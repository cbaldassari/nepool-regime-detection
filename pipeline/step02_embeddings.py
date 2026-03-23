"""
step02_embeddings.py
====================
NEPOOL Regime Detection Pipeline — Step 02: Chronos-2 Embedding Extraction

Input  : results/preprocessed.parquet   (43,764 rows × 12 cols)
Output : results/embeddings.parquet     (N_windows × [datetime + emb_0…emb_D-1])
Plots  : results/step02/

Design
------
Context length (data-driven, confirmed by ACF analysis):
  • Dominant cycles: 24h (daily), 168h (weekly), ~720h (monthly dispatch cycle)
  • ACF of log_return/arcsinh_lmp still significant at 720h (0.34–0.45)
  • Context = 720h: captures ≥4 weekly cycles + 1 monthly dispatch cycle

Sliding windows:
  • Stride = 6h  →  (43764 − 720) / 6 ≈ 7,174 windows
  • Timestamp = LAST observation of each window
  • Windows straddling the 25h gap (2023-06-15) are skipped automatically

Chronos-2 encoder (amazon/chronos-2 — 120M params, released Oct 2025):
  • Native multivariate — all 11 channels passed as a single group per window
  • Group attention captures cross-channel dependencies (e.g. LMP vs fuel mix)
  • Per window: embed(group of 11 series × 720 steps) → (11, n_patches, D_model)
  • Mean-pool over patch dimension: (11, D_model)
  • Concatenate channels: final embedding = 11 × D_model per window
  • D_model inferred dynamically at runtime (expected ~512)

Execution modes:
  • TEST_MODE = True  → 50 windows, local CPU, no Ray  (quick sanity check)
  • TEST_MODE = False → all windows, Ray cluster 3×GPU  (production run)

Install dependencies:
    pip install "chronos-forecasting[chronos2]" ray[default]
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

CONTEXT_LEN   = C.MEAN_REVERSION["context_len"]  # 720h
STRIDE_H      = 6
CHRONOS_MODEL = "amazon/chronos-2"          # multivariate, 120M params, D_model 768
N_GPUS        = 8                           # max cap — actual actors = min(N_GPUS, cluster_gpus) auto-detected at runtime
BATCH_SIZE    = 32                          # reduced from 256: OOM with larger batches on RTX 3080 Ti (12 GB)
MAX_RETRIES   = 3                           # Ray actor retries on node failure

# ── Esperimento ──────────────────────────────────────────────────────────────
# Passa da linea di comando:  python step02_embeddings.py --exp A
#   A -> log_return          (raw)
#   B -> arcsinh_lmp         (raw)
#   C -> mstl_resid_lr       (MSTL residuo di log_return)
#   D -> mstl_resid_arcsinh  (MSTL residuo di arcsinh_lmp)
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--exp", default="A", choices=["A","B","C","D","E","F","G","H","I","J","K","L"])
EXPERIMENT = _parser.parse_known_args()[0].exp

_ILR_COLS = ["ilr_1", "ilr_2", "ilr_3", "ilr_4", "ilr_5", "ilr_6", "ilr_7"]

_EXP_MAP = {
    # ── Price only (univariate) ───────────────────────────────────────────────
    "A": ["log_return"],            # stationary return (arcsinh diff)
    "B": ["arcsinh_lmp"],           # price level, negative-safe
    "C": ["mstl_resid_lr"],         # deseasonalized return
    "D": ["mstl_resid_arcsinh"],    # deseasonalized level (TOPSIS winner)
    "E": ["log_lmp_shifted"],       # log-scale level (alt. to arcsinh)
    "F": ["lmp_clipped"],           # raw level without extreme spikes
    "G": ["mstl_resid_log"],        # deseasonalized log-level
    "H": ["quantile_transform"],    # distribution-free normal mapping
    "I": ["rolling_zscore_24h"],    # adaptive local normalization
    # ── ILR ablation study (multivariate) ────────────────────────────────────
    "J": _ILR_COLS,                              # ILR only   (7 ch, no price)
    "K": ["log_return"]          + _ILR_COLS,   # A + ILR    (8 ch)
    "L": ["mstl_resid_arcsinh"]  + _ILR_COLS,   # D + ILR    (8 ch) — best expected
}
FEAT_COLS  = _EXP_MAP[EXPERIMENT]
EXP_DIR    = Path(C.RESULTS_DIR) / f"exp_{EXPERIMENT}"

PLOT_DIR   = EXP_DIR / "step02"
GAP_FROM   = pd.Timestamp("2023-06-14 23:00:00")
GAP_TO     = pd.Timestamp("2023-06-16 00:00:00")

# TEST_MODE: True  → 50 windows, local CPU (quick sanity check, no Ray needed)
#            False → all windows, Ray cluster 3×GPU  (production run)
TEST_MODE  = False   # True = 50 win CPU (debug) | False = Ray cluster 3×GPU (production)


# ═══════════════════════════════════════════════════════════════════════════
#  Sliding window construction
# ═══════════════════════════════════════════════════════════════════════════

def build_windows(out: pd.DataFrame, max_windows: int = None
                  ) -> tuple:
    """
    Build (N_windows, CONTEXT_LEN, N_features) sliding window tensor.

    Rules:
      • Timestamp = datetime of the LAST observation in the window
      • Skip windows straddling the 25h gap (2023-06-15):
        detected by checking actual vs expected time span of the window
      • max_windows: cap total windows (used in TEST_MODE)

    Returns: (windows, timestamps, n_skipped)
    """
    dt = out["datetime"].values
    X  = out[FEAT_COLS].values.astype(np.float32)
    T  = len(X)

    windows, timestamps = [], []
    skipped = 0

    for i in range(0, T - CONTEXT_LEN + 1, STRIDE_H):
        if max_windows and len(windows) >= max_windows:
            break
        end     = i + CONTEXT_LEN - 1
        t_start = pd.Timestamp(dt[i])
        t_end   = pd.Timestamp(dt[end])

        # Skip windows that cross the 25h data gap
        if t_start <= GAP_FROM and t_end >= GAP_TO:
            skipped += 1
            continue

        # Skip windows with internal discontinuity (> 2h between any steps)
        actual_span_h   = (t_end - t_start).total_seconds() / 3600
        expected_span_h = CONTEXT_LEN - 1
        if abs(actual_span_h - expected_span_h) > 2:
            skipped += 1
            continue

        windows.append(X[i : i + CONTEXT_LEN])
        timestamps.append(t_end)

    return (np.stack(windows, axis=0).astype(np.float32),
            np.array(timestamps),
            skipped)


# ═══════════════════════════════════════════════════════════════════════════
#  Chronos-2 encoder  (shared logic)
# ═══════════════════════════════════════════════════════════════════════════

def load_chronos(model_name: str, device: str):
    """
    Load Chronos-2 pipeline and return (pipeline, encoder_hidden_dim).
    Uses Chronos2Pipeline which supports native multivariate (group) input.
    Hidden dim inferred via dummy forward pass.

    embed() returns a list of tensors, each of shape:
      (n_variates, num_patches + 2, d_model)
    The +2 comes from [REG] token (index 0) and masked output patch (index -1).
    """
    import torch
    from chronos import Chronos2Pipeline

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    pipeline = Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
    )

    # Infer D_model: pass 2 dummy series via 3D tensor (1 window, 2 variates, 10 steps)
    dummy = torch.zeros(1, 2, 10, dtype=torch.float32)
    emb_list, _ = pipeline.embed(dummy)
    # emb_list[0] shape: (n_variates, num_patches + 2, d_model)
    hidden_dim = emb_list[0].shape[-1]

    return pipeline, hidden_dim


def embed_batch_chronos(pipeline, batch: np.ndarray, device: str) -> np.ndarray:
    """
    Embed a batch of windows using Chronos-2 multivariate encoder.

    Chronos-2 natively supports 3D batched input (batch, n_variates, history_length).
    All B windows are passed in one call — no Python loop over windows.

    Input format: (B, F, T) tensor  [batch, variates, time]
    embed() returns: list of B tensors, each (F, num_patches+2, D_model)
      - index 0   : [REG] global summary token
      - index 1…-2: actual patch embeddings
      - index -1  : masked output patch token

    Per window: mean-pool over actual patches → (F, D_model) → flatten → (F×D_model,)

    batch  : (B, CONTEXT_LEN, N_features)
    returns: (B, N_features × hidden_dim)  float32
    """
    import torch

    B, T, F = batch.shape

    # Reshape (B, T, F) → (B, F, T) as required by Chronos-2 embed API.
    # Keep the tensor on CPU — Chronos2Pipeline (loaded with device_map=device)
    # handles device placement internally via its DataLoader.  Moving to CUDA
    # here causes "cannot pin torch.cuda.FloatTensor" because the pipeline's
    # DataLoader has pin_memory=True, which only works on dense CPU tensors.
    x = torch.tensor(batch.transpose(0, 2, 1), dtype=torch.float32)  # CPU

    # batch_size controls how many windows Chronos processes simultaneously on GPU.
    # Keeping it small (4) limits peak VRAM: with 10ch×720steps the activations
    # for a full batch of 32 exceed 12 GB on RTX 3080 Ti.
    # emb_list: list of B tensors, each (F, num_patches+2, D_model)
    EMBED_BATCH = 4
    emb_list, _ = pipeline.embed(x, batch_size=EMBED_BATCH)

    all_embs = []
    for emb in emb_list:
        # Mean-pool over actual patches only (skip [REG] at 0, masked patch at -1)
        emb_pooled = emb[:, 1:-1, :].mean(dim=1)   # (F, D_model)
        emb_flat   = emb_pooled.reshape(-1).float().cpu().numpy()  # (F × D_model,)
        all_embs.append(emb_flat)

    return np.stack(all_embs, axis=0)   # (B, F × D_model)


# ═══════════════════════════════════════════════════════════════════════════
#  ChronosActor — module-level class (required for Ray Client serialization)
#
#  IMPORTANT: @ray.remote classes/functions defined *inside* a function cannot
#  be pickled by Ray Client (ray://...). The class must live at module level.
#  ray.remote() is applied as a plain function inside extract_ray() so that
#  the `ray` import stays lazy (only needed in production mode).
# ═══════════════════════════════════════════════════════════════════════════

class ChronosActor:
    """
    Chronos-2 embedding actor for Ray distributed execution.
    One instance per GPU node — loads the model once, processes chunks in batches.
    Defined at module level so Ray Client can pickle and send it to the cluster.

    SERIALIZATION NOTE
    ------------------
    Ray Client (ray://) always uses pickle5 for ALL data crossing the client↔cluster
    boundary, including ray.put() objects.  pickle5 activates numpy's buffer protocol,
    whose internal _frombuffer() signature changed between numpy versions.  Because the
    client runs Python 3.12.7 and the cluster Python 3.12.3, passing numpy arrays in
    either direction raises:
        TypeError: _frombuffer() takes 4 positional arguments but 5 were given

    Fix: never cross the boundary with numpy arrays.  Convert to a plain Python tuple
        (raw_bytes: bytes, shape: tuple[int,...], dtype_str: str)
    before ray.put() and reconstruct with np.frombuffer() on the worker.  The bytes
    object is serialised as a literal in pickle — no buffer protocol, no ABI issue.
    The same applies to the return value (result numpy → bytes tuple → client).
    """
    def __init__(self, model_name: str):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ChronosActor: CUDA not available on this Ray worker. "
                "Check that num_gpus=1 is correctly allocated and CUDA drivers are installed."
            )
        self.device   = f"cuda:{torch.cuda.current_device()}"
        print(f"  [Actor] GPU: {torch.cuda.get_device_name()} | device={self.device}", flush=True)
        pipeline, _   = load_chronos(model_name, self.device)
        self.pipeline = pipeline

    def embed_chunk(self, chunk_payload: tuple) -> tuple:
        """
        Accept / return plain Python tuples to avoid pickle5 numpy ABI mismatch.

        chunk_payload : (raw_bytes, shape, dtype_str)
            raw_bytes  — chunk.tobytes()
            shape      — e.g. (2315, 720, 11)
            dtype_str  — e.g. '<f4'

        returns       : (raw_bytes, shape, dtype_str) for the embedding matrix
        """
        raw_bytes, shape, dtype_str = chunk_payload
        chunk = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()

        results = []
        for start in range(0, len(chunk), BATCH_SIZE):
            batch = chunk[start : start + BATCH_SIZE]
            results.append(embed_batch_chronos(self.pipeline, batch, self.device))
        result = np.concatenate(results, axis=0)

        # Return as bytes tuple — same reason as input
        return result.tobytes(), result.shape, result.dtype.str


# ═══════════════════════════════════════════════════════════════════════════
#  LOCAL extraction  (TEST_MODE — CPU, no Ray)
# ═══════════════════════════════════════════════════════════════════════════

def extract_local(windows: np.ndarray) -> np.ndarray:
    """
    Single-process CPU extraction for TEST_MODE (50 windows, ~2 min).
    Uses the same embed_batch_chronos logic as ChronosActor.
    """
    pipeline, _ = load_chronos(CHRONOS_MODEL, "cpu")
    results = []
    n = len(windows)
    for start in range(0, n, BATCH_SIZE):
        batch = windows[start : start + BATCH_SIZE]
        results.append(embed_batch_chronos(pipeline, batch, "cpu"))
        done = min(start + BATCH_SIZE, n)
        pct  = done / n * 100
        print(f"  LOCAL [{done:>3}/{n}]  ({pct:>5.1f}%)", flush=True)
    return np.concatenate(results, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
#  RAY extraction  (production — Ray cluster, datalab-rayclnt.unitus.it)
# ═══════════════════════════════════════════════════════════════════════════

def extract_ray(windows: np.ndarray) -> np.ndarray:
    """
    Distributed embedding extraction via Ray.
    Splits windows into N_GPUS equal chunks, one actor per GPU.

    Robustness:
      - max_retries=MAX_RETRIES  : Ray re-schedules the task on a healthy node
        if the original node crashes (ObjectLostError / node failure).
      - BATCH_SIZE=32            : reduced to limit GPU memory pressure per batch.
      - ray.wait loop            : collects results as they arrive; on failure
        the future is retried rather than propagating immediately.

    NOTE: ChronosActor is defined at module level (above) — required for Ray
    Client serialization. ray.remote() is applied here as a plain function.
    """
    import ray

    ray.init(
        "ray://datalab-rayclnt.unitus.it:10001",
        ignore_reinit_error=True,
        runtime_env={"pip": ["chronos-forecasting[chronos2]"]},
    )

    # Count DISTINCT alive nodes with at least 1 GPU.
    # Using ray.nodes() instead of cluster_resources()["GPU"] avoids the case
    # where a single node reports 2 GPU slots (causing SPREAD to put 2 actors there).
    # n_actors = min(N_GPUS, n_gpu_nodes) → never more than 1 actor per node.
    cluster_gpus  = int(ray.cluster_resources().get("GPU", 0))
    gpu_nodes     = [n for n in ray.nodes()
                     if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0]
    n_gpu_nodes   = len(gpu_nodes)
    n_actors      = min(N_GPUS, n_gpu_nodes) if n_gpu_nodes > 0 else 1
    node_ips      = [n["NodeManagerAddress"] for n in gpu_nodes]
    print(f"  ✓  GPU nodes: {n_gpu_nodes}  {node_ips}  →  {n_actors} actors  "
          f"(cluster GPU slots: {cluster_gpus})", flush=True)

    # SPREAD placement group — guarantees 1 actor per node (never 2 on same node).
    # Prevents RAM OOM from 2 Chronos-2 instances loading simultaneously on same host.
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    pg = placement_group(
        [{"GPU": 1, "CPU": 1} for _ in range(n_actors)],
        strategy="STRICT_SPREAD",   # GUARANTEES 1 bundle per node — fails if impossible
    )
    ray.get(pg.ready(), timeout=60)
    print(f"  Placement group STRICT_SPREAD ready — {n_actors} bundles, 1 per node", flush=True)

    # No auto-restart: when an actor dies we redistribute its work manually.
    # max_restarts=0 lets failures surface immediately so we can failover.
    RemoteChronosActor = ray.remote(
        num_gpus=1,
        max_restarts=0,
        max_task_retries=0,
    )(ChronosActor)

    # ── Split work into fine-grained batches (not 1 chunk per actor) ─────────
    # Each batch = BATCH_SIZE windows.  Batches are dispatched dynamically to
    # whichever actor is free.  If an actor dies, its in-flight batch is
    # re-queued and sent to a surviving actor automatically.
    n           = len(windows)
    all_batches = [windows[i : i + BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
    n_batches   = len(all_batches)

    actors = [
        RemoteChronosActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        ).remote(CHRONOS_MODEL)
        for i in range(n_actors)
    ]

    print(f"\n  {n} windows  →  {n_batches} batches × {BATCH_SIZE}  |  "
          f"{n_actors} actors  |  cluster GPUs: {cluster_gpus}", flush=True)

    # ── Dynamic dispatch with failover ────────────────────────────────────────
    # pending_queue : batches not yet dispatched
    # flight        : { future : (actor_idx, batch_idx, batch_data) }
    # dead_actors   : set of actor indices that have crashed
    # results       : { batch_idx : np.ndarray }
    pending_queue = list(range(n_batches))   # batch indices to process
    flight        = {}                       # future → (actor_idx, batch_idx, batch)
    dead_actors   = set()
    results       = {}
    actor_busy    = [False] * n_actors

    def _alive_actors():
        return [i for i in range(n_actors) if i not in dead_actors]

    def _dispatch_next(actor_idx):
        """Send next queued batch to actor_idx. Returns False if queue empty."""
        if not pending_queue:
            return False
        bidx  = pending_queue.pop(0)
        batch = all_batches[bidx]
        payload = (batch.tobytes(), batch.shape, batch.dtype.str)
        ref  = ray.put(payload)
        fut  = actors[actor_idx].embed_chunk.remote(ref)
        flight[fut] = (actor_idx, bidx, batch)
        actor_busy[actor_idx] = True
        return True

    # Initial dispatch: 1 batch per alive actor
    for i in _alive_actors():
        _dispatch_next(i)

    n_done = 0
    while flight:
        done_refs, _ = ray.wait(list(flight.keys()), num_returns=1, timeout=300)

        if not done_refs:
            # Timeout — cancel all in-flight and requeue
            print(f"  ⏱  timeout waiting for results — requeuing in-flight batches",
                  flush=True)
            for fut, (aidx, bidx, batch) in list(flight.items()):
                pending_queue.insert(0, bidx)
                dead_actors.add(aidx)
                try: ray.cancel(fut, force=True)
                except Exception: pass
            flight.clear()
            alive = _alive_actors()
            if not alive:
                raise RuntimeError("All Ray actors are dead — cannot continue.")
            for i in alive:
                _dispatch_next(i)
            continue

        fut = done_refs[0]
        actor_idx, bidx, batch = flight.pop(fut)
        actor_busy[actor_idx]  = False

        try:
            raw_bytes, shape, dtype_str = ray.get(fut)
            arr = np.frombuffer(raw_bytes,
                                dtype=np.dtype(dtype_str)).reshape(shape).copy()
            results[bidx] = arr
            n_done += 1
            pct = n_done / n_batches * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  RAY [{bar}] {n_done}/{n_batches} batches  ({pct:>5.1f}%)  "
                  f"actor={actor_idx}", flush=True)
        except Exception as e:
            # Actor crashed — mark dead, requeue batch, dispatch to surviving actors
            print(f"  ⚠  actor {actor_idx} died ({type(e).__name__}) — "
                  f"requeuing batch {bidx}, redistributing to survivors", flush=True)
            dead_actors.add(actor_idx)
            pending_queue.insert(0, bidx)   # put failed batch back at front

            alive = _alive_actors()
            if not alive:
                raise RuntimeError(
                    f"All {n_actors} Ray actors are dead. "
                    f"Processed {n_done}/{n_batches} batches before failure."
                )
            print(f"  ✓  Surviving actors: {alive}", flush=True)

        # Keep alive actors busy
        for i in _alive_actors():
            if not actor_busy[i]:
                dispatched = _dispatch_next(i)
                if dispatched:
                    actor_busy[i] = True

    ray.shutdown()
    # Reconstruct in original order
    ordered = np.concatenate([results[i] for i in range(n_batches)], axis=0)
    return ordered


# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostic plots  (post-embedding)
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(emb_df: pd.DataFrame) -> None:
    """
    Post-extraction diagnostic figures saved to results/step02/.

    03_embedding_norms.png    — L2 norm over time (anomaly indicator)
    04_embedding_variance.png — variance per dimension + cumulative (→ UMAP n_components)
    05_embedding_tsne.png     — t-SNE 2D coloured by month and LMP
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    dt_col   = pd.to_datetime(emb_df["datetime"])
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    E        = emb_df[emb_cols].values.astype(np.float32)

    # ── 03: L2 norms over time ───────────────────────────────────────────────
    norms = np.linalg.norm(E, axis=1)
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(dt_col, norms, lw=0.5, color="#2c6fad", alpha=0.8)
    ax.axhline(np.percentile(norms, 99), color="red", lw=0.8, ls="--",
               label="99th pct")
    ax.set_title("Embedding L2-norm over time — spikes signal unusual regimes",
                 fontweight="bold")
    ax.set_ylabel("‖emb‖₂");  ax.legend(fontsize=8)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_embedding_norms.png")
    plt.close(fig)
    print("  [plot] 03_embedding_norms.png")

    # ── 04: Variance per dimension ───────────────────────────────────────────
    var_sorted = np.sort(E.var(axis=0))[::-1]
    cumvar     = np.cumsum(var_sorted) / var_sorted.sum()
    n90 = np.searchsorted(cumvar, 0.90) + 1
    n95 = np.searchsorted(cumvar, 0.95) + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].semilogy(range(1, len(var_sorted) + 1), var_sorted,
                     lw=1.2, color="#5a9e6f")
    axes[0].set_title("Variance per embedding dimension (log-y)")
    axes[0].set_xlabel("Dimension rank");  axes[0].set_ylabel("Variance")

    axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100, lw=1.5, color="#e07b39")
    axes[1].axhline(90, color="red",  lw=0.8, ls="--")
    axes[1].axhline(95, color="blue", lw=0.8, ls="--")
    axes[1].axvline(n90, color="red",  lw=0.8, ls="--", label=f"90% → {n90} dims")
    axes[1].axvline(n95, color="blue", lw=0.8, ls="--", label=f"95% → {n95} dims")
    axes[1].set_title("Cumulative variance explained")
    axes[1].set_xlabel("Number of dimensions");  axes[1].set_ylabel("% variance")
    axes[1].legend(fontsize=8)
    fig.suptitle("Embedding dimensionality — informs UMAP n_components",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_embedding_variance.png")
    plt.close(fig)
    print(f"  [plot] 04_embedding_variance.png  (90%→{n90} dims, 95%→{n95} dims)")

    # ── 05: t-SNE ────────────────────────────────────────────────────────────
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        idx   = np.random.default_rng(42).choice(len(E), min(3000, len(E)),
                                                  replace=False)
        idx.sort()
        E_sub = StandardScaler().fit_transform(E[idx])
        Z     = TSNE(n_components=2, perplexity=40, n_iter=1000,
                     random_state=42, n_jobs=-1).fit_transform(E_sub)
        month = dt_col.iloc[idx].dt.month.values

        pre      = pd.read_parquet(Path(C.RESULTS_DIR) / "preprocessed.parquet")
        pre["datetime"] = pd.to_datetime(pre["datetime"])
        lmp_vals = dt_col.iloc[idx].map(pre.set_index("datetime")["lmp"]).values
        lmp_clip = np.clip(lmp_vals, 0, np.percentile(lmp_vals[~np.isnan(lmp_vals)], 98))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sc  = axes[0].scatter(Z[:, 0], Z[:, 1], c=month, cmap="hsv",
                              s=4, alpha=0.6, vmin=1, vmax=12)
        plt.colorbar(sc, ax=axes[0], label="Month")
        axes[0].set_title("t-SNE — coloured by month")
        axes[0].set_xlabel("t-SNE 1");  axes[0].set_ylabel("t-SNE 2")

        sc2 = axes[1].scatter(Z[:, 0], Z[:, 1], c=lmp_clip, cmap="YlOrRd",
                              s=4, alpha=0.6)
        plt.colorbar(sc2, ax=axes[1], label="LMP $/MWh")
        axes[1].set_title("t-SNE — coloured by LMP (clipped 98th pct)")
        axes[1].set_xlabel("t-SNE 1");  axes[1].set_ylabel("t-SNE 2")

        fig.suptitle("t-SNE projection of Chronos-2 embeddings", fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "05_embedding_tsne.png")
        plt.close(fig)
        print("  [plot] 05_embedding_tsne.png")
    except Exception as e:
        print(f"  [plot] 05_embedding_tsne.png  SKIPPED ({e})")

    # ── 06: Per-channel embedding norm ───────────────────────────────────────
    # Reshape (N, 11×D) → (N, 11, D) to check that each feature channel
    # contributed meaningfully to the final embedding.
    # A near-zero channel bar indicates that Chronos-2 collapsed that channel
    # (constant input, saturation, or tokenisation failure).
    n_feat = len(FEAT_COLS)
    if E.shape[1] % n_feat == 0:
        D       = E.shape[1] // n_feat
        E_3d    = E.reshape(len(E), n_feat, D)
        ch_norms = np.linalg.norm(E_3d, axis=2).mean(axis=0)   # (11,)
        ch_std   = np.linalg.norm(E_3d, axis=2).std(axis=0)

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(n_feat)
        bars = ax.bar(x, ch_norms, yerr=ch_std, capsize=4,
                      color=plt.cm.tab10(x / n_feat), alpha=0.8)
        ax.axhline(0.01 * ch_norms.mean(), color="red", lw=0.8, ls="--",
                   label="1% mean (silence threshold)")
        ax.set_xticks(x)
        ax.set_xticklabels(FEAT_COLS, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Mean L2 norm (± std)");
        ax.set_title("Per-channel embedding norms — all channels should be active",
                     fontweight="bold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "06_channel_norms.png")
        plt.close(fig)
        print("  [plot] 06_channel_norms.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> pd.DataFrame:
    t0 = time.time()

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 02: Chronos-2 Embeddings")
    print("═" * 65)
    print(f"  Input   : {C.RESULTS_DIR}/preprocessed.parquet")
    print(f"  Output  : {C.RESULTS_DIR}/embeddings.parquet")
    print(f"  Plots   : {PLOT_DIR}/")
    print(f"  Model   : {CHRONOS_MODEL}  (multivariate, 120M params)")
    print(f"  Context : {CONTEXT_LEN}h  (30 days)")
    print(f"  Stride  : {STRIDE_H}h")
    print(f"  Cluster : ray://datalab-rayclnt.unitus.it:10001  ({N_GPUS}×GPU)")
    print()
    print("  Embedding strategy:")
    print("    1 price channel per experiment (A/B/C/D) → Chronos-2 univariate encoder.")
    print("    Per window: embed(720 steps) → mean-pool patches → embedding vector.")
    print("─" * 65)

    n_steps = 6
    def _step(n: int, label: str) -> None:
        print(f"\n  ▶ [{n}/{n_steps}] {label} ···", flush=True)
    def _done(n: int, label: str, detail: str = "") -> None:
        suffix = f"  ({detail})" if detail else ""
        print(f"  ✓ [{n}/{n_steps}] {label}{suffix}", flush=True)

    mode_str = "TEST (50 win, CPU)" if TEST_MODE else "PRODUCTION (Ray cluster)"
    print(f"  Mode    : {mode_str}", flush=True)
    print("─" * 65, flush=True)

    # ── 1. Load ─────────────────────────────────────────────────────────────
    _step(1, "Load preprocessed data")
    out = pd.read_parquet(Path(C.RESULTS_DIR) / "preprocessed.parquet")
    out["datetime"] = pd.to_datetime(out["datetime"])
    _done(1, "Load preprocessed data", f"{len(out):,} righe  ×  {len(FEAT_COLS)} canali")

    # ── 2. Sliding windows ──────────────────────────────────────────────────
    _step(2, "Build sliding windows")
    max_w   = 50 if TEST_MODE else None
    windows, timestamps, n_skipped = build_windows(out, max_windows=max_w)
    N = len(windows)
    _done(2, "Build sliding windows",
          f"{N:,} finestre  |  {n_skipped} skipped  |  shape ({N}, {CONTEXT_LEN}, {len(FEAT_COLS)})")
    print(f"     Range: {pd.Timestamp(timestamps[0]).date()}  →  "
          f"{pd.Timestamp(timestamps[-1]).date()}", flush=True)

    # ── 3. Embeddings ────────────────────────────────────────────────────────
    _step(3, "Extract Chronos-2 embeddings")
    try:
        if TEST_MODE:
            print("  [TEST MODE] Running on CPU — 50 windows only", flush=True)
            embeddings = extract_local(windows)
        else:
            embeddings = extract_ray(windows)
    except ImportError as e:
        print(f"\n  ERROR: missing dependency — {e}")
        print("  Install with:")
        print("    pip install chronos-forecasting[chronos2]")
        print("    pip install torch --extra-index-url "
              "https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    emb_dim = embeddings.shape[1]
    _done(3, "Extract Chronos-2 embeddings", f"shape ({N}, {emb_dim})")

    # ── 4b. Embedding sanity checks ──────────────────────────────────────────
    # Fail-fast: NaN/Inf in embeddings propagate silently through PCA and UMAP.
    # Better to crash here with a clear message than get garbage clusters.
    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"Embedding sanity check FAILED: "
            f"{nan_count} NaN values, {inf_count} Inf values detected in "
            f"embeddings matrix ({embeddings.shape}).\n"
            f"Likely cause: bfloat16 overflow or degenerate input windows.\n"
            f"Fix: inspect windows with extreme values, or switch torch_dtype to float32."
        )

    # Per-channel norm check — embedding is (N, 11 × D_model).
    # Reshape to (N, 11, D_model) and compute per-channel mean L2 norm.
    # A channel with mean norm < 1% of the average is effectively silent.
    D = emb_dim // len(FEAT_COLS)
    if emb_dim % len(FEAT_COLS) == 0:
        E_3d = embeddings.reshape(len(embeddings), len(FEAT_COLS), D)
        ch_norms = np.linalg.norm(E_3d, axis=2).mean(axis=0)   # (11,)
        threshold = 0.01 * ch_norms.mean()
        silent = [FEAT_COLS[i] for i, v in enumerate(ch_norms) if v < threshold]
        if silent:
            print(f"\n  ⚠  WARNING: {len(silent)} channel(s) with near-zero embedding "
                  f"norm (<1% mean): {silent}")
            print(f"     Per-channel norms: "
                  f"{dict(zip(FEAT_COLS, ch_norms.round(2)))}")
        else:
            print(f"\n  ✓  Embedding sanity: 0 NaN, 0 Inf, all {len(FEAT_COLS)} channels active")
            print(f"     Per-channel norms: "
                  f"{dict(zip(FEAT_COLS, ch_norms.round(2)))}")
    else:
        print(f"\n  ✓  Embedding sanity: 0 NaN, 0 Inf  "
              f"(emb_dim {emb_dim} not divisible by {len(FEAT_COLS)} — skip per-channel check)")

    # ── 4. Save embeddings ───────────────────────────────────────────────────
    _step(4, "Save embeddings")
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df   = pd.DataFrame(embeddings, columns=emb_cols)
    emb_df.insert(0, "datetime", pd.to_datetime(timestamps))
    out_path = EXP_DIR / "embeddings.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(out_path, index=False)
    _done(4, "Save embeddings", f"{out_path.stat().st_size / 1e6:.1f} MB  →  {out_path}")

    # ── 5. Plots ─────────────────────────────────────────────────────────────
    _step(5, "Diagnostic plots")
    make_plots(emb_df)
    _done(5, "Diagnostic plots", f"→ {PLOT_DIR}/")

    # ── Report ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    file_mb = out_path.stat().st_size / 1e6
    norms   = np.linalg.norm(embeddings, axis=1)

    print("\n" + "─" * 65)
    print("  EMBEDDING REPORT")
    print("─" * 65)
    print(f"  Cluster          : ray://datalab-rayclnt.unitus.it:10001")
    print(f"  Input rows       : {len(out):>10,}")
    print(f"  Context length   : {CONTEXT_LEN:>10}h")
    print(f"  Stride           : {STRIDE_H:>10}h")
    print(f"  Windows produced : {N:>10,}")
    print(f"  Windows skipped  : {n_skipped:>10,}")
    print(f"  Channels         : {len(FEAT_COLS):>10}")
    print(f"  Encoder hidden   : {emb_dim // len(FEAT_COLS):>10}")
    print(f"  Embedding dim    : {emb_dim:>10,}  ({len(FEAT_COLS)} × "
          f"{emb_dim // len(FEAT_COLS)})")
    print(f"  Output size      : {file_mb:>9.1f} MB")
    print(f"  Elapsed          : {elapsed:>9.1f}s")
    print()
    print(f"  L2 norm — mean={norms.mean():.2f}  std={norms.std():.2f}  "
          f"min={norms.min():.2f}  max={norms.max():.2f}")
    print()
    print("  Next → Step 03: cuML UMAP + HDBSCAN su embeddings raw (7680D)")
    print("─" * 65)
    print(f"  Embeddings → {out_path}")
    print(f"  Plots      → {PLOT_DIR}/")
    print("═" * 65 + "\n")

    return emb_df


if __name__ == "__main__":
    main()
