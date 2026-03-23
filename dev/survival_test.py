"""
survival_test.py
================
NEPOOL Regime Detection — GPU/Model Survival Test

Scopo: determinare quale variante di Chronos-2 è stabile sui 12GB VRAM
       della RTX 3080 Ti, e quale batch size massimo è sostenibile.

Test matrix:
  Modelli     : amazon/chronos-2  (120M)  →  amazon/chronos-2-large  (710M)
  Batch sizes : 4, 8, 16, 32
  Finestre    : N_WINDOWS per actor (default 200)
  Nodi        : 1 actor per GPU (3 GPU totali)

Metriche riportate per ogni combinazione (modello × batch):
  • VRAM peak  (GB)          — torch.cuda.max_memory_allocated
  • Throughput (finestre/s)  — N_WINDOWS / elapsed
  • Stabilità               — conteggio NaN + Inf negli embeddings
  • Status                  — OK / OOM / ERROR

Output:
  results/survival_test.csv   — tabella completa
  (stampato anche su stdout)

Uso:
  python survival_test.py

Note:
  • Se un batch size va in OOM il test salta i batch size maggiori
    per quel modello sullo stesso nodo.
  • Context length e FEAT_COLS sono identici a step02 per massima
    rappresentatività del carico reale.
  • bfloat16 su GPU (come step02), float32 su CPU.
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  Configurazione
# ═══════════════════════════════════════════════════════════════════════════

RAY_ADDRESS  = "ray://datalab-rayclnt.unitus.it:10001"
N_WINDOWS    = 200          # finestre per actor per run (abbastanza per stabilizzare VRAM)
CONTEXT_LEN  = C.MEAN_REVERSION["context_len"]   # 720h — identico a step02
STRIDE_H     = 6

MODELS_TO_TEST = [
    "amazon/chronos-2",        # 120M params
    "amazon/chronos-2-large",  # 710M params
]

BATCH_SIZES = [4, 8, 16, 32]

# Stesse feature di step02 (10 canali dopo modifica arcsinh)
FEAT_COLS = [
    "arcsinh_lmp", "log_return", "total_mw",
    "ilr_1", "ilr_2", "ilr_3", "ilr_4", "ilr_5", "ilr_6", "ilr_7",
]

GAP_FROM = pd.Timestamp("2023-06-14 23:00:00")
GAP_TO   = pd.Timestamp("2023-06-16 00:00:00")

RESULTS_DIR = Path(C.RESULTS_DIR)


# ═══════════════════════════════════════════════════════════════════════════
#  Costruzione finestre (identico a step02.build_windows)
# ═══════════════════════════════════════════════════════════════════════════

def build_windows(out: pd.DataFrame, max_windows: int) -> tuple:
    dt = out["datetime"].values
    X  = out[FEAT_COLS].values.astype(np.float32)
    T  = len(X)

    windows, timestamps = [], []
    skipped = 0

    for i in range(0, T - CONTEXT_LEN + 1, STRIDE_H):
        if len(windows) >= max_windows:
            break
        end     = i + CONTEXT_LEN - 1
        t_start = pd.Timestamp(dt[i])
        t_end   = pd.Timestamp(dt[end])

        if t_start <= GAP_FROM and t_end >= GAP_TO:
            skipped += 1
            continue

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
#  SurvivalActor — module-level (richiesto da Ray Client)
# ═══════════════════════════════════════════════════════════════════════════

class SurvivalActor:
    """
    Carica Chronos-2 su GPU e processa N finestre con un dato batch size.
    Riporta VRAM peak, throughput, NaN/Inf count.
    Definita a livello di modulo per compatibilità con Ray Client (pickle5).
    """

    def __init__(self, model_name: str):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponibile su questo worker.")
        self.device     = f"cuda:{torch.cuda.current_device()}"
        self.model_name = model_name
        self.gpu_name   = torch.cuda.get_device_name()
        print(f"  [Actor] {self.gpu_name} | {self.device} | {model_name}", flush=True)

        from chronos import Chronos2Pipeline
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=self.device,
            dtype=__import__("torch").bfloat16,
        )

    def run(self, payload: tuple, batch_size: int) -> dict:
        """
        payload : (raw_bytes, shape, dtype_str)
        Ritorna dizionario con metriche (tutto serializzabile, no numpy).
        """
        import torch
        import numpy as np

        raw_bytes, shape, dtype_str = payload
        windows = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
        N = len(windows)

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)

        t0      = time.time()
        all_emb = []
        oom     = False

        try:
            for start in range(0, N, batch_size):
                batch = windows[start : start + batch_size]   # (B, T, F)
                B, T, F = batch.shape

                x = torch.tensor(
                    batch.transpose(0, 2, 1),   # (B, F, T)
                    dtype=torch.float32,
                )
                emb_list, _ = self.pipeline.embed(x, batch_size=B)
                for emb in emb_list:
                    pooled = emb[:, 1:-1, :].mean(dim=1)   # (F, D)
                    flat   = pooled.reshape(-1).float().cpu().numpy()
                    all_emb.append(flat)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom = True
            else:
                return {
                    "status"    : f"ERROR: {str(e)[:120]}",
                    "vram_peak" : -1.0,
                    "throughput": 0.0,
                    "nan_count" : -1,
                    "inf_count" : -1,
                    "gpu_name"  : self.gpu_name,
                    "n_done"    : len(all_emb),
                }

        elapsed = time.time() - t0
        torch.cuda.synchronize(self.device)
        vram_peak_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3

        if oom:
            return {
                "status"    : "OOM",
                "vram_peak" : vram_peak_gb,
                "throughput": 0.0,
                "nan_count" : -1,
                "inf_count" : -1,
                "gpu_name"  : self.gpu_name,
                "n_done"    : len(all_emb),
            }

        E = np.stack(all_emb, axis=0)
        return {
            "status"    : "OK",
            "vram_peak" : float(vram_peak_gb),
            "throughput": float(N / elapsed) if elapsed > 0 else 0.0,
            "nan_count" : int(np.isnan(E).sum()),
            "inf_count" : int(np.isinf(E).sum()),
            "emb_dim"   : int(E.shape[1]),
            "gpu_name"  : self.gpu_name,
            "n_done"    : N,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import ray

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  GPU/Model Survival Test")
    print("═" * 65)
    print(f"  Ray      : {RAY_ADDRESS}")
    print(f"  Modelli  : {MODELS_TO_TEST}")
    print(f"  Batch sz : {BATCH_SIZES}")
    print(f"  Finestre : {N_WINDOWS} per actor")
    print(f"  Context  : {CONTEXT_LEN}h  |  Canali: {len(FEAT_COLS)}")
    print("─" * 65)

    # ── 1. Carica dati ──────────────────────────────────────────────────
    print("\n[1/3] Carico preprocessed.parquet ...")
    pre_path = RESULTS_DIR / "preprocessed.parquet"
    if not pre_path.exists():
        print(f"  ERRORE: {pre_path} non trovato. Esegui step01 prima.")
        sys.exit(1)

    out = pd.read_parquet(pre_path)
    out["datetime"] = pd.to_datetime(out["datetime"])

    # Verifica che le colonne arcsinh_lmp esistano
    missing = [c for c in FEAT_COLS if c not in out.columns]
    if missing:
        print(f"  ERRORE: colonne mancanti in preprocessed.parquet: {missing}")
        print(f"  Riesegui step01 dopo la modifica arcsinh.")
        sys.exit(1)

    windows, timestamps, n_skipped = build_windows(out, max_windows=N_WINDOWS)
    N = len(windows)
    print(f"  {N} finestre costruite  ({n_skipped} skipped)  "
          f"shape=({N}, {CONTEXT_LEN}, {len(FEAT_COLS)})", flush=True)

    payload = (windows.tobytes(), windows.shape, windows.dtype.str)

    # ── 2. Connetti Ray ─────────────────────────────────────────────────
    print("\n[2/3] Connessione a Ray ...")
    ray.init(
        RAY_ADDRESS,
        ignore_reinit_error=True,
        runtime_env={"pip": ["chronos-forecasting[chronos2]"]},
    )
    n_gpus = int(ray.cluster_resources().get("GPU", 0))
    n_cpus = int(ray.cluster_resources().get("CPU", 0))
    print(f"  Cluster: {n_gpus} GPU  |  {n_cpus} CPU", flush=True)

    if n_gpus == 0:
        print("  ERRORE: nessuna GPU disponibile sul cluster.")
        ray.shutdown()
        sys.exit(1)

    RemoteActor = ray.remote(num_gpus=1)(SurvivalActor)

    # ── 3. Test matrix ──────────────────────────────────────────────────
    print("\n[3/3] Test matrix ...")
    results = []

    for model in MODELS_TO_TEST:
        print(f"\n  {'─'*55}")
        print(f"  Modello: {model}")
        print(f"  {'─'*55}")

        oom_hit = False   # se OOM → salta batch size maggiori

        for bs in BATCH_SIZES:
            if oom_hit:
                results.append({
                    "model"     : model,
                    "batch_size": bs,
                    "status"    : "SKIPPED (prev OOM)",
                    "vram_peak" : float("nan"),
                    "throughput": float("nan"),
                    "nan_count" : -1,
                    "inf_count" : -1,
                    "gpu_name"  : "n/a",
                })
                print(f"  batch={bs:>2}  SKIPPED (prev OOM)")
                continue

            print(f"  batch={bs:>2}  avvio actor ...", flush=True)
            try:
                actor  = RemoteActor.remote(model)
                ref    = actor.run.remote(payload, bs)
                result = ray.get(ref, timeout=600)
            except Exception as e:
                result = {
                    "status"    : f"RAY_ERROR: {str(e)[:80]}",
                    "vram_peak" : float("nan"),
                    "throughput": 0.0,
                    "nan_count" : -1,
                    "inf_count" : -1,
                    "gpu_name"  : "n/a",
                    "n_done"    : 0,
                }

            status  = result["status"]
            vram    = result.get("vram_peak", float("nan"))
            tput    = result.get("throughput", 0.0)
            nans    = result.get("nan_count", -1)
            infs    = result.get("inf_count", -1)
            gpu     = result.get("gpu_name", "n/a")
            emb_dim = result.get("emb_dim", "?")

            stability = "✓ stable" if (nans == 0 and infs == 0) else f"✗ NaN={nans} Inf={infs}"
            vram_str  = f"{vram:.2f} GB" if not np.isnan(vram) else "n/a"
            tput_str  = f"{tput:.1f} win/s" if tput > 0 else "n/a"

            print(
                f"  batch={bs:>2}  [{status:<6}]  "
                f"VRAM={vram_str:<9}  "
                f"tput={tput_str:<12}  "
                f"{stability}  "
                f"emb_dim={emb_dim}"
            )

            results.append({
                "model"     : model,
                "batch_size": bs,
                "status"    : status,
                "vram_peak" : vram,
                "throughput": tput,
                "nan_count" : nans,
                "inf_count" : infs,
                "emb_dim"   : emb_dim,
                "gpu_name"  : gpu,
            })

            if status == "OOM":
                oom_hit = True

    ray.shutdown()

    # ── Report finale ───────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "survival_test.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\n" + "═" * 65)
    print("  SURVIVAL TEST — RISULTATI")
    print("═" * 65)
    print(df.to_string(index=False))

    # Raccomandazione
    ok_rows = df[df["status"] == "OK"].copy()
    print("\n" + "─" * 65)
    print("  RACCOMANDAZIONE")
    print("─" * 65)

    if ok_rows.empty:
        print("  ⚠  Nessuna combinazione stabile trovata.")
        print("     Verifica driver CUDA e dipendenze chronos-forecasting.")
    else:
        # Preferisci il modello più grande stabile con il batch size più alto
        ok_rows["model_size"] = ok_rows["model"].apply(
            lambda m: 1 if "large" in m else 0
        )
        best = (ok_rows
                .sort_values(["model_size", "batch_size", "throughput"],
                             ascending=[False, False, False])
                .iloc[0])
        print(f"  Modello     : {best['model']}")
        print(f"  Batch size  : {int(best['batch_size'])}")
        print(f"  VRAM peak   : {best['vram_peak']:.2f} GB / 12.0 GB")
        print(f"  Throughput  : {best['throughput']:.1f} finestre/s")
        print(f"  Embedding   : {best.get('emb_dim', '?')} dim")
        print(f"\n  → Aggiorna CHRONOS_MODEL e BATCH_SIZE in step02_embeddings.py")

    print(f"\n  CSV salvato → {out_csv}")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
