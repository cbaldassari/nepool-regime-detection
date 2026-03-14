"""
step06_montecarlo.py
====================
NEPOOL Regime Detection Pipeline — Step 06: Monte Carlo Simulation

Input  : results/transition_matrix.parquet    (K×K Markov P)
         results/mean_reversion_params.parquet (per-regime OU params from step04)
         results/regime_stats.parquet          (stationary π, median_lmp)
         results/preprocessed.parquet          (to read most recent observed log_lmp)
Output : results/mc_price_paths.parquet       (N_TRAJ × HORIZON_H  log_lmp paths)
         results/mc_regime_paths.parquet       (N_TRAJ × HORIZON_STEPS regime sequences)
         results/mc_summary.parquet            (per-hour quantiles + VaR/CVaR)
Plots  : results/step06/

Design
------
Each trajectory simulates HORIZON_H = 720 hours (30 days) forward from t=now.

Regime dynamics — checked every STRIDE_H = 6h:
    regime_{t+6h} ~ Categorical(P[regime_t, :])
    Sampled from the empirical first-order Markov transition matrix (step05).

Price dynamics — Ornstein-Uhlenbeck, Euler-Maruyama at Δt = 1h:
    log_lmp_{t+1} = log_lmp_t + θ_r (μ_r − log_lmp_t) Δt + σ_r √Δt ε_t
    ε_t ~ N(0,1)

OU parameters (θ, μ, σ) per regime from step04 (horizon_h = 1, Δt = 1h).
If step04 flagged a regime as invalid (R² < min_r2), θ is set to 0 (random walk)
and σ is the empirical log-return std of that regime.

Initial conditions:
    regime_0  : sampled from stationary distribution π
    log_lmp_0 : most recent observed log_lmp in preprocessed.parquet
                (overridable via INIT_LOG_LMP constant below)

Ray cluster: ray://datalab-rayclnt.unitus.it:10001
    N_TRAJ trajectories split into N_WORKERS chunks.
    Each @ray.remote worker receives (chunk_size, random_seed, P, ou_params,
    regime_labels, init_log_lmp, pi) → returns price array + regime array.
    Pure numpy — no GPU required.
    Results collected with ray.wait for live progress reporting.
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

RAY_ADDRESS  = "ray://datalab-rayclnt.unitus.it:10001"
N_TRAJ       = C.MONTE_CARLO["n_trajectories"]   # 1000
HORIZON_H    = C.MONTE_CARLO["horizon_h"]         # 720h (30 days)
STRIDE_H     = C.MONTE_CARLO["regime_switch_every_h"]  # 6h
HORIZON_STEPS = HORIZON_H // STRIDE_H             # 120 regime steps
DT           = 1.0                                 # simulation Δt = 1h
N_WORKERS    = 10                                  # Ray workers (no GPU needed)
CHUNK_SIZE   = N_TRAJ // N_WORKERS                 # trajectories per worker

INIT_LOG_LMP = None   # None = use most recent observed; or set a float to override
OU_HORIZON_H = 1      # use θ/μ/σ estimated at 1h horizon from step04

PLOT_DIR     = Path(C.RESULTS_DIR) / "step06"
NOISE_LABEL  = -1


# ═══════════════════════════════════════════════════════════════════════════
#  Load and prepare inputs
# ═══════════════════════════════════════════════════════════════════════════

def load_inputs() -> dict:
    """
    Load transition matrix, OU params, regime stats, and initial log_lmp.

    Returns a dict with all arrays needed by the Ray workers.
    Everything is converted to plain Python / numpy for Ray serialisation.
    """
    base = Path(C.RESULTS_DIR)

    # ── Transition matrix ────────────────────────────────────────────────────
    P_df    = pd.read_parquet(base / "transition_matrix.parquet")
    labels  = [int(c.replace("to_", "")) for c in P_df.columns]
    P       = P_df.values.astype(np.float64)
    K       = len(labels)

    # ── Stationary distribution π ────────────────────────────────────────────
    stats_df = pd.read_parquet(base / "regime_stats.parquet")
    pi       = np.array([
        float(stats_df.loc[stats_df["regime"] == r, "stationary_prob"].values[0])
        for r in labels
    ])
    pi /= pi.sum()   # ensure sums to 1

    # ── OU parameters (θ, μ, σ) per regime at horizon_h = OU_HORIZON_H ──────
    ou_df = pd.read_parquet(base / "mean_reversion_params.parquet")
    ou_h  = ou_df[ou_df["horizon_h"] == OU_HORIZON_H].copy()

    ou_params = {}
    for r in labels:
        row = ou_h[ou_h["regime"] == r]
        if len(row) == 0 or not bool(row.iloc[0].get("valid", True)):
            # Fallback: random walk with regime median σ
            stat_row = stats_df[stats_df["regime"] == r]
            sigma_fb = float(stat_row["mean_lmp"].values[0]) * 0.02  # 2% of mean LMP
            ou_params[r] = {"theta": 0.0,
                            "mu":    float(np.log(max(stat_row["median_lmp"].values[0], 1e-3))),
                            "sigma": sigma_fb,
                            "valid": False}
        else:
            ou_params[r] = {"theta": float(row.iloc[0]["theta"]),
                            "mu":    float(row.iloc[0]["mu"]),
                            "sigma": float(row.iloc[0]["sigma"]),
                            "valid": True}

    # ── Initial log_lmp ──────────────────────────────────────────────────────
    if INIT_LOG_LMP is not None:
        init_log_lmp = float(INIT_LOG_LMP)
    else:
        pre = pd.read_parquet(base / "preprocessed.parquet")
        init_log_lmp = float(pre["log_lmp"].dropna().iloc[-1])

    print(f"  Initial log_lmp  : {init_log_lmp:.4f}  "
          f"(LMP ≈ {np.exp(init_log_lmp):.2f} $/MWh)")
    print(f"  OU params source : horizon_h = {OU_HORIZON_H}h")
    for r in labels:
        p = ou_params[r]
        tag = "✓" if p["valid"] else "⚠ fallback"
        print(f"    R{r}: θ={p['theta']:.4f}  μ={p['mu']:.4f}  "
              f"σ={p['sigma']:.4f}   [{tag}]")

    return {
        "P":            P,
        "labels":       labels,
        "K":            K,
        "pi":           pi,
        "ou_params":    ou_params,
        "init_log_lmp": init_log_lmp,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Ray worker
# ═══════════════════════════════════════════════════════════════════════════

@staticmethod
def _simulate_chunk(
    chunk_size:   int,
    seed:         int,
    P:            np.ndarray,
    labels:       list,
    pi:           np.ndarray,
    ou_params:    dict,
    init_log_lmp: float,
    horizon_h:    int,
    stride_h:     int,
    dt:           float,
) -> tuple:
    """
    Core simulation logic — runs inside each Ray worker.

    Simulates `chunk_size` independent trajectories.

    Returns
    -------
    price_paths  : (chunk_size, horizon_h)  float32  — log_lmp at each hour
    regime_paths : (chunk_size, horizon_h // stride_h)  int16  — regime at each step
    """
    rng           = np.random.default_rng(seed)
    K             = len(labels)
    horizon_steps = horizon_h // stride_h
    label_arr     = np.array(labels, dtype=np.int32)

    # Cumulative probability rows for Categorical sampling (vectorised)
    P_cdf = np.cumsum(P, axis=1)

    price_paths  = np.zeros((chunk_size, horizon_h),    dtype=np.float32)
    regime_paths = np.zeros((chunk_size, horizon_steps), dtype=np.int16)

    # Theta / mu / sigma arrays indexed by label position
    theta_arr = np.array([ou_params[r]["theta"] for r in labels], dtype=np.float64)
    mu_arr    = np.array([ou_params[r]["mu"]    for r in labels], dtype=np.float64)
    sigma_arr = np.array([ou_params[r]["sigma"] for r in labels], dtype=np.float64)

    for traj in range(chunk_size):
        # ── Initial conditions ───────────────────────────────────────────────
        reg_idx = int(rng.choice(K, p=pi))           # sample from π
        log_x   = float(init_log_lmp)

        h = 0   # current hour pointer
        for step in range(horizon_steps):
            # Record current regime
            regime_paths[traj, step] = label_arr[reg_idx]

            # Simulate OU for stride_h hours under regime reg_idx
            theta = theta_arr[reg_idx]
            mu    = mu_arr[reg_idx]
            sigma = sigma_arr[reg_idx]
            noise = rng.standard_normal(stride_h) * (sigma * np.sqrt(dt))

            for k in range(stride_h):
                if h < horizon_h:
                    log_x          += theta * (mu - log_x) * dt + noise[k]
                    price_paths[traj, h] = log_x
                    h += 1

            # Sample next regime from P[reg_idx, :]
            u       = rng.random()
            reg_idx = int(np.searchsorted(P_cdf[reg_idx], u))
            reg_idx = min(reg_idx, K - 1)   # clamp for float rounding

    return price_paths, regime_paths


def make_ray_worker():
    """
    Build and return the Ray remote function.
    Defined at runtime to avoid serialisation issues with module-level decorators.
    """
    import ray

    @ray.remote
    def simulate_chunk(
        chunk_size, seed, P, labels, pi, ou_params,
        init_log_lmp, horizon_h, stride_h, dt,
    ):
        return _simulate_chunk(
            chunk_size, seed, P, labels, pi, ou_params,
            init_log_lmp, horizon_h, stride_h, dt,
        )

    return simulate_chunk


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def run_montecarlo(inp: dict) -> tuple:
    """
    Dispatch all chunks to the Ray cluster and collect results.

    Returns (price_paths, regime_paths) as full (N_TRAJ, …) arrays.
    """
    import ray
    from tqdm import tqdm

    ray.init(RAY_ADDRESS, ignore_reinit_error=True)
    simulate_chunk = make_ray_worker()

    # Put large shared objects in Ray object store once
    P_ref         = ray.put(inp["P"])
    ou_ref        = ray.put(inp["ou_params"])
    pi_ref        = ray.put(inp["pi"])

    # Dispatch all chunks simultaneously
    futures = []
    for w in range(N_WORKERS):
        chunk = CHUNK_SIZE + (N_TRAJ % N_WORKERS if w == N_WORKERS - 1 else 0)
        fut = simulate_chunk.remote(
            chunk,
            seed         = C.RANDOM_STATE + w,
            P            = P_ref,
            labels       = inp["labels"],
            pi           = pi_ref,
            ou_params    = ou_ref,
            init_log_lmp = inp["init_log_lmp"],
            horizon_h    = HORIZON_H,
            stride_h     = STRIDE_H,
            dt           = DT,
        )
        futures.append(fut)

    print(f"\n  Dispatched {N_TRAJ} trajectories across {N_WORKERS} Ray workers")
    print(f"  Cluster : {RAY_ADDRESS}")

    # Collect in completion order
    price_chunks  = []
    regime_chunks = []
    pending       = list(futures)

    with tqdm(total=N_WORKERS, desc="  Collecting", ncols=60,
              bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        while pending:
            done, pending = ray.wait(pending, num_returns=1, timeout=120)
            for f in done:
                prices, regimes = ray.get(f)
                price_chunks.append(prices)
                regime_chunks.append(regimes)
                pbar.update(1)

    ray.shutdown()

    price_paths  = np.concatenate(price_chunks,  axis=0)   # (N_TRAJ, HORIZON_H)
    regime_paths = np.concatenate(regime_chunks, axis=0)   # (N_TRAJ, HORIZON_STEPS)

    return price_paths, regime_paths


# ═══════════════════════════════════════════════════════════════════════════
#  Summary statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_summary(price_paths: np.ndarray) -> pd.DataFrame:
    """
    Per-hour quantile summary of log_lmp paths, plus VaR and CVaR in LMP space.

    Columns:
      hour, q05, q25, q50, q75, q95   — log_lmp quantiles
      lmp_q05 … lmp_q95               — exp(log_lmp) quantiles
      lmp_mean, lmp_std               — mean and std of LMP distribution
      var_95                          — 95% Value-at-Risk (5th pct of LMP)
      cvar_95                         — 95% CVaR (expected LMP below VaR)
    """
    N, H    = price_paths.shape
    lmp_paths = np.exp(price_paths.astype(np.float64))

    records = []
    for h in range(H):
        lmp_h = lmp_paths[:, h]
        log_h = price_paths[:, h]
        q05   = float(np.percentile(lmp_h, 5))
        records.append({
            "hour":    h + 1,
            "q05":     float(np.percentile(log_h, 5)),
            "q25":     float(np.percentile(log_h, 25)),
            "q50":     float(np.percentile(log_h, 50)),
            "q75":     float(np.percentile(log_h, 75)),
            "q95":     float(np.percentile(log_h, 95)),
            "lmp_q05": q05,
            "lmp_q25": float(np.percentile(lmp_h, 25)),
            "lmp_q50": float(np.percentile(lmp_h, 50)),
            "lmp_q75": float(np.percentile(lmp_h, 75)),
            "lmp_q95": float(np.percentile(lmp_h, 95)),
            "lmp_mean": float(lmp_h.mean()),
            "lmp_std":  float(lmp_h.std()),
            "var_95":   q05,                                   # 5th pct = 95% VaR (downside)
            "cvar_95":  float(lmp_h[lmp_h <= q05].mean())
                        if (lmp_h <= q05).any() else q05,
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_price_fan(summary: pd.DataFrame, init_log_lmp: float):
    """
    01 — Fan chart of LMP distribution over the forecast horizon.

    Shaded bands: [5%-95%] (light), [25%-75%] (medium), median (solid).
    Initial LMP shown as reference point at hour 0.
    """
    h    = summary["hour"].values
    init = float(np.exp(init_log_lmp))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col_base, ylabel, transform in [
        (axes[0], "lmp", "LMP $/MWh",    lambda x: x),
        (axes[1], "q",   "log LMP",       lambda x: x),
    ]:
        if col_base == "lmp":
            q05 = summary["lmp_q05"].values
            q25 = summary["lmp_q25"].values
            q50 = summary["lmp_q50"].values
            q75 = summary["lmp_q75"].values
            q95 = summary["lmp_q95"].values
            y0  = init
        else:
            q05 = summary["q05"].values
            q25 = summary["q25"].values
            q50 = summary["q50"].values
            q75 = summary["q75"].values
            q95 = summary["q95"].values
            y0  = init_log_lmp

        ax.axhline(y0, color="grey", lw=0.8, ls="--", alpha=0.6, label="Initial value")
        ax.fill_between(h, q05, q95, alpha=0.20, color="#2c6fad", label="5%–95%")
        ax.fill_between(h, q25, q75, alpha=0.40, color="#2c6fad", label="25%–75%")
        ax.plot(h, q50, color="#1a3f6f", lw=1.5, label="Median")
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    axes[0].set_title("LMP forecast fan chart  ($/MWh)", fontweight="bold")
    axes[1].set_title("log-LMP forecast fan chart",      fontweight="bold")
    fig.suptitle(
        f"Monte Carlo forecast — {N_TRAJ} trajectories, horizon = {HORIZON_H}h",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_price_fan.png")
    plt.close(fig)
    print("  [plot] 01_price_fan.png")


def plot_terminal_distribution(price_paths: np.ndarray):
    """
    02 — Histogram of terminal LMP (at HORIZON_H) with VaR / CVaR annotations.
    """
    terminal_lmp = np.exp(price_paths[:, -1].astype(np.float64))
    var95  = float(np.percentile(terminal_lmp, 5))
    cvar95 = float(terminal_lmp[terminal_lmp <= var95].mean())
    median = float(np.median(terminal_lmp))

    clip = float(np.percentile(terminal_lmp, 99))
    plot_data = terminal_lmp[terminal_lmp <= clip]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(plot_data, bins=60, color="#5a9e6f", edgecolor="white", alpha=0.8)
    ax.axvline(var95,  color="red",    lw=1.5, ls="--", label=f"VaR 95% = ${var95:.2f}")
    ax.axvline(cvar95, color="darkred",lw=1.5, ls=":",  label=f"CVaR 95% = ${cvar95:.2f}")
    ax.axvline(median, color="navy",   lw=1.5,          label=f"Median = ${median:.2f}")
    ax.set_xlabel(f"Terminal LMP at hour {HORIZON_H}  ($/MWh, clipped 99th pct)")
    ax.set_ylabel("Trajectory count")
    ax.set_title(
        f"Terminal LMP distribution — {N_TRAJ} Monte Carlo trajectories",
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_terminal_distribution.png")
    plt.close(fig)
    print("  [plot] 02_terminal_distribution.png")


def plot_sample_paths(price_paths: np.ndarray, n_show: int = 50, seed: int = 42):
    """
    03 — Spaghetti plot of n_show randomly selected trajectories.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(price_paths), min(n_show, len(price_paths)), replace=False)
    lmp_show = np.exp(price_paths[idx].astype(np.float64))
    h        = np.arange(1, HORIZON_H + 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(len(idx)):
        ax.plot(h, lmp_show[i], lw=0.5, alpha=0.4, color="#2c6fad")

    med = np.exp(np.median(price_paths, axis=0))
    ax.plot(h, med, color="black", lw=1.5, label="Median trajectory")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("LMP $/MWh")
    ax.set_title(f"Sample Monte Carlo paths ({n_show} of {N_TRAJ}  shown)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_sample_paths.png")
    plt.close(fig)
    print("  [plot] 03_sample_paths.png")


def plot_regime_frequency(regime_paths: np.ndarray, labels: list):
    """
    04 — Fraction of trajectories in each regime at each time step.

    Shows how the regime distribution evolves from the initial draw (from π)
    toward the stationary distribution as the forecast horizon extends.
    """
    K     = len(labels)
    steps = regime_paths.shape[1]
    freq  = np.zeros((steps, K), dtype=np.float64)

    for ki, r in enumerate(labels):
        freq[:, ki] = (regime_paths == r).mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    bottom  = np.zeros(steps)
    x       = np.arange(steps) * STRIDE_H

    palette = plt.cm.tab10
    for ki, r in enumerate(labels):
        ax.fill_between(x, bottom, bottom + freq[:, ki],
                        alpha=0.75, label=f"Regime {r}",
                        color=palette(ki / K))
        bottom += freq[:, ki]

    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Fraction of trajectories")
    ax.set_title(
        "Regime frequency over forecast horizon\n"
        "(should converge to stationary distribution π as horizon → ∞)",
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper right", fontsize=8, ncol=min(K, 5))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_regime_frequency.png")
    plt.close(fig)
    print("  [plot] 04_regime_frequency.png")


def plot_var_curve(summary: pd.DataFrame):
    """
    05 — VaR and CVaR curves over the forecast horizon.
    """
    h = summary["hour"].values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(h, summary["lmp_q50"],  color="navy",    lw=1.5,         label="Median LMP")
    ax.plot(h, summary["var_95"],   color="red",     lw=1.2, ls="--",label="VaR 95% (5th pct)")
    ax.plot(h, summary["cvar_95"],  color="darkred", lw=1.2, ls=":", label="CVaR 95%")
    ax.fill_between(h, summary["lmp_q05"], summary["lmp_q95"],
                    alpha=0.12, color="navy")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("LMP $/MWh")
    ax.set_title("Value-at-Risk and CVaR curves over forecast horizon",
                 fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_var_curve.png")
    plt.close(fig)
    print("  [plot] 05_var_curve.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 06: Monte Carlo Simulation")
    print("═" * 65)
    print(f"  Ray cluster  : {RAY_ADDRESS}")
    print(f"  Trajectories : {N_TRAJ}")
    print(f"  Horizon      : {HORIZON_H}h  ({HORIZON_H // 24} days)")
    print(f"  Regime step  : every {STRIDE_H}h  ({HORIZON_STEPS} steps)")
    print(f"  Price step   : Δt = {DT}h  (Euler-Maruyama OU)")
    print(f"  Workers      : {N_WORKERS}  ×  {CHUNK_SIZE} trajectories each")
    print(f"  Output       : {C.RESULTS_DIR}/mc_*.parquet")
    print(f"  Plots        : {PLOT_DIR}/")
    print("─" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    # ── 1. Load inputs ───────────────────────────────────────────────────────
    print("\n  [1/5] Loading inputs …")
    inp = load_inputs()

    # ── 2. Monte Carlo via Ray ───────────────────────────────────────────────
    print("\n  [2/5] Running Monte Carlo on Ray cluster …")
    price_paths, regime_paths = run_montecarlo(inp)

    print(f"\n  price_paths  shape : {price_paths.shape}")
    print(f"  regime_paths shape : {regime_paths.shape}")

    # ── 3. Summary statistics ────────────────────────────────────────────────
    print("\n  [3/5] Computing summary statistics …")
    summary = compute_summary(price_paths)

    # ── 4. Save outputs ──────────────────────────────────────────────────────
    print("\n  [4/5] Saving outputs …")
    base = Path(C.RESULTS_DIR)

    # Price paths: store as float32 parquet with hour columns
    hours  = [f"h{i+1:03d}" for i in range(HORIZON_H)]
    pp_df  = pd.DataFrame(price_paths.astype(np.float32), columns=hours)
    pp_df.insert(0, "traj_id", np.arange(N_TRAJ))
    pp_df.to_parquet(base / "mc_price_paths.parquet", index=False)
    print(f"  Saved: mc_price_paths.parquet   ({pp_df.shape[0]}×{pp_df.shape[1]-1} hours)")

    # Regime paths
    steps  = [f"s{i+1:03d}" for i in range(HORIZON_STEPS)]
    rp_df  = pd.DataFrame(regime_paths.astype(np.int16), columns=steps)
    rp_df.insert(0, "traj_id", np.arange(N_TRAJ))
    rp_df.to_parquet(base / "mc_regime_paths.parquet", index=False)
    print(f"  Saved: mc_regime_paths.parquet  ({rp_df.shape[0]}×{rp_df.shape[1]-1} steps)")

    # Summary
    summary.to_parquet(base / "mc_summary.parquet", index=False)
    print(f"  Saved: mc_summary.parquet       ({len(summary)} rows)")

    # ── 5. Plots ─────────────────────────────────────────────────────────────
    print("\n  [5/5] Generating plots …")
    plot_price_fan(summary, inp["init_log_lmp"])
    plot_terminal_distribution(price_paths)
    plot_sample_paths(price_paths)
    plot_regime_frequency(regime_paths, inp["labels"])
    plot_var_curve(summary)

    # ── Report ───────────────────────────────────────────────────────────────
    elapsed     = time.time() - t0
    terminal    = np.exp(price_paths[:, -1].astype(np.float64))
    var95       = float(np.percentile(terminal, 5))
    cvar95      = float(terminal[terminal <= var95].mean())
    median_term = float(np.median(terminal))

    print("\n" + "─" * 65)
    print("  MONTE CARLO REPORT")
    print("─" * 65)
    print(f"  Trajectories       : {N_TRAJ:,}")
    print(f"  Horizon            : {HORIZON_H}h  ({HORIZON_H // 24} days)")
    print(f"  Initial LMP        : {np.exp(inp['init_log_lmp']):.2f} $/MWh")
    print()
    print(f"  Terminal LMP distribution (hour {HORIZON_H}):")
    print(f"    Median            : {median_term:.2f} $/MWh")
    print(f"    5th pct  (VaR 95%): {var95:.2f} $/MWh")
    print(f"    CVaR 95%          : {cvar95:.2f} $/MWh")
    print(f"    95th pct          : {float(np.percentile(terminal, 95)):.2f} $/MWh")
    print()
    print(f"  Elapsed            : {elapsed:.1f}s")
    print("─" * 65)
    print(f"  mc_price_paths.parquet  → {base / 'mc_price_paths.parquet'}")
    print(f"  mc_regime_paths.parquet → {base / 'mc_regime_paths.parquet'}")
    print(f"  mc_summary.parquet      → {base / 'mc_summary.parquet'}")
    print(f"  Plots                   → {PLOT_DIR}/")
    print()
    print("  Next → Step 07: Backtest")
    print("═" * 65 + "\n")

    return price_paths, regime_paths, summary


if __name__ == "__main__":
    main()
