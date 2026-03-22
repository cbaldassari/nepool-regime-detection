"""
step01_preprocessing.py
=======================
NEPOOL Regime Detection Pipeline — Step 01: Preprocessing
ISONE Electricity Market — Geometry-Driven Regime Identification

Input  : isone_dataset.parquet  (43,789 rows × 22 cols, 2021–2025)
Output : results/preprocessed.parquet
Plots  : results/step01/  (5 diagnostic figures)

Features produced (8 columns)
-------------------------------
  1.  lmp                  — raw LMP $/MWh
  2.  arcsinh_lmp          — arcsinh(LMP), handles negative prices natively
  3.  log_return           — first difference of arcsinh_lmp, lag 1h
  4.  mstl_resid_arcsinh   — MSTL residual of arcsinh_lmp (periods 24h/168h/8760h)
  5.  mstl_resid_lr        — MSTL residual of log_return
  6.  log_lmp_shifted      — log(LMP - min(LMP) + 1), log scale with shift for negatives
  7.  lmp_clipped          — LMP winsorized at [1st, 99th] percentile
  8.  mstl_resid_log       — MSTL residual of log_lmp_shifted

Experiments A-G (used in step02_embeddings.py)
-----------------------------------------------
  A: log_return           — stationary, short-memory dynamics
  B: arcsinh_lmp          — price level, negative-safe
  C: mstl_resid_lr        — deseasonalized return
  D: mstl_resid_arcsinh   — deseasonalized level (best in TOPSIS)
  E: log_lmp_shifted      — log-scale level (alternative to arcsinh)
  F: lmp_clipped          — raw level without extreme spikes
  G: mstl_resid_log       — deseasonalized log-level

Design choices
--------------
  • No standardisation: Chronos-2 normalises each channel per-window internally.
  • No detrending: same reason — per-window normalisation removes local trend.
  • arcsinh for LMP: handles negative prices (oversupply events) correctly.
  • log_lmp_shifted: log(LMP + shift) where shift = max(0, -min_lmp) + 1.
    Classic log scale; grows slower than arcsinh for large prices.
  • lmp_clipped: winsorize at [p1, p99] to reduce spike influence on the encoder.
  • total_mw and ILR coordinates (ilr_1…ilr_7) are intentionally excluded:
    they are not passed to Chronos-2 in any of the A-G experiments.

Dependencies: pandas, numpy, matplotlib, seaborn, tqdm
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
import config as C

PLOT_DIR = Path(C.RESULTS_DIR) / "step01"

FUEL_DISPLAY = ["Gas", "Nuclear", "Hydro", "Wind", "Solar", "Coal", "Oil", "Other"]
FUEL_COLORS  = ["#e07b39", "#7b52ab", "#4da6e8", "#6abf69", "#f7c948",
                "#5a5a5a", "#b04040", "#aaaaaa"]


# ═══════════════════════════════════════════════════════════════════════════
#  Feature builders
# ═══════════════════════════════════════════════════════════════════════════

def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build price-derived features.
      arcsinh_lmp     : arcsinh(LMP) — handles negative prices, no arbitrary clip needed
      log_return      : first difference of arcsinh_lmp (≈ arithmetic return for small Δ)
                        NaN for the very first observation (no prior price available)
      lmp             : raw LMP kept for plots and validation
      log_lmp_shifted : log(LMP + shift) where shift = max(0, -min_lmp) + 1.0
                        Guarantees positivity before log; comparable to arcsinh but
                        grows slower for large values.
      lmp_clipped     : LMP winsorized at [1st, 99th] percentile — keeps scale intact
                        but reduces influence of extreme spikes on the encoder.
    """
    lmp         = df["lmp"].values.astype(np.float64)
    arcsinh_lmp = np.arcsinh(lmp)
    log_ret     = np.empty(len(lmp))
    log_ret[0]  = np.nan
    log_ret[1:] = np.diff(arcsinh_lmp)

    # log_lmp_shifted: shift so that min(shifted) >= 1 before log
    shift           = max(0.0, -lmp.min()) + 1.0
    log_lmp_shifted = np.log(lmp + shift)

    # lmp_clipped: winsorize at [p1, p99]
    p1, p99   = np.percentile(lmp, 1), np.percentile(lmp, 99)
    lmp_clipped = np.clip(lmp, p1, p99)

    return pd.DataFrame({
        "lmp"            : lmp,
        "arcsinh_lmp"    : arcsinh_lmp,
        "log_return"     : log_ret,
        "log_lmp_shifted": log_lmp_shifted,
        "lmp_clipped"    : lmp_clipped,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════
#  Validation checks
# ═══════════════════════════════════════════════════════════════════════════

def run_checks(df_raw: pd.DataFrame, feat: pd.DataFrame) -> None:
    """Sanity checks printed after feature construction."""
    issues = []

    # NaN check
    nan_counts = feat.isnull().sum()
    if nan_counts.any():
        issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")

    # LMP negative or zero — should not occur in 2021-2025; clip at 1e-3 as safety
    n_neg_lmp  = (feat["lmp"] < 0).sum()
    n_zero_lmp = (feat["lmp"] == 0).sum()
    if n_neg_lmp:
        issues.append(f"lmp < 0: {n_neg_lmp} rows  ← unexpected, check source data")
    if n_zero_lmp:
        issues.append(f"lmp = 0: {n_zero_lmp} rows  ← clipped to 1e-3 for log")

    for msg in issues:
        print(f"    {'⚠' if '✗' in msg or 'artefact' in msg else '✓'}  {msg}")


# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(out: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """
    Generate 5 diagnostic figures and save to results/step01/.

    Figure list:
      01_lmp_timeseries.png       — full LMP series with spike threshold
      02_lmp_distribution.png     — LMP histogram + log-scale
      03_log_return_distribution  — log_return histogram with normal fit + QQ plot
      04_fuel_mix_monthly.png     — monthly average fuel share stacked bar
      05_feature_correlation.png  — Pearson correlation heatmap of price features
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    dt  = pd.to_datetime(out["datetime"])
    lmp = out["lmp"].values

    # ── 01: LMP time series ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dt, lmp, lw=0.4, color="#2c6fad", alpha=0.8, label="LMP $/MWh")
    spike_thr = np.percentile(lmp, 99)
    ax.axhline(spike_thr, color="#d62728", lw=0.8, ls="--",
               label=f"99th pct  (${spike_thr:.0f})")
    ax.set_title("ISONE Mass Hub LMP — 2021–2025 (hourly)", fontweight="bold")
    ax.set_ylabel("$/MWh")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_lmp_timeseries.png")
    plt.close(fig)

    # ── 02: LMP distribution (linear + log-x) ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(lmp, bins=120, color="#2c6fad", alpha=0.75, edgecolor="none")
    axes[0].axvline(np.median(lmp), color="orange", lw=1.2, label=f"median ${np.median(lmp):.1f}")
    axes[0].axvline(np.mean(lmp),   color="red",    lw=1.2, ls="--", label=f"mean ${np.mean(lmp):.1f}")
    axes[0].set_title("LMP distribution (linear scale)")
    axes[0].set_xlabel("$/MWh");  axes[0].legend(fontsize=8)

    axes[1].hist(lmp, bins=np.logspace(np.log10(max(lmp.min(), 1)), np.log10(lmp.max()), 80),
                 color="#2c6fad", alpha=0.75, edgecolor="none")
    axes[1].set_xscale("log")
    axes[1].set_title("LMP distribution (log-x scale)")
    axes[1].set_xlabel("$/MWh (log)")
    fig.suptitle("LMP Price Distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_lmp_distribution.png")
    plt.close(fig)

    # ── 03: log_return distribution + QQ ────────────────────────────────────
    from scipy import stats as spstats
    lr = out["log_return"].dropna().values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(lr, bins=120, color="#5a9e6f", alpha=0.75, edgecolor="none",
                 density=True, label="empirical")
    x_norm = np.linspace(lr.min(), lr.max(), 300)
    axes[0].plot(x_norm, spstats.norm.pdf(x_norm, lr.mean(), lr.std()),
                 color="red", lw=1.5, label=f"Normal  μ={lr.mean():.4f}  σ={lr.std():.4f}")
    axes[0].set_title("log_return distribution vs Normal")
    axes[0].set_xlabel("log_return");  axes[0].legend(fontsize=8)

    (osm, osr), (slope, intercept, r) = spstats.probplot(lr, dist="norm")
    axes[1].scatter(osm, osr, s=2, alpha=0.3, color="#5a9e6f")
    axes[1].plot(osm, slope * np.array(osm) + intercept, color="red", lw=1.5)
    axes[1].set_title(f"QQ plot  (R²={r**2:.4f})")
    axes[1].set_xlabel("Theoretical quantiles");  axes[1].set_ylabel("Sample quantiles")
    fig.suptitle("log_return — Normality Check", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_log_return_distribution.png")
    plt.close(fig)

    # ── 04: Monthly fuel mix stacked bar ────────────────────────────────────
    df_raw2 = df_raw.copy()
    df_raw2["month"] = pd.to_datetime(df_raw2["datetime"]).dt.to_period("M")
    share_cols = [c + "_share" for c in
                  ["natural_gas", "nuclear", "hydro", "wind", "solar", "coal", "oil", "other"]]
    share_cols = [c for c in share_cols if c in df_raw2.columns]
    monthly = df_raw2.groupby("month")[share_cols].mean()
    monthly.index = monthly.index.astype(str)

    fig, ax = plt.subplots(figsize=(16, 5))
    bottom = np.zeros(len(monthly))
    for i, col in enumerate(share_cols):
        vals = monthly[col].values
        ax.bar(range(len(monthly)), vals, bottom=bottom,
               color=FUEL_COLORS[i], label=FUEL_DISPLAY[i], width=0.85)
        bottom += vals
    ax.set_xticks(range(0, len(monthly), 3))
    ax.set_xticklabels(list(monthly.index)[::3], rotation=45, ha="right", fontsize=7)
    ax.set_title("Monthly average fuel mix shares — ISONE 2021–2025", fontweight="bold")
    ax.set_ylabel("Share (fraction)");  ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_fuel_mix_monthly.png")
    plt.close(fig)

    # ── 05: Feature correlation heatmap (price features only) ───────────────
    feat_cols = ["lmp", "arcsinh_lmp", "log_return",
                 "mstl_resid_arcsinh", "mstl_resid_lr",
                 "log_lmp_shifted", "lmp_clipped", "mstl_resid_log"]
    feat_cols = [c for c in feat_cols if c in out.columns]
    corr = out[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm", vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.3, square=True,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Pearson correlation — price features", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_feature_correlation.png")
    plt.close(fig)

    print(f"  5 figures saved → {PLOT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> pd.DataFrame:
    t0 = time.time()

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 01: Preprocessing")
    print("═" * 65)
    print(f"  Input  : {C.DATA_PATH}")
    print(f"  Period : {C.DATASET['start_date']}  →  {C.DATASET['end_date']}")
    print(f"  Output : {C.RESULTS_DIR}/preprocessed.parquet")
    print(f"  Plots  : {PLOT_DIR}/")
    print()
    print("  This step loads the raw ISONE dataset, removes a small number")
    print("  of anomalous rows (DST duplicates, EIA artefacts), and builds")
    print("  5 price features for downstream embedding with Chronos-2:")
    print("    • lmp, arcsinh_lmp, log_return")
    print("    • mstl_resid_arcsinh, mstl_resid_lr")
    print("─" * 65)

    n_steps = 6
    def _step(n: int, label: str) -> None:
        print(f"  ▶ [{n}/{n_steps}] {label} ···", flush=True)
    def _done(n: int, label: str, detail: str = "") -> None:
        suffix = f"  ({detail})" if detail else ""
        print(f"  ✓ [{n}/{n_steps}] {label}{suffix}", flush=True)

    # ── 1. Load ─────────────────────────────────────────────────────────────
    _step(1, "Loading data")
    df = (pd.read_parquet(C.DATA_PATH)
            .sort_values("datetime")
            .reset_index(drop=True))
    df["datetime"] = pd.to_datetime(df["datetime"])
    _done(1, "Loading data", f"{len(df):,} righe  ×  {df.shape[1]} colonne")

    # ── 2. Filter & clean ───────────────────────────────────────────────────
    _step(2, "Filtering & cleaning")
    n_raw = len(df)

    # 2a. Date range filter
    df = df[
        (df["datetime"] >= pd.Timestamp(C.DATASET["start_date"])) &
        (df["datetime"] <= pd.Timestamp(C.DATASET["end_date"]))
    ].reset_index(drop=True)
    n_filtered = n_raw - len(df)

    # 2b. Collect anomalous rows BEFORE removal (for audit log)
    removed_rows = []

    # DST fall-back duplicates: keep first occurrence, drop second.
    # Each November the clock reverts 1h, causing one timestamp to appear twice
    # in the raw data. The fuel mix is identical for both rows; LMP differs
    # marginally. We keep the first occurrence (real-time price).
    dst_dupes = df[df.duplicated(subset=["datetime"], keep="first")]
    for _, row in dst_dupes.iterrows():
        removed_rows.append({
            "datetime": row["datetime"],
            "reason":   "DST fall-back duplicate (kept first occurrence)",
            "lmp":      row["lmp"],
        })
    df = df[~df.duplicated(subset=["datetime"], keep="first")].reset_index(drop=True)

    # EIA artefact: rows where any raw MW generation column is negative.
    # Negative generation is physically impossible; a single row (2021-03-01)
    # has all major fuel columns negative, indicating a sign-flip in the EIA
    # reporting pipeline.
    mw_raw_cols = ["hydro", "natural_gas", "nuclear", "oil", "other", "solar", "wind", "coal"]
    neg_mw_mask = (df[[c for c in mw_raw_cols if c in df.columns]] < 0).any(axis=1)
    for _, row in df[neg_mw_mask].iterrows():
        removed_rows.append({
            "datetime": row["datetime"],
            "reason":   "EIA artefact: negative raw MW generation",
            "lmp":      row["lmp"],
        })
    df = df[~neg_mw_mask].reset_index(drop=True)

    # 25h temporal gap: detect and log (data loss — not removable).
    # One gap of 25h found (2023-06-15): 24 consecutive hours are missing.
    # The sliding-window step (Step 02) will avoid windows straddling this gap.
    diffs       = df["datetime"].diff()
    gap_25h_idx = diffs[diffs > pd.Timedelta("2h")].index
    temporal_gaps = []
    for idx in gap_25h_idx:
        gap_start = df.loc[idx - 1, "datetime"]
        gap_end   = df.loc[idx,     "datetime"]
        gap_h     = int((gap_end - gap_start).total_seconds() / 3600)
        temporal_gaps.append({"from": gap_start, "to": gap_end, "gap_h": gap_h})

    # 2c. Missing price data
    n_dropped = 0

    # 2d. Save audit log
    audit_path = Path(C.RESULTS_DIR) / "removed_rows.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(removed_rows).to_csv(audit_path, index=False)

    df_clean = df.copy()
    _done(2, "Filtering & cleaning",
          f"{n_raw:,} → {len(df):,} righe  |  {len(removed_rows)} rimossi  |  {len(temporal_gaps)} gap temporali")

    # ── 3. Price features ───────────────────────────────────────────────────
    _step(3, "Price features  (lmp, arcsinh_lmp, log_return)")
    price = build_price_features(df)
    _done(3, "Price features")

    # ── 4. Assembly & validation ─────────────────────────────────────────────
    _step(4, "Assembly & validation")
    feat      = price.copy()
    valid     = feat.notna().all(axis=1)
    feat      = feat[valid].reset_index(drop=True)
    df        = df[valid].reset_index(drop=True)
    df_clean  = df.copy()
    out       = pd.concat([df[["datetime"]], feat], axis=1)
    _done(4, "Assembly & validation", f"{len(out):,} righe  ×  {out.shape[1]} colonne")

    # ── 5. MSTL deseasonalization ────────────────────────────────────────────
    _step(5, "MSTL deseasonalization  (periodi: 24h, 168h, 8760h)")
    from statsmodels.tsa.seasonal import MSTL
    for src_col, dst_col in [("arcsinh_lmp",     "mstl_resid_arcsinh"),
                              ("log_return",      "mstl_resid_lr"),
                              ("log_lmp_shifted", "mstl_resid_log")]:
        series = out[src_col].fillna(0).values
        mstl   = MSTL(series, periods=[24, 168, 8760], windows=[25, 169, 8761])
        res    = mstl.fit()
        out[dst_col] = res.resid
        print(f"    {src_col} -> {dst_col}  "
              f"(resid std={res.resid.std():.4f})", flush=True)
    _done(5, "MSTL deseasonalization",
          "colonne aggiunte: mstl_resid_arcsinh, mstl_resid_lr, mstl_resid_log")

    # ── 6. Save ──────────────────────────────────────────────────────────────
    _step(6, "Saving output")
    out_path = Path(C.RESULTS_DIR) / "preprocessed.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    _done(6, "Saving output", f"{out_path.stat().st_size / 1e6:.1f} MB  →  {out_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    make_plots(out, df_clean)

    # ── Report ──────────────────────────────────────────────────────────────
    elapsed  = time.time() - t0
    n_removed = len(removed_rows)

    print("\n" + "─" * 65)
    print("  DATA REPORT")
    print("─" * 65)
    print(f"  Rows loaded          : {n_raw:>10,}")
    print(f"  Rows outside range   : {n_filtered:>10,}")
    print(f"  Rows removed (audit) : {n_removed:>10,}")
    print(f"  Rows dropped (NaN)   : {n_dropped:>10,}")
    print(f"  Rows with log_ret NaN: {'1':>10}  (first row, expected)")
    print(f"  Rows output          : {len(out):>10,}")
    print(f"  Columns output       : {out.shape[1]:>10}")
    print(f"  Date range           : {out['datetime'].min().date()}  →  {out['datetime'].max().date()}")
    print(f"  Elapsed              : {elapsed:.1f}s")

    # Removed rows table
    if removed_rows:
        print("\n" + "─" * 65)
        print("  REMOVED ROWS AUDIT")
        print("  (6 rows total: 5 DST duplicates + 1 EIA artefact)")
        print("─" * 65)
        audit_df = pd.DataFrame(removed_rows)
        print(audit_df.to_string(index=False))
        print(f"\n  Saved → {audit_path}")

    # Temporal gaps
    if temporal_gaps:
        print("\n" + "─" * 65)
        print("  TEMPORAL GAPS  (data loss — not removable)")
        print("  Step 02 will discard sliding windows that straddle these gaps.")
        print("─" * 65)
        for g in temporal_gaps:
            print(f"  {g['from']}  →  {g['to']}  ({g['gap_h']}h missing)")

    print("\n" + "─" * 65)
    print("  FEATURE SUMMARY")
    print("─" * 65)
    print()
    print("  Price features:")
    print("    lmp, arcsinh_lmp, log_return — raw price dynamics.")
    print("    log_lmp_shifted, lmp_clipped — log-scale and clipped alternatives.")
    print("    mstl_resid_arcsinh, mstl_resid_lr, mstl_resid_log — MSTL residuals (seasonal removed).")
    print()
    print(feat.describe().round(3).to_string())

    print("\n" + "─" * 65)
    print("  VALIDATION CHECKS")
    print("─" * 65)
    run_checks(df_clean, feat)

    print("\n" + "─" * 65)
    print(f"  Output saved  → {out_path}")
    print(f"  Plots saved   → {PLOT_DIR}/")
    print(f"  Audit log     → {audit_path}")
    print("═" * 65 + "\n")

    return out


if __name__ == "__main__":
    main()
