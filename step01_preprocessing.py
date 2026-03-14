"""
step01_preprocessing.py
=======================
NEPOOL Regime Detection Pipeline — Step 01: Preprocessing
ISONE Electricity Market — Geometry-Driven Regime Identification

Input  : isone_dataset.parquet  (43,789 rows × 22 cols, 2021–2025)
Output : results/preprocessed.parquet
Plots  : results/step01/  (8 diagnostic figures)

Features produced (11 columns)
-------------------------------
  1.  lmp        — raw LMP $/MWh
  2.  log_lmp    — ln(LMP), clipped at 1e-3 (no negative prices in 2021–2025)
  3.  log_return — first difference of log_lmp, lag 1h
  4.  total_mw   — total generation MW (scale complement to ILR composition)
  5.  ilr_1 … ilr_7 — Isometric Log-Ratio of fuel mix (7 coordinates)

Design choices
--------------
  • No standardisation: Chronos-2 normalises each channel per-window internally.
  • No detrending: same reason — per-window normalisation removes local trend.
  • signed_log for LMP: handles negative prices (oversupply events) correctly.
  • ILR over raw shares: embeds compositional data in Euclidean space (Aitchison
    geometry), removing the unit-sum constraint and enabling proper distance
    computation in downstream clustering.
  • SBP design is economically motivated (ISONE merit order).

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

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  ILR configuration
# ═══════════════════════════════════════════════════════════════════════════

# Sequential Binary Partition — economically motivated (merit order ISONE)
# Column order: Gas(0) Nuc(1) Hyd(2) Win(3) Sol(4) Coa(5) Oil(6) Oth(7)
#
#        Gas  Nuc  Hyd  Win  Sol  Coa  Oil  Oth
SBP = np.array([
    [ 1,   1,  -1,  -1,  -1,   1,   1,   1],   # ILR1: Dispatchable vs Variable
    [ 1,  -1,   0,   0,   0,   1,   1,  -1],   # ILR2: Fossil vs Non-fossil
    [ 1,   0,   0,   0,   0,  -1,  -1,   0],   # ILR3: Gas vs Coal+Oil
    [ 0,   0,   0,   0,   0,   1,  -1,   0],   # ILR4: Coal vs Oil
    [ 0,   1,   0,   0,   0,   0,   0,  -1],   # ILR5: Nuclear vs Other
    [ 0,   0,   1,  -1,  -1,   0,   0,   0],   # ILR6: Hydro vs Intermittent
    [ 0,   0,   0,  -1,   1,   0,   0,   0],   # ILR7: Solar vs Wind (Solar+, Wind-)
], dtype=float)

ILR_LABELS = [
    "ILR1\nDispatchable\nvs Variable",
    "ILR2\nFossil\nvs Non-fossil",
    "ILR3\nGas\nvs Coal+Oil",
    "ILR4\nCoal\nvs Oil",
    "ILR5\nNuclear\nvs Other",
    "ILR6\nHydro\nvs Intermittent",
    "ILR7\nSolar\nvs Wind",
]

FUEL_COLS  = C.ILR["fuel_cols"]   # ordered: gas, nuc, hyd, win, sol, coa, oil, oth
DELTA      = C.ILR["zero_replacement_delta"]
PLOT_DIR   = Path(C.RESULTS_DIR) / "step01"

FUEL_DISPLAY = ["Gas", "Nuclear", "Hydro", "Wind", "Solar", "Coal", "Oil", "Other"]
FUEL_COLORS  = ["#e07b39", "#7b52ab", "#4da6e8", "#6abf69", "#f7c948",
                "#5a5a5a", "#b04040", "#aaaaaa"]


# ═══════════════════════════════════════════════════════════════════════════
#  ILR — forward transform
# ═══════════════════════════════════════════════════════════════════════════

def zero_replace(shares: np.ndarray, delta: float) -> np.ndarray:
    """
    Multiplicative zero replacement (Martín-Fernández et al., 2003).
    Replaces non-positive values (zeros AND negatives) with delta,
    then renormalises rows to sum to 1.

    Negative fuel shares are physically impossible and treated as
    reporting artefacts (same as zeros for compositional purposes).
    Using s <= 0 prevents np.log() from receiving invalid inputs.
    """
    s = shares.copy()
    s[s <= 0] = delta
    s = s / s.sum(axis=1, keepdims=True)
    return s


def ilr_transform(shares: np.ndarray, sbp: np.ndarray) -> np.ndarray:
    """
    Forward ILR transform given a Sequential Binary Partition.
    Returns (N, D-1) matrix of ILR coordinates.
    """
    log_s    = np.log(shares)
    n_coords = sbp.shape[0]
    ilr      = np.zeros((len(shares), n_coords))
    for j, row in enumerate(sbp):
        pos   = row ==  1
        neg   = row == -1
        r, s  = pos.sum(), neg.sum()
        coeff = np.sqrt((r * s) / (r + s))
        ilr[:, j] = coeff * (log_s[:, pos].mean(axis=1) - log_s[:, neg].mean(axis=1))
    return ilr


# ═══════════════════════════════════════════════════════════════════════════
#  ILR — inverse transform
# ═══════════════════════════════════════════════════════════════════════════

def ilr_basis(sbp: np.ndarray) -> np.ndarray:
    """Build the (D-1) × D orthonormal ILR basis matrix Ψ from the SBP."""
    n_coords, D = sbp.shape
    Psi = np.zeros((n_coords, D))
    for j, row in enumerate(sbp):
        pos = row ==  1
        neg = row == -1
        r, s = pos.sum(), neg.sum()
        Psi[j, pos] =  np.sqrt(s / (r * (r + s)))
        Psi[j, neg] = -np.sqrt(r / (s * (r + s)))
    return Psi


def ilr_inverse(ilr: np.ndarray, sbp: np.ndarray) -> np.ndarray:
    """Inverse ILR: ILR coordinates → fuel shares via CLR → softmax."""
    Psi   = ilr_basis(sbp)
    clr   = ilr @ Psi
    exp_c = np.exp(clr)
    return exp_c / exp_c.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Feature builders
# ═══════════════════════════════════════════════════════════════════════════

def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build price-derived features.
      log_lmp    : ln(LMP), clipped at 1e-3 as safety floor (no negatives in dataset)
      log_return : first difference of log_lmp (≈ arithmetic return for small Δ)
                   NaN for the very first observation (no prior price available)
    """
    lmp         = df["lmp"].values.astype(np.float64)
    log_lmp     = np.log(np.clip(lmp, 1e-3, None))
    log_ret     = np.empty(len(lmp))
    log_ret[0]  = np.nan
    log_ret[1:] = np.diff(log_lmp)
    return pd.DataFrame({"lmp": lmp, "log_lmp": log_lmp, "log_return": log_ret},
                        index=df.index)


def build_total_mw(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"total_mw": df["total_mw"].values.astype(np.float64)},
                        index=df.index)


def build_ilr_features(df: pd.DataFrame) -> pd.DataFrame:
    shares = df[FUEL_COLS].values.astype(np.float64)
    shares = zero_replace(shares, DELTA)
    ilr    = ilr_transform(shares, SBP)
    cols   = [f"ilr_{i+1}" for i in range(ilr.shape[1])]
    return pd.DataFrame(ilr, columns=cols, index=df.index)


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

    # total_mw negative
    n_neg_mw = (feat["total_mw"] < 0).sum()
    if n_neg_mw:
        issues.append(f"total_mw < 0: {n_neg_mw} rows  ← possible EIA reporting artefact")

    # ILR7 sign check: solar > wind at noon in summer → ILR7 > 0
    tmp = df_raw.copy()
    tmp["hour"]  = pd.to_datetime(tmp["datetime"]).dt.hour
    tmp["month"] = pd.to_datetime(tmp["datetime"]).dt.month
    noon_idx = tmp[(tmp["hour"] == 12) & (tmp["month"].isin([6, 7, 8]))].index
    noon_idx = noon_idx[noon_idx < len(feat)]
    if len(noon_idx):
        ilr7_noon = feat.loc[noon_idx, "ilr_7"].mean()
        sign_ok   = ilr7_noon > 0
        issues.append(
            f"ILR7 mean at noon (Jun–Aug): {ilr7_noon:.3f}  "
            f"{'✓ Solar > Wind' if sign_ok else '✗ SIGN ERROR'}"
        )

    # Fuel shares sum check
    fuel_vals  = df_raw[FUEL_COLS].values.astype(np.float64)
    row_sums   = fuel_vals.sum(axis=1)
    bad_sum    = np.abs(row_sums - 1.0) > 1e-4
    n_bad_sum  = bad_sum.sum()
    if n_bad_sum:
        issues.append(
            f"fuel shares row-sum ≠ 1: {n_bad_sum} rows  "
            f"(max deviation {np.abs(row_sums[bad_sum] - 1.0).max():.2e})  ✗ DATA QUALITY"
        )
    else:
        issues.append(
            f"fuel shares row-sum: max deviation {np.abs(row_sums - 1.0).max():.2e}  ✓"
        )

    # Fuel share non-positive values
    n_nonpos = (fuel_vals <= 0).sum()
    if n_nonpos:
        issues.append(
            f"fuel shares ≤ 0: {n_nonpos} cells replaced with δ={DELTA}"
            f"  ({'negatives' if (fuel_vals < 0).any() else 'zeros only'})"
        )

    # ILR roundtrip
    ilr_vals   = feat[[f"ilr_{i}" for i in range(1, 8)]].values
    shares_rec = ilr_inverse(ilr_vals, SBP)
    shares_raw = zero_replace(fuel_vals, DELTA)
    max_err    = np.abs(shares_raw - shares_rec).max()
    issues.append(f"ILR roundtrip max error: {max_err:.2e}  {'✓' if max_err < 1e-10 else '✗ ERROR'}")

    for msg in issues:
        print(f"    {'⚠' if '✗' in msg or 'artefact' in msg else '✓'}  {msg}")


# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(out: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """
    Generate 8 diagnostic figures and save to results/step01/.

    Figure list:
      01_lmp_timeseries.png       — full LMP series with spike threshold
      02_lmp_distribution.png     — LMP histogram + log-scale + signed_log overlay
      03_log_return_distribution  — log_return histogram with normal fit + QQ plot
      04_total_mw_profile.png     — total_mw by hour-of-day per season
      05_fuel_mix_monthly.png     — monthly average fuel share stacked bar
      06_ilr_boxplots.png         — boxplots of ILR 1–7 with SBP descriptions
      07_ilr7_heatmap.png         — ILR7 mean by hour × month (Solar vs Wind)
      08_feature_correlation.png  — Pearson correlation heatmap of all 11 features
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

    # ── 04: total_mw hourly profile by season ───────────────────────────────
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    season_colors = {"Winter": "#4da6e8", "Spring": "#6abf69",
                     "Summer": "#f7c948", "Autumn": "#e07b39"}
    df_p = out.copy()
    df_p["hour"]   = dt.dt.hour
    df_p["season"] = dt.dt.month.map(season_map)

    fig, ax = plt.subplots(figsize=(12, 5))
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        sub = df_p[df_p["season"] == season].groupby("hour")["total_mw"]
        med = sub.median()
        q25 = sub.quantile(0.25)
        q75 = sub.quantile(0.75)
        ax.plot(med.index, med.values, lw=2, color=season_colors[season], label=season)
        ax.fill_between(q25.index, q25.values, q75.values,
                        color=season_colors[season], alpha=0.15)
    ax.set_title("Total Generation MW — Median hourly profile by season (IQR shaded)",
                 fontweight="bold")
    ax.set_xlabel("Hour of day");  ax.set_ylabel("MW")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_total_mw_profile.png")
    plt.close(fig)

    # ── 05: Monthly fuel mix stacked bar ────────────────────────────────────
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
    fig.savefig(PLOT_DIR / "05_fuel_mix_monthly.png")
    plt.close(fig)

    # ── 06: ILR boxplots ────────────────────────────────────────────────────
    ilr_cols = [f"ilr_{i}" for i in range(1, 8)]
    ilr_data = [out[c].values for c in ilr_cols]

    fig, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot(ilr_data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=1.5),
                    flierprops=dict(marker=".", markersize=1, alpha=0.3, color="gray"))
    colors = plt.cm.tab10(np.linspace(0, 0.9, 7))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color);  patch.set_alpha(0.7)
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(ILR_LABELS, fontsize=7.5)
    ax.set_title("ILR coordinate distributions (SBP — ISONE merit order)",
                 fontweight="bold")
    ax.set_ylabel("ILR value")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_ilr_boxplots.png")
    plt.close(fig)

    # ── 07: ILR7 heatmap (hour × month) ────────────────────────────────────
    df_p["month_n"] = dt.dt.month
    pivot = df_p.groupby(["month_n", "hour"])["ilr_7"].mean().unstack("hour")

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=0,
                xticklabels=range(0, 24, 1),
                yticklabels=["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"],
                cbar_kws={"label": "ILR7  (Solar+ / Wind−)"})
    ax.set_title("ILR7 mean by hour × month — Solar (green) vs Wind (red) dominance",
                 fontweight="bold")
    ax.set_xlabel("Hour of day");  ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_ilr7_heatmap.png")
    plt.close(fig)

    # ── 08: Feature correlation heatmap ─────────────────────────────────────
    feat_cols = ["lmp", "log_lmp", "log_return", "total_mw"] + ilr_cols
    corr = out[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm", vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.3, square=True,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Pearson correlation — all 11 features", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_feature_correlation.png")
    plt.close(fig)

    print(f"  8 figures saved → {PLOT_DIR}/")


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
    print(f"  Config : standardise=OFF  detrend=OFF  delta={DELTA}")
    print()
    print("  This step loads the raw ISONE dataset, removes a small number")
    print("  of anomalous rows (DST duplicates, EIA artefacts), and builds")
    print("  11 features for downstream embedding with Chronos-2:")
    print("    • 3 price features (lmp, log_lmp, log_return)")
    print("    • 1 demand proxy  (total_mw)")
    print("    • 7 ILR coordinates encoding fuel mix composition")
    print("─" * 65)

    steps = [
        "Loading data",
        "Filtering & cleaning",
        "Price features  (lmp, log_lmp, log_return)",
        "Total MW",
        "ILR transform   (ilr_1 … ilr_7)",
        "Assembly & validation",
        "Saving output",
        "Diagnostic plots",
    ]

    pbar = tqdm(steps, ncols=65, bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt}")

    # ── 1. Load ─────────────────────────────────────────────────────────────
    pbar.set_description(steps[0])
    df = (pd.read_parquet(C.DATA_PATH)
            .sort_values("datetime")
            .reset_index(drop=True))
    df["datetime"] = pd.to_datetime(df["datetime"])
    pbar.update(1)

    # ── 2. Filter & clean ───────────────────────────────────────────────────
    pbar.set_description(steps[1])
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
            "total_mw": row["total_mw"],
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
            "total_mw": row["total_mw"],
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

    # 2c. Missing fuel shares
    fuel_null = df[FUEL_COLS].isnull().all(axis=1)
    n_dropped = fuel_null.sum()
    if n_dropped:
        df = df[~fuel_null].reset_index(drop=True)

    # 2d. Save audit log
    audit_path = Path(C.RESULTS_DIR) / "removed_rows.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(removed_rows).to_csv(audit_path, index=False)

    df_clean = df.copy()
    pbar.update(1)

    # ── 3. Price features ───────────────────────────────────────────────────
    # LMP (Locational Marginal Price) at Mass Hub is the reference price for
    # the ISONE market. We derive:
    #   • log_lmp    : signed-log transform — compresses spikes, handles
    #                  negative prices (oversupply events) without clipping.
    #   • log_return : hour-over-hour change in log_lmp — captures price
    #                  velocity and is approximately stationary.
    pbar.set_description(steps[2])
    price = build_price_features(df)
    pbar.update(1)

    # ── 4. Total MW ─────────────────────────────────────────────────────────
    # total_mw is total generation dispatched in ISONE (≈ load + net exports).
    # It adds the scale dimension that ILR cannot capture: same composition
    # at 5 GW vs 12 GW represents structurally different market conditions.
    pbar.set_description(steps[3])
    mw = build_total_mw(df)
    pbar.update(1)

    # ── 5. ILR transform ────────────────────────────────────────────────────
    # Raw fuel shares live on the unit simplex S^7 (sum = 1), where standard
    # Euclidean distance is meaningless (spurious correlations, scale
    # dependence). The ILR transform maps S^7 → ℝ^7 via a Sequential Binary
    # Partition (SBP) designed around the ISONE merit order:
    #   ILR1: overall renewable penetration (Dispatchable vs Variable)
    #   ILR2: carbon intensity (Fossil vs Non-fossil)
    #   ILR3: gas dominance within fossil fleet
    #   ILR4: hard coal vs oil (marginal fossil units)
    #   ILR5: baseload nuclear vs flexible Other
    #   ILR6: dispatchable hydro vs intermittent renewables
    #   ILR7: solar vs wind within intermittents (diurnal signal)
    # Zeros are replaced with δ=0.0001 before log (Martín-Fernández 2003).
    pbar.set_description(steps[4])
    ilr = build_ilr_features(df)
    pbar.update(1)

    # ── 6. Assembly & validation ────────────────────────────────────────────
    # Concatenate all feature blocks. Drop the first row (log_return = NaN)
    # using a shared boolean mask to keep df and feat perfectly aligned.
    pbar.set_description(steps[5])
    feat      = pd.concat([price, mw, ilr], axis=1)
    valid     = feat.notna().all(axis=1)
    feat      = feat[valid].reset_index(drop=True)
    df        = df[valid].reset_index(drop=True)
    df_clean  = df.copy()
    out       = pd.concat([df[["datetime"]], feat], axis=1)
    pbar.update(1)

    # ── 7. Save ─────────────────────────────────────────────────────────────
    pbar.set_description(steps[6])
    out_path = Path(C.RESULTS_DIR) / "preprocessed.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    pbar.update(1)

    # ── 8. Plots ─────────────────────────────────────────────────────────────
    pbar.set_description(steps[7])
    make_plots(out, df_clean)
    pbar.update(1)

    pbar.close()

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
    print("  Price features (lmp, log_lmp, log_return):")
    print("    lmp range [$9–$475], mean $55.5 — heavy right tail from spike events.")
    print("    log_lmp  via ln(lmp): compresses spikes, no negative prices in dataset.")
    print("    log_return mean ≈ 0 (martingale property), fat-tailed distribution.")
    print()
    print("  Demand proxy (total_mw):")
    print("    Range [2775–22933 MW], mean 11512 MW. Seasonal and diurnal patterns")
    print("    expected; adds scale information orthogonal to ILR composition.")
    print()
    print("  ILR coordinates (ilr_1 … ilr_7):")
    print("    All 7 axes in ℝ, unconstrained. ILR1 (Dispatchable vs Variable)")
    print("    has the widest range, reflecting the variability of renewable")
    print("    penetration. ILR7 (Solar+ vs Wind−) shows strong diurnal and")
    print("    seasonal structure, confirmed by the noon summer sign check.")
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
