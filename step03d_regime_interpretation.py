"""
step03d_regime_interpretation.py
=================================
Verbose economic interpretation of the regimes produced by step03_clustering.

For each regime this script:
  1. Collects all windows assigned to it (from regimes.parquet)
  2. For each window extracts the 720h context from preprocessed.parquet
  3. Computes mean / std / percentiles for each feature
  4. Inverts arcsinh_lmp → LMP via sinh()
  5. Inverts ILR coordinates → fuel shares via the SBP basis
  6. Prints a verbose economic narrative for each regime
  7. Saves a battery of diagnostic plots to results/step03d/

Input  : results/regimes.parquet, results/preprocessed.parquet, results/umap.parquet
Output : results/step03d/  (figures)
         stdout (verbose interpretation)

Run    : python step03d_regime_interpretation.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  ILR inverse (same SBP as step01)
# ═══════════════════════════════════════════════════════════════════════════

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

FUEL_DISPLAY = ["Gas", "Nuclear", "Hydro", "Wind", "Solar", "Coal", "Oil", "Other"]
FUEL_COLORS  = ["#e07b39", "#7b52ab", "#4da6e8", "#6abf69", "#f7c948",
                "#5a5a5a", "#b04040", "#aaaaaa"]

ILR_COLS = [f"ilr_{i}" for i in range(1, 8)]


def ilr_basis(sbp: np.ndarray) -> np.ndarray:
    """Build (D-1) × D orthonormal ILR basis matrix Ψ from the SBP."""
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
    """Inverse ILR: (N, 7) → (N, 8) fuel shares (rows sum to 1)."""
    Psi   = ilr_basis(sbp)
    clr   = ilr @ Psi
    exp_c = np.exp(clr)
    return exp_c / exp_c.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

CONTEXT_H = C.MEAN_REVERSION["context_len"]   # 720h

SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Autumn", 10: "Autumn", 11: "Autumn"}

PEAK_HOURS = set(range(7, 23))   # HE07–HE22

REGIME_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#1b9e77", "#d95f02",
    "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
]


def season_name(month):
    return SEASON_MAP.get(month, "?")


def dominant_season(months: pd.Series) -> str:
    counts = months.map(season_name).value_counts()
    if counts.empty:
        return "?"
    top = counts.index[0]
    pct = counts.iloc[0] / counts.sum() * 100
    return f"{top} ({pct:.0f}%)"


def peak_fraction(hours: pd.Series) -> float:
    return hours.isin(PEAK_HOURS).mean()


def lmp_narrative(lmp_mean, lmp_p5, lmp_p95):
    if lmp_mean < 10:
        level = "very low / near-zero or negative"
    elif lmp_mean < 30:
        level = "low (off-peak / renewable surplus)"
    elif lmp_mean < 60:
        level = "moderate (typical dispatch)"
    elif lmp_mean < 100:
        level = "elevated (demand stress)"
    elif lmp_mean < 200:
        level = "high (grid stress / scarcity)"
    else:
        level = "extreme (spike / emergency)"
    spread = lmp_p95 - lmp_p5
    if spread < 30:
        vol = "very low volatility"
    elif spread < 80:
        vol = "moderate volatility"
    elif spread < 200:
        vol = "high volatility"
    else:
        vol = "extreme volatility (spike-prone)"
    return level, vol


def fuel_narrative(fuel_means: dict) -> str:
    parts = []
    gas_pct = fuel_means.get("Gas", 0) * 100
    nuc_pct = fuel_means.get("Nuclear", 0) * 100
    hyd_pct = fuel_means.get("Hydro", 0) * 100
    win_pct = fuel_means.get("Wind", 0) * 100
    sol_pct = fuel_means.get("Solar", 0) * 100
    coa_pct = fuel_means.get("Coal", 0) * 100
    oil_pct = fuel_means.get("Oil", 0) * 100
    ren_pct = win_pct + sol_pct + hyd_pct

    # Gas characterisation
    if gas_pct > 50:
        parts.append(f"gas-dominated ({gas_pct:.0f}%)")
    elif gas_pct > 35:
        parts.append(f"gas-heavy ({gas_pct:.0f}%)")
    else:
        parts.append(f"gas-moderate ({gas_pct:.0f}%)")

    # Nuclear baseload
    if nuc_pct > 30:
        parts.append(f"strong nuclear baseload ({nuc_pct:.0f}%)")
    elif nuc_pct > 20:
        parts.append(f"normal nuclear baseload ({nuc_pct:.0f}%)")
    else:
        parts.append(f"reduced nuclear ({nuc_pct:.0f}% — possible Millstone/Seabrook outage?)")

    # Renewables
    if ren_pct > 40:
        parts.append(f"high renewables ({ren_pct:.0f}% wind+solar+hydro)")
    elif ren_pct > 25:
        parts.append(f"moderate renewables ({ren_pct:.0f}%)")
    else:
        parts.append(f"low renewables ({ren_pct:.0f}%)")

    # Emergency fuels
    if oil_pct > 5:
        parts.append(f"⚠️  SIGNIFICANT OIL ({oil_pct:.1f}%) — likely cold-weather peaking event")
    elif oil_pct > 1:
        parts.append(f"oil activated ({oil_pct:.1f}% — grid under stress)")
    if coa_pct > 3:
        parts.append(f"⚠️  COAL ACTIVE ({coa_pct:.1f}%) — unusual for ISONE")

    return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  Load data
# ═══════════════════════════════════════════════════════════════════════════

RESULTS = Path(C.RESULTS_DIR)
OUT_DIR = RESULTS / "step03d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading parquet files …", flush=True)
regimes_df = pd.read_parquet(RESULTS / "regimes.parquet")
regimes_df["datetime"] = pd.to_datetime(regimes_df["datetime"])

pre_df = pd.read_parquet(RESULTS / "preprocessed.parquet")
if "datetime" in pre_df.columns:
    pre_df = pre_df.set_index("datetime")
pre_df.index = pd.to_datetime(pre_df.index)
pre_df = pre_df.sort_index()

umap_df = pd.read_parquet(RESULTS / "umap.parquet")
umap_df["datetime"] = pd.to_datetime(umap_df["datetime"])

print(f"  regimes : {len(regimes_df):,} windows")
print(f"  preproc : {len(pre_df):,} hours")
print(f"  umap    : {len(umap_df):,} points")

# Merge UMAP coords into regimes
umap_merged = regimes_df.merge(umap_df, on="datetime", how="left")

# ═══════════════════════════════════════════════════════════════════════════
#  Per-window feature extraction
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nExpanding windows ({len(regimes_df)} windows × up to {CONTEXT_H}h context) …",
      flush=True)

FEAT_STATS = ["arcsinh_lmp", "log_return", "total_mw"] + ILR_COLS

records = []

for _, row in tqdm(regimes_df.iterrows(), total=len(regimes_df), ncols=80):
    wdt    = row["datetime"]
    regime = row["regime"]
    if regime < 0:
        continue

    start  = wdt - pd.Timedelta(hours=CONTEXT_H - 1)
    win    = pre_df.loc[start:wdt, FEAT_STATS]

    if len(win) < CONTEXT_H // 4:    # skip near-gap windows
        continue

    rec = {"datetime": wdt, "regime": int(regime)}
    for col in FEAT_STATS:
        if col in win.columns:
            rec[f"{col}_mean"] = float(win[col].mean())
            rec[f"{col}_std"]  = float(win[col].std())
    rec["hour_mean"]  = float(win.index.hour.mean())
    rec["month_mode"] = int(win.index.month.value_counts().idxmax())
    rec["peak_frac"]  = float(win.index.hour.isin(PEAK_HOURS).mean())
    records.append(rec)

wstats = pd.DataFrame(records)
print(f"  Valid windows: {len(wstats):,} / {len(regimes_df):,}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
#  Invert arcsinh → LMP and ILR → fuel shares
# ═══════════════════════════════════════════════════════════════════════════

# LMP via sinh
wstats["lmp_mean"] = np.sinh(wstats["arcsinh_lmp_mean"])

# ILR inverse
ilr_mat = wstats[[f"ilr_{i}_mean" for i in range(1, 8)]].values
fuel_mat = ilr_inverse(ilr_mat, SBP)   # (N_windows, 8)
for j, fname in enumerate(FUEL_DISPLAY):
    wstats[f"fuel_{fname}"] = fuel_mat[:, j]

# ═══════════════════════════════════════════════════════════════════════════
#  Per-regime aggregates
# ═══════════════════════════════════════════════════════════════════════════

regime_ids   = sorted(wstats["regime"].unique())
regime_sizes = wstats.groupby("regime").size().to_dict()
total_wins   = len(wstats)

# ═══════════════════════════════════════════════════════════════════════════
#  ── PRINT VERBOSE INTERPRETATION ──
# ═══════════════════════════════════════════════════════════════════════════

SEPARATOR = "═" * 72

print(f"\n\n{SEPARATOR}")
print("  ISONE REGIME INTERPRETATION — VERBOSE ECONOMIC NARRATIVE")
print(f"  {len(regime_ids)} regimes  |  "
      f"{len(wstats):,} windows  |  noise excluded")
print(SEPARATOR)

regime_summaries = {}

for rid in regime_ids:
    sub = wstats[wstats["regime"] == rid]
    n   = len(sub)
    pct = n / total_wins * 100

    # ── Price statistics ────────────────────────────────────────────────
    lmp_mean = sub["lmp_mean"].mean()
    lmp_med  = sub["lmp_mean"].median()
    lmp_p5   = sub["lmp_mean"].quantile(0.05)
    lmp_p95  = sub["lmp_mean"].quantile(0.95)
    lmp_std  = sub["lmp_mean"].std()
    logret_std = sub.get("log_return_std", pd.Series([np.nan])).mean()

    lmp_level, lmp_vol = lmp_narrative(lmp_mean, lmp_p5, lmp_p95)

    # ── Fuel shares ─────────────────────────────────────────────────────
    fuel_means = {f: sub[f"fuel_{f}"].mean() for f in FUEL_DISPLAY}
    fuel_str   = fuel_narrative(fuel_means)
    top_fuels  = sorted(fuel_means.items(), key=lambda x: -x[1])[:3]

    # ── Load ────────────────────────────────────────────────────────────
    load_mean = sub["total_mw_mean"].mean() if "total_mw_mean" in sub else np.nan

    # ── Temporal patterns ───────────────────────────────────────────────
    # Reconstruct approximate datetimes for season + hour analysis
    regime_dts = sub["datetime"]
    months = regime_dts.dt.month
    dom_season = dominant_season(months)
    peak_frac  = sub["peak_frac"].mean()

    # ── ILR coordinates ─────────────────────────────────────────────────
    ilr_means = [sub[f"ilr_{i}_mean"].mean() for i in range(1, 8)]

    # Store summary for comparison table
    regime_summaries[rid] = {
        "n_windows": n, "pct": pct,
        "lmp_mean": lmp_mean, "lmp_p5": lmp_p5, "lmp_p95": lmp_p95,
        "load_mean": load_mean, "peak_frac": peak_frac,
        "fuel_means": fuel_means, "dom_season": dom_season,
    }

    # ── Print ────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  REGIME  {rid:>2d}   ({n:,} windows, {pct:.1f}% of total)")
    print(f"{'─' * 72}")

    print(f"\n  PRICE")
    print(f"    Mean LMP      : ${lmp_mean:>7.2f}/MWh  [P5 ${lmp_p5:.1f} – P95 ${lmp_p95:.1f}]")
    print(f"    Median LMP    : ${lmp_med:>7.2f}/MWh")
    print(f"    Price level   : {lmp_level}")
    print(f"    Price spread  : {lmp_vol}")

    print(f"\n  LOAD")
    print(f"    Mean load     : {load_mean:,.0f} MW")
    print(f"    Peak-hour frac: {peak_frac:.1%}  (HE07–HE22)")

    print(f"\n  FUEL MIX  (mean window shares)")
    for fname, fval in sorted(fuel_means.items(), key=lambda x: -x[1]):
        bar = "█" * int(round(fval * 30))
        print(f"    {fname:>8s} : {fval*100:5.1f}%  {bar}")
    print(f"\n  Characterisation: {fuel_str}")

    print(f"\n  ILR COORDINATES (mean over windows)")
    ilr_labels = [
        "ILR1 Dispatchable vs Variable  ",
        "ILR2 Fossil vs Non-fossil      ",
        "ILR3 Gas vs Coal+Oil           ",
        "ILR4 Coal vs Oil               ",
        "ILR5 Nuclear vs Other          ",
        "ILR6 Hydro vs Intermittent     ",
        "ILR7 Solar vs Wind             ",
    ]
    for lbl, val in zip(ilr_labels, ilr_means):
        sign = "▲" if val > 0.2 else ("▼" if val < -0.2 else "●")
        print(f"    {lbl}: {val:+.3f}  {sign}")

    print(f"\n  TEMPORAL CONTEXT")
    print(f"    Dominant season : {dom_season}")
    month_counts = months.value_counts().sort_index()
    month_str = "  ".join(f"{pd.Timestamp(2021, m, 1).strftime('%b')}:{c:>3d}"
                           for m, c in month_counts.items())
    print(f"    Monthly dist.   : {month_str}")

    # ── Economic narrative ───────────────────────────────────────────────
    print(f"\n  ECONOMIC NARRATIVE")
    # Generate a narrative based on the quantitative features
    gas_pct = fuel_means["Gas"] * 100
    oil_pct = fuel_means["Oil"] * 100
    ren_pct = (fuel_means["Wind"] + fuel_means["Solar"] + fuel_means["Hydro"]) * 100
    nuc_pct = fuel_means["Nuclear"] * 100

    if lmp_mean > 150 or oil_pct > 8:
        print("    🔴 STRESS / SCARCITY REGIME")
        print(f"    Prices are extremely elevated (${lmp_mean:.0f}/MWh avg) with high")
        print(f"    price volatility. Oil peakers active at {oil_pct:.1f}% — consistent with")
        print(f"    cold-weather demand surge or supply emergency. This regime likely")
        print(f"    corresponds to polar vortex / winter peak events where expensive")
        print(f"    last-resort generation sets the price.")
    elif lmp_mean > 80 or oil_pct > 2:
        print("    🟠 ELEVATED / GRID-STRESS REGIME")
        print(f"    Prices above normal (${lmp_mean:.0f}/MWh) with gas as dominant fuel ({gas_pct:.0f}%).")
        if oil_pct > 2:
            print(f"    Oil dispatch ({oil_pct:.1f}%) signals demand pushing into expensive peaking units.")
        print(f"    Likely corresponds to demand peaks (hot summer days or cold snaps)")
        print(f"    where the merit order is steeply ascending.")
    elif lmp_mean < 15 or ren_pct > 45:
        print("    🟢 RENEWABLE SURPLUS / LOW-PRICE REGIME")
        print(f"    Prices near zero or slightly negative (${lmp_mean:.0f}/MWh) with high")
        print(f"    renewable penetration ({ren_pct:.0f}% wind+solar+hydro). Gas displaced")
        print(f"    to minimum output. Consistent with spring/overnight wind surplus,")
        print(f"    ISO-NE minimum load periods, or midday solar oversupply.")
    elif ren_pct > 30 and lmp_mean < 40:
        print("    🟡 MODERATE RENEWABLE MIX REGIME")
        print(f"    Below-normal prices (${lmp_mean:.0f}/MWh) with meaningful renewable share")
        print(f"    ({ren_pct:.0f}%). Gas still dominant ({gas_pct:.0f}%) but renewables compress margin.")
        print(f"    Likely transition shoulder hours or mild-weather periods.")
    elif gas_pct > 55 and lmp_mean > 40:
        print("    🔵 GAS-MARGINAL REGIME (normal operations)")
        print(f"    Typical ISONE dispatch (${lmp_mean:.0f}/MWh): gas sets the price at {gas_pct:.0f}%")
        print(f"    of the mix. Nuclear baseload ({nuc_pct:.0f}%) running normally.")
        print(f"    This is the baseline operating regime for the market.")
    else:
        print("    🟤 MIXED / TRANSITION REGIME")
        print(f"    Prices moderate (${lmp_mean:.0f}/MWh) with diverse fuel mix.")
        print(f"    Gas {gas_pct:.0f}%, renewables {ren_pct:.0f}%, nuclear {nuc_pct:.0f}%.")
        print(f"    May represent transition periods between structural regimes.")


# ═══════════════════════════════════════════════════════════════════════════
#  Comparison table
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n\n{SEPARATOR}")
print("  REGIME COMPARISON TABLE")
print(SEPARATOR)
hdr = (f"{'Rgm':>3}  {'N':>5}  {'%tot':>5}  "
       f"{'LMP_mean':>8}  {'LMP_p5':>7}  {'LMP_p95':>7}  "
       f"{'Load(MW)':>9}  {'Gas%':>5}  {'Oil%':>5}  {'Ren%':>5}  "
       f"{'Season':>18}")
print(f"\n  {hdr}")
print("  " + "─" * len(hdr))

for rid in regime_ids:
    s = regime_summaries[rid]
    fm = s["fuel_means"]
    ren = (fm["Wind"] + fm["Solar"] + fm["Hydro"]) * 100
    print(f"  {rid:>3d}  {s['n_windows']:>5d}  {s['pct']:>5.1f}  "
          f"{s['lmp_mean']:>8.2f}  {s['lmp_p5']:>7.2f}  {s['lmp_p95']:>7.2f}  "
          f"{s['load_mean']:>9,.0f}  {fm['Gas']*100:>5.1f}  "
          f"{fm['Oil']*100:>5.2f}  {ren:>5.1f}  "
          f"{s['dom_season']:>18}")


# ═══════════════════════════════════════════════════════════════════════════
#  ── PLOTS ──
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n\nGenerating plots → {OUT_DIR} …", flush=True)

sns.set_theme(style="whitegrid", font_scale=0.9)
palette = {rid: REGIME_COLORS[i % len(REGIME_COLORS)]
           for i, rid in enumerate(regime_ids)}
noise_color = "#cccccc"


# ── 1. UMAP scatter coloured by regime ──────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 7))
for rid in [-1] + regime_ids:
    mask = umap_merged["regime"] == rid
    if mask.sum() == 0:
        continue
    color = noise_color if rid == -1 else palette[rid]
    label = "Noise" if rid == -1 else f"Regime {rid}"
    ax.scatter(umap_merged.loc[mask, "umap_1"],
               umap_merged.loc[mask, "umap_2"],
               c=color, s=3, alpha=0.5, label=label, linewidths=0)
ax.legend(markerscale=3, loc="upper right", fontsize=7, framealpha=0.7)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_title("UMAP projection — coloured by regime", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "01_umap_scatter.png", dpi=150)
plt.close(fig)
print("  saved: 01_umap_scatter.png")


# ── 2. Regime timeline (calendar view) ──────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 4))
for rid in regime_ids:
    sub = wstats[wstats["regime"] == rid]
    ax.scatter(sub["datetime"], [rid] * len(sub),
               c=palette[rid], s=2, alpha=0.4, linewidths=0)
ax.set_yticks(regime_ids)
ax.set_yticklabels([f"R{r}" for r in regime_ids], fontsize=8)
ax.set_xlabel("Date")
ax.set_title("Regime timeline — window endpoints coloured by regime", fontsize=11, fontweight="bold")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
fig.tight_layout()
fig.savefig(OUT_DIR / "02_regime_timeline.png", dpi=150)
plt.close(fig)
print("  saved: 02_regime_timeline.png")


# ── 3. LMP box plots per regime ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
data_boxes = [wstats.loc[wstats["regime"] == rid, "lmp_mean"].values
              for rid in regime_ids]
bp = ax.boxplot(data_boxes, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=0.8),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
for patch, rid in zip(bp["boxes"], regime_ids):
    patch.set_facecolor(palette[rid])
    patch.set_alpha(0.75)
ax.set_xticks(range(1, len(regime_ids) + 1))
ax.set_xticklabels([f"R{r}" for r in regime_ids])
ax.set_ylabel("Mean window LMP  ($/MWh)")
ax.set_title("LMP distribution per regime", fontsize=11, fontweight="bold")
ax.axhline(0, color="red", linewidth=0.7, linestyle="--", alpha=0.5, label="$0/MWh")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "03_lmp_boxplot.png", dpi=150)
plt.close(fig)
print("  saved: 03_lmp_boxplot.png")


# ── 4. Stacked bar — fuel mix per regime ────────────────────────────────

fuel_matrix = np.zeros((len(regime_ids), len(FUEL_DISPLAY)))
for i, rid in enumerate(regime_ids):
    sub = wstats[wstats["regime"] == rid]
    for j, fname in enumerate(FUEL_DISPLAY):
        fuel_matrix[i, j] = sub[f"fuel_{fname}"].mean()

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(regime_ids))
bottom = np.zeros(len(regime_ids))
for j, (fname, fcolor) in enumerate(zip(FUEL_DISPLAY, FUEL_COLORS)):
    ax.bar(x, fuel_matrix[:, j] * 100, bottom=bottom * 100,
           color=fcolor, label=fname, width=0.7, edgecolor="white", linewidth=0.3)
    bottom += fuel_matrix[:, j]
ax.set_xticks(x)
ax.set_xticklabels([f"R{r}" for r in regime_ids])
ax.set_ylabel("Share (%)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title("Mean fuel mix per regime", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT_DIR / "04_fuel_mix_bar.png", dpi=150)
plt.close(fig)
print("  saved: 04_fuel_mix_bar.png")


# ── 5. Radar chart — ILR coordinates per regime ─────────────────────────

ilr_short = ["ILR1\nDisp/Var", "ILR2\nFoss/NonF",
             "ILR3\nGas/Coal+Oil", "ILR4\nCoal/Oil",
             "ILR5\nNuc/Oth", "ILR6\nHyd/Int", "ILR7\nSol/Wind"]
n_ilr = 7
angles = np.linspace(0, 2 * np.pi, n_ilr, endpoint=False).tolist()
angles += angles[:1]  # close polygon

nrows = 3
ncols = math.ceil(len(regime_ids) / nrows)

import math

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3),
                          subplot_kw=dict(polar=True))
axes_flat = np.array(axes).flatten()

for ax in axes_flat:
    ax.set_visible(False)

for i, rid in enumerate(regime_ids):
    sub  = wstats[wstats["regime"] == rid]
    vals = [sub[f"ilr_{k}_mean"].mean() for k in range(1, 8)]
    vals += vals[:1]
    ax = axes_flat[i]
    ax.set_visible(True)
    ax.plot(angles, vals, color=palette[rid], linewidth=1.5)
    ax.fill(angles, vals, color=palette[rid], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ilr_short, size=5)
    ax.set_title(f"R{rid}\n(N={regime_summaries[rid]['n_windows']})",
                 size=8, fontweight="bold", pad=8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

fig.suptitle("ILR coordinates per regime (radar)", fontsize=11, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "05_ilr_radar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: 05_ilr_radar.png")


# ── 6. Monthly distribution heatmap ─────────────────────────────────────

month_counts = np.zeros((len(regime_ids), 12), dtype=int)
for i, rid in enumerate(regime_ids):
    sub = wstats[wstats["regime"] == rid]
    mc  = sub["datetime"].dt.month.value_counts()
    for m in range(1, 13):
        month_counts[i, m - 1] = mc.get(m, 0)

# Normalise by column (month) to get regime distribution within each month
month_norm = month_counts / (month_counts.sum(axis=0, keepdims=True) + 1e-9)

fig, ax = plt.subplots(figsize=(12, max(4, len(regime_ids) * 0.55 + 1.5)))
im = ax.imshow(month_norm, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=month_norm.max())
ax.set_xticks(range(12))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_yticks(range(len(regime_ids)))
ax.set_yticklabels([f"R{r}" for r in regime_ids])
ax.set_title("Regime prevalence by calendar month\n(fraction of windows within each month)",
             fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="Fraction of month's windows")
for i in range(len(regime_ids)):
    for j in range(12):
        v = month_norm[i, j]
        if v > 0.05:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=7, color="black" if v < 0.6 else "white")
fig.tight_layout()
fig.savefig(OUT_DIR / "06_monthly_heatmap.png", dpi=150)
plt.close(fig)
print("  saved: 06_monthly_heatmap.png")


# ── 7. LMP vs gas share scatter (one point per regime) ──────────────────

fig, ax = plt.subplots(figsize=(7, 5))
for rid in regime_ids:
    s  = regime_summaries[rid]
    fm = s["fuel_means"]
    ax.scatter(fm["Gas"] * 100, s["lmp_mean"],
               c=palette[rid], s=regime_summaries[rid]["n_windows"] / 10,
               alpha=0.85, linewidths=0.5, edgecolors="black",
               zorder=3)
    ax.annotate(f"R{rid}", (fm["Gas"] * 100, s["lmp_mean"]),
                textcoords="offset points", xytext=(5, 3), fontsize=8)
ax.set_xlabel("Gas share (% of mix)")
ax.set_ylabel("Mean window LMP ($/MWh)")
ax.set_title("LMP vs Gas share — bubble size ∝ regime size",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "07_lmp_vs_gas.png", dpi=150)
plt.close(fig)
print("  saved: 07_lmp_vs_gas.png")


print(f"\n✓ All plots saved to {OUT_DIR}")
print("Done.")
