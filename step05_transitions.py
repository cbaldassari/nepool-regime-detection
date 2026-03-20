"""
step05_transitions.py
=====================
NEPOOL Regime Detection Pipeline — Step 05: Markov Transition Matrix

Input  : results/regimes.parquet           (N_windows × [datetime, regime])
         results/umap.parquet              (N_windows × [datetime, umap_1, umap_2])
         results/preprocessed.parquet      (full hourly series)
Output : results/transition_matrix.parquet (K×K empirical P, labelled DataFrame)
         results/regime_stats.parquet      (per-regime statistics table)
Plots  : results/step05/

Design
------
Observations are spaced at STRIDE_H=6h. Each consecutive pair (t, t+6h) defines
one transition. Pairs spanning data gaps (> MAX_GAP_H = 7h) are discarded.
Noise windows (regime = -1) are excluded from transition counts but reported.

Transition matrix P[i,j]:
  P[i,j] = P(regime_{t+6h} = j | regime_t = i)
          = count(i→j) / Σ_j count(i→j)

Markov chain properties:
  • Stationary distribution π  — left eigenvector of P for eigenvalue 1
      Interpretation: fraction of long-run time spent in each regime.
      Computed from the closed-form solution, confirmed by bootstrapping.

  • Mean sojourn time E[T_i]  — expected hours in regime i before switching
      E[T_i] = 1 / (1 - P[i,i]) × STRIDE_H
      Based on the geometric distribution of consecutive run lengths.

  • Stationarity test          — compare P estimated on first vs second half
      Relative Frobenius distance ‖P_1 − P_2‖_F / mean(‖P_1‖_F, ‖P_2‖_F).
      < 0.10 = time-homogeneous chain; > 0.15 = structural break present.

  • Bootstrap CI on π          — 1000 resamples of per-row multinomial
      Propagates uncertainty in transition counts into π uncertainty.

Plots produced:
  01_transition_heatmap.png      — annotated P matrix + stationary π on right axis
  02_regime_timeline.png         — color-coded regime bar chart over calendar time
  03_run_lengths.png             — histogram of consecutive run lengths per regime
  04_stationary_distribution.png — π bar chart with 95% bootstrap CI
  05_regime_characterisation.png — LMP box plots + seasonal composition
  06_umap_transitions.png        — UMAP scatter + arrows for most probable transitions

Install:
    pip install pandas numpy matplotlib seaborn scikit-learn
    (no Ray needed — all operations are O(N_windows) numpy/pandas)
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
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

STRIDE_H    = 6      # hours between consecutive windows (must match step02)
MAX_GAP_H   = 7      # max hours between window timestamps to count as transition
N_BOOTSTRAP = 1000   # multinomial resamples for bootstrap CI on π
NOISE_LABEL = -1     # HDBSCAN noise label

SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3,  4, 5],
    "Summer": [6,  7, 8],
    "Fall":   [9, 10, 11],
}

PALETTE  = plt.cm.tab10
PLOT_DIR = Path(C.RESULTS_DIR) / "step05"


# ═══════════════════════════════════════════════════════════════════════════
#  Transition matrix
# ═══════════════════════════════════════════════════════════════════════════

def build_transition_matrix(
    regimes:   pd.Series,
    datetimes: pd.Series,
) -> tuple:
    """
    Build empirical first-order Markov transition matrix from regime sequence.

    A "transition" is a consecutive pair (row_i, row_{i+1}) in regimes.parquet
    where datetimes[i+1] - datetimes[i] ≤ MAX_GAP_H (excludes cross-gap pairs).
    Pairs involving NOISE_LABEL on either side are discarded.

    Returns
    -------
    P      : (K, K) ndarray  — row-stochastic probability matrix
    labels : list[int]       — regime labels in row/column order
    counts : (K, K) ndarray  — raw integer transition counts
    """
    reg  = regimes.values
    dt   = pd.to_datetime(datetimes).values

    labels = sorted(r for r in np.unique(reg) if r != NOISE_LABEL)
    K      = len(labels)
    idx    = {r: i for i, r in enumerate(labels)}
    counts = np.zeros((K, K), dtype=np.int64)

    for t in range(len(reg) - 1):
        r_from, r_to = reg[t], reg[t + 1]

        # Skip noise windows
        if r_from == NOISE_LABEL or r_to == NOISE_LABEL:
            continue

        # Skip transitions that span a data gap
        gap_h = (dt[t + 1] - dt[t]) / np.timedelta64(1, "h")
        if gap_h > MAX_GAP_H:
            continue

        counts[idx[r_from], idx[r_to]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.where(row_sums > 0, counts / row_sums, 0.0)

    return P, labels, counts


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Compute stationary distribution π via left eigenvector of P.

    π satisfies: π P = π  (row vector).
    Equivalently: P^T π = π  (column vector) → eigenvector of P^T for λ=1.
    Result is normalised to sum to 1 and guaranteed non-negative.
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi  = eigenvectors[:, idx].real
    pi  = np.abs(pi)
    pi /= pi.sum()
    return pi


def bootstrap_stationary_ci(
    P: np.ndarray,
    counts: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int   = C.RANDOM_STATE,
) -> tuple:
    """
    Bootstrap 95% CI on π by resampling per-row transition counts.

    Per bootstrap iteration:
      1. For each row i, draw n_i counts from Multinomial(n_i, P[i,·]).
      2. Re-normalise → P_boot.
      3. Compute π_boot from P_boot.

    Returns (π_mean, π_lower_2.5pct, π_upper_97.5pct).
    """
    rng      = np.random.default_rng(seed)
    K        = P.shape[0]
    pi_boots = []

    for _ in range(n_boot):
        P_boot = np.zeros_like(P)
        for i in range(K):
            n_i = int(counts[i].sum())
            if n_i == 0:
                P_boot[i] = P[i]
                continue
            sample = rng.multinomial(n_i, P[i])
            s      = sample.sum()
            P_boot[i] = sample / s if s > 0 else P[i]
        try:
            pi_boots.append(stationary_distribution(P_boot))
        except Exception:
            continue

    arr = np.stack(pi_boots)
    return (arr.mean(axis=0),
            np.percentile(arr, 2.5,  axis=0),
            np.percentile(arr, 97.5, axis=0))


def stationarity_test(
    regimes:   pd.Series,
    datetimes: pd.Series,
) -> dict:
    """
    Test time-homogeneity by comparing P estimated on first vs second half.

    Relative Frobenius distance:
        d = ‖P₁ − P₂‖_F / mean(‖P₁‖_F, ‖P₂‖_F)

    Interpretation:
        d < 0.10  → chain is approximately stationary
        0.10–0.20 → mild non-stationarity (seasonal structure still present)
        d > 0.20  → structural break — two-regime MLE or time-varying P needed

    Returns a dict with P_first, P_second, frobenius_distance,
    relative_frobenius_dist, is_stationary.
    """
    n   = len(regimes)
    mid = n // 2

    P1, L1, _ = build_transition_matrix(regimes.iloc[:mid],  datetimes.iloc[:mid])
    P2, L2, _ = build_transition_matrix(regimes.iloc[mid:],  datetimes.iloc[mid:])

    if L1 != L2:
        return {
            "is_stationary": None,
            "note": f"different regime sets in two halves — cannot compare "
                    f"(first: {L1}, second: {L2})",
        }

    fro_diff = float(np.linalg.norm(P1 - P2, "fro"))
    fro_ref  = float((np.linalg.norm(P1, "fro") + np.linalg.norm(P2, "fro")) / 2)
    rel_dist = fro_diff / (fro_ref + 1e-12)

    return {
        "P_first":                 P1,
        "P_second":                P2,
        "frobenius_distance":      fro_diff,
        "relative_frobenius_dist": rel_dist,
        "is_stationary":           rel_dist < 0.15,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Regime statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_run_lengths(regimes: pd.Series) -> dict:
    """
    Compute consecutive run lengths (hours) per regime.

    A "run" is an uninterrupted block of identical regime labels in the
    6h-stride sequence. Run length = block size × STRIDE_H.

    Returns {regime_label: list[int]  (hours)}.
    """
    runs  = {}
    reg   = regimes.values
    i     = 0
    while i < len(reg):
        r = reg[i]
        j = i
        while j < len(reg) and reg[j] == r:
            j += 1
        length_h = (j - i) * STRIDE_H
        if r != NOISE_LABEL:
            runs.setdefault(r, []).append(length_h)
        i = j
    return runs


def compute_regime_stats(
    regime_df: pd.DataFrame,
    pre:       pd.DataFrame,
    P:         np.ndarray,
    labels:    list,
    pi:        np.ndarray,
    runs:      dict,
) -> pd.DataFrame:
    """
    Build per-regime statistics table.

    Joins regime timestamps with the hourly preprocessed series to obtain LMP
    values at each window endpoint. Statistics include price quantiles, seasonal
    fractions, peak-hour fraction, self-transition probability, and stationary π.

    Columns
    -------
    regime, n_obs, pct_time, mean_duration_h, median_duration_h,
    mean_lmp, median_lmp, p5_lmp, p95_lmp,
    winter_pct, spring_pct, summer_pct, fall_pct, dominant_season,
    peak_pct (hours 08-21), self_transition, stationary_prob,
    mean_sojourn_h  (= 1/(1−P[i,i]) × STRIDE_H, geometric mean run length)
    """
    merged = regime_df.merge(
        pre[["datetime", "lmp"]],
        on="datetime",
        how="left",
    )
    merged["hour"]  = pd.to_datetime(merged["datetime"]).dt.hour
    merged["month"] = pd.to_datetime(merged["datetime"]).dt.month

    total_obs = (merged["regime"] != NOISE_LABEL).sum()
    rows      = []

    for k, r in enumerate(labels):
        sub = merged[merged["regime"] == r]
        n   = len(sub)

        # Season fractions
        season_pct = {
            sname: sub["month"].isin(months).sum() / n * 100 if n > 0 else 0.0
            for sname, months in SEASONS.items()
        }
        dom_season = max(season_pct, key=season_pct.get) if n > 0 else "?"

        # Peak hours (08-21 local)
        peak_pct = (sub["hour"].between(8, 21).sum() / n * 100 if n > 0 else 0.0)

        # LMP statistics
        lmps = sub["lmp"].dropna()

        # Run lengths
        rlen = runs.get(r, [STRIDE_H])

        # Mean sojourn time from geometric distribution
        p_self        = float(P[k, k])
        mean_sojourn  = (1.0 / (1.0 - p_self) * STRIDE_H
                         if p_self < 1.0 else float("inf"))

        rows.append({
            "regime":            r,
            "n_obs":             n,
            "pct_time":          round(n / total_obs * 100, 2) if total_obs > 0 else 0.0,
            "mean_duration_h":   round(float(np.mean(rlen)),   1),
            "median_duration_h": round(float(np.median(rlen)), 1),
            "mean_lmp":          round(float(lmps.mean()),          2) if len(lmps) > 0 else np.nan,
            "median_lmp":        round(float(lmps.median()),        2) if len(lmps) > 0 else np.nan,
            "p5_lmp":            round(float(lmps.quantile(0.05)),  2) if len(lmps) > 0 else np.nan,
            "p95_lmp":           round(float(lmps.quantile(0.95)),  2) if len(lmps) > 0 else np.nan,
            "winter_pct":        round(season_pct["Winter"], 1),
            "spring_pct":        round(season_pct["Spring"], 1),
            "summer_pct":        round(season_pct["Summer"], 1),
            "fall_pct":          round(season_pct["Fall"],   1),
            "dominant_season":   dom_season,
            "peak_pct":          round(peak_pct, 1),
            "self_transition":   round(p_self,   4),
            "mean_sojourn_h":    round(mean_sojourn, 1),
            "stationary_prob":   round(float(pi[k]), 4),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════

def _regime_color(r, labels):
    i = labels.index(r) if r in labels else -1
    return PALETTE(i / max(len(labels), 1)) if i >= 0 else (0.7, 0.7, 0.7, 0.5)


def plot_transition_heatmap(P, labels, counts, pi):
    """
    01 — Heatmap of P[i,j] with count annotations and π on right axis.

    The diagonal (self-transition) is the persistence of each regime.
    Values are annotated as "probability\\n(n=count)".
    """
    K     = len(labels)
    annot = np.empty((K, K), dtype=object)
    for i in range(K):
        for j in range(K):
            annot[i, j] = f"{P[i,j]:.2f}\n(n={counts[i,j]:,})"

    fig, ax = plt.subplots(figsize=(max(6, K * 1.4), max(5, K * 1.1)))
    sns.heatmap(
        P,
        annot=annot, fmt="",
        cmap="YlOrRd", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        xticklabels=[f"→ R{r}" for r in labels],
        yticklabels=[f"R{r} →" for r in labels],
        ax=ax,
    )
    ax.set_title(
        "Empirical Markov Transition Matrix  P[i→j]\n"
        "(rows = from-regime, cols = to-regime  |  value = probability, n = count)",
        fontweight="bold",
    )
    ax.set_xlabel("To regime")
    ax.set_ylabel("From regime")

    # Right y-axis: stationary probability π
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(K) + 0.5)
    ax2.set_yticklabels([f"π = {pi[i]:.3f}" for i in range(K)], fontsize=8)
    ax2.set_ylabel("Stationary probability π", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_transition_heatmap.png")
    plt.close(fig)
    print("  [plot] 01_transition_heatmap.png")


def plot_regime_timeline(regime_df, labels):
    """
    02 — Color-coded vertical bars over calendar time.

    Each bar is one 6h window. Noise windows are shown in light grey.
    Allows visual inspection of regime persistence, clustering, and seasonality.
    """
    dt  = pd.to_datetime(regime_df["datetime"])
    reg = regime_df["regime"].values
    K   = len(labels)

    colors  = [PALETTE(i / K)                    for i in range(K)]
    col_map = {r: colors[i] for i, r in enumerate(labels)}
    col_map[NOISE_LABEL] = (0.75, 0.75, 0.75, 0.35)

    fig, ax = plt.subplots(figsize=(16, 3))
    for i in range(len(regime_df)):
        ax.axvline(dt.iloc[i], color=col_map.get(reg[i], "grey"), lw=0.35, alpha=0.9)

    from matplotlib.patches import Patch
    handles  = [Patch(color=colors[i], label=f"Regime {r}")
                for i, r in enumerate(labels)]
    handles += [Patch(color=(0.75, 0.75, 0.75), label="Noise (-1)")]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              ncol=min(len(handles), 8))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_xlim(dt.iloc[0], dt.iloc[-1])
    ax.set_yticks([])
    ax.set_title(
        "Regime timeline  (each vertical bar = one 6h window — colour = regime)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_regime_timeline.png")
    plt.close(fig)
    print("  [plot] 02_regime_timeline.png")


def plot_run_lengths(runs, labels):
    """
    03 — Histogram of consecutive run lengths per regime.

    A long right tail indicates a highly persistent regime.
    Mean (red dashed) and median (blue dotted) are marked.
    """
    K     = len(labels)
    ncols = min(K, 3)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, (ax, r) in enumerate(zip(axes_flat, labels)):
        data = runs.get(r, [])
        if not data:
            ax.set_title(f"Regime {r}  (no data)")
            ax.axis("off")
            continue
        color = PALETTE(ax_i / K)
        ax.hist(data, bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(data),   color="red",  lw=1.3, ls="--",
                   label=f"mean = {np.mean(data):.0f}h")
        ax.axvline(np.median(data), color="navy", lw=1.3, ls=":",
                   label=f"median = {np.median(data):.0f}h")
        ax.set_title(f"Regime {r}  (n={len(data)} runs)", fontweight="bold")
        ax.set_xlabel("Run length (hours)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7)

    # Hide unused subplots
    for ax in axes_flat[K:]:
        ax.axis("off")

    fig.suptitle(
        "Distribution of consecutive run lengths per regime\n"
        "(longer = more persistent — right tail = rare multi-day events)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_run_lengths.png")
    plt.close(fig)
    print("  [plot] 03_run_lengths.png")


def plot_stationary_distribution(pi, pi_lo, pi_hi, labels, stats_df):
    """
    04 — Stationary distribution bar chart with bootstrap CI.

    π[i] = fraction of long-run time the Markov chain spends in regime i.
    Error bars are 2.5% / 97.5% bootstrap quantiles.
    Mean sojourn time is printed inside each bar.
    """
    K    = len(labels)
    x    = np.arange(K)
    yerr = np.stack([pi - pi_lo, pi_hi - pi], axis=0)

    fig, ax = plt.subplots(figsize=(max(7, K * 1.5), 4))
    bars = ax.bar(x, pi, yerr=yerr, capsize=6,
                  color=[PALETTE(i / K) for i in range(K)],
                  alpha=0.85,
                  error_kw={"lw": 1.8, "ecolor": "black"})

    # Annotate mean sojourn time inside bars
    for i, r in enumerate(labels):
        row = stats_df[stats_df["regime"] == r]
        if len(row) > 0:
            soj = row.iloc[0]["mean_sojourn_h"]
            label_str = f"{soj:.0f}h" if soj < 1e4 else "∞"
            ax.text(i, pi[i] / 2, label_str, ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Regime {r}" for r in labels])
    ax.set_ylabel("Stationary probability π")
    ax.set_title(
        "Stationary distribution  π  (long-run fraction of time per regime)\n"
        "Error bars = 95% bootstrap CI  |  white label = mean sojourn time",
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_stationary_distribution.png")
    plt.close(fig)
    print("  [plot] 04_stationary_distribution.png")


def plot_regime_characterisation(regime_df, pre, labels):
    """
    05 — LMP box plots (left) + seasonal composition stacked bars (right).

    Shows whether each regime has distinct price levels and seasonal bias.
    LMP is clipped at the 99th percentile to suppress spike distortion.
    """
    merged = regime_df.merge(
        pre[["datetime", "lmp"]], on="datetime", how="left"
    )
    active = merged[merged["regime"] != NOISE_LABEL].copy()
    active["month"]  = pd.to_datetime(active["datetime"]).dt.month
    active["season"] = active["month"].apply(
        lambda m: next((s for s, ms in SEASONS.items() if m in ms), "?")
    )
    active["regime_str"] = active["regime"].astype(str)

    lmp_cap = float(active["lmp"].quantile(0.99))
    active["lmp_clip"] = active["lmp"].clip(upper=lmp_cap)

    K       = len(labels)
    order   = [str(r) for r in labels]
    palette = {str(r): PALETTE(i / K) for i, r in enumerate(labels)}

    season_order  = ["Winter", "Spring", "Summer", "Fall"]
    season_colors = {
        "Winter": "#5b9bd5",
        "Spring": "#70ad47",
        "Summer": "#ffc000",
        "Fall":   "#ed7d31",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: LMP box plot
    sns.boxplot(
        data=active, x="regime_str", y="lmp_clip",
        order=order, palette=palette, ax=axes[0],
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )
    axes[0].set_xlabel("Regime")
    axes[0].set_ylabel(f"LMP $/MWh  (clipped at {lmp_cap:.0f})")
    axes[0].set_title("LMP distribution per regime", fontweight="bold")

    # Right: seasonal stacked bar
    x      = np.arange(K)
    bottom = np.zeros(K)
    for sname in season_order:
        vals = np.array([
            (active[active["regime"] == r]["season"] == sname).sum()
            / max((active["regime"] == r).sum(), 1) * 100
            for r in labels
        ])
        axes[1].bar(x, vals, bottom=bottom, label=sname,
                    color=season_colors[sname], alpha=0.85)
        bottom += vals

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"R{r}" for r in labels])
    axes[1].set_ylabel("% of observations")
    axes[1].set_title("Seasonal composition per regime", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())

    fig.suptitle("Regime characterisation — LMP and seasonality", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_regime_characterisation.png")
    plt.close(fig)
    print("  [plot] 05_regime_characterisation.png")


def plot_umap_transitions(umap_df, regime_df, P, labels):
    """
    06 — UMAP scatter coloured by regime with arrows for most likely transitions.

    For each regime i, one curved arrow points to the most probable destination
    regime j ≠ i. Arrow width scales with P[i,j]. Self-loops are not shown.

    This plot links the geometric (embedding space) view with the dynamic
    (transition matrix) view of the regimes.
    """
    df = umap_df.merge(regime_df[["datetime", "regime"]], on="datetime", how="left")
    K  = len(labels)

    # Compute centroid of each regime in UMAP space
    centroids = {}
    for r in labels:
        sub = df[df["regime"] == r][["umap_1", "umap_2"]]
        if len(sub) > 0:
            centroids[r] = sub.mean().values

    idx = {r: i for i, r in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(9, 7))

    # Noise scatter (background)
    noise_sub = df[df["regime"] == NOISE_LABEL]
    ax.scatter(noise_sub["umap_1"], noise_sub["umap_2"],
               c="lightgrey", s=3, alpha=0.25, label="Noise", rasterized=True)

    # Regime scatter
    for i, r in enumerate(labels):
        sub = df[df["regime"] == r]
        ax.scatter(sub["umap_1"], sub["umap_2"],
                   c=[PALETTE(i / K)], s=5, alpha=0.45,
                   label=f"Regime {r}", rasterized=True)

    # Centroid labels
    for r, c in centroids.items():
        ax.text(c[0], c[1], f"R{r}", fontsize=10, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75))

    # Arrows: most probable non-self transition from each regime
    for i, r in enumerate(labels):
        if r not in centroids:
            continue
        row = P[idx[r]].copy()
        row[idx[r]] = 0.0          # exclude self-loop
        j_best = int(np.argmax(row))
        p_best = float(row[j_best])
        r_to   = labels[j_best]
        if r_to not in centroids or p_best < 1e-3:
            continue
        ax.annotate(
            "",
            xy=centroids[r_to], xytext=centroids[r],
            arrowprops=dict(
                arrowstyle="-|>",
                lw=max(0.8, p_best * 6),
                color="black",
                connectionstyle="arc3,rad=0.15",
                alpha=0.75,
            ),
        )
        # Annotate probability on arrow midpoint
        mid = (centroids[r] + centroids[r_to]) / 2
        ax.text(mid[0], mid[1], f"{p_best:.2f}", fontsize=7, ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="lightyellow", alpha=0.8))

    ax.set_title(
        "UMAP embedding — regime clusters with most probable cross-regime transitions\n"
        "(arrows = most likely next regime from each regime, excluding self-loop)",
        fontweight="bold",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_umap_transitions.png", dpi=150)
    plt.close(fig)
    print("  [plot] 06_umap_transitions.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 05: Transition Matrix")
    print("═" * 65)
    print(f"  Input  : {C.RESULTS_DIR}/regimes.parquet")
    print(f"  Input  : {C.RESULTS_DIR}/umap.parquet")
    print(f"  Input  : {C.RESULTS_DIR}/preprocessed.parquet")
    print(f"  Output : {C.RESULTS_DIR}/transition_matrix.parquet")
    print(f"  Output : {C.RESULTS_DIR}/regime_stats.parquet")
    print(f"  Plots  : {PLOT_DIR}/")
    print()
    print("  Design:")
    print(f"    Stride: {STRIDE_H}h — consecutive pairs define one transition")
    print(f"    Max gap: {MAX_GAP_H}h — cross-gap pairs excluded (data discontinuities)")
    print(f"    Noise windows (regime=-1) excluded from transition counts")
    print(f"    Bootstrap CI on π: {N_BOOTSTRAP} multinomial resamples per row")
    print("─" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    # ── 1. Load inputs ───────────────────────────────────────────────────────
    print("\n  [1/7] Loading inputs …")
    reg_df  = pd.read_parquet(Path(C.RESULTS_DIR) / "regimes.parquet")
    umap_df = pd.read_parquet(Path(C.RESULTS_DIR) / "umap.parquet")
    pre     = pd.read_parquet(Path(C.RESULTS_DIR) / "preprocessed.parquet")

    for df in (reg_df, umap_df, pre):
        df["datetime"] = pd.to_datetime(df["datetime"])

    n_total = len(reg_df)
    n_noise = int((reg_df["regime"] == NOISE_LABEL).sum())
    labels  = sorted(r for r in reg_df["regime"].unique() if r != NOISE_LABEL)
    K       = len(labels)

    print(f"  Windows    : {n_total:,}")
    print(f"  Regimes    : {K}  (labels: {labels})")
    print(f"  Noise (-1) : {n_noise:,}  ({n_noise / n_total * 100:.1f}%)")

    # ── 2. Transition matrix ─────────────────────────────────────────────────
    print("\n  [2/7] Building transition matrix …")
    P, labels, counts = build_transition_matrix(reg_df["regime"],
                                                reg_df["datetime"])
    n_transitions = int(counts.sum())
    print(f"  Valid transitions : {n_transitions:,}")

    # ── 3. Stationary distribution + bootstrap CI ────────────────────────────
    print(f"\n  [3/7] Stationary distribution + bootstrap CI (n={N_BOOTSTRAP}) …")
    pi                    = stationary_distribution(P)
    pi_mean, pi_lo, pi_hi = bootstrap_stationary_ci(P, counts)

    # ── 4. Stationarity test ─────────────────────────────────────────────────
    print("\n  [4/7] Stationarity test (first half vs second half) …")
    stat_result = stationarity_test(reg_df["regime"], reg_df["datetime"])

    # ── 5. Regime statistics ─────────────────────────────────────────────────
    print("\n  [5/7] Computing regime statistics …")
    runs     = compute_run_lengths(reg_df["regime"])
    stats_df = compute_regime_stats(reg_df, pre, P, labels, pi, runs)

    # ── 6. Save outputs ──────────────────────────────────────────────────────
    print("\n  [6/7] Saving outputs …")
    P_df = pd.DataFrame(
        P,
        index=[f"from_{r}" for r in labels],
        columns=[f"to_{r}"  for r in labels],
    )
    P_df.to_parquet(Path(C.RESULTS_DIR) / "transition_matrix.parquet")
    stats_df.to_parquet(Path(C.RESULTS_DIR) / "regime_stats.parquet", index=False)
    print(f"  Saved: transition_matrix.parquet  ({P_df.shape[0]}×{P_df.shape[1]})")
    print(f"  Saved: regime_stats.parquet       ({len(stats_df)} rows)")

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    print("\n  [7/7] Generating plots …")
    plot_transition_heatmap(P, labels, counts, pi)
    plot_regime_timeline(reg_df, labels)
    plot_run_lengths(runs, labels)
    plot_stationary_distribution(pi_mean, pi_lo, pi_hi, labels, stats_df)
    plot_regime_characterisation(reg_df, pre, labels)
    plot_umap_transitions(umap_df, reg_df, P, labels)

    # ── Report ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    print("\n" + "─" * 65)
    print("  TRANSITION MATRIX REPORT")
    print("─" * 65)
    print(f"  Regimes             : {K}")
    print(f"  Total transitions   : {n_transitions:,}")
    print(f"  Noise excluded      : {n_noise:,} windows")
    print()
    hdr = f"  {'Regime':<8} {'π':>6}  {'P(i,i)':>8}  {'E[T_i]':>10}  "  \
          f"{'Median LMP':>12}  {'Dom.Season':<12}"
    print(hdr)
    print("  " + "─" * 64)
    for _, row in stats_df.iterrows():
        soj = row["mean_sojourn_h"]
        soj_str = f"{soj:.0f}h" if soj < 1e4 else "  ∞"
        print(
            f"  R{row['regime']:<7}  "
            f"{row['stationary_prob']:>6.3f}  "
            f"{row['self_transition']:>8.3f}  "
            f"{soj_str:>10}  "
            f"{row['median_lmp']:>12.2f}  "
            f"{row['dominant_season']:<12}"
        )

    print()
    if isinstance(stat_result.get("relative_frobenius_dist"), float):
        d  = stat_result["relative_frobenius_dist"]
        ok = "✓ stationary" if stat_result["is_stationary"] else "⚠ non-stationary"
        print(f"  Stationarity test   : relative Frobenius = {d:.3f}  [{ok}]")
        print(f"  (threshold: < 0.15 = time-homogeneous chain)")
    else:
        note = stat_result.get("note", "")
        print(f"  Stationarity test   : skipped — {note}")

    print(f"\n  Elapsed             : {elapsed:.1f}s")
    print("─" * 65)
    print(f"  transition_matrix.parquet → "
          f"{Path(C.RESULTS_DIR) / 'transition_matrix.parquet'}")
    print(f"  regime_stats.parquet      → "
          f"{Path(C.RESULTS_DIR) / 'regime_stats.parquet'}")
    print(f"  Plots                     → {PLOT_DIR}/")
    print()
    print("  Next → Step 06: Monte Carlo simulation of regime paths")
    print("═" * 65 + "\n")

    return P_df, stats_df


if __name__ == "__main__":
    main()
