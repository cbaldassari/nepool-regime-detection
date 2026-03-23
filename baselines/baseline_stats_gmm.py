"""
step03g_stats_gmm.py
====================
Clustering su statistiche riassuntive per finestra — senza embedding neurali.

Pipeline
--------
  preprocessed.parquet
    -> per ogni finestra di CONTEXT_H ore (stride STRIDE_H):
       calcola ~25 statistiche interpretabili (LMP, load, log_return, fuel mix)
    -> StandardScaler
    -> GMM sweep K=K_MIN..K_MAX con BIC per selezione K ottimo
    -> plots diagnostici

Features per finestra
---------------------
  LMP (sinh di arcsinh_lmp):
    mean, std, p5, p25, p50, p75, p95, iqr,
    spike_ratio (p95/p50), neg_fraction

  log_return:
    mean, std, skewness, kurtosis

  load (total_mw):
    mean, std, peak_mean, offpeak_mean, peak_offpeak_ratio

  ILR coords (7):
    mean di ciascuna

  tempo:
    month_sin, month_cos  (codifica ciclica del mese)

Output (results/step03g/)
--------------------------
  01_bic_aic.png           BIC e AIC al variare di K
  02_feature_importance.png varianza spiegata da ogni feature per regime
  03_community_sizes.png   dimensione regimi per K ottimo
  04_timeline.png          timeline regimi
  05_monthly_heatmap.png   presenza mensile
  06_lmp_boxplot.png       distribuzione LMP per regime
  07_load_boxplot.png      distribuzione load per regime
  08_fuel_mix.png          fuel mix medio per regime
  window_features.parquet  feature per finestra
  labels_best.parquet      etichette K ottimo
  labels_K{k}.parquet      etichette per ogni K
  best_params.json         parametri ottimi

Uso
---
  python step03g_stats_gmm.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

PREPROCESSED    = Path("results/preprocessed.parquet")
OUT_DIR         = Path("results/step03g")

CONTEXT_H       = 720     # ore per finestra (30 giorni)
STRIDE_H        = 6       # stride tra finestre
PEAK_HOURS      = set(range(7, 23))

# GMM
K_MIN           = 3
K_MAX           = 20
COV_TYPES       = ["full", "diag"]
N_INIT          = 10

SEED            = 42
DPI             = 150
MPL_STYLE       = "seaborn-v0_8-whitegrid"

# ILR
ILR_COLS        = [f"ilr_{i}" for i in range(1, 8)]
FUEL_NAMES      = ["Gas", "Nuclear", "Hydro", "Wind", "Solar", "Coal", "Oil", "Other"]
FUEL_COLORS     = ["#e07b39", "#7b52ab", "#4da6e8", "#6abf69",
                   "#f7c948", "#5a5a5a", "#b04040", "#aaaaaa"]

SBP = np.array([
    [ 1,   1,  -1,  -1,  -1,   1,   1,   1],
    [ 1,  -1,   0,   0,   0,   1,   1,  -1],
    [ 1,   0,   0,   0,   0,  -1,  -1,   0],
    [ 0,   0,   0,   0,   0,   1,  -1,   0],
    [ 0,   1,   0,   0,   0,   0,   0,  -1],
    [ 0,   0,   1,  -1,  -1,   0,   0,   0],
    [ 0,   0,   0,  -1,   1,   0,   0,   0],
], dtype=float)


def _ilr_basis(sbp):
    n_coords, D = sbp.shape
    Psi = np.zeros((n_coords, D))
    for j, row in enumerate(sbp):
        pos = row == 1;  neg = row == -1
        r, s = pos.sum(), neg.sum()
        Psi[j, pos] =  np.sqrt(s / (r * (r + s)))
        Psi[j, neg] = -np.sqrt(r / (s * (r + s)))
    return Psi


def ilr_inverse(ilr: np.ndarray) -> np.ndarray:
    Psi   = _ilr_basis(SBP)
    clr   = ilr @ Psi
    exp_c = np.exp(clr)
    return exp_c / exp_c.sum(axis=1, keepdims=True)


CMAP = plt.get_cmap("tab20")

def regime_colors(n):
    return [CMAP(i % 20) for i in range(n)]


SEASON_MAP = {12: "Winter", 1: "Winter",  2: "Winter",
              3:  "Spring", 4: "Spring",  5: "Spring",
              6:  "Summer", 7: "Summer",  8: "Summer",
              9:  "Autumn", 10: "Autumn", 11: "Autumn"}

# =============================================================================
# CALCOLO FEATURE PER FINESTRA
# =============================================================================

def compute_window_features(pre_df):
    """
    Scorre le finestre con stride STRIDE_H e calcola statistiche riassuntive.
    Ritorna DataFrame con una riga per finestra.
    """
    print("Calcolo feature per finestra...", flush=True)
    pre_df = pre_df.sort_index()
    timestamps = pre_df.index

    # genera timestamp di fine finestra
    start_idx = CONTEXT_H - 1
    window_ends = timestamps[start_idx::STRIDE_H]

    records = []
    n = len(window_ends)

    for i, ts in enumerate(window_ends):
        if i % 500 == 0:
            print(f"  {i}/{n}...", flush=True)

        start = ts - pd.Timedelta(hours=CONTEXT_H - 1)
        win   = pre_df.loc[start:ts]

        if len(win) < CONTEXT_H // 2:
            continue

        rec = {"timestamp": ts}

        # -- LMP --
        if "arcsinh_lmp" in win.columns:
            lmp = np.sinh(win["arcsinh_lmp"].values)
            rec["lmp_mean"]      = lmp.mean()
            rec["lmp_std"]       = lmp.std()
            rec["lmp_p5"]        = np.percentile(lmp, 5)
            rec["lmp_p25"]       = np.percentile(lmp, 25)
            rec["lmp_p50"]       = np.percentile(lmp, 50)
            rec["lmp_p75"]       = np.percentile(lmp, 75)
            rec["lmp_p95"]       = np.percentile(lmp, 95)
            rec["lmp_iqr"]       = rec["lmp_p75"] - rec["lmp_p25"]
            p50 = rec["lmp_p50"] if rec["lmp_p50"] != 0 else 1.0
            rec["lmp_spike_ratio"] = rec["lmp_p95"] / abs(p50)
            rec["lmp_neg_frac"]  = (lmp < 0).mean()

        # -- log_return --
        if "log_return" in win.columns:
            lr = win["log_return"].dropna().values
            if len(lr) > 3:
                rec["lr_mean"]     = lr.mean()
                rec["lr_std"]      = lr.std()
                rec["lr_skew"]     = float(scipy_stats.skew(lr))
                rec["lr_kurt"]     = float(scipy_stats.kurtosis(lr))

        # -- load --
        if "total_mw" in win.columns:
            load = win["total_mw"].values
            win_hours = pd.DatetimeIndex(win.index).hour
            peak_mask    = np.isin(win_hours, list(PEAK_HOURS))
            offpeak_mask = ~peak_mask
            rec["load_mean"]   = load.mean()
            rec["load_std"]    = load.std()
            rec["load_peak"]   = load[peak_mask].mean()   if peak_mask.sum() > 0 else np.nan
            rec["load_offpeak"]= load[offpeak_mask].mean() if offpeak_mask.sum() > 0 else np.nan
            if rec.get("load_offpeak", 0) and rec["load_offpeak"] != 0:
                rec["load_po_ratio"] = rec["load_peak"] / rec["load_offpeak"]

        # -- ILR (fuel mix) --
        ilr_avail = [c for c in ILR_COLS if c in win.columns]
        if len(ilr_avail) == 7:
            ilr_means = win[ilr_avail].mean().values
            for j, col in enumerate(ILR_COLS):
                rec[f"{col}_mean"] = ilr_means[j]
            # fuel shares medie
            fuel_shares = ilr_inverse(ilr_means.reshape(1, 7))[0]
            for name, share in zip(FUEL_NAMES, fuel_shares):
                rec[f"fuel_{name}"] = share

        # -- tempo (codifica ciclica) --
        rec["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
        rec["month_cos"] = np.cos(2 * np.pi * ts.month / 12)
        rec["season"]    = SEASON_MAP.get(ts.month, "Unknown")

        records.append(rec)

    df = pd.DataFrame(records)
    print(f"  finestre calcolate: {len(df)}", flush=True)
    return df


# =============================================================================
# GMM BIC
# =============================================================================

def run_gmm_bic(X_scaled):
    print(f"\nGMM sweep K={K_MIN}..{K_MAX}...", flush=True)
    results = []
    for cov_type in COV_TYPES:
        for k in range(K_MIN, K_MAX + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                n_init=N_INIT,
                random_state=SEED,
                max_iter=300,
            )
            gmm.fit(X_scaled)
            bic    = gmm.bic(X_scaled)
            aic    = gmm.aic(X_scaled)
            labels = gmm.predict(X_scaled)
            results.append(dict(K=k, cov_type=cov_type, bic=bic, aic=aic,
                                labels=labels, gmm=gmm))
            print(f"  K={k:2d}  cov={cov_type:<5}  BIC={bic:12.1f}  AIC={aic:12.1f}",
                  flush=True)
    return results


def select_best(results):
    best = min(results, key=lambda r: r["bic"])
    print(f"\nBest: K={best['K']}  cov={best['cov_type']}  BIC={best['bic']:.1f}",
          flush=True)
    return best


# =============================================================================
# PLOT
# =============================================================================

def plot_bic_aic(results):
    """01 - BIC e AIC."""
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("labels", "gmm")} for r in results])
    colors_cov = {"full": "#4477AA", "diag": "#EE6677"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for metric, ax in zip(["bic", "aic"], axes):
        for cov_type, grp in df.groupby("cov_type"):
            grp = grp.sort_values("K")
            color = colors_cov.get(cov_type, "gray")
            ax.plot(grp["K"], grp[metric], marker="o", ms=5,
                    lw=1.5, color=color, label=cov_type)
            idx_min = grp[metric].idxmin()
            ax.scatter(grp.loc[idx_min, "K"], grp.loc[idx_min, metric],
                       s=100, color=color, zorder=5, marker="*")
        ax.set_xlabel("K")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs K  (stella = minimo)")
        ax.legend(fontsize=8)
    fig.suptitle("Selezione K via BIC/AIC — GMM su statistiche finestra", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_bic_aic.png", dpi=DPI)
    plt.close(fig)
    print("  01_bic_aic.png", flush=True)


def plot_feature_importance(df_feat, labels, feat_cols, best_K):
    """02 - Varianza inter-regime per ogni feature (eta^2 approssimato)."""
    eta2 = {}
    for col in feat_cols:
        vals = df_feat[col].values
        grand_mean = vals.mean()
        ss_total = ((vals - grand_mean)**2).sum()
        ss_between = sum(
            (labels == k).sum() * (vals[labels == k].mean() - grand_mean)**2
            for k in range(best_K)
            if (labels == k).sum() > 0
        )
        eta2[col] = ss_between / ss_total if ss_total > 0 else 0.0

    eta2_s = pd.Series(eta2).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(5, len(feat_cols) * 0.3)))
    colors = ["#EE6677" if v > 0.1 else "#4477AA" for v in eta2_s.values]
    ax.barh(eta2_s.index, eta2_s.values, color=colors, alpha=0.8)
    ax.axvline(0.01, color="gray", lw=1, ls=":", label="piccolo (0.01)")
    ax.axvline(0.06, color="gray", lw=1, ls="--", label="medio (0.06)")
    ax.axvline(0.14, color="gray", lw=1, ls="-",  label="grande (0.14)")
    ax.set_xlabel("eta^2 (varianza inter-regime)")
    ax.set_title(f"Importanza feature per separazione regimi  K={best_K}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_feature_importance.png", dpi=DPI)
    plt.close(fig)
    print("  02_feature_importance.png", flush=True)


def plot_community_sizes(labels, best_K):
    """03 - Dimensione regimi."""
    sizes  = pd.Series(labels).value_counts().sort_index()
    K      = len(sizes)
    colors = regime_colors(K)
    total  = sizes.sum()
    fig, ax = plt.subplots(figsize=(max(6, K * 0.7), 4))
    bars = ax.bar(range(K), sizes.values, color=colors, edgecolor="white", lw=0.5)
    for bar, val in zip(bars, sizes.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + total * 0.003,
                f"{val/total:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"R{i}" for i in range(K)], fontsize=8)
    ax.set_ylabel("Finestre")
    ax.set_title(f"Dimensione regimi  K={best_K}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_community_sizes.png", dpi=DPI)
    plt.close(fig)
    print("  03_community_sizes.png", flush=True)


def plot_timeline(labels, timestamps, best_K):
    """04 - Timeline."""
    ts     = pd.to_datetime(timestamps).reset_index(drop=True)
    K      = len(set(labels))
    colors = regime_colors(K)
    fig, ax = plt.subplots(figsize=(14, 3))
    for k in range(K):
        mask = labels == k
        ax.scatter(ts[mask], np.ones(mask.sum()), c=[colors[k]],
                   s=4, marker="|", linewidths=1.5, label=f"R{k}")
    ax.set_yticks([])
    ax.set_xlabel("Data")
    ax.set_title(f"Timeline regimi  K={best_K}")
    if K <= 12:
        ax.legend(loc="upper right", ncol=min(K, 8),
                  fontsize=7, markerscale=3, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_timeline.png", dpi=DPI)
    plt.close(fig)
    print("  04_timeline.png", flush=True)


def plot_monthly_heatmap(labels, timestamps, best_K):
    """05 - Presenza mensile."""
    ts     = pd.to_datetime(timestamps).reset_index(drop=True)
    months = ts.dt.to_period("M").astype(str)
    K      = len(set(labels))
    df = pd.DataFrame({"month": months, "regime": labels})
    ct = df.groupby(["month", "regime"]).size().unstack(fill_value=0)
    ct = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(max(8, len(ct) // 3), max(4, K * 0.4)))
    im = ax.imshow(ct.T.values, aspect="auto", cmap="Blues",
                   vmin=0, vmax=ct.values.max())
    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels(ct.index, rotation=90, fontsize=6)
    ax.set_yticks(range(ct.shape[1]))
    ax.set_yticklabels([f"R{c}" for c in ct.columns], fontsize=7)
    ax.set_title(f"Presenza mensile  K={best_K}")
    fig.colorbar(im, ax=ax, label="Frazione finestre")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_monthly_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  05_monthly_heatmap.png", flush=True)


def plot_lmp_boxplot(df_feat, labels, best_K):
    """06 - LMP per regime."""
    if "lmp_mean" not in df_feat.columns:
        return
    K      = len(set(labels))
    colors = regime_colors(K)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title in zip(axes,
                               ["lmp_mean", "lmp_std"],
                               ["LMP medio ($/MWh)", "LMP std ($/MWh)"]):
        data = [df_feat.loc[labels == k, col].dropna().values for k in range(K)]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={"color": "black", "lw": 1.5},
                        whiskerprops={"lw": 0.8},
                        flierprops={"ms": 2, "alpha": 0.3},
                        showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels([f"R{k}" for k in range(K)], fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title)

    fig.suptitle(f"LMP per regime  K={best_K}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_lmp_boxplot.png", dpi=DPI)
    plt.close(fig)
    print("  06_lmp_boxplot.png", flush=True)


def plot_load_boxplot(df_feat, labels, best_K):
    """07 - Load per regime."""
    if "load_mean" not in df_feat.columns:
        return
    K      = len(set(labels))
    colors = regime_colors(K)
    fig, ax = plt.subplots(figsize=(max(6, K * 0.7), 4))
    data = [df_feat.loc[labels == k, "load_mean"].dropna().values for k in range(K)]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "black", "lw": 1.5},
                    whiskerprops={"lw": 0.8},
                    flierprops={"ms": 2, "alpha": 0.3},
                    showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f"R{k}" for k in range(K)], fontsize=8)
    ax.set_ylabel("Load medio (MW)")
    ax.set_title(f"Load per regime  K={best_K}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_load_boxplot.png", dpi=DPI)
    plt.close(fig)
    print("  07_load_boxplot.png", flush=True)


def plot_fuel_mix(df_feat, labels, best_K):
    """08 - Fuel mix medio per regime."""
    fuel_cols = [f"fuel_{f}" for f in FUEL_NAMES if f"fuel_{f}" in df_feat.columns]
    if not fuel_cols:
        return
    K      = len(set(labels))
    fuel_means = pd.DataFrame(
        {k: df_feat.loc[labels == k, fuel_cols].mean() for k in range(K)}
    ).T

    fig, ax = plt.subplots(figsize=(max(8, K * 0.7), 5))
    bottom = np.zeros(K)
    x = np.arange(K)
    for col, color in zip(fuel_cols, FUEL_COLORS):
        vals = fuel_means[col].values
        ax.bar(x, vals, bottom=bottom, color=color, alpha=0.85,
               label=col.replace("fuel_", ""), edgecolor="white", lw=0.3)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{k}" for k in range(K)], fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Quota fuel mix")
    ax.set_title(f"Fuel mix medio per regime  K={best_K}")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_fuel_mix.png", dpi=DPI)
    plt.close(fig)
    print("  08_fuel_mix.png", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        plt.style.use(MPL_STYLE)
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Carica dati
    print(f"Carico {PREPROCESSED}...", flush=True)
    pre_df = pd.read_parquet(PREPROCESSED)
    pre_df.index = pd.to_datetime(pre_df.index)
    pre_df = pre_df.sort_index()
    print(f"  shape: {pre_df.shape}", flush=True)

    # Feature per finestra
    feat_path = OUT_DIR / "window_features.parquet"
    if feat_path.exists():
        print(f"Carico features cached da {feat_path}...", flush=True)
        df_feat = pd.read_parquet(feat_path)
    else:
        df_feat = compute_window_features(pre_df)
        df_feat.to_parquet(feat_path, index=False)
        print(f"  salvato {feat_path}", flush=True)

    # Seleziona colonne numeriche per clustering (escludi timestamp, season, fuel_*)
    exclude = {"timestamp", "season"} | {f"fuel_{f}" for f in FUEL_NAMES}
    feat_cols = [c for c in df_feat.columns
                 if c not in exclude and df_feat[c].dtype != object]
    feat_cols = [c for c in feat_cols if df_feat[c].notna().mean() > 0.9]
    print(f"\nFeature per clustering ({len(feat_cols)}): {feat_cols}", flush=True)

    # Imputa NaN residui con mediana
    X = df_feat[feat_cols].copy()
    X = X.fillna(X.median())

    # Scala
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # GMM BIC
    results = run_gmm_bic(X_scaled)
    best    = select_best(results)

    best_K      = best["K"]
    best_labels = best["labels"]
    timestamps  = df_feat["timestamp"]
    ts_clean    = pd.to_datetime(timestamps).reset_index(drop=True)

    # Salva etichette
    pd.DataFrame({"timestamp": ts_clean, "regime": best_labels}
                 ).to_parquet(OUT_DIR / "labels_best.parquet", index=False)

    for r in results:
        pd.DataFrame({"timestamp": ts_clean, "regime": r["labels"]}
                     ).to_parquet(
                         OUT_DIR / f"labels_K{r['K']:02d}_{r['cov_type']}.parquet",
                         index=False)

    with open(OUT_DIR / "best_params.json", "w") as f:
        json.dump({"K": best_K, "cov_type": best["cov_type"],
                   "bic": best["bic"], "aic": best["aic"],
                   "n_features": len(feat_cols),
                   "features": feat_cols}, f, indent=2)

    # Plots
    print("\nProduco figure...", flush=True)
    plot_bic_aic(results)
    plot_feature_importance(df_feat, best_labels, feat_cols, best_K)
    plot_community_sizes(best_labels, best_K)
    plot_timeline(best_labels, timestamps, best_K)
    plot_monthly_heatmap(best_labels, timestamps, best_K)
    plot_lmp_boxplot(df_feat, best_labels, best_K)
    plot_load_boxplot(df_feat, best_labels, best_K)
    plot_fuel_mix(df_feat, best_labels, best_K)

    print(f"\nDone. K ottimo = {best_K} ({best['cov_type']})  "
          f"BIC={best['bic']:.1f}  Output in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
