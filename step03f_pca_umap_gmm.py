"""
step03f_pca_umap_gmm.py
=======================
Clustering su embedding Chronos2 con pipeline:

  Embedding 7680D
    -> PCA (PCA_COMPONENTS)
    -> UMAP (UMAP_COMPONENTS, cosine)
    -> GMM con BIC per selezione K ottimo

Pipeline
--------
  1. StandardScaler + PCA(20D)   -- rimuove rumore ad alta frequenza
  2. UMAP(10D, cosine)           -- comprime il manifold in spazio piu' denso
  3. GMM sweep K=K_MIN..K_MAX   -- BIC seleziona K ottimo
  4. Plots diagnostici

Output (results/exp_{X}/step03f/)
----------------------------------
  01_umap2d.png          proiezione UMAP 2D colorata per regime
  02_bic_aic.png         BIC e AIC al variare di K
  03_community_sizes.png distribuzione dimensioni per K ottimo
  04_timeline.png        timeline regime
  05_monthly_heatmap.png presenza mensile per regime
  06_pca2d.png           PCA 2D scatter
  07_umap3d.html         UMAP 3D interattivo (plotly)
  08_hourly_heatmap.png  presenza per ora del giorno
  09_weekday_heatmap.png presenza per giorno della settimana
  10_quality_report.png  report qualità econometrica (6 pannelli)
  labels_best.parquet    etichette K ottimo
  labels_K{k}.parquet    etichette per ogni K testato
  best_params.json       K ottimo, parametri e metriche chiave
  quality_report.json    metriche complete per step03_compare

Uso
---
  python step03f_pca_umap_gmm.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, f_oneway, kruskal
from umap import UMAP

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

# ── Esperimento ──────────────────────────────────────────────────────────────
# Passa da linea di comando:  python step03f_pca_umap_gmm.py --exp A
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--exp",     default="A", choices=["A","B","C","D","E","F","G"])
_parser.add_argument("--no-umap", action="store_true",
                     help="Salta UMAP: GMM gira direttamente su PCA(20D)")
_args      = _parser.parse_known_args()[0]
EXPERIMENT = _args.exp
NO_UMAP    = _args.no_umap

EMBEDDINGS_PATH = Path(f"results/exp_{EXPERIMENT}/embeddings.parquet")
# --no-umap salva in step03f_pca per non sovrascrivere i risultati UMAP
OUT_DIR = Path(f"results/exp_{EXPERIMENT}/{'step03f_pca' if NO_UMAP else 'step03f'}")

# Step 1 — PCA
PCA_COMPONENTS   = 20

# Step 2 — UMAP
UMAP_COMPONENTS  = 3
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_METRIC      = "cosine"

# Step 3 — GMM BIC
K_MIN            = 3
K_MAX            = 20
COV_TYPES        = ["full", "diag"]   # tipi covarianza da testare
N_INIT           = 5                  # inizializzazioni GMM per stabilita

# FORCE_K: se None, K viene scelto automaticamente via composite score
#           (0.25*sil + 0.20*DB + 0.25*eta2 + 0.15*transition_rate + 0.15*cramer_season)
# Se impostato (es. FORCE_K=8), ignora il composite score e usa K fisso.
FORCE_K          = None   # None = automatico (raccomandato)
FORCE_COV        = "full"

SEED             = 42
DPI              = 150
MPL_STYLE        = "seaborn-v0_8-whitegrid"

# Parallelismo CPU — -1 = tutti i core disponibili
N_JOBS           = -1

# =============================================================================
# HELPERS
# =============================================================================

CMAP = plt.get_cmap("tab20")

def regime_colors(n):
    return [CMAP(i % 20) for i in range(n)]


def load_embeddings():
    print(f"Carico embeddings da {EMBEDDINGS_PATH}...", flush=True)
    df = pd.read_parquet(EMBEDDINGS_PATH)
    ts_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if ts_col:
        timestamps = pd.to_datetime(df[ts_col[0]])
        E = df.drop(columns=ts_col).values.astype(np.float32)
    else:
        timestamps = pd.Series(range(len(df)))
        E = df.values.astype(np.float32)
    print(f"  shape: {E.shape}", flush=True)
    return E, timestamps


# =============================================================================
# PIPELINE
# =============================================================================

def step1_pca(E):
    print(f"\nStep 1 — StandardScaler + PCA({PCA_COMPONENTS}D)...", flush=True)
    scaler = StandardScaler()
    E_s    = scaler.fit_transform(E)
    # Fit full PCA first to compute scree plot, then keep PCA_COMPONENTS
    pca_full = PCA(random_state=SEED)
    pca_full.fit(E_s)
    pca    = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    E_pca  = pca.fit_transform(E_s)
    var    = pca.explained_variance_ratio_.sum()
    print(f"  varianza spiegata con {PCA_COMPONENTS} PC: {var:.1%}", flush=True)
    # n_components to reach 95% variance
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n95    = int(np.searchsorted(cumvar, 0.95)) + 1
    print(f"  PC per 95% varianza: {n95}  (scelto: {PCA_COMPONENTS})", flush=True)
    return E_pca, pca, pca_full


def step2_umap(E_pca):
    print(f"\nStep 2 — UMAP({UMAP_COMPONENTS}D, {UMAP_METRIC})...", flush=True)
    reducer = UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
        low_memory=False,
        verbose=False,
        n_jobs=N_JOBS,
    )
    E_umap = reducer.fit_transform(E_pca)
    print(f"  shape UMAP: {E_umap.shape}", flush=True)
    return E_umap, reducer


def _fit_one_gmm(k: int, cov_type: str, E_umap: np.ndarray) -> dict:
    """Fit un singolo GMM — chiamato in parallelo da step3_gmm_bic."""
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=cov_type,
        n_init=N_INIT,
        random_state=SEED,
        max_iter=200,
    )
    gmm.fit(E_umap)
    return dict(
        K=k, cov_type=cov_type,
        bic=gmm.bic(E_umap), aic=gmm.aic(E_umap),
        labels=gmm.predict(E_umap), gmm=gmm,
    )


def step3_gmm_bic(E_umap):
    from joblib import Parallel, delayed

    combos = [(k, cov) for cov in COV_TYPES for k in range(K_MIN, K_MAX + 1)]
    n_total = len(combos)
    print(f"\nStep 3 — GMM sweep K={K_MIN}..{K_MAX}  "
          f"({n_total} fit, n_jobs={N_JOBS})...", flush=True)

    results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_fit_one_gmm)(k, cov, E_umap) for k, cov in combos
    )

    # stampa ordinata
    for r in sorted(results, key=lambda x: (x["cov_type"], x["K"])):
        print(f"  K={r['K']:2d}  cov={r['cov_type']:<5}  "
              f"BIC={r['bic']:12.1f}  AIC={r['aic']:12.1f}", flush=True)

    return results


def select_best(results, E_umap: np.ndarray, timestamps: pd.Series):
    """
    Seleziona K ottimo tramite composite score (geometrico + econometrico).

    Score = 0.25*sil_n + 0.20*db_n + 0.25*eta2_n + 0.15*tr_n + 0.15*cramer_n
    dove *_n indica la metrica normalizzata in [0,1] sull'intervallo K_MIN..K_MAX.

    Valuta solo covariance_type = FORCE_COV (default "full").
    Se FORCE_K e' impostato, lo usa come override senza calcolare il composite.

    Riferimenti:
      Silhouette: Rousseeuw (1987)
      eta-squared: Hamilton (1989)
      Composite score K: letteratura clustering multi-criterio
    """
    if FORCE_K is not None:
        matches = [r for r in results
                   if r["K"] == FORCE_K and r["cov_type"] == FORCE_COV]
        if matches:
            best = matches[0]
            print(f"\nBest (forzato FORCE_K): K={best['K']}  cov={best['cov_type']}  "
                  f"BIC={best['bic']:.1f}", flush=True)
            return best

    # Filtra al tipo di covarianza scelto
    cands = [r for r in results if r["cov_type"] == FORCE_COV]
    if not cands:
        cands = results  # fallback: tutti

    # Carica LMP per eta-squared
    pre_path = Path("results/preprocessed.parquet")
    lmp_arr  = None
    if pre_path.exists():
        pre = pd.read_parquet(pre_path)[["datetime", "lmp"]]
        pre["datetime"] = pd.to_datetime(pre["datetime"])
        ts_s = pd.to_datetime(timestamps).reset_index(drop=True)
        merged = pd.DataFrame({"datetime": ts_s}).merge(pre, on="datetime", how="left")
        lmp_arr = merged["lmp"].values

    print("\n  Composite score per K:", flush=True)
    scored = []
    for r in cands:
        lbs = r["labels"]
        K   = r["K"]

        sil = float(silhouette_score(E_umap, lbs, random_state=SEED, n_jobs=N_JOBS))
        db  = float(davies_bouldin_score(E_umap, lbs))
        tr  = float(np.sum(lbs[1:] != lbs[:-1]) / max(len(lbs) - 1, 1))

        ts_ = pd.to_datetime(timestamps).reset_index(drop=True)
        season = ts_.dt.month.map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                                    6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
        ct     = pd.crosstab(lbs, season).values
        cramer = _cramers_v(ct) if ct.shape[1] > 1 else 0.0

        eta2 = 0.0
        if lmp_arr is not None:
            groups = [lmp_arr[lbs == k] for k in range(K)]
            groups = [g[~np.isnan(g)] for g in groups if len(g) > 1]
            if len(groups) >= 2:
                eta2 = _eta_squared(groups)

        scored.append({**r, "_sil": sil, "_db": db, "_tr": tr,
                       "_cramer": cramer, "_eta2": eta2})

    def _norm(vals, direction):
        arr = np.array(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.ones(len(arr)) * 0.5
        n = (arr - mn) / (mx - mn)
        return n if direction == "max" else (1.0 - n)

    sil_n   = _norm([r["_sil"]   for r in scored], "max")
    db_n    = _norm([r["_db"]    for r in scored], "min")
    eta2_n  = _norm([r["_eta2"]  for r in scored], "max")
    tr_n    = _norm([r["_tr"]    for r in scored], "min")
    cramer_n = _norm([r["_cramer"] for r in scored], "max")

    composite = (0.25 * sil_n + 0.20 * db_n + 0.25 * eta2_n
                 + 0.15 * tr_n + 0.15 * cramer_n)

    for i, r in enumerate(scored):
        print(f"    K={r['K']:2d}  sil={r['_sil']:.3f}  DB={r['_db']:.3f}"
              f"  eta2={r['_eta2']:.3f}  tr={r['_tr']:.4f}"
              f"  V_s={r['_cramer']:.3f}  -> score={composite[i]:.4f}",
              flush=True)

    best_idx = int(np.argmax(composite))
    best     = scored[best_idx]
    best["composite_score"] = float(composite[best_idx])

    print(f"\n  => Best (composite): K={best['K']}  cov={best['cov_type']}"
          f"  score={best['composite_score']:.4f}  BIC={best['bic']:.1f}",
          flush=True)
    return best


# =============================================================================
# PLOT
# =============================================================================

def plot_scree(pca_full) -> None:
    """00 - Scree plot: varianza spiegata cumulata vs numero di PC.

    Mostra quante componenti principali bastano per catturare il 95% della
    varianza degli embedding Chronos-2. Giustifica la scelta di PCA_COMPONENTS=20.
    """
    evr  = pca_full.explained_variance_ratio_
    cumv = np.cumsum(evr)

    n95  = int(np.searchsorted(cumv, 0.95)) + 1
    n99  = int(np.searchsorted(cumv, 0.99)) + 1
    nsel = PCA_COMPONENTS

    n_show = min(60, len(evr))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) varianza individuale
    ax = axes[0]
    ax.bar(range(1, n_show + 1), evr[:n_show] * 100,
           color="#4477AA", alpha=0.75, width=0.8)
    ax.axvline(nsel, color="red", lw=1.5, ls="--",
               label=f"scelto = {nsel} PC  ({cumv[nsel-1]:.1%})")
    ax.set_xlabel("Componente principale")
    ax.set_ylabel("Varianza spiegata (%)")
    ax.set_title("Varianza individuale per PC")
    ax.legend(fontsize=8)

    # (b) varianza cumulata
    ax = axes[1]
    ax.plot(range(1, n_show + 1), cumv[:n_show] * 100,
            color="#4477AA", lw=2, marker=".")
    ax.axhline(95, color="orange", lw=1, ls="--", label="95%")
    ax.axhline(99, color="red",    lw=1, ls="--", label="99%")
    ax.axvline(nsel, color="red", lw=1.5, ls="--",
               label=f"scelto = {nsel} PC")
    ax.fill_between(range(1, nsel + 1), cumv[:nsel] * 100,
                    alpha=0.15, color="#4477AA")
    ax.set_xlabel("Numero di PC")
    ax.set_ylabel("Varianza cumulata (%)")
    ax.set_title(f"Varianza cumulata  |  95%={n95} PC  |  99%={n99} PC")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 102)

    fig.suptitle(
        f"Scree plot PCA — Exp {EXPERIMENT}  |  "
        f"Embedding {pca_full.n_features_in_}D → {nsel}D scelto  "
        f"({cumv[nsel-1]:.1%} varianza)",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "00_screeplot.png", dpi=DPI)
    plt.close(fig)
    print("  00_screeplot.png", flush=True)


def plot_umap2d(E_umap, labels, best_K):
    """01 - UMAP 2D colorato per regime."""
    # se UMAP_COMPONENTS > 2, riproietta a 2D per visualizzazione
    if E_umap.shape[1] > 2:
        red2 = UMAP(n_components=2, n_neighbors=UMAP_N_NEIGHBORS,
                    min_dist=UMAP_MIN_DIST, metric="euclidean",
                    random_state=SEED, verbose=False)
        E2 = red2.fit_transform(E_umap)
    else:
        E2 = E_umap

    K      = len(set(labels))
    colors = regime_colors(K)

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(K):
        mask = labels == k
        ax.scatter(E2[mask, 0], E2[mask, 1], c=[colors[k]],
                   s=5, alpha=0.5, label=f"R{k}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP 2D — K={best_K} regimi")
    if K <= 12:
        ax.legend(markerscale=2, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_umap2d.png", dpi=DPI)
    plt.close(fig)
    print("  01_umap2d.png", flush=True)


def plot_bic_aic(results):
    """02 - BIC e AIC al variare di K per ogni tipo covarianza."""
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("labels", "gmm")} for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors_cov = {"full": "#4477AA", "diag": "#EE6677",
                  "tied": "#228833", "spherical": "#CCBB44"}

    for metric, ax in zip(["bic", "aic"], axes):
        for cov_type, grp in df.groupby("cov_type"):
            grp = grp.sort_values("K")
            color = colors_cov.get(cov_type, "gray")
            ax.plot(grp["K"], grp[metric], marker="o", ms=5, lw=1.5,
                    color=color, label=cov_type)
            # evidenzia minimo
            idx_min = grp[metric].idxmin()
            ax.scatter(grp.loc[idx_min, "K"], grp.loc[idx_min, metric],
                       s=80, color=color, zorder=5, marker="*")

        ax.set_xlabel("K")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs K  (stella = minimo)")
        ax.legend(fontsize=8)

    fig.suptitle("Selezione K via BIC/AIC (GMM su UMAP 10D)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_bic_aic.png", dpi=DPI)
    plt.close(fig)
    print("  02_bic_aic.png", flush=True)


def plot_community_sizes(labels, best_K):
    """03 - Distribuzione dimensioni per K ottimo."""
    sizes = pd.Series(labels).value_counts().sort_index()
    K     = len(sizes)
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
    """04 - Timeline regime."""
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
    """05 - Presenza mensile per regime."""
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


def plot_pca2d(E_umap, labels, best_K):
    """06 - PCA 2D sullo spazio UMAP."""
    pca2   = PCA(n_components=2, random_state=SEED)
    E2     = pca2.fit_transform(E_umap)
    K      = len(set(labels))
    colors = regime_colors(K)

    fig, ax = plt.subplots(figsize=(7, 6))
    for k in range(K):
        mask = labels == k
        ax.scatter(E2[mask, 0], E2[mask, 1], c=[colors[k]],
                   s=5, alpha=0.5, label=f"R{k}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA 2D su spazio UMAP  K={best_K}")
    if K <= 12:
        ax.legend(markerscale=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_pca2d.png", dpi=DPI)
    plt.close(fig)
    print("  06_pca2d.png", flush=True)


def plot_hourly_heatmap(labels, timestamps, best_K):
    """08 - Frazione finestre per regime × ora del giorno (post-hoc)."""
    ts     = pd.to_datetime(timestamps).reset_index(drop=True)
    K      = len(set(labels))

    df  = pd.DataFrame({"hour": ts.dt.hour, "regime": labels})
    ct  = df.groupby(["hour", "regime"]).size().unstack(fill_value=0)
    ct  = ct.div(ct.sum(axis=1), axis=0)   # normalizza per ora

    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), 5))
    im = ax.imshow(ct.T.values, aspect="auto", cmap="Blues",
                   vmin=0, vmax=ct.values.max())
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                       rotation=90, fontsize=6)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"R{c}" for c in ct.columns], fontsize=7)
    ax.set_title(f"Regime × Ora del giorno  K={best_K}  (post-hoc, fraction)")
    ax.set_xlabel("Ora")
    fig.colorbar(im, ax=ax, label="Frazione finestre")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_hourly_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  08_hourly_heatmap.png", flush=True)


def plot_weekday_heatmap(labels, timestamps, best_K):
    """09 - Frazione finestre per regime × giorno settimana (post-hoc)."""
    ts     = pd.to_datetime(timestamps).reset_index(drop=True)
    K      = len(set(labels))

    day_names = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
    df  = pd.DataFrame({"dow": ts.dt.dayofweek, "regime": labels})
    ct  = df.groupby(["dow", "regime"]).size().unstack(fill_value=0)
    ct  = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), 4))
    im = ax.imshow(ct.T.values, aspect="auto", cmap="Blues",
                   vmin=0, vmax=ct.values.max())
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names, fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"R{c}" for c in ct.columns], fontsize=7)
    ax.set_title(f"Regime × Giorno settimana  K={best_K}  (post-hoc, fraction)")
    ax.set_xlabel("Giorno")
    fig.colorbar(im, ax=ax, label="Frazione finestre")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "09_weekday_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  09_weekday_heatmap.png", flush=True)


def plot_umap3d(E_umap, labels, timestamps, best_K):
    """07 - Scatter 3D interattivo UMAP colorato per regime (Plotly HTML)."""
    import plotly.graph_objects as go

    ts  = pd.to_datetime(timestamps).reset_index(drop=True)
    K   = len(set(labels))

    # palette tab20 come hex
    cmap = plt.get_cmap("tab20")
    palette = [
        "#{:02x}{:02x}{:02x}".format(
            int(cmap(i % 20)[0] * 255),
            int(cmap(i % 20)[1] * 255),
            int(cmap(i % 20)[2] * 255),
        )
        for i in range(K)
    ]

    # se UMAP_COMPONENTS < 3 non possiamo fare 3D
    if E_umap.shape[1] < 3:
        print("  07_umap3d.html: UMAP_COMPONENTS < 3, skip", flush=True)
        return

    traces = []
    for k in range(K):
        mask = labels == k
        traces.append(go.Scatter3d(
            x=E_umap[mask, 0],
            y=E_umap[mask, 1],
            z=E_umap[mask, 2],
            mode="markers",
            name=f"R{k}",
            text=ts[mask].dt.strftime("%Y-%m-%d"),
            hovertemplate="<b>R" + str(k) + "</b><br>%{text}<extra></extra>",
            marker=dict(size=3, color=palette[k], opacity=0.7),
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"UMAP 3D — K={best_K} regimi (Chronos2 embeddings)",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
        width=900,
        height=700,
    )

    out_path = OUT_DIR / "07_umap3d.html"
    fig.write_html(str(out_path))
    print(f"  07_umap3d.html  (apri nel browser per interazione 3D)", flush=True)


# =============================================================================
# QUALITY ASSESSMENT — metriche di bontà del clustering
# =============================================================================

def _cramers_v(contingency: np.ndarray) -> float:
    """Cramér's V: effect size del chi-quadro [0,1]. 0=nessuna associazione."""
    chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    n    = contingency.sum()
    r, c = contingency.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else 0.0


def _eta_squared(groups: list[np.ndarray]) -> float:
    """Eta-quadro: frazione di varianza spiegata dai gruppi (effect size ANOVA)."""
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum(((x - grand_mean) ** 2).sum() for g in groups for x in [g])
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def compute_quality_metrics(E_umap: np.ndarray, labels: np.ndarray,
                            timestamps: pd.Series) -> dict:
    """
    Calcola tutte le metriche di qualità del clustering nello spazio UMAP.

    Metriche geometriche (spazio UMAP dove gira il GMM):
      silhouette        — coesione+separazione [-1,1], più alto = meglio
      davies_bouldin    — compattezza/separazione [0,∞), più basso = meglio
      calinski_harabasz — between/within variance ratio, più alto = meglio

    Distribuzione regimi:
      entropy_norm      — entropia normalizzata [0,1], 1 = regimi bilanciati
      transition_rate   — freq. cambio regime, più basso = più persistente

    Test temporali (chi-quadro + Cramér's V):
      cramer_hour   — associazione regime × ora del giorno
      cramer_dow    — associazione regime × giorno settimana
      cramer_month  — associazione regime × mese
      cramer_season — associazione regime × stagione
      p_hour/dow/month/season — p-value chi-quadro (< 0.05 = associazione significativa)

    Persistenza (sojourn time):
      sojourn_mean_{k}  — durata media del regime k (ore)
      sojourn_median_{k}
      sojourn_cv_{k}    — coefficiente di variazione (stabilità)

    Matrice di transizione:
      transition_entropy — entropia media delle righe della matrice di transizione
                           (0 = transizioni deterministiche, log K = casuali)
    """
    K   = len(np.unique(labels))
    ts  = pd.to_datetime(timestamps).reset_index(drop=True)
    out = {}

    # ── Metriche geometriche ─────────────────────────────────────────────
    out["silhouette"]        = round(float(silhouette_score(E_umap, labels,
                                     random_state=SEED, n_jobs=N_JOBS)), 4) if K > 1 else 0.0
    out["davies_bouldin"]    = round(float(davies_bouldin_score(E_umap, labels)),
                                     4) if K > 1 else 9999.0
    out["calinski_harabasz"] = round(float(calinski_harabasz_score(E_umap, labels)),
                                     1) if K > 1 else 0.0

    # ── Distribuzione regimi ─────────────────────────────────────────────
    counts = np.bincount(labels)
    p      = counts / counts.sum()
    p_pos  = p[p > 0]
    H      = -np.sum(p_pos * np.log(p_pos))
    out["entropy_norm"]    = round(float(H / np.log(K)) if K > 1 else 0.0, 4)
    out["transition_rate"] = round(
        float(np.sum(labels[1:] != labels[:-1]) / max(len(labels) - 1, 1)), 4)

    # ── Test temporali (chi-quadro + Cramér's V) ─────────────────────────
    df_t = pd.DataFrame({
        "regime" : labels,
        "hour"   : ts.dt.hour,
        "dow"    : ts.dt.dayofweek,
        "month"  : ts.dt.month,
        "season" : ts.dt.month.map({12:0,1:0,2:0, 3:1,4:1,5:1,
                                     6:2,7:2,8:2, 9:3,10:3,11:3}),
    })
    for unit in ["hour", "dow", "month", "season"]:
        ct   = pd.crosstab(df_t["regime"], df_t[unit]).values
        chi2, pval, _, _ = chi2_contingency(ct, correction=False)
        out[f"cramer_{unit}"] = round(_cramers_v(ct), 4)
        out[f"p_{unit}"]      = round(float(pval), 6)

    # ── Sojourn time per regime ───────────────────────────────────────────
    sojourn: dict[int, list[int]] = {k: [] for k in range(K)}
    run_len = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            run_len += 1
        else:
            sojourn[labels[i - 1]].append(run_len)
            run_len = 1
    sojourn[labels[-1]].append(run_len)

    sojourn_stats: dict[str, float] = {}
    for k in range(K):
        d = np.array(sojourn[k], dtype=float)
        if len(d) == 0:
            continue
        sojourn_stats[f"sojourn_mean_R{k}"]   = round(float(d.mean()), 2)
        sojourn_stats[f"sojourn_median_R{k}"] = round(float(np.median(d)), 2)
        sojourn_stats[f"sojourn_cv_R{k}"]     = round(
            float(d.std() / d.mean()) if d.mean() > 0 else 0.0, 4)
    out.update(sojourn_stats)

    # ── Entropia della matrice di transizione ────────────────────────────
    trans = np.zeros((K, K), dtype=float)
    for i in range(len(labels) - 1):
        trans[labels[i], labels[i + 1]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = trans / row_sums
    row_ent = []
    for row in T:
        r = row[row > 0]
        row_ent.append(-np.sum(r * np.log(r)))
    out["transition_entropy"] = round(float(np.mean(row_ent)), 4)
    out["transition_matrix"]  = T.round(4).tolist()

    return out


def compute_price_separation(labels: np.ndarray, timestamps: pd.Series) -> dict:
    """
    Separa i prezzi LMP per regime e testa se le distribuzioni sono
    statisticamente distinte (ANOVA parametrica + Kruskal-Wallis non-parametrica).

    Carica preprocessed.parquet dalla path standard.
    Restituisce dict vuoto se il file non esiste.
    """
    pre_path = Path("results") / "preprocessed.parquet"
    if not pre_path.exists():
        return {}

    pre = pd.read_parquet(pre_path)[["datetime", "lmp", "arcsinh_lmp",
                                      "log_return"]].copy()
    pre["datetime"] = pd.to_datetime(pre["datetime"])

    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    df = pd.DataFrame({"datetime": ts, "regime": labels})
    df = df.merge(pre, on="datetime", how="left")

    if df["lmp"].isna().all():
        return {}

    K      = len(np.unique(labels))
    groups = [df.loc[df["regime"] == k, "lmp"].dropna().values for k in range(K)]
    groups = [g for g in groups if len(g) > 1]

    if len(groups) < 2:
        return {}

    f_stat, p_anova   = f_oneway(*groups)
    h_stat, p_kruskal = kruskal(*groups)

    out = {
        "price_anova_F"      : round(float(f_stat), 3),
        "price_anova_p"      : round(float(p_anova), 6),
        "price_kruskal_H"    : round(float(h_stat), 3),
        "price_kruskal_p"    : round(float(p_kruskal), 6),
        "price_eta_squared"  : round(_eta_squared(groups), 4),
    }
    for k in range(K):
        g = df.loc[df["regime"] == k, "lmp"].dropna()
        if len(g):
            out[f"lmp_mean_R{k}"]   = round(float(g.mean()), 2)
            out[f"lmp_median_R{k}"] = round(float(g.median()), 2)
            out[f"lmp_std_R{k}"]    = round(float(g.std()), 2)
    return out


def save_quality_report(quality: dict, price: dict,
                        best_K: int, best_cov: str) -> None:
    """Salva quality_report.json in OUT_DIR."""
    report = {
        "experiment"       : EXPERIMENT,
        "K"                : best_K,
        "cov_type"         : best_cov,
        "clustering_quality": {
            k: v for k, v in quality.items()
            if k not in ("transition_matrix",)
            and not k.startswith("sojourn_")
        },
        "sojourn_time"     : {k: v for k, v in quality.items()
                              if k.startswith("sojourn_")},
        "transition_matrix": quality.get("transition_matrix", []),
        "price_separation" : price,
    }
    with open(OUT_DIR / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  quality_report.json", flush=True)


def plot_quality_report(quality: dict, price: dict,
                        labels: np.ndarray, best_K: int) -> None:
    """10 - Figura riassuntiva della qualità econometrica del clustering."""
    K      = best_K
    colors = regime_colors(K)

    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # ── (0,0) Silhouette / DB / CH normalizzati ───────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    metrics = {
        "Silhouette\n(↑)": quality.get("silhouette", 0),
        "1−DB/10\n(↑)":    max(0, 1 - quality.get("davies_bouldin", 0) / 10),
        "CH/1e4\n(↑)":     min(1, quality.get("calinski_harabasz", 0) / 1e4),
        "Entropy\n(↑)":    quality.get("entropy_norm", 0),
    }
    ax.bar(range(len(metrics)), list(metrics.values()),
           color=["#4477AA","#EE6677","#228833","#CCBB44"], alpha=0.85)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(list(metrics.keys()), fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title("Qualità geometrica", fontsize=8, fontweight="bold")
    ax.axhline(0.5, color="gray", lw=0.6, ls="--", alpha=0.5)

    # ── (0,1) Cramér's V per scala temporale ─────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    units  = ["hour", "dow", "month", "season"]
    labels_u = ["Ora\n(24)", "Giorno\n(7)", "Mese\n(12)", "Stagione\n(4)"]
    cv = [quality.get(f"cramer_{u}", 0) for u in units]
    pv = [quality.get(f"p_{u}", 1)    for u in units]
    bar_colors = ["#228833" if p < 0.05 else "#AAAAAA" for p in pv]
    ax.bar(range(4), cv, color=bar_colors, alpha=0.85)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_u, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cramér's V", fontsize=7)
    ax.set_title("Associazione temporale\n(verde = p<0.05)", fontsize=8,
                 fontweight="bold")
    ax.axhline(0.1, color="orange", lw=0.8, ls="--", alpha=0.7,
               label="V=0.1 (debole)")
    ax.axhline(0.3, color="red",    lw=0.8, ls="--", alpha=0.7,
               label="V=0.3 (moderata)")
    ax.legend(fontsize=6)

    # ── (0,2) Price separation ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    if price:
        means = [price.get(f"lmp_mean_R{k}", np.nan) for k in range(K)]
        stds  = [price.get(f"lmp_std_R{k}",  0)      for k in range(K)]
        valid = [(k, m, s) for k, m, s in zip(range(K), means, stds)
                 if not np.isnan(m)]
        if valid:
            ks, ms, ss = zip(*valid)
            ax.bar(range(len(ks)), ms, yerr=ss, capsize=3,
                   color=[colors[k] for k in ks], alpha=0.85)
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([f"R{k}" for k in ks], fontsize=7)
            ax.set_ylabel("LMP $/MWh", fontsize=7)
            eta = price.get("price_eta_squared", 0)
            p_a = price.get("price_anova_p", 1)
            ax.set_title(f"LMP per regime\nη²={eta:.3f}  p={p_a:.4f}",
                         fontsize=8, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "preprocessed.parquet\nnon disponibile",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title("LMP per regime", fontsize=8, fontweight="bold")

    # ── (1,0-1) Sojourn time ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    means_s  = [quality.get(f"sojourn_mean_R{k}",   0) for k in range(K)]
    medians_s = [quality.get(f"sojourn_median_R{k}", 0) for k in range(K)]
    x = np.arange(K)
    ax.bar(x - 0.2, means_s,   0.35, label="Media",   color="#4477AA", alpha=0.85)
    ax.bar(x + 0.2, medians_s, 0.35, label="Mediana", color="#EE6677", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{k}" for k in range(K)], fontsize=8)
    ax.set_ylabel("Finestre consecutive", fontsize=8)
    ax.set_title("Sojourn time per regime  (1 finestra = 6h)",
                 fontsize=8, fontweight="bold")
    ax.legend(fontsize=7)

    # ── (1,2) Entropia transizioni ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    te = quality.get("transition_entropy", 0)
    ax.barh(["Transition\nentropy"], [te], color="#AA3377", alpha=0.85)
    ax.barh(["Max entropy\n(log K)"],  [np.log(K) if K > 1 else 1],
            color="#AAAAAA", alpha=0.4)
    ax.set_xlim(0, max(np.log(K) * 1.1, 0.1) if K > 1 else 1.1)
    ax.set_title("Prevedibilità transizioni\n(basso = prevedibili)",
                 fontsize=8, fontweight="bold")
    ax.text(te + 0.02, 0, f"{te:.3f}", va="center", fontsize=8)

    # ── (2,0-2) Matrice di transizione ───────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    T  = np.array(quality.get("transition_matrix", np.eye(K).tolist()))
    im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"R{k}" for k in range(K)], fontsize=7)
    ax.set_yticklabels([f"R{k}" for k in range(K)], fontsize=7)
    ax.set_xlabel("Regime destinazione", fontsize=8)
    ax.set_ylabel("Regime origine",      fontsize=8)
    ax.set_title("Matrice di transizione  (riga = origine, colonna = destinazione)",
                 fontsize=8, fontweight="bold")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{T[i,j]:.2f}", ha="center", va="center",
                    fontsize=max(5, 8 - K), color="white" if T[i,j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Probabilità di transizione", shrink=0.6)

    fig.suptitle(
        f"Quality Report — Exp {EXPERIMENT}  K={K}  "
        f"sil={quality.get('silhouette',0):.3f}  "
        f"DB={quality.get('davies_bouldin',0):.3f}  "
        f"η²={price.get('price_eta_squared', float('nan')):.3f}",
        fontweight="bold", fontsize=10)

    fig.savefig(OUT_DIR / "10_quality_report.png", dpi=DPI)
    plt.close(fig)
    print("  10_quality_report.png", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        plt.style.use(MPL_STYLE)
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    E, timestamps = load_embeddings()

    # Pipeline
    E_pca, _, pca_full = step1_pca(E)

    if NO_UMAP:
        print("\nModalita' PCA-only (--no-umap): GMM su PCA(20D), UMAP saltato.",
              flush=True)
        E_cluster = E_pca
    else:
        E_cluster, _ = step2_umap(E_pca)

    results = step3_gmm_bic(E_cluster)
    best    = select_best(results, E_cluster, timestamps)

    best_K      = best["K"]
    best_labels = best["labels"]
    ts_clean    = pd.to_datetime(timestamps).reset_index(drop=True)

    # Salva etichette
    pd.DataFrame({"timestamp": ts_clean, "regime": best_labels}
                 ).to_parquet(OUT_DIR / "labels_best.parquet", index=False)

    for r in results:
        pd.DataFrame({"timestamp": ts_clean, "regime": r["labels"]}
                     ).to_parquet(
                         OUT_DIR / f"labels_K{r['K']:02d}_{r['cov_type']}.parquet",
                         index=False)

    # Quality assessment (computed in the same space where GMM ran)
    print("\nQuality assessment...", flush=True)
    quality = compute_quality_metrics(E_cluster, best_labels, ts_clean)
    price   = compute_price_separation(best_labels, ts_clean)
    save_quality_report(quality, price, best_K, best["cov_type"])

    # Arricchisce best_params.json con le metriche chiave
    with open(OUT_DIR / "best_params.json", "w") as f:
        json.dump({
            "K"               : best_K,
            "cov_type"        : best["cov_type"],
            "bic"             : best["bic"],
            "aic"             : best["aic"],
            "composite_score" : best.get("composite_score"),
            "pca_components"  : PCA_COMPONENTS,
            "umap_components" : UMAP_COMPONENTS,
            # metriche chiave per step03_compare
            "silhouette"      : quality.get("silhouette"),
            "davies_bouldin"  : quality.get("davies_bouldin"),
            "calinski_harabasz": quality.get("calinski_harabasz"),
            "entropy_norm"    : quality.get("entropy_norm"),
            "transition_rate" : quality.get("transition_rate"),
            "transition_entropy": quality.get("transition_entropy"),
            "price_eta_squared": price.get("price_eta_squared"),
            "cramer_hour"     : quality.get("cramer_hour"),
            "cramer_month"    : quality.get("cramer_month"),
        }, f, indent=2)

    # Plots
    print("\nProduco figure...", flush=True)
    plot_scree(pca_full)
    plot_umap2d(E_cluster, best_labels, best_K)
    plot_bic_aic(results)
    plot_community_sizes(best_labels, best_K)
    plot_timeline(best_labels, timestamps, best_K)
    plot_monthly_heatmap(best_labels, timestamps, best_K)
    plot_pca2d(E_cluster, best_labels, best_K)
    if not NO_UMAP:
        plot_umap3d(E_cluster, best_labels, timestamps, best_K)
    plot_hourly_heatmap(best_labels, timestamps, best_K)
    plot_weekday_heatmap(best_labels, timestamps, best_K)
    plot_quality_report(quality, price, best_labels, best_K)

    print(f"\nDone. K ottimo = {best_K} ({best['cov_type']})  "
          f"Output in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
