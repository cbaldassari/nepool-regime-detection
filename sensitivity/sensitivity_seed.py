"""
step03f_sensitivity_seed.py
===========================
Sensitività al seed UMAP per l'esperimento vincitore (Exp D).

UMAP è stocastico: seed diversi producono proiezioni diverse.
Questo script mostra che le metriche econometriche chiave (η², transition_rate,
cramer_season) sono stabili al variare del seed, mentre le metriche geometriche
(silhouette, DB) variano di più — il che è atteso perché misurano lo stesso
spazio che cambia.

Procedura:
  1. PCA(20D) — deterministico, eseguito una sola volta
  2. Per ogni seed in SEEDS:
       UMAP(3D) -> GMM K=8 full -> metriche
  3. Coefficiente di variazione (CV = std/mean) per ogni metrica
  4. Plot metriche per seed + CV

Uso
---
  python step03f_sensitivity_seed.py [--exp D]

Output
------
  results/exp_{X}/step03f/12_sensitivity_seed.png
  results/exp_{X}/step03f/sensitivity_seed.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from umap import UMAP

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--exp", default="D",
                     choices=["A","B","C","D","E","F","G","H","I",
                              "J","K","L","M","N","O"])
EXPERIMENT = _parser.parse_known_args()[0].exp

SEEDS = [42, 123, 456, 789, 1234]   # seed da testare

EMBEDDINGS_PATH  = Path(f"results/exp_{EXPERIMENT}/embeddings.parquet")
PREPROC_PATH     = Path("results/preprocessed.parquet")
OUT_DIR          = Path(f"results/exp_{EXPERIMENT}/step03f")

PCA_COMPONENTS   = 20
UMAP_COMPONENTS  = 3
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_METRIC      = "cosine"
FORCE_K          = 8
COV_TYPE         = "full"
N_INIT           = 5
DPI              = 150

# =============================================================================
# HELPERS
# =============================================================================

def _cramers_v(ct):
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n = ct.sum(); r, c = ct.shape
    return float(np.sqrt(chi2 / (n*(min(r,c)-1)))) if min(r,c)>1 else 0.

def _eta_squared(groups):
    gm  = np.concatenate(groups).mean()
    ssb = sum(len(g)*(g.mean()-gm)**2 for g in groups)
    sst = sum(((x-gm)**2).sum() for g in groups for x in [g])
    return float(ssb/sst) if sst > 0 else 0.

def load_pca() -> tuple[np.ndarray, pd.Series]:
    """Carica embeddings e applica PCA — deterministico."""
    df   = pd.read_parquet(EMBEDDINGS_PATH)
    ts_c = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    ts   = pd.to_datetime(df[ts_c[0]]) if ts_c else pd.Series(range(len(df)))
    E    = df.drop(columns=ts_c).values.astype(np.float32)
    E_s  = StandardScaler().fit_transform(E)
    E_pca = PCA(n_components=PCA_COMPONENTS, random_state=42).fit_transform(E_s)
    return E_pca, ts.reset_index(drop=True)

def load_lmp(ts: pd.Series) -> pd.Series | None:
    if not PREPROC_PATH.exists():
        return None
    pre = pd.read_parquet(PREPROC_PATH)[["datetime","lmp"]]
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    df  = pd.DataFrame({"datetime": ts}).merge(pre, on="datetime", how="left")
    return df["lmp"]

def compute_metrics(E_umap: np.ndarray, labels: np.ndarray,
                    ts: pd.Series, lmp: pd.Series | None) -> dict:
    K   = len(np.unique(labels))
    ts_ = pd.to_datetime(ts).reset_index(drop=True)

    sil = float(silhouette_score(E_umap, labels, random_state=42))
    db  = float(davies_bouldin_score(E_umap, labels))
    ch  = float(calinski_harabasz_score(E_umap, labels))
    tr  = float(np.sum(labels[1:]!=labels[:-1]) / max(len(labels)-1, 1))

    season = ts_.dt.month.map({12:0,1:0,2:0,3:1,4:1,5:1,
                                6:2,7:2,8:2,9:3,10:3,11:3})
    ct  = pd.crosstab(labels, season).values
    cv_s = _cramers_v(ct) if ct.shape[1]>1 else 0.

    eta2 = 0.0
    if lmp is not None:
        groups = [lmp.values[labels==k] for k in range(K)]
        groups = [g[~np.isnan(g)] for g in groups if len(g)>1]
        if len(groups) >= 2:
            eta2 = _eta_squared(groups)

    runs, cur, cnt = [], labels[0], 1
    for l in labels[1:]:
        if l == cur: cnt += 1
        else: runs.append(cnt); cur, cnt = l, 1
    runs.append(cnt)

    return {
        "silhouette"      : round(sil,  4),
        "davies_bouldin"  : round(db,   4),
        "calinski_harabasz": round(ch,  1),
        "transition_rate" : round(tr,   4),
        "cramer_season"   : round(cv_s, 4),
        "eta_squared"     : round(eta2, 4),
        "sojourn_mean_h"  : round(float(np.mean(runs)) * 6, 2),
    }

# =============================================================================
# PLOT
# =============================================================================

def plot_seed_sensitivity(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)

    metrics = [
        ("silhouette",       "Silhouette (↑)",          "#4477AA"),
        ("davies_bouldin",   "Davies-Bouldin (↓)",       "#EE6677"),
        ("eta_squared",      "Price η² (↑)",             "#AA3377"),
        ("transition_rate",  "Transition rate (↓)",      "#CCBB44"),
        ("cramer_season",    "Cramér stagione (↑)",      "#228833"),
        ("sojourn_mean_h",   "Sojourn medio ore (↑)",    "#66CCEE"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (col, title, color) in zip(axes, metrics):
        vals   = df[col].values
        seeds  = df["seed"].values
        mean_v = vals.mean()
        std_v  = vals.std()
        cv     = std_v / (abs(mean_v) + 1e-12)

        ax.bar(range(len(seeds)), vals, color=color, alpha=0.75)
        ax.axhline(mean_v, color="black", lw=1.5, ls="--", label=f"media={mean_v:.4f}")
        ax.fill_between([-0.5, len(seeds)-0.5],
                        mean_v - std_v, mean_v + std_v,
                        alpha=0.12, color=color)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"seed\n{s}" for s in seeds], fontsize=7)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.legend(fontsize=6)
        ax.text(0.97, 0.92, f"CV={cv:.4f}", transform=ax.transAxes,
                fontsize=7, ha="right", color="gray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.suptitle(
        f"Sensitività al seed UMAP — Exp {EXPERIMENT}  K=8\n"
        f"CV basso = risultati stabili indipendentemente dall'inizializzazione",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "12_sensitivity_seed.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  12_sensitivity_seed.png", flush=True)

# =============================================================================
# MAIN
# =============================================================================

def main():
    try: plt.style.use("seaborn-v0_8-whitegrid")
    except Exception: pass

    print(f"\nSensitivity analysis UMAP seed — Exp {EXPERIMENT}", flush=True)

    print("\n[1/3] PCA(20D)...", flush=True)
    E_pca, ts = load_pca()
    lmp       = load_lmp(ts)
    print(f"  shape PCA: {E_pca.shape}", flush=True)

    print("\n[2/3] UMAP + GMM per ogni seed...", flush=True)
    rows = []
    for seed in SEEDS:
        print(f"  seed={seed}...", end=" ", flush=True)
        E_umap = UMAP(
            n_components=UMAP_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC,
            random_state=seed, verbose=False,
        ).fit_transform(E_pca)

        gmm    = GaussianMixture(n_components=FORCE_K, covariance_type=COV_TYPE,
                                  n_init=N_INIT, random_state=seed, max_iter=200)
        gmm.fit(E_umap)
        labels = gmm.predict(E_umap)

        m = compute_metrics(E_umap, labels, ts, lmp)
        m["seed"] = seed
        rows.append(m)
        print(f"sil={m['silhouette']:.3f}  η²={m['eta_squared']:.3f}"
              f"  tr={m['transition_rate']:.4f}  V_s={m['cramer_season']:.3f}",
              flush=True)

    # CV per ogni metrica
    df = pd.DataFrame(rows)
    print("\n  Stabilita' al seed (CV = std/mean):", flush=True)
    key_metrics = ["silhouette","davies_bouldin","eta_squared",
                   "transition_rate","cramer_season","sojourn_mean_h"]
    for col in key_metrics:
        cv = df[col].std() / (abs(df[col].mean()) + 1e-12)
        print(f"    {col:<22} CV={cv:.4f}"
              f"  [{df[col].min():.4f} – {df[col].max():.4f}]", flush=True)

    # salva json
    with open(OUT_DIR / "sensitivity_seed.json", "w") as f:
        json.dump({"experiment": EXPERIMENT, "seeds": SEEDS,
                   "results": rows}, f, indent=2)
    print("\n  sensitivity_seed.json", flush=True)

    print("\n[3/3] Figure...", flush=True)
    plot_seed_sensitivity(rows)

    print(f"\nDone. Output in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
