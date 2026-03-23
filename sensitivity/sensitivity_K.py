"""
step03f_sensitivity_K.py
========================
Analisi di sensitività al numero di cluster K per l'esperimento vincitore.

Le etichette per K=3..20 sono gia' salvate in step03f da step03f_pca_umap_gmm.py.
Questo script:
  1. Ri-proietta gli embedding in UMAP (stesso spazio usato in step03f)
  2. Per ogni K in K_RANGE carica labels_K{k:02d}_full.parquet
  3. Calcola metriche geometriche (silhouette, DB) e econometriche
     (eta², transition_rate, cramer_season) sullo spazio UMAP corretto
  4. Produce una figura con le metriche al variare di K

Obiettivo: mostrare che i risultati chiave sono stabili per K=6..10,
rendendo K=8 difendibile anche senza un minimo BIC netto.

Uso
---
  python step03f_sensitivity_K.py [--exp D]

Output
------
  results/exp_{X}/step03f/11_sensitivity_K.png
  results/exp_{X}/step03f/sensitivity_K.json
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
from sklearn.preprocessing import StandardScaler
from umap import UMAP

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--exp", default="D", choices=["A","B","C","D","E","F","G"])
EXPERIMENT = _parser.parse_known_args()[0].exp

K_RANGE          = range(3, 21)     # tutti i K disponibili
K_HIGHLIGHT      = range(6, 11)     # intervallo da evidenziare nel plot

EMBEDDINGS_PATH  = Path(f"results/exp_{EXPERIMENT}/embeddings.parquet")
LABELS_DIR       = Path(f"results/exp_{EXPERIMENT}/step03f")
OUT_DIR          = LABELS_DIR
PREPROC_PATH     = Path("results/preprocessed.parquet")

PCA_COMPONENTS   = 20
UMAP_COMPONENTS  = 3
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_METRIC      = "cosine"
SEED             = 42
DPI              = 150

# =============================================================================
# HELPERS
# =============================================================================

def _cramers_v(contingency: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    n    = contingency.sum()
    r, c = contingency.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else 0.0


def _eta_squared(groups: list[np.ndarray]) -> float:
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum(((x - grand_mean) ** 2).sum() for g in groups for x in [g])
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def load_umap() -> tuple[np.ndarray, pd.Series]:
    """Carica embeddings e proietta in UMAP — stesso spazio di step03f."""
    print(f"  Carico embeddings da {EMBEDDINGS_PATH}...", flush=True)
    df   = pd.read_parquet(EMBEDDINGS_PATH)
    ts_c = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    ts   = pd.to_datetime(df[ts_c[0]]) if ts_c else pd.Series(range(len(df)))
    E    = df.drop(columns=ts_c).values.astype(np.float32)

    print(f"  PCA({PCA_COMPONENTS}D)...", flush=True)
    E_s   = StandardScaler().fit_transform(E)
    E_pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit_transform(E_s)

    print(f"  UMAP({UMAP_COMPONENTS}D)...", flush=True)
    E_umap = UMAP(
        n_components=UMAP_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC,
        random_state=SEED, verbose=False,
    ).fit_transform(E_pca)

    return E_umap, ts.reset_index(drop=True)


def load_labels(k: int) -> np.ndarray | None:
    p = LABELS_DIR / f"labels_K{k:02d}_full.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)["regime"].values.astype(int)


def load_lmp(ts: pd.Series) -> pd.Series | None:
    if not PREPROC_PATH.exists():
        return None
    pre = pd.read_parquet(PREPROC_PATH)[["datetime", "lmp"]]
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    df  = pd.DataFrame({"datetime": ts}).merge(pre, on="datetime", how="left")
    return df["lmp"]


def compute_metrics(E_umap: np.ndarray, labels: np.ndarray,
                    ts: pd.Series, lmp: pd.Series | None) -> dict:
    K = len(np.unique(labels))

    # geometriche
    sil = float(silhouette_score(E_umap, labels, random_state=SEED)) if K > 1 else 0.0
    db  = float(davies_bouldin_score(E_umap, labels))                if K > 1 else 9999.0
    ch  = float(calinski_harabasz_score(E_umap, labels))             if K > 1 else 0.0

    # persistenza
    tr = float(np.sum(labels[1:] != labels[:-1]) / max(len(labels)-1, 1))

    # cramer stagione
    ts_dt  = pd.to_datetime(ts).reset_index(drop=True)
    season = ts_dt.dt.month.map({12:0,1:0,2:0, 3:1,4:1,5:1,
                                  6:2,7:2,8:2, 9:3,10:3,11:3})
    ct     = pd.crosstab(labels, season).values
    cv_s   = _cramers_v(ct) if ct.shape[1] > 1 else 0.0

    # eta² prezzi
    eta2 = 0.0
    if lmp is not None:
        groups = [lmp[labels == k].dropna().values for k in range(K)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            eta2 = _eta_squared(groups)

    # sojourn medio (finestre)
    runs, cur, cnt = [], labels[0], 1
    for l in labels[1:]:
        if l == cur: cnt += 1
        else: runs.append(cnt); cur, cnt = l, 1
    runs.append(cnt)
    sojourn_mean = float(np.mean(runs))

    return {
        "K"              : K,
        "silhouette"     : round(sil,  4),
        "davies_bouldin" : round(db,   4),
        "calinski_harabasz": round(ch, 1),
        "transition_rate": round(tr,   4),
        "cramer_season"  : round(cv_s, 4),
        "eta_squared"    : round(eta2, 4),
        "sojourn_mean_h" : round(sojourn_mean * 6, 2),  # ore
    }


# =============================================================================
# PLOT
# =============================================================================

def plot_sensitivity(rows: list[dict]) -> None:
    df = pd.DataFrame(rows).sort_values("K")

    metrics = [
        ("silhouette",      "Silhouette (↑)",         "#4477AA", "max"),
        ("davies_bouldin",  "Davies-Bouldin (↓)",      "#EE6677", "min"),
        ("eta_squared",     "Price η² (↑)",            "#AA3377", "max"),
        ("transition_rate", "Transition rate (↓)",     "#CCBB44", "min"),
        ("cramer_season",   "Cramer stagione (↑)",     "#228833", "max"),
        ("sojourn_mean_h",  "Sojourn medio (ore) (↑)", "#66CCEE", "max"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (col, title, color, direction) in zip(axes, metrics):
        vals = df[col].values
        ks   = df["K"].values

        ax.plot(ks, vals, color=color, lw=2, marker="o", ms=5, zorder=3)

        # evidenzia K=8
        k8_idx = np.where(ks == 8)[0]
        if len(k8_idx):
            ax.scatter(8, vals[k8_idx[0]], s=120, color=color,
                       zorder=5, edgecolors="black", linewidths=1.5)
            ax.annotate("K=8", (8, vals[k8_idx[0]]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)

        # banda K=6..10
        mask_h = (ks >= 6) & (ks <= 10)
        if mask_h.any():
            ax.axvspan(5.7, 10.3, alpha=0.08, color=color, zorder=1)
            # coefficiente di variazione nell'intervallo (stabilita')
            cv = np.std(vals[mask_h]) / (np.abs(np.mean(vals[mask_h])) + 1e-12)
            ax.text(0.97, 0.05, f"CV K6-10: {cv:.3f}",
                    transform=ax.transAxes, fontsize=7, ha="right",
                    color="gray")

        ax.set_xlabel("K", fontsize=8)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xticks(list(K_RANGE))
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Sensitività a K — Exp {EXPERIMENT}\n"
        f"Zona ombreggiata = K=6..10  |  punto nero = K=8 selezionato",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "11_sensitivity_K.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  11_sensitivity_K.png", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    print(f"\nSensitivity analysis K — Exp {EXPERIMENT}", flush=True)

    # UMAP (run una volta sola)
    print("\n[1/3] PCA + UMAP...", flush=True)
    E_umap, ts = load_umap()

    # LMP per eta²
    lmp = load_lmp(ts)

    # metriche per ogni K
    print("\n[2/3] Calcolo metriche per K=3..20...", flush=True)
    rows = []
    for k in K_RANGE:
        labels = load_labels(k)
        if labels is None:
            print(f"  K={k}: file non trovato — skip", flush=True)
            continue
        m = compute_metrics(E_umap, labels, ts, lmp)
        rows.append(m)
        print(f"  K={k:2d}  sil={m['silhouette']:.3f}  DB={m['davies_bouldin']:.3f}"
              f"  η²={m['eta_squared']:.3f}  tr={m['transition_rate']:.3f}"
              f"  V_s={m['cramer_season']:.3f}  sojourn={m['sojourn_mean_h']:.0f}h",
              flush=True)

    # CV nell'intervallo K=6..10 per ogni metrica
    df_h = pd.DataFrame([r for r in rows if 6 <= r["K"] <= 10])
    print("\n  Stabilita' K=6..10 (CV = std/mean):", flush=True)
    for col in ["silhouette","davies_bouldin","eta_squared",
                "transition_rate","cramer_season","sojourn_mean_h"]:
        cv = df_h[col].std() / (df_h[col].mean().item() + 1e-12)
        print(f"    {col:<22} CV={cv:.4f}", flush=True)

    # salva json
    with open(OUT_DIR / "sensitivity_K.json", "w") as f:
        json.dump({"experiment": EXPERIMENT, "results": rows}, f, indent=2)
    print("\n  sensitivity_K.json", flush=True)

    # plot
    print("\n[3/3] Figure...", flush=True)
    plot_sensitivity(rows)

    print(f"\nDone. Output in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
