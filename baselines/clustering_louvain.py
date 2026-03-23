"""
step03c_louvain.py
==================
Community detection su k-NN graph via Louvain con griglia ottimale su K.

Pipeline
--------
  embeddings.parquet
    -> StandardScaler -> PCA (95% varianza)
    -> k-NN graph (K_NEIGHBORS, cosine similarity)
    -> binary search su risoluzione per ogni K_TARGET in [K_MIN, K_MAX]
    -> N_RUNS run per stabilita (NMI)
    -> criteri multipli: silhouette, Calinski-Harabasz, Davies-Bouldin,
                         stabilita NMI, modularita Q
    -> score composito normalizzato
    -> plots

Output (results/step03c/)
--------------------------
  01_metrics_vs_K.png          tutti i criteri al variare di K
  02_composite_score.png       score composito + K scelto
  03_stability_vs_K.png        NMI stabilita per ogni K
  04_community_sizes.png       distribuzione dimensioni per K ottimo
  05_timeline.png              timeline regime per K ottimo
  06_pca2d.png                 scatter PCA-2D per K ottimo
  07_monthly_heatmap.png       presenza mensile per K ottimo
  grid_results.csv             tabella completa metriche
  labels_best.parquet          etichette per K ottimo
  labels_K{k}.parquet          etichette per ogni K testato

Uso
---
  python step03c_louvain.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score,
                              calinski_harabasz_score,
                              davies_bouldin_score,
                              normalized_mutual_info_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

EMBEDDINGS_PATH = Path("results/embeddings.parquet")
OUT_DIR         = Path("results/step03c")

# PCA
PCA_VARIANCE    = 0.95          # frazione di varianza da mantenere

# k-NN graph
K_NEIGHBORS     = 15            # vicini per il grafo
METRIC          = "cosine"      # metrica distanza

# Griglia K target
K_MIN           = 5
K_MAX           = 30
K_STEP          = 1             # passo nella griglia K

# Stabilita
N_RUNS          = 10            # run Louvain per ogni K (per NMI stabilita)

# Score composito: pesi per ogni metrica (devono sommare a 1)
# silhouette e CH e Q vanno massimizzati; DB va minimizzato (viene invertito)
WEIGHTS = {
    "silhouette":  0.25,
    "CH":          0.20,
    "DB":          0.20,   # invertito: score = 1 / (1 + DB)
    "stability":   0.20,
    "modularity":  0.15,
}

# Subsample per metriche costose
SIL_SUBSAMPLE   = 3000

# Binary search su risoluzione
RES_LO          = 0.001
RES_HI          = 50.0
BS_MAX_ITER     = 40

SEED            = 42
DPI             = 150
MPL_STYLE       = "seaborn-v0_8-whitegrid"

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


def pca_reduce(E):
    print(f"StandardScaler + PCA({PCA_VARIANCE*100:.0f}% varianza)...", flush=True)
    scaler = StandardScaler()
    E_s = scaler.fit_transform(E)
    pca = PCA(n_components=PCA_VARIANCE, random_state=SEED)
    E_pca = pca.fit_transform(E_s)
    print(f"  componenti: {E_pca.shape[1]}  varianza: {pca.explained_variance_ratio_.sum():.1%}",
          flush=True)
    return E_pca


def build_knn_graph(E_pca):
    print(f"k-NN graph (k={K_NEIGHBORS}, {METRIC})...", flush=True)
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric=METRIC, algorithm="brute")
    nn.fit(E_pca)
    distances, indices = nn.kneighbors(E_pca)
    similarities = 1.0 - distances[:, 1:]
    neighbors_idx = indices[:, 1:]

    G = nx.Graph()
    G.add_nodes_from(range(len(E_pca)))
    for i in range(len(E_pca)):
        for j_pos, j in enumerate(neighbors_idx[i]):
            w = float(similarities[i, j_pos])
            if w > 0:
                if G.has_edge(i, j):
                    G[i][j]["weight"] = max(G[i][j]["weight"], w)
                else:
                    G.add_edge(i, j, weight=w)

    print(f"  nodi: {G.number_of_nodes():,}  archi: {G.number_of_edges():,}", flush=True)
    return G


def louvain_labels(G, resolution, seed):
    """Ritorna array di etichette interi per la partizione Louvain."""
    communities = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    labels = np.zeros(G.number_of_nodes(), dtype=int)
    for cid, comm in enumerate(communities):
        for node in comm:
            labels[node] = cid
    return labels, communities


def find_resolution_for_k(G, target_k, lo=RES_LO, hi=RES_HI, max_iter=BS_MAX_ITER):
    """Binary search: trova risoluzione che produce esattamente target_k comunita."""
    best_res, best_k = lo, 0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        comms = louvain_communities(G, weight="weight", resolution=mid, seed=SEED)
        k = len(comms)
        if k == target_k:
            return mid, k
        elif k < target_k:
            lo = mid
        else:
            hi = mid
        if abs(k - target_k) < abs(best_k - target_k):
            best_res, best_k = mid, k
    return best_res, best_k


def compute_metrics(G, E_pca, labels, communities):
    """Calcola silhouette, CH, DB, modularity su un set di etichette."""
    rng = np.random.default_rng(SEED)
    n = len(E_pca)
    K = len(set(labels))

    if K < 2 or K >= n:
        return dict(silhouette=np.nan, CH=np.nan, DB=np.nan, modularity=np.nan)

    idx = (rng.choice(n, min(SIL_SUBSAMPLE, n), replace=False)
           if n > SIL_SUBSAMPLE else np.arange(n))

    sil = silhouette_score(E_pca[idx], labels[idx], metric="cosine")
    ch  = calinski_harabasz_score(E_pca, labels)
    db  = davies_bouldin_score(E_pca, labels)
    Q   = modularity(G, communities, weight="weight")

    return dict(silhouette=sil, CH=ch, DB=db, modularity=Q)


def compute_stability(G, resolution, n_runs=N_RUNS):
    """Esegui n_runs Louvain e calcola NMI medio tra tutte le coppie."""
    all_labels = []
    for s in range(n_runs):
        lbl, _ = louvain_labels(G, resolution, seed=SEED + s)
        all_labels.append(lbl)

    nmis = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            nmis.append(normalized_mutual_info_score(all_labels[i], all_labels[j]))

    return float(np.mean(nmis)), all_labels[0]


# =============================================================================
# GRIGLIA PRINCIPALE
# =============================================================================

def run_grid(G, E_pca):
    K_targets = list(range(K_MIN, K_MAX + 1, K_STEP))
    print(f"\nGriglia K={K_MIN}..{K_MAX} ({len(K_targets)} valori), "
          f"{N_RUNS} run/K per stabilita...", flush=True)

    records = []
    labels_dict = {}

    for k_target in K_targets:
        # 1. binary search risoluzione
        res, k_found = find_resolution_for_k(G, k_target)

        # 2. stabilita: N_RUNS run, NMI medio
        stability, stable_labels = compute_stability(G, res, N_RUNS)
        _, communities = louvain_labels(G, res, seed=SEED)

        # 3. metriche
        m = compute_metrics(G, E_pca, stable_labels, communities)
        k_actual = len(set(stable_labels))

        rec = dict(
            K_target=k_target,
            K_actual=k_actual,
            resolution=res,
            silhouette=m["silhouette"],
            CH=m["CH"],
            DB=m["DB"],
            modularity=m["modularity"],
            stability=stability,
        )
        records.append(rec)
        labels_dict[k_target] = stable_labels

        print(f"  K={k_target:3d} (trovato={k_actual:3d}) "
              f"sil={m['silhouette']:.4f}  CH={m['CH']:8.1f}  "
              f"DB={m['DB']:.4f}  Q={m['modularity']:.4f}  "
              f"stab={stability:.4f}", flush=True)

    return pd.DataFrame(records), labels_dict


# =============================================================================
# SCORE COMPOSITO
# =============================================================================

def compute_composite(df):
    """
    Normalizza ogni metrica in [0,1] e calcola score composito pesato.
    DB viene invertito: score_DB = 1 / (1 + DB), poi normalizzato.
    """
    mm = MinMaxScaler()

    df = df.copy()
    df["DB_inv"] = 1.0 / (1.0 + df["DB"])

    cols_norm = {
        "silhouette": "silhouette",
        "CH":         "CH",
        "DB_inv":     "DB",
        "stability":  "stability",
        "modularity": "modularity",
    }

    for col_raw, weight_key in cols_norm.items():
        vals = df[col_raw].values.reshape(-1, 1)
        normed = mm.fit_transform(vals).flatten()
        df[f"norm_{weight_key}"] = normed

    df["composite"] = (
        WEIGHTS["silhouette"] * df["norm_silhouette"] +
        WEIGHTS["CH"]         * df["norm_CH"] +
        WEIGHTS["DB"]         * df["norm_DB"] +
        WEIGHTS["stability"]  * df["norm_stability"] +
        WEIGHTS["modularity"] * df["norm_modularity"]
    )

    return df


# =============================================================================
# PLOT
# =============================================================================

def plot_metrics_vs_K(df):
    """01 - Tutti i criteri al variare di K."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    metrics = [
        ("silhouette",  "Silhouette (max)",        "#4477AA", False),
        ("CH",          "Calinski-Harabasz (max)", "#EE6677", False),
        ("DB",          "Davies-Bouldin (min)",    "#228833", False),
        ("stability",   "Stabilita NMI (max)",     "#CCBB44", False),
        ("modularity",  "Modularita Q (max)",      "#AA3377", False),
        ("composite",   "Score composito",         "#000000", True),
    ]

    Ks = df["K_target"].values

    for ax, (col, title, color, is_composite) in zip(axes, metrics):
        vals = df[col].values if not is_composite else df["composite"].values
        ax.plot(Ks, vals, color=color, lw=1.8, marker="o", ms=4)

        if is_composite:
            best_idx = int(np.nanargmax(vals))
            ax.axvline(Ks[best_idx], color="red", lw=1.2, ls="--",
                       label=f"K ottimo={Ks[best_idx]}")
            ax.legend(fontsize=8)

        ax.set_xlabel("K")
        ax.set_ylabel(col)
        ax.set_title(title, fontsize=9)

    fig.suptitle("Metriche di clustering al variare di K (Louvain)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_metrics_vs_K.png", dpi=DPI)
    plt.close(fig)
    print("  01_metrics_vs_K.png salvato", flush=True)


def plot_composite(df):
    """02 - Score composito con K ottimo evidenziato."""
    best_idx = int(np.nanargmax(df["composite"].values))
    best_K   = int(df["K_target"].iloc[best_idx])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["K_target"], df["composite"], color="#4477AA", alpha=0.7,
           edgecolor="white", linewidth=0.5)
    ax.axvline(best_K, color="red", lw=2, ls="--", label=f"K ottimo = {best_K}")
    ax.set_xlabel("K")
    ax.set_ylabel("Score composito")
    ax.set_title("Score composito normalizzato (silhouette + CH + DB + stabilita + modularita)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_composite_score.png", dpi=DPI)
    plt.close(fig)
    print(f"  02_composite_score.png salvato  (K ottimo = {best_K})", flush=True)
    return best_K


def plot_stability(df):
    """03 - Stabilita NMI per ogni K."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["K_target"], df["stability"], color="#CCBB44",
            lw=1.8, marker="o", ms=4)
    ax.set_xlabel("K")
    ax.set_ylabel("NMI medio tra run")
    ax.set_title(f"Stabilita Louvain ({N_RUNS} run per K)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_stability_vs_K.png", dpi=DPI)
    plt.close(fig)
    print("  03_stability_vs_K.png salvato", flush=True)


def plot_community_sizes(labels, best_K):
    """04 - Distribuzione dimensioni comunita per K ottimo."""
    sizes = pd.Series(labels).value_counts().sort_index()
    K = len(sizes)
    colors = regime_colors(K)
    total = sizes.sum()

    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), 4))
    bars = ax.bar(range(K), sizes.values, color=colors, edgecolor="white", lw=0.5)
    for bar, val in zip(bars, sizes.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total * 0.003,
                f"{val/total:.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"C{i}" for i in range(K)], fontsize=7)
    ax.set_ylabel("Finestre")
    ax.set_title(f"Dimensione comunita  K={best_K}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_community_sizes.png", dpi=DPI)
    plt.close(fig)
    print("  04_community_sizes.png salvato", flush=True)


def plot_timeline(labels, timestamps, best_K):
    """05 - Timeline regime."""
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    K  = len(set(labels))
    colors = regime_colors(K)

    fig, ax = plt.subplots(figsize=(14, 3))
    for k in range(K):
        mask = labels == k
        ax.scatter(ts[mask], np.ones(mask.sum()), c=[colors[k]], s=4,
                   marker="|", linewidths=1.5, label=f"C{k}")
    ax.set_yticks([])
    ax.set_xlabel("Data")
    ax.set_title(f"Timeline comunita Louvain  K={best_K}")
    if K <= 15:
        ax.legend(loc="upper right", ncol=min(K, 8), fontsize=7,
                  markerscale=3, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_timeline.png", dpi=DPI)
    plt.close(fig)
    print("  05_timeline.png salvato", flush=True)


def plot_pca2d(labels, E_pca, timestamps, best_K):
    """06 - PCA-2D scatter + timeline."""
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    K  = len(set(labels))
    colors = regime_colors(K)

    pca2 = PCA(n_components=2, random_state=SEED)
    E2   = pca2.fit_transform(E_pca)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for k in range(K):
        mask = labels == k
        ax.scatter(E2[mask, 0], E2[mask, 1], c=[colors[k]], s=6,
                   alpha=0.6, label=f"C{k}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA-2D  K={best_K}")
    if K <= 15:
        ax.legend(markerscale=2, fontsize=7)

    ax2 = axes[1]
    numeric_ts = (ts - ts.min()).dt.total_seconds().values
    ax2.scatter(numeric_ts, np.ones(len(ts)), c=labels,
                cmap=mcolors.ListedColormap(colors[:K]), s=4,
                marker="|", linewidths=1.5)
    n_ticks = 6
    tick_idx = np.linspace(0, len(ts) - 1, n_ticks, dtype=int)
    ax2.set_xticks(numeric_ts[tick_idx])
    ax2.set_xticklabels([ts.iloc[i].strftime("%Y-%m") for i in tick_idx],
                        rotation=30, ha="right")
    ax2.set_yticks([])
    ax2.set_title("Timeline")

    fig.suptitle(f"Louvain K={best_K}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_pca2d.png", dpi=DPI)
    plt.close(fig)
    print("  06_pca2d.png salvato", flush=True)


def plot_monthly_heatmap(labels, timestamps, best_K):
    """07 - Heatmap presenza mensile."""
    ts     = pd.to_datetime(timestamps).reset_index(drop=True)
    months = ts.dt.to_period("M").astype(str)
    K      = len(set(labels))

    df = pd.DataFrame({"month": months, "community": labels})
    ct = df.groupby(["month", "community"]).size().unstack(fill_value=0)
    ct = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(8, len(ct) // 3), max(4, K * 0.35)))
    im = ax.imshow(ct.T.values, aspect="auto", cmap="Blues",
                   vmin=0, vmax=ct.values.max())
    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels(ct.index, rotation=90, fontsize=6)
    ax.set_yticks(range(ct.shape[1]))
    ax.set_yticklabels([f"C{c}" for c in ct.columns], fontsize=7)
    ax.set_xlabel("Mese")
    ax.set_ylabel("Comunita")
    ax.set_title(f"Presenza mensile  K={best_K}")
    fig.colorbar(im, ax=ax, label="Frazione finestre")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_monthly_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  07_monthly_heatmap.png salvato", flush=True)


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
    E_pca         = pca_reduce(E)
    G             = build_knn_graph(E_pca)

    # Griglia
    df_grid, labels_dict = run_grid(G, E_pca)

    # Score composito
    df_grid = compute_composite(df_grid)

    # Salva CSV
    df_grid.to_csv(OUT_DIR / "grid_results.csv", index=False)
    print(f"\nGrid results salvato in {OUT_DIR}/grid_results.csv", flush=True)

    # Salva etichette per ogni K
    ts_clean = pd.to_datetime(timestamps).reset_index(drop=True)
    for k_target, lbl in labels_dict.items():
        pd.DataFrame({"timestamp": ts_clean, "community": lbl}
                     ).to_parquet(OUT_DIR / f"labels_K{k_target:02d}.parquet", index=False)

    # K ottimo
    best_idx = int(np.nanargmax(df_grid["composite"].values))
    best_K   = int(df_grid["K_target"].iloc[best_idx])
    best_row = df_grid.iloc[best_idx]
    print(f"\n-- K ottimo = {best_K} -----------------------------------------")
    print(f"  silhouette : {best_row['silhouette']:.4f}")
    print(f"  CH         : {best_row['CH']:.1f}")
    print(f"  DB         : {best_row['DB']:.4f}")
    print(f"  stabilita  : {best_row['stability']:.4f}")
    print(f"  modularita : {best_row['modularity']:.4f}")
    print(f"  composite  : {best_row['composite']:.4f}")

    best_labels = labels_dict[best_K]
    pd.DataFrame({"timestamp": ts_clean, "community": best_labels}
                 ).to_parquet(OUT_DIR / "labels_best.parquet", index=False)

    # Plots
    print("\nProduco figure...", flush=True)
    plot_metrics_vs_K(df_grid)
    plot_composite(df_grid)
    plot_stability(df_grid)
    plot_community_sizes(best_labels, best_K)
    plot_timeline(best_labels, timestamps, best_K)
    plot_pca2d(best_labels, E_pca, timestamps, best_K)
    plot_monthly_heatmap(best_labels, timestamps, best_K)

    print(f"\nDone. Output in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
