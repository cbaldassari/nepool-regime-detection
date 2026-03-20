"""
step03b_dendrogram.py
=====================
NEPOOL Regime Detection — Step 03b: Hierarchical Clustering Diagnostics

Scopo
-----
Analisi esplorativa della struttura gerarchica degli embedding Chronos-2
tramite Agglomerative Clustering (Ward linkage) applicato direttamente
agli embedding PCA-ridotti.  Nessun UMAP, nessun Ray, gira in locale.

Pipeline
--------
  embeddings.parquet (7680D)
    → StandardScaler
    → PCA (PCA_COMPONENTS D)
    → Agglomerative / Ward  →  dendrogramma + merge distances
    → taglio a K=K_MIN..K_MAX  →  etichette per ogni K
    → plot diagnostici

Output
------
  results/step03b/01_dendrogram.png          dendrogramma troncato (ultime N_SHOW fusioni)
  results/step03b/02_merge_distances.png     distanze di merge + suggerimento K
  results/step03b/03_labels_heatmap.png      stabilità etichette al variare di K
  results/step03b/04_silhouette_vs_k.png     silhouette score per K=K_MIN..K_MAX
  results/step03b/05_pca2d_k{K}.png          PCA-2D scatter per ogni K suggerito
  results/step03b/labels_K{k}.parquet        datetime + regime per ogni K testato

Run
---
  python step03b_dendrogram.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance   import pdist
from sklearn.preprocessing    import StandardScaler
from sklearn.decomposition    import PCA
from sklearn.metrics          import silhouette_score

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  ▶▶  CONFIGURAZIONE  ◀◀
# ═══════════════════════════════════════════════════════════════════════════

# Dimensioni PCA prima del clustering
PCA_COMPONENTS = 64      # 64D cattura ~80% varianza; aumenta per più dettaglio

# Range di K da esplorare nel taglio del dendrogramma
K_MIN = 3
K_MAX = 12

# Quante fusioni mostrare nel dendrogramma troncato
# (le ultime N_SHOW su 6944 totali — le più informative)
N_SHOW = 80

# Linkage method: "ward" minimizza varianza intra-cluster (raccomandato)
# Alternativa: "average", "complete"
LINKAGE_METHOD = "ward"

# Silhouette: subsample per velocità (None = tutti i punti)
SIL_SUBSAMPLE = 3000

# Palette colori per i cluster
PALETTE = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#888888","#1b9e77","#d95f02",
    "#7570b3","#e7298a","#66a61e","#e6ab02",
]

OUT_DIR      = Path(C.RESULTS_DIR) / "step03b"
RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════════════════
#  Caricamento e PCA
# ═══════════════════════════════════════════════════════════════════════════

def load_and_reduce():
    emb_path = Path(C.RESULTS_DIR) / "embeddings.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} non trovato — esegui step02 prima.")

    print("Caricamento embeddings…", flush=True)
    emb_df     = pd.read_parquet(emb_path)
    timestamps = pd.to_datetime(emb_df["datetime"])
    E_raw      = emb_df[[c for c in emb_df.columns
                          if c.startswith("emb_")]].values.astype(np.float32)
    print(f"  {len(E_raw):,} finestre × {E_raw.shape[1]} dims", flush=True)

    print(f"PCA {E_raw.shape[1]}D → {PCA_COMPONENTS}D…", flush=True)
    E_scaled = StandardScaler().fit_transform(E_raw)
    pca      = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE,
                   svd_solver="randomized")
    E        = pca.fit_transform(E_scaled).astype(np.float32)
    var_exp  = pca.explained_variance_ratio_.sum()
    print(f"  ✓ varianza spiegata: {var_exp:.1%}", flush=True)

    # PCA 2D per visualizzazione
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    Z2   = pca2.fit_transform(E_scaled).astype(np.float32)
    var2 = pca2.explained_variance_ratio_.sum()
    print(f"  ✓ PCA-2D varianza: {var2:.1%}", flush=True)

    return timestamps, E, Z2


# ═══════════════════════════════════════════════════════════════════════════
#  Linkage
# ═══════════════════════════════════════════════════════════════════════════

def compute_linkage(E):
    """
    Ward linkage su E (N × D).
    Per Ward, scipy usa l'algoritmo di Lance-Williams → O(N² log N), ~1-5 min per N=7000.
    """
    print(f"\nAgglomerative clustering ({LINKAGE_METHOD} linkage)…", flush=True)
    print(f"  N={len(E):,}  D={E.shape[1]}  → questo può richiedere 1-5 minuti",
          flush=True)
    import time
    t0  = time.time()
    Z   = linkage(E, method=LINKAGE_METHOD, metric="euclidean")
    print(f"  ✓ linkage completato in {time.time()-t0:.1f}s", flush=True)
    return Z


# ═══════════════════════════════════════════════════════════════════════════
#  Suggerimento K automatico
# ═══════════════════════════════════════════════════════════════════════════

def suggest_k(Z, k_min=K_MIN, k_max=K_MAX):
    """
    Analizza le ultime (k_max + 5) distanze di merge.
    Suggerisce K dove c'è il salto relativo più grande
    (accelerazione della distanza di fusione).
    Restituisce k_suggested, merge_distances_array.
    """
    # Le distanze di merge sono nella colonna 2 di Z, ordinate dal basso
    merge_dists = Z[:, 2]

    # Guarda le ultime k_max+10 fusioni (le più rilevanti per il taglio)
    tail = merge_dists[-(k_max + 10):]

    # Accelerazione = seconda derivata discreta delle distanze
    accel = np.diff(tail, 2)

    # Il massimo dell'accelerazione corrisponde al punto di "gomito"
    # Indice nell'array tail: accel[i] corrisponde a tail[i+2]
    # Il numero di cluster associato è len(Z) + 1 - (len(merge_dists) - (k_max+10) + i + 2)
    # Più semplicemente: contiamo dalla fine
    n = len(merge_dists)
    best_i    = int(np.argmax(accel))  # indice in accel
    # posizione in merge_dists: n - (k_max+10) + best_i + 2
    pos_in_Z  = n - (k_max + 10) + best_i + 2
    k_suggested = n + 1 - pos_in_Z   # numero cluster al quel taglio

    # Clamp nel range richiesto
    k_suggested = max(k_min, min(k_max, k_suggested))
    return k_suggested, merge_dists


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_dendrogram(Z, k_suggested):
    """01 — Dendrogramma troncato alle ultime N_SHOW fusioni."""
    fig, ax = plt.subplots(figsize=(16, 6))

    ddata = dendrogram(
        Z,
        ax             = ax,
        truncate_mode  = "lastp",
        p              = N_SHOW,
        leaf_rotation  = 90,
        leaf_font_size = 7,
        show_contracted= True,
        color_threshold= Z[-(k_suggested), 2],   # colora i rami sotto il taglio
    )

    # Linea di taglio per K suggerito
    cut_height = (Z[-(k_suggested), 2] + Z[-(k_suggested-1), 2]) / 2
    ax.axhline(cut_height, color="red", lw=1.5, ls="--",
               label=f"Taglio suggerito  K={k_suggested}")
    ax.legend(fontsize=9)
    ax.set_title(f"Dendrogramma (Ward linkage, ultime {N_SHOW} fusioni su {len(Z)})\n"
                 f"K suggerito = {k_suggested}",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Finestre (o cluster aggregati)")
    ax.set_ylabel("Distanza di merge (Ward)")
    fig.tight_layout()
    out = OUT_DIR / "01_dendrogram.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [plot] {out.name}")


def plot_merge_distances(Z, k_suggested):
    """02 — Distanze di merge + accelerazione + linea di taglio."""
    merge_dists = Z[:, 2]
    n           = len(merge_dists)

    # Mostra le ultime K_MAX*3 fusioni per chiarezza
    window   = K_MAX * 3
    tail_d   = merge_dists[-window:]
    tail_k   = np.arange(window, 0, -1)   # K corrispondente (da window a 1)

    accel    = np.diff(tail_d, 2)
    accel_k  = tail_k[1:-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    # Distanze di merge
    ax1.plot(tail_k, tail_d, marker="o", ms=4, lw=1.5, color="#377eb8")
    ax1.axvline(k_suggested, color="red", lw=1.5, ls="--",
                label=f"K suggerito = {k_suggested}")
    ax1.set_ylabel("Distanza di merge (Ward)")
    ax1.set_title("Distanze di merge al variare di K", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.invert_xaxis()
    ax1.set_xlabel("K (numero cluster)")

    # Accelerazione (seconda derivata)
    ax2.bar(accel_k, accel, color="#e41a1c", alpha=0.7, width=0.6)
    ax2.axvline(k_suggested, color="red", lw=1.5, ls="--",
                label=f"K suggerito = {k_suggested}")
    ax2.set_ylabel("Accelerazione distanza (Δ²)")
    ax2.set_title("Accelerazione delle distanze di merge  (picco = K ottimale)",
                  fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.invert_xaxis()
    ax2.set_xlabel("K (numero cluster)")

    fig.tight_layout()
    out = OUT_DIR / "02_merge_distances.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [plot] {out.name}")


def plot_silhouette(E, labels_dict):
    """04 — Silhouette score per ogni K testato."""
    ks    = sorted(labels_dict.keys())
    sils  = []
    rng   = np.random.default_rng(RANDOM_STATE)

    print("  Calcolo silhouette…", flush=True)
    for k in ks:
        lab = labels_dict[k]
        if len(np.unique(lab)) < 2:
            sils.append(np.nan); continue
        idx = (rng.choice(len(E), min(SIL_SUBSAMPLE, len(E)), replace=False)
               if len(E) > SIL_SUBSAMPLE else np.arange(len(E)))
        sils.append(float(silhouette_score(E[idx], lab[idx], metric="euclidean")))
        print(f"    K={k}  sil={sils[-1]:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ks, sils, marker="o", ms=7, lw=2, color="#4daf4a")
    best_k = ks[int(np.nanargmax(sils))]
    ax.axvline(best_k, color="red", lw=1.5, ls="--",
               label=f"Best silhouette K={best_k}")
    for k, s in zip(ks, sils):
        if not np.isnan(s):
            ax.annotate(f"{s:.3f}", (k, s), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score per K  (Ward linkage, PCA embeddings)",
                 fontweight="bold")
    ax.set_xticks(ks)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "04_silhouette_vs_k.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [plot] {out.name}")
    return best_k


def plot_labels_heatmap(timestamps, labels_dict):
    """
    03 — Heatmap stabilità etichette: per ogni finestra (riga = mese-anno),
    mostra come cambia l'assegnazione al variare di K.
    Permette di vedere quali finestre sono stabili e quali oscillano.
    """
    ks = sorted(labels_dict.keys())

    # Aggrega per mese per leggibilità (altrimenti 6945 righe)
    ts    = pd.to_datetime(timestamps)
    months = ts.to_period("M").astype(str)
    unique_months = sorted(months.unique())

    # Matrice: righe = mesi, colonne = K
    mat = np.zeros((len(unique_months), len(ks)), dtype=float)
    for j, k in enumerate(ks):
        lab = labels_dict[k]
        for i, m in enumerate(unique_months):
            mask = months == m
            if mask.sum() > 0:
                # moda del regime in quel mese per quel K
                vals, cnts = np.unique(lab[mask], return_counts=True)
                mat[i, j]  = float(vals[np.argmax(cnts)])

    fig, ax = plt.subplots(figsize=(max(8, len(ks) * 1.0),
                                    max(6, len(unique_months) * 0.25)))
    # Usa una colormap discreta
    n_max = max(k for k in ks)
    cmap  = plt.get_cmap("tab10", n_max)
    im    = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=n_max,
                      interpolation="nearest")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"K={k}" for k in ks], fontsize=9)
    ax.set_yticks(range(0, len(unique_months), max(1, len(unique_months)//20)))
    ax.set_yticklabels(
        [unique_months[i] for i in range(0, len(unique_months),
                                          max(1, len(unique_months)//20))],
        fontsize=7
    )
    ax.set_xlabel("K (numero cluster scelto)")
    ax.set_ylabel("Mese")
    ax.set_title("Stabilità assegnazione regime al variare di K\n"
                 "(colore = regime dominante nel mese)",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, label="Regime", shrink=0.6)
    fig.tight_layout()
    out = OUT_DIR / "03_labels_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] {out.name}")


def plot_pca2d_for_k(Z2, labels, k, timestamps):
    """05 — PCA-2D scatter per un dato K."""
    reg_ids   = sorted(np.unique(labels).tolist())
    palette   = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(reg_ids)}
    ts        = pd.to_datetime(timestamps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter PCA-2D
    ax = axes[0]
    for r in reg_ids:
        m   = labels == r
        col = palette[r]
        ax.scatter(Z2[m, 0], Z2[m, 1], c=col, s=5, alpha=0.5,
                   linewidths=0, label=f"R{r} (n={m.sum()})")
        cx, cy = Z2[m, 0].mean(), Z2[m, 1].mean()
        ax.text(cx, cy, str(r), fontsize=8, fontweight="bold",
                ha="center", va="center", color="white",
                bbox=dict(boxstyle="circle,pad=0.15", fc=col, ec="none", alpha=0.85),
                zorder=5)
    ax.set_title(f"PCA-2D  —  K={k} regimi (Ward)", fontweight="bold")
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    ax.legend(markerscale=3, fontsize=7, ncol=max(1, k // 6),
              loc="best", framealpha=0.7)

    # Timeline
    ax = axes[1]
    for r in reg_ids:
        m = labels == r
        ax.scatter(ts[m], np.full(m.sum(), r),
                   c=palette[r], s=3, alpha=0.5, linewidths=0)
    ax.set_yticks(reg_ids)
    ax.set_yticklabels([f"R{r}" for r in reg_ids], fontsize=8)
    ax.set_xlabel("Data"); ax.set_ylabel("Regime")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_title(f"Timeline regimi  K={k}", fontweight="bold")

    fig.suptitle(f"Ward linkage — K={k} cluster", fontweight="bold", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / f"05_pca2d_k{k:02d}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [plot] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", font_scale=0.95)
    plt.rcParams.update({"savefig.facecolor": "white", "savefig.bbox": "tight"})

    # ── Carica e riduce ──────────────────────────────────────────────────
    timestamps, E, Z2 = load_and_reduce()

    # ── Linkage ──────────────────────────────────────────────────────────
    Z = compute_linkage(E)

    # ── K suggerito da accelerazione distanze ────────────────────────────
    k_suggested, merge_dists = suggest_k(Z)
    print(f"\nK suggerito (max accelerazione distanze): {k_suggested}", flush=True)

    # ── Taglia per K_MIN..K_MAX e calcola etichette ──────────────────────
    print(f"\nTaglio dendrogramma per K={K_MIN}..{K_MAX}…", flush=True)
    labels_dict = {}
    for k in range(K_MIN, K_MAX + 1):
        labels_dict[k] = fcluster(Z, k, criterion="maxclust").astype(np.int16) - 1
        # fcluster restituisce 1..K, convertiamo a 0..K-1

    # ── Salva etichette ──────────────────────────────────────────────────
    for k, lab in labels_dict.items():
        out = OUT_DIR / f"labels_K{k:02d}.parquet"
        pd.DataFrame({"datetime": timestamps, "regime": lab}).to_parquet(out, index=False)
    print(f"  Salvati labels_K{K_MIN:02d}..labels_K{K_MAX:02d}.parquet in {OUT_DIR}")

    # ── Plot ─────────────────────────────────────────────────────────────
    print(f"\nGenerazione plot → {OUT_DIR}…", flush=True)
    plot_dendrogram(Z, k_suggested)
    plot_merge_distances(Z, k_suggested)
    plot_labels_heatmap(timestamps, labels_dict)
    best_sil_k = plot_silhouette(E, labels_dict)

    # PCA-2D per K suggerito, best silhouette, e K=5 (riferimento)
    for k in sorted({k_suggested, best_sil_k, 5, 6, 7} & set(labels_dict)):
        plot_pca2d_for_k(Z2, labels_dict[k], k, timestamps)

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RISULTATI DENDROGRAMMA")
    print("═" * 60)
    print(f"  K suggerito (accelerazione) : {k_suggested}")
    print(f"  K migliore (silhouette)     : {best_sil_k}")
    print(f"\n  Distribuzione per K={k_suggested}:")
    lab = labels_dict[k_suggested]
    for r, cnt in zip(*np.unique(lab, return_counts=True)):
        print(f"    R{r:2d} : {cnt:5,} finestre ({cnt/len(lab):.1%})")
    print(f"\n  Plot e labels salvati in {OUT_DIR}")
    print("═" * 60)


if __name__ == "__main__":
    main()
