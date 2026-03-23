"""
step03_compare.py
=================
Confronto tra esperimenti A/B/C/D per eleggere il clustering migliore.

Metodologia
-----------
Le metriche vengono caricate da ``quality_report.json`` prodotto da step03f
per ciascun esperimento.  NON si ricalcolano PCA/UMAP qui: le metriche
geometriche (silhouette, DB, CH) devono essere valutate nello stesso spazio
UMAP dove gira il GMM — altrimenti confrontano spazi diversi.

Per la griglia UMAP 2D viene eseguita una ri-proiezione SOLO a scopo
visivo (chiaramente annotata nel plot).

Metriche TOPSIS
---------------
  silhouette         0.25  — coesione+separazione nello spazio GMM
  davies_bouldin     0.20  — compattezza (complementare, robusta agli outlier)
  price_eta_squared  0.20  — separazione econometrica dei prezzi LMP
  transition_rate    0.15  — persistenza regime (domain-specific)
  entropy_norm       0.10  — bilancio regimi (evita soluzioni degeneri)
  cramer_month       0.10  — associazione stagionale (rilevanza interpretativa)

Ranking finale — TOPSIS (Hwang & Yoon 1981)
-------------------------------------------
  1. Normalizzazione vettoriale
  2. Ponderazione
  3. Soluzioni ideali A+ e A-
  4. Distanza euclidea da A+ e A-
  5. C_i = D- / (D+ + D-)  ∈ [0,1], più alto = migliore

Output
------
  results/comparison/
    comparison_table.csv   — tutte le metriche + TOPSIS
    01_metrics.png         — barplot metriche normalizzate + score
    02_umap2d_grid.png     — 4 UMAP 2D affiancati (solo visualizzazione)
    winner.json
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

EXPERIMENTS = ["A", "B", "C", "D", "E", "F", "G"]
EXP_LABELS  = {
    "A": "log_return",
    "B": "arcsinh_lmp",
    "C": "mstl_resid_lr",
    "D": "mstl_resid_arcsinh",
    "E": "log_lmp_shifted",
    "F": "lmp_clipped",
    "G": "mstl_resid_log",
}

# Parametri UMAP per la griglia visiva (deve coincidere con step03f)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_METRIC      = "cosine"
PCA_COMPONENTS   = 20
SEED             = 42
DPI              = 150

RESULTS_DIR = Path("results")
OUT_DIR     = RESULTS_DIR / "comparison"

# =============================================================================
# TOPSIS CRITERIA — due set separati
# =============================================================================

# TOPSIS-G: criteri geometrici (interni allo spazio UMAP/GMM)
# Rousseeuw 1987; Davies & Bouldin 1979
TOPSIS_GEOM = {
    #  colonna              direzione  peso
    "silhouette"        : ("max", 0.50),
    "davies_bouldin"    : ("min", 0.30),
    "calinski_harabasz" : ("max", 0.20),
}

# TOPSIS-E: criteri econometrici (esterni — non dipendono dallo spazio UMAP)
# η²: Hamilton 1989; sojourn/transition: Huisman & Mahieu 2003; Cramér: Wichern 1988
TOPSIS_ECON = {
    #  colonna              direzione  peso
    "price_eta_squared" : ("max", 0.40),  # separazione distribuz. prezzi LMP
    "transition_rate"   : ("min", 0.30),  # persistenza regime (basso = meglio)
    "cramer_season"     : ("max", 0.20),  # struttura stagionale
    "entropy_norm"      : ("max", 0.10),  # bilanciamento regimi
}

# =============================================================================
# HELPERS
# =============================================================================

CMAP = plt.get_cmap("tab20")

def regime_colors(n):
    return [CMAP(i / max(n, 1)) for i in range(n)]


def load_experiment(exp: str) -> dict | None:
    """
    Carica quality_report.json, labels_best.parquet e best_params.json per
    l'esperimento.  Restituisce None se i file non esistono.
    """
    step03f_dir  = RESULTS_DIR / f"exp_{exp}" / "step03f"
    qr_path      = step03f_dir / "quality_report.json"
    labels_path  = step03f_dir / "labels_best.parquet"
    emb_path     = RESULTS_DIR / f"exp_{exp}" / "embeddings.parquet"

    for p in [qr_path, labels_path]:
        if not p.exists():
            print(f"  ⚠  exp_{exp}: {p.name} non trovato — skip", flush=True)
            return None

    with open(qr_path) as f:
        qr = json.load(f)

    lbl_df = pd.read_parquet(labels_path)
    labels = lbl_df["regime"].values
    ts     = pd.to_datetime(lbl_df["timestamp"])

    cq = qr.get("clustering_quality", {})
    ps = qr.get("price_separation", {})

    # ── Valori medi su 5 seed (opzione A — robusti all'inizializzazione UMAP)
    # Se sensitivity_seed.json esiste, sostituisce price_eta_squared e
    # transition_rate con le medie; altrimenti usa il valore seed=42.
    seed_path = step03f_dir / "sensitivity_seed.json"
    eta2_mean  = ps.get("price_eta_squared", 0.0)
    eta2_std   = 0.0
    tr_mean    = cq.get("transition_rate", 1.0)
    cs_mean    = cq.get("cramer_season", 0.0)
    if seed_path.exists():
        with open(seed_path) as f:
            ss = json.load(f)
        sr = ss.get("results", [])
        if sr:
            import numpy as _np
            eta2_mean = float(_np.mean([r.get("eta_squared", 0) for r in sr]))
            eta2_std  = float(_np.std( [r.get("eta_squared", 0) for r in sr]))
            tr_mean   = float(_np.mean([r.get("transition_rate", 1) for r in sr]))
            cs_mean   = float(_np.mean([r.get("cramer_season", 0) for r in sr]))

    data = {
        "exp"    : exp,
        "K"      : qr.get("K"),
        "cov"    : qr.get("cov_type"),
        "labels" : labels,
        "ts"     : ts,
        "quality": cq,
        "price"  : ps,
        # metriche estratte per TOPSIS (medie su 5 seed dove disponibili)
        "silhouette"        : cq.get("silhouette", 0.0),
        "davies_bouldin"    : cq.get("davies_bouldin", 9999.0),
        "calinski_harabasz" : cq.get("calinski_harabasz", 0.0),
        "entropy_norm"      : cq.get("entropy_norm", 0.0),
        "transition_rate"   : tr_mean,
        "transition_entropy": cq.get("transition_entropy", 0.0),
        "cramer_hour"       : cq.get("cramer_hour", 0.0),
        "cramer_month"      : cq.get("cramer_month", 0.0),
        "cramer_season"     : cs_mean,
        "price_eta_squared" : eta2_mean,
        "price_eta_squared_std": eta2_std,
        "price_anova_p"     : ps.get("price_anova_p", 1.0),
        # embeddings solo per la griglia visiva
        "_emb_path": emb_path if emb_path.exists() else None,
    }
    return data


# =============================================================================
# TOPSIS  (Hwang & Yoon, 1981)
# =============================================================================

def topsis(df: pd.DataFrame, criteria: dict, suffix: str) -> pd.DataFrame:
    """
    TOPSIS multi-criteria ranking (Hwang & Yoon 1981).

    Passi:
      1. Normalizzazione vettoriale: r_ij = x_ij / ||x_j||
      2. Matrice pesata: v_ij = w_j * r_ij
      3. Soluzione ideale positiva A+ e negativa A-
      4. Distanze: D+_i = ||v_i - A+||,  D-_i = ||v_i - A-||
      5. Indice di vicinanza: C_i = D- / (D+ + D-)  ∈ [0,1]

    suffix: stringa aggiunta ai nomi delle colonne risultato (es. "_geom", "_econ")
    """
    df   = df.copy()
    cols = list(criteria.keys())
    dirs = [criteria[c][0] for c in cols]
    wts  = np.array([criteria[c][1] for c in cols])

    X = df[cols].values.astype(float)

    norms = np.sqrt((X ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    R = X / norms
    V = R * wts

    A_pos = np.where([d == "max" for d in dirs], V.max(axis=0), V.min(axis=0))
    A_neg = np.where([d == "max" for d in dirs], V.min(axis=0), V.max(axis=0))

    D_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    denom = D_pos + D_neg
    denom[denom == 0] = 1e-12
    C = D_neg / denom

    df[f"topsis_score{suffix}"] = np.round(C, 4)
    df[f"topsis_rank{suffix}"]  = (-C).argsort().argsort() + 1
    return df


# =============================================================================
# PLOTS
# =============================================================================

def _normalise_for_plot(df: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    """Normalizza le colonne dei criteri in [0,1] orientate 'più alto = meglio'."""
    norm = df[list(criteria.keys())].copy().astype(float)
    for col, (direction, _) in criteria.items():
        rng = norm[col].max() - norm[col].min()
        if rng < 1e-12:
            norm[col] = 0.5
        elif direction == "min":
            norm[col] = 1 - (norm[col] - norm[col].min()) / rng
        else:
            norm[col] = (norm[col] - norm[col].min()) / rng
    return norm


def plot_metrics(df: pd.DataFrame) -> None:
    """01 - Figura a due blocchi: TOPSIS-G (geometrico) e TOPSIS-E (econometrico)."""

    exp_labels = [f"Exp {r['exp']}\n({r['feature']})" for _, r in df.iterrows()]
    x = np.arange(len(df))

    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── riga 0: TOPSIS-G (geometrico — interno) ───────────────────────────
    ax_g  = fig.add_subplot(gs[0, :2])
    ax_gs = fig.add_subplot(gs[0, 2])

    norm_g  = _normalise_for_plot(df, TOPSIS_GEOM)
    cols_g  = list(TOPSIS_GEOM.keys())
    nice_g  = ["Silhouette ↑\n(w=0.50)", "Davies-Bouldin ↑*\n(w=0.30)",
               "Calinski-H ↑\n(w=0.20)"]
    clrs_g  = ["#4477AA", "#EE6677", "#228833"]
    width_g = 0.25

    for i, (col, label, color) in enumerate(zip(cols_g, nice_g, clrs_g)):
        ax_g.bar(x + i * width_g, norm_g[col].values, width_g,
                 label=label, color=color, alpha=0.85)

    ax_g.set_xticks(x + width_g)
    ax_g.set_xticklabels(exp_labels, fontsize=8)
    ax_g.set_ylim(0, 1.2)
    ax_g.set_ylabel("Score normalizzato [0-1]")
    ax_g.set_title("TOPSIS-G  |  Criteri geometrici (interni allo spazio UMAP)",
                   fontweight="bold")
    ax_g.legend(fontsize=7, loc="upper right")
    # stella vincitore geometrico
    w_g = df["topsis_rank_geom"].values.argmin()
    ax_g.annotate("*", xy=(w_g + width_g, 1.13), ha="center",
                  fontsize=16, color="#4477AA", fontweight="bold")

    scores_g  = df["topsis_score_geom"].values
    order_g   = df["topsis_rank_geom"].values.argsort()
    bc_g = ["#4477AA" if r == 1 else "#AAAAAA" for r in df["topsis_rank_geom"].values]
    bars = ax_gs.barh(range(len(df)), scores_g[order_g], color=[bc_g[i] for i in order_g], alpha=0.85)
    ax_gs.set_yticks(range(len(df)))
    ax_gs.set_yticklabels([exp_labels[i] for i in order_g], fontsize=8)
    ax_gs.set_xlabel("TOPSIS-G score  C_i")
    ax_gs.set_title("Ranking geometrico", fontweight="bold")
    ax_gs.set_xlim(0, 1.05)
    for bar, val in zip(bars, scores_g[order_g]):
        ax_gs.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", fontsize=8)
    ax_gs.invert_yaxis()

    # ── riga 1: TOPSIS-E (econometrico — esterno) ─────────────────────────
    ax_e  = fig.add_subplot(gs[1, :2])
    ax_es = fig.add_subplot(gs[1, 2])

    norm_e  = _normalise_for_plot(df, TOPSIS_ECON)
    cols_e  = list(TOPSIS_ECON.keys())
    nice_e  = ["Price η² ↑\n(w=0.40)", "Transition rate ↑*\n(w=0.30)",
               "Cramer stagione ↑\n(w=0.20)", "Entropy ↑\n(w=0.10)"]
    clrs_e  = ["#AA3377", "#CCBB44", "#66CCEE", "#228833"]
    width_e = 0.20

    for i, (col, label, color) in enumerate(zip(cols_e, nice_e, clrs_e)):
        ax_e.bar(x + i * width_e, norm_e[col].values, width_e,
                 label=label, color=color, alpha=0.85)

    ax_e.set_xticks(x + width_e * 1.5)
    ax_e.set_xticklabels(exp_labels, fontsize=8)
    ax_e.set_ylim(0, 1.2)
    ax_e.set_ylabel("Score normalizzato [0-1]")
    ax_e.set_title("TOPSIS-E  |  Criteri econometrici (esterni — indipendenti da UMAP)",
                   fontweight="bold")
    ax_e.legend(fontsize=7, loc="upper right")
    w_e = df["topsis_rank_econ"].values.argmin()
    ax_e.annotate("*", xy=(w_e + width_e * 1.5, 1.13), ha="center",
                  fontsize=16, color="#AA3377", fontweight="bold")

    scores_e = df["topsis_score_econ"].values
    order_e  = df["topsis_rank_econ"].values.argsort()
    bc_e = ["#AA3377" if r == 1 else "#AAAAAA" for r in df["topsis_rank_econ"].values]
    bars = ax_es.barh(range(len(df)), scores_e[order_e], color=[bc_e[i] for i in order_e], alpha=0.85)
    ax_es.set_yticks(range(len(df)))
    ax_es.set_yticklabels([exp_labels[i] for i in order_e], fontsize=8)
    ax_es.set_xlabel("TOPSIS-E score  C_i")
    ax_es.set_title("Ranking econometrico", fontweight="bold")
    ax_es.set_xlim(0, 1.05)
    for bar, val in zip(bars, scores_e[order_e]):
        ax_es.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", fontsize=8)
    ax_es.invert_yaxis()

    # verifica convergenza
    geom_winner = df.loc[df["topsis_rank_geom"] == 1, "exp"].values[0]
    econ_winner = df.loc[df["topsis_rank_econ"] == 1, "exp"].values[0]
    convergence = "CONVERGONO su Exp " + geom_winner if geom_winner == econ_winner \
                  else f"NON convergono: geom=Exp {geom_winner}, econ=Exp {econ_winner}"

    fig.suptitle(
        f"Confronto A/B/C/D — TOPSIS duale  (Hwang & Yoon 1981)\n"
        f"Ranking geometrico (G) vs econometrico (E)  |  {convergence}",
        fontsize=9, fontweight="bold")
    fig.savefig(OUT_DIR / "01_metrics.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  01_metrics.png", flush=True)


def _umap2d_for_vis(emb_path: Path) -> np.ndarray | None:
    """Ri-proietta embeddings a 2D SOLO per visualizzazione."""
    if emb_path is None or not emb_path.exists():
        return None
    df    = pd.read_parquet(emb_path)
    cols  = [c for c in df.columns if c.startswith("emb_")]
    if not cols:
        return None
    E     = df[cols].values.astype(np.float32)
    E_s   = StandardScaler().fit_transform(E)
    E_pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit_transform(E_s)
    E2    = UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
        verbose=False,
    ).fit_transform(E_pca)
    return E2


def plot_umap2d_grid(loaded: list[dict]) -> None:
    """02 - Griglia 2×2 UMAP 2D — solo visualizzazione, metriche da quality_report."""
    n    = len(loaded)
    ncol = 2
    nrow = (n + 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 5 * nrow))
    axes = np.array(axes).flatten()

    for ax, d in zip(axes, loaded):
        exp    = d["exp"]
        labels = d["labels"]
        K      = len(np.unique(labels))
        colors = regime_colors(K)

        print(f"    UMAP 2D vis exp_{exp}...", end=" ", flush=True)
        E2 = _umap2d_for_vis(d["_emb_path"])
        print("ok", flush=True)

        if E2 is not None:
            for k in range(K):
                mask = labels == k
                ax.scatter(E2[mask, 0], E2[mask, 1], c=[colors[k]],
                           s=4, alpha=0.5, label=f"R{k}")
        else:
            ax.text(0.5, 0.5, "embeddings non disponibili",
                    ha="center", va="center", transform=ax.transAxes)

        ax.set_title(
            f"Exp {exp} — {EXP_LABELS[exp]}\n"
            f"K={K}  sil={d['silhouette']:.3f}  DB={d['davies_bouldin']:.3f}"
            f"  η²={d['price_eta_squared']:.3f}",
            fontsize=8)
        ax.set_xlabel("UMAP 1", fontsize=7)
        ax.set_ylabel("UMAP 2", fontsize=7)
        if K <= 12:
            ax.legend(markerscale=2, fontsize=6, loc="best",
                      handlelength=0.8, borderpad=0.4)

    for ax in axes[len(loaded):]:
        ax.set_visible(False)

    fig.suptitle(
        "UMAP 2D — confronto esperimenti A/B/C/D\n"
        "(ri-proiezione a scopo visivo — metriche da step03f quality_report.json)",
        fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_umap2d_grid.png", dpi=DPI)
    plt.close(fig)
    print("  02_umap2d_grid.png", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Confronto Esperimenti A/B/C/D")
    print("═" * 65)

    # ── 1. Carica quality_report.json per ogni esperimento ────────────────
    print("\n  [1/3] Caricamento quality_report.json...", flush=True)
    loaded = []
    for exp in EXPERIMENTS:
        d = load_experiment(exp)
        if d is not None:
            loaded.append(d)
            feat = EXP_LABELS[exp]
            std_str = (f" +/-{d['price_eta_squared_std']:.3f}"
                       if d['price_eta_squared_std'] > 0 else "")
            print(f"    exp_{exp}: K={d['K']}  ({feat})"
                  f"  sil={d['silhouette']:.3f}"
                  f"  η²={d['price_eta_squared']:.3f}{std_str}"
                  f"  (media 5 seed)", flush=True)

    if len(loaded) < 2:
        print("\n  ⚠  Meno di 2 esperimenti disponibili — impossibile confrontare.")
        return

    # ── 2. TOPSIS duale ───────────────────────────────────────────────────
    print("\n  [2/3] TOPSIS ranking (geometrico + econometrico)...", flush=True)
    rows = []
    for d in loaded:
        rows.append({
            "exp"               : d["exp"],
            "feature"           : EXP_LABELS[d["exp"]],
            "K"                 : d["K"],
            "cov_type"          : d["cov"],
            # criteri geometrici (TOPSIS-G)
            "silhouette"        : d["silhouette"],
            "davies_bouldin"    : d["davies_bouldin"],
            "calinski_harabasz" : d["calinski_harabasz"],
            # criteri econometrici (TOPSIS-E)
            "price_eta_squared" : d["price_eta_squared"],
            "transition_rate"   : d["transition_rate"],
            "cramer_season"     : d["quality"].get("cramer_season", 0.0),
            "entropy_norm"      : d["entropy_norm"],
            # informativi
            "transition_entropy": d["transition_entropy"],
            "cramer_hour"       : d["cramer_hour"],
            "cramer_month"      : d["cramer_month"],
            "price_anova_p"     : d["price_anova_p"],
        })

    df = pd.DataFrame(rows)
    df = topsis(df, TOPSIS_GEOM, "_geom")
    df = topsis(df, TOPSIS_ECON, "_econ")

    geom_winner = df.loc[df["topsis_rank_geom"] == 1, "exp"].values[0]
    econ_winner = df.loc[df["topsis_rank_econ"] == 1, "exp"].values[0]

    print(f"\n  TOPSIS-G (geometrico):    Exp {geom_winner}", flush=True)
    print(f"  TOPSIS-E (econometrico):  Exp {econ_winner}", flush=True)
    if geom_winner == econ_winner:
        print(f"  ★  CONVERGONO: vincitore robusto = Exp {econ_winner}", flush=True)
    else:
        print(f"  ⚠  Ranking divergenti — analisi ulteriore necessaria", flush=True)

    # ── 3. Output ─────────────────────────────────────────────────────────
    print("\n  [3/3] Salvataggio output...", flush=True)

    df.to_csv(OUT_DIR / "comparison_table.csv", index=False)
    print("  comparison_table.csv", flush=True)

    # winner = econometrico (esterno, non circolare)
    winner_row = df.loc[df["topsis_rank_econ"] == 1].iloc[0]
    with open(OUT_DIR / "winner.json", "w") as f:
        json.dump({
            "winner_exp"            : winner_row["exp"],
            "feature"               : winner_row["feature"],
            "K"                     : int(winner_row["K"]),
            "topsis_geom_score"     : float(winner_row["topsis_score_geom"]),
            "topsis_geom_rank"      : int(winner_row["topsis_rank_geom"]),
            "topsis_econ_score"     : float(winner_row["topsis_score_econ"]),
            "topsis_econ_rank"      : int(winner_row["topsis_rank_econ"]),
            "rankings_converge"     : geom_winner == econ_winner,
            "price_eta_squared"     : float(winner_row["price_eta_squared"]),
            "transition_rate"       : float(winner_row["transition_rate"]),
            "cramer_season"         : float(winner_row["cramer_season"]),
            "silhouette"            : float(winner_row["silhouette"]),
            "davies_bouldin"        : float(winner_row["davies_bouldin"]),
            "weights_geom"          : {k: v[1] for k, v in TOPSIS_GEOM.items()},
            "weights_econ"          : {k: v[1] for k, v in TOPSIS_ECON.items()},
        }, f, indent=2)
    print("  winner.json", flush=True)

    plot_metrics(df)
    plot_umap2d_grid(loaded)

    # ── Report finale ─────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  RANKING FINALE")
    print("-" * 65)
    print(df[["exp", "feature", "K",
              "silhouette", "davies_bouldin",
              "price_eta_squared", "transition_rate", "cramer_season",
              "topsis_score_geom", "topsis_rank_geom",
              "topsis_score_econ", "topsis_rank_econ"]].to_string(index=False))
    print(f"\n  TOPSIS-G vincitore: Exp {geom_winner}")
    print(f"  TOPSIS-E vincitore: Exp {econ_winner}")
    if geom_winner == econ_winner:
        print(f"  ★  Ranking convergenti — Exp {econ_winner} e' il vincitore robusto")
    print(f"  Output → {OUT_DIR}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
