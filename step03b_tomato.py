"""
step03b_tomato.py
=================
NEPOOL Regime Detection — Step 03b: ToMATo Clustering (alternativa a HDBSCAN)

Differenze rispetto a step03:
  • Nessuna PCA, nessun UMAP, nessun HDBSCAN
  • ToMATo (Topological Mode Analysis Tool) direttamente sugli embedding 8448D
  • Basato su omologia persistente — trova i regimi come "bacini di attrazione"
    della funzione di densità stimata via kNN (Distance to Measure)
  • CPU only (gudhi non ha backend GPU)
  • Grid search locale: k × n_clusters
  • Score: silhouette (DBCV non disponibile per ToMATo)

Dipendenze:
  pip install gudhi scikit-learn

Input  : results/embeddings.parquet
Output : results/step03b/tomato_results.csv
         results/step03b/best_params.json
         results/step03b/regimes.parquet
Plots  : results/step03b/
"""

import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  Configurazione
# ═══════════════════════════════════════════════════════════════════════════

RANDOM_STATE  = 42
OUT_DIR       = Path(C.RESULTS_DIR) / "step03b"

# Grid search: parametri ToMATo
# k          — vicini per la stima di densità (Distance to Measure)
# n_clusters — numero di regimi da forzare (taglia il dendrogramma di persistenza)
SEARCH_SPACE = {
    "k"          : [5, 10, 15, 20, 30, 50],
    "n_clusters" : [2, 3, 4, 5, 6, 7, 8, 10],
}

# ── Modalità sviluppo ─────────────────────────────────────────────────────
FRESH_START = True   # True → riparte da zero, False → riprende dal CSV

# ═══════════════════════════════════════════════════════════════════════════
#  Singolo trial ToMATo
# ═══════════════════════════════════════════════════════════════════════════

def run_tomato(E: np.ndarray, k: int, n_clusters: int) -> dict:
    """
    Esegue ToMATo con i parametri dati e restituisce un dizionario di risultati.

    ToMATo costruisce:
      1. Grafo kNN sull'input (8448D, distanza euclidea)
      2. Stima densità con Distance to Measure (DTM)
      3. Merge gerarchico dei picchi di densità
      4. Taglia il dendrogramma a n_clusters cluster

    Score: silhouette sui punti non-rumore (ToMATo non produce -1/rumore
    nativamente — tutti i punti vengono assegnati ad un cluster).
    """
    try:
        from gudhi.clustering.tomato import Tomato
        from sklearn.metrics import silhouette_score

        t = Tomato(
            graph_type   = "knn",
            density_type = "DTM",   # Distance to Measure — robusto in alta dim.
            k            = k,
            n_clusters   = n_clusters,
        )
        t.fit(E)
        labels = t.labels_

        n_found    = len(set(labels))
        noise_frac = 0.0   # ToMATo assegna tutti i punti

        if n_found >= 2:
            score = float(silhouette_score(E, labels, metric="euclidean",
                                           sample_size=min(3000, len(E)),
                                           random_state=RANDOM_STATE))
        else:
            score = float("nan")

        # Persistenza massima nel diagramma (misura di separazione topologica)
        if hasattr(t, "diagram_") and t.diagram_ is not None and len(t.diagram_) > 0:
            persistence = float(t.diagram_[-1]) if len(t.diagram_) >= n_clusters else float("nan")
        else:
            persistence = float("nan")

        return {
            "k"           : k,
            "n_clusters"  : n_clusters,
            "n_found"     : n_found,
            "noise_frac"  : noise_frac,
            "silhouette"  : score,
            "persistence" : persistence,
            "status"      : "ok",
        }

    except Exception as e:
        return {
            "k"           : k,
            "n_clusters"  : n_clusters,
            "n_found"     : 0,
            "noise_frac"  : 1.0,
            "silhouette"  : float("nan"),
            "persistence" : float("nan"),
            "status"      : f"error: {str(e)[:80]}",
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Grid search (locale, CPU)
# ═══════════════════════════════════════════════════════════════════════════

def grid_search(E: np.ndarray, ckpt_path: Path) -> pd.DataFrame:
    import itertools

    all_combos = list(itertools.product(
        SEARCH_SPACE["k"],
        SEARCH_SPACE["n_clusters"],
    ))
    N_TRIALS = len(all_combos)

    # Resume
    done_keys = set()
    results   = []
    if ckpt_path.exists():
        ckpt_df = pd.read_csv(ckpt_path)
        for _, row in ckpt_df.iterrows():
            results.append(row.to_dict())
            done_keys.add((int(row["k"]), int(row["n_clusters"])))

    todo = [(k, nc) for k, nc in all_combos if (k, nc) not in done_keys]
    n_done = len(done_keys)

    print(f"  Grid: {N_TRIALS} trial  "
          f"({'da zero' if n_done == 0 else f'{n_done} già fatti → {len(todo)} rimasti'})")

    if not todo:
        print("  ✓ Tutti i trial già completati.")
        return pd.DataFrame(results).sort_values("silhouette", ascending=False,
                                                  na_position="last").reset_index(drop=True)

    CSV_COLS     = ["k", "n_clusters", "n_found", "noise_frac",
                    "silhouette", "persistence", "status"]
    write_header = not ckpt_path.exists()
    best_sil     = max(
        (r["silhouette"] for r in results
         if r.get("silhouette") is not None and not np.isnan(float(r["silhouette"]))),
        default=float("-inf"),
    )

    print(f"\n  {'#':>4}  {'k':>4}  {'nc':>3}  {'sil':>7}  {'found':>5}  note")
    print(f"  {'─'*4}  {'─'*4}  {'─'*3}  {'─'*7}  {'─'*5}  {'─'*6}")

    t_start = time.time()

    for i, (k, nc) in enumerate(todo):
        r = run_tomato(E, k, nc)

        row_df = pd.DataFrame([{col: r.get(col) for col in CSV_COLS}])
        row_df.to_csv(ckpt_path, mode="a", header=write_header, index=False)
        write_header = False
        results.append(r)

        sil     = r.get("silhouette")
        val     = float(sil) if sil is not None and not np.isnan(float(sil)) \
                  else float("-inf")
        is_best = np.isfinite(val) and val > best_sil
        if is_best:
            best_sil = val

        sc_str  = f"{val:.4f}" if np.isfinite(val) else "   nan"
        elapsed = time.time() - t_start
        rate    = (i + 1) / elapsed if elapsed > 0 else 0
        eta     = (len(todo) - i - 1) / rate if rate > 0 else 0
        eta_str = f"{int(eta//60)}m{int(eta%60):02d}s"

        status_str = "" if r["status"] == "ok" else f"  ⚠ {r['status']}"
        print(
            f"  {n_done+i+1:>4}/{N_TRIALS}  "
            f"k={k:>3}  nc={nc:>2}  "
            f"sil={sc_str}  found={int(r['n_found']):>3}"
            f"{'  ← best!' if is_best else ''}"
            f"{status_str}",
            flush=True,
        )

        if (n_done + i + 1) % 10 == 0:
            print(f"  ── {n_done+i+1}/{N_TRIALS} trial  ETA {eta_str} ──", flush=True)

    df = pd.DataFrame(results)
    return df.sort_values("silhouette", ascending=False,
                          na_position="last").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Plot
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(grid_df: pd.DataFrame, labels: np.ndarray, timestamps):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    n_reg   = len(set(labels))
    palette = sns.color_palette("tab10", n_colors=max(n_reg, 1))

    # 01 heatmap silhouette
    k_vals  = sorted(grid_df["k"].dropna().unique())
    nc_vals = sorted(grid_df["n_clusters"].dropna().unique())
    pivot   = (grid_df.groupby(["k", "n_clusters"])["silhouette"]
                       .max()
                       .unstack("n_clusters")
                       .reindex(index=k_vals, columns=nc_vals))

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlGn",
                linewidths=0.3)
    ax.set_title("ToMATo grid search — silhouette score", fontweight="bold")
    ax.set_xlabel("n_clusters");  ax.set_ylabel("k")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_grid_heatmap.png");  plt.close(fig)
    print("  [plot] 01_grid_heatmap.png")

    # 02 timeline regimi
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.scatter(pd.to_datetime(timestamps), labels, c=labels,
               cmap="tab10", s=1, alpha=0.6)
    ax.set_title("Regime label over time — ToMATo", fontweight="bold")
    ax.set_xlabel("Date");  ax.set_ylabel("Regime")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_timeline.png");  plt.close(fig)
    print("  [plot] 02_timeline.png")

    # 03 distribuzione
    u, c = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(max(6, len(u)), 4))
    bars = ax.bar([str(r) for r in u], c,
                  color=[palette[r % len(palette)] for r in u])
    ax.bar_label(bars, fmt="%d", fontsize=8)
    ax.set_title("Regime size distribution — ToMATo", fontweight="bold")
    ax.set_xlabel("Regime");  ax.set_ylabel("N windows")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_distribution.png");  plt.close(fig)
    print("  [plot] 03_distribution.png")

    # 04 persistenza vs silhouette (scatter diagnostico)
    valid = grid_df.dropna(subset=["silhouette"])
    if not valid.empty and "persistence" in valid.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(valid["persistence"], valid["silhouette"],
                        c=valid["n_clusters"], cmap="tab10", s=40, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="n_clusters")
        ax.set_xlabel("Max persistence");  ax.set_ylabel("Silhouette")
        ax.set_title("ToMATo: persistenza topologica vs silhouette", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "04_persistence_vs_silhouette.png");  plt.close(fig)
        print("  [plot] 04_persistence_vs_silhouette.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path   = OUT_DIR / "tomato_checkpoint.csv"
    results_path = OUT_DIR / "tomato_results.csv"
    best_path   = OUT_DIR / "best_params.json"
    regime_path = OUT_DIR / "regimes.parquet"

    if FRESH_START:
        for p in [ckpt_path, results_path, best_path, regime_path]:
            if p.exists():
                p.unlink()
                print(f"  [fresh] rimosso {p.name}")

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 03b: ToMATo Clustering")
    print("═" * 65)
    print(f"  Input   : results/embeddings.parquet  (8448D raw)")
    print(f"  Metodo  : ToMATo (gudhi) — nessuna PCA/UMAP/HDBSCAN")
    print(f"  Score   : silhouette (CPU sklearn)")
    print(f"  Grid    : k={SEARCH_SPACE['k']}  ×  nc={SEARCH_SPACE['n_clusters']}")
    print(f"  Trials  : {len(SEARCH_SPACE['k']) * len(SEARCH_SPACE['n_clusters'])}")
    print("─" * 65)

    # ── 1. Embeddings (raw, nessuna riduzione) ───────────────────────────
    print("\n[1/4] Loading embeddings (raw 8448D)...")
    emb_df     = pd.read_parquet(Path(C.RESULTS_DIR) / "embeddings.parquet")
    timestamps = pd.to_datetime(emb_df["datetime"])
    E          = emb_df[[c for c in emb_df.columns
                          if c.startswith("emb_")]].values.astype(np.float32)
    print(f"  {len(E):,} windows × {E.shape[1]:,} dims")

    # Verifica gudhi
    try:
        from gudhi.clustering.tomato import Tomato
        print("  gudhi OK")
    except ImportError:
        print("  ✗ gudhi non installato — esegui: pip install gudhi")
        return

    # ── 2. Grid search ───────────────────────────────────────────────────
    if results_path.exists():
        print(f"\n[2/4] Risultati già presenti — carico da {results_path}")
        grid_df = pd.read_csv(results_path)
        print(f"  {len(grid_df)} trial  |  {grid_df['silhouette'].notna().sum()} validi")
    else:
        print(f"\n[2/4] Grid search ToMATo ({E.shape[1]}D)...")
        grid_df = grid_search(E, ckpt_path)
        grid_df.to_csv(results_path, index=False)
        print(f"\n  Salvato → {results_path}")

    print(f"\n  Top 5:")
    print(grid_df.dropna(subset=["silhouette"]).head(5).to_string(index=False))

    # ── 3. Best params + regime labels ───────────────────────────────────
    print(f"\n[3/4] Best params + final clustering...")
    valid = grid_df.dropna(subset=["silhouette"])
    if valid.empty:
        raise RuntimeError("Nessun trial valido (tutti NaN).")
    best = valid.iloc[0].to_dict()
    with open(best_path, "w") as f:
        json.dump({k: (int(v)   if isinstance(v, (np.integer,  int))   else
                       float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in best.items()}, f, indent=2)
    print(f"  Best: k={int(best['k'])}  n_clusters={int(best['n_clusters'])}"
          f"  silhouette={float(best['silhouette']):.4f}")

    # Final run con best params
    from gudhi.clustering.tomato import Tomato
    print("  Final ToMATo fit...", flush=True)
    t_final = Tomato(graph_type="knn", density_type="DTM",
                     k=int(best["k"]), n_clusters=int(best["n_clusters"]))
    t_final.fit(E)
    labels = t_final.labels_.astype(np.int16)

    pd.DataFrame({"datetime": timestamps,
                  "regime": labels}
                 ).to_parquet(regime_path, index=False)
    print(f"  Salvato → {regime_path}")

    # ── 4. Plot ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Plot...")
    make_plots(grid_df, labels, timestamps)

    # ── Report ────────────────────────────────────────────────────────────
    n_reg   = len(set(labels))
    elapsed = time.time() - t0

    print("\n" + "─" * 65)
    print("  RISULTATI ToMATo")
    print("─" * 65)
    print(f"  Regimi trovati    : {n_reg}")
    print(f"  Best silhouette   : {float(best['silhouette']):.4f}")
    print(f"  Best k            : {int(best['k'])}")
    print(f"  Best n_clusters   : {int(best['n_clusters'])}")
    print(f"  Tempo totale      : {elapsed:.1f}s")
    sizes = pd.Series(labels).value_counts().sort_index()
    print(f"\n  Regime sizes:")
    for r, sz in sizes.items():
        print(f"    regime {r:2d} : {sz:5,} windows ({sz/len(labels):.1%})")
    print("\n  → Confronta con step03 (HDBSCAN) in results/step03b/")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
