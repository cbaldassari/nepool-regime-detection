"""
step03_clustering.py
====================
NEPOOL Regime Detection — Step 03: GMM Clustering

Pipeline
--------
  embeddings.parquet (7680D)
    → PCA 64D   (client-side, CPU, ~5s)   preserva ~85% varianza
    → GMM grid search  (K × cov_type)     Ray, task CPU, ~2-5s ciascuno
    → best K via BIC                       criterio standard per GMM
    → fit finale GMM                       hard labels + posteriors

Nessun UMAP nel clustering.  PCA-2D usata solo per visualizzazione.

Perché GMM
----------
  • Modello probabilistico: P(regime_k | embedding_t) → soft assignments
  • BIC dà selezione del modello statisticamente fondata (penalizza complessità)
  • Concettualmente affine ai Regime Switching Models in econometria
  • Task CPU puri → Ray funziona su tutti i nodi senza dipendenze GPU
  • Nessun iperparametro difficile (niente min_dist, min_cluster_size, ecc.)

Output
------
  results/grid_results.csv        tutti i trial con BIC / AIC / silhouette
  results/best_params.json        miglior (K, cov_type)
  results/regimes.parquet         datetime + regime (hard assignment)
  results/posteriors.parquet      datetime + P(regime_k | x) per k=0..K-1
  results/step03/                 plot diagnostici
"""

import json, math, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  Configurazione
# ═══════════════════════════════════════════════════════════════════════════

RAY_ADDRESS   = "ray://datalab-rayclnt.unitus.it:10001"
PLOT_DIR      = Path(C.RESULTS_DIR) / "step03"
RANDOM_STATE  = 42

# PCA pre-riduzione per GMM (non per UMAP — non c'è più UMAP)
# 64D cattura ~80-85% della varianza; abbassa il costo computazionale del GMM
# e riduce il rischio di matrici di covarianza singolari ("full" in alta dim).
PCA_COMPONENTS = 64

# Grid search
K_GRID      = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
COV_TYPES   = ["full", "tied", "diag"]   # "spherical" raramente utile in finanza
N_INIT      = 5       # inizializzazioni random per stabilità
REG_COVAR   = 1e-4    # regolarizzazione covarianza (evita singolarità in alta dim)
MAX_ITER    = 300

# CPU per task Ray
CPUS_PER_TASK = 2

# True  → ricomincia da zero (cancella CSV, parquet, PNG step03)
# False → riprende dal checkpoint se esiste
FRESH_START = False


# ═══════════════════════════════════════════════════════════════════════════
#  Ray remote: fit GMM per una configurazione (K, cov_type)
# ═══════════════════════════════════════════════════════════════════════════

def _gmm_trial_local(E, K, cov_type, n_init, reg_covar, max_iter, seed):
    """Fit GMM (K, cov_type) su E.  Restituisce dict con metriche."""
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics  import silhouette_score

    result = {"K": K, "cov_type": cov_type,
              "bic": float("nan"), "aic": float("nan"),
              "silhouette": float("nan"), "converged": False,
              "n_iter": 0, "error": None}
    try:
        gmm = GaussianMixture(
            n_components   = K,
            covariance_type= cov_type,
            n_init         = n_init,
            reg_covar      = reg_covar,
            random_state   = seed,
            max_iter       = max_iter,
        )
        gmm.fit(E)
        labels = gmm.predict(E)
        result["bic"]       = float(gmm.bic(E))
        result["aic"]       = float(gmm.aic(E))
        result["converged"] = bool(gmm.converged_)
        result["n_iter"]    = int(gmm.n_iter_)

        n_unique = len(np.unique(labels))
        if n_unique >= 2:
            idx = (np.random.default_rng(seed).choice(len(E), min(3000, len(E)),
                                                        replace=False)
                   if len(E) > 3000 else np.arange(len(E)))
            result["silhouette"] = float(
                silhouette_score(E[idx], labels[idx], metric="euclidean"))
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ── Versione Ray remote ────────────────────────────────────────────────────
try:
    import ray as _ray_check   # noqa: F401  (test import senza init)

    import ray
    @ray.remote(num_cpus=CPUS_PER_TASK)
    def gmm_trial_remote(E_bytes, shape, dtype_str,
                         K, cov_type, n_init, reg_covar, max_iter, seed):
        import numpy as np
        E = np.frombuffer(E_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
        return _gmm_trial_local(E, K, cov_type, n_init, reg_covar, max_iter, seed)

except ImportError:
    pass   # Ray non installato sul client — si usa il fallback CPU locale


# ═══════════════════════════════════════════════════════════════════════════
#  Grid search
# ═══════════════════════════════════════════════════════════════════════════

def grid_search(E, ckpt_path):
    """
    Lancia un trial GMM per ogni (K, cov_type) via Ray.
    Checkpoint CSV: salva ogni trial completato → resume automatico.
    Restituisce DataFrame con tutti i risultati, ordinato per BIC.
    """
    import ray

    # Carica checkpoint
    done = {}
    if ckpt_path.exists():
        ck = pd.read_csv(ckpt_path)
        for _, row in ck.iterrows():
            done[(int(row["K"]), str(row["cov_type"]))] = row.to_dict()

    all_configs = [(K, ct) for K in K_GRID for ct in COV_TYPES]
    todo        = [(K, ct) for K, ct in all_configs if (K, ct) not in done]
    print(f"  Totale trial: {len(all_configs)}  |  "
          f"già completati: {len(done)}  |  da fare: {len(todo)}", flush=True)

    if not todo:
        results = list(done.values())
    else:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True,
                 log_to_driver=False)
        print(f"  Ray risorse: {ray.cluster_resources()}", flush=True)

        E_bytes    = E.tobytes()
        E_shape    = E.shape
        E_dtype    = str(E.dtype)
        futures    = {}
        for K, ct in todo:
            ref = gmm_trial_remote.remote(
                E_bytes, E_shape, E_dtype,
                K, ct, N_INIT, REG_COVAR, MAX_ITER, RANDOM_STATE)
            futures[ref] = (K, ct)

        results = list(done.values())
        n_done  = 0
        while futures:
            ready, _ = ray.wait(list(futures.keys()), num_returns=1, timeout=120)
            if not ready:
                print("  ⚠  Nessun risultato in 120s — continuo ad aspettare…",
                      flush=True)
                continue
            ref = ready[0]
            K, ct = futures.pop(ref)
            try:
                res = ray.get(ref)
                results.append(res)
                n_done += 1
                bic_str = f"BIC={res['bic']:.1f}" if not np.isnan(res['bic']) else "BIC=NaN"
                sil_str = f"sil={res['silhouette']:.3f}" if not np.isnan(res['silhouette']) else "sil=NaN"
                conv    = "✓" if res["converged"] else "✗conv"
                err_str = f"  ERR: {res['error']}" if res["error"] else ""
                print(f"  [{n_done:>3d}/{len(todo)}] K={K:>2d}  {ct:<8s}  "
                      f"{bic_str}  {sil_str}  {conv}{err_str}", flush=True)
            except Exception as exc:
                print(f"  ✗ K={K} {ct}: {exc}", flush=True)
                results.append({"K": K, "cov_type": ct,
                                 "bic": float("nan"), "aic": float("nan"),
                                 "silhouette": float("nan"),
                                 "converged": False, "error": str(exc)})

            # checkpoint incrementale
            pd.DataFrame(results).to_csv(ckpt_path, index=False)

        ray.shutdown()

    df = pd.DataFrame(results).sort_values("bic", na_position="last")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Fit finale
# ═══════════════════════════════════════════════════════════════════════════

def final_fit(E, best_K, best_cov):
    """
    Fit GMM finale con i migliori iperparametri.
    Restituisce labels (hard) e posteriors (soft, shape N×K).
    """
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(
        n_components   = best_K,
        covariance_type= best_cov,
        n_init         = N_INIT * 2,    # più init per il fit finale
        reg_covar      = REG_COVAR,
        random_state   = RANDOM_STATE,
        max_iter       = MAX_ITER,
    )
    gmm.fit(E)
    labels     = gmm.predict(E).astype(np.int16)
    posteriors = gmm.predict_proba(E).astype(np.float32)
    print(f"  Converged: {gmm.converged_}  |  "
          f"n_iter: {gmm.n_iter_}  |  "
          f"BIC: {gmm.bic(E):.1f}", flush=True)
    return labels, posteriors


# ═══════════════════════════════════════════════════════════════════════════
#  Visualizzazione 2D (PCA, non UMAP)
# ═══════════════════════════════════════════════════════════════════════════

def pca_2d(E_raw):
    """
    Proiezione 2D via PCA per visualizzazione.
    Non viene usata per il clustering — solo per i plot.
    E_raw: embedding normalizzato ma NON ancora ridotto a PCA_COMPONENTS.
    """
    from sklearn.decomposition import PCA as _PCA
    pca2 = _PCA(n_components=2, random_state=RANDOM_STATE)
    return pca2.fit_transform(E_raw)


# ═══════════════════════════════════════════════════════════════════════════
#  Plot
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(grid_df, Z2, labels, posteriors, timestamps):
    """
    01  BIC / AIC curve per K e per cov_type
    02  Silhouette curve
    03  PCA-2D scatter colorato per regime
    04  PCA-2D — 4 pannelli (regime / anno / stagione / ora)
    05  Timeline regime nel tempo
    06  Distribuzione dimensione regimi
    07  Heatmap posterior media per regime (quanto è "puro" ogni regime)
    """
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", font_scale=0.95)
    plt.rcParams.update({
        "figure.dpi"       : 150,
        "savefig.bbox"     : "tight",
        "savefig.facecolor": "white",
        "axes.spines.top"  : False,
        "axes.spines.right": False,
    })

    ts       = pd.to_datetime(timestamps)
    reg_ids  = sorted(np.unique(labels).tolist())
    n_reg    = len(reg_ids)
    palette  = sns.color_palette("tab10", n_colors=max(n_reg, 1))
    reg_color = {r: palette[i % len(palette)] for i, r in enumerate(reg_ids)}
    point_colors = [reg_color[lb] for lb in labels]

    valid_df = grid_df.dropna(subset=["bic"])

    # ════════════════════════════════════════════════════════════════════
    # 01  BIC / AIC curve
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ct in COV_TYPES:
        sub = valid_df[valid_df["cov_type"] == ct].sort_values("K")
        if sub.empty:
            continue
        axes[0].plot(sub["K"], sub["bic"], marker="o", label=ct, linewidth=1.5)
        axes[1].plot(sub["K"], sub["aic"], marker="o", label=ct, linewidth=1.5)
    for ax, metric in zip(axes, ["BIC", "AIC"]):
        ax.set_xlabel("K (numero regimi)"); ax.set_ylabel(metric)
        ax.set_title(f"{metric} per K e cov_type", fontweight="bold")
        ax.legend(fontsize=9); ax.set_xticks(K_GRID)
        ax.axvline(int(valid_df.loc[valid_df["bic"].idxmin(), "K"]),
                   color="red", ls="--", lw=0.8, alpha=0.6, label="best K (BIC)")
    fig.suptitle("Selezione modello GMM via BIC / AIC", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_bic_aic.png"); plt.close(fig)
    print("  [plot] 01_bic_aic.png")

    # ════════════════════════════════════════════════════════════════════
    # 02  Silhouette curve
    # ════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 4))
    for ct in COV_TYPES:
        sub = valid_df[valid_df["cov_type"] == ct].sort_values("K")
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["silhouette"], marker="s", label=ct, linewidth=1.5)
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette per K e cov_type", fontweight="bold")
    ax.legend(fontsize=9); ax.set_xticks(K_GRID)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_silhouette.png"); plt.close(fig)
    print("  [plot] 02_silhouette.png")

    # ════════════════════════════════════════════════════════════════════
    # 03  PCA-2D scatter
    # ════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in reg_ids:
        m   = labels == r
        col = reg_color[r]
        ax.scatter(Z2[m, 0], Z2[m, 1], c=[col],
                   s=6, alpha=0.55, linewidths=0, label=f"R{r}")
        cx, cy = Z2[m, 0].mean(), Z2[m, 1].mean()
        ax.text(cx, cy, str(r), fontsize=8, fontweight="bold",
                ha="center", va="center", color="white",
                bbox=dict(boxstyle="circle,pad=0.15", fc=col, ec="none", alpha=0.85),
                zorder=5)
    ax.set_title(f"PCA-2D  —  {n_reg} regimi GMM  "
                 f"(visualizzazione, non usata per clustering)",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    legend_els = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=reg_color[r], markersize=7,
                          label=f"R{r}  (n={int((labels==r).sum())})")
                  for r in reg_ids]
    ax.legend(handles=legend_els, fontsize=8,
              ncol=max(1, n_reg // 6), loc="best", framealpha=0.7)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_pca2d_regimes.png"); plt.close(fig)
    print("  [plot] 03_pca2d_regimes.png")

    # ════════════════════════════════════════════════════════════════════
    # 04  PCA-2D — 4 pannelli contestuali
    # ════════════════════════════════════════════════════════════════════
    fig, axes4 = plt.subplots(2, 2, figsize=(14, 11))
    # 04a regime
    ax = axes4[0, 0]
    for r in reg_ids:
        m = labels == r
        ax.scatter(Z2[m, 0], Z2[m, 1], c=[reg_color[r]],
                   s=4, alpha=0.5, linewidths=0)
        cx, cy = Z2[m, 0].mean(), Z2[m, 1].mean()
        ax.text(cx, cy, str(r), fontsize=7, fontweight="bold",
                ha="center", va="center", color="white",
                bbox=dict(boxstyle="circle,pad=0.12", fc=reg_color[r],
                          ec="none", alpha=0.85), zorder=4)
    ax.set_title("Regime", fontweight="bold")
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    # 04b anno
    ax  = axes4[0, 1]
    sc  = ax.scatter(Z2[:, 0], Z2[:, 1], c=ts.dt.year.values,
                     cmap="plasma", s=4, alpha=0.5, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Year", shrink=0.85)
    ax.set_title("Per anno", fontweight="bold")
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    # 04c stagione
    ax    = axes4[1, 0]
    month = ts.dt.month.values
    sl    = np.where(np.isin(month, [3,4,5]),  0,
            np.where(np.isin(month, [6,7,8]),  1,
            np.where(np.isin(month, [9,10,11]),2, 3)))
    snames = ["Spring","Summer","Fall","Winter"]
    scolors= ["#2ecc71","#e74c3c","#e67e22","#3498db"]
    for sid, sn, sc_ in zip(range(4), snames, scolors):
        m = sl == sid
        ax.scatter(Z2[m,0], Z2[m,1], c=[sc_], s=4, alpha=0.45,
                   linewidths=0, label=sn)
    ax.legend(markerscale=3, fontsize=8, framealpha=0.7)
    ax.set_title("Per stagione", fontweight="bold")
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    # 04d ora
    ax = axes4[1, 1]
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=ts.dt.hour.values,
                    cmap="twilight_shifted", s=4, alpha=0.5,
                    linewidths=0, vmin=0, vmax=23)
    cb = plt.colorbar(sc, ax=ax, label="Hour of day", shrink=0.85)
    cb.set_ticks([0, 6, 12, 18, 23])
    ax.set_title("Per ora del giorno", fontweight="bold")
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")

    fig.suptitle("PCA-2D — pannelli contestuali", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_pca2d_context.png"); plt.close(fig)
    print("  [plot] 04_pca2d_context.png")

    # ════════════════════════════════════════════════════════════════════
    # 05  Timeline
    # ════════════════════════════════════════════════════════════════════
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 5),
        gridspec_kw={"height_ratios": [1, 2.5]}
    )
    years_u  = sorted(ts.dt.year.unique())
    bottoms  = np.zeros(len(years_u))
    for r in reg_ids:
        fracs = []
        for yr in years_u:
            m_yr  = ts.dt.year == yr
            total = m_yr.sum()
            fracs.append((m_yr & (labels == r)).sum() / total if total else 0)
        ax_top.bar(years_u, fracs, bottom=bottoms,
                   color=reg_color[r], width=0.8, label=f"R{r}")
        bottoms += np.array(fracs)
    ax_top.set_ylabel("Frazione", fontsize=8)
    ax_top.set_title("Quota annua per regime", fontsize=9, fontweight="bold")
    ax_top.legend(ncol=n_reg, fontsize=7, loc="upper left", framealpha=0.6)
    ax_top.set_xlim(years_u[0] - 0.5, years_u[-1] + 0.5)

    for r in reg_ids:
        m = labels == r
        ax_bot.scatter(ts[m], np.full(m.sum(), r),
                       c=[reg_color[r]], s=3, alpha=0.5, linewidths=0)
    ax_bot.set_yticks(reg_ids)
    ax_bot.set_yticklabels([f"R{r}" for r in reg_ids], fontsize=8)
    ax_bot.set_xlabel("Data"); ax_bot.set_ylabel("Regime")
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bot.xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_timeline.png"); plt.close(fig)
    print("  [plot] 05_timeline.png")

    # ════════════════════════════════════════════════════════════════════
    # 06  Dimensione regimi
    # ════════════════════════════════════════════════════════════════════
    u, cnt = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(max(6, n_reg * 0.9), 4))
    bars = ax.bar([f"R{r}" for r in u], cnt,
                  color=[reg_color[r] for r in u],
                  edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)
    ax.set_title("Distribuzione dimensione regimi", fontweight="bold", fontsize=11)
    ax.set_xlabel("Regime"); ax.set_ylabel("N finestre")
    ax.set_ylim(0, cnt.max() * 1.12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_distribution.png"); plt.close(fig)
    print("  [plot] 06_distribution.png")

    # ════════════════════════════════════════════════════════════════════
    # 07  Heatmap posteriors media: quanto è "puro" ogni regime
    # ════════════════════════════════════════════════════════════════════
    # Per ogni regime r (riga): vettore medio delle probabilità posterior
    # di tutte le finestre assegnate a r.
    # Diagonale alta = regimi ben separati.  Off-diagonale alta = overlap.
    post_mean = np.zeros((n_reg, n_reg))
    for i, r in enumerate(reg_ids):
        m = labels == r
        if m.sum() > 0:
            post_mean[i] = posteriors[m].mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(5, n_reg * 0.7), max(4, n_reg * 0.65)))
    im = ax.imshow(post_mean, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_reg)); ax.set_xticklabels([f"R{r}" for r in reg_ids])
    ax.set_yticks(range(n_reg)); ax.set_yticklabels([f"R{r}" for r in reg_ids])
    ax.set_xlabel("Componente GMM")
    ax.set_ylabel("Regime assegnato (hard)")
    ax.set_title("Posterior medio per regime  (diagonale = purezza)",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, label="P(k | x)", shrink=0.85)
    for i in range(n_reg):
        for j in range(n_reg):
            v = post_mean[i, j]
            if v > 0.04:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if v < 0.6 else "white")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_posterior_heatmap.png"); plt.close(fig)
    print("  [plot] 07_posterior_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    grid_path  = Path(C.RESULTS_DIR) / "grid_results.csv"
    ckpt_path  = Path(C.RESULTS_DIR) / "grid_checkpoint.csv"
    best_path  = Path(C.RESULTS_DIR) / "best_params.json"
    regime_path= Path(C.RESULTS_DIR) / "regimes.parquet"
    post_path  = Path(C.RESULTS_DIR) / "posteriors.parquet"
    emb_path   = Path(C.RESULTS_DIR) / "embeddings.parquet"

    # ── Fresh start ──────────────────────────────────────────────────────
    if FRESH_START:
        for p in [grid_path, ckpt_path, best_path, regime_path, post_path]:
            if p.exists():
                p.unlink()
                print(f"  [fresh] rimosso {p.name}")
        if PLOT_DIR.exists():
            for png in PLOT_DIR.glob("*.png"):
                png.unlink()
                print(f"  [fresh] rimosso {png.name}")

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 03: GMM Clustering")
    print("═" * 65)
    print(f"  Ray       : {RAY_ADDRESS}")
    print(f"  Pipeline  : 7680D → PCA({PCA_COMPONENTS}D) → GMM(K×cov_type)")
    print(f"  Grid      : K={K_GRID}  ×  cov_type={COV_TYPES}")
    print(f"  Totale trial: {len(K_GRID) * len(COV_TYPES)}")
    print("─" * 65)

    # ── 1. Carica embeddings ─────────────────────────────────────────────
    print(f"\n[1/5] Loading embeddings ({emb_path.name})…")
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} non trovato — esegui prima step02_embeddings.py")
    emb_df     = pd.read_parquet(emb_path)
    timestamps = pd.to_datetime(emb_df["datetime"])
    E_raw      = emb_df[[c for c in emb_df.columns
                          if c.startswith("emb_")]].values.astype(np.float32)
    print(f"  {len(E_raw):,} finestre × {E_raw.shape[1]} dims", flush=True)

    # ── 2. PCA pre-riduzione ─────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition  import PCA as _PCA

    print(f"\n[2/5] PCA {E_raw.shape[1]}D → {PCA_COMPONENTS}D…", flush=True)
    E_scaled = StandardScaler().fit_transform(E_raw)
    pca      = _PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE,
                    svd_solver="randomized")
    E        = pca.fit_transform(E_scaled).astype(np.float32)
    var_exp  = pca.explained_variance_ratio_.sum()
    print(f"  ✓ PCA → {E.shape[1]}D  (varianza spiegata: {var_exp:.1%})", flush=True)

    # PCA 2D per visualizzazione
    pca2     = _PCA(n_components=2, random_state=RANDOM_STATE)
    Z2       = pca2.fit_transform(E_scaled).astype(np.float32)
    print(f"  ✓ PCA-2D per plot  (varianza: {pca2.explained_variance_ratio_.sum():.1%})",
          flush=True)

    # ── 3. Grid search ───────────────────────────────────────────────────
    if grid_path.exists():
        print(f"\n[3/5] Grid results già presenti — carico da {grid_path.name}")
        grid_df = pd.read_csv(grid_path)
        print(f"  {len(grid_df)} trial  |  "
              f"{grid_df['bic'].notna().sum()} validi (BIC non-NaN)")
    else:
        n_done_ck = len(pd.read_csv(ckpt_path)) if ckpt_path.exists() else 0
        print(f"\n[3/5] Grid search GMM  ({len(K_GRID)}×{len(COV_TYPES)} = "
              f"{len(K_GRID)*len(COV_TYPES)} trial)…")
        if n_done_ck:
            print(f"  Resume: {n_done_ck} trial già nel checkpoint.")
        grid_df = grid_search(E, ckpt_path)
        grid_df.to_csv(grid_path, index=False)
        print(f"  Salvato → {grid_path}")

    print(f"\n  Top 5 per BIC:")
    top5 = grid_df.dropna(subset=["bic"]).head(5)
    print(top5[["K","cov_type","bic","aic","silhouette","converged"]].to_string(index=False))

    # ── 4. Best params ───────────────────────────────────────────────────
    print(f"\n[4/5] Selezione best params (min BIC)…")
    valid = grid_df.dropna(subset=["bic"])
    if valid.empty:
        raise RuntimeError("Nessun trial valido (tutti NaN).")
    best     = valid.iloc[0].to_dict()
    best_K   = int(best["K"])
    best_cov = str(best["cov_type"])
    print(f"  Best: K={best_K}  cov_type={best_cov}  "
          f"BIC={best['bic']:.1f}  sil={best['silhouette']:.3f}")
    with open(best_path, "w") as f:
        json.dump({k: (int(v)   if isinstance(v, (np.integer,  int))   else
                       float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in best.items()}, f, indent=2)
    print(f"  Salvato → {best_path}")

    # ── 5. Fit finale ────────────────────────────────────────────────────
    if regime_path.exists() and post_path.exists():
        print(f"\n[5/5] Output finali già presenti — carico da disco.")
        regime_df = pd.read_parquet(regime_path)
        labels    = regime_df["regime"].values
        post_df   = pd.read_parquet(post_path)
        posteriors= post_df[[c for c in post_df.columns
                              if c.startswith("p_")]].values
    else:
        print(f"\n[5/5] Fit finale GMM  K={best_K}  cov={best_cov}…")
        labels, posteriors = final_fit(E, best_K, best_cov)

        pd.DataFrame({"datetime": timestamps,
                      "regime": labels}).to_parquet(regime_path, index=False)
        post_cols = {f"p_{k}": posteriors[:, k] for k in range(best_K)}
        pd.DataFrame({"datetime": timestamps, **post_cols}
                     ).to_parquet(post_path, index=False)
        print(f"  Salvato → {regime_path}")
        print(f"  Salvato → {post_path}")

    # ── Plot ─────────────────────────────────────────────────────────────
    print(f"\n  Plot → {PLOT_DIR}")
    make_plots(grid_df, Z2, labels, posteriors, timestamps)

    # ── Report ────────────────────────────────────────────────────────────
    n_reg   = len(np.unique(labels))
    elapsed = time.time() - t0
    print("\n" + "─" * 65)
    print("  RISULTATI")
    print("─" * 65)
    print(f"  Regimi trovati : {n_reg}")
    print(f"  Best K         : {best_K}")
    print(f"  Best cov_type  : {best_cov}")
    print(f"  Best BIC       : {best['bic']:.1f}")
    print(f"  Best silhouette: {best['silhouette']:.4f}")
    print(f"  Tempo totale   : {elapsed:.1f}s")
    sizes = pd.Series(labels).value_counts().sort_index()
    print(f"\n  Regime sizes:")
    for r, sz in sizes.items():
        print(f"    regime {r:2d} : {sz:5,} windows ({sz/len(labels):.1%})")
    print("\n  → Step 03d: interpretazione regimi")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
