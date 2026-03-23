"""
step02_justification.py
=======================
Giustificazione empirica di Chronos-2 rispetto a features manuali.

Domanda
-------
  "Perché un modello da 120M parametri invece di features calcolate a mano?"

Risposta
--------
  Stessa pipeline (PCA → UMAP → GMM, stesso K) applicata a:
    A) Chronos-2 embeddings  (Exp F, lmp_clipped)
    B) Features manuali      (12 statistiche per finestra)

  Se Chronos-2 vince su silhouette, η², ANOVA-F, la scelta è giustificata.

Features manuali per finestra (720h)
-------------------------------------
  mean, std, skewness, kurtosis, p10, p90,
  acf1, acf24, acf168, spike_rate, range_norm, hurst

Output  (results/step02_justification/)
----------------------------------------
  comparison_table.csv      — metriche a confronto
  01_comparison_bar.png     — barchart silhouette / η² / DB
  02_umap_chronos.png       — UMAP colorato per regime (Chronos-2)
  02_umap_manual.png        — UMAP colorato per regime (features manuali)
  03_tsne_comparison.png    — t-SNE side-by-side

Uso
---
  python diagnostics/step02_justification.py
  python diagnostics/step02_justification.py --exp F   # default
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
import config as C

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="F",
                    choices=["A","B","C","D","E","F","G","H","I"])
args = parser.parse_args()
EXP = args.exp

RESULTS_DIR = ROOT / C.RESULTS_DIR
EMB_PATH    = RESULTS_DIR / f"step02/exp_{EXP}/embeddings.parquet"
LABELS_PATH = RESULTS_DIR / f"step03f/exp_{EXP}/labels_best.parquet"
PREP_PATH   = RESULTS_DIR / "preprocessed.parquet"
OUT_DIR     = RESULTS_DIR / "step02_justification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_H = 720
STRIDE_H  = 6
SEED      = C.RANDOM_STATE
N_JOBS    = -1
DPI       = 150

# =============================================================================
# 1. Carica embeddings e labels Chronos-2
# =============================================================================

print("\n" + "═"*65)
print(f"  Step02 Justification — Exp {EXP}")
print("═"*65)

if not EMB_PATH.exists():
    print(f"\n  ERRORE: embeddings non trovati in {EMB_PATH}")
    print("  Esegui prima step02_embeddings.py --exp F")
    sys.exit(1)

print(f"\n[1] Carico embeddings Exp {EXP}...")
emb_df  = pd.read_parquet(EMB_PATH)
emb_df["datetime"] = pd.to_datetime(emb_df["datetime"])
emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
E_chronos = emb_df[emb_cols].values.astype(np.float32)
print(f"    embedding shape: {E_chronos.shape}")

# K dal clustering già fatto (se disponibile), altrimenti BIC
K_best = None
if LABELS_PATH.exists():
    lbl_df  = pd.read_parquet(LABELS_PATH)
    K_best  = int(lbl_df["label"].nunique())
    print(f"    K dal clustering Exp {EXP}: {K_best}")
else:
    print(f"    labels_best.parquet non trovato — K sarà selezionato via BIC")

# =============================================================================
# 2. Calcola features manuali sulle stesse finestre
# =============================================================================

print("\n[2] Calcolo features manuali...")

prep = pd.read_parquet(PREP_PATH)
prep["datetime"] = pd.to_datetime(prep["datetime"])

# colonna da usare: stessa trasformazione dell'esperimento
_EXP_COL = {
    "A": "log_return", "B": "arcsinh_lmp", "C": "mstl_resid_lr",
    "D": "mstl_resid_arcsinh", "E": "log_lmp_shifted",
    "F": "lmp_clipped", "G": "mstl_resid_log",
    "H": "quantile_transform", "I": "rolling_zscore_24h",
}
col = _EXP_COL[EXP]
if col not in prep.columns:
    print(f"  ERRORE: colonna '{col}' non trovata in preprocessed.parquet")
    sys.exit(1)

series = prep.set_index("datetime")[col].sort_index().dropna()
times  = emb_df["datetime"].values   # timestamp finali delle finestre

def hurst_rs(x: np.ndarray) -> float:
    """Stima R/S dell'esponente di Hurst."""
    n = len(x)
    if n < 20:
        return 0.5
    lags = [n // 8, n // 4, n // 2, n]
    rs_vals = []
    for lag in lags:
        sub = x[:lag]
        mean_ = sub.mean()
        devs  = np.cumsum(sub - mean_)
        R     = devs.max() - devs.min()
        S     = sub.std()
        if S > 0:
            rs_vals.append(R / S)
    if not rs_vals:
        return 0.5
    return float(np.log(np.mean(rs_vals)) / np.log(np.mean(lags)))


def acf_lag(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return 0.0
    c0 = np.var(x)
    if c0 == 0:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


feats = []
for t in times:
    t_end   = pd.Timestamp(t)
    t_start = t_end - pd.Timedelta(hours=CONTEXT_H - 1)
    w = series.loc[t_start:t_end].values
    if len(w) < CONTEXT_H // 2:
        feats.append([np.nan] * 12)
        continue
    std_  = w.std()
    feats.append([
        w.mean(),
        std_,
        float(pd.Series(w).skew()),
        float(pd.Series(w).kurtosis()),
        float(np.percentile(w, 10)),
        float(np.percentile(w, 90)),
        acf_lag(w, 1),
        acf_lag(w, 24),
        acf_lag(w, 168),
        float(np.mean(np.abs(w - w.mean()) > 2 * std_)) if std_ > 0 else 0.0,
        float((w.max() - w.min()) / std_) if std_ > 0 else 0.0,
        hurst_rs(w),
    ])

feat_names = ["mean","std","skew","kurt","p10","p90",
              "acf1","acf24","acf168","spike_rate","range_norm","hurst"]
F_manual = np.array(feats, dtype=np.float32)

# rimuovi righe con NaN
valid = ~np.isnan(F_manual).any(axis=1)
E_chronos_v = E_chronos[valid]
F_manual_v  = F_manual[valid]
times_v     = times[valid]
print(f"    finestre valide: {valid.sum()} / {len(valid)}")

# =============================================================================
# 3. Pipeline PCA → UMAP → GMM su entrambi
# =============================================================================

def run_pipeline(X: np.ndarray, name: str, K: int | None) -> dict:
    """PCA(20D) → UMAP(2D) → GMM → metriche."""
    print(f"\n[3] Pipeline {name}...")

    # PCA
    n_pca = min(20, X.shape[1], X.shape[0] - 1)
    pca   = PCA(n_components=n_pca, random_state=SEED)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    var20 = pca.explained_variance_ratio_.sum()
    print(f"    PCA {n_pca}D → {var20*100:.1f}% varianza")

    # UMAP
    try:
        from umap import UMAP
        X_umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.0,
                      metric="euclidean", random_state=SEED,
                      n_jobs=N_JOBS).fit_transform(X_pca)
    except ImportError:
        from sklearn.manifold import TSNE
        X_umap = TSNE(n_components=2, random_state=SEED,
                      n_jobs=N_JOBS).fit_transform(X_pca)
        print("    UMAP non disponibile — uso t-SNE")

    # GMM: usa K noto o cerca con BIC
    if K is not None:
        gmm    = GaussianMixture(n_components=K, covariance_type="full",
                                  n_init=5, random_state=SEED)
        gmm.fit(X_umap)
        labels = gmm.predict(X_umap)
        bic    = gmm.bic(X_umap)
        print(f"    GMM K={K}  BIC={bic:.0f}")
    else:
        best_bic, best_labels, K = np.inf, None, 2
        for k in range(2, 21):
            g = GaussianMixture(n_components=k, covariance_type="full",
                                n_init=3, random_state=SEED)
            g.fit(X_umap)
            b = g.bic(X_umap)
            if b < best_bic:
                best_bic, best_labels, K = b, g.predict(X_umap), k
        labels = best_labels
        print(f"    GMM BIC → K={K}")

    # Metriche
    sil = float(silhouette_score(X_umap, labels, random_state=SEED))
    db  = float(davies_bouldin_score(X_umap, labels))

    # η² ANOVA sul prezzo LMP
    if PREP_PATH.exists():
        prep_  = pd.read_parquet(PREP_PATH)
        prep_["datetime"] = pd.to_datetime(prep_["datetime"])
        lmp_map = prep_.set_index("datetime")["lmp"]
        lmp_win = pd.Series(times_v).map(lmp_map).values
        groups = [lmp_win[labels == k] for k in np.unique(labels)]
        groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) > 1]
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            ss_between = sum(len(g) * (g.mean() - np.concatenate(groups).mean())**2
                             for g in groups)
            ss_total   = np.concatenate(groups).var() * (sum(len(g) for g in groups) - 1)
            eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0
        else:
            f_stat, p_val, eta2 = 0.0, 1.0, 0.0
    else:
        f_stat, p_val, eta2 = 0.0, 1.0, 0.0

    print(f"    Silhouette={sil:.4f}  DB={db:.4f}  η²={eta2:.4f}  "
          f"ANOVA-F={f_stat:.1f}  p={p_val:.2e}")

    return {
        "name": name, "K": K,
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "eta2": round(eta2, 4),
        "anova_F": round(float(f_stat), 2),
        "anova_p": float(p_val),
        "pca_var20": round(float(var20), 4),
        "umap": X_umap,
        "labels": labels,
    }


K_use = K_best  # stesso K per entrambi (confronto equo)
res_chronos = run_pipeline(E_chronos_v, f"Chronos-2 (Exp {EXP})", K_use)
res_manual  = run_pipeline(F_manual_v,  "Features manuali",        K_use)

# =============================================================================
# 4. Tabella comparativa
# =============================================================================

print("\n[4] Tabella comparativa...")
metrics = ["silhouette", "davies_bouldin", "eta2", "anova_F", "anova_p", "K"]
rows = []
for r in [res_chronos, res_manual]:
    rows.append({m: r[m] for m in metrics} | {"Metodo": r["name"]})
df_cmp = pd.DataFrame(rows).set_index("Metodo")
df_cmp.to_csv(OUT_DIR / "comparison_table.csv")
print(df_cmp.to_string())
print(f"\n  Salvato: {OUT_DIR / 'comparison_table.csv'}")

# =============================================================================
# 5. Grafici
# =============================================================================

plt.rcParams.update({"figure.dpi": DPI, "savefig.bbox": "tight",
                     "savefig.facecolor": "white"})

# ── 01: barchart metriche ─────────────────────────────────────────────────
print("\n[5] Plot 01_comparison_bar.png...")
metric_plot = {
    "Silhouette ↑":      ("silhouette", True),
    "Davies-Bouldin ↓":  ("davies_bouldin", False),
    "η² (ANOVA) ↑":      ("eta2", True),
}
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
colors = ["#2c6fad", "#e07b39"]
for ax, (title, (key, higher_better)) in zip(axes, metric_plot.items()):
    vals   = [res_chronos[key], res_manual[key]]
    labels = [f"Chronos-2\nExp {EXP}", "Features\nmanuali"]
    bars   = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    winner_idx = 0 if (higher_better and vals[0] >= vals[1]) or \
                      (not higher_better and vals[0] <= vals[1]) else 1
    bars[winner_idx].set_edgecolor("gold")
    bars[winner_idx].set_linewidth(2.5)

fig.suptitle(f"Chronos-2 vs Features Manuali — Exp {EXP} (K={res_chronos['K']})",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "01_comparison_bar.png", dpi=DPI)
plt.close(fig)
print(f"  Salvato: {OUT_DIR / '01_comparison_bar.png'}")

# ── 02: UMAP scatter per entrambi ────────────────────────────────────────
for res, fname in [(res_chronos, "02_umap_chronos.png"),
                   (res_manual,  "02_umap_manual.png")]:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(res["umap"][:, 0], res["umap"][:, 1],
                    c=res["labels"], cmap="tab20",
                    s=3, alpha=0.5, linewidths=0)
    ax.set_title(f"UMAP — {res['name']}  K={res['K']}\n"
                 f"Sil={res['silhouette']:.4f}  η²={res['eta2']:.4f}",
                 fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=DPI)
    plt.close(fig)
    print(f"  Salvato: {OUT_DIR / fname}")

# ── 03: t-SNE side-by-side ───────────────────────────────────────────────
try:
    from sklearn.manifold import TSNE
    print("\n[5] Plot 03_tsne_comparison.png (può richiedere 1-2 min)...")
    idx  = np.random.default_rng(SEED).choice(len(E_chronos_v),
                                               min(3000, len(E_chronos_v)),
                                               replace=False)
    idx.sort()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (X, res, title) in zip(axes, [
        (E_chronos_v[idx], res_chronos, f"Chronos-2 Exp {EXP}"),
        (F_manual_v[idx],  res_manual,  "Features manuali"),
    ]):
        Xs = StandardScaler().fit_transform(X)
        Z  = TSNE(n_components=2, perplexity=40, n_iter=1000,
                  random_state=SEED, n_jobs=N_JOBS).fit_transform(Xs)
        ax.scatter(Z[:, 0], Z[:, 1], c=res["labels"][idx],
                   cmap="tab20", s=4, alpha=0.6, linewidths=0)
        ax.set_title(f"t-SNE — {title}\nSil={res['silhouette']:.4f}  "
                     f"η²={res['eta2']:.4f}", fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"t-SNE: struttura dei cluster a confronto (K={res_chronos['K']})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_tsne_comparison.png", dpi=DPI)
    plt.close(fig)
    print(f"  Salvato: {OUT_DIR / '03_tsne_comparison.png'}")
except Exception as e:
    print(f"  t-SNE saltato: {e}")

# =============================================================================
# 6. Riepilogo
# =============================================================================
print("\n" + "═"*65)
print("  RISULTATO")
print("═"*65)
print(df_cmp[["silhouette","davies_bouldin","eta2","anova_F"]].to_string())

sil_win = "Chronos-2" if res_chronos["silhouette"] >= res_manual["silhouette"] \
          else "Features manuali"
db_win  = "Chronos-2" if res_chronos["davies_bouldin"] <= res_manual["davies_bouldin"] \
          else "Features manuali"
eta_win = "Chronos-2" if res_chronos["eta2"] >= res_manual["eta2"] \
          else "Features manuali"

print(f"\n  Silhouette  → vince: {sil_win}")
print(f"  Davies-Bouldin → vince: {db_win}")
print(f"  η²          → vince: {eta_win}")
print(f"\n  Output: {OUT_DIR}")
