"""
step03f_baseline_features.py
============================
Baseline di confronto per Chronos-2: stesse finestre (720h, stride 6h),
stessa feature (mstl_resid_arcsinh = Exp D), ma vettore di feature manuali
invece dell'embedding neurale.

Feature manuali per finestra (12 features):
  mean        — livello medio del residuo
  std         — deviazione standard (volatilita')
  skewness    — asimmetria (spike positivi o negativi)
  kurtosis    — code pesanti
  p10, p90    — percentili per forma della distribuzione
  acf1        — autocorrelazione lag-1 (memoria a breve)
  acf24       — autocorrelazione lag-24 (ciclo giornaliero residuo)
  acf168      — autocorrelazione lag-168 (ciclo settimanale residuo)
  spike_rate  — frequenza di valori > 2 std dal mean
  range_norm  — (max-min)/std (ampiezza relativa)
  hurst       — esponente di Hurst (R/S) stima persistenza

Pipeline identica a step03f:
  PCA(20D) -> UMAP(3D, cosine) -> GMM K=8 full

Output
------
  results/exp_baseline/step03f/
    quality_report.json
    labels_best.parquet
    01..10_*.png

Confronto
---------
  Al termine stampa tabella comparativa Chronos-2 (Exp D) vs Baseline.

Uso
---
  python step03f_baseline_features.py
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
from scipy.stats import skew, kurtosis, chi2_contingency, f_oneway
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

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

_ROOT = Path(__file__).parent.parent.resolve()

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

PREPROC_PATH  = _ROOT / C.RESULTS_DIR / "preprocessed.parquet"
FEAT_COL      = "mstl_resid_arcsinh"   # stessa feature di Exp D
CONTEXT_LEN   = 720                    # ore (= finestra Chronos-2)
STRIDE_H      = 6
GAP_FROM      = pd.Timestamp("2023-06-14 23:00:00")
GAP_TO        = pd.Timestamp("2023-06-16 00:00:00")

OUT_DIR       = _ROOT / C.RESULTS_DIR / "exp_baseline" / "step03f"
CHRONOS_QR    = _ROOT / C.RESULTS_DIR / "exp_D" / "step03f" / "quality_report.json"

PCA_COMPONENTS   = 20
UMAP_COMPONENTS  = 3
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_METRIC      = "cosine"
FORCE_K          = 8
COV_TYPE         = "full"
N_INIT           = 5
SEED             = 42
DPI              = 150

CMAP = plt.get_cmap("tab20")
def regime_colors(n): return [CMAP(i / max(n, 1)) for i in range(n)]

# =============================================================================
# HURST EXPONENT (R/S method)
# =============================================================================

def hurst_rs(x: np.ndarray) -> float:
    """Stima dell'esponente di Hurst con metodo R/S (Hurst 1951)."""
    N = len(x)
    if N < 20:
        return 0.5
    lags  = [int(N // k) for k in [2, 4, 8] if N // k >= 10]
    if not lags:
        return 0.5
    rs_vals = []
    for lag in lags:
        chunks = [x[i:i+lag] for i in range(0, N - lag + 1, lag)]
        rs_chunk = []
        for c in chunks:
            c = c - c.mean()
            Z = np.cumsum(c)
            R = Z.max() - Z.min()
            S = c.std()
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append((np.log(lag), np.log(np.mean(rs_chunk))))
    if len(rs_vals) < 2:
        return 0.5
    xs, ys = zip(*rs_vals)
    H = np.polyfit(xs, ys, 1)[0]
    return float(np.clip(H, 0.0, 1.0))


# =============================================================================
# FEATURE MANUALI PER FINESTRA
# =============================================================================

def window_features(w: np.ndarray) -> np.ndarray:
    """
    Calcola 12 feature manuali su una finestra 1D di lunghezza CONTEXT_LEN.
    """
    w   = w.astype(float)
    mu  = w.mean()
    s   = w.std() + 1e-8

    # ACF lag k: correlazione tra w e w shiftato di k
    def acf(lag):
        if lag >= len(w):
            return 0.0
        a, b = w[:-lag], w[lag:]
        denom = np.std(a) * np.std(b)
        return float(np.corrcoef(a, b)[0, 1]) if denom > 0 else 0.0

    spike_rate = float(np.mean(np.abs(w - mu) > 2 * s))
    rng_norm   = float((w.max() - w.min()) / s)
    h          = hurst_rs(w)

    return np.array([
        mu,                         # 0 mean
        s,                          # 1 std
        float(skew(w)),             # 2 skewness
        float(kurtosis(w)),         # 3 kurtosis
        float(np.percentile(w, 10)),# 4 p10
        float(np.percentile(w, 90)),# 5 p90
        acf(1),                     # 6 acf lag-1
        acf(24),                    # 7 acf lag-24
        acf(168),                   # 8 acf lag-168
        spike_rate,                 # 9 spike rate
        rng_norm,                   # 10 range normalizzato
        h,                          # 11 Hurst
    ], dtype=np.float32)


# =============================================================================
# BUILD WINDOWS
# =============================================================================

def build_feature_matrix() -> tuple[np.ndarray, pd.Series]:
    """Costruisce (N_windows, 12) con feature manuali e relativi timestamp."""
    print(f"  Carico {PREPROC_PATH}...", flush=True)
    df = pd.read_parquet(PREPROC_PATH)[["datetime", FEAT_COL]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    dt = df["datetime"].values
    X  = df[FEAT_COL].values.astype(np.float32)
    N  = len(X)

    feats, timestamps = [], []
    n_skip = 0

    for start in range(0, N - CONTEXT_LEN, STRIDE_H):
        end = start + CONTEXT_LEN
        if end >= N:
            break
        t_start = pd.Timestamp(dt[start])
        t_end   = pd.Timestamp(dt[end - 1])
        # salta finestre che attraversano il gap
        if t_start <= GAP_TO and t_end >= GAP_FROM:
            n_skip += 1
            continue
        feats.append(window_features(X[start:end]))
        timestamps.append(t_end)

    F  = np.stack(feats)
    ts = pd.Series(timestamps)
    print(f"  {len(F):,} finestre  ({n_skip} skippate per gap)  shape={F.shape}",
          flush=True)
    return F, ts


# =============================================================================
# QUALITY METRICS  (stesso calcolo di step03f)
# =============================================================================

def _cramers_v(ct):
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n = ct.sum(); r, c = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r,c)-1)))) if min(r,c)>1 else 0.

def _eta_squared(groups):
    gm = np.concatenate(groups).mean()
    ssb = sum(len(g)*(g.mean()-gm)**2 for g in groups)
    sst = sum(((x-gm)**2).sum() for g in groups for x in [g])
    return float(ssb/sst) if sst > 0 else 0.

def compute_quality(E: np.ndarray, labels: np.ndarray,
                    ts: pd.Series) -> dict:
    K   = len(np.unique(labels))
    ts_ = pd.to_datetime(ts).reset_index(drop=True)
    out = {}
    out["silhouette"]        = round(float(silhouette_score(E, labels, random_state=SEED)), 4)
    out["davies_bouldin"]    = round(float(davies_bouldin_score(E, labels)), 4)
    out["calinski_harabasz"] = round(float(calinski_harabasz_score(E, labels)), 1)
    counts = np.bincount(labels); p = counts/counts.sum(); p_pos = p[p>0]
    out["entropy_norm"]    = round(float(-np.sum(p_pos*np.log(p_pos))/np.log(K)), 4)
    out["transition_rate"] = round(float(np.sum(labels[1:]!=labels[:-1])/max(len(labels)-1,1)), 4)

    for unit, getter in [("season", lambda t: t.dt.month.map(
                          {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}))]:
        col = getter(ts_)
        ct  = pd.crosstab(labels, col).values
        out[f"cramer_{unit}"] = round(_cramers_v(ct), 4)

    # sojourn
    runs, cur, cnt = [], labels[0], 1
    for l in labels[1:]:
        if l == cur: cnt += 1
        else: runs.append(cnt); cur, cnt = l, 1
    runs.append(cnt)
    out["sojourn_mean_h"] = round(float(np.mean(runs)) * 6, 2)

    # LMP separation
    pre_path = Path("results/preprocessed.parquet")
    if pre_path.exists():
        pre = pd.read_parquet(pre_path)[["datetime","lmp"]]
        pre["datetime"] = pd.to_datetime(pre["datetime"])
        df  = pd.DataFrame({"datetime": ts_}).merge(pre, on="datetime", how="left")
        groups = [df.loc[df.index[labels==k], "lmp"].dropna().values for k in range(K)]
        groups = [g for g in groups if len(g)>1]
        if len(groups) >= 2:
            f_stat, p_anova = f_oneway(*groups)
            out["price_eta_squared"] = round(_eta_squared(groups), 4)
            out["price_anova_F"]     = round(float(f_stat), 3)
            out["price_anova_p"]     = round(float(p_anova), 6)
            for k in range(K):
                g = df.loc[df.index[labels==k], "lmp"].dropna()
                if len(g):
                    out[f"lmp_mean_R{k}"] = round(float(g.mean()), 2)
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    try: plt.style.use("seaborn-v0_8-whitegrid")
    except Exception: pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBaseline: feature manuali vs Chronos-2  (feature = mstl_resid_arcsinh)",
          flush=True)

    # ── 1. Feature matrix ─────────────────────────────────────────────────
    print("\n[1/4] Costruisco feature manuali...", flush=True)
    F, ts = build_feature_matrix()

    # ── 2. PCA + UMAP ─────────────────────────────────────────────────────
    print("\n[2/4] PCA + UMAP...", flush=True)
    F_s   = StandardScaler().fit_transform(F)
    # PCA: feature solo 12, prendiamo tutte (o min(12, PCA_COMPONENTS))
    n_pca = min(PCA_COMPONENTS, F_s.shape[1])
    F_pca = PCA(n_components=n_pca, random_state=SEED).fit_transform(F_s)
    var   = PCA(n_components=n_pca, random_state=SEED).fit(F_s).explained_variance_ratio_.sum()
    print(f"  PCA({n_pca}D)  varianza spiegata: {var:.1%}", flush=True)

    F_umap = UMAP(
        n_components=UMAP_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC,
        random_state=SEED, verbose=False,
    ).fit_transform(F_pca)
    print(f"  UMAP shape: {F_umap.shape}", flush=True)

    # ── 3. GMM K=8 ────────────────────────────────────────────────────────
    print(f"\n[3/4] GMM K={FORCE_K} {COV_TYPE}...", flush=True)
    gmm    = GaussianMixture(n_components=FORCE_K, covariance_type=COV_TYPE,
                              n_init=N_INIT, random_state=SEED, max_iter=200)
    gmm.fit(F_umap)
    labels = gmm.predict(F_umap)
    bic    = gmm.bic(F_umap)
    print(f"  BIC={bic:.1f}", flush=True)

    # ── 4. Quality metrics ────────────────────────────────────────────────
    print("\n[4/4] Quality metrics...", flush=True)
    quality = compute_quality(F_umap, labels, ts)

    # salva labels
    pd.DataFrame({"timestamp": ts, "regime": labels}
                 ).to_parquet(OUT_DIR / "labels_best.parquet", index=False)

    # salva quality report
    report = {"experiment": "baseline", "K": FORCE_K, "feature": FEAT_COL,
              "method": "manual_features_12D", "bic": round(bic, 1),
              "quality": quality}
    with open(OUT_DIR / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  quality_report.json", flush=True)

    # ── Confronto con Chronos-2 (Exp D) ───────────────────────────────────
    print("\n" + "=" * 65)
    print("  CONFRONTO: Feature Manuali  vs  Chronos-2 (Exp D)")
    print("=" * 65)

    chronos_q = {}
    if CHRONOS_QR.exists():
        with open(CHRONOS_QR) as f:
            qr = json.load(f)
        chronos_q = {**qr.get("clustering_quality", {}),
                     **qr.get("price_separation", {})}

    rows = [
        ("silhouette (↑)",      quality.get("silhouette"),
                                chronos_q.get("silhouette")),
        ("davies_bouldin (↓)",  quality.get("davies_bouldin"),
                                chronos_q.get("davies_bouldin")),
        ("calinski_harabasz (↑)",quality.get("calinski_harabasz"),
                                 chronos_q.get("calinski_harabasz")),
        ("transition_rate (↓)", quality.get("transition_rate"),
                                chronos_q.get("transition_rate")),
        ("cramer_season (↑)",   quality.get("cramer_season"),
                                chronos_q.get("cramer_season")),
        ("sojourn_mean_h (↑)",  quality.get("sojourn_mean_h"),
                                None),
        ("price_eta² (↑)",      quality.get("price_eta_squared"),
                                chronos_q.get("price_eta_squared")),
        ("price_anova_F (↑)",   quality.get("price_anova_F"),
                                chronos_q.get("price_anova_F")),
    ]

    print(f"  {'Metrica':<26}  {'Baseline':>10}  {'Chronos-2':>10}  Winner")
    print("  " + "-" * 58)
    for name, base_val, chron_val in rows:
        bv  = f"{base_val:.4f}" if base_val is not None else "n/a"
        cv  = f"{chron_val:.4f}" if chron_val is not None else "n/a"
        # determina vincitore (↑ = higher better, ↓ = lower better)
        if base_val is not None and chron_val is not None:
            higher_better = "(↑)" in name
            win = "Chronos" if (chron_val > base_val) == higher_better else "Baseline"
        else:
            win = ""
        print(f"  {name:<26}  {bv:>10}  {cv:>10}  {win}")
    print("=" * 65 + "\n")

    # plot comparativo semplice
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics_plot = ["silhouette","davies_bouldin","transition_rate",
                    "cramer_season","price_eta_squared"]
    labels_plot  = ["Silhouette↑","DB↓","Trans.rate↓","Cramér seas↑","η²↑"]
    # normalizza [0,1] ciascuna metrica
    directions   = ["max","min","min","max","max"]

    base_vals  = [quality.get(m, 0) for m in metrics_plot]
    chron_vals = [chronos_q.get(m, 0) for m in metrics_plot]

    # normalizza con min/max tra i due
    norm_b, norm_c = [], []
    for bv, cv, d in zip(base_vals, chron_vals, directions):
        mn, mx = min(bv,cv), max(bv,cv)
        rng = mx - mn if mx > mn else 1e-12
        nb = (bv - mn) / rng
        nc = (cv - mn) / rng
        if d == "min":
            nb, nc = 1 - nb, 1 - nc
        norm_b.append(nb); norm_c.append(nc)

    x = np.arange(len(metrics_plot))
    w = 0.35
    axes[0].bar(x - w/2, norm_b, w, label="Feature manuali", color="#EE6677", alpha=0.85)
    axes[0].bar(x + w/2, norm_c, w, label="Chronos-2 (Exp D)", color="#4477AA", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_plot, fontsize=8)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Score normalizzato [0-1]  (piu' alto = meglio)")
    axes[0].set_title("Confronto metriche normalizzate", fontweight="bold")
    axes[0].legend(fontsize=8)

    # radar-like: valori assoluti chiave
    keys  = ["silhouette","transition_rate","price_eta_squared"]
    names = ["Silhouette","Trans.rate*","Price η²"]
    bvs   = [quality.get(k, 0) for k in keys]
    cvs   = [chronos_q.get(k, 0) for k in keys]
    axes[1].bar(np.arange(3) - 0.18, bvs, 0.35,
                label="Feature manuali", color="#EE6677", alpha=0.85)
    axes[1].bar(np.arange(3) + 0.18, cvs, 0.35,
                label="Chronos-2", color="#4477AA", alpha=0.85)
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_title("Metriche chiave (valori assoluti)\n* trans.rate: lower is better",
                      fontweight="bold", fontsize=8)
    axes[1].legend(fontsize=8)

    fig.suptitle("Baseline: Feature Manuali vs Chronos-2 (Exp D)\n"
                 "Stessa feature (mstl_resid_arcsinh) | stessa pipeline PCA+UMAP+GMM K=8",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "00_chronos_vs_baseline.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  00_chronos_vs_baseline.png", flush=True)
    print(f"  Output -> {OUT_DIR}/\n", flush=True)


if __name__ == "__main__":
    main()
