"""
step03h_comparison.py
=====================
Confronto dei 4 rami sperimentali (A, B, C, D).

Rami
----
  A  log_return          (raw)
  B  arcsinh_lmp         (raw)
  C  mstl_resid_lr       (MSTL residuo log_return)
  D  mstl_resid_arcsinh  (MSTL residuo arcsinh_lmp)

Test
----
  1. BIC curve overlay           — quale ramo ha il gomito piu' netto
  2. Metriche di qualita' a K=8  — silhouette, DB, CH per ogni ramo
  3. Seasonal confounding        — V di Cramer regime x stagione (basso = destagionalizzato)
  4. Accordo inter-ramo (ARI)    — matrice ARI tra coppie di rami a K=8
  5. Eta^2 su LMP e load         — varianza economica spiegata dal clustering

Input
-----
  results/exp_{X}/step03f/grid_results.csv
  results/exp_{X}/step03f/labels_best.parquet
  results/preprocessed.parquet

Output (results/step03h/)
--------------------------
  01_bic_overlay.png         BIC curves dei 4 rami
  02_quality_metrics.png     silhouette / DB / CH a K=8
  03_seasonal_confounding.png V di Cramer per ramo
  04_ari_matrix.png          matrice ARI inter-ramo
  05_eta2_economic.png       eta^2 su LMP e load
  comparison_report.txt      report testuale

Uso
---
  python step03h_comparison.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

EXPERIMENTS     = ["A", "B", "C", "D"]
EXP_LABELS      = {
    "A": "log_return (raw)",
    "B": "arcsinh_lmp (raw)",
    "C": "mstl_resid_lr",
    "D": "mstl_resid_arcsinh",
}
EXP_COLORS      = {"A": "#4477AA", "B": "#EE6677", "C": "#228833", "D": "#CCBB44"}

PREPROCESSED    = Path("results/preprocessed.parquet")
OUT_DIR         = Path("results/step03h")

K_COMPARE       = 8       # K a cui confrontare le metriche puntuali
CONTEXT_H       = 720
STRIDE_H        = 6
PEAK_HOURS      = set(range(7, 23))

SEASON_MAP = {12: "Winter", 1: "Winter",  2: "Winter",
              3:  "Spring", 4: "Spring",  5: "Spring",
              6:  "Summer", 7: "Summer",  8: "Summer",
              9:  "Autumn", 10: "Autumn", 11: "Autumn"}

DPI             = 150
MPL_STYLE       = "seaborn-v0_8-whitegrid"

# =============================================================================
# CARICAMENTO
# =============================================================================

def load_grid(exp):
    path = Path(f"results/exp_{exp}/step03f/grid_results.csv")
    if not path.exists():
        print(f"  [WARN] {path} non trovato — ramo {exp} saltato", flush=True)
        return None
    return pd.read_csv(path)


def load_labels(exp):
    # prova prima labels_best, poi labels_K08_full
    for fname in ["labels_best.parquet", f"labels_K{K_COMPARE:02d}_full.parquet"]:
        path = Path(f"results/exp_{exp}/step03f/{fname}")
        if path.exists():
            df = pd.read_parquet(path)
            ts_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
            lbl_col = [c for c in df.columns if c != ts_col][0]
            return pd.to_datetime(df[ts_col]), df[lbl_col].values
    print(f"  [WARN] labels non trovati per ramo {exp}", flush=True)
    return None, None


def load_preprocessed():
    pre = pd.read_parquet(PREPROCESSED)
    pre.index = pd.to_datetime(pre["datetime"] if "datetime" in pre.columns else pre.index)
    pre = pre.sort_index()
    return pre


def build_window_lmp_load(timestamps, pre):
    """Per ogni finestra calcola LMP medio e load medio."""
    records = []
    for ts in pd.to_datetime(timestamps):
        start = ts - pd.Timedelta(hours=CONTEXT_H - 1)
        win   = pre.loc[start:ts]
        rec   = {"timestamp": ts}
        if "arcsinh_lmp" in win.columns:
            rec["lmp_mean"] = np.sinh(win["arcsinh_lmp"].mean())
        if "total_mw" in win.columns:
            rec["load_mean"] = win["total_mw"].mean()
        rec["season"] = SEASON_MAP.get(ts.month, "Unknown")
        records.append(rec)
    return pd.DataFrame(records)


# =============================================================================
# TEST
# =============================================================================

def compute_seasonal_cramer_v(labels, timestamps):
    """V di Cramer tra regime e stagione."""
    ts      = pd.to_datetime(timestamps).reset_index(drop=True)
    seasons = ts.apply(lambda t: SEASON_MAP.get(t.month, "Unknown"))
    ct      = pd.crosstab(labels, seasons)
    chi2, p, dof, _ = scipy_stats.chi2_contingency(ct)
    n  = ct.values.sum()
    k  = min(ct.shape) - 1
    V  = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0
    return V, p


def compute_eta2(labels, values):
    """Eta^2 = SS_between / SS_total."""
    values  = np.array(values, dtype=float)
    mask    = ~np.isnan(values)
    labels  = np.array(labels)[mask]
    values  = values[mask]
    grand   = values.mean()
    regimes = np.unique(labels)
    ss_b    = sum((labels == r).sum() * (values[labels == r].mean() - grand)**2
                  for r in regimes)
    ss_t    = ((values - grand)**2).sum()
    return ss_b / ss_t if ss_t > 0 else 0.0


def get_metrics_at_K(grid_df, k=K_COMPARE):
    """Estrae metriche a K specifico dalla grid (cov=full)."""
    row = grid_df[(grid_df["K"] == k) & (grid_df["cov_type"] == "full")]
    if len(row) == 0:
        row = grid_df[grid_df["K"] == k].iloc[:1]
    if len(row) == 0:
        return {}
    row = row.iloc[0]
    return {
        "bic":        row.get("bic", np.nan),
        "aic":        row.get("aic", np.nan),
        "silhouette": row.get("silhouette", np.nan),
        "CH":         row.get("CH", np.nan),
        "DB":         row.get("DB", np.nan),
    }


# =============================================================================
# PLOT
# =============================================================================

def plot_bic_overlay(grids):
    """01 - BIC curves dei 4 rami (solo cov=full)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for exp, grid in grids.items():
        if grid is None:
            continue
        sub = grid[grid["cov_type"] == "full"].sort_values("K")
        ax.plot(sub["K"], sub["bic"], marker="o", ms=4, lw=1.8,
                color=EXP_COLORS[exp], label=EXP_LABELS[exp])
        idx_min = sub["bic"].idxmin()
        ax.scatter(sub.loc[idx_min, "K"], sub.loc[idx_min, "bic"],
                   s=100, color=EXP_COLORS[exp], zorder=5, marker="*")
    ax.axvline(K_COMPARE, color="gray", lw=1, ls="--", label=f"K={K_COMPARE}")
    ax.set_xlabel("K")
    ax.set_ylabel("BIC")
    ax.set_title("BIC vs K — confronto rami (cov=full, stella=minimo)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_bic_overlay.png", dpi=DPI)
    plt.close(fig)
    print("  01_bic_overlay.png", flush=True)


def plot_quality_metrics(metrics_dict):
    """02 - Silhouette / DB / CH a K=K_COMPARE."""
    exps   = [e for e in EXPERIMENTS if e in metrics_dict]
    metric_names = ["silhouette", "DB", "CH"]
    titles       = ["Silhouette (max)", "Davies-Bouldin (min)", "Calinski-Harabasz (max)"]
    colors       = [EXP_COLORS[e] for e in exps]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, metric_names, titles):
        vals = [metrics_dict[e].get(metric, np.nan) for e in exps]
        bars = ax.bar(exps, vals, color=colors, alpha=0.8, edgecolor="white")
        ax.set_title(f"{title}\nK={K_COMPARE}")
        ax.set_ylabel(metric)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels([EXP_LABELS[e] for e in exps], fontsize=7, rotation=15)

    fig.suptitle(f"Metriche di qualita' a K={K_COMPARE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_quality_metrics.png", dpi=DPI)
    plt.close(fig)
    print("  02_quality_metrics.png", flush=True)


def plot_seasonal_confounding(cramer_dict):
    """03 - V di Cramer regime x stagione per ramo."""
    exps  = [e for e in EXPERIMENTS if e in cramer_dict]
    vals  = [cramer_dict[e][0] for e in exps]
    colors = [EXP_COLORS[e] for e in exps]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(exps, vals, color=colors, alpha=0.8, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(exps)))
    ax.set_xticklabels([EXP_LABELS[e] for e in exps], fontsize=8, rotation=15)
    ax.set_ylabel("V di Cramer")
    ax.set_title("Seasonal confounding — V di Cramer regime x stagione\n"
                 "(basso = regimi indipendenti dalla stagione)")
    ax.set_ylim(0, max(vals) * 1.2)
    # linee riferimento
    for v, lbl in [(0.1, "piccolo"), (0.3, "medio"), (0.5, "grande")]:
        ax.axhline(v, color="gray", lw=0.8, ls="--", alpha=0.6, label=lbl)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_seasonal_confounding.png", dpi=DPI)
    plt.close(fig)
    print("  03_seasonal_confounding.png", flush=True)


def plot_ari_matrix(labels_dict):
    """04 - Matrice ARI inter-ramo."""
    exps = [e for e in EXPERIMENTS if e in labels_dict]
    n    = len(exps)
    mat  = np.eye(n)

    for i, ea in enumerate(exps):
        for j, eb in enumerate(exps):
            if i < j:
                # allinea per timestamp comune
                ts_a, la = labels_dict[ea]
                ts_b, lb = labels_dict[eb]
                df_a = pd.DataFrame({"ts": pd.to_datetime(ts_a), "la": la})
                df_b = pd.DataFrame({"ts": pd.to_datetime(ts_b), "lb": lb})
                merged = df_a.merge(df_b, on="ts")
                if len(merged) > 0:
                    ari = adjusted_rand_score(merged["la"], merged["lb"])
                else:
                    ari = np.nan
                mat[i, j] = mat[j, i] = ari

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=1.0)
    fig.colorbar(im, ax=ax, label="ARI")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels_tick = [EXP_LABELS[e] for e in exps]
    ax.set_xticklabels(labels_tick, fontsize=7, rotation=20)
    ax.set_yticklabels(labels_tick, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(mat[i,j]) < 0.7 else "white")
    ax.set_title(f"ARI inter-ramo a K={K_COMPARE}\n(1=identici, 0=casuali)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_ari_matrix.png", dpi=DPI)
    plt.close(fig)
    print("  04_ari_matrix.png", flush=True)


def plot_eta2_economic(eta2_dict):
    """05 - Eta^2 su LMP e load per ramo."""
    exps   = [e for e in EXPERIMENTS if e in eta2_dict]
    colors = [EXP_COLORS[e] for e in exps]
    feats  = ["lmp_mean", "load_mean"]
    titles = ["LMP medio (eta^2)", "Load medio (eta^2)"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, feat, title in zip(axes, feats, titles):
        vals = [eta2_dict[e].get(feat, np.nan) for e in exps]
        bars = ax.bar(exps, vals, color=colors, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels([EXP_LABELS[e] for e in exps], fontsize=7, rotation=15)
        ax.set_ylabel("eta^2")
        ax.set_title(title)
        ax.axhline(0.14, color="gray", lw=0.8, ls="--", label="grande (0.14)")
        ax.axhline(0.06, color="gray", lw=0.8, ls=":",  label="medio (0.06)")
        ax.legend(fontsize=7)

    fig.suptitle(f"Varianza economica spiegata dal clustering (K={K_COMPARE})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_eta2_economic.png", dpi=DPI)
    plt.close(fig)
    print("  05_eta2_economic.png", flush=True)


# =============================================================================
# REPORT
# =============================================================================

def write_report(grids, metrics_dict, cramer_dict, eta2_dict):
    lines = []
    lines.append("=" * 65)
    lines.append("COMPARISON REPORT — 4 rami sperimentali")
    lines.append("=" * 65)
    lines.append(f"  A: {EXP_LABELS['A']}")
    lines.append(f"  B: {EXP_LABELS['B']}")
    lines.append(f"  C: {EXP_LABELS['C']}")
    lines.append(f"  D: {EXP_LABELS['D']}")

    lines.append(f"\n--- Metriche a K={K_COMPARE} (cov=full) ---")
    lines.append(f"{'Ramo':<6} {'BIC':>12} {'Sil':>8} {'DB':>8} {'CH':>8}")
    lines.append("-" * 45)
    for e in EXPERIMENTS:
        if e not in metrics_dict:
            continue
        m = metrics_dict[e]
        lines.append(f"{e:<6} {m.get('bic',float('nan')):>12.1f} "
                     f"{m.get('silhouette',float('nan')):>8.4f} "
                     f"{m.get('DB',float('nan')):>8.4f} "
                     f"{m.get('CH',float('nan')):>8.1f}")

    lines.append("\n--- Seasonal confounding (V di Cramer) ---")
    lines.append(f"{'Ramo':<6} {'V Cramer':>10} {'p-value':>12}")
    lines.append("-" * 30)
    for e in EXPERIMENTS:
        if e not in cramer_dict:
            continue
        V, p = cramer_dict[e]
        lines.append(f"{e:<6} {V:>10.4f} {p:>12.2e}")

    lines.append("\n--- Eta^2 economico ---")
    lines.append(f"{'Ramo':<6} {'LMP eta^2':>12} {'Load eta^2':>12}")
    lines.append("-" * 32)
    for e in EXPERIMENTS:
        if e not in eta2_dict:
            continue
        d = eta2_dict[e]
        lines.append(f"{e:<6} {d.get('lmp_mean', float('nan')):>12.4f} "
                     f"{d.get('load_mean', float('nan')):>12.4f}")

    lines.append("\n" + "=" * 65)

    path = OUT_DIR / "comparison_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    for line in lines:
        print(line, flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        plt.style.use(MPL_STYLE)
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carico dati...", flush=True)
    pre = load_preprocessed()

    grids       = {}
    labels_dict = {}
    metrics_dict = {}
    cramer_dict  = {}
    eta2_dict    = {}
    win_dict     = {}

    for exp in EXPERIMENTS:
        print(f"\n--- Ramo {exp}: {EXP_LABELS[exp]} ---", flush=True)
        grid = load_grid(exp)
        grids[exp] = grid

        ts, lbl = load_labels(exp)
        if ts is None:
            continue

        labels_dict[exp] = (ts, lbl)

        if grid is not None:
            metrics_dict[exp] = get_metrics_at_K(grid)

        # window stats (LMP, load)
        print(f"  Calcolo window stats...", flush=True)
        wdf = build_window_lmp_load(ts, pre)
        win_dict[exp] = wdf

        cramer_dict[exp] = compute_seasonal_cramer_v(lbl, ts)

        # eta^2
        e2 = {}
        for feat in ["lmp_mean", "load_mean"]:
            if feat in wdf.columns:
                e2[feat] = compute_eta2(lbl, wdf[feat].values)
        eta2_dict[exp] = e2

    # Plots
    print("\nProduco figure...", flush=True)
    plot_bic_overlay(grids)
    if metrics_dict:
        plot_quality_metrics(metrics_dict)
    if cramer_dict:
        plot_seasonal_confounding(cramer_dict)
    if len(labels_dict) >= 2:
        plot_ari_matrix(labels_dict)
    if eta2_dict:
        plot_eta2_economic(eta2_dict)

    write_report(grids, metrics_dict, cramer_dict, eta2_dict)

    print(f"\nDone. Output in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
