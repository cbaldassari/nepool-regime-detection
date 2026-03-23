"""
step03e_validation.py
=====================
Validazione DESCRITTIVA dei regimi prodotti da qualsiasi step di clustering.
Scopo: verificare che i regimi siano economicamente distinti e stabili nel tempo.
NON e' validazione predittiva (nessun train/test split, nessun forecast).

Test implementati
-----------------
  1. Profilo economico per regime
       media/mediana/IQR di LMP, load, fuel mix per regime

  2. Separazione statistica univariata
       Kruskal-Wallis H + eta^2 (varianza spiegata)  per ogni feature
       Dunn post-hoc pairwise (Bonferroni) tra coppie di regimi

  3. Persistenza temporale
       durata media dei run consecutivi (in finestre e in ore)
       self-transition probability per regime
       matrice di transizione P[i->j]

  4. Coerenza stagionale
       chi-quadro regime x stagione + V di Cramer

  5. Separazione multivariata (PERMANOVA semplificato)
       F-ratio = varianza between / varianza within sulle feature standardizzate
       R^2 = SS_between / SS_total

Input (configurabile)
---------------------
  LABELS_PATH   : parquet con colonne [timestamp, community/regime]
  PREPROCESSED  : results/preprocessed.parquet

Output (results/step03e/)
--------------------------
  01_economic_profiles.png    boxplot LMP e load per regime
  02_fuel_mix.png             fuel mix medio per regime
  03_kruskal_wallis.png       H statistic e eta^2 per feature
  04_dunn_heatmap.png         p-value pairwise Dunn (LMP)
  05_persistence.png          distribuzione durata run per regime
  06_transition_matrix.png    heatmap matrice di transizione
  07_seasonal_heatmap.png     distribuzione stagionale per regime
  validation_report.txt       report testuale con tutti i risultati
  validation_stats.csv        tabella completa metriche per regime

Uso
---
  python step03e_validation.py

  Per cambiare sorgente di etichette, modifica LABELS_PATH nella sezione
  CONFIGURAZIONE qui sotto.
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

# Etichette da validare — cambia qui per usare step03b, step03 o step03c
LABELS_PATH   = Path("results/step03c/labels_best.parquet")
LABEL_COL     = None   # None = auto-detect (prima colonna non-timestamp)

PREPROCESSED  = Path("results/preprocessed.parquet")
OUT_DIR       = Path("results/step03e")

# Ore di contesto per finestra (deve coincidere con step02)
CONTEXT_H     = 720

# Ore "peak"
PEAK_HOURS    = set(range(7, 23))

# Feature da testare nella validazione statistica
FEATURES_TEST = ["arcsinh_lmp", "total_mw", "ilr_1", "ilr_2", "ilr_3",
                 "ilr_4", "ilr_5", "ilr_6", "ilr_7"]

# Stride usato in step02 (ore tra una finestra e la prossima)
WINDOW_STRIDE_H = 6

DPI           = 150
MPL_STYLE     = "seaborn-v0_8-whitegrid"

# =============================================================================
# ILR inverse  (SBP identico a step01)
# =============================================================================

#        Gas  Nuc  Hyd  Win  Sol  Coa  Oil  Oth
SBP = np.array([
    [ 1,   1,  -1,  -1,  -1,   1,   1,   1],
    [ 1,  -1,   0,   0,   0,   1,   1,  -1],
    [ 1,   0,   0,   0,   0,  -1,  -1,   0],
    [ 0,   0,   0,   0,   0,   1,  -1,   0],
    [ 0,   1,   0,   0,   0,   0,   0,  -1],
    [ 0,   0,   1,  -1,  -1,   0,   0,   0],
    [ 0,   0,   0,  -1,   1,   0,   0,   0],
], dtype=float)

FUEL_NAMES  = ["Gas", "Nuclear", "Hydro", "Wind", "Solar", "Coal", "Oil", "Other"]
ILR_COLS    = [f"ilr_{i}" for i in range(1, 8)]

SEASON_MAP = {12: "Winter", 1: "Winter",  2: "Winter",
              3:  "Spring", 4: "Spring",  5: "Spring",
              6:  "Summer", 7: "Summer",  8: "Summer",
              9:  "Autumn", 10: "Autumn", 11: "Autumn"}


def _ilr_basis(sbp):
    n_coords, D = sbp.shape
    Psi = np.zeros((n_coords, D))
    for j, row in enumerate(sbp):
        pos = row ==  1;  neg = row == -1
        r, s = pos.sum(), neg.sum()
        Psi[j, pos] =  np.sqrt(s / (r * (r + s)))
        Psi[j, neg] = -np.sqrt(r / (s * (r + s)))
    return Psi


def ilr_inverse(ilr: np.ndarray) -> np.ndarray:
    """(N,7) ILR coords -> (N,8) fuel shares (rows sum to 1)."""
    Psi   = _ilr_basis(SBP)
    clr   = ilr @ Psi
    exp_c = np.exp(clr)
    return exp_c / exp_c.sum(axis=1, keepdims=True)


CMAP = plt.get_cmap("tab20")

def regime_colors(n):
    return [CMAP(i % 20) for i in range(n)]

# =============================================================================
# CARICAMENTO DATI
# =============================================================================

def load_data():
    print("Carico labels...", flush=True)
    lab_df = pd.read_parquet(LABELS_PATH)
    ts_col = next(c for c in lab_df.columns
                  if "time" in c.lower() or "date" in c.lower())
    lab_df[ts_col] = pd.to_datetime(lab_df[ts_col])
    lab_df = lab_df.sort_values(ts_col).reset_index(drop=True)

    # individua colonna etichette
    global LABEL_COL
    if LABEL_COL is None:
        LABEL_COL = next(c for c in lab_df.columns if c != ts_col)
    lab_df = lab_df.rename(columns={ts_col: "timestamp", LABEL_COL: "regime"})
    # rimuovi eventuale rumore (-1)
    lab_df = lab_df[lab_df["regime"] >= 0].copy()

    print(f"  finestre: {len(lab_df)}  regimi: {lab_df['regime'].nunique()}", flush=True)

    print("Carico preprocessed...", flush=True)
    pre_df = pd.read_parquet(PREPROCESSED)
    pre_df.index = pd.to_datetime(pre_df.index)
    pre_df = pre_df.sort_index()
    print(f"  preprocessed: {len(pre_df)} righe", flush=True)

    return lab_df, pre_df


# =============================================================================
# ESPANSIONE FINESTRE -> STATISTICHE PER FINESTRA
# =============================================================================

def build_window_stats(lab_df, pre_df):
    """
    Per ogni finestra calcola la media delle feature sul periodo di CONTEXT_H ore.
    Ritorna un DataFrame con una riga per finestra.
    """
    print("Calcolo statistiche per finestra...", flush=True)
    avail_feats = [f for f in FEATURES_TEST if f in pre_df.columns]
    records = []

    for _, row in lab_df.iterrows():
        ts  = row["timestamp"]
        reg = int(row["regime"])
        start = ts - pd.Timedelta(hours=CONTEXT_H - 1)
        win = pre_df.loc[start:ts, avail_feats] if avail_feats else pd.DataFrame()

        if len(win) < CONTEXT_H // 4:
            continue

        rec = {"timestamp": ts, "regime": reg}

        # medie feature
        for f in avail_feats:
            if f in win.columns:
                rec[f"{f}_mean"] = win[f].mean()

        # LMP in scala originale
        if "arcsinh_lmp" in win.columns:
            rec["lmp_mean"]   = np.sinh(win["arcsinh_lmp"].mean())
            rec["lmp_median"] = np.sinh(win["arcsinh_lmp"].median())
            rec["lmp_p95"]    = np.sinh(np.percentile(win["arcsinh_lmp"], 95))
            rec["lmp_p5"]     = np.sinh(np.percentile(win["arcsinh_lmp"], 5))

        # fuel mix medio
        ilr_avail = [c for c in ILR_COLS if c in win.columns]
        if len(ilr_avail) == 7:
            ilr_mat  = win[ilr_avail].mean().values.reshape(1, 7)
            fuel_shares = ilr_inverse(ilr_mat)[0]
            for name, share in zip(FUEL_NAMES, fuel_shares):
                rec[f"fuel_{name}"] = share

        # stagione (moda del mese)
        rec["season"] = SEASON_MAP.get(ts.month, "Unknown")

        # ora media peak
        if "arcsinh_lmp" in win.columns:
            win_hours = win.copy()
            win_hours["hour"] = pd.DatetimeIndex(win.index).hour
            peak_mask = win_hours["hour"].isin(PEAK_HOURS)
            rec["peak_frac"] = peak_mask.mean()

        records.append(rec)

    ws = pd.DataFrame(records)
    print(f"  finestre valide: {len(ws)}", flush=True)
    return ws


# =============================================================================
# TEST 1: PROFILO ECONOMICO
# =============================================================================

def regime_profiles(ws):
    """Calcola media/mediana/IQR per regime."""
    feat_cols = [c for c in ["lmp_mean", "arcsinh_lmp_mean", "total_mw_mean"] if c in ws.columns]
    fuel_cols = [c for c in ws.columns if c.startswith("fuel_")]

    profiles = {}
    for reg, grp in ws.groupby("regime"):
        p = {"n": len(grp)}
        for c in feat_cols + fuel_cols:
            p[f"{c}_median"] = grp[c].median()
            p[f"{c}_q25"]    = grp[c].quantile(0.25)
            p[f"{c}_q75"]    = grp[c].quantile(0.75)
        profiles[reg] = p

    return pd.DataFrame(profiles).T


# =============================================================================
# TEST 2: KRUSKAL-WALLIS + ETA^2 + DUNN
# =============================================================================

def kruskal_wallis(ws):
    """Kruskal-Wallis H e eta^2 per ogni feature."""
    results = []
    regimes = sorted(ws["regime"].unique())

    mean_cols = {f: f"{f}_mean" for f in FEATURES_TEST if f"{f}_mean" in ws.columns}
    if "lmp_mean" in ws.columns:
        mean_cols["lmp"] = "lmp_mean"

    for feat_key, col in mean_cols.items():
        groups = [ws.loc[ws["regime"] == r, col].dropna().values for r in regimes]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        H, p = stats.kruskal(*groups)
        N = sum(len(g) for g in groups)
        # eta^2 per Kruskal-Wallis
        k = len(groups)
        eta2 = (H - k + 1) / (N - k)
        eta2 = max(0.0, eta2)
        results.append(dict(feature=feat_key, H=H, p_value=p, eta2=eta2, N=N))

    return pd.DataFrame(results).sort_values("eta2", ascending=False)


def dunn_test(ws, feature_col="lmp_mean"):
    """
    Dunn test pairwise con correzione Bonferroni.
    Implementazione manuale senza dipendenze extra.
    """
    if feature_col not in ws.columns:
        return None

    regimes = sorted(ws["regime"].unique())
    n_reg   = len(regimes)
    all_vals = ws[feature_col].dropna().values
    N = len(all_vals)

    # ranghi globali
    ranked = stats.rankdata(ws[feature_col].fillna(ws[feature_col].median()).values)
    ws = ws.copy()
    ws["_rank"] = ranked

    # media ranghi per regime
    R_bar = {r: ws.loc[ws["regime"] == r, "_rank"].mean() for r in regimes}
    n_r   = {r: (ws["regime"] == r).sum() for r in regimes}

    # fattore di correzione per ties
    _, tie_counts = np.unique(ranked, return_counts=True)
    tie_corr = 1 - (np.sum(tie_counts**3 - tie_counts)) / (N**3 - N)

    results = {}
    pairs = list(combinations(regimes, 2))
    n_comp = len(pairs)

    for (i, j) in pairs:
        ni, nj = n_r[i], n_r[j]
        se = np.sqrt(
            (N * (N + 1) / 12.0 - np.sum(tie_counts**3 - tie_counts) / (12.0 * (N - 1)))
            * (1.0/ni + 1.0/nj)
        )
        if se == 0:
            p_adj = 1.0
        else:
            z = (R_bar[i] - R_bar[j]) / se
            p_raw = 2 * stats.norm.sf(abs(z))
            p_adj = min(1.0, p_raw * n_comp)   # Bonferroni
        results[(i, j)] = p_adj

    # costruisci matrice
    mat = pd.DataFrame(np.ones((n_reg, n_reg)), index=regimes, columns=regimes)
    for (i, j), p in results.items():
        mat.loc[i, j] = p
        mat.loc[j, i] = p
    np.fill_diagonal(mat.values, np.nan)

    return mat


# =============================================================================
# TEST 3: PERSISTENZA TEMPORALE
# =============================================================================

def temporal_persistence(lab_df):
    """Calcola run consecutivi per regime e matrice di transizione."""
    df = lab_df.sort_values("timestamp").copy()
    labels = df["regime"].values
    regimes = sorted(df["regime"].unique())
    K = len(regimes)
    reg2idx = {r: i for i, r in enumerate(regimes)}

    # run length encoding
    runs = []
    i = 0
    while i < len(labels):
        r = labels[i]
        j = i
        while j < len(labels) and labels[j] == r:
            j += 1
        runs.append({"regime": r, "length_windows": j - i,
                     "length_hours": (j - i) * WINDOW_STRIDE_H})
        i = j
    runs_df = pd.DataFrame(runs)

    # statistiche durata per regime
    pers = (runs_df.groupby("regime")["length_windows"]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={"mean": "avg_windows", "median": "med_windows",
                              "std": "std_windows", "count": "n_runs"}))
    pers["avg_hours"]  = pers["avg_windows"] * WINDOW_STRIDE_H
    pers["med_hours"]  = pers["med_windows"] * WINDOW_STRIDE_H

    # matrice di transizione
    trans = np.zeros((K, K), dtype=int)
    for t in range(len(labels) - 1):
        fi = reg2idx[labels[t]]
        fj = reg2idx[labels[t + 1]]
        trans[fi, fj] += 1

    # normalizza per riga -> probabilita
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans / row_sums

    trans_df = pd.DataFrame(trans_prob, index=regimes, columns=regimes)

    return pers, trans_df, runs_df


# =============================================================================
# TEST 4: COERENZA STAGIONALE
# =============================================================================

def seasonal_coherence(ws):
    """Chi-quadro regime x stagione + V di Cramer."""
    ct = pd.crosstab(ws["regime"], ws["season"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0
    return dict(chi2=chi2, p_value=p, dof=dof, cramer_v=cramer_v,
                contingency=ct)


# =============================================================================
# TEST 5: PERMANOVA SEMPLIFICATO (F-ratio su feature standardizzate)
# =============================================================================

def permanova_simple(ws):
    """
    F-ratio = MS_between / MS_within sulle feature standardizzate.
    R^2 = SS_between / SS_total.
    Non usa permutazioni (N troppo grande) - e' solo il F-ratio osservato.
    """
    feat_cols = [c for c in ws.columns
                 if c.endswith("_mean") and not c.startswith("fuel")]
    feat_cols = [c for c in feat_cols if ws[c].notna().all()]
    if not feat_cols:
        return None

    X = StandardScaler().fit_transform(ws[feat_cols].values)
    labels = ws["regime"].values
    regimes = sorted(np.unique(labels))
    K = len(regimes)
    N = len(X)

    grand_mean = X.mean(axis=0)
    SS_between = sum(
        (labels == r).sum() * np.sum((X[labels == r].mean(axis=0) - grand_mean)**2)
        for r in regimes
    )
    SS_within = sum(
        np.sum((X[labels == r] - X[labels == r].mean(axis=0))**2)
        for r in regimes
    )
    SS_total = SS_between + SS_within

    MS_between = SS_between / (K - 1)
    MS_within  = SS_within  / (N - K)
    F = MS_between / MS_within if MS_within > 0 else np.inf
    R2 = SS_between / SS_total if SS_total > 0 else 0.0
    p_approx = stats.f.sf(F, K - 1, N - K)

    return dict(F=F, R2=R2, p_value=p_approx,
                SS_between=SS_between, SS_within=SS_within,
                K=K, N=N, features=feat_cols)


# =============================================================================
# PLOT
# =============================================================================

def plot_economic_profiles(ws, regimes_list):
    """01 - Boxplot LMP e load per regime."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = regime_colors(len(regimes_list))

    # LMP
    ax = axes[0]
    data_lmp = [ws.loc[ws["regime"] == r, "lmp_mean"].dropna().values
                for r in regimes_list]
    bp = ax.boxplot(data_lmp, patch_artist=True, medianprops={"color": "black", "lw": 1.5},
                    whiskerprops={"lw": 0.8}, flierprops={"ms": 2, "alpha": 0.4})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f"R{r}" for r in regimes_list], fontsize=8)
    ax.set_ylabel("LMP medio ($/MWh)")
    ax.set_title("LMP per regime")

    # Load
    ax2 = axes[1]
    if "total_mw_mean" in ws.columns:
        data_load = [ws.loc[ws["regime"] == r, "total_mw_mean"].dropna().values
                     for r in regimes_list]
        bp2 = ax2.boxplot(data_load, patch_artist=True,
                          medianprops={"color": "black", "lw": 1.5},
                          whiskerprops={"lw": 0.8}, flierprops={"ms": 2, "alpha": 0.4})
        for patch, color in zip(bp2["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_xticklabels([f"R{r}" for r in regimes_list], fontsize=8)
        ax2.set_ylabel("Load medio (MW)")
        ax2.set_title("Load per regime")
    else:
        ax2.text(0.5, 0.5, "total_mw non disponibile", ha="center", va="center",
                 transform=ax2.transAxes)

    fig.suptitle("Profili economici per regime", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_economic_profiles.png", dpi=DPI)
    plt.close(fig)
    print("  01_economic_profiles.png", flush=True)


def plot_fuel_mix(ws, regimes_list):
    """02 - Fuel mix medio per regime (stacked bar)."""
    fuel_cols = [f"fuel_{f}" for f in FUEL_NAMES if f"fuel_{f}" in ws.columns]
    if not fuel_cols:
        print("  02_fuel_mix.png: fuel columns non disponibili, skip", flush=True)
        return

    fuel_means = ws.groupby("regime")[fuel_cols].mean().loc[regimes_list]

    FUEL_COLORS = ["#e07b39", "#7b52ab", "#4da6e8", "#6abf69",
                   "#f7c948", "#5a5a5a", "#b04040", "#aaaaaa"]

    fig, ax = plt.subplots(figsize=(max(8, len(regimes_list) * 0.7), 5))
    bottom = np.zeros(len(regimes_list))
    x = np.arange(len(regimes_list))

    for i, (col, color) in enumerate(zip(fuel_cols, FUEL_COLORS)):
        vals = fuel_means[col].values
        ax.bar(x, vals, bottom=bottom, color=color, alpha=0.85,
               label=FUEL_NAMES[i], edgecolor="white", lw=0.3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"R{r}" for r in regimes_list], fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Quota fuel mix")
    ax.set_title("Fuel mix medio per regime")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_fuel_mix.png", dpi=DPI)
    plt.close(fig)
    print("  02_fuel_mix.png", flush=True)


def plot_kruskal(kw_df):
    """03 - H statistic e eta^2 per feature."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.barh(kw_df["feature"], kw_df["H"], color="#4477AA", alpha=0.8)
    ax.set_xlabel("H statistic (Kruskal-Wallis)")
    ax.set_title("Separazione statistica per feature")

    ax2 = axes[1]
    bars = ax2.barh(kw_df["feature"], kw_df["eta2"], color="#EE6677", alpha=0.8)
    ax2.set_xlabel("eta^2 (varianza spiegata dal regime)")
    ax2.set_title("Effect size (eta^2)")
    # linee riferimento
    for threshold, label, ls in [(0.01, "piccolo", ":"), (0.06, "medio", "--"), (0.14, "grande", "-")]:
        ax2.axvline(threshold, color="gray", lw=1, ls=ls, label=label)
    ax2.legend(fontsize=7)

    # p-value come annotazione
    for i, (_, row) in enumerate(kw_df.iterrows()):
        p_str = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01
                                                        else ("*" if row["p_value"] < 0.05 else "ns"))
        ax.text(row["H"] * 1.01, i, p_str, va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_kruskal_wallis.png", dpi=DPI)
    plt.close(fig)
    print("  03_kruskal_wallis.png", flush=True)


def plot_dunn(dunn_mat):
    """04 - Heatmap p-value Dunn pairwise (LMP)."""
    if dunn_mat is None:
        return
    K = len(dunn_mat)
    fig, ax = plt.subplots(figsize=(max(5, K * 0.5), max(4, K * 0.45)))

    # log10(p) per visualizzare meglio
    with np.errstate(divide="ignore"):
        log_p = -np.log10(dunn_mat.values.astype(float))
    log_p[np.isnan(log_p)] = 0

    im = ax.imshow(log_p, cmap="Reds", vmin=0, vmax=max(3, log_p[~np.isnan(log_p)].max()))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("-log10(p) Bonferroni")

    regs = list(dunn_mat.index)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"R{r}" for r in regs], fontsize=7, rotation=45)
    ax.set_yticklabels([f"R{r}" for r in regs], fontsize=7)
    ax.set_title("Dunn test pairwise (LMP) — piu' scuro = piu' significativo")

    # soglie
    for i in range(K):
        for j in range(K):
            if i != j:
                p = dunn_mat.iloc[i, j]
                if not np.isnan(p):
                    marker = "***" if p < 0.001 else ("**" if p < 0.01
                              else ("*" if p < 0.05 else ""))
                    if marker:
                        ax.text(j, i, marker, ha="center", va="center",
                                fontsize=6, color="white")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_dunn_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  04_dunn_heatmap.png", flush=True)


def plot_persistence(runs_df, pers_df, regimes_list):
    """05 - Distribuzione durata run per regime."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = regime_colors(len(regimes_list))

    # boxplot durata
    ax = axes[0]
    data = [runs_df.loc[runs_df["regime"] == r, "length_hours"].values
            for r in regimes_list]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "black", "lw": 1.5},
                    whiskerprops={"lw": 0.8}, flierprops={"ms": 2, "alpha": 0.4},
                    showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f"R{r}" for r in regimes_list], fontsize=8)
    ax.set_ylabel("Durata run (ore)")
    ax.set_title("Persistenza temporale per regime")

    # self-transition probability
    ax2 = axes[1]
    # dalla matrice di transizione
    ax2.set_title("Durata mediana run (ore)")
    ax2.bar(range(len(regimes_list)),
            [pers_df.loc[r, "med_hours"] if r in pers_df.index else 0
             for r in regimes_list],
            color=colors, alpha=0.8, edgecolor="white")
    ax2.set_xticks(range(len(regimes_list)))
    ax2.set_xticklabels([f"R{r}" for r in regimes_list], fontsize=8)
    ax2.set_ylabel("Ore")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_persistence.png", dpi=DPI)
    plt.close(fig)
    print("  05_persistence.png", flush=True)


def plot_transition_matrix(trans_df):
    """06 - Heatmap matrice di transizione."""
    K = len(trans_df)
    fig, ax = plt.subplots(figsize=(max(5, K * 0.5), max(4, K * 0.45)))

    im = ax.imshow(trans_df.values, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="P(j|i)")

    regs = list(trans_df.index)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"R{r}" for r in regs], fontsize=7, rotation=45)
    ax.set_yticklabels([f"R{r}" for r in regs], fontsize=7)
    ax.set_xlabel("Regime successivo j")
    ax.set_ylabel("Regime corrente i")
    ax.set_title("Matrice di transizione P[i -> j]")

    # annotazioni per valori alti
    for i in range(K):
        for j in range(K):
            v = trans_df.iloc[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if v < 0.6 else "white")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_transition_matrix.png", dpi=DPI)
    plt.close(fig)
    print("  06_transition_matrix.png", flush=True)


def plot_seasonal_heatmap(contingency, seasonal_res):
    """07 - Heatmap distribuzione stagionale per regime."""
    ct = contingency.div(contingency.sum(axis=1), axis=0)
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    ct = ct.reindex(columns=[s for s in season_order if s in ct.columns])

    K = len(ct)
    fig, ax = plt.subplots(figsize=(6, max(4, K * 0.4)))
    im = ax.imshow(ct.values, cmap="YlOrRd", vmin=0, vmax=ct.values.max(), aspect="auto")
    fig.colorbar(im, ax=ax, label="Frazione finestre")

    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, fontsize=9)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"R{r}" for r in ct.index], fontsize=8)
    ax.set_title(
        f"Distribuzione stagionale per regime\n"
        f"chi2={seasonal_res['chi2']:.1f}  p={seasonal_res['p_value']:.2e}  "
        f"V={seasonal_res['cramer_v']:.3f}"
    )

    for i in range(K):
        for j in range(len(ct.columns)):
            v = ct.iloc[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=7, color="black" if v < 0.5 else "white")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_seasonal_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  07_seasonal_heatmap.png", flush=True)


# =============================================================================
# REPORT TESTUALE
# =============================================================================

def write_report(kw_df, seasonal_res, perm_res, pers_df, regimes_list, ws):
    lines = []
    lines.append("=" * 65)
    lines.append("VALIDATION REPORT — Regime Detection (descrittivo)")
    lines.append(f"Labels: {LABELS_PATH}")
    lines.append(f"K = {len(regimes_list)}  finestre = {len(ws)}")
    lines.append("=" * 65)

    lines.append("\n--- 1. SEPARAZIONE STATISTICA (Kruskal-Wallis) ---")
    lines.append(f"{'Feature':<20} {'H':>8} {'p-value':>12} {'eta^2':>8} {'sig':>5}")
    lines.append("-" * 55)
    for _, row in kw_df.iterrows():
        sig = ("***" if row["p_value"] < 0.001 else
               ("**" if row["p_value"] < 0.01 else
                ("*"  if row["p_value"] < 0.05 else "ns")))
        lines.append(f"{row['feature']:<20} {row['H']:>8.1f} {row['p_value']:>12.2e} "
                     f"{row['eta2']:>8.4f} {sig:>5}")

    lines.append("\n--- 2. COERENZA STAGIONALE ---")
    lines.append(f"  chi^2 = {seasonal_res['chi2']:.2f}  "
                 f"p = {seasonal_res['p_value']:.2e}  "
                 f"V di Cramer = {seasonal_res['cramer_v']:.3f}")
    lines.append("  (V > 0.1 = effetto piccolo, > 0.3 = medio, > 0.5 = grande)")

    lines.append("\n--- 3. PERMANOVA (F-ratio osservato) ---")
    if perm_res:
        lines.append(f"  F = {perm_res['F']:.2f}  R^2 = {perm_res['R2']:.4f}  "
                     f"p ~ {perm_res['p_value']:.2e}")
        lines.append(f"  Feature usate: {', '.join(perm_res['features'])}")
    else:
        lines.append("  Non disponibile")

    lines.append("\n--- 4. PERSISTENZA TEMPORALE ---")
    lines.append(f"{'Regime':<10} {'n_runs':>8} {'avg_h':>8} {'med_h':>8} {'std_w':>8}")
    lines.append("-" * 45)
    for r in regimes_list:
        if r in pers_df.index:
            row = pers_df.loc[r]
            lines.append(f"R{r:<9} {int(row['n_runs']):>8} {row['avg_hours']:>8.1f} "
                         f"{row['med_hours']:>8.1f} {row['std_windows']:>8.2f}")

    lines.append("\n--- 5. PROFILO LMP PER REGIME ---")
    if "lmp_mean" in ws.columns:
        lmp_stats = ws.groupby("regime")["lmp_mean"].agg(["mean", "median", "std"])
        lines.append(f"{'Regime':<10} {'mean':>8} {'median':>8} {'std':>8}")
        lines.append("-" * 36)
        for r in regimes_list:
            if r in lmp_stats.index:
                row = lmp_stats.loc[r]
                lines.append(f"R{r:<9} {row['mean']:>8.2f} {row['median']:>8.2f} "
                             f"{row['std']:>8.2f}")

    lines.append("\n" + "=" * 65)

    report_path = OUT_DIR / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
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

    lab_df, pre_df = load_data()
    ws = build_window_stats(lab_df, pre_df)

    if len(ws) == 0:
        print("ERRORE: nessuna finestra valida trovata.", flush=True)
        return

    regimes_list = sorted(ws["regime"].unique())
    K = len(regimes_list)
    print(f"\nK = {K}  regimi: {regimes_list[:10]}{'...' if K > 10 else ''}", flush=True)

    # Salva statistiche per finestra
    ws.to_csv(OUT_DIR / "validation_stats.csv", index=False)

    # Test
    print("\nKruskal-Wallis...", flush=True)
    kw_df = kruskal_wallis(ws)

    print("Dunn test (LMP)...", flush=True)
    dunn_mat = dunn_test(ws, feature_col="lmp_mean")

    print("Persistenza temporale...", flush=True)
    pers_df, trans_df, runs_df = temporal_persistence(lab_df)

    print("Coerenza stagionale...", flush=True)
    seas_res = seasonal_coherence(ws)

    print("PERMANOVA...", flush=True)
    perm_res = permanova_simple(ws)

    # Plots
    print("\nProduco figure...", flush=True)
    plot_economic_profiles(ws, regimes_list)
    plot_fuel_mix(ws, regimes_list)
    plot_kruskal(kw_df)
    plot_dunn(dunn_mat)
    plot_persistence(runs_df, pers_df, regimes_list)
    plot_transition_matrix(trans_df)
    plot_seasonal_heatmap(seas_res["contingency"], seas_res)

    # Report
    print("\nReport...", flush=True)
    write_report(kw_df, seas_res, perm_res, pers_df, regimes_list, ws)

    print(f"\nDone. Output in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
