"""
step03c_diagnostics.py
======================
NEPOOL Regime Detection — Step 03c: Regime Diagnostics

Due test di validazione post-clustering:

1. MUTUAL INFORMATION TEST
   Valuta se aggiungere gas price (Algonquin) e/o temperatura
   aumenterebbe la separazione tra regimi.
   Usa MI normalizzata (NMI) tra feature candidata e regime label.
   Input: regimes.parquet + preprocessed.parquet
   Output: results/mi_test.csv + plot

2. PERSISTENCE TEST
   Verifica che i regimi siano temporalmente stabili e non troppo
   "nervosi" (switching frequente).
   Metriche:
     - Durata media per regime (ore)
     - Autocorrelazione della sequenza di regimi (lag 1-24h)
     - Matrice di transizione normalizzata
     - Persistence score: P(regime_t = regime_{t+1})
   Input: regimes.parquet
   Output: results/persistence_test.csv + plot

Output plots: results/step03c/
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
import config as C

PLOT_DIR    = Path(C.RESULTS_DIR) / "step03c"
RESULTS_DIR = Path(C.RESULTS_DIR)


# ═══════════════════════════════════════════════════════════════════════════
#  1. MUTUAL INFORMATION TEST
# ═══════════════════════════════════════════════════════════════════════════

def mutual_information_test(pre: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la Normalized Mutual Information (NMI) tra ogni feature
    candidata e i regime labels.

    NMI = 0  → nessuna informazione condivisa con i regimi
    NMI = 1  → feature determina completamente il regime

    Feature testate:
      - arcsinh_lmp, log_return, total_mw, ilr_1…ilr_7  (già nel modello)
      - lmp                                               (già nel modello, raw)
      - hour_of_day, day_of_week, month                  (temporali implicite)

    Feature candidate ESTERNE (da aggiungere se MI alta):
      - gas_price_proxy: non disponibile nel dataset, NMI stimata su correlati
        Come proxy usiamo ilr_3 (Gas vs Coal+Oil) che cattura la pressione
        del gas sul dispatch — in assenza del dato Algonquin.

    Nota: il dato Algonquin Gas Price e la temperatura non sono nel dataset
    corrente. Questo test usa le feature disponibili per stimare il valore
    informativo aggiuntivo di variabili temporali esplicite.
    """
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.preprocessing import KBinsDiscretizer

    # Merge su datetime
    df = pre.merge(regimes, on="datetime", how="inner")
    df = df[df["regime"] >= 0]   # escludi noise
    labels = df["regime"].values

    # Feature da testare
    feature_groups = {
        "In modello": [
            "arcsinh_lmp", "log_return", "total_mw",
            "ilr_1", "ilr_2", "ilr_3", "ilr_4", "ilr_5", "ilr_6", "ilr_7",
        ],
        "Temporali (non in modello)": [],
    }

    # Aggiungi feature temporali
    df["hour_of_day"]  = pd.to_datetime(df["datetime"]).dt.hour
    df["day_of_week"]  = pd.to_datetime(df["datetime"]).dt.dayofweek
    df["month"]        = pd.to_datetime(df["datetime"]).dt.month
    feature_groups["Temporali (non in modello)"] = ["hour_of_day", "day_of_week", "month"]

    results = []
    kbd = KBinsDiscretizer(n_bins=20, encode="ordinal", strategy="quantile")

    for group, features in feature_groups.items():
        for feat in features:
            if feat not in df.columns:
                continue
            x = df[feat].values.reshape(-1, 1)
            try:
                x_disc = kbd.fit_transform(x).ravel().astype(int)
                nmi = normalized_mutual_info_score(labels, x_disc, average_method="arithmetic")
            except Exception:
                nmi = float("nan")
            results.append({"feature": feat, "group": group, "NMI": nmi})

    return pd.DataFrame(results).sort_values("NMI", ascending=False).reset_index(drop=True)


def plot_mi(mi_df: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2c6fad" if g == "In modello" else "#e07b39"
              for g in mi_df["group"]]
    bars = ax.barh(mi_df["feature"], mi_df["NMI"], color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(fc="#2c6fad", alpha=0.85, label="Già nel modello"),
        Patch(fc="#e07b39", alpha=0.85, label="Non nel modello (temporali)"),
    ]
    ax.legend(handles=legend_els, fontsize=9)
    ax.set_xlabel("Normalized Mutual Information (NMI)")
    ax.set_title("Mutual Information tra feature e regime labels\n"
                 "NMI alto → feature informativa per la separazione dei regimi",
                 fontweight="bold")
    ax.set_xlim(0, min(1.05, mi_df["NMI"].max() * 1.2))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_mutual_information.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  [plot] 01_mutual_information.png")


# ═══════════════════════════════════════════════════════════════════════════
#  2. PERSISTENCE TEST
# ═══════════════════════════════════════════════════════════════════════════

def persistence_test(regimes: pd.DataFrame) -> dict:
    """
    Valuta la stabilità temporale dei regimi.

    Metriche:
      persistence_score  : P(regime_t == regime_{t+1}) per ogni regime
      mean_duration_h    : durata media consecutiva in ore per regime
      acf_lag1           : autocorrelazione lag-1 della sequenza di regimi
      transition_matrix  : P(regime_j | regime_i) normalizzata per riga
    """
    df = regimes.sort_values("datetime").reset_index(drop=True)
    labels = df["regime"].values
    reg_ids = sorted(set(labels[labels >= 0]))

    # ── Persistence score per regime ─────────────────────────────────────
    persistence = {}
    for r in reg_ids:
        mask = labels == r
        idx  = np.where(mask)[0]
        if len(idx) < 2:
            persistence[r] = float("nan")
            continue
        # P(next = r | current = r)
        next_labels = labels[np.minimum(idx + 1, len(labels) - 1)]
        persistence[r] = float((next_labels == r).mean())

    overall_persistence = float((labels[:-1] == labels[1:]).mean())

    # ── Durata media run consecutivi ─────────────────────────────────────
    mean_duration = {}
    for r in reg_ids:
        runs = []
        count = 0
        for lb in labels:
            if lb == r:
                count += 1
            elif count > 0:
                runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        mean_duration[r] = float(np.mean(runs)) if runs else 0.0

    # ── ACF lag-1 della sequenza regime ──────────────────────────────────
    # Usa solo finestre non-noise per evitare distorsioni
    clean = labels[labels >= 0].astype(float)
    if len(clean) > 2:
        acf_lag1 = float(np.corrcoef(clean[:-1], clean[1:])[0, 1])
    else:
        acf_lag1 = float("nan")

    # ── Matrice di transizione ────────────────────────────────────────────
    n_reg = len(reg_ids)
    r_map = {r: i for i, r in enumerate(reg_ids)}
    trans = np.zeros((n_reg, n_reg))
    for i in range(len(labels) - 1):
        a, b = labels[i], labels[i + 1]
        if a >= 0 and b >= 0:
            trans[r_map[a], r_map[b]] += 1

    row_sums = trans.sum(axis=1, keepdims=True)
    trans_norm = np.divide(trans, row_sums, where=row_sums > 0)

    return {
        "persistence_per_regime": persistence,
        "overall_persistence"   : overall_persistence,
        "mean_duration_h"       : mean_duration,
        "acf_lag1"              : acf_lag1,
        "transition_matrix"     : trans_norm,
        "regime_ids"            : reg_ids,
    }


def plot_persistence(res: dict) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)

    reg_ids  = res["regime_ids"]
    pers     = [res["persistence_per_regime"][r] for r in reg_ids]
    dur      = [res["mean_duration_h"][r] for r in reg_ids]
    palette  = sns.color_palette("tab10", len(reg_ids))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Persistence score ────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar([f"R{r}" for r in reg_ids], pers,
                  color=palette, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.axhline(res["overall_persistence"], color="red", lw=1.2, ls="--",
               label=f"Overall  {res['overall_persistence']:.3f}")
    ax.axhline(0.9, color="green", lw=0.8, ls=":", alpha=0.7,
               label="Target > 0.90")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(regime_t = regime_{t+1})")
    ax.set_title("Persistence score per regime", fontweight="bold")
    ax.legend(fontsize=8)

    # ── Durata media ─────────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar([f"R{r}" for r in reg_ids], dur,
                  color=palette, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f h", fontsize=8, padding=2)
    ax.set_ylabel("Ore consecutive medie")
    ax.set_title("Durata media regime (ore)", fontweight="bold")

    # ── Matrice di transizione ────────────────────────────────────────────
    ax = axes[2]
    tm = res["transition_matrix"]
    labels_str = [f"R{r}" for r in reg_ids]
    sns.heatmap(tm, ax=ax, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels_str, yticklabels=labels_str,
                linewidths=0.4, vmin=0, vmax=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matrice di transizione\nP(col | row)", fontweight="bold")
    ax.set_xlabel("Regime successivo")
    ax.set_ylabel("Regime corrente")

    fig.suptitle(
        f"Persistence Test  —  ACF lag-1: {res['acf_lag1']:.3f}  |  "
        f"Overall persistence: {res['overall_persistence']:.3f}",
        fontweight="bold", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_persistence.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  [plot] 02_persistence.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 03c: Diagnostics")
    print("═" * 65)
    print(f"  Input  : results/regimes.parquet")
    print(f"           results/preprocessed.parquet")
    print(f"  Output : results/step03c/")
    print("─" * 65)

    # ── Carica dati ──────────────────────────────────────────────────────
    regime_path = RESULTS_DIR / "regimes.parquet"
    pre_path    = RESULTS_DIR / "preprocessed.parquet"

    if not regime_path.exists():
        print(f"  ERRORE: {regime_path} non trovato. Esegui step03 prima.")
        sys.exit(1)
    if not pre_path.exists():
        print(f"  ERRORE: {pre_path} non trovato. Esegui step01 prima.")
        sys.exit(1)

    regimes = pd.read_parquet(regime_path)
    regimes["datetime"] = pd.to_datetime(regimes["datetime"])
    pre     = pd.read_parquet(pre_path)
    pre["datetime"] = pd.to_datetime(pre["datetime"])

    n_reg   = len(set(regimes["regime"].values)) - (1 if -1 in regimes["regime"].values else 0)
    noise_n = (regimes["regime"] == -1).sum()
    print(f"  Regimi: {n_reg}  |  Noise: {noise_n} ({noise_n/len(regimes):.1%})")
    print(f"  Finestre totali: {len(regimes):,}")
    print()

    # ── 1. Mutual Information Test ────────────────────────────────────────
    print("[1/2] Mutual Information Test ...")
    mi_df = mutual_information_test(pre, regimes)
    mi_path = RESULTS_DIR / "mi_test.csv"
    mi_df.to_csv(mi_path, index=False)
    plot_mi(mi_df)

    print(f"\n  NMI risultati (top 5):")
    for _, row in mi_df.head(5).iterrows():
        bar = "█" * int(row["NMI"] * 30)
        print(f"    {row['feature']:<18}  {row['NMI']:.3f}  {bar}")

    # Interpretazione
    print()
    temporal_nmi = mi_df[mi_df["group"] == "Temporali (non in modello)"]["NMI"].max()
    model_nmi    = mi_df[mi_df["group"] == "In modello"]["NMI"].max()
    if temporal_nmi > 0.3:
        print(f"  ⚠  Feature temporali esplicite hanno NMI alto ({temporal_nmi:.3f})")
        print(f"     Considera aggiunta di hour_of_day/month come canale.")
    else:
        print(f"  ✓  Feature temporali NMI basso ({temporal_nmi:.3f}) — Chronos-2")
        print(f"     le cattura già implicitamente dalla sequenza temporale.")
    print(f"  Salvato → {mi_path}")

    # ── 2. Persistence Test ───────────────────────────────────────────────
    print(f"\n[2/2] Persistence Test ...")
    res = persistence_test(regimes)
    plot_persistence(res)

    # Salva CSV
    pers_rows = []
    for r in res["regime_ids"]:
        pers_rows.append({
            "regime"          : r,
            "persistence"     : res["persistence_per_regime"][r],
            "mean_duration_h" : res["mean_duration_h"][r],
        })
    pers_df = pd.DataFrame(pers_rows)
    pers_path = RESULTS_DIR / "persistence_test.csv"
    pers_df.to_csv(pers_path, index=False)

    print(f"\n  Persistence per regime:")
    for _, row in pers_df.iterrows():
        ok = "✓" if row["persistence"] > 0.9 else "⚠"
        print(f"    {ok}  R{int(row['regime'])}  "
              f"persistence={row['persistence']:.3f}  "
              f"durata_media={row['mean_duration_h']:.1f}h")

    print(f"\n  Overall persistence : {res['overall_persistence']:.3f}")
    print(f"  ACF lag-1          : {res['acf_lag1']:.3f}")

    if res["overall_persistence"] < 0.85:
        print(f"\n  ⚠  Regimi instabili (persistence < 0.85).")
        print(f"     Aumenta min_cluster_size in step03 per ridurre lo switching.")
    else:
        print(f"\n  ✓  Regimi stabili (persistence ≥ 0.85).")

    print(f"\n  Salvato → {pers_path}")

    # ── Report ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "─" * 65)
    print(f"  Plots   → {PLOT_DIR}/")
    print(f"  Elapsed : {elapsed:.1f}s")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
