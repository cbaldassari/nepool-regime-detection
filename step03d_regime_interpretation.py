"""
step03d_regime_interpretation.py
=================================
Interpretazione economica verbose dei regimi prodotti da step03.

Architettura a due fasi
-----------------------
  FASE 1 — COMPUTE  (lenta, ~1-2 min)
    Espande ogni finestra di 720h dal preprocessed.parquet,
    inverte ILR → fuel shares, inverte arcsinh → LMP.
    Salva i risultati in:
      results/step03d/window_stats.parquet   (una riga per finestra)
      results/step03d/regime_stats.parquet   (una riga per regime)

    ► Salta automaticamente se i file esistono già.
    ► Forza il ricalcolo impostando FORCE_RECOMPUTE = True  (riga ~50)

  FASE 2 — PLOT  (veloce, sempre eseguita)
    Ogni grafico è una funzione indipendente.
    Puoi modificare una funzione e ri-eseguire lo script:
    la Fase 1 verrà saltata e vedrai subito il grafico aggiornato.
    Le funzioni plot sono in fondo al file, ciascuna ben separata.

Input  : results/regimes.parquet
         results/preprocessed.parquet
         results/umap.parquet
Output : results/step03d/window_stats.parquet   (dati finestra)
         results/step03d/regime_stats.parquet    (dati regime aggregati)
         results/step03d/interpretation.txt      (testo verbose)
         results/step03d/01_umap_scatter.png
         results/step03d/02_regime_timeline.png
         results/step03d/03_lmp_boxplot.png
         results/step03d/04_fuel_mix_bar.png
         results/step03d/05_ilr_radar.png
         results/step03d/06_monthly_heatmap.png
         results/step03d/07_lmp_vs_load.png
         results/step03d/08_fuel_detail.png

Run    : python step03d_regime_interpretation.py
"""

import sys
import math
import warnings
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates  as mdates
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  ▶▶  CONFIGURAZIONE  ◀◀
#  Queste sono le variabili che probabilmente vorrai modificare.
# ═══════════════════════════════════════════════════════════════════════════

# True  → ricalcola sempre la Fase 1 (espansione finestre)
# False → salta se window_stats.parquet esiste già  ← modalità normale
FORCE_RECOMPUTE = False

# Cartella di output
OUT_DIR = Path(C.RESULTS_DIR) / "step03d"

# Ore di contesto per finestra (deve corrispondere a step02)
CONTEXT_H = C.MEAN_REVERSION["context_len"]   # 720

# Ore "peak" per calcolo peak_fraction
PEAK_HOURS = set(range(7, 23))   # HE07–HE22

# Palette regimi (se hai più di 14 regimi, la lista si cicla)
REGIME_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#888888", "#1b9e77", "#d95f02",
    "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
]

# Stile matplotlib globale
MPL_STYLE   = "whitegrid"    # "whitegrid" | "darkgrid" | "white" | "ticks"
MPL_CONTEXT = "notebook"     # "paper" | "notebook" | "talk" | "poster"
DPI         = 150

# ═══════════════════════════════════════════════════════════════════════════
#  ILR inverse  (SBP identico a step01)
# ═══════════════════════════════════════════════════════════════════════════

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
FUEL_COLORS = ["#e07b39", "#7b52ab", "#4da6e8", "#6abf69",
               "#f7c948", "#5a5a5a", "#b04040", "#aaaaaa"]
ILR_COLS    = [f"ilr_{i}" for i in range(1, 8)]

ILR_SHORT = [
    "Disp/Var", "Foss/NonF", "Gas/Coal+Oil",
    "Coal/Oil", "Nuc/Oth",  "Hyd/Int",  "Sol/Wind",
]


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
    """(N,7) ILR coords → (N,8) fuel shares (rows sum to 1)."""
    Psi   = _ilr_basis(SBP)
    clr   = ilr @ Psi
    exp_c = np.exp(clr)
    return exp_c / exp_c.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Utilità
# ═══════════════════════════════════════════════════════════════════════════

SEASON_MAP = {12: "Winter", 1: "Winter",  2: "Winter",
              3:  "Spring", 4: "Spring",  5: "Spring",
              6:  "Summer", 7: "Summer",  8: "Summer",
              9:  "Autumn", 10: "Autumn", 11: "Autumn"}


def _regime_palette(regime_ids):
    return {rid: REGIME_PALETTE[i % len(REGIME_PALETTE)]
            for i, rid in enumerate(sorted(regime_ids))}


def _apply_style():
    sns.set_theme(style=MPL_STYLE, context=MPL_CONTEXT, font_scale=0.9)
    plt.rcParams.update({
        "figure.dpi"        : DPI,
        "savefig.dpi"       : DPI,
        "savefig.bbox"      : "tight",
        "savefig.facecolor" : "white",
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  FASE 1 — COMPUTE
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(regimes_df, pre_df):
    """
    Per ogni finestra in regimes_df, estrae le 720h di dati da pre_df
    e calcola media/std delle feature.  Restituisce window_stats DataFrame.

    Colonne prodotte per ogni feature X:
        X_mean, X_std

    Colonne aggiuntive:
        lmp_mean, lmp_p5, lmp_p95     ← sinh(arcsinh_lmp)
        fuel_Gas … fuel_Other          ← ILR inverse
        month_mode, peak_frac
    """
    FEAT = ["arcsinh_lmp", "log_return", "total_mw"] + ILR_COLS

    records = []
    for _, row in tqdm(regimes_df.iterrows(), total=len(regimes_df),
                       desc="  espansione finestre", ncols=80):
        wdt    = row["datetime"]
        regime = int(row["regime"])
        if regime < 0:
            continue

        start = wdt - pd.Timedelta(hours=CONTEXT_H - 1)
        win   = pre_df.loc[start:wdt, [c for c in FEAT if c in pre_df.columns]]

        if len(win) < CONTEXT_H // 4:
            continue

        rec = {"datetime": wdt, "regime": regime}
        for col in FEAT:
            if col in win.columns:
                rec[f"{col}_mean"] = float(win[col].mean())
                rec[f"{col}_std"]  = float(win[col].std())

        # LMP via sinh (valori tipici $/MWh)
        if "arcsinh_lmp_mean" in rec:
            lmp_vals = np.sinh(win["arcsinh_lmp"].values)
            rec["lmp_mean"] = float(lmp_vals.mean())
            rec["lmp_p5"]   = float(np.percentile(lmp_vals, 5))
            rec["lmp_p95"]  = float(np.percentile(lmp_vals, 95))
            rec["lmp_std"]  = float(lmp_vals.std())

        # Fuel shares via ILR inverse
        ilr_present = [c for c in ILR_COLS if c in win.columns]
        if len(ilr_present) == 7:
            ilr_mat  = win[ILR_COLS].values
            fuel_mat = ilr_inverse(ilr_mat)   # (T, 8)
            for j, fname in enumerate(FUEL_NAMES):
                rec[f"fuel_{fname}"] = float(fuel_mat[:, j].mean())

        # Temporale
        rec["month_mode"] = int(win.index.month.value_counts().idxmax())
        rec["peak_frac"]  = float(win.index.hour.isin(PEAK_HOURS).mean())

        records.append(rec)

    wstats = pd.DataFrame(records)
    print(f"  Finestre valide: {len(wstats):,} / {len(regimes_df):,}")
    return wstats


def aggregate_regimes(wstats):
    """Aggrega window_stats per regime → regime_stats DataFrame."""
    rows = []
    for rid, sub in wstats.groupby("regime"):
        row = {"regime": rid, "n_windows": len(sub)}
        for col in wstats.columns:
            if col in ("regime", "datetime"):
                continue
            if pd.api.types.is_numeric_dtype(wstats[col]):
                row[f"{col}_mean"] = sub[col].mean()
                row[f"{col}_std"]  = sub[col].std()
        row["dom_season"] = (sub["datetime"].dt.month
                             .map(SEASON_MAP)
                             .value_counts().index[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def load_or_compute():
    """Carica window_stats e regime_stats da disco, o li calcola."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ws_path = OUT_DIR / "window_stats.parquet"
    rs_path = OUT_DIR / "regime_stats.parquet"

    if not FORCE_RECOMPUTE and ws_path.exists() and rs_path.exists():
        print("  Statistiche già presenti → carico da disco  "
              "(FORCE_RECOMPUTE=False)")
        wstats = pd.read_parquet(ws_path)
        rstats = pd.read_parquet(rs_path)
        wstats["datetime"] = pd.to_datetime(wstats["datetime"])
        return wstats, rstats

    print("Caricamento dati …")
    regimes_df = pd.read_parquet(Path(C.RESULTS_DIR) / "regimes.parquet")
    regimes_df["datetime"] = pd.to_datetime(regimes_df["datetime"])

    pre_df = pd.read_parquet(Path(C.RESULTS_DIR) / "preprocessed.parquet")
    if "datetime" in pre_df.columns:
        pre_df = pre_df.set_index("datetime")
    pre_df.index = pd.to_datetime(pre_df.index)
    pre_df = pre_df.sort_index()

    print(f"  {len(regimes_df):,} finestre  |  {len(pre_df):,} ore")

    print("\nFase 1: espansione finestre e inversione trasformate …")
    wstats = compute_stats(regimes_df, pre_df)
    rstats = aggregate_regimes(wstats)

    wstats.to_parquet(ws_path, index=False)
    rstats.to_parquet(rs_path, index=False)
    print(f"  Salvato → {ws_path}")
    print(f"  Salvato → {rs_path}")

    return wstats, rstats


# ═══════════════════════════════════════════════════════════════════════════
#  INTERPRETAZIONE VERBOSA
# ═══════════════════════════════════════════════════════════════════════════

def _lmp_label(lmp_mean):
    if lmp_mean < 10:   return "quasi-zero / negativo (surplus rinnovabili)"
    if lmp_mean < 30:   return "basso (off-peak)"
    if lmp_mean < 60:   return "moderato (normale)"
    if lmp_mean < 100:  return "elevato (stress domanda)"
    if lmp_mean < 200:  return "alto (scarsità)"
    return                     "estremo (emergenza / spike)"


def _economic_narrative(sub_mean):
    lmp   = sub_mean.get("lmp_mean", np.nan)
    gas   = sub_mean.get("fuel_Gas",   0) * 100
    oil   = sub_mean.get("fuel_Oil",   0) * 100
    nuc   = sub_mean.get("fuel_Nuclear", 0) * 100
    ren   = (sub_mean.get("fuel_Wind",  0) +
             sub_mean.get("fuel_Solar", 0) +
             sub_mean.get("fuel_Hydro", 0)) * 100

    if oil > 8 or lmp > 150:
        tag = "🔴 STRESS / SCARCITÀ"
        txt = (f"Prezzi estremi (${lmp:.0f}/MWh). Oil-peaker attivi ({oil:.1f}%) "
               f"— evento cold-weather o emergenza offerta. Regime tipicamente "
               f"invernale di breve durata ma ad alto impatto.")
    elif oil > 2 or lmp > 80:
        tag = "🟠 STRESS MODERATO"
        txt = (f"Prezzi elevati (${lmp:.0f}/MWh). Oil parzialmente attivo ({oil:.1f}%). "
               f"Gas domina al {gas:.0f}%. Probabile picco domanda estivo o "
               f"freddo invernale non estremo.")
    elif ren > 45 or lmp < 15:
        tag = "🟢 SURPLUS RINNOVABILI"
        txt = (f"Prezzi molto bassi o negativi (${lmp:.0f}/MWh). Forte penetrazione "
               f"rinnovabili ({ren:.0f}% wind+solar+hydro). Gas ridotto al minimo. "
               f"Tipico: primavera/notte vento, mezzogiorno solare.")
    elif ren > 28 and lmp < 45:
        tag = "🟡 MIX RINNOVABILE MODERATO"
        txt = (f"Prezzi sotto-norma (${lmp:.0f}/MWh), buona quota rinnovabili ({ren:.0f}%). "
               f"Gas ancora marginale ({gas:.0f}%). Ore di transizione o "
               f"meteo favorevole.")
    elif gas > 55:
        tag = "🔵 GAS MARGINALE (operatività normale)"
        txt = (f"Dispatch tipico ISONE: gas margina al {gas:.0f}%, nucleare baseload "
               f"({nuc:.0f}%). LMP normale (${lmp:.0f}/MWh). "
               f"Regime di \"business as usual\".")
    else:
        tag = "🟤 MISTO / TRANSIZIONE"
        txt = (f"LMP moderato (${lmp:.0f}/MWh), fuel mix diversificato: "
               f"gas {gas:.0f}%, rinn. {ren:.0f}%, nuc {nuc:.0f}%.")

    return tag, textwrap.fill(txt, width=68, subsequent_indent="    ")


def print_interpretation(wstats, rstats, out_txt=None):
    """Stampa la narrativa economica su stdout e (opzionale) su file .txt."""
    regime_ids = sorted(wstats["regime"].unique())
    total      = len(wstats)
    palette    = _regime_palette(regime_ids)

    lines = []
    SEP   = "═" * 72

    lines.append(SEP)
    lines.append("  ISONE — INTERPRETAZIONE REGIMI   "
                 f"({len(regime_ids)} regimi, {total:,} finestre)")
    lines.append(SEP)

    for rid in regime_ids:
        sub  = wstats[wstats["regime"] == rid]
        n    = len(sub)
        pct  = n / total * 100

        lmp_mean = sub["lmp_mean"].mean()
        lmp_p5   = sub["lmp_p5"].mean()
        lmp_p95  = sub["lmp_p95"].mean()
        load     = sub["total_mw_mean"].mean() if "total_mw_mean" in sub else np.nan
        peak_f   = sub["peak_frac"].mean()

        fuel_m = {f: sub[f"fuel_{f}"].mean() for f in FUEL_NAMES
                  if f"fuel_{f}" in sub}
        ren    = sum(fuel_m.get(f, 0) for f in ["Wind", "Solar", "Hydro"]) * 100

        ilr_m  = [sub[f"ilr_{k}_mean"].mean() for k in range(1, 8)
                  if f"ilr_{k}_mean" in sub]

        months = sub["datetime"].dt.month
        dom_season = months.map(SEASON_MAP).value_counts().index[0]
        month_pct  = months.map(SEASON_MAP).value_counts()
        season_str = "  ".join(f"{s}:{c}" for s, c in month_pct.items())

        tag, narrative = _economic_narrative(
            {"lmp_mean": lmp_mean, **{f"fuel_{f}": v for f, v in fuel_m.items()}}
        )

        lines.append(f"\n{'─' * 72}")
        lines.append(f"  REGIME {rid:>2d}   {tag}   "
                     f"({n:,} finestre, {pct:.1f}%)")
        lines.append(f"{'─' * 72}")
        lines.append(f"\n  PREZZO")
        lines.append(f"    Media  ${lmp_mean:>7.2f}/MWh   [P5 ${lmp_p5:.1f} – "
                     f"P95 ${lmp_p95:.1f}]    {_lmp_label(lmp_mean)}")
        lines.append(f"  CARICO")
        lines.append(f"    Media  {load:>10,.0f} MW   "
                     f"Peak-hour frac: {peak_f:.1%}")
        lines.append(f"  FUEL MIX (media finestre)")
        for fname, fval in sorted(fuel_m.items(), key=lambda x: -x[1]):
            bar = "█" * max(1, int(round(fval * 32)))
            lines.append(f"    {fname:>8s}  {fval*100:5.1f}%  {bar}")
        if ilr_m:
            lines.append(f"  ILR COORDS (media)")
            for lbl, val in zip(ILR_SHORT, ilr_m):
                mark = "▲" if val > 0.3 else ("▼" if val < -0.3 else "●")
                lines.append(f"    {lbl:<18s}  {val:+.3f}  {mark}")
        lines.append(f"  STAGIONE DOMINANTE: {dom_season}   [{season_str}]")
        lines.append(f"\n  NARRATIVA\n    {narrative}")

    # Tabella riassuntiva
    lines.append(f"\n\n{SEP}")
    lines.append("  TABELLA COMPARATIVA")
    lines.append(SEP)
    hdr = (f"{'R':>3}  {'N':>5}  {'%':>4}  "
           f"{'LMP_mean':>8}  {'LMP_p5':>7}  {'LMP_p95':>7}  "
           f"{'Gas%':>5}  {'Oil%':>5}  {'Ren%':>5}  {'Stagione':>8}")
    lines.append(f"\n  {hdr}")
    lines.append("  " + "─" * len(hdr))
    for rid in regime_ids:
        sub  = wstats[wstats["regime"] == rid]
        n    = len(sub)
        pct  = n / total * 100
        lmp  = sub["lmp_mean"].mean()
        lp5  = sub["lmp_p5"].mean()
        lp95 = sub["lmp_p95"].mean()
        fm   = {f: sub[f"fuel_{f}"].mean() for f in FUEL_NAMES if f"fuel_{f}" in sub}
        ren  = sum(fm.get(f, 0) for f in ["Wind", "Solar", "Hydro"]) * 100
        seas = sub["datetime"].dt.month.map(SEASON_MAP).value_counts().index[0]
        lines.append(
            f"  {rid:>3d}  {n:>5d}  {pct:>4.1f}  "
            f"{lmp:>8.2f}  {lp5:>7.2f}  {lp95:>7.2f}  "
            f"{fm.get('Gas',0)*100:>5.1f}  {fm.get('Oil',0)*100:>5.2f}  "
            f"{ren:>5.1f}  {seas:>8}"
        )

    text = "\n".join(lines)
    print(text)

    if out_txt:
        out_txt.write_text(text, encoding="utf-8")
        print(f"\n  Salvato → {out_txt}")


# ═══════════════════════════════════════════════════════════════════════════
#  FASE 2 — PLOT
#  ─────────────────────────────────────────────────────────────────────────
#  Ogni funzione è indipendente: riceve wstats (e a volte altri df),
#  genera UN file PNG e lo salva in OUT_DIR.
#  Per modificare un grafico: edita la funzione e ri-esegui lo script
#  → Fase 1 verrà saltata, solo i PNG vengono rigenerati.
# ═══════════════════════════════════════════════════════════════════════════

def plot_umap_scatter(wstats):
    """
    01 — UMAP 2D colorato per regime.
    Carica umap.parquet direttamente (non dipende da wstats per i punti).
    """
    umap_df = pd.read_parquet(Path(C.RESULTS_DIR) / "umap.parquet")
    umap_df["datetime"] = pd.to_datetime(umap_df["datetime"])
    regimes_df = pd.read_parquet(Path(C.RESULTS_DIR) / "regimes.parquet")
    regimes_df["datetime"] = pd.to_datetime(regimes_df["datetime"])
    merged = umap_df.merge(regimes_df, on="datetime", how="left")

    regime_ids = sorted(r for r in merged["regime"].unique() if r >= 0)
    palette    = _regime_palette(regime_ids)

    fig, ax = plt.subplots(figsize=(8, 7))
    # noise
    noise = merged[merged["regime"] == -1]
    if len(noise):
        ax.scatter(noise["umap_1"], noise["umap_2"],
                   c="#cccccc", s=3, alpha=0.3, linewidths=0, label="Noise")
    # regimi
    for rid in regime_ids:
        m = merged["regime"] == rid
        ax.scatter(merged.loc[m, "umap_1"], merged.loc[m, "umap_2"],
                   c=palette[rid], s=5, alpha=0.55, linewidths=0,
                   label=f"R{rid} (n={(m).sum()})")
    ax.legend(markerscale=3, fontsize=7, loc="best",
              ncol=max(1, len(regime_ids) // 7), framealpha=0.7)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    n_noise = (merged["regime"] == -1).sum()
    noise_pct = n_noise / len(merged) * 100
    ax.set_title(f"UMAP 2D — {len(regime_ids)} regimi  "
                 f"(noise = {noise_pct:.1f}%)",
                 fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "01_umap_scatter.png"
    fig.savefig(out); plt.close(fig)
    print(f"  [plot] {out.name}")


def plot_regime_timeline(wstats):
    """
    02 — Timeline: un punto per finestra, y = regime.
    Top: stacked bar quota annua per regime.
    Bottom: scatter temporale.
    """
    regime_ids = sorted(wstats["regime"].unique())
    palette    = _regime_palette(regime_ids)
    ts         = pd.to_datetime(wstats["datetime"])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(15, 6),
        gridspec_kw={"height_ratios": [1, 2.5]}
    )
    # --- Top: stacked bar per anno ---
    years = sorted(ts.dt.year.unique())
    bottoms = np.zeros(len(years))
    for rid in regime_ids:
        fracs = []
        for yr in years:
            m_yr = ts.dt.year == yr
            tot  = m_yr.sum()
            fracs.append((m_yr & (wstats["regime"] == rid)).sum() / tot
                         if tot else 0)
        ax_top.bar(years, fracs, bottom=bottoms,
                   color=palette[rid], width=0.75, label=f"R{rid}")
        bottoms += np.array(fracs)
    ax_top.set_ylabel("Frazione", fontsize=8)
    ax_top.set_title("Quota annua per regime", fontsize=9, fontweight="bold")
    ax_top.legend(ncol=len(regime_ids), fontsize=7, loc="upper left",
                  framealpha=0.6)
    ax_top.set_xlim(years[0] - 0.5, years[-1] + 0.5)

    # --- Bottom: scatter timeline ---
    for rid in regime_ids:
        m = wstats["regime"] == rid
        ax_bot.scatter(ts[m], np.full(m.sum(), rid),
                       c=palette[rid], s=3, alpha=0.5, linewidths=0)
    ax_bot.set_yticks(regime_ids)
    ax_bot.set_yticklabels([f"R{r}" for r in regime_ids], fontsize=8)
    ax_bot.set_xlabel("Data"); ax_bot.set_ylabel("Regime")
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_bot.xaxis.set_major_locator(mdates.YearLocator())

    fig.suptitle("Regime nel tempo", fontweight="bold", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "02_regime_timeline.png"
    fig.savefig(out); plt.close(fig)
    print(f"  [plot] {out.name}")


def plot_lmp_boxplot(wstats):
    """
    03 — Box-plot LMP per regime.
    Ogni box = distribuzione delle lmp_mean delle finestre in quel regime.
    """
    regime_ids = sorted(wstats["regime"].unique())
    palette    = _regime_palette(regime_ids)

    data_boxes = [wstats.loc[wstats["regime"] == rid, "lmp_mean"].dropna().values
                  for rid in regime_ids]

    fig, ax = plt.subplots(figsize=(max(8, len(regime_ids) * 0.9), 5))
    bp = ax.boxplot(data_boxes, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, rid in zip(bp["boxes"], regime_ids):
        patch.set_facecolor(palette[rid])
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(regime_ids) + 1))
    ax.set_xticklabels([f"R{r}" for r in regime_ids])
    ax.set_ylabel("LMP medio di finestra  ($/MWh)")
    ax.set_title("Distribuzione LMP per regime", fontweight="bold")
    ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.5)
    fig.tight_layout()
    out = OUT_DIR / "03_lmp_boxplot.png"
    fig.savefig(out); plt.close(fig)
    print(f"  [plot] {out.name}")


def plot_fuel_mix_bar(wstats):
    """
    04 — Stacked bar: fuel mix medio per regime.
    """
    regime_ids = sorted(wstats["regime"].unique())

    fuel_matrix = np.zeros((len(regime_ids), len(FUEL_NAMES)))
    for i, rid in enumerate(regime_ids):
        sub = wstats[wstats["regime"] == rid]
        for j, fname in enumerate(FUEL_NAMES):
            col = f"fuel_{fname}"
            if col in sub:
                fuel_matrix[i, j] = sub[col].mean()

    fig, ax = plt.subplots(figsize=(max(9, len(regime_ids) * 0.9), 5))
    x      = np.arange(len(regime_ids))
    bottom = np.zeros(len(regime_ids))
    for j, (fname, fcolor) in enumerate(zip(FUEL_NAMES, FUEL_COLORS)):
        ax.bar(x, fuel_matrix[:, j] * 100, bottom=bottom * 100,
               color=fcolor, label=fname, width=0.72,
               edgecolor="white", linewidth=0.3)
        bottom += fuel_matrix[:, j]
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{r}" for r in regime_ids])
    ax.set_ylabel("Share (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Fuel mix medio per regime", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.8)
    fig.tight_layout()
    out = OUT_DIR / "04_fuel_mix_bar.png"
    fig.savefig(out); plt.close(fig)
    print(f"  [plot] {out.name}")


def plot_ilr_radar(wstats):
    """
    05 — Radar chart ILR per regime (un pannello per regime).
    """
    regime_ids = sorted(wstats["regime"].unique())
    palette    = _regime_palette(regime_ids)
    n_ilr      = 7
    angles     = np.linspace(0, 2 * np.pi, n_ilr, endpoint=False).tolist()
    angles    += angles[:1]

    ncols = min(4, len(regime_ids))
    nrows = math.ceil(len(regime_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 3.2),
                              subplot_kw=dict(polar=True))
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat:
        ax.set_visible(False)

    for i, rid in enumerate(regime_ids):
        sub  = wstats[wstats["regime"] == rid]
        vals = [sub[f"ilr_{k}_mean"].mean() if f"ilr_{k}_mean" in sub else 0.0
                for k in range(1, 8)]
        vals += vals[:1]
        ax = axes_flat[i]
        ax.set_visible(True)
        ax.plot(angles, vals, color=palette[rid], linewidth=1.8)
        ax.fill(angles, vals, color=palette[rid], alpha=0.22)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(ILR_SHORT, size=6)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"R{rid}  (n={len(sub)})", size=8,
                     fontweight="bold", pad=10)

    fig.suptitle("Coordinate ILR per regime (radar)",
                 fontweight="bold", fontsize=11, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "05_ilr_radar.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] {out.name}")


def plot_monthly_heatmap(wstats):
    """
    06 — Heatmap: prevalenza regime per mese di calendario.
    Ogni cella = frazione delle finestre di quel mese assegnate al regime.
    """
    regime_ids = sorted(wstats["regime"].unique())
    month_counts = np.zeros((len(regime_ids), 12), dtype=int)
    for i, rid in enumerate(regime_ids):
        mc = pd.to_datetime(wstats.loc[wstats["regime"] == rid, "datetime"]).dt.month.value_counts()
        for m in range(1, 13):
            month_counts[i, m - 1] = int(mc.get(m, 0))
    # normalizza per colonna (mese)
    col_sums = month_counts.sum(axis=0, keepdims=True) + 1e-9
    norm     = month_counts / col_sums

    fig, ax = plt.subplots(figsize=(13, max(3.5, len(regime_ids) * 0.55 + 1.5)))
    im = ax.imshow(norm, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=norm.max())
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Gen","Feb","Mar","Apr","Mag","Giu",
                         "Lug","Ago","Set","Ott","Nov","Dic"])
    ax.set_yticks(range(len(regime_ids)))
    ax.set_yticklabels([f"R{r}" for r in regime_ids])
    ax.set_title("Prevalenza regime per mese  (frazione finestre del mese)",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, label="Frazione", shrink=0.85)
    for i in range(len(regime_ids)):
        for j in range(12):
            v = norm[i, j]
            if v > 0.04:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=7,
                        color="black" if v < 0.55 else "white")
    fig.tight_layout()
    out = OUT_DIR / "06_monthly_heatmap.png"
    fig.savefig(out); plt.close(fig)
    print(f"  [plot] {out.name}")


def plot_lmp_vs_load(wstats):
    """
    07 — Scatter LMP medio vs carico medio per regime.
    Bolla: dimensione ∝ numero finestre.  Colore: regime.
    """
    regime_ids = sorted(wstats["regime"].unique())
    palette    = _regime_palette(regime_ids)

    fig, ax = plt.subplots(figsize=(8, 6))
    for rid in regime_ids:
        sub  = wstats[wstats["regime"] == rid]
        lmp  = sub["lmp_mean"].mean()
        load = sub["total_mw_mean"].mean() if "total_mw_mean" in sub else np.nan
        size = len(sub) / 8
        ax.scatter(load, lmp, c=palette[rid], s=size,
                   alpha=0.85, linewidths=0.6, edgecolors="black", zorder=3)
        ax.annotate(f"R{rid}", (load, lmp),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel("Carico medio finestra (MW)")
    ax.set_ylabel("LMP medio finestra ($/MWh)")
    ax.set_title("LMP vs Carico per regime  (bolla ∝ n finestre)",
                 fontweight="bold")
    ax.axhline(0, color="red", lw=0.7, ls="--", alpha=0.4)
    fig.tight_layout()
    out = OUT_DIR / "07_lmp_vs_load.png"
    fig.savefig(out); plt.close(fig)
    print(f"  [plot] {out.name}")


def plot_fuel_detail(wstats):
    """
    08 — Fuel detail: box-plot separato per Gas, Oil, Rinnovabili.
    Utile per identificare regimi stress (oil elevato) e surplus (ren elevato).
    """
    regime_ids = sorted(wstats["regime"].unique())
    palette    = _regime_palette(regime_ids)
    xlabels    = [f"R{r}" for r in regime_ids]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    specs = [
        ("fuel_Gas",   "Gas share (%)",          "#e07b39"),
        ("fuel_Oil",   "Oil share (%)",           "#b04040"),
    ]
    # terza colonna: rinnovabili aggregate
    for ax, (col, ylabel, color) in zip(axes[:2], specs):
        data = [wstats.loc[wstats["regime"] == rid, col].dropna().values * 100
                for rid in regime_ids]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", lw=1.4),
                        flierprops=dict(marker=".", ms=2, alpha=0.3))
        for patch in bp["boxes"]:
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(regime_ids) + 1))
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Rinnovabili = Wind + Solar + Hydro
    ax = axes[2]
    ren_col = "ren_share"
    wstats_tmp = wstats.copy()
    wstats_tmp[ren_col] = (
        wstats_tmp.get("fuel_Wind",  pd.Series(0, index=wstats_tmp.index)).fillna(0) +
        wstats_tmp.get("fuel_Solar", pd.Series(0, index=wstats_tmp.index)).fillna(0) +
        wstats_tmp.get("fuel_Hydro", pd.Series(0, index=wstats_tmp.index)).fillna(0)
    )
    data = [wstats_tmp.loc[wstats_tmp["regime"] == rid, ren_col].values * 100
            for rid in regime_ids]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", lw=1.4),
                    flierprops=dict(marker=".", ms=2, alpha=0.3))
    for patch in bp["boxes"]:
        patch.set_facecolor("#4da6e8"); patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(regime_ids) + 1))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Rinnovabili share (Wind+Solar+Hydro) %")
    ax.set_title("Rinnovabili share (%)", fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    fig.suptitle("Dettaglio fuel per regime", fontweight="bold", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "08_fuel_detail.png"
    fig.savefig(out); plt.close(fig)
    print(f"  [plot] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # ── Fase 1: compute (o carica da disco) ─────────────────────────────
    wstats, rstats = load_or_compute()

    # ── Interpretazione verbosa → stdout + file ──────────────────────────
    print("\n" + "═" * 72)
    print("  INTERPRETAZIONE ECONOMICA")
    print("═" * 72)
    print_interpretation(wstats, rstats,
                         out_txt=OUT_DIR / "interpretation.txt")

    # ── Fase 2: plot ─────────────────────────────────────────────────────
    print(f"\nFase 2: generazione plot → {OUT_DIR} …")
    plot_umap_scatter(wstats)
    plot_regime_timeline(wstats)
    plot_lmp_boxplot(wstats)
    plot_fuel_mix_bar(wstats)
    plot_ilr_radar(wstats)
    plot_monthly_heatmap(wstats)
    plot_lmp_vs_load(wstats)
    plot_fuel_detail(wstats)

    print(f"\n✓ Fatto.  Output in {OUT_DIR}")


if __name__ == "__main__":
    main()
