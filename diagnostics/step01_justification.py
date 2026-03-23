"""
step01_justification.py
=======================
Giustificazione empirica della soglia di clip p1/p99 (Exp F).

Domanda del reviewer
---------------------
  "Perché p1/p99 e non p5/p95 o winsorize a ±3σ?"

Output  (results/step01_justification/)
-----------------------------------------
  01_distribution_raw.png   — distribuzione LMP raw con soglie p1/p99
  02_clip_comparison.png    — % outlier rimossi per soglie alternative

Uso
---
  python diagnostics/step01_justification.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
import config as C

RESULTS_DIR = ROOT / C.RESULTS_DIR
PREP_PATH   = RESULTS_DIR / "preprocessed.parquet"
RAW_PATH    = ROOT / "isone_dataset.parquet"
OUT_DIR     = RESULTS_DIR / "step01_justification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
plt.rcParams.update({"figure.dpi": DPI, "savefig.bbox": "tight",
                     "savefig.facecolor": "white"})

print("\n" + "═"*65)
print("  Step01 Justification — Clip p1/p99")
print("═"*65)

# ── carica LMP raw ────────────────────────────────────────────────────────────
if not PREP_PATH.exists():
    print(f"  ERRORE: {PREP_PATH} non trovato — esegui step01_preprocessing.py")
    sys.exit(1)

prep = pd.read_parquet(PREP_PATH)
prep["datetime"] = pd.to_datetime(prep["datetime"])
prep = prep.set_index("datetime").sort_index()

# usa la colonna lmp se presente, altrimenti dal raw
if "lmp" in prep.columns:
    lmp = prep["lmp"].dropna()
else:
    raw = pd.read_parquet(RAW_PATH)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.set_index("datetime").sort_index()
    lmp_col = next((c for c in raw.columns if "lmp" in c.lower()), None)
    if lmp_col is None:
        print("  ERRORE: colonna LMP non trovata")
        sys.exit(1)
    lmp = raw[lmp_col].reindex(prep.index).dropna()

print(f"\n  LMP: {len(lmp)} osservazioni  "
      f"min={lmp.min():.1f}  max={lmp.max():.1f}  "
      f"skew={lmp.skew():.2f}")

p1  = float(lmp.quantile(0.01))
p99 = float(lmp.quantile(0.99))
n_out_p1p99 = int(((lmp < p1) | (lmp > p99)).sum())
print(f"  p1={p1:.2f}  p99={p99:.2f}  outlier rimossi: "
      f"{n_out_p1p99} ({100*n_out_p1p99/len(lmp):.2f}%)")

# ── Plot 01: distribuzione LMP raw ────────────────────────────────────────────
print("\n[1] Plot 01_distribution_raw.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# istogramma (clip per visualizzazione)
vis_lo, vis_hi = lmp.quantile(0.001), lmp.quantile(0.999)
ax1.hist(lmp.clip(vis_lo, vis_hi), bins=120,
         color="#2c6fad", alpha=0.75, edgecolor="none")
ax1.axvline(p1,  color="red",    lw=1.5, ls="--", label=f"p1  = {p1:.1f} $/MWh")
ax1.axvline(p99, color="orange", lw=1.5, ls="--", label=f"p99 = {p99:.1f} $/MWh")
ax1.set_title("LMP raw — distribuzione", fontweight="bold")
ax1.set_xlabel("LMP ($/MWh)")
ax1.set_ylabel("Frequenza")
ax1.legend(fontsize=9)

# serie storica (campionata)
sample = lmp.iloc[::4]
ax2.plot(sample.index, sample.values, lw=0.4, color="#2c6fad", alpha=0.7)
ax2.axhline(p1,  color="red",    lw=1.2, ls="--", label=f"p1 = {p1:.1f}")
ax2.axhline(p99, color="orange", lw=1.2, ls="--", label=f"p99 = {p99:.1f}")
ax2.set_title("LMP raw — serie storica", fontweight="bold")
ax2.set_xlabel("Data")
ax2.set_ylabel("LMP ($/MWh)")
ax2.tick_params(axis="x", rotation=30)
ax2.legend(fontsize=9)

fig.suptitle("LMP Raw: distribuzione e soglie di clip", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "01_distribution_raw.png", dpi=DPI)
plt.close(fig)
print(f"  Salvato: {OUT_DIR / '01_distribution_raw.png'}")

# ── Plot 02: confronto soglie di clip ─────────────────────────────────────────
print("\n[2] Plot 02_clip_comparison.png...")

mu, sigma = lmp.mean(), lmp.std()
soglie = {
    "p1/p99":   (lmp.quantile(0.01),  lmp.quantile(0.99)),
    "p2/p98":   (lmp.quantile(0.02),  lmp.quantile(0.98)),
    "p5/p95":   (lmp.quantile(0.05),  lmp.quantile(0.95)),
    "±2σ":      (mu - 2*sigma, mu + 2*sigma),
    "±3σ":      (mu - 3*sigma, mu + 3*sigma),
    "±4σ":      (mu - 4*sigma, mu + 4*sigma),
}

labels, pcts, lo_vals, hi_vals = [], [], [], []
for name, (lo, hi) in soglie.items():
    n_out = int(((lmp < lo) | (lmp > hi)).sum())
    labels.append(name)
    pcts.append(100 * n_out / len(lmp))
    lo_vals.append(lo)
    hi_vals.append(hi)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# barchart % outlier
colors = ["#2c6fad" if l == "p1/p99" else "#aec7e8" for l in labels]
bars = ax1.bar(labels, pcts, color=colors, alpha=0.85, width=0.6)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_title("% osservazioni rimosse per soglia", fontweight="bold")
ax1.set_xlabel("Soglia di clip")
ax1.set_ylabel("% outlier rimossi")
for bar, v in zip(bars, pcts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
bars[0].set_edgecolor("gold")
bars[0].set_linewidth(2.5)
ax1.text(0.02, 0.97, "★ scelta Exp F", transform=ax1.transAxes,
         va="top", ha="left", fontsize=9, color="goldenrod", fontweight="bold")

# intervalli [lo, hi] per soglia
ax2.barh(labels, [h - l for l, h in zip(lo_vals, hi_vals)],
         left=lo_vals, color=colors, alpha=0.7, height=0.5)
ax2.axvline(lmp.min(), color="gray", lw=0.8, ls=":")
ax2.axvline(lmp.max(), color="gray", lw=0.8, ls=":")
ax2.set_title("Intervallo [lo, hi] per ogni soglia", fontweight="bold")
ax2.set_xlabel("LMP ($/MWh)")
for i, (l, h) in enumerate(zip(lo_vals, hi_vals)):
    ax2.text(l - 1, i, f"{l:.0f}", ha="right", va="center", fontsize=8)
    ax2.text(h + 1, i, f"{h:.0f}", ha="left",  va="center", fontsize=8)

fig.suptitle("Confronto soglie di clip — giustificazione scelta p1/p99",
             fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "02_clip_comparison.png", dpi=DPI)
plt.close(fig)
print(f"  Salvato: {OUT_DIR / '02_clip_comparison.png'}")

# ── Riepilogo ─────────────────────────────────────────────────────────────────
print("\n" + "═"*65)
print("  RIEPILOGO")
print("═"*65)
for name, pct, (lo, hi) in zip(labels, pcts, soglie.values()):
    marker = " ← scelta Exp F" if name == "p1/p99" else ""
    print(f"  {name:8s}  [{lo:8.1f}, {hi:8.1f}]  rimossi: {pct:.2f}%{marker}")
print(f"\n  Output: {OUT_DIR}")
