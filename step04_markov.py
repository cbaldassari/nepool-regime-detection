"""
step04_markov.py
================
Analisi della catena di Markov sui regimi del clustering vincitore (Exp D).

Test e analisi
--------------
1. Test ordine-1 di Markov (Anderson & Goodman 1957)
   H0: P(X_t | X_{t-1}, X_{t-2}) = P(X_t | X_{t-1})
   Statistica: likelihood ratio chi-quadro
   df = K * (K-1)^2

2. Test di stazionarieta' (Bartlett 1951)
   H0: la matrice di transizione e' costante nel tempo
   Split in due sotto-periodi, confronto chi-quadro

3. Matrice di transizione empirica P con IC bootstrap (95%)

4. Distribuzione stazionaria pi = lim P^t
   Interpretazione: frazione di tempo attesa in ciascun regime

5. Mean First Passage Time (MFPT)
   m_ij = tempo atteso (in finestre) per raggiungere j partendo da i

Output (results/step04/)
------------------------
  markov_report.json     test statistici + pi + MFPT
  01_transition.png      heatmap matrice P con IC
  02_stationary.png      distribuzione stazionaria
  03_mfpt.png            heatmap MFPT
  04_stationarity.png    confronto P primo/secondo semestre

Uso
---
  python step04_markov.py [--exp D]
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
from scipy.stats import chi2

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--exp", default="D", choices=["A","B","C","D"])
EXPERIMENT = _parser.parse_known_args()[0].exp

LABELS_PATH = Path(f"results/exp_{EXPERIMENT}/step03f/labels_best.parquet")
OUT_DIR     = Path("results/step04")
SEED        = 42
DPI         = 150
N_BOOTSTRAP = 1000   # campioni bootstrap per IC sulla matrice P

# =============================================================================
# HELPERS
# =============================================================================

CMAP = plt.get_cmap("tab20")

def regime_colors(n):
    return [CMAP(i / max(n, 1)) for i in range(n)]


def load_labels() -> tuple[np.ndarray, pd.Series]:
    df     = pd.read_parquet(LABELS_PATH)
    labels = df["regime"].values.astype(int)
    ts     = pd.to_datetime(df["timestamp"])
    K      = len(np.unique(labels))
    print(f"  Caricato: {len(labels):,} finestre  K={K}  exp={EXPERIMENT}",
          flush=True)
    return labels, ts


# =============================================================================
# MATRICE DI TRANSIZIONE
# =============================================================================

def transition_matrix(labels: np.ndarray,
                      K: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Restituisce:
      T_counts  — conteggi grezzi N_{ij}
      P         — probabilita' di transizione (normalizzata per riga)
    K puo' essere passato esplicitamente per garantire dimensione fissa
    (utile nel bootstrap dove alcune sequenze potrebbero non contenere tutti i regimi).
    """
    if K is None:
        K = len(np.unique(labels))
    T = np.zeros((K, K), dtype=float)
    for i in range(len(labels) - 1):
        T[labels[i], labels[i + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums
    return T, P


def bootstrap_ci(labels: np.ndarray, alpha: float = 0.05,
                 n_boot: int = N_BOOTSTRAP, rng_seed: int = SEED
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    IC bootstrap (percentile) al livello (1-alpha) per ogni cella di P.
    Ricampiona sequenze di run-length per preservare la struttura temporale.
    Restituisce P_lo, P_hi  (shape K x K).
    """
    rng = np.random.default_rng(rng_seed)
    K   = len(np.unique(labels))

    # calcola run-length encoding
    runs = []
    cur_regime, cur_len = labels[0], 1
    for l in labels[1:]:
        if l == cur_regime:
            cur_len += 1
        else:
            runs.append((cur_regime, cur_len))
            cur_regime, cur_len = l, 1
    runs.append((cur_regime, cur_len))

    boot_P = np.zeros((n_boot, K, K))
    for b in range(n_boot):
        idx    = rng.integers(0, len(runs), size=len(runs))
        seq    = np.concatenate([np.full(runs[i][1], runs[i][0]) for i in idx])
        _, Pb  = transition_matrix(seq, K=K)
        boot_P[b] = Pb

    lo = np.percentile(boot_P, 100 * alpha / 2,     axis=0)
    hi = np.percentile(boot_P, 100 * (1 - alpha/2), axis=0)
    return lo, hi


# =============================================================================
# TEST DI MARKOV ORDINE 1  (Anderson & Goodman 1957)
# =============================================================================

def test_markov_order1(labels: np.ndarray) -> dict:
    """
    Likelihood ratio test: H0 = catena ordine-1  vs  H1 = catena ordine-2.

    Statistica:
      L2 = 2 * sum_{i,j,k} n_{ijk} * ln( n_{ijk} * n_j / (n_{ij} * n_{jk}) )

    df = K * (K-1)^2   (Anderson & Goodman 1957, Theorem 2)

    Se p > 0.05 -> non si rigetta H0 -> catena ordine-1 giustificata.
    """
    K = len(np.unique(labels))

    # conteggi triplette n_{ijk}
    n3 = np.zeros((K, K, K), dtype=float)
    for t in range(len(labels) - 2):
        n3[labels[t], labels[t+1], labels[t+2]] += 1

    # marginali
    n_ij = n3.sum(axis=2)   # shape K x K
    n_jk = n3.sum(axis=0)   # shape K x K
    n_j  = n3.sum(axis=(0, 2))  # shape K

    L2 = 0.0
    for i in range(K):
        for j in range(K):
            for k in range(K):
                num = n3[i, j, k] * n_j[j]
                den = n_ij[i, j] * n_jk[j, k]
                if num > 0 and den > 0:
                    L2 += 2 * n3[i, j, k] * np.log(num / den)

    df   = K * (K - 1) ** 2
    pval = float(chi2.sf(L2, df))
    return {
        "test"       : "Anderson-Goodman order-1 vs order-2",
        "statistic"  : round(L2, 3),
        "df"         : df,
        "p_value"    : round(pval, 6),
        "reject_H0"  : pval < 0.05,
        "conclusion" : ("Ordine-2 significativo — la catena non e' Markov ordine-1"
                        if pval < 0.05 else
                        "Non si rigetta Markov ordine-1 (p > 0.05)"),
    }


# =============================================================================
# TEST DI STAZIONARIETA'  (Bartlett 1951)
# =============================================================================

def test_stationarity(labels: np.ndarray, ts: pd.Series,
                      n_splits: int = 2) -> dict:
    """
    Divide la serie in n_splits sotto-periodi di uguale lunghezza,
    stima P in ciascuno e testa se sono uguali con un likelihood ratio
    chi-quadro.

    df = (n_splits - 1) * K * (K - 1)
    """
    K   = len(np.unique(labels))
    N   = len(labels)
    sz  = N // n_splits
    splits = [labels[i*sz : (i+1)*sz] for i in range(n_splits)]

    # P globale (pooled)
    T_pool, P_pool = transition_matrix(labels)

    L2 = 0.0
    for seg in splits:
        T_s, _ = transition_matrix(seg, K=K)
        for i in range(K):
            n_i = T_s[i].sum()
            if n_i == 0:
                continue
            for j in range(K):
                obs = T_s[i, j]
                exp = P_pool[i, j] * n_i
                if obs > 0 and exp > 0:
                    L2 += 2 * obs * np.log(obs / exp)

    df   = (n_splits - 1) * K * (K - 1)
    pval = float(chi2.sf(L2, df))

    # split dates per report
    split_dates = [str(pd.to_datetime(ts.values[i*sz]).date())
                   for i in range(n_splits + 1)
                   if i * sz < N]

    return {
        "test"        : f"Stationarity test ({n_splits} splits)",
        "split_dates" : split_dates,
        "statistic"   : round(L2, 3),
        "df"          : df,
        "p_value"     : round(pval, 6),
        "reject_H0"   : pval < 0.05,
        "conclusion"  : ("Matrice non stazionaria — struttura cambia nel tempo"
                         if pval < 0.05 else
                         "Non si rigetta stazionarieta' (p > 0.05)"),
    }


# =============================================================================
# DISTRIBUZIONE STAZIONARIA
# =============================================================================

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Distribuzione stazionaria pi tale che pi @ P = pi, sum(pi) = 1.
    Calcolata come autovettore sinistro associato all'autovalore 1.
    """
    K      = P.shape[0]
    eigvals, eigvecs = np.linalg.eig(P.T)
    # autovettore corrispondente all'autovalore piu' vicino a 1
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi  = np.real(eigvecs[:, idx])
    pi  = np.abs(pi) / np.abs(pi).sum()
    return pi


# =============================================================================
# MEAN FIRST PASSAGE TIME
# =============================================================================

def mean_first_passage_time(P: np.ndarray) -> np.ndarray:
    """
    Matrice MFPT M dove M[i,j] = tempo atteso (finestre) per raggiungere
    j partendo da i.

    Metodo: soluzione del sistema lineare (Kemeny & Snell 1960)
      M[i,j] = 1 + sum_{k != j} P[i,k] * M[k,j]   per i != j
      M[j,j] = 1 / pi[j]
    """
    K  = P.shape[0]
    pi = stationary_distribution(P)
    M  = np.zeros((K, K))

    for j in range(K):
        # sistema: (I - P + e * pi_j^T) @ m_j = e
        # dove e = vettore di uni, eccetto la riga j
        A = np.eye(K) - P
        A[:, j] = pi          # colonna j sostituita con pi
        b = np.ones(K)
        b[j] = 0
        try:
            m_j = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            m_j = np.full(K, np.nan)
        M[:, j] = m_j
        M[j, j] = 1.0 / pi[j] if pi[j] > 0 else np.inf

    return M


# =============================================================================
# PLOTS
# =============================================================================

def plot_transition(P: np.ndarray, P_lo: np.ndarray, P_hi: np.ndarray,
                    K: int) -> None:
    """01 - Heatmap P con IC bootstrap."""
    colors = regime_colors(K)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # pannello sinistro: heatmap P
    ax = axes[0]
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"R{k}" for k in range(K)], fontsize=8)
    ax.set_yticklabels([f"R{k}" for k in range(K)], fontsize=8)
    ax.set_xlabel("Destinazione"); ax.set_ylabel("Origine")
    ax.set_title("Matrice di transizione P", fontweight="bold")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center",
                    fontsize=max(5, 8 - K),
                    color="white" if P[i,j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="P(j|i)")

    # pannello destro: auto-probabilita' con IC
    ax2 = axes[1]
    self_probs = np.diag(P)
    lo_diag    = np.diag(P_lo)
    hi_diag    = np.diag(P_hi)
    yerr       = [self_probs - lo_diag, hi_diag - self_probs]
    bar_colors = [colors[k] for k in range(K)]
    ax2.bar(range(K), self_probs, color=bar_colors, alpha=0.85,
            yerr=yerr, capsize=4, ecolor="black", error_kw={"lw": 1.2})
    ax2.set_xticks(range(K))
    ax2.set_xticklabels([f"R{k}" for k in range(K)], fontsize=8)
    ax2.set_ylabel("P(stessa regime al passo successivo)")
    ax2.set_title("Auto-probabilita' di transizione\n(barre di errore = IC 95% bootstrap)",
                  fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1/K, color="gray", lw=0.8, ls="--", alpha=0.6,
                label=f"Random (1/K={1/K:.2f})")
    ax2.legend(fontsize=7)

    fig.suptitle(f"Markov Transition — Exp {EXPERIMENT}  K={K}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_transition.png", dpi=DPI)
    plt.close(fig)
    print("  01_transition.png", flush=True)


def plot_stationary(pi: np.ndarray, K: int, price_means: dict) -> None:
    """02 - Distribuzione stazionaria con LMP medio."""
    colors = regime_colors(K)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(K), pi * 100, color=colors, alpha=0.85)
    ax1.set_xticks(range(K))
    ax1.set_xticklabels([f"R{k}" for k in range(K)], fontsize=8)
    ax1.set_ylabel("% tempo atteso nel regime (lungo periodo)")
    ax1.set_title("Distribuzione stazionaria pi", fontweight="bold")
    for k, v in enumerate(pi):
        ax1.text(k, v * 100 + 0.3, f"{v*100:.1f}%", ha="center", fontsize=7)

    # scatter pi vs LMP medio
    means = [price_means.get(f"lmp_mean_R{k}", np.nan) for k in range(K)]
    valid = [(k, pi[k], m) for k, m in enumerate(means) if not np.isnan(m)]
    if valid:
        ks, pis, ms = zip(*valid)
        sc = ax2.scatter(ms, [p*100 for p in pis],
                         c=[colors[k] for k in ks], s=120, zorder=5)
        for k, p, m in valid:
            ax2.annotate(f"R{k}", (m, p*100), textcoords="offset points",
                         xytext=(5, 3), fontsize=7)
        ax2.set_xlabel("LMP medio ($/MWh)")
        ax2.set_ylabel("% tempo nel regime (pi)")
        ax2.set_title("Prezzo vs frequenza regime\n(lungo periodo)",
                      fontweight="bold")

    fig.suptitle(f"Distribuzione stazionaria — Exp {EXPERIMENT}  K={K}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_stationary.png", dpi=DPI)
    plt.close(fig)
    print("  02_stationary.png", flush=True)


def plot_mfpt(M: np.ndarray, K: int) -> None:
    """03 - Heatmap Mean First Passage Time (finestre da 6h)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    # converti in giorni (1 finestra = 6h = 0.25 giorni)
    M_days = M * 0.25
    vmax   = np.nanpercentile(M_days, 95)
    im     = ax.imshow(M_days, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=vmax)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"R{k}" for k in range(K)], fontsize=8)
    ax.set_yticklabels([f"R{k}" for k in range(K)], fontsize=8)
    ax.set_xlabel("Regime destinazione")
    ax.set_ylabel("Regime origine")
    ax.set_title(f"Mean First Passage Time (giorni)\nExp {EXPERIMENT}  K={K}",
                 fontweight="bold")
    for i in range(K):
        for j in range(K):
            val = M_days[i, j]
            if not np.isnan(val) and not np.isinf(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=max(5, 7 - K),
                        color="white" if val > vmax * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Giorni attesi", shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_mfpt.png", dpi=DPI)
    plt.close(fig)
    print("  03_mfpt.png", flush=True)


def plot_stationarity(labels: np.ndarray, ts: pd.Series, K: int) -> None:
    """04 - Confronto P primo vs secondo semestre."""
    N   = len(labels)
    h1  = labels[:N//2]
    h2  = labels[N//2:]
    _, P1 = transition_matrix(h1, K=K)
    _, P2 = transition_matrix(h2, K=K)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    d1 = str(pd.to_datetime(ts.values[0]).date())
    d2 = str(pd.to_datetime(ts.values[N//2]).date())
    d3 = str(pd.to_datetime(ts.values[-1]).date())

    for ax, P, title in zip(axes, [P1, P2],
                            [f"Prima meta'\n{d1} – {d2}",
                             f"Seconda meta'\n{d2} – {d3}"]):
        im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([f"R{k}" for k in range(K)], fontsize=7)
        ax.set_yticklabels([f"R{k}" for k in range(K)], fontsize=7)
        ax.set_title(title, fontweight="bold", fontsize=8)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center",
                        fontsize=max(4, 7 - K),
                        color="white" if P[i,j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f"Test stazionarieta' — P e' stabile nel tempo?  Exp {EXPERIMENT}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_stationarity.png", dpi=DPI)
    plt.close(fig)
    print("  04_stationarity.png", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 04 — Markov chain analysis  (exp={EXPERIMENT})", flush=True)

    labels, ts = load_labels()
    K = len(np.unique(labels))

    # ── Matrice di transizione ────────────────────────────────────────────
    print("\n[1/5] Matrice di transizione + IC bootstrap...", flush=True)
    T_counts, P = transition_matrix(labels)
    print(f"  {int(T_counts.sum()):,} transizioni osservate", flush=True)
    P_lo, P_hi  = bootstrap_ci(labels)
    print(f"  IC bootstrap completato ({N_BOOTSTRAP} campioni)", flush=True)

    # ── Test ordine-1 ─────────────────────────────────────────────────────
    print("\n[2/5] Test Markov ordine-1 (Anderson & Goodman 1957)...", flush=True)
    res_order = test_markov_order1(labels)
    print(f"  chi2({res_order['df']}) = {res_order['statistic']:.1f}"
          f"  p = {res_order['p_value']:.4f}", flush=True)
    print(f"  → {res_order['conclusion']}", flush=True)

    # ── Test stazionarieta' ───────────────────────────────────────────────
    print("\n[3/5] Test stazionarieta' (2 sotto-periodi)...", flush=True)
    res_stat = test_stationarity(labels, ts, n_splits=2)
    print(f"  chi2({res_stat['df']}) = {res_stat['statistic']:.1f}"
          f"  p = {res_stat['p_value']:.4f}", flush=True)
    print(f"  → {res_stat['conclusion']}", flush=True)

    # ── Distribuzione stazionaria ─────────────────────────────────────────
    print("\n[4/5] Distribuzione stazionaria + MFPT...", flush=True)
    pi = stationary_distribution(P)
    M  = mean_first_passage_time(P)
    for k in range(K):
        print(f"  R{k}: pi={pi[k]*100:.1f}%  MFPT ritorno={M[k,k]*0.25:.1f} gg",
              flush=True)

    # ── Salva report ──────────────────────────────────────────────────────
    # carica LMP means dal quality report per i plot
    qr_path = Path(f"results/exp_{EXPERIMENT}/step03f/quality_report.json")
    price_means = {}
    if qr_path.exists():
        with open(qr_path) as f:
            price_means = json.load(f).get("price_separation", {})

    report = {
        "experiment"           : EXPERIMENT,
        "K"                    : K,
        "n_transitions"        : int(T_counts.sum()),
        "test_order1"          : res_order,
        "test_stationarity"    : res_stat,
        "stationary_dist"      : {f"R{k}": round(float(pi[k]), 4)
                                  for k in range(K)},
        "mfpt_days"            : {
            f"R{i}_to_R{j}": round(float(M[i,j]) * 0.25, 2)
            for i in range(K) for j in range(K)
            if not np.isnan(M[i,j]) and not np.isinf(M[i,j])
        },
        "return_time_days"     : {f"R{k}": round(float(M[k,k]) * 0.25, 2)
                                  for k in range(K)
                                  if not np.isinf(M[k,k])},
        "transition_matrix"    : P.round(4).tolist(),
        "transition_matrix_lo" : P_lo.round(4).tolist(),
        "transition_matrix_hi" : P_hi.round(4).tolist(),
    }
    with open(OUT_DIR / "markov_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  markov_report.json", flush=True)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[5/5] Figure...", flush=True)
    plot_transition(P, P_lo, P_hi, K)
    plot_stationary(pi, K, price_means)
    plot_mfpt(M, K)
    plot_stationarity(labels, ts, K)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  MARKOV ANALYSIS SUMMARY  —  Exp {EXPERIMENT}  K={K}")
    print("=" * 60)
    print(f"  Ordine-1:      {'OK' if not res_order['reject_H0'] else 'RIGETTATO'}"
          f"  (p={res_order['p_value']:.4f})")
    print(f"  Stazionarieta: {'OK' if not res_stat['reject_H0'] else 'RIGETTATA'}"
          f"  (p={res_stat['p_value']:.4f})")
    print(f"  Regime piu' frequente (pi):  "
          f"R{pi.argmax()} ({pi.max()*100:.1f}%)")
    print(f"  Return time piu' breve:  "
          f"R{np.diag(M).argmin()} ({np.diag(M).min()*0.25:.1f} gg)")
    print(f"  Output → {OUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
