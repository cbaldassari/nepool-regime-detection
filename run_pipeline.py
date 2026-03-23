"""
run_pipeline.py
===============
Launcher della pipeline NEPOOL Regime Detection in locale.

Struttura
---------
  [1]   Preprocessing      — step01_preprocessing.py
  [2b]  Baseline features  — baseline_features.py + baseline_stats_gmm.py
                               (opzionale, --baselines)
  [2]   Embeddings         — step02_embeddings.py  (per exp, sequenziale)
  [3]   PCA+UMAP+GMM       — step03_pca_umap_gmm.py (per exp, sequenziale)
  [3b]  Sensitivity        — sensitivity_K.py + sensitivity_seed.py
                               (opzionale, --sensitivity; per exp)
  [4]   Compare TOPSIS     — step03_compare.py → winner.json
  [3c]  Alt clustering     — clustering_dendrogram.py
                               (opzionale, --baselines; sul vincitore)
  [3d]  Diagnostics        — step03_interpretation.py + altri
                               (opzionale, --diagnostics; sul vincitore)
  [5]   Markov             — step04_markov.py (sul vincitore)

Esperimenti A-O:
  A-I  : price-only (diverse trasformazioni)
  J-L  : ILR raw (fuel-mix isometric log-ratio)
  M-O  : ILR detrended (MSTL residui ILR)

Uso
---
  python run_pipeline.py                        # tutti gli esperimenti A-O
  python run_pipeline.py --exp A B D            # solo alcuni esperimenti
  python run_pipeline.py --skip-preproc         # salta step01 (già fatto)
  python run_pipeline.py --skip-emb             # salta step02
  python run_pipeline.py --sensitivity          # aggiunge [3b] sensitivity K/seed
  python run_pipeline.py --baselines            # aggiunge [2b] baseline e [3c] dendrogram
  python run_pipeline.py --diagnostics          # aggiunge [3d] diagnostics sul vincitore
  python run_pipeline.py --markov-exp D         # forza exp per Markov/alt-clustering/diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

PYTHON      = sys.executable
ALL_EXPS    = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
               "J", "K", "L",   # ILR raw ablation
               "M", "N", "O"]   # ILR detrended ablation
RESULTS_DIR = Path("results")
PROJECT_DIR = Path(__file__).parent.resolve()


# =============================================================================
# HELPERS
# =============================================================================

def S(name: str) -> str:
    """Path assoluto di uno script nella directory del progetto."""
    return str(PROJECT_DIR / name)


def run_local(script: str, *args: str, label: str = "") -> bool:
    """Esegue uno script come subprocess e stampa stato + tempo."""
    cmd = [PYTHON, script, *args]
    tag = label or Path(script).name
    bar = "-" * 65
    print(f"\n{bar}\n  >> {tag}\n{bar}", flush=True)
    t0  = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    ret = subprocess.run(cmd, env=env, cwd=str(PROJECT_DIR))
    elapsed = time.time() - t0
    status  = "OK  " if ret.returncode == 0 else "FAIL"
    print(f"  [{status}] {tag}  ({elapsed:.1f}s)", flush=True)
    return ret.returncode == 0


def run_sequence(tasks: list[tuple], label: str) -> list[str]:
    """
    Esegue una lista di task sequenzialmente.
    tasks: lista di (script, args_list, task_label)
    Restituisce lista di label falliti.
    """
    bar = "=" * 65
    n   = len(tasks)
    print(f"\n{bar}\n  {label}  ({n} task)\n{bar}", flush=True)
    failed = []
    t_start = time.time()
    for i, (script, args, task_label) in enumerate(tasks, 1):
        print(f"\n  [{i}/{n}] {task_label}", flush=True)
        ok = run_local(script, *args, label=task_label)
        if not ok:
            failed.append(task_label)
    elapsed = time.time() - t_start
    print(f"\n  Completati: {n - len(failed)}/{n}  totale={elapsed:.1f}s  "
          f"({'tutti OK' if not failed else f'{len(failed)} falliti'})",
          flush=True)
    return failed


def read_winner() -> str | None:
    p = RESULTS_DIR / "comparison" / "winner.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
        return d.get("winner_exp") or d.get("winner")


def _header(msg: str) -> None:
    print(f"\n{'=' * 65}\n  {msg}\n{'=' * 65}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline NEPOOL in locale",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exp", nargs="+", default=ALL_EXPS,
        choices=ALL_EXPS, metavar="EXP",
    )
    parser.add_argument("--skip-preproc", action="store_true",
                        help="Salta step01 (preprocessing già fatto)")
    parser.add_argument("--skip-emb",     action="store_true",
                        help="Salta step02 (embeddings già calcolati)")
    parser.add_argument("--sensitivity",  action="store_true",
                        help="Aggiunge [3b] sensitivity-K e sensitivity-seed per ogni exp")
    parser.add_argument("--baselines",    action="store_true",
                        help="Aggiunge [2b] baseline features e [3c] dendrogram sul vincitore")
    parser.add_argument("--diagnostics",  action="store_true",
                        help="Aggiunge [3d] diagnostics sul vincitore")
    parser.add_argument("--markov-exp",   default=None, metavar="EXP",
                        help="Forza exp per Markov/alt-clustering/diagnostics (ignora winner.json)")
    args    = parser.parse_args()
    exps    = args.exp
    t_start = time.time()
    failed  = []

    _header(
        f"NEPOOL Regime Detection — Pipeline locale\n"
        f"  Esperimenti : {exps}\n"
        f"  Python      : {PYTHON}"
    )

    # ── 1. Preprocessing ─────────────────────────────────────────────────────
    _header("[1] Preprocessing")
    if args.skip_preproc:
        print("  SALTATO (--skip-preproc)", flush=True)
    elif not run_local(S("pipeline/step01_preprocessing.py"), label="step01"):
        failed.append("step01")

    # ── 2b. Baseline features (opzionale) ────────────────────────────────────
    if args.baselines:
        _header("[2b] Baseline features")
        for script, lbl in [
            ("baselines/baseline_features.py",  "baseline_features"),
            ("baselines/baseline_stats_gmm.py", "baseline_stats_gmm"),
        ]:
            if not run_local(S(script), label=lbl):
                failed.append(lbl)

    # ── 2. Embeddings Chronos-2 (sequenziale per exp) ────────────────────────
    if not args.skip_emb:
        tasks = [
            (S("pipeline/step02_embeddings.py"), ["--exp", exp], f"emb_{exp}")
            for exp in exps
        ]
        failed += run_sequence(tasks, "[2] Embeddings Chronos-2")
    else:
        print("\n  [2] Embeddings: SALTATO (--skip-emb)", flush=True)

    # ── 3. PCA -> UMAP -> GMM (sequenziale per exp) ──────────────────────────
    tasks = [
        (S("pipeline/step03_pca_umap_gmm.py"), ["--exp", exp], f"clust_{exp}")
        for exp in exps
    ]
    failed += run_sequence(tasks, "[3] PCA+UMAP+GMM (K automatico)")

    # ── 3b. Sensitivity (opzionale) ──────────────────────────────────────────
    if args.sensitivity:
        tasks_k = [
            (S("sensitivity/sensitivity_K.py"), ["--exp", exp], f"sens_K_{exp}")
            for exp in exps
        ]
        tasks_s = [
            (S("sensitivity/sensitivity_seed.py"), ["--exp", exp], f"sens_seed_{exp}")
            for exp in exps
        ]
        failed += run_sequence(tasks_k, "[3b] Sensitivity-K")
        failed += run_sequence(tasks_s, "[3b] Sensitivity-seed")

    # ── 4. Compare TOPSIS ────────────────────────────────────────────────────
    _header("[4] Compare (TOPSIS)")
    if not run_local(S("pipeline/step03_compare.py"), label="step03_compare"):
        failed.append("step03_compare")

    # ── 3c. Clustering alternativi sul vincitore (opzionale) ──────────────────
    if args.baselines:
        alt_exp = args.markov_exp or read_winner()
        _header(f"[3c] Clustering alternativi  exp={alt_exp}")
        if alt_exp:
            if not run_local(S("baselines/clustering_dendrogram.py"),
                             "--exp", alt_exp, label=f"dendrogram_{alt_exp}"):
                failed.append(f"dendrogram_{alt_exp}")
        else:
            print("  winner.json non trovato — usa --markov-exp", flush=True)

    # ── 3d. Diagnostics sul vincitore (opzionale) ─────────────────────────────
    if args.diagnostics:
        diag_exp = args.markov_exp or read_winner()
        _header(f"[3d] Diagnostics  exp={diag_exp}")
        if diag_exp:
            for script, lbl, extra in [
                ("diagnostics/step03_interpretation.py", f"interpretation_{diag_exp}", ["--exp", diag_exp]),
                ("diagnostics/step01_justification.py",  "step01_justification",       []),
                ("diagnostics/step02_justification.py",  "step02_justification",       ["--exp", diag_exp]),
                ("diagnostics/step03_diagnostics.py",    "step03_diagnostics",         ["--exp", diag_exp]),
                ("diagnostics/step03_validation.py",     "step03_validation",          ["--exp", diag_exp]),
            ]:
                if not run_local(S(script), *extra, label=lbl):
                    failed.append(lbl)
        else:
            print("  winner.json non trovato — usa --markov-exp", flush=True)

    # ── 5. Markov ─────────────────────────────────────────────────────────────
    markov_exp = args.markov_exp or read_winner()
    _header(f"[5] Markov  exp={markov_exp}")
    if markov_exp:
        if not run_local(S("pipeline/step04_markov.py"), "--exp", markov_exp,
                         label=f"step04_markov_{markov_exp}"):
            failed.append(f"step04_{markov_exp}")
    else:
        print("  winner.json non trovato — usa --markov-exp", flush=True)

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    _header("RIEPILOGO")
    print(f"  Tempo totale : {elapsed / 60:.1f} min", flush=True)
    if failed:
        print(f"  FALLITI ({len(failed)}) : {failed}", flush=True)
    else:
        print("  Tutti gli step completati con successo.", flush=True)


if __name__ == "__main__":
    main()
