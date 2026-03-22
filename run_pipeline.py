"""
run_pipeline.py
===============
Launcher della pipeline NEPOOL Regime Detection su cluster Ray.

Strategia
---------
  Step 1  : locale, sequenziale (preprocessing unico e veloce)
  Step 2  : Ray task per ogni esperimento (gestisce GPU/CPU automaticamente)
  Step 3f : Ray task per ogni esperimento (CPU, joblib interno)
  Step 3b : Ray task per ogni esperimento (sensitivity, opzionale)
  Step 4  : locale, sequenziale (dipende dall'output del 3)
  Step 5  : locale, sequenziale (dipende dal vincitore)

Gli step 2 e 3 per gli esperimenti A-G vengono sottomessi tutti insieme
come task Ray e girano in parallelo sui nodi disponibili del cluster.

Risorse per task
----------------
  step02  : CPU_PER_EMB cpu + GPU_PER_EMB gpu  (default: 4 cpu, 1 gpu)
  step03f : CPU_PER_CLUST cpu                  (default: 8 cpu, 0 gpu)
  Modifica le costanti qui sotto per adattarle al tuo cluster.

Uso
---
  python run_pipeline.py                    # tutti gli esperimenti A-G
  python run_pipeline.py --exp A B D        # solo alcuni esperimenti
  python run_pipeline.py --skip-emb        # salta step02
  python run_pipeline.py --sensitivity     # aggiunge sensitivity_K/seed
  python run_pipeline.py --markov-exp D    # forza esperimento per Markov
  python run_pipeline.py --ray-address auto  # connetti a cluster Ray esterno
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
ALL_EXPS    = ["A", "B", "C", "D", "E", "F", "G"]
RESULTS_DIR = Path("results")

# Risorse Ray per task — 8 CPU fisici per nodo/worker
CPU_PER_EMB   = 8    # core per step02 (embedding) — nodo intero
GPU_PER_EMB   = 1    # gpu per step02 (0 = cpu-only)
CPU_PER_CLUST = 8    # core per step03f (clustering) — nodo intero


# =============================================================================
# RAY TASK
# =============================================================================

def _make_ray_run():
    """
    Definisce il task Ray a runtime (dopo ray.init) per evitare
    import di ray al top-level prima dell'inizializzazione.
    """
    import ray

    @ray.remote
    def ray_run(script: str, args: list[str], label: str) -> tuple[str, bool, float]:
        """Task Ray: lancia script come subprocess e restituisce (label, ok, sec)."""
        cmd = [PYTHON, script, *args]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        t0  = time.time()
        ret = subprocess.run(cmd, env=env)
        return label, ret.returncode == 0, time.time() - t0

    return ray_run


# =============================================================================
# HELPERS
# =============================================================================

def run_local(script: str, *args: str, label: str = "") -> bool:
    """Esegue script localmente (step1, step4, step5)."""
    cmd = [PYTHON, script, *args]
    tag = label or script
    bar = "-" * 65
    print(f"\n{bar}\n  >> {tag}\n{bar}", flush=True)
    t0  = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    ret = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    status  = "OK  " if ret.returncode == 0 else "FAIL"
    print(f"  [{status}] {tag}  ({elapsed:.1f}s)", flush=True)
    return ret.returncode == 0


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / max(total, 1))
    bar    = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total}"


def run_parallel_ray(ray_run, tasks: list[tuple], label: str) -> list[str]:
    """
    Sottomette una lista di task Ray e aspetta il completamento.
    Mostra una barra di progresso man mano che i task terminano.

    tasks: lista di (script, args_list, task_label, num_cpus, num_gpus)
    Restituisce lista di label falliti.
    """
    import ray

    bar = "=" * 65
    n   = len(tasks)
    print(f"\n{bar}\n  {label}  ({n} task Ray)\n{bar}", flush=True)

    # sottometti tutti i task
    pending = {}   # future -> task_label
    for script, args, task_label, n_cpu, n_gpu in tasks:
        remote_fn = ray_run.options(num_cpus=n_cpu, num_gpus=n_gpu)
        fut = remote_fn.remote(script, args, task_label)
        pending[fut] = task_label
        print(f"  >> sottomesso: {task_label}", flush=True)

    # attendi con barra di progresso
    print(flush=True)
    done   = 0
    failed = []
    remaining = list(pending.keys())

    while remaining:
        # ray.wait restituisce appena almeno 1 task e' pronto
        ready, remaining = ray.wait(remaining, num_returns=1, timeout=None)
        for fut in ready:
            lbl, ok, elapsed = ray.get(fut)
            done += 1
            status = "OK  " if ok else "FAIL"
            print(f"  [{status}] {lbl}  ({elapsed:.1f}s)  "
                  f"{_progress_bar(done, n)}", flush=True)
            if not ok:
                failed.append(lbl)

    print(f"\n  Completati: {done}/{n}  "
          f"({'tutti OK' if not failed else f'{len(failed)} falliti'})",
          flush=True)
    return failed


def read_winner() -> str | None:
    p = RESULTS_DIR / "comparison" / "winner.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f).get("winner")


def _header(msg: str) -> None:
    print(f"\n{'=' * 65}\n  {msg}\n{'=' * 65}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline NEPOOL su cluster Ray",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exp", nargs="+", default=ALL_EXPS,
        choices=ALL_EXPS, metavar="EXP",
    )
    parser.add_argument("--skip-emb",    action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--markov-exp",  default=None, metavar="EXP")
    parser.add_argument(
        "--ray-address", default="ray://datalab-rayclnt.unitus.it:10001",
        metavar="ADDR",
        help="Indirizzo Ray (default: cluster Unitus)",
    )
    args    = parser.parse_args()
    exps    = args.exp
    t_start = time.time()
    failed  = []

    # ── Init Ray ─────────────────────────────────────────────────────────────
    import ray
    try:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    except ConnectionError:
        print("  Ray cluster non trovato — avvio Ray in modalita' locale",
              flush=True)
        ray.init(ignore_reinit_error=True)
    resources = ray.cluster_resources()
    _header(
        f"NEPOOL Regime Detection — Ray Pipeline\n"
        f"  Esperimenti : {exps}\n"
        f"  Ray cluster : CPU={resources.get('CPU', '?'):.0f}"
        f"  GPU={resources.get('GPU', 0):.0f}"
    )

    ray_run = _make_ray_run()

    # ── 1. Preprocessing (locale) ────────────────────────────────────────────
    _header("[1] Preprocessing")
    if not run_local("step01_preprocessing.py", label="step01"):
        failed.append("step01")

    # ── 2. Embeddings Chronos-2 (Ray, parallelo per exp) ─────────────────────
    if not args.skip_emb:
        tasks = [
            ("step02_embeddings.py", ["--exp", exp],
             f"emb_{exp}", CPU_PER_EMB, GPU_PER_EMB)
            for exp in exps
        ]
        failed += run_parallel_ray(ray_run, tasks, "[2] Embeddings Chronos-2")
    else:
        print("\n  [2] Embeddings: SALTATO (--skip-emb)", flush=True)

    # ── 3. PCA -> UMAP -> GMM (Ray, parallelo per exp) ───────────────────────
    tasks = [
        ("step03f_pca_umap_gmm.py", ["--exp", exp],
         f"clust_{exp}", CPU_PER_CLUST, 0)
        for exp in exps
    ]
    failed += run_parallel_ray(ray_run, tasks, "[3] PCA+UMAP+GMM (K automatico)")

    # ── 3b. Sensitivity (Ray, opzionale) ─────────────────────────────────────
    if args.sensitivity:
        tasks_k = [
            ("step03f_sensitivity_K.py", ["--exp", exp],
             f"sens_K_{exp}", CPU_PER_CLUST, 0)
            for exp in exps
        ]
        tasks_s = [
            ("step03f_sensitivity_seed.py", ["--exp", exp],
             f"sens_seed_{exp}", CPU_PER_CLUST, 0)
            for exp in exps
        ]
        failed += run_parallel_ray(ray_run, tasks_k,    "[3b] Sensitivity-K")
        failed += run_parallel_ray(ray_run, tasks_s,    "[3b] Sensitivity-seed")

    # ── 4. Compare TOPSIS (locale) ───────────────────────────────────────────
    _header("[4] Compare (TOPSIS)")
    if not run_local("step03_compare.py", label="step03_compare"):
        failed.append("step03_compare")

    # ── 5. Markov (locale) ───────────────────────────────────────────────────
    markov_exp = args.markov_exp or read_winner()
    _header(f"[5] Markov  exp={markov_exp}")
    if markov_exp:
        if not run_local("step04_markov.py", "--exp", markov_exp,
                         label=f"step04_markov exp={markov_exp}"):
            failed.append(f"step04_{markov_exp}")
    else:
        print("  winner.json non trovato — usa --markov-exp", flush=True)

    ray.shutdown()

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
