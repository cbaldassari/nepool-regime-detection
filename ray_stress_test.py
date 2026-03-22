"""
ray_stress_test.py
==================
Test di carico sul cluster Ray — verifica che la CPU salga nella dashboard.

Lancia N task su tutti i nodi disponibili, ognuno usa tutti i core
del nodo per un calcolo CPU-intensivo (moltiplicazione matrici numpy).

Uso
---
  python ray_stress_test.py                  # default: 3 task x 30 sec
  python ray_stress_test.py --tasks 6        # 6 task (2 per nodo)
  python ray_stress_test.py --duration 60    # 60 secondi per task
"""

import argparse
import time
import ray

RAY_ADDRESS = "ray://10.4.4.7:10001"


@ray.remote
def cpu_stress(task_id: int, duration_s: int, n_cpu: int) -> dict:
    """
    Carico CPU puro: moltiplica matrici grandi in loop per `duration_s` secondi.
    Usa tutti i thread disponibili tramite numpy (OpenBLAS/MKL).
    """
    import numpy as np
    import os
    import socket

    os.environ["OMP_NUM_THREADS"]      = str(n_cpu)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_cpu)
    os.environ["MKL_NUM_THREADS"]      = str(n_cpu)

    host = socket.gethostname()
    pid  = os.getpid()
    size = 2000   # matrice 2000x2000 float64 — ~30MB, riempie la cache L3

    t0       = time.time()
    iters    = 0
    while time.time() - t0 < duration_s:
        A = np.random.randn(size, size).astype(np.float64)
        B = np.random.randn(size, size).astype(np.float64)
        _ = A @ B
        iters += 1

    elapsed = time.time() - t0
    return {
        "task_id" : task_id,
        "host"    : host,
        "pid"     : pid,
        "elapsed" : round(elapsed, 1),
        "iters"   : iters,
        "gflops"  : round(iters * 2 * size**3 / elapsed / 1e9, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",    type=int, default=3,
                        help="Numero di task (default: 3)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Durata per task in secondi (default: 30)")
    parser.add_argument("--cpus",     type=int, default=8,
                        help="CPU per task (default: 8)")
    args = parser.parse_args()

    print(f"\nConnessione a {RAY_ADDRESS}...")
    try:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    except ConnectionError:
        print("  cluster non trovato — modalita' locale")
        ray.init(ignore_reinit_error=True)

    res   = ray.cluster_resources()
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    print(f"Cluster: CPU={res.get('CPU',0):.0f}  GPU={res.get('GPU',0):.0f}"
          f"  nodi={len(nodes)}")

    print(f"\nLancio {args.tasks} task x {args.duration}s x {args.cpus} CPU...")
    print("Apri la dashboard Ray per vedere la CPU salire.\n")

    t0      = time.time()
    futures = [
        cpu_stress.options(num_cpus=args.cpus).remote(i, args.duration, args.cpus)
        for i in range(args.tasks)
    ]

    for i, fut in enumerate(ray.get(futures)):
        print(f"  task {fut['task_id']}  host={fut['host']}  "
              f"pid={fut['pid']}  {fut['elapsed']}s  "
              f"{fut['iters']} iter  {fut['gflops']} GFLOPS")

    print(f"\nTotale: {time.time()-t0:.1f}s")
    ray.shutdown()


if __name__ == "__main__":
    main()
