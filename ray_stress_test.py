"""
ray_stress_test.py
==================
Stress test CPU + GPU sul cluster Ray.

Lancia task CPU e GPU in parallelo su tutti i nodi disponibili.

Uso
---
  python ray_stress_test.py                  # CPU+GPU, 3 task, 30s
  python ray_stress_test.py --cpu-only       # solo CPU
  python ray_stress_test.py --gpu-only       # solo GPU
  python ray_stress_test.py --duration 60    # 60 secondi
  python ray_stress_test.py --tasks 6        # 6 task per tipo
"""

import argparse
import time
import ray

RAY_ADDRESS = "ray://10.4.4.7:10001"


@ray.remote
def cpu_stress(task_id: int, duration_s: int, n_cpu: int) -> dict:
    """Carico CPU: moltiplicazione matrici numpy in loop."""
    import numpy as np
    import os
    import socket

    os.environ["OMP_NUM_THREADS"]      = str(n_cpu)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_cpu)
    os.environ["MKL_NUM_THREADS"]      = str(n_cpu)

    host = socket.gethostname()
    size = 2000

    t0, iters = time.time(), 0
    while time.time() - t0 < duration_s:
        A = np.random.randn(size, size).astype(np.float64)
        _ = A @ A.T
        iters += 1

    elapsed = time.time() - t0
    return {
        "type"    : "CPU",
        "task_id" : task_id,
        "host"    : host,
        "elapsed" : round(elapsed, 1),
        "iters"   : iters,
        "gflops"  : round(iters * 2 * size**3 / elapsed / 1e9, 1),
        "device"  : "cpu",
    }


@ray.remote
def gpu_stress(task_id: int, duration_s: int) -> dict:
    """Carico GPU: moltiplicazione matrici torch.cuda in loop."""
    import socket
    import os

    host = socket.gethostname()

    try:
        import torch
        if not torch.cuda.is_available():
            return {"type": "GPU", "task_id": task_id, "host": host,
                    "error": "CUDA non disponibile"}

        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        size   = 4096   # matrice 4096x4096 float32 — ~64MB VRAM

        A = torch.randn(size, size, device=device, dtype=torch.float32)
        B = torch.randn(size, size, device=device, dtype=torch.float32)

        # warmup
        _ = A @ B
        torch.cuda.synchronize()

        t0, iters = time.time(), 0
        while time.time() - t0 < duration_s:
            _ = A @ B
            torch.cuda.synchronize()
            iters += 1

        elapsed = time.time() - t0
        mem_mb  = torch.cuda.memory_allocated(0) / 1e6

        return {
            "type"    : "GPU",
            "task_id" : task_id,
            "host"    : host,
            "elapsed" : round(elapsed, 1),
            "iters"   : iters,
            "tflops"  : round(iters * 2 * size**3 / elapsed / 1e12, 2),
            "device"  : name,
            "vram_mb" : round(mem_mb, 1),
        }

    except Exception as e:
        return {"type": "GPU", "task_id": task_id, "host": host, "error": str(e)}


def _init_ray():
    try:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    except ConnectionError:
        print("  cluster non trovato — modalita' locale")
        ray.init(ignore_reinit_error=True)


def _print_result(r: dict) -> None:
    if "error" in r:
        print(f"  [{r['type']}] task {r['task_id']}  host={r['host']}"
              f"  ERRORE: {r['error']}")
        return
    if r["type"] == "CPU":
        print(f"  [CPU] task {r['task_id']}  host={r['host']}"
              f"  {r['elapsed']}s  {r['iters']} iter  {r['gflops']} GFLOPS")
    else:
        print(f"  [GPU] task {r['task_id']}  host={r['host']}"
              f"  {r['elapsed']}s  {r['iters']} iter  {r['tflops']} TFLOPS"
              f"  device={r['device']}  VRAM={r['vram_mb']}MB")


def main():
    parser = argparse.ArgumentParser(description="Ray CPU+GPU stress test")
    parser.add_argument("--tasks",    type=int,  default=3)
    parser.add_argument("--duration", type=int,  default=30)
    parser.add_argument("--cpus",     type=int,  default=None,
                        help="Core per task CPU (default: 7 se GPU attiva, 8 se cpu-only)")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu-only", action="store_true")
    args = parser.parse_args()

    run_cpu = not args.gpu_only
    run_gpu = not args.cpu_only

    # se CPU e GPU girano insieme, cede 1 core al task GPU per nodo
    if args.cpus is not None:
        n_cpu = args.cpus
    elif run_cpu and run_gpu:
        n_cpu = 7   # 7 CPU task + 1 CPU per GPU task = 8 totali per nodo
    else:
        n_cpu = 8

    print(f"\nConnessione a {RAY_ADDRESS}...")
    _init_ray()

    res   = ray.cluster_resources()
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    print(f"Cluster: CPU={res.get('CPU',0):.0f}  GPU={res.get('GPU',0):.0f}"
          f"  nodi attivi={len(nodes)}")
    for i, n in enumerate(nodes):
        r = n.get("Resources", {})
        print(f"  nodo {i+1}: CPU={r.get('CPU',0):.0f}  GPU={r.get('GPU',0):.0f}"
              f"  [{n.get('NodeManagerAddress','?')}]")

    futures = []
    n_tasks = 0

    if run_cpu and run_gpu:
        print(f"\nLancio {args.tasks} task CPU + {args.tasks} task GPU in contemporanea")
        print(f"  CPU: {n_cpu} core/task  |  GPU: 1 GPU + 1 core/task")
        print(f"  Layout per nodo: {n_cpu} CPU (stress) + 1 CPU + 1 GPU (stress) = {n_cpu+1} CPU totali")
    elif run_cpu:
        print(f"\nLancio {args.tasks} task CPU ({args.duration}s, {n_cpu} core)")
    else:
        print(f"\nLancio {args.tasks} task GPU ({args.duration}s, 1 GPU)")

    if run_cpu:
        for i in range(args.tasks):
            futures.append(
                cpu_stress.options(num_cpus=n_cpu).remote(i, args.duration, n_cpu)
            )
            n_tasks += 1

    if run_gpu:
        for i in range(args.tasks):
            futures.append(
                gpu_stress.options(num_cpus=1, num_gpus=1).remote(i, args.duration)
            )
            n_tasks += 1

    avail = ray.available_resources()
    used_cpu = res.get("CPU", 0) - avail.get("CPU", 0)
    used_gpu = res.get("GPU", 0) - avail.get("GPU", 0)
    print(f"\n{n_tasks} task in esecuzione  |  "
          f"cluster in uso: CPU={used_cpu:.0f}/{res.get('CPU',0):.0f}"
          f"  GPU={used_gpu:.0f}/{res.get('GPU',0):.0f}")
    print("(guarda la dashboard Ray)\n")

    t0 = time.time()
    for result in ray.get(futures):
        _print_result(result)

    print(f"\nTotale: {time.time()-t0:.1f}s")
    ray.shutdown()


if __name__ == "__main__":
    main()
