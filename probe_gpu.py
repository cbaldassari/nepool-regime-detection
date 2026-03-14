"""
probe_gpu.py
============
Probe GPU/CUDA environment on every Ray worker node.
Run before install_rapids.py to determine the correct package variant.

Usage:
    python probe_gpu.py
"""

import ray
import socket

RAY_ADDRESS = "ray://datalab-rayclnt.unitus.it:10001"


@ray.remote
def probe():
    import subprocess
    import sys
    import socket

    info = {"hostname": socket.gethostname(), "ip": "?", "python": sys.version.split()[0]}

    # ── IP ───────────────────────────────────────────────────────────────────
    try:
        info["ip"] = socket.gethostbyname(socket.gethostname())
    except Exception:
        pass

    # ── nvidia-smi ───────────────────────────────────────────────────────────
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        info["nvidia_smi"] = r.stdout.strip()
    except Exception as e:
        info["nvidia_smi"] = f"ERROR: {e}"

    # ── CUDA runtime version (via nvcc) ──────────────────────────────────────
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        # extract "release X.Y" from output
        for token in r.stdout.split():
            if token.startswith("V"):
                info["nvcc"] = token[1:]   # e.g. "12.4.131"
                break
        else:
            info["nvcc"] = r.stdout.strip()
    except Exception as e:
        info["nvcc"] = f"not found: {e}"

    # ── CUDA via ctypes ───────────────────────────────────────────────────────
    try:
        import ctypes
        lib = ctypes.cdll.LoadLibrary("libcuda.so.1")
        ver = ctypes.c_int(0)
        lib.cuDriverGetVersion(ctypes.byref(ver))
        v = ver.value
        info["cuda_driver"] = f"{v // 1000}.{(v % 1000) // 10}"
    except Exception as e:
        info["cuda_driver"] = f"ERROR: {e}"

    # ── cuML already installed? ───────────────────────────────────────────────
    try:
        import cuml
        info["cuml"] = cuml.__version__
    except ImportError:
        info["cuml"] = "not installed"

    # ── cupy already installed? ───────────────────────────────────────────────
    try:
        import cupy as cp
        info["cupy"] = cp.__version__
    except ImportError:
        info["cupy"] = "not installed"

    # ── RAM + CPU ────────────────────────────────────────────────────────────
    try:
        import psutil
        mem_gb = psutil.virtual_memory().total / 1024**3
        info["ram_gb"]  = f"{mem_gb:.0f}"
        info["cpu_count"] = psutil.cpu_count(logical=False)
    except Exception:
        pass

    # ── disk space on / ─────────────────────────────────────────────────────
    try:
        r = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
        lines = r.stdout.strip().splitlines()
        if len(lines) >= 2:
            info["disk"] = lines[1]
    except Exception:
        pass

    return info


def main():
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    n_cpus = int(ray.cluster_resources().get("CPU", 1))
    n_gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"\nCluster: {n_cpus} CPUs, {n_gpus} GPUs")
    print("─" * 70)

    # Launch one probe per GPU (num_gpus=1 forces each to land on a different node)
    probes = [probe.options(num_gpus=1).remote() for _ in range(n_gpus)]
    results = ray.get(probes)

    for r in results:
        print(f"\n  Node: {r.get('hostname')}  ({r.get('ip')})")
        print(f"    Python       : {r.get('python')}")
        print(f"    GPU          : {r.get('nvidia_smi')}")
        print(f"    CUDA driver  : {r.get('cuda_driver')}")
        print(f"    nvcc         : {r.get('nvcc')}")
        print(f"    cuML         : {r.get('cuml')}")
        print(f"    cupy         : {r.get('cupy')}")
        print(f"    RAM          : {r.get('ram_gb')} GB,  CPUs: {r.get('cpu_count')}")
        print(f"    Disk         : {r.get('disk')}")

    print("\n" + "─" * 70)
    # Summarise CUDA version to know which pip package to use
    cuda_versions = set()
    for r in results:
        v = r.get("cuda_driver", "")
        if v and not v.startswith("ERROR"):
            major = int(v.split(".")[0])
            cuda_versions.add(major)

    if cuda_versions:
        major = max(cuda_versions)
        pkg = f"cuml-cu{major if major >= 12 else '11'}"
        print(f"\n  → Recommended package: {pkg}")
        print(f"    Run: python install_rapids.py")
    else:
        print("\n  ⚠  Could not determine CUDA version — check nvidia-smi manually")

    print()
    ray.shutdown()


if __name__ == "__main__":
    main()
