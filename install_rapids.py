"""
install_rapids.py
=================
Install RAPIDS cuML on every GPU worker node of the Ray cluster.

RAPIDS provides GPU-accelerated UMAP (10-50x faster than CPU umap-learn)
and HDBSCAN. We install it directly on the nodes via subprocess pip calls
(not via runtime_env, which would re-download on every Ray restart).

Usage:
    python install_rapids.py            # auto-detect CUDA version
    python install_rapids.py --cuda 12  # force CUDA 12 variant
    python install_rapids.py --dry-run  # print commands without running

Package sizes (approximate, downloaded once per node):
    cuml-cu12  ~800 MB
    cupy-cuda12x ~100 MB

After install, restart the Ray workers or relaunch step03 — the packages
will be available on all nodes without runtime_env overhead.
"""

import sys
import ray
import socket
import subprocess
import argparse

RAY_ADDRESS = "ray://datalab-rayclnt.unitus.it:10001"

# RAPIDS pypi index (official NVIDIA pip index)
NVIDIA_INDEX = "https://pypi.nvidia.com"


def detect_cuda_major() -> int:
    """Detect CUDA major version from nvidia-smi on the local node (proxy)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        # driver version doesn't map directly to CUDA — use nvcc instead
        r2 = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        for token in r2.stdout.split():
            if token.startswith("V"):
                major = int(token[1:].split(".")[0])
                return major
    except Exception:
        pass
    return 12   # safe default — CUDA 12 is current standard


@ray.remote(num_gpus=1)
def install_on_node(cuda_major: int, dry_run: bool = False) -> dict:
    """
    Install cuML + cupy on this worker node via pip.
    Runs as a Ray task with num_gpus=1 so Ray schedules exactly one
    task per GPU node.
    """
    import subprocess
    import sys
    import socket

    hostname = socket.gethostname()
    results  = {"hostname": hostname, "steps": []}

    # ── Determine package variant ────────────────────────────────────────────
    if cuda_major >= 12:
        cuml_pkg = "cuml-cu12"
        cupy_pkg = "cupy-cuda12x"
    else:
        cuml_pkg = "cuml-cu11"
        cupy_pkg = "cupy-cuda11x"

    packages = [
        cuml_pkg,
        cupy_pkg,
    ]

    def run_pip(pkg: str) -> dict:
        cmd = [
            sys.executable, "-m", "pip", "install",
            pkg,
            "--extra-index-url", "https://pypi.nvidia.com",
            "--quiet",
            "--no-warn-script-location",
            "--break-system-packages",   # Ubuntu 22.04+ externally-managed-environment
        ]
        if dry_run:
            return {"pkg": pkg, "cmd": " ".join(cmd), "rc": 0, "dry_run": True}

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        return {
            "pkg"    : pkg,
            "rc"     : r.returncode,
            "stdout" : r.stdout[-300:].strip() if r.stdout else "",
            "stderr" : r.stderr[-300:].strip() if r.stderr else "",
        }

    for pkg in packages:
        step = run_pip(pkg)
        results["steps"].append(step)

    # ── Verify ──────────────────────────────────────────────────────────────
    if not dry_run:
        try:
            import cuml
            results["cuml_version"] = cuml.__version__
            results["ok"] = True
        except ImportError as e:
            results["cuml_version"] = f"IMPORT FAILED: {e}"
            results["ok"] = False
        try:
            import cupy as cp
            results["cupy_version"] = cp.__version__
            results["cuda_version"] = str(cp.cuda.runtime.runtimeGetVersion())
        except ImportError as e:
            results["cupy_version"] = f"IMPORT FAILED: {e}"
    else:
        results["ok"] = True
        results["dry_run"] = True

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=None,
                        help="CUDA major version to target (11 or 12). Auto-detected if omitted.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pip commands without executing them.")
    args = parser.parse_args()

    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    n_gpus = int(ray.cluster_resources().get("GPU", 0))

    if n_gpus == 0:
        print("⚠  No GPU resources found in cluster. Exiting.")
        ray.shutdown()
        sys.exit(1)

    # Detect CUDA version
    cuda_major = args.cuda
    if cuda_major is None:
        print("  Auto-detecting CUDA version via probe task...")

        @ray.remote(num_gpus=1)
        def _detect():
            import subprocess, re
            # Try nvcc: output is like "Cuda compilation tools, release 12.4, V12.4.131"
            try:
                r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
                m = re.search(r"release\s+(\d+)\.\d+", r.stdout)
                if m:
                    return int(m.group(1))
            except Exception:
                pass
            # Fallback: nvidia-smi header line "CUDA Version: 12.4"
            try:
                r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                m = re.search(r"CUDA Version:\s*(\d+)\.\d+", r.stdout)
                if m:
                    return int(m.group(1))
            except Exception:
                pass
            return 12  # safe default

        cuda_major = ray.get(_detect.remote())
        # RAPIDS only publishes cu11 and cu12 variants — cap accordingly
        if cuda_major > 12:
            print(f"  Detected CUDA {cuda_major} — RAPIDS max supported is 12, using cu12")
            cuda_major = 12
        print(f"  Using CUDA major version: {cuda_major}")

    pkg_variant = f"cuml-cu{cuda_major if cuda_major >= 12 else '11'}"
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Installing {pkg_variant} on {n_gpus} GPU nodes")
    print(f"  Index: {NVIDIA_INDEX}")
    print(f"  This may take 5-10 minutes per node (first install ~900 MB)\n")

    # Launch one install task per GPU node
    futures = [install_on_node.remote(cuda_major, args.dry_run) for _ in range(n_gpus)]
    results = ray.get(futures)

    print("─" * 65)
    all_ok = True
    for r in results:
        host = r.get("hostname", "?")
        ok   = r.get("ok", False)
        print(f"\n  Node: {host}  {'✓ OK' if ok else '✗ FAILED'}")

        if r.get("dry_run"):
            for step in r.get("steps", []):
                print(f"    [dry-run] {step.get('cmd')}")
        else:
            for step in r.get("steps", []):
                rc  = step.get("rc", -1)
                pkg = step.get("pkg", "?")
                status = "✓" if rc == 0 else f"✗ (rc={rc})"
                print(f"    {status}  pip install {pkg}")
                if rc != 0 and step.get("stderr"):
                    print(f"       stderr: {step['stderr'][:200]}")

            print(f"    cuML   : {r.get('cuml_version', 'N/A')}")
            print(f"    cupy   : {r.get('cupy_version', 'N/A')}")
            if "cuda_version" in r:
                v = int(r["cuda_version"])
                print(f"    CUDA RT: {v // 1000}.{(v % 1000) // 10}")

        if not ok:
            all_ok = False

    print("\n" + "─" * 65)
    if args.dry_run:
        print("  Dry run complete — no packages were installed.")
    elif all_ok:
        print("  ✓  RAPIDS installed successfully on all nodes.")
        print("  → You can now run step03 with cuML UMAP acceleration.")
        print("  → Remove runtime_env pip entries for umap-learn/hdbscan")
        print("    (cuml includes its own UMAP; keep hdbscan for DBCV).")
    else:
        print("  ⚠  Some nodes failed. Check stderr above.")
        print("  Common causes:")
        print("    - CUDA version mismatch (try --cuda 11 or --cuda 12)")
        print("    - Insufficient disk space (cuML needs ~2GB free)")
        print("    - Network timeout (try again — PyPI can be slow)")

    print()
    ray.shutdown()


if __name__ == "__main__":
    main()
