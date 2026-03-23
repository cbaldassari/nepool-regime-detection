"""
fix_cuml_deepdive.py
====================
Deep diagnostic for the persistent cuML import failure.

The previous fix confirmed all lib dirs are present, but import still fails.
This script:
  1. Shows the FULL error (no truncation)
  2. Lists actual files in nvidia/cusparse/lib/ (symlinks vs real files)
  3. Creates missing SONAME symlinks if needed
  4. Tries ctypes pre-loading as alternative to LD_LIBRARY_PATH
  5. Writes a permanent sitecustomize.py fix using ctypes preload

Usage:
    python fix_cuml_deepdive.py
"""

import ray

RAY_ADDRESS = "ray://datalab-rayclnt.unitus.it:10001"

BASE = "/usr/local/lib/python3.12/dist-packages"


@ray.remote(num_gpus=1)
def deepdive() -> dict:
    import os, sys, subprocess, socket, glob, ctypes

    hostname = socket.gethostname()
    result = {"hostname": hostname}

    # ── 1. Full error message (no truncation) ────────────────────────────────
    env = os.environ.copy()
    # Build full LD_LIBRARY_PATH from all nvidia/* and RAPIDS lib64 dirs
    lib_dirs = (
        glob.glob(f"{BASE}/nvidia/*/lib")   +
        glob.glob(f"{BASE}/nvidia/*/lib64") +
        glob.glob(f"{BASE}/*/lib64")        +
        glob.glob(f"{BASE}/*.libs")
    )
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)

    r = subprocess.run(
        [sys.executable, "-c", "import cuml"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    result["full_error"] = r.stderr   # no truncation

    # ── 2. List contents of the key failing directories ──────────────────────
    key_dirs = {
        "cusparse" : f"{BASE}/nvidia/cusparse/lib",
        "librmm"   : f"{BASE}/librmm/lib64",
        "rl_logger": f"{BASE}/rapids_logger/lib64",
        "libcuml"  : f"{BASE}/libcuml/lib64",
    }
    dir_listings = {}
    for name, d in key_dirs.items():
        if os.path.isdir(d):
            entries = []
            for f in sorted(os.listdir(d)):
                full = os.path.join(d, f)
                if os.path.islink(full):
                    entries.append(f"{f} -> {os.readlink(full)}")
                else:
                    size = os.path.getsize(full) // 1024
                    entries.append(f"{f} ({size}KB)")
            dir_listings[name] = entries
        else:
            dir_listings[name] = [f"DIRECTORY NOT FOUND: {d}"]
    result["dir_listings"] = dir_listings

    # ── 3. Check for missing SONAME symlinks and create them ─────────────────
    symlinks_created = []
    # Pattern: libcusparse.so.12.x.y.z → create libcusparse.so.12 symlink
    for d in [f"{BASE}/nvidia/cusparse/lib",
              f"{BASE}/librmm/lib64",
              f"{BASE}/rapids_logger/lib64"]:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith(".so"):
                continue
            # e.g. f = "libcusparse.so.12.5.4.0"
            parts = f.split(".so.")
            if len(parts) != 2:
                continue
            base_name = parts[0]  # "libcusparse"
            versions = parts[1].split(".")
            if not versions:
                continue
            soname = f"{base_name}.so.{versions[0]}"   # "libcusparse.so.12"
            soname_path = os.path.join(d, soname)
            if not os.path.exists(soname_path):
                try:
                    os.symlink(f, soname_path)
                    symlinks_created.append(f"{soname_path} -> {f}")
                except Exception as e:
                    symlinks_created.append(f"FAILED {soname_path}: {e}")
    result["symlinks_created"] = symlinks_created

    # ── 4. ctypes preload approach ───────────────────────────────────────────
    # Load RAPIDS deps explicitly in dependency order before importing cuml.
    # This bypasses rpath issues by ensuring all .so are already in memory.
    preload_order = [
        # CUDA runtime deps (from nvidia-* packages)
        f"{BASE}/nvidia/cuda_runtime/lib/libcudart.so.12",
        f"{BASE}/nvidia/cublas/lib/libcublas.so.12",
        f"{BASE}/nvidia/cublas/lib/libcublasLt.so.12",
        f"{BASE}/nvidia/cusparse/lib/libcusparse.so.12",
        f"{BASE}/nvidia/cusolver/lib/libcusolver.so.11",
        f"{BASE}/nvidia/cusolver/lib/libcusolver.so.11.7",  # alt name
        f"{BASE}/nvidia/cufft/lib/libcufft.so.11",
        f"{BASE}/nvidia/curand/lib/libcurand.so.10",
        f"{BASE}/nvidia/nvjitlink/lib/libnvJitLink.so.12",
        # RAPIDS core libs in dependency order
        f"{BASE}/rapids_logger/lib64/librapids_logger.so",
        f"{BASE}/librmm/lib64/librmm.so",
        f"{BASE}/libraft/lib64/libraft.so",
        f"{BASE}/libcuml/lib64/libcuml++.so",
    ]

    preload_results = {}
    for lib in preload_order:
        if not os.path.exists(lib):
            # Try without version suffix (find matching file)
            d = os.path.dirname(lib)
            base = os.path.basename(lib).split(".so")[0] + ".so"
            candidates = [f for f in (os.listdir(d) if os.path.isdir(d) else [])
                         if f.startswith(base.split(".so")[0]) and ".so" in f]
            if candidates:
                lib = os.path.join(d, sorted(candidates)[0])
            else:
                preload_results[os.path.basename(lib)] = "FILE NOT FOUND"
                continue
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            preload_results[os.path.basename(lib)] = "OK"
        except OSError as e:
            preload_results[os.path.basename(lib)] = f"FAILED: {e}"
    result["preload_results"] = preload_results

    # ── 5. Test import after ctypes preload ───────────────────────────────────
    # Build a Python script that pre-loads via ctypes then imports cuml
    preload_script = f"""
import ctypes, os, glob

BASE = {BASE!r}
lib_dirs = (
    glob.glob(BASE + "/nvidia/*/lib")   +
    glob.glob(BASE + "/nvidia/*/lib64") +
    glob.glob(BASE + "/*/lib64")        +
    glob.glob(BASE + "/*.libs")
)
os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs)

preload = [
    BASE + "/rapids_logger/lib64/librapids_logger.so",
    BASE + "/librmm/lib64/librmm.so",
    BASE + "/libraft/lib64/libraft.so",
    BASE + "/libcuml/lib64/libcuml++.so",
]
for lib in preload:
    if os.path.exists(lib):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

import cuml
from cuml.manifold import UMAP
import numpy as np
Z = UMAP(n_components=2, n_neighbors=10, random_state=42).fit_transform(
    np.random.randn(200, 10).astype("float32"))
print(f"cuML {{cuml.__version__}}  UMAP OK  shape={{tuple(Z.shape)}}")
"""

    r2 = subprocess.run(
        [sys.executable, "-c", preload_script],
        capture_output=True, text=True, timeout=60, env=env,
    )
    result["ctypes_test"] = {
        "rc"     : r2.returncode,
        "stdout" : r2.stdout.strip(),
        "stderr" : r2.stderr[-500:].strip(),
    }

    # ── 6. If ctypes preload works, write permanent sitecustomize ─────────────
    if r2.returncode == 0:
        for site_dir in sys.path:
            if ("dist-packages" in site_dir or "site-packages" in site_dir) \
                    and os.path.isdir(site_dir):
                sc = os.path.join(site_dir, "sitecustomize.py")
                existing = open(sc).read() if os.path.exists(sc) else ""
                if "rapids_preload" not in existing:
                    with open(sc, "a") as f:
                        f.write(f"""
# rapids_preload — auto-generated by fix_cuml_deepdive.py
import ctypes as _ctypes, os as _os, glob as _glob
_BASE = {BASE!r}
_lib_dirs = (
    _glob.glob(_BASE + "/nvidia/*/lib")   +
    _glob.glob(_BASE + "/nvidia/*/lib64") +
    _glob.glob(_BASE + "/*/lib64")        +
    _glob.glob(_BASE + "/*.libs")
)
_os.environ["LD_LIBRARY_PATH"] = ":".join(_lib_dirs)
for _lib in [
    _BASE + "/rapids_logger/lib64/librapids_logger.so",
    _BASE + "/librmm/lib64/librmm.so",
    _BASE + "/libraft/lib64/libraft.so",
    _BASE + "/libcuml/lib64/libcuml++.so",
]:
    if _os.path.exists(_lib):
        try: _ctypes.CDLL(_lib, mode=_ctypes.RTLD_GLOBAL)
        except Exception: pass
""")
                result["sitecustomize"] = sc
                break
        result["status"] = "OK — ctypes preload works, sitecustomize written"
    else:
        result["status"] = "FAILED — even ctypes preload cannot import cuml"

    return result


def main():
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    n_gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"\nCluster GPUs: {n_gpus}  — deep diagnostic\n")

    futures = [deepdive.remote() for _ in range(n_gpus)]
    results = ray.get(futures)

    seen = set()
    for r in results:
        host = r.get("hostname", "?")
        if host in seen:
            continue
        seen.add(host)

        print(f"\n{'═'*65}")
        print(f"  Node: {host}  —  {r.get('status', '?')}")
        print(f"{'─'*65}")

        # Directory listings
        for name, entries in r.get("dir_listings", {}).items():
            print(f"\n  [{name}]")
            for e in entries[:6]:
                print(f"    {e}")
            if len(entries) > 6:
                print(f"    ... ({len(entries)-6} more)")

        # Symlinks created
        sl = r.get("symlinks_created", [])
        if sl:
            print(f"\n  Symlinks created: {len(sl)}")
            for s in sl:
                print(f"    {s}")

        # ctypes preload results
        print(f"\n  ctypes preload:")
        for lib, status in r.get("preload_results", {}).items():
            icon = "✓" if status == "OK" else "✗"
            print(f"    {icon} {lib}: {status[:80]}")

        # ctypes test result
        ct = r.get("ctypes_test", {})
        ok = "✓ OK" if ct.get("rc") == 0 else "✗ FAILED"
        print(f"\n  Import after ctypes preload: {ok}")
        if ct.get("stdout"):
            print(f"    {ct['stdout']}")
        if ct.get("rc") != 0 and ct.get("stderr"):
            # Show last part of error (most relevant)
            err_lines = ct["stderr"].splitlines()
            print(f"    Last error lines:")
            for line in err_lines[-5:]:
                print(f"      {line}")

        # sitecustomize
        if "sitecustomize" in r:
            print(f"\n  ✓ sitecustomize written: {r['sitecustomize']}")

        # Full error (only if still failing)
        if ct.get("rc") != 0:
            print(f"\n  Full original error:")
            for line in r.get("full_error", "").splitlines()[-15:]:
                print(f"    {line}")

    print(f"\n{'═'*65}\n")
    ray.shutdown()


if __name__ == "__main__":
    main()
