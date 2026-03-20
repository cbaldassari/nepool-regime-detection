"""
step03_clustering.py
====================
NEPOOL Regime Detection — Step 03: UMAP + Grid Search + HDBSCAN

Backend
-------
  GPU (cuML)   — priorità, se cuML disponibile sui nodi.
                  • cuml.manifold.UMAP  (10-100× più veloce di CPU)
                  • cuml.cluster.HDBSCAN con relative_validity_ (DBCV reale)
                  • 1 task / GPU  → 3 concurrent (uno per nodo)
  CPU fallback  — umap-learn + sklearn HDBSCAN + silhouette score
                  • 2 CPU / task  → 12 concurrent

Input  : results/embeddings.parquet
Output : results/grid_results.csv      (tutti i trial + score)
         results/best_params.json      (migliori iperparametri)
         results/umap.parquet          (proiezione 2D ottimale)
         results/regimes.parquet       (etichetta regime per finestra)
Plots  : results/step03/

Logica di resume
----------------
  ┌─ grid_results.csv esiste?
  │     SÌ  → salta la ricerca, carica CSV
  │     NO  → cerca optuna_study.db
  │             esiste? → riprende i trial completati
  │             no?     → parte da zero
  └─ umap.parquet + regimes.parquet esistono?
        SÌ  → salta final_run
        NO  → esegue final_run con best params
"""

import json, math, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  Configurazione
# ═══════════════════════════════════════════════════════════════════════════

UMAP_N_COMPONENTS     = 20   # dimensioni per clustering HDBSCAN
UMAP_VIZ_COMPONENTS   = 2    # dimensioni per visualizzazione 2D
RANDOM_STATE          = 42
RAY_ADDRESS       = "ray://datalab-rayclnt.unitus.it:10001"
CPUS_PER_TASK     = 2          # usato solo in modalità CPU
PLOT_DIR          = Path(C.RESULTS_DIR) / "step03"

# ── Modalità sviluppo ────────────────────────────────────────────────────
# True  → cancella tutti i checkpoint ad ogni run (sempre da zero)
# False → riprende da dove era rimasto (comportamento produzione)
FRESH_START = True

# Path dove pip installa i pacchetti di sistema sui nodi
CUML_BASE = "/usr/local/lib/python3.12/dist-packages"

SEARCH_SPACE = {
    "n_neighbors"      : [5, 10, 15, 20, 30, 50, 75, 100],
    "min_dist"         : [0.0, 0.05, 0.1, 0.2, 0.3],
    "min_cluster_size" : [100, 150, 200, 300, 500, 750, 1000, 1500],
}
N_TRIALS       = math.prod(len(v) for v in SEARCH_SPACE.values())   # 320 full grid
MAX_NOISE_FRAC = 0.05   # trial con noise > 5%  → score=nan, esclusi dal ranking
MAX_CLUSTERS   = 8      # trial con k > MAX_CLUSTERS → score=nan (troppi regimi)



# ═══════════════════════════════════════════════════════════════════════════
#  cuML preload  (risolve "undefined symbol" su certi setup)
# ═══════════════════════════════════════════════════════════════════════════

def _cuml_preload():
    """
    Carica le shared library RAPIDS via ctypes prima di importare cuml.
    Serve quando LD_LIBRARY_PATH non contiene automaticamente i path
    delle librerie nvidia-* installate da pip.
    """
    import os, glob, ctypes
    B = CUML_BASE
    lib_dirs = (
        glob.glob(f"{B}/nvidia/*/lib")   +
        glob.glob(f"{B}/nvidia/*/lib64") +
        glob.glob(f"{B}/*/lib64")        +
        glob.glob(f"{B}/*.libs")
    )
    if lib_dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            ":".join(lib_dirs) + (":" + existing if existing else "")
        )
    # Ordine di caricamento: dipendenze prima di cuml++
    for lib_path in [
        f"{B}/rapids_logger/lib64/librapids_logger.so",
        f"{B}/librmm/lib64/librmm.so",
        f"{B}/libraft/lib64/libraft.so",
        f"{B}/libcuml/lib64/libcuml++.so",
    ]:
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  Fase A — UMAP only (module-level, GPU-first con fallback CPU)
#
#  UMAP dipende solo da (n_neighbors, min_dist), non da min_cluster_size.
#  Con lo spazio 8×5×8 = 320 trial ci sono solo 40 coppie (nn,md) uniche.
#  Questa funzione calcola l'embedding UMAP 20D e lo restituisce come bytes
#  tuple (no numpy ABI crossing).  Il risultato viene salvato su disco dal
#  client — immune al crash, riusabile senza Ray.
# ═══════════════════════════════════════════════════════════════════════════

def compute_umap_only(E_payload: tuple, n_neighbors: int, min_dist: float) -> tuple:
    """
    Calcola solo UMAP 20D su (n_neighbors, min_dist).
    Restituisce Z come bytes tuple per evitare il mismatch pickle5.
    """
    import numpy as np
    raw_bytes, shape, dtype_str = E_payload
    E = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()

    # ── Tentativo GPU (cuML) ─────────────────────────────────────────────
    try:
        import os, glob, ctypes as _ctypes
        _B = "/usr/local/lib/python3.12/dist-packages"
        _dirs = (glob.glob(f"{_B}/nvidia/*/lib")   + glob.glob(f"{_B}/nvidia/*/lib64") +
                 glob.glob(f"{_B}/*/lib64")         + glob.glob(f"{_B}/*.libs"))
        if _dirs:
            os.environ["LD_LIBRARY_PATH"] = ":".join(_dirs)
        for _lib in [f"{_B}/rapids_logger/lib64/librapids_logger.so",
                     f"{_B}/librmm/lib64/librmm.so",
                     f"{_B}/libraft/lib64/libraft.so",
                     f"{_B}/libcuml/lib64/libcuml++.so"]:
            if os.path.exists(_lib):
                try: _ctypes.CDLL(_lib, mode=_ctypes.RTLD_GLOBAL)
                except OSError: pass

        from cuml.manifold import UMAP as cuUMAP
        import cupy as cp

        E_gpu = cp.asarray(E.astype(np.float32))
        Z_gpu = cuUMAP(
            n_components = UMAP_N_COMPONENTS,
            n_neighbors  = n_neighbors,
            min_dist     = min_dist,
            metric       = "cosine",
            random_state = RANDOM_STATE,
            n_epochs     = 100,
        ).fit_transform(E_gpu)
        Z = cp.asnumpy(Z_gpu).astype(np.float32)
        return (Z.tobytes(), Z.shape, Z.dtype.str, "GPU")

    except Exception:
        pass

    # ── Fallback CPU ─────────────────────────────────────────────────────
    import umap as _umap
    Z = _umap.UMAP(
        n_components = UMAP_N_COMPONENTS,
        n_neighbors  = n_neighbors,
        min_dist     = min_dist,
        metric       = "cosine",
        random_state = RANDOM_STATE,
        low_memory   = True,
        n_epochs     = 100,
        n_jobs       = 1,
    ).fit_transform(E).astype(np.float32)
    return (Z.tobytes(), Z.shape, Z.dtype.str, "CPU")


# ═══════════════════════════════════════════════════════════════════════════
#  Fase B — HDBSCAN sweep su embedding UMAP pre-calcolato
# ═══════════════════════════════════════════════════════════════════════════

def run_hdbscan_sweep_from_z(Z_payload: tuple, n_neighbors: int, min_dist: float,
                              mcs_list: list) -> list:
    """
    Esegue UMAP una sola volta per (n_neighbors, min_dist), poi fa sweep di
    HDBSCAN su tutti i valori di min_cluster_size in mcs_list.
    Dato l'embedding UMAP 20D pre-calcolato, esegue solo HDBSCAN per tutti i
    min_cluster_size in mcs_list.  UMAP non viene mai ricalcolata qui.

    Z_payload : (raw_bytes, shape, dtype_str, backend_str)
        backend_str indica se Z è stato prodotto da GPU o CPU (per il report).
    """
    import numpy as np
    raw_bytes, shape, dtype_str, umap_backend = Z_payload
    Z = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()

    results     = []
    gpu_err_msg = None

    # ── Tentativo GPU (cuML HDBSCAN) ─────────────────────────────────────
    try:
        import os, glob, ctypes as _ctypes
        _B = "/usr/local/lib/python3.12/dist-packages"
        _dirs = (glob.glob(f"{_B}/nvidia/*/lib")   + glob.glob(f"{_B}/nvidia/*/lib64") +
                 glob.glob(f"{_B}/*/lib64")         + glob.glob(f"{_B}/*.libs"))
        if _dirs:
            os.environ["LD_LIBRARY_PATH"] = ":".join(_dirs)
        for _lib in [f"{_B}/rapids_logger/lib64/librapids_logger.so",
                     f"{_B}/librmm/lib64/librmm.so",
                     f"{_B}/libraft/lib64/libraft.so",
                     f"{_B}/libcuml/lib64/libcuml++.so"]:
            if os.path.exists(_lib):
                try: _ctypes.CDLL(_lib, mode=_ctypes.RTLD_GLOBAL)
                except OSError: pass

        from cuml.cluster import HDBSCAN as cuHDBSCAN
        import cupy as cp

        Z_gpu = cp.asarray(Z.astype(np.float32))

        for mcs in mcs_list:
            hdb = cuHDBSCAN(
                min_cluster_size         = mcs,
                metric                   = "euclidean",
                cluster_selection_method = "eom",
                gen_min_span_tree        = True,
            )
            hdb.fit(Z_gpu)
            labels     = cp.asnumpy(hdb.labels_)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_frac = float((labels == -1).mean())

            if n_clusters > MAX_CLUSTERS:
                score, metric_used = float("nan"), "too_many_clusters"
            elif noise_frac > MAX_NOISE_FRAC:
                score, metric_used = float("nan"), "noise_exceed"
            elif n_clusters >= 2 and hasattr(hdb, "relative_validity_"):
                score, metric_used = float(hdb.relative_validity_), "dbcv"
            elif n_clusters >= 2:
                from sklearn.metrics import silhouette_score
                mask = labels != -1
                score = (float(silhouette_score(Z[mask], labels[mask]))
                         if mask.sum() > n_clusters else float("nan"))
                metric_used = "silhouette"
            else:
                score, metric_used = float("nan"), "nan"

            results.append({
                "n_neighbors"      : n_neighbors,
                "min_dist"         : min_dist,
                "min_cluster_size" : mcs,
                "n_clusters"       : n_clusters,
                "noise_frac"       : noise_frac,
                "dbcv"             : score,
                "backend"          : umap_backend,
                "metric"           : metric_used,
            })
        return results

    except Exception as e:
        gpu_err_msg = str(e)[:80]

    # ── Fallback CPU (sklearn HDBSCAN) ────────────────────────────────────
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import silhouette_score

    for mcs in mcs_list:
        labels     = HDBSCAN(
            min_cluster_size         = mcs,
            metric                   = "euclidean",
            cluster_selection_method = "eom",
        ).fit_predict(Z)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = float((labels == -1).mean())
        mask       = labels != -1

        if n_clusters > MAX_CLUSTERS:
            score = float("nan")
        elif noise_frac > MAX_NOISE_FRAC:
            score = float("nan")
        elif n_clusters >= 2 and mask.sum() > n_clusters:
            score = float(silhouette_score(Z[mask], labels[mask], metric="euclidean"))
        else:
            score = float("nan")

        results.append({
            "n_neighbors"      : n_neighbors,
            "min_dist"         : min_dist,
            "min_cluster_size" : mcs,
            "n_clusters"       : n_clusters,
            "noise_frac"       : noise_frac,
            "dbcv"             : score,
            "backend"          : "CPU",
            "metric"           : "silhouette",
            "gpu_err"          : gpu_err_msg,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Grid search su Ray  (auto GPU / CPU)
# ═══════════════════════════════════════════════════════════════════════════

def grid_search(E_pca: np.ndarray, ckpt_path: Path) -> pd.DataFrame:
    """
    Grid search a due fasi con cache UMAP su disco.

    Fase A — UMAP (Ray, GPU-first):
        40 task (uno per coppia nn×md), ogni task calcola solo UMAP 20D.
        Risultato salvato in results/umap_cache/nn{nn:03d}_md{md:.3f}.npy.
        Su resume: il file .npy esiste → UMAP saltata completamente.

    Fase B — HDBSCAN sweep (Ray, GPU-first):
        40 task, ciascuno carica Z dalla cache e fa sweep dei 8 mcs.
        Checkpoint CSV scritto dopo ogni task.
        Su resume: le coppie (nn,md,mcs) già nel CSV vengono saltate.

    Vantaggi rispetto al design precedente:
    - Crash durante UMAP → lavoro UMAP non perso (file .npy già scritto)
    - Cambio parametri HDBSCAN → nessun ricalcolo UMAP
    - Separazione netta del collo di bottiglia (UMAP) dal sweep veloce (HDBSCAN)
    """
    import ray, itertools

    # ── Ray ─────────────────────────────────────────────────────────────
    print(f"  Connessione a Ray: {RAY_ADDRESS} ...", flush=True)
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True, log_to_driver=False)
    print("  Ray connesso.", flush=True)

    EXPECTED_GPUS = 3
    _gpus = int(ray.cluster_resources().get("GPU", 0))
    if _gpus == 0:
        print(f"  ⚠ Nessuna GPU — attendo max 60s...", flush=True)
        _t0 = time.time()
        while int(ray.cluster_resources().get("GPU", 0)) == 0:
            if time.time() - _t0 > 60: break
            time.sleep(5)
        _gpus = int(ray.cluster_resources().get("GPU", 0))
    elif _gpus < EXPECTED_GPUS:
        _t0 = time.time()
        while int(ray.cluster_resources().get("GPU", 0)) < EXPECTED_GPUS:
            if time.time() - _t0 > 30: break
            time.sleep(5)
        _gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"  {'✓' if _gpus >= EXPECTED_GPUS else '⚠'} GPU disponibili: {_gpus}", flush=True)

    n_cpu = int(ray.cluster_resources().get("CPU", 1))
    n_gpu = _gpus

    N_CONCURRENT = n_gpu if n_gpu > 0 else max(1, n_cpu // CPUS_PER_TASK)
    use_gpu      = n_gpu > 0

    if use_gpu:
        print(f"  Modalità GPU  — {n_gpu} GPU  N_CONCURRENT={N_CONCURRENT}")
        print(f"  Backend UMAP  : cuML UMAP   Backend HDBSCAN: cuML HDBSCAN (DBCV)")
    else:
        print(f"  Modalità CPU  — {n_cpu} CPU  N_CONCURRENT={N_CONCURRENT}")
        print(f"  Backend UMAP  : umap-learn  Backend HDBSCAN: sklearn (silhouette)")

    mcs_list   = SEARCH_SPACE["min_cluster_size"]
    all_groups = list(itertools.product(
        SEARCH_SPACE["n_neighbors"],
        SEARCH_SPACE["min_dist"],
    ))
    N_GROUPS   = len(all_groups)

    # ── UMAP cache dir ───────────────────────────────────────────────────
    umap_cache_dir = ckpt_path.parent / "umap_cache"
    umap_cache_dir.mkdir(exist_ok=True)

    def _umap_cache_path(nn, md):
        return umap_cache_dir / f"nn{int(nn):03d}_md{float(md):.3f}.npy"

    # ────────────────────────────────────────────────────────────────────
    # FASE A — UMAP
    # ────────────────────────────────────────────────────────────────────
    umap_todo = [(nn, md) for nn, md in all_groups
                 if not _umap_cache_path(nn, md).exists()]
    n_umap_cached = N_GROUPS - len(umap_todo)

    print(f"\n  [A] UMAP  — {N_GROUPS} coppie nn×md  "
          f"({n_umap_cached} già in cache, {len(umap_todo)} da calcolare)", flush=True)

    if umap_todo:
        if use_gpu:
            ray_umap = ray.remote(num_gpus=1)(compute_umap_only)
        else:
            ray_umap = ray.remote(num_cpus=CPUS_PER_TASK)(compute_umap_only)

        E_ref        = ray.put((E_pca.tobytes(), E_pca.shape, E_pca.dtype.str))
        umap_iter    = iter(umap_todo)
        umap_flight  = {}   # ref → (nn, md)
        umap_done    = 0
        umap_failed  = []
        MAX_UMAP_S   = 300  # timeout per singolo task UMAP

        for _ in range(min(N_CONCURRENT, len(umap_todo))):
            nn, md = next(umap_iter)
            ref    = ray_umap.remote(E_ref, nn, md)
            umap_flight[ref] = (nn, md)
        submitted_u = len(umap_flight)

        print(f"  {'umap':>4}  {'nn':>4} {'md':>5}  {'backend':>7}  {'status':>12}",
              flush=True)

        while umap_flight:
            done_refs, _ = ray.wait(list(umap_flight.keys()), num_returns=1,
                                    timeout=MAX_UMAP_S)
            if not done_refs:
                hung_ref     = next(iter(umap_flight))
                nn, md       = umap_flight.pop(hung_ref)
                print(f"  ⏱ timeout UMAP nn={nn} md={md} — saltato", flush=True)
                try: ray.cancel(hung_ref, force=True)
                except Exception: pass
                umap_failed.append((nn, md))
                umap_done += 1
                if submitted_u < len(umap_todo):
                    try:
                        nn2, md2 = next(umap_iter)
                        ref2     = ray_umap.remote(E_ref, nn2, md2)
                        umap_flight[ref2] = (nn2, md2)
                        submitted_u += 1
                    except StopIteration: pass
                continue

            ref      = done_refs[0]
            nn, md   = umap_flight.pop(ref)
            try:
                raw_bytes, shape, dtype_str, bk = ray.get(ref)
                Z = np.frombuffer(raw_bytes,
                                  dtype=np.dtype(dtype_str)).reshape(shape).copy()
                np.save(_umap_cache_path(nn, md), Z)
                print(f"  {umap_done+1:>4}/{N_GROUPS}  "
                      f"nn={int(nn):>3} md={float(md):.2f}  "
                      f"bk={bk:>3}  saved→cache", flush=True)
            except Exception as e:
                print(f"  ⚠ UMAP fallita nn={nn} md={md}: {e}", flush=True)
                umap_failed.append((nn, md))

            umap_done += 1
            if submitted_u < len(umap_todo):
                try:
                    nn2, md2 = next(umap_iter)
                    ref2     = ray_umap.remote(E_ref, nn2, md2)
                    umap_flight[ref2] = (nn2, md2)
                    submitted_u += 1
                except StopIteration: pass

        if umap_failed:
            print(f"  ⚠ {len(umap_failed)} coppie UMAP fallite — escluse da HDBSCAN.",
                  flush=True)

    # ────────────────────────────────────────────────────────────────────
    # FASE B — HDBSCAN sweep su embedding in cache
    # ────────────────────────────────────────────────────────────────────
    done_keys = set()
    results   = []
    if ckpt_path.exists():
        ckpt_df = pd.read_csv(ckpt_path)
        for _, row in ckpt_df.iterrows():
            results.append(row.to_dict())
            done_keys.add((row["n_neighbors"], row["min_dist"], row["min_cluster_size"]))

    hdb_todo = []
    for nn, md in all_groups:
        if not _umap_cache_path(nn, md).exists():
            continue   # UMAP fallita, skip
        remaining = [mcs for mcs in mcs_list if (nn, md, mcs) not in done_keys]
        if remaining:
            hdb_todo.append((nn, md, remaining))

    n_hdb_done = len(done_keys)
    print(f"\n  [B] HDBSCAN — {N_GROUPS} gruppi × {len(mcs_list)} mcs = {N_TRIALS} trial  "
          f"({n_hdb_done} già fatti, {len(hdb_todo)} gruppi rimasti)", flush=True)

    if not hdb_todo:
        print("  ✓ Tutti i trial HDBSCAN già completati.", flush=True)
    else:
        if use_gpu:
            ray_hdb = ray.remote(num_gpus=1)(run_hdbscan_sweep_from_z)
        else:
            ray_hdb = ray.remote(num_cpus=CPUS_PER_TASK)(run_hdbscan_sweep_from_z)

        CSV_COLS     = ["n_neighbors", "min_dist", "min_cluster_size",
                        "n_clusters", "noise_frac", "dbcv", "backend", "metric"]
        write_header = not ckpt_path.exists()
        hdb_iter     = iter(hdb_todo)
        hdb_flight   = {}   # ref → (nn, md, rem_mcs)
        hdb_done_g   = 0
        best_dbcv    = max(
            (r["dbcv"] for r in results
             if r.get("dbcv") is not None and not np.isnan(float(r["dbcv"]))),
            default=float("-inf"),
        )
        MAX_HDB_S    = 180   # timeout per singolo sweep HDBSCAN

        def _dispatch_hdb():
            nn, md, rem_mcs = next(hdb_iter)
            Z   = np.load(_umap_cache_path(nn, md)).astype(np.float32)
            bk  = "GPU" if use_gpu else "CPU"
            Z_p = (Z.tobytes(), Z.shape, Z.dtype.str, bk)
            ref = ray_hdb.remote(ray.put(Z_p), nn, md, rem_mcs)
            return ref, (nn, md, rem_mcs)

        for _ in range(min(N_CONCURRENT, len(hdb_todo))):
            ref, combo = _dispatch_hdb()
            hdb_flight[ref] = combo
        submitted_h = len(hdb_flight)

        print(f"\n  {'gr':>4}  {'nn':>4} {'md':>5}  {'mcs_range':>14}  "
              f"{'best_score':>10}  {'k_range':>8}  {'bk':>3}", flush=True)

        t_start = time.time()
        while hdb_flight:
            done_refs, _ = ray.wait(list(hdb_flight.keys()), num_returns=1,
                                    timeout=MAX_HDB_S)
            if not done_refs:
                hung_ref          = next(iter(hdb_flight))
                nn, md, rem_mcs   = hdb_flight.pop(hung_ref)
                print(f"  ⏱ timeout HDBSCAN nn={nn} md={md} — saltato", flush=True)
                try: ray.cancel(hung_ref, force=True)
                except Exception: pass
                hdb_done_g += 1
                if submitted_h < len(hdb_todo):
                    try:
                        ref, combo = _dispatch_hdb()
                        hdb_flight[ref] = combo
                        submitted_h += 1
                    except StopIteration: pass
                continue

            ref             = done_refs[0]
            nn, md, rem_mcs = hdb_flight.pop(ref)
            try:
                r_list = ray.get(ref)
            except Exception as e:
                print(f"  ⚠ HDBSCAN fallito nn={nn} md={md}: {e}", flush=True)
                hdb_done_g += 1
                if submitted_h < len(hdb_todo):
                    try:
                        ref, combo = _dispatch_hdb()
                        hdb_flight[ref] = combo
                        submitted_h += 1
                    except StopIteration: pass
                continue

            batch_df = pd.DataFrame([{k: r.get(k) for k in CSV_COLS} for r in r_list])
            batch_df.to_csv(ckpt_path, mode="a", header=write_header, index=False)
            write_header = False
            results.extend(r_list)

            valid_scores = [float(r["dbcv"]) for r in r_list
                            if r.get("dbcv") is not None
                            and not np.isnan(float(r["dbcv"]))]
            group_best = max(valid_scores) if valid_scores else float("-inf")
            is_best    = np.isfinite(group_best) and group_best > best_dbcv
            if is_best:
                best_dbcv = group_best

            k_vals    = [int(r["n_clusters"]) for r in r_list]
            sc_str    = f"{group_best:.4f}" if np.isfinite(group_best) else "    nan"
            best_str  = f"global={best_dbcv:.4f}" if np.isfinite(best_dbcv) else "global=n/a"
            bk_short  = r_list[0].get("backend", "?")[:3]
            elapsed_s = time.time() - t_start
            rate      = (hdb_done_g + 1) / elapsed_s if elapsed_s > 0 else 0
            rem_est   = (N_GROUPS - hdb_done_g - 1) / rate if rate > 0 else 0
            eta_str   = f"{int(rem_est//60)}m{int(rem_est%60):02d}s" if rate > 0 else "?"

            print(
                f"  {hdb_done_g+1:>4}/{N_GROUPS}  "
                f"nn={int(nn):>3} md={float(md):.2f}  "
                f"mcs=[{min(rem_mcs):>4}..{max(rem_mcs):>4}]  "
                f"best={sc_str}  k=[{min(k_vals)}..{max(k_vals)}]  {bk_short}  "
                f"{best_str}  ETA {eta_str}",
                flush=True
            )
            if is_best:
                best_r = max(r_list, key=lambda r: float(r["dbcv"])
                             if (r.get("dbcv") and not np.isnan(float(r["dbcv"]))) else -1e9)
                print(f"  {'':>4}  ↑ nuovo best! mcs={int(best_r['min_cluster_size'])}  "
                      f"k={int(best_r['n_clusters'])}  noise={float(best_r['noise_frac']):.1%}",
                      flush=True)

            hdb_done_g += 1
            if submitted_h < len(hdb_todo):
                try:
                    ref, combo = _dispatch_hdb()
                    hdb_flight[ref] = combo
                    submitted_h += 1
                except StopIteration: pass

    if not results:
        raise RuntimeError("Zero risultati — tutti i Ray task sono falliti.")

    return _to_df(results)


def _to_df(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    return df.sort_values("dbcv", ascending=False, na_position="last"
                          ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Final run — module-level worker functions
#
#  IMPORTANT: functions decorated with @ray.remote (or wrapped via ray.remote())
#  that are defined *inside* another function cannot be pickled by Ray Client
#  (ray://).  They must live at module level.
#
#  SERIALIZATION: same bytes-tuple pattern as run_one / ChronosActor — numpy
#  arrays are never passed directly across the Ray Client boundary.
#  Input  E_payload : (raw_bytes, shape, dtype_str)
#  Output            : (Z2_raw, Z2_shape, Z2_dtype, labels_raw, labels_shape, labels_dtype)
# ═══════════════════════════════════════════════════════════════════════════

def _final_run_gpu(E_payload: tuple, nn: int, md: float, mcs: int) -> tuple:
    """GPU worker (cuML) for final UMAP + HDBSCAN.  Module-level for Ray Client."""
    import numpy as np
    import os, glob, ctypes as _ctypes

    # Reconstruct numpy array from bytes tuple
    raw_bytes, shape, dtype_str = E_payload
    E = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()

    _B = "/usr/local/lib/python3.12/dist-packages"
    _dirs = (glob.glob(f"{_B}/nvidia/*/lib")   + glob.glob(f"{_B}/nvidia/*/lib64") +
             glob.glob(f"{_B}/*/lib64")         + glob.glob(f"{_B}/*.libs"))
    if _dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_dirs)
    for _lib in [f"{_B}/rapids_logger/lib64/librapids_logger.so",
                 f"{_B}/librmm/lib64/librmm.so",
                 f"{_B}/libraft/lib64/libraft.so",
                 f"{_B}/libcuml/lib64/libcuml++.so"]:
        if os.path.exists(_lib):
            try: _ctypes.CDLL(_lib, mode=_ctypes.RTLD_GLOBAL)
            except OSError: pass

    from cuml.manifold import UMAP   as cuUMAP
    from cuml.cluster  import HDBSCAN as cuHDBSCAN
    import cupy as cp

    E_gpu = cp.asarray(E.astype(np.float32))

    # Step 1: clustering (20D) — cosine metric per transformer embeddings
    Z20_gpu = cuUMAP(
        n_components = UMAP_N_COMPONENTS,
        n_neighbors  = int(nn),
        min_dist     = float(md),
        metric       = "cosine",
        random_state = RANDOM_STATE,
        n_epochs     = 300,
    ).fit_transform(E_gpu)
    hdb = cuHDBSCAN(
        min_cluster_size         = int(mcs),
        metric                   = "euclidean",
        cluster_selection_method = "eom",
        gen_min_span_tree        = True,
    )
    hdb.fit(Z20_gpu)
    labels = cp.asnumpy(hdb.labels_).astype(np.int16)

    # Step 2: visualizzazione (2D)
    Z2_gpu = cuUMAP(
        n_components = UMAP_VIZ_COMPONENTS,
        n_neighbors  = 15,
        min_dist     = 0.1,
        metric       = "euclidean",
        random_state = RANDOM_STATE,
        n_epochs     = 300,
    ).fit_transform(Z20_gpu)
    Z2 = cp.asnumpy(Z2_gpu).astype(np.float32)

    # Return as bytes tuples — avoid pickle5 numpy ABI mismatch on the way back
    return (Z2.tobytes(),     Z2.shape,     Z2.dtype.str,
            labels.tobytes(), labels.shape, labels.dtype.str)


def _final_run_cpu(E_payload: tuple, nn: int, md: float, mcs: int) -> tuple:
    """CPU worker (umap-learn + sklearn) for final UMAP + HDBSCAN.  Module-level for Ray Client."""
    import numpy as np, umap
    from sklearn.cluster import HDBSCAN

    # Reconstruct numpy array from bytes tuple
    raw_bytes, shape, dtype_str = E_payload
    E = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()

    # Step 1: clustering (20D) — cosine metric per transformer embeddings
    Z20 = umap.UMAP(
        n_components = UMAP_N_COMPONENTS,
        n_neighbors  = int(nn),
        min_dist     = float(md),
        metric       = "cosine",
        random_state = RANDOM_STATE,
        low_memory   = True,
        n_epochs     = 200,
        n_jobs       = -1,
    ).fit_transform(E)
    labels = HDBSCAN(
        min_cluster_size         = int(mcs),
        metric                   = "euclidean",
        cluster_selection_method = "eom",
    ).fit_predict(Z20).astype(np.int16)

    # Step 2: visualizzazione (2D)
    Z2 = umap.UMAP(
        n_components = UMAP_VIZ_COMPONENTS,
        n_neighbors  = 15,
        min_dist     = 0.1,
        metric       = "euclidean",
        random_state = RANDOM_STATE,
        low_memory   = True,
        n_jobs       = -1,
    ).fit_transform(Z20).astype(np.float32)

    # Return as bytes tuples — avoid pickle5 numpy ABI mismatch on the way back
    return (Z2.tobytes(),     Z2.shape,     Z2.dtype.str,
            labels.tobytes(), labels.shape, labels.dtype.str)


# ═══════════════════════════════════════════════════════════════════════════
#  Final run con best params  (GPU se disponibile)
# ═══════════════════════════════════════════════════════════════════════════

def final_run(E: np.ndarray, best: dict, use_gpu: bool):
    """
    Due passi:
      1. UMAP ?D → 20D  (best params)  → HDBSCAN → labels
      2. UMAP 20D →  2D (fisso)        → coordinate per visualizzazione

    Worker functions (_final_run_gpu / _final_run_cpu) are defined at module
    level — required for Ray Client serialization.  numpy arrays cross the
    client↔cluster boundary as bytes tuples (same pattern as run_one).
    """
    import ray
    print(f"  Connessione a Ray: {RAY_ADDRESS} ...", flush=True)
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True, log_to_driver=False)
    print("  Ray connesso.", flush=True)

    print(f"  Params: nn={int(best['n_neighbors'])}  "
          f"md={float(best['min_dist']):.2f}  "
          f"mcs={int(best['min_cluster_size'])}  "
          f"score={float(best['dbcv']):.4f}")
    print(f"  Backend: {'GPU (cuML)' if use_gpu else 'CPU (umap-learn)'}")
    print(f"  Step 1: UMAP {E.shape[1]}D → {UMAP_N_COMPONENTS}D → HDBSCAN")
    print(f"  Step 2: UMAP {UMAP_N_COMPONENTS}D → {UMAP_VIZ_COMPONENTS}D (visualizzazione)")

    # Serialise input as bytes tuple — avoids pickle5 numpy ABI mismatch
    E_payload = (E.tobytes(), E.shape, E.dtype.str)
    E_ref     = ray.put(E_payload)

    worker_fn = _final_run_gpu if use_gpu else _final_run_cpu
    resources = {"num_gpus": 1} if use_gpu else {"num_cpus": CPUS_PER_TASK}
    remote_fn = ray.remote(**resources)(worker_fn)

    result = ray.get(remote_fn.remote(
        E_ref,
        best["n_neighbors"], best["min_dist"], best["min_cluster_size"]
    ))

    # Unpack bytes tuples returned by worker
    Z2_raw, Z2_shape, Z2_dtype, lb_raw, lb_shape, lb_dtype = result
    Z_2d   = np.frombuffer(Z2_raw, dtype=np.dtype(Z2_dtype)).reshape(Z2_shape).copy()
    labels = np.frombuffer(lb_raw, dtype=np.dtype(lb_dtype)).reshape(lb_shape).copy()
    return Z_2d, labels


# ═══════════════════════════════════════════════════════════════════════════
#  Plot
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(grid_df, Z, labels, timestamps):
    """
    Genera tutti i plot di diagnostica.

    01  Grid-search heatmap  (DBCV / silhouette score)
    02  UMAP 2D — regimi (scatter + centroidi + contorni KDE)
    03  UMAP 2D — 4 pannelli: regime / anno / stagione / ora del giorno
    04  UMAP 2D — densità hexbin per regime
    05  Timeline regime nel tempo
    06  Distribuzione dimensione regimi
    """
    import matplotlib.dates as mdates
    from matplotlib.lines  import Line2D
    from scipy.stats        import gaussian_kde

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", font_scale=0.95)
    plt.rcParams.update({
        "figure.dpi"       : 150,
        "savefig.bbox"     : "tight",
        "savefig.facecolor": "white",
        "axes.spines.top"  : False,
        "axes.spines.right": False,
    })

    ts          = pd.to_datetime(timestamps)
    n_reg       = len(set(labels)) - (1 if -1 in labels else 0)
    reg_ids     = sorted(set(labels) - {-1})
    palette     = sns.color_palette("tab10", n_colors=max(n_reg, 1))
    reg_color   = {r: palette[i % len(palette)] for i, r in enumerate(reg_ids)}
    noise_c     = (0.75, 0.75, 0.75)

    # Colore per punto (noise in grigio)
    point_colors = np.array([noise_c if lb == -1 else reg_color[lb] for lb in labels])

    # ── Utilità: KDE contorno per un regime ─────────────────────────────
    def _kde_contour(ax, pts, color, levels=3, alpha=0.25):
        if len(pts) < 10:
            return
        try:
            kde  = gaussian_kde(pts.T, bw_method="scott")
            xmn, ymn = pts[:, 0].min(), pts[:, 1].min()
            xmx, ymx = pts[:, 0].max(), pts[:, 1].max()
            xi = np.linspace(xmn, xmx, 80)
            yi = np.linspace(ymn, ymx, 80)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            ax.contourf(Xi, Yi, Zi, levels=levels, colors=[color],
                        alpha=alpha, zorder=0)
            ax.contour(Xi, Yi, Zi, levels=levels, colors=[color],
                       linewidths=0.6, alpha=0.6, zorder=1)
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════
    # 01  Grid-search heatmap
    # ════════════════════════════════════════════════════════════════════
    grid_agg = (grid_df
                .groupby(["n_neighbors", "min_dist", "min_cluster_size"], as_index=False)
                ["dbcv"].max())
    mcs_vals = sorted(grid_agg["min_cluster_size"].dropna().unique())
    nn_vals  = sorted(grid_agg["n_neighbors"].dropna().unique())
    md_vals  = sorted(grid_agg["min_dist"].dropna().unique())
    ncols    = len(mcs_vals)
    fig, axes = plt.subplots(1, ncols, figsize=(max(12, 2.5 * ncols), 5), sharey=True)
    if ncols == 1:
        axes = [axes]
    vmin = grid_df["dbcv"].min()
    vmax = grid_df["dbcv"].max()
    for ax, mcs in zip(axes, mcs_vals):
        sub   = grid_agg[grid_agg["min_cluster_size"] == mcs]
        pivot = sub.pivot(index="n_neighbors", columns="min_dist",
                          values="dbcv").reindex(index=nn_vals, columns=md_vals)
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlGn",
                    vmin=vmin, vmax=vmax, linewidths=0.4,
                    cbar=(ax == axes[-1]), annot_kws={"size": 7})
        ax.set_title(f"mcs={int(mcs)}", fontsize=9, fontweight="bold")
        ax.set_xlabel("min_dist", fontsize=8)
        ax.set_ylabel("n_neighbors" if ax == axes[0] else "", fontsize=8)
    metric_note = ""
    if "metric" in grid_df.columns:
        metrics = grid_df["metric"].dropna().unique().tolist()
        metric_note = f"  [{', '.join(metrics)}]"
    fig.suptitle(f"Grid-search score{metric_note}", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_grid_heatmap.png"); plt.close(fig)
    print("  [plot] 01_grid_heatmap.png")

    # ════════════════════════════════════════════════════════════════════
    # 02  UMAP 2D — regimi con contorni KDE e centroidi
    # ════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(9, 7))

    # noise
    mask_n = labels == -1
    if mask_n.any():
        ax.scatter(Z[mask_n, 0], Z[mask_n, 1], c=[noise_c],
                   s=4, alpha=0.25, linewidths=0, zorder=2, label="noise")

    # regimi
    for r in reg_ids:
        m   = labels == r
        col = reg_color[r]
        _kde_contour(ax, Z[m], col, levels=4, alpha=0.18)
        ax.scatter(Z[m, 0], Z[m, 1], c=[col],
                   s=6, alpha=0.55, linewidths=0, zorder=3, label=f"R{r}")
        cx, cy = Z[m, 0].mean(), Z[m, 1].mean()
        ax.text(cx, cy, str(r), fontsize=8, fontweight="bold",
                ha="center", va="center", color="white",
                bbox=dict(boxstyle="circle,pad=0.15", fc=col, ec="none", alpha=0.85),
                zorder=5)

    ax.set_title(
        f"UMAP 2D  —  {n_reg} regimi  (noise = {mask_n.mean():.1%})",
        fontweight="bold", fontsize=11,
    )
    ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
    legend_els = (
        [Line2D([0], [0], marker="o", color="w", markerfacecolor=reg_color[r],
                markersize=7, label=f"R{r}  (n={int((labels==r).sum())})")
         for r in reg_ids] +
        ([Line2D([0], [0], marker="o", color="w", markerfacecolor=noise_c,
                 markersize=7, label=f"noise  (n={int(mask_n.sum())})")] if mask_n.any() else [])
    )
    ax.legend(handles=legend_els, fontsize=8,
              ncol=max(1, (n_reg + 1) // 6), loc="best",
              framealpha=0.7, edgecolor="0.8")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_umap_regimes.png"); plt.close(fig)
    print("  [plot] 02_umap_regimes.png")

    # ════════════════════════════════════════════════════════════════════
    # 03  UMAP 2D — 4 pannelli contestuali
    # ════════════════════════════════════════════════════════════════════
    fig, axes4 = plt.subplots(2, 2, figsize=(14, 11))

    # 03a  Per regime
    ax = axes4[0, 0]
    if mask_n.any():
        ax.scatter(Z[mask_n, 0], Z[mask_n, 1], c=[noise_c],
                   s=3, alpha=0.2, linewidths=0)
    for r in reg_ids:
        m = labels == r
        ax.scatter(Z[m, 0], Z[m, 1], c=[reg_color[r]],
                   s=5, alpha=0.5, linewidths=0)
        cx, cy = Z[m, 0].mean(), Z[m, 1].mean()
        ax.text(cx, cy, str(r), fontsize=7, fontweight="bold",
                ha="center", va="center", color="white",
                bbox=dict(boxstyle="circle,pad=0.12", fc=reg_color[r],
                          ec="none", alpha=0.85), zorder=4)
    ax.set_title("Regime", fontweight="bold"); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # 03b  Per anno
    ax   = axes4[0, 1]
    years = ts.dt.year.values
    sc   = ax.scatter(Z[:, 0], Z[:, 1], c=years, cmap="plasma",
                      s=4, alpha=0.5, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Year", shrink=0.85)
    ax.set_title("Colored by year", fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # 03c  Per stagione
    ax = axes4[1, 0]
    month = ts.dt.month.values
    season_label = np.where(np.isin(month, [3,  4,  5]),  0,   # Spring
                   np.where(np.isin(month, [6,  7,  8]),  1,   # Summer
                   np.where(np.isin(month, [9, 10, 11]),  2,   # Fall
                                                          3)))
    season_names  = ["Spring", "Summer", "Fall", "Winter"]
    season_colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]
    for s_id, s_name, s_col in zip(range(4), season_names, season_colors):
        m = season_label == s_id
        ax.scatter(Z[m, 0], Z[m, 1], c=[s_col],
                   s=4, alpha=0.45, linewidths=0, label=s_name)
    ax.legend(markerscale=3, fontsize=8, framealpha=0.7, edgecolor="0.8")
    ax.set_title("Colored by season", fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # 03d  Per ora del giorno
    ax  = axes4[1, 1]
    hour = ts.dt.hour.values
    sc   = ax.scatter(Z[:, 0], Z[:, 1], c=hour, cmap="twilight_shifted",
                      s=4, alpha=0.5, linewidths=0, vmin=0, vmax=23)
    cb   = plt.colorbar(sc, ax=ax, label="Hour of day", shrink=0.85)
    cb.set_ticks([0, 6, 12, 18, 23])
    ax.set_title("Colored by hour of day", fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    fig.suptitle("UMAP 2D — context panels", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_umap_context.png"); plt.close(fig)
    print("  [plot] 03_umap_context.png")

    # ════════════════════════════════════════════════════════════════════
    # 04  UMAP 2D — densità hexbin per regime (un pannello per regime)
    # ════════════════════════════════════════════════════════════════════
    ncols4 = min(4, n_reg)
    nrows4 = math.ceil(n_reg / ncols4) if n_reg > 0 else 1
    fig, axes_h = plt.subplots(nrows4, ncols4,
                                figsize=(4 * ncols4, 3.5 * nrows4),
                                sharex=True, sharey=True)
    axes_h = np.array(axes_h).ravel()
    xmn, xmx = Z[:, 0].min(), Z[:, 0].max()
    ymn, ymx = Z[:, 1].min(), Z[:, 1].max()
    for i, r in enumerate(reg_ids):
        ax  = axes_h[i]
        m   = labels == r
        col = reg_color[r]
        cmap_r = sns.light_palette(col, as_cmap=True)
        ax.hexbin(Z[m, 0], Z[m, 1], gridsize=30, cmap=cmap_r,
                  extent=[xmn, xmx, ymn, ymx], mincnt=1, linewidths=0)
        # background grigio per tutti i punti
        ax.scatter(Z[:, 0], Z[:, 1], c=[(0.9, 0.9, 0.9)],
                   s=1, alpha=0.15, linewidths=0, zorder=0)
        ax.set_title(f"R{r}  (n={int(m.sum())})", fontweight="bold",
                     fontsize=9, color=col)
        ax.set_xlabel("UMAP-1", fontsize=7); ax.set_ylabel("UMAP-2", fontsize=7)
    # Nascondi pannelli in eccesso
    for ax in axes_h[n_reg:]:
        ax.set_visible(False)
    fig.suptitle("UMAP 2D — regime density (hexbin)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_umap_hexbin.png"); plt.close(fig)
    print("  [plot] 04_umap_hexbin.png")

    # ════════════════════════════════════════════════════════════════════
    # 05  Timeline
    # ════════════════════════════════════════════════════════════════════
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 5), gridspec_kw={"height_ratios": [1, 3]}
    )
    # Top: stacked area per anno → fraction per regime
    years_u = sorted(ts.dt.year.unique())
    fracs   = {r: [] for r in reg_ids}
    for yr in years_u:
        m_yr = ts.dt.year == yr
        total = m_yr.sum()
        for r in reg_ids:
            fracs[r].append((m_yr & (labels == r)).sum() / total if total else 0)
    bottoms = np.zeros(len(years_u))
    for r in reg_ids:
        ax_top.bar(years_u, fracs[r], bottom=bottoms,
                   color=reg_color[r], width=0.8, label=f"R{r}")
        bottoms += np.array(fracs[r])
    ax_top.set_ylabel("Fraction", fontsize=8)
    ax_top.set_title("Regime fraction per year", fontsize=9, fontweight="bold")
    ax_top.legend(fontsize=7, ncol=n_reg, loc="upper left",
                  framealpha=0.6, edgecolor="0.8")
    ax_top.set_xlim(years_u[0] - 0.5, years_u[-1] + 0.5)

    # Bottom: scatter timeline
    for r in reg_ids:
        m = labels == r
        ax_bot.scatter(ts[m], np.full(m.sum(), r), c=[reg_color[r]],
                       s=2, alpha=0.5, linewidths=0)
    if mask_n.any():
        ax_bot.scatter(ts[mask_n], np.full(mask_n.sum(), -1), c=[noise_c],
                       s=1, alpha=0.3, linewidths=0)
    ax_bot.set_ylabel("Regime", fontsize=9)
    ax_bot.set_xlabel("Date", fontsize=9)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bot.xaxis.set_major_locator(mdates.YearLocator())
    ax_bot.set_yticks([-1] + reg_ids)
    ax_bot.set_yticklabels(["noise"] + [f"R{r}" for r in reg_ids], fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_timeline.png"); plt.close(fig)
    print("  [plot] 05_timeline.png")

    # ════════════════════════════════════════════════════════════════════
    # 06  Distribuzione dimensione regimi
    # ════════════════════════════════════════════════════════════════════
    u, cnt = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(max(6, len(u) * 0.9), 4))
    x_labels = ["noise" if r == -1 else f"R{r}" for r in u]
    bar_cols  = [noise_c if r == -1 else reg_color.get(r, (0.5, 0.5, 0.5)) for r in u]
    bars = ax.bar(x_labels, cnt, color=bar_cols, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)
    ax.set_title("Regime size distribution", fontweight="bold", fontsize=11)
    ax.set_xlabel("Regime"); ax.set_ylabel("N windows")
    ax.set_ylim(0, cnt.max() * 1.12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_distribution.png"); plt.close(fig)
    print("  [plot] 06_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    grid_path   = Path(C.RESULTS_DIR) / "grid_results.csv"
    ckpt_path   = Path(C.RESULTS_DIR) / "grid_checkpoint.csv"
    best_path   = Path(C.RESULTS_DIR) / "best_params.json"
    umap_path   = Path(C.RESULTS_DIR) / "umap.parquet"
    regime_path = Path(C.RESULTS_DIR) / "regimes.parquet"
    emb_raw_path = Path(C.RESULTS_DIR) / "embeddings.parquet"   # raw embeddings (no PCA)

    # ── Fresh start ──────────────────────────────────────────────────────
    if FRESH_START:
        # pca_path NON viene rimosso — è prodotto da step02, non da step03
        for p in [grid_path, ckpt_path, best_path, umap_path, regime_path]:
            if p.exists():
                p.unlink()
                print(f"  [fresh] rimosso {p.name}")

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 03: UMAP + Grid + HDBSCAN")
    print("═" * 65)
    print(f"  Ray     : {RAY_ADDRESS}")
    print(f"  Space   : {N_TRIALS} trials  full grid  (8×5×8)")
    print(f"  Pipeline: 7680D (raw embeddings) →[cuML UMAP]→ {UMAP_N_COMPONENTS}D →[cuML HDBSCAN]")
    print(f"  Viz     : {UMAP_N_COMPONENTS}D →[UMAP]→ {UMAP_VIZ_COMPONENTS}D")
    print(f"  Noise   : max {MAX_NOISE_FRAC:.0%}  |  Max regimi: {MAX_CLUSTERS}")
    print("─" * 65)

    # ── 1. Carica embeddings raw (prodotto da step02) ────────────────────
    # PCA rimossa: cuML UMAP gestisce 7680D direttamente su GPU.
    print("\n[1/5] Loading embeddings raw (da step02)...")
    if not emb_raw_path.exists():
        raise FileNotFoundError(
            f"File non trovato: {emb_raw_path}\n"
            f"Esegui prima step02_embeddings.py — genera embeddings.parquet."
        )
    emb_df     = pd.read_parquet(emb_raw_path)
    timestamps = pd.to_datetime(emb_df["datetime"])
    E          = emb_df[[c for c in emb_df.columns
                          if c.startswith("emb_")]].values.astype(np.float32)
    print(f"  {len(E):,} windows × {E.shape[1]} dims  ← {emb_raw_path.name}", flush=True)

    # ── 2. Grid search (con resume automatico via checkpoint CSV) ────────
    # Passo 1: se esiste un CSV verifica che la griglia salvata coincida con
    # quella corrente; se non coincide (es. mcs cambiati) lo cancella.
    if grid_path.exists():
        _tmp = pd.read_csv(grid_path)
        saved_mcs   = set(_tmp["min_cluster_size"].dropna().astype(int).unique())
        current_mcs = set(SEARCH_SPACE["min_cluster_size"])
        if not saved_mcs.issubset(current_mcs):
            print(f"\n[2/5] ⚠  CSV con mcs={sorted(saved_mcs)} — griglia cambiata "
                  f"({sorted(current_mcs)}).")
            print(f"  ↺  Cancello grid_results.csv e checkpoint — ripart da zero.")
            grid_path.unlink()
            if ckpt_path.exists():
                ckpt_path.unlink()

    # Passo 2: carica o esegui il grid search
    if grid_path.exists():
        print(f"\n[2/5] Grid results già presenti — carico da {grid_path}")
        grid_df = pd.read_csv(grid_path)
        # Applica i filtri correnti sui trial caricati (utile se i vincoli sono
        # cambiati rispetto alla run che ha prodotto il CSV).
        mask_bad = (
            (grid_df["n_clusters"] > MAX_CLUSTERS) |
            (grid_df["noise_frac"] > MAX_NOISE_FRAC)
        )
        if mask_bad.any():
            print(f"  ⚠  {mask_bad.sum()} trial violano i vincoli correnti "
                  f"(k>{MAX_CLUSTERS} o noise>{MAX_NOISE_FRAC:.0%}) "
                  f"→ dbcv azzerato.")
            grid_df.loc[mask_bad, "dbcv"] = float("nan")
        print(f"  {len(grid_df)} trial  |  "
              f"{grid_df['dbcv'].notna().sum()} validi "
              f"(k≤{MAX_CLUSTERS}, noise≤{MAX_NOISE_FRAC:.0%})")
    else:
        n_already = len(pd.read_csv(ckpt_path)) if ckpt_path.exists() else 0
        print(f"\n[2/5] Grid search  ({E.shape[1]}D → {UMAP_N_COMPONENTS}D → HDBSCAN)...")
        if n_already:
            print(f"  Resume: {n_already} trial già nel checkpoint — continuo da lì.")
        MAX_CONN_RETRIES = 10
        for _attempt in range(MAX_CONN_RETRIES):
            try:
                grid_df = grid_search(E, ckpt_path)
                break
            except ConnectionError as _e:
                if _attempt < MAX_CONN_RETRIES - 1:
                    _wait = 30 * (_attempt + 1)
                    print(f"\n  ⚠  Connessione Ray persa: {_e}")
                    print(f"  ↺  Trial salvati nel checkpoint CSV.")
                    print(f"  ↺  Riconnessione tra {_wait}s "
                          f"(tentativo {_attempt+2}/{MAX_CONN_RETRIES})...")
                    try:
                        import ray; ray.shutdown()
                    except Exception:
                        pass
                    time.sleep(_wait)
                else:
                    raise
        # Salva CSV finale ordinato per dbcv
        grid_df.to_csv(grid_path, index=False)
        print(f"\n  Salvato → {grid_path}")

    print(f"\n  Top 5:")
    print(grid_df.dropna(subset=["dbcv"]).head(5).to_string(index=False))

    # ── 3. Best params ───────────────────────────────────────────────────
    print(f"\n[3/5] Best params...")
    valid = grid_df.dropna(subset=["dbcv"])
    if valid.empty:
        raise RuntimeError("Nessun trial valido (tutti NaN).")
    best = valid.iloc[0].to_dict()
    with open(best_path, "w") as f:
        json.dump({k: (int(v)   if isinstance(v, (np.integer,  int))   else
                       float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in best.items()}, f, indent=2)
    print(f"  Salvato → {best_path}")

    # ── 4. Final run ─────────────────────────────────────────────────────
    import ray
    print("  Verifica risorse cluster...", flush=True)
    try:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True,
                 log_to_driver=False)
        use_gpu = int(ray.cluster_resources().get("GPU", 0)) > 0
        ray.shutdown()
    except ConnectionError:
        print("  ⚠  Ray non raggiungibile — uso CPU per final run.", flush=True)
        use_gpu = False

    if umap_path.exists() and regime_path.exists():
        print(f"\n[4/5] Output finali già presenti — carico da disco.")
        umap_df   = pd.read_parquet(umap_path)
        Z         = umap_df[["umap_1", "umap_2"]].values
        regime_df = pd.read_parquet(regime_path)
        labels    = regime_df["regime"].values
    else:
        print(f"\n[4/5] Final UMAP + HDBSCAN con best params...")
        Z, labels = final_run(E, best, use_gpu=use_gpu)
        pd.DataFrame({"datetime": timestamps,
                      "umap_1": Z[:, 0], "umap_2": Z[:, 1]}
                     ).to_parquet(umap_path, index=False)
        pd.DataFrame({"datetime": timestamps,
                      "regime": labels.astype(np.int16)}
                     ).to_parquet(regime_path, index=False)
        print(f"  Salvato → {umap_path}")
        print(f"  Salvato → {regime_path}")

    # ── 5. Plot ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Plot...")
    make_plots(grid_df, Z, labels, timestamps)

    # ── Report ────────────────────────────────────────────────────────────
    n_reg      = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    elapsed    = time.time() - t0

    # Mostra quanti trial hanno girato su GPU vs CPU
    if "backend" in grid_df.columns:
        bk_counts = grid_df["backend"].value_counts().to_dict()
        bk_str = "  ".join(f"{k}:{v}" for k, v in bk_counts.items())
    else:
        bk_str = "n/a"

    print("\n" + "─" * 65)
    print("  RISULTATI")
    print("─" * 65)
    print(f"  Regimi trovati    : {n_reg}")
    print(f"  Noise fraction    : {noise_frac:.1%}")
    print(f"  Best score        : {float(best['dbcv']):.4f}")
    print(f"  Best n_neighbors  : {int(best['n_neighbors'])}")
    print(f"  Best min_dist     : {float(best['min_dist']):.2f}")
    print(f"  Best mcs          : {int(best['min_cluster_size'])}")
    print(f"  Backend trials    : {bk_str}")
    print(f"  Tempo totale      : {elapsed:.1f}s")
    if n_reg >= 2:
        sizes = pd.Series(labels[labels >= 0]).value_counts().sort_index()
        print(f"\n  Regime sizes:")
        for r, sz in sizes.items():
            print(f"    regime {r:2d} : {sz:5,} windows ({sz/len(labels):.1%})")
    print("\n  → Step 04: OU parameter estimation per regime")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
