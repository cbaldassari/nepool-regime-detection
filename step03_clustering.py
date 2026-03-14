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
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(line_buffering=True)   # flush immediato ad ogni print
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  Configurazione
# ═══════════════════════════════════════════════════════════════════════════

PCA_VARIANCE          = 0.90 # pre-riduzione CPU: componenti che spiegano 90% varianza
UMAP_N_COMPONENTS     = 20   # dimensioni per clustering HDBSCAN
UMAP_VIZ_COMPONENTS   = 2    # dimensioni per visualizzazione 2D
RANDOM_STATE          = 42
RAY_ADDRESS       = "ray://datalab-rayclnt.unitus.it:10001"
CPUS_PER_TASK     = 2          # usato solo in modalità CPU
PLOT_DIR          = Path(C.RESULTS_DIR) / "step03"

# ── Modalità sviluppo ────────────────────────────────────────────────────
# True  → cancella tutti i checkpoint ad ogni run (sempre da zero)
# False → riprende da dove era rimasto (comportamento produzione)
FRESH_START = False

# Path dove pip installa i pacchetti di sistema sui nodi
CUML_BASE = "/usr/local/lib/python3.12/dist-packages"

SEARCH_SPACE = {
    "n_neighbors"      : [5, 10, 15, 20, 30, 50, 75, 100],
    "min_dist"         : [0.0, 0.05, 0.1, 0.2, 0.3],
    "min_cluster_size" : [30, 50, 75, 100, 150, 200, 300, 500],
}
N_TRIALS      = math.prod(len(v) for v in SEARCH_SPACE.values())   # 320 full grid
MAX_NOISE_FRAC = 0.05   # trial con noise > 5% → score=nan, esclusi dal ranking



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
#  Singolo trial — GPU prima, CPU come fallback
# ═══════════════════════════════════════════════════════════════════════════

def run_one(E_pca: np.ndarray, n_neighbors: int, min_dist: float,
            min_cluster_size: int) -> dict:
    import numpy as np

    gpu_err_msg = None

    # ── Tentativo GPU (cuML) ─────────────────────────────────────────────
    try:
        # Preload inline (evita problemi di serializzazione closure in Ray Client)
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

        from cuml.manifold import UMAP   as cuUMAP
        from cuml.cluster  import HDBSCAN as cuHDBSCAN
        import cupy as cp

        E_gpu = cp.asarray(E_pca.astype(np.float32))

        # UMAP 8448D → 20D per clustering
        Z_gpu = cuUMAP(
            n_components = UMAP_N_COMPONENTS,     # 20D
            n_neighbors  = n_neighbors,
            min_dist     = min_dist,
            metric       = "euclidean",
            random_state = RANDOM_STATE,
            n_epochs     = 100,
        ).fit_transform(E_gpu)

        hdb = cuHDBSCAN(
            min_cluster_size         = min_cluster_size,
            metric                   = "euclidean",
            cluster_selection_method = "eom",
            gen_min_span_tree        = True,   # abilita relative_validity_
        )
        hdb.fit(Z_gpu)

        labels = cp.asnumpy(hdb.labels_)
        Z      = cp.asnumpy(Z_gpu)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = float((labels == -1).mean())

        # Vincolo noise: se troppo rumore il trial è invalido
        if noise_frac > MAX_NOISE_FRAC:
            score       = float("nan")
            metric_used = "noise_exceed"
        elif n_clusters >= 2 and hasattr(hdb, "relative_validity_"):
            score       = float(hdb.relative_validity_)
            metric_used = "dbcv"
        elif n_clusters >= 2:
            from sklearn.metrics import silhouette_score
            mask = labels != -1
            score = (float(silhouette_score(Z[mask], labels[mask]))
                     if mask.sum() > n_clusters else float("nan"))
            metric_used = "silhouette"
        else:
            score       = float("nan")
            metric_used = "nan"

        return {
            "n_neighbors"      : n_neighbors,
            "min_dist"         : min_dist,
            "min_cluster_size" : min_cluster_size,
            "n_clusters"       : n_clusters,
            "noise_frac"       : noise_frac,
            "dbcv"             : score,
            "backend"          : "GPU",
            "metric"           : metric_used,
        }

    except Exception as e:
        gpu_err_msg = str(e)[:80]

    # ── Fallback CPU (umap-learn + sklearn) ──────────────────────────────
    import umap
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import silhouette_score

    # UMAP 8448D → 20D per clustering
    Z = umap.UMAP(
        n_components = UMAP_N_COMPONENTS,     # 20D
        n_neighbors  = n_neighbors,
        min_dist     = min_dist,
        metric       = "euclidean",
        random_state = RANDOM_STATE,
        low_memory   = True,
        n_epochs     = 100,
        n_jobs       = 1,
    ).fit_transform(E_pca)

    labels = HDBSCAN(
        min_cluster_size         = min_cluster_size,
        metric                   = "euclidean",
        cluster_selection_method = "eom",
    ).fit_predict(Z)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = float((labels == -1).mean())
    mask       = labels != -1

    if noise_frac > MAX_NOISE_FRAC:
        score = float("nan")
    elif n_clusters >= 2 and mask.sum() > n_clusters:
        score = float(silhouette_score(Z[mask], labels[mask], metric="euclidean"))
    else:
        score = float("nan")

    return {
        "n_neighbors"      : n_neighbors,
        "min_dist"         : min_dist,
        "min_cluster_size" : min_cluster_size,
        "n_clusters"       : n_clusters,
        "noise_frac"       : noise_frac,
        "dbcv"             : score,
        "backend"          : f"CPU",
        "metric"           : "silhouette",
        "gpu_err"          : gpu_err_msg,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Grid search su Ray  (auto GPU / CPU)
# ═══════════════════════════════════════════════════════════════════════════

def grid_search(E_pca: np.ndarray) -> pd.DataFrame:
    import ray, optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Ray ─────────────────────────────────────────────────────────────
    print(f"  Connessione a Ray: {RAY_ADDRESS} ...", flush=True)
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True, log_to_driver=False)
    print("  Ray connesso.", flush=True)

    MAX_TASK_SECS = 480   # 8 min — task oltre questo limite vengono cancellati

    n_cpu = int(ray.cluster_resources().get("CPU", 1))
    n_gpu = int(ray.cluster_resources().get("GPU", 0))

    if n_gpu > 0:
        N_CONCURRENT = n_gpu
        ray_run_one  = ray.remote(num_gpus=1)(run_one)
        print(f"  Modalità GPU  — {n_gpu} GPU  →  {N_CONCURRENT} concurrent tasks")
        print(f"  Backend       : cuML UMAP + cuML HDBSCAN (DBCV reale)")
    else:
        N_CONCURRENT = n_cpu // CPUS_PER_TASK
        ray_run_one  = ray.remote(num_cpus=CPUS_PER_TASK)(run_one)
        print(f"  Modalità CPU  — {n_cpu} CPU  →  {N_CONCURRENT} concurrent tasks "
              f"({CPUS_PER_TASK} CPU/task)")
        print(f"  Backend       : umap-learn + sklearn HDBSCAN (silhouette)")

    # ── Optuna study (checkpoint SQLite) ────────────────────────────────
    db_path    = Path(C.RESULTS_DIR) / "optuna_study.db"
    study_name = "step03_umap_hdbscan"
    study = optuna.create_study(
        study_name     = study_name,
        storage        = f"sqlite:///{db_path}",
        direction      = "maximize",
        sampler        = optuna.samplers.GridSampler(SEARCH_SPACE),
        load_if_exists = True,
    )

    # Recupera trial già completati (resume)
    results = [
        {
            "n_neighbors"      : t.params["n_neighbors"],
            "min_dist"         : t.params["min_dist"],
            "min_cluster_size" : t.params["min_cluster_size"],
            "n_clusters"       : t.user_attrs.get("n_clusters", 0),
            "noise_frac"       : t.user_attrs.get("noise_frac",  1.0),
            "dbcv"             : t.value,
            "backend"          : t.user_attrs.get("backend", "?"),
            "metric"           : t.user_attrs.get("metric",  "?"),
        }
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    n_done = len(results)
    n_left = N_TRIALS - n_done

    print(f"  Full grid: {N_TRIALS} trial  "
          f"({'da zero' if n_done == 0 else f'{n_done} già fatti → {n_left} rimasti'})")

    if n_left <= 0:
        print("  ✓ Tutti i trial già completati — carico dal DB.")
        return _to_df(results)

    # ── Loop ask-and-tell parallelo ──────────────────────────────────────
    E_ref     = ray.put(E_pca)
    in_flight = {}
    submitted = 0
    best_dbcv = max(
        (r["dbcv"] for r in results if r["dbcv"] and not np.isnan(r["dbcv"])),
        default=float("-inf")
    )
    first_backend = None   # per rilevare se il primo trial è GPU o CPU

    def _dispatch():
        trial = study.ask()
        # suggest_categorical: TPE rispetta i valori discreti del search space
        nn  = trial.suggest_categorical("n_neighbors",      SEARCH_SPACE["n_neighbors"])
        md  = trial.suggest_categorical("min_dist",         SEARCH_SPACE["min_dist"])
        mcs = trial.suggest_categorical("min_cluster_size", SEARCH_SPACE["min_cluster_size"])
        return ray_run_one.remote(E_ref, nn, md, mcs), trial

    for _ in range(min(N_CONCURRENT, n_left)):
        ref, trial = _dispatch()
        in_flight[ref] = trial
        submitted += 1

    print(f"\n  {'#':>4}  {'nn':>4} {'md':>5} {'mcs':>5}  "
          f"{'score':>7}  {'k':>3}  {'noise':>6}  {'bk':>3}  note")
    print(f"  {'─'*4}  {'─'*4} {'─'*5} {'─'*5}  "
          f"{'─'*7}  {'─'*3}  {'─'*6}  {'─'*3}  {'─'*6}")

    bar_fmt = "  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"

    with tqdm(total=N_TRIALS, initial=n_done, desc="  trials",
              ncols=88, bar_format=bar_fmt) as pbar:
        pbar.set_postfix(best=f"{best_dbcv:.4f}" if np.isfinite(best_dbcv) else "n/a",
                         refresh=False)

        while in_flight:
            done_refs, _ = ray.wait(list(in_flight.keys()), num_returns=1,
                                    timeout=MAX_TASK_SECS)

            # ── Timeout: nessun task completato entro MAX_TASK_SECS ──────
            if not done_refs:
                hung_ref   = next(iter(in_flight))
                hung_trial = in_flight.pop(hung_ref)
                tqdm.write(
                    f"  ⏱  timeout ({MAX_TASK_SECS}s) — "
                    f"trial {n_done+1} cancellato  "
                    f"(nn={hung_trial.params.get('n_neighbors','?')} "
                    f"md={hung_trial.params.get('min_dist','?')} "
                    f"mcs={hung_trial.params.get('min_cluster_size','?')})"
                )
                try:
                    ray.cancel(hung_ref, force=True)
                except Exception:
                    pass
                try:
                    study.tell(hung_trial, values=None,
                               state=optuna.trial.TrialState.FAIL)
                except (RuntimeError, ValueError):
                    pass
                n_done += 1
                pbar.update(1)
                if submitted < n_left:
                    try:
                        ref, trial = _dispatch()
                        in_flight[ref] = trial
                        submitted += 1
                    except Exception:
                        pass
                continue

            ref   = done_refs[0]
            trial = in_flight.pop(ref)

            try:
                r   = ray.get(ref)
                val = r["dbcv"] if (r["dbcv"] and not np.isnan(r["dbcv"])) \
                                else float("-inf")

                # Salva attributi nel DB Optuna
                trial.set_user_attr("n_clusters", int(r["n_clusters"]))
                trial.set_user_attr("noise_frac", float(r["noise_frac"]))
                trial.set_user_attr("backend",    r.get("backend", "?"))
                trial.set_user_attr("metric",     r.get("metric",  "?"))
                if r.get("gpu_err"):
                    trial.set_user_attr("gpu_err", r["gpu_err"])

                try:
                    study.tell(trial, val)
                except (RuntimeError, ValueError):
                    pass   # GridSampler chiama study.stop() all'ultimo trial

                results.append(r)

                is_best = np.isfinite(val) and val > best_dbcv
                if is_best:
                    best_dbcv = val

                # Backend tag corto
                bk = r.get("backend", "?")
                bk_short = "GPU" if bk == "GPU" else "CPU"

                # Avviso se primo trial GPU fallisce
                if first_backend is None:
                    first_backend = bk_short
                    if bk_short == "CPU" and n_gpu > 0:
                        gpu_err = r.get("gpu_err", "unknown")
                        tqdm.write(f"\n  ⚠  GPU non disponibile su questo nodo: {gpu_err}")
                        tqdm.write(f"  ⚠  Esecuzione in modalità CPU fallback.\n")

                sc_str = (f"{r['dbcv']:.4f}"
                          if (r["dbcv"] and not np.isnan(r["dbcv"])) else "   nan")
                tqdm.write(
                    f"  {n_done+1:>4}  "
                    f"{int(r['n_neighbors']):>4} "
                    f"{float(r['min_dist']):>5.2f} "
                    f"{int(r['min_cluster_size']):>5}  "
                    f"{sc_str:>7}  "
                    f"{int(r['n_clusters']):>3}  "
                    f"{r['noise_frac']:>6.1%}  "
                    f"{bk_short:>3}"
                    f"{'  ← best!' if is_best else ''}"
                )
                pbar.set_postfix(
                    best=f"{best_dbcv:.4f}" if np.isfinite(best_dbcv) else "n/a",
                    refresh=False,
                )

            except Exception as e:
                try:
                    study.tell(trial, values=None,
                               state=optuna.trial.TrialState.FAIL)
                except (RuntimeError, ValueError):
                    pass
                tqdm.write(f"  ⚠  trial {n_done+1} fallito: {e}")

            n_done += 1
            pbar.update(1)

            if submitted < n_left:
                try:
                    ref, trial = _dispatch()
                    in_flight[ref] = trial
                    submitted += 1
                except Exception:
                    pass   # grid esaurita

    if not results:
        raise RuntimeError("Zero risultati — tutti i Ray task sono falliti.")

    return _to_df(results)


def _to_df(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    return df.sort_values("dbcv", ascending=False, na_position="last"
                          ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Final run con best params  (GPU se disponibile)
# ═══════════════════════════════════════════════════════════════════════════

def final_run(E: np.ndarray, best: dict, use_gpu: bool):
    """
    Due passi:
      1. UMAP 8448D → 20D  (best params)  → HDBSCAN → labels
      2. UMAP 20D  →  2D   (fisso)        → coordinate per visualizzazione
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

    if use_gpu:
        @ray.remote(num_gpus=1)
        def _run(E, nn, md, mcs):
            import numpy as np
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

            from cuml.manifold import UMAP   as cuUMAP
            from cuml.cluster  import HDBSCAN as cuHDBSCAN
            import cupy as cp

            E_gpu = cp.asarray(E.astype(np.float32))

            # Step 1: clustering (20D)
            Z20_gpu = cuUMAP(
                n_components = UMAP_N_COMPONENTS,
                n_neighbors  = int(nn),
                min_dist     = float(md),
                metric       = "euclidean",
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

            # Step 2: visualizzazione (2D) partendo dai 20D
            Z2_gpu = cuUMAP(
                n_components = UMAP_VIZ_COMPONENTS,
                n_neighbors  = 15,
                min_dist     = 0.1,
                metric       = "euclidean",
                random_state = RANDOM_STATE,
                n_epochs     = 300,
            ).fit_transform(Z20_gpu)

            return cp.asnumpy(Z2_gpu).astype(np.float32), labels
    else:
        @ray.remote(num_cpus=CPUS_PER_TASK)
        def _run(E, nn, md, mcs):
            import numpy as np, umap
            from sklearn.cluster import HDBSCAN

            # Step 1: clustering (20D)
            Z20 = umap.UMAP(
                n_components = UMAP_N_COMPONENTS,
                n_neighbors  = int(nn),
                min_dist     = float(md),
                metric       = "euclidean",
                random_state = RANDOM_STATE,
                low_memory   = True,
                n_epochs     = 200,
                n_jobs       = -1,
            ).fit_transform(E)
            labels = HDBSCAN(
                min_cluster_size         = int(mcs),
                metric                   = "euclidean",
                cluster_selection_method = "eom",
            ).fit_predict(Z20)

            # Step 2: visualizzazione (2D)
            Z2 = umap.UMAP(
                n_components = UMAP_VIZ_COMPONENTS,
                n_neighbors  = 15,
                min_dist     = 0.1,
                metric       = "euclidean",
                random_state = RANDOM_STATE,
                low_memory   = True,
                n_jobs       = -1,
            ).fit_transform(Z20)

            return Z2.astype(np.float32), labels.astype(np.int16)

    Z_2d, labels = ray.get(_run.remote(
        ray.put(E),
        best["n_neighbors"], best["min_dist"], best["min_cluster_size"]
    ))
    return Z_2d, labels


# ═══════════════════════════════════════════════════════════════════════════
#  Plot
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(grid_df, Z, labels, timestamps):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    n_reg   = len(set(labels)) - (1 if -1 in labels else 0)
    palette = sns.color_palette("tab10", n_colors=max(n_reg, 1))
    noise_c = (0.7, 0.7, 0.7)

    # 01 heatmap
    # Aggrega duplicati (stessa combo eseguita più volte) → tieni il massimo
    grid_agg = (grid_df
                .groupby(["n_neighbors", "min_dist", "min_cluster_size"], as_index=False)
                ["dbcv"].max())

    mcs_vals = sorted(grid_agg["min_cluster_size"].dropna().unique())
    nn_vals  = sorted(grid_agg["n_neighbors"].dropna().unique())
    md_vals  = sorted(grid_agg["min_dist"].dropna().unique())
    fig, axes = plt.subplots(1, len(mcs_vals), figsize=(16, 4), sharey=True)
    for ax, mcs in zip(axes, mcs_vals):
        sub   = grid_agg[grid_agg["min_cluster_size"] == mcs]
        pivot = sub.pivot(index="n_neighbors", columns="min_dist",
                          values="dbcv").reindex(index=nn_vals, columns=md_vals)
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlGn",
                    vmin=grid_df["dbcv"].min(), vmax=grid_df["dbcv"].max(),
                    linewidths=0.3, cbar=(ax == axes[-1]))
        ax.set_title(f"mcs={mcs}", fontsize=9)
        ax.set_xlabel("min_dist")
        ax.set_ylabel("n_neighbors" if ax == axes[0] else "")

    # Nota metrica
    metric_note = ""
    if "metric" in grid_df.columns:
        metrics = grid_df["metric"].dropna().unique()
        metric_note = f" — metrica: {', '.join(metrics)}"
    fig.suptitle(f"Grid search score{metric_note}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_grid_heatmap.png"); plt.close(fig)
    print("  [plot] 01_grid_heatmap.png")

    # 02 UMAP regimes
    fig, ax = plt.subplots(figsize=(10, 8))
    mask_n = labels == -1
    if mask_n.any():
        ax.scatter(Z[mask_n, 0], Z[mask_n, 1], c=[noise_c], s=2, alpha=0.3, label="noise")
    for r in sorted(set(labels) - {-1}):
        m = labels == r
        ax.scatter(Z[m, 0], Z[m, 1], c=[palette[r % len(palette)]],
                   s=3, alpha=0.6, label=f"regime {r}")
    ax.set_title(f"UMAP 2D — {n_reg} regimi (noise={mask_n.mean():.1%})",
                 fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=4, fontsize=8, ncol=max(1, n_reg // 8))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_umap_regimes.png"); plt.close(fig)
    print("  [plot] 02_umap_regimes.png")

    # 03 UMAP by year
    years = pd.to_datetime(timestamps).dt.year.values
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=years, cmap="plasma", s=2, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Year")
    ax.set_title("UMAP 2D — coloured by year", fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_umap_time.png"); plt.close(fig)
    print("  [plot] 03_umap_time.png")

    # 04 timeline
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.scatter(pd.to_datetime(timestamps), labels, c=labels,
               cmap="tab10", s=1, alpha=0.6)
    ax.set_title("Regime label over time", fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Regime")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_timeline.png"); plt.close(fig)
    print("  [plot] 04_timeline.png")

    # 05 distribution
    u, c = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(max(6, len(u)), 4))
    bars = ax.bar(
        [str(r) if r != -1 else "noise" for r in u], c,
        color=[noise_c if r == -1 else palette[r % len(palette)] for r in u]
    )
    ax.bar_label(bars, fmt="%d", fontsize=8)
    ax.set_title("Regime size distribution", fontweight="bold")
    ax.set_xlabel("Regime"); ax.set_ylabel("N windows")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_distribution.png"); plt.close(fig)
    print("  [plot] 05_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    grid_path   = Path(C.RESULTS_DIR) / "grid_results.csv"
    best_path   = Path(C.RESULTS_DIR) / "best_params.json"
    umap_path   = Path(C.RESULTS_DIR) / "umap.parquet"
    regime_path = Path(C.RESULTS_DIR) / "regimes.parquet"

    # ── Fresh start ──────────────────────────────────────────────────────
    if FRESH_START:
        for p in [grid_path, best_path, umap_path, regime_path,
                  Path(C.RESULTS_DIR) / "optuna_study.db"]:
            if p.exists():
                p.unlink()
                print(f"  [fresh] rimosso {p.name}")

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 03: UMAP + Grid + HDBSCAN")
    print("═" * 65)
    print(f"  Ray     : {RAY_ADDRESS}")
    print(f"  Space   : {N_TRIALS} trials  full grid  (8×5×8)")
    print(f"  Pipeline: 8448D →[PCA {PCA_VARIANCE:.0%}var]→ ?D →[UMAP]→ {UMAP_N_COMPONENTS}D →[HDBSCAN]")
    print(f"  Viz     : {UMAP_N_COMPONENTS}D →[UMAP]→ {UMAP_VIZ_COMPONENTS}D")
    print(f"  Noise   : max {MAX_NOISE_FRAC:.0%}")
    print("─" * 65)

    # ── 1. Embeddings + PCA pre-riduzione ───────────────────────────────
    print("\n[1/5] Loading embeddings + PCA pre-riduzione...")
    emb_df     = pd.read_parquet(Path(C.RESULTS_DIR) / "embeddings.parquet")
    timestamps = pd.to_datetime(emb_df["datetime"])
    E_raw      = emb_df[[c for c in emb_df.columns
                          if c.startswith("emb_")]].values.astype(np.float32)
    print(f"  {len(E_raw):,} windows × {E_raw.shape[1]:,} dims")

    # PCA CPU: riduce 8448D → n componenti che spiegano PCA_VARIANCE della varianza
    print(f"  PCA {E_raw.shape[1]}D → {PCA_VARIANCE:.0%} varianza  (CPU, sklearn)...", flush=True)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    E   = pca.fit_transform(E_raw).astype(np.float32)
    print(f"  → {E.shape[1]}D  ({pca.explained_variance_ratio_.sum():.1%} varianza spiegata)")

    # ── 2. Grid search (con resume automatico) ───────────────────────────
    if grid_path.exists():
        print(f"\n[2/5] Grid results già presenti — carico da {grid_path}")
        grid_df = pd.read_csv(grid_path)
        print(f"  {len(grid_df)} trial  |  "
              f"{grid_df['dbcv'].notna().sum()} validi")
    else:
        print(f"\n[2/5] Grid search  ({E.shape[1]}D → {UMAP_N_COMPONENTS}D → HDBSCAN)...")
        MAX_CONN_RETRIES = 10
        for _attempt in range(MAX_CONN_RETRIES):
            try:
                grid_df = grid_search(E)
                break
            except ConnectionError as _e:
                if _attempt < MAX_CONN_RETRIES - 1:
                    _wait = 30 * (_attempt + 1)
                    print(f"\n  ⚠  Connessione Ray persa: {_e}")
                    print(f"  ↺  I trial completati sono nel DB.")
                    print(f"  ↺  Riconnessione tra {_wait}s "
                          f"(tentativo {_attempt+2}/{MAX_CONN_RETRIES})...")
                    try:
                        import ray; ray.shutdown()
                    except Exception:
                        pass
                    time.sleep(_wait)
                else:
                    raise
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
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    use_gpu = int(ray.cluster_resources().get("GPU", 0)) > 0
    ray.shutdown()

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
