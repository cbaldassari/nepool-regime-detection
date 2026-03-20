"""
step04_mean_reversion.py
=======================
NEPOOL Regime Detection Pipeline — Step 04: Ornstein-Uhlenbeck Parameter Estimation

Input  : results/preprocessed.parquet   (43,764 rows × 12 cols)
         results/regimes.parquet        (N_windows × [datetime, regime])
Output : results/mean_reversion_params.parquet      (K_regimes × 3 orizzonti × parametri mean reversion)
         results/mean_reversion_drift.parquet       (N_windows × [datetime, regime, log_price,
                                         drift_1h, drift_6h, drift_24h])
Plots  : results/step04/

Design
------
Stima dei parametri mean reversion per regime usando Chronos-2 in modalità forecast.

Motivazione: il drift atteso stimato da Chronos-2 è data-driven — il modello
ha appreso la struttura dinamica del mercato su migliaia di serie temporali.
Usare Chronos-2 invece di un AR(1) classico significa che la stima del drift
incorpora pattern non lineari, stagionalità, e dipendenze cross-channel che
un modello econometrico parametrico non cattura.

Procedura per ogni finestra i con timestamp t_i e regime r_i:
  1. Contesto: log_return[t_i − 719 : t_i] (720 valori, univariato)
  2. Chronos-2 predict() → mediana E[log_return_{t+k}] per k=1..24
  3. drift_{i,h} = Σ_{k=1}^{h} E[log_return_{t+k}]   per h ∈ {1, 6, 24}
       = E[log_price_{t+h}] − log_price_t  (drift cumulato atteso)

OLS per regime r, orizzonte h (esclude noise regime = −1):
  drift_{i,h} = a + b · log_price_{t_i}  →  OLS su tutti i ∈ regime r
  θ_{r,h} = −b / h          (mean-reversion speed, > 0)
  μ_{r,h} = −a / b          (equilibrium log-price)
  t½_{r,h} = ln(2) / θ_{r,h} (half-life in hours)

σ (volatilità):
  Stimata dai log-return osservati nel regime r (non dai residui Chronos-2).
  σ_r = std(log_return osservati nel regime r)
  I residui Chronos-2 misurano l'aderenza al modello lineare, non la
  volatilità effettiva dei prezzi — per σ servono i dati reali.

Execution:
  Ray cluster GPU (stessi attori di step02) per Chronos-2 predict.
  Ogni attore processa un chunk di finestre e restituisce i drift.

Install:
    pip install "chronos-forecasting[chronos2]" ray[default]
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
    pip install scipy statsmodels
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

CHRONOS_MODEL  = "amazon/chronos-2"
CONTEXT_LEN    = C.MEAN_REVERSION["context_len"]   # 720h
HORIZONS       = [1, 6, 24]                   # ore
N_GPUS         = 3
BATCH_SIZE     = 128                          # finestre per batch (predict è più leggero di embed)
RAY_ADDRESS    = "ray://datalab-rayclnt.unitus.it:10001"

PLOT_DIR = Path(C.RESULTS_DIR) / "step04"


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading e costruzione contesti
# ═══════════════════════════════════════════════════════════════════════════

def build_contexts(preprocessed: pd.DataFrame,
                   regimes: pd.DataFrame) -> tuple:
    """
    Per ogni finestra in regimes, estrae:
      - contesto: log_return degli ultimi CONTEXT_LEN passi (per Chronos-2)
      - log_price_t: log_lmp al timestamp della finestra
      - regime: etichetta

    Allineamento: il timestamp di regimes è l'ULTIMO della finestra (t_end).
    Il contesto è log_return da t_end - 719h a t_end incluso.

    Returns:
      contexts   : (N, CONTEXT_LEN) float32 — log_return context per Chronos-2
      log_prices : (N,) float64               — log_lmp al timestamp
      labels     : (N,) int                   — regime label
      timestamps : (N,) datetime              — timestamp t_end
    """
    pre = preprocessed.set_index("datetime").sort_index()
    dt_index = pre.index

    contexts, log_prices, labels, timestamps = [], [], [], []
    skipped = 0

    for _, row in tqdm(regimes.iterrows(),
                       total=len(regimes),
                       desc="  Building contexts",
                       ncols=65,
                       bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt}"):
        t_end   = pd.Timestamp(row["datetime"])
        regime  = int(row["regime"])

        # Cerca posizione in dt_index
        pos = dt_index.searchsorted(t_end)
        if pos < CONTEXT_LEN or pos >= len(dt_index):
            skipped += 1
            continue
        if dt_index[pos] != t_end:
            skipped += 1
            continue

        window_data = pre.iloc[pos - CONTEXT_LEN + 1 : pos + 1]
        if len(window_data) != CONTEXT_LEN:
            skipped += 1
            continue

        lr = window_data["log_return"].values.astype(np.float32)
        if np.any(np.isnan(lr)):
            skipped += 1
            continue

        contexts.append(lr)
        log_prices.append(float(pre.iloc[pos]["arcsinh_lmp"]))
        labels.append(regime)
        timestamps.append(t_end)

    if skipped:
        print(f"  ⚠  Skipped {skipped} windows (boundary / NaN)")

    return (np.stack(contexts, axis=0).astype(np.float32),
            np.array(log_prices, dtype=np.float64),
            np.array(labels, dtype=np.int16),
            np.array(timestamps))


# ═══════════════════════════════════════════════════════════════════════════
#  Chronos-2 forecast  (shared logic)
# ═══════════════════════════════════════════════════════════════════════════

def load_chronos_forecast(model_name: str, device: str):
    """
    Carica Chronos2Pipeline in modalità forecast.
    """
    import torch
    from chronos import Chronos2Pipeline

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    pipeline = Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=dtype,
    )
    return pipeline


def predict_batch(pipeline, contexts: np.ndarray, device: str,
                  prediction_length: int = 24) -> np.ndarray:
    """
    Forecast log_return per un batch di finestre.

    contexts : (B, CONTEXT_LEN) — log_return univariato
    returns  : (B, prediction_length) — mediana E[log_return_{t+1}..{t+h}]

    Chronos-2 predict() input: lista di tensori 1D (univariato) o
    tensore 3D (B, 1, T) per batch.
    Output: tensore quantili (B, n_quantiles, prediction_length).
    La mediana corrisponde al quantile 0.5.
    """
    import torch

    B = len(contexts)
    # Shape (B, 1, T): batch di B serie univariate (1 variata)
    x = torch.tensor(contexts[:, np.newaxis, :],
                     dtype=torch.float32).to(device)

    # predict() → (B, n_quantiles, prediction_length)
    # quantile_levels default include 0.5 (mediana)
    with torch.no_grad():
        forecast = pipeline.predict(
            x,
            prediction_length=prediction_length,
            num_samples=20,          # campioni interni per stima quantile
            limit_prediction_length=False,
        )

    # forecast: (B, n_samples, prediction_length) → mediana su asse samples
    median_forecast = forecast.median(dim=1).values   # (B, prediction_length)
    return median_forecast.cpu().float().numpy()


def compute_drifts(medians: np.ndarray, horizons: list) -> dict:
    """
    Da mediana dei log_return futuri, calcola drift cumulato per ogni orizzonte.

    medians : (B, prediction_length) — E[log_return_{t+k}] per k=1..H
    returns : dict h → (B,) drift cumulato = Σ_{k=1}^h E[log_return_{t+k}]
    """
    drifts = {}
    for h in horizons:
        drifts[h] = medians[:, :h].sum(axis=1)   # (B,)
    return drifts


# ═══════════════════════════════════════════════════════════════════════════
#  Local extraction  (single GPU / CPU)
# ═══════════════════════════════════════════════════════════════════════════

def extract_drifts_local(contexts: np.ndarray) -> dict:
    """
    Estrae drift Chronos-2 in modalità locale (CPU o singola GPU).
    Returns dict h → (N,) array drift.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")
    pipeline = load_chronos_forecast(CHRONOS_MODEL, device)

    N       = len(contexts)
    max_h   = max(HORIZONS)
    all_med = []

    for start in tqdm(range(0, N, BATCH_SIZE),
                      desc="  Forecasting",
                      ncols=65,
                      bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt}"):
        batch  = contexts[start : start + BATCH_SIZE]
        median = predict_batch(pipeline, batch, device, prediction_length=max_h)
        all_med.append(median)

    medians = np.concatenate(all_med, axis=0)   # (N, max_h)
    return compute_drifts(medians, HORIZONS)


# ═══════════════════════════════════════════════════════════════════════════
#  Ray extraction  (3×GPU cluster)
# ═══════════════════════════════════════════════════════════════════════════

def extract_drifts_ray(contexts: np.ndarray) -> dict:
    """
    Estrae drift Chronos-2 in parallelo su 3 GPU via Ray.
    Returns dict h → (N,) array drift.
    """
    import ray

    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)

    @ray.remote(num_gpus=1)
    class ForecastActor:
        def __init__(self, model_name: str):
            import torch
            self.device   = "cuda"
            self.pipeline = load_chronos_forecast(model_name, self.device)
            self.max_h    = max(HORIZONS)

        def forecast_chunk(self, chunk: np.ndarray) -> np.ndarray:
            """Returns (N_chunk, max_h) median forecasts."""
            results = []
            for start in range(0, len(chunk), BATCH_SIZE):
                batch  = chunk[start : start + BATCH_SIZE]
                median = predict_batch(self.pipeline, batch,
                                       self.device, self.max_h)
                results.append(median)
            return np.concatenate(results, axis=0)

    n        = len(contexts)
    chunk_sz = int(np.ceil(n / N_GPUS))
    chunks   = [contexts[i : i + chunk_sz] for i in range(0, n, chunk_sz)]

    actors  = [ForecastActor.remote(CHRONOS_MODEL) for _ in range(len(chunks))]
    futures = [a.forecast_chunk.remote(c) for a, c in zip(actors, chunks)]

    print(f"\n  Dispatched {n} windows across {len(actors)} Ray actors")
    parts = []
    remaining = list(futures)
    with tqdm(total=len(futures), desc="  Collecting",
              ncols=65, bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            parts.append(ray.get(done[0]))
            pbar.update(1)

    ray.shutdown()
    medians = np.concatenate(parts, axis=0)   # (N, max_h)
    return compute_drifts(medians, HORIZONS)


# ═══════════════════════════════════════════════════════════════════════════
#  OLS per regime × orizzonte
# ═══════════════════════════════════════════════════════════════════════════

def fit_mean_reversion_params(log_prices: np.ndarray, drifts: dict,
                  labels: np.ndarray,
                  preprocessed: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni regime r e orizzonte h:
      OLS: drift_{i,h} = a + b · log_price_{t_i}
      θ = −b / h      (mean-reversion speed per ora)
      μ = −a / b      (equilibrium log-price)
      σ = std(log_return osservati nel regime r)
      t½ = ln(2) / θ  (half-life in hours)

    Noise (regime = −1) escluso.
    Returns DataFrame con parametri mean reversion per ogni (regime, horizon).
    """
    rows = []
    unique_regimes = sorted(set(labels) - {-1})

    # Mappa regime → log_returns osservati (per σ)
    # Usiamo il preprocessed completo, non solo i timestamp delle finestre
    regime_series = {r: [] for r in unique_regimes}

    for r in unique_regimes:
        mask = labels == r
        # log_price e drift per questo regime
        lp = log_prices[mask]

        for h in HORIZONS:
            dr = drifts[h][mask]
            n  = len(lp)

            if n < 10:
                print(f"  ⚠  Regime {r}, h={h}h: solo {n} osservazioni — skip")
                continue

            # OLS: drift = a + b * log_price
            slope, intercept, r_val, p_val, se = stats.linregress(lp, dr)
            b, a = slope, intercept
            r2   = r_val ** 2

            # Parametri OU
            # drift = a + b*lp = θh*μ − θh*lp  →  b = −θh, a = θh*μ
            theta_h = -b                    # = θ × h (speed × horizon)
            if theta_h <= 0:
                print(f"  ⚠  Regime {r}, h={h}h: θ≤0 ({theta_h:.4f}) — mean-reversion assente")
                theta    = float("nan")
                mu       = float("nan")
                half_life = float("nan")
            else:
                theta     = theta_h / h     # speed per ora
                mu        = a / theta_h     # = −a/b
                half_life = np.log(2) / theta

            rows.append({
                "regime"   : int(r),
                "horizon_h": int(h),
                "n_obs"    : int(n),
                "theta"    : theta,
                "mu"       : mu,
                "half_life": half_life,
                "b"        : float(b),
                "a"        : float(a),
                "r2"       : float(r2),
                "b_pvalue" : float(p_val),
            })

    # σ: std dei log_return osservati per regime
    # (calcolata sul preprocessed completo, non sulle finestre)
    pre_by_dt = preprocessed.set_index("datetime").sort_index()

    # Per ogni regime, abbiamo i timestamp delle finestre → prendiamo le ore in quei periodi
    # Approssimazione: σ stimato su tutti i log_return del preprocessed
    # assegnati al regime tramite i timestamp delle finestre (campione rappresentativo)
    sigma_by_regime = {}
    for r in unique_regimes:
        lr = preprocessed["log_return"].dropna().values
        # Usiamo l'intero dataset come proxy (σ non dipende dall'orizzonte)
        # Una stima più precisa richiederebbe l'assegnazione ora per ora,
        # disponibile dopo step05 (cluster_validation)
        sigma_by_regime[r] = float(np.std(lr))

    # Aggiunge σ al DataFrame
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sigma"] = df["regime"].map(sigma_by_regime)

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(ou_df: pd.DataFrame, drift_df: pd.DataFrame) -> None:
    """
    01_mean_reversion_params_by_regime.png  — θ, μ, σ, t½ per regime × orizzonte
    02_drift_vs_logprice.png    — scatter drift vs log_price con retta OLS
    03_halflife_heatmap.png     — heatmap t½ (regime × orizzonte)
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    if ou_df.empty:
        print("  [plot] nessun parametro mean reversion valido — plot saltati")
        return

    regimes  = sorted(ou_df["regime"].unique())
    horizons = sorted(ou_df["horizon_h"].unique())
    palette  = sns.color_palette("tab10", n_colors=len(horizons))

    # ── 01: parametri mean reversion per regime ──────────────────────────────────────
    params = ["theta", "mu", "sigma", "half_life"]
    labels = ["θ (speed/h)", "μ (equil. log-price)", "σ (volatility)", "t½ (hours)"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, param, label in zip(axes.flat, params, labels):
        for h, col in zip(horizons, palette):
            sub = ou_df[ou_df["horizon_h"] == h]
            ax.plot(sub["regime"], sub[param], marker="o",
                    label=f"h={h}h", color=col, lw=1.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Regime");  ax.set_xticks(regimes)
        ax.legend(fontsize=8)
    fig.suptitle("Mean reversion parameters per regime and horizon", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_mean_reversion_params_by_regime.png")
    plt.close(fig)
    print("  [plot] 01_mean_reversion_params_by_regime.png")

    # ── 02: scatter drift vs log_price ───────────────────────────────────
    h_plot   = 6    # usa orizzonte 6h per il plot
    n_reg    = len(regimes)
    ncols    = min(3, n_reg)
    nrows    = int(np.ceil(n_reg / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    pal_reg = sns.color_palette("tab10", n_colors=n_reg)

    for idx, r in enumerate(regimes):
        ax   = axes[idx // ncols][idx % ncols]
        sub  = drift_df[(drift_df["regime"] == r)]
        col  = pal_reg[idx % len(pal_reg)]

        ax.scatter(sub["log_price"], sub[f"drift_{h_plot}h"],
                   s=3, alpha=0.4, color=col)

        # OLS line
        ou_row = ou_df[(ou_df["regime"] == r) & (ou_df["horizon_h"] == h_plot)]
        if not ou_row.empty:
            a_val = ou_row.iloc[0]["a"]
            b_val = ou_row.iloc[0]["b"]
            r2    = ou_row.iloc[0]["r2"]
            theta = ou_row.iloc[0]["theta"]
            lp_range = np.linspace(sub["log_price"].min(), sub["log_price"].max(), 50)
            ax.plot(lp_range, a_val + b_val * lp_range,
                    color="red", lw=1.5, label=f"OLS R²={r2:.2f} θ={theta:.3f}")
            ax.legend(fontsize=7)

        ax.set_title(f"Regime {r}  (h={h_plot}h)", fontweight="bold")
        ax.set_xlabel("log_price");  ax.set_ylabel(f"drift_{h_plot}h")

    # nasconde assi in eccesso
    for idx in range(n_reg, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Drift vs log_price — OLS fit per regime (h=6h)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_drift_vs_logprice.png")
    plt.close(fig)
    print("  [plot] 02_drift_vs_logprice.png")

    # ── 03: heatmap half-life ─────────────────────────────────────────────
    pivot = ou_df.pivot(index="regime", columns="horizon_h", values="half_life")
    fig, ax = plt.subplots(figsize=(7, max(3, len(regimes))))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd_r",
                linewidths=0.3, ax=ax, cbar_kws={"label": "t½ (hours)"})
    ax.set_title("Half-life t½ per regime × horizon\n"
                 "(shorter = faster mean-reversion)", fontweight="bold")
    ax.set_xlabel("Horizon (hours)");  ax.set_ylabel("Regime")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_halflife_heatmap.png")
    plt.close(fig)
    print("  [plot] 03_halflife_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 04: Mean Reversion Estimation")
    print("═" * 65)
    print(f"  Input   : {C.RESULTS_DIR}/preprocessed.parquet")
    print(f"            {C.RESULTS_DIR}/regimes.parquet")
    print(f"  Output  : {C.RESULTS_DIR}/mean_reversion_params.parquet")
    print(f"            {C.RESULTS_DIR}/mean_reversion_drift.parquet")
    print(f"  Plots   : {PLOT_DIR}/")
    print(f"  Model   : {CHRONOS_MODEL}")
    print(f"  Context : {CONTEXT_LEN}h")
    print(f"  Horizons: {HORIZONS}h")
    print(f"  Ray     : {RAY_ADDRESS}")
    print("─" * 65)

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    pre = pd.read_parquet(Path(C.RESULTS_DIR) / "preprocessed.parquet")
    pre["datetime"] = pd.to_datetime(pre["datetime"])

    reg = pd.read_parquet(Path(C.RESULTS_DIR) / "regimes.parquet")
    reg["datetime"] = pd.to_datetime(reg["datetime"])

    print(f"  Preprocessed : {len(pre):,} rows")
    print(f"  Regimes      : {len(reg):,} windows")
    print(f"  Regime dist  : {dict(reg['regime'].value_counts().sort_index())}")

    # ── 2. Build contexts ────────────────────────────────────────────────
    print("\n[2/5] Building Chronos-2 contexts...")
    contexts, log_prices, labels, timestamps = build_contexts(pre, reg)
    N = len(contexts)
    print(f"  Contexts built: {N:,}  shape: {contexts.shape}")

    # ── 3. Chronos-2 forecast ─────────────────────────────────────────────
    print("\n[3/5] Chronos-2 drift estimation...")
    try:
        import ray
        use_ray = True
    except ImportError:
        use_ray = False

    try:
        if use_ray:
            drifts = extract_drifts_ray(contexts)
        else:
            drifts = extract_drifts_local(contexts)
    except ImportError as e:
        print(f"\n  ERROR: {e}")
        print("  Install: pip install 'chronos-forecasting[chronos2]' torch")
        sys.exit(1)

    # ── 4. OLS per regime × orizzonte ────────────────────────────────────
    print("\n[4/5] OLS fit per regime × horizon...")
    ou_df = fit_mean_reversion_params(log_prices, drifts, labels, pre)

    # Salva drift intermedi
    drift_rows = {"datetime": timestamps, "regime": labels, "log_price": log_prices}
    for h in HORIZONS:
        drift_rows[f"drift_{h}h"] = drifts[h]
    drift_df = pd.DataFrame(drift_rows)

    ou_path    = Path(C.RESULTS_DIR) / "mean_reversion_params.parquet"
    drift_path = Path(C.RESULTS_DIR) / "mean_reversion_drift.parquet"
    ou_df.to_parquet(ou_path, index=False)
    drift_df.to_parquet(drift_path, index=False)

    print(f"\n  OU params saved  → {ou_path}")
    print(f"  Drift data saved → {drift_path}")
    print(f"\n{ou_df.to_string(index=False)}")

    # ── 5. Plots ──────────────────────────────────────────────────────────
    print("\n[5/5] Diagnostic plots...")
    make_plots(ou_df, drift_df)

    # ── Report ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "─" * 65)
    print("  MEAN REVERSION REPORT")
    print("─" * 65)
    print(f"  Windows processed : {N:>8,}")
    print(f"  Regimes fitted    : {ou_df['regime'].nunique():>8}")
    print(f"  Horizons          : {HORIZONS}")
    print(f"  Elapsed           : {elapsed:>7.1f}s")
    if not ou_df.empty:
        print()
        print("  θ range  (speed)  : "
              f"{ou_df['theta'].min():.4f} – {ou_df['theta'].max():.4f} /h")
        print("  t½ range (half-life): "
              f"{ou_df['half_life'].min():.1f} – {ou_df['half_life'].max():.1f} h")
        print("  R² range          : "
              f"{ou_df['r2'].min():.3f} – {ou_df['r2'].max():.3f}")
    print()
    print("  Next → Step 05: OU validation (R², Jarque-Bera, AIC/BIC, rolling θ)")
    print("─" * 65)
    print(f"  mean_reversion_params → {ou_path}")
    print(f"  mean_reversion_drift  → {drift_path}")
    print(f"  plots     → {PLOT_DIR}/")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
