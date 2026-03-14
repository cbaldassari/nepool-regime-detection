"""
config.py
=========
Central configuration for the NEPOOL Regime Detection Pipeline.
ISONE Electricity Market — Geometry-Driven Regime Identification

How to use
----------
Every feature and hyperparameter can be toggled here without touching
pipeline code. Set a flag to True/False to include/exclude a feature
or behaviour. Comments explain the trade-off for each choice.

Feature philosophy
------------------
Features are divided into three groups:
  A) PRICE DYNAMICS  — how the price moves (level, changes, volatility)
  B) FUEL MIX (ILR) — what is being dispatched (compositional geometry)
  C) DEMAND / GRID  — how much is consumed / network stress

Group A and B are always active (they are the core of the methodology).
Group C is optional: each feature can be switched on independently.
Including demand/grid features makes regimes more "operational";
excluding them makes regimes more "structural" and less calendar-driven.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════════

DATA_PATH    = "isone_dataset.parquet"   # input: merged LMP + EIA dataset
RESULTS_DIR  = "results"                 # output folder
RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════

DATASET = {
    # Temporal coverage (inclusive). Change to widen/narrow the training window.
    # Note: extending before 2021 introduces a different market structure
    # (pre-solar-boom, different baseload mix). Recommended: keep 2021-2025.
    "start_date": "2021-01-01",
    "end_date":   "2025-12-31",

    # Out-of-sample backtest period (Step 13). Must be within [start_date, end_date].
    "backtest_start": "2025-01-01",
    "backtest_end":   "2025-12-31",
}

# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE FLAGS — GROUP A: PRICE DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════
# These capture how the price moves. Always active (core features).

FEATURES_PRICE = {

    # Log-return at lag 1h: ln(LMP_t / LMP_{t-1}).
    # Captures short-term price momentum. Symmetric, additive, scale-free.
    # ALWAYS INCLUDE — foundational feature.
    "log_return_lag1": True,

    # Log-return at lag 6h: ln(LMP_t / LMP_{t-6}).
    # Captures medium-term dynamics (half a trading session).
    # Useful to distinguish spike regimes (short memory) from
    # stress regimes (sustained elevation over several hours).
    # Recommended: ON.
    "log_return_lag6": True,

    # Log-return at lag 24h: ln(LMP_t / LMP_{t-24}).
    # Captures day-over-day price change. Helps separate
    # multi-day stress events from intraday anomalies.
    # Recommended: ON.
    "log_return_lag24": True,

    # Normalised log-price: (ln(LMP_t) - mu) / sigma.
    # Captures the *level* of the price (not just its change).
    # Without this, two hours with very different price levels but
    # similar log-returns would land in the same cluster.
    # ALWAYS INCLUDE — foundational feature.
    "log_price_norm": True,

    # Rolling 24h volatility of log-returns: std(log_return, 24h window).
    # Captures local volatility regime. A regime can have low price but
    # high volatility (e.g. spring with unstable renewables) — without
    # this feature, it would merge with quiet low-price regimes.
    # Recommended: ON.
    "rolling_vol_24h": True,

    # Rolling 168h (1-week) volatility of log-returns.
    # Captures longer-term volatility persistence.
    # Use with caution: high correlation with rolling_vol_24h.
    # Recommended: OFF by default to avoid redundancy.
    "rolling_vol_168h": False,
}

# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE FLAGS — GROUP B: FUEL MIX (ILR)
# ═══════════════════════════════════════════════════════════════════════════
# Isometric Log-Ratio transform of the 8 fuel shares.
# Always active — this is the geometric core of the pipeline.
# Individual ILR coordinates can be disabled for ablation studies only.

FEATURES_ILR = {

    # ILR_1: Dispatchable vs Variable (Gas+Nuc+Coal+Oil+Other vs Hydro+Wind+Solar)
    # Captures how "renewable" the mix is. High positive = traditional dispatch.
    "ilr1_dispatch_vs_variable": True,

    # ILR_2: Fossil vs Non-fossil (Gas+Coal+Oil vs Nuclear+Other)
    # Inside dispatchable: CO2 intensity of the mix.
    "ilr2_fossil_vs_nonfossil": True,

    # ILR_3: Gas vs Solid+Liquid fossils (Gas vs Coal+Oil)
    # Key stress signal: coal and oil activate only under extreme conditions.
    "ilr3_gas_vs_solidliquid": True,

    # ILR_4: Coal vs Oil
    # Both marginal in ISNE; their ratio signals type of stress event.
    "ilr4_coal_vs_oil": True,

    # ILR_5: Nuclear vs Other (baseload non-fossil)
    # Captures nuclear outage effects (Millstone, Seabrook).
    "ilr5_nuclear_vs_other": True,

    # ILR_6: Hydro vs Intermittent (Hydro vs Wind+Solar)
    # Seasonal signal: high hydro in spring snowmelt, high solar in summer.
    "ilr6_hydro_vs_intermittent": True,

    # ILR_7: Solar vs Wind
    # Intraday cycle (solar high at noon, zero at night) + meteorological.
    "ilr7_solar_vs_wind": True,
}

# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE FLAGS — GROUP C: DEMAND & GRID
# ═══════════════════════════════════════════════════════════════════════════
# Optional features. Including these makes regimes more "operational"
# (tied to demand levels and network conditions). Excluding them makes
# regimes more "structural" (pure price + dispatch geometry).
#
# Recommendation: run clustering WITHOUT Group C, then add Group C features
# as descriptive statistics in the regime characterisation table.

FEATURES_DEMAND = {

    # Normalised system load: (total_mw - mu_L) / sigma_L.
    # Strong seasonal and intraday signal. If ON, clusters will partly
    # reflect demand cycles (summer peak, winter heating, mild shoulder).
    # Risk: regimes become "calendar-driven" rather than "structural".
    # Recommended: OFF for clustering, ON for post-hoc characterisation.
    "norm_load": False,

    # Net load: (total_mw - wind_mw - solar_mw) normalised.
    # Captures the residual demand that dispatchable units must cover.
    # More informative than raw load for renewable-rich grids.
    # Recommended: OFF (same reasoning as norm_load, with similar risk).
    "net_load_norm": False,

    # Normalised LMP congestion component: (congestion / abs(LMP)) clipped.
    # Signals transmission constraints (positive = import-constrained node,
    # negative = export-constrained). Noisy but occasionally very informative
    # during scarcity events.
    # Recommended: OFF by default; enable if congestion regimes are of interest.
    "congestion_norm": False,

    # Hour-of-day cyclical encoding: [sin(2π*h/24), cos(2π*h/24)].
    # Powerful feature but strongly calendar-driven. If ON, clustering will
    # likely identify "peak hour" vs "off-peak hour" as the dominant split,
    # overshadowing structural market dynamics.
    # Recommended: OFF for structural regime detection.
    "hour_cyclical": False,
}

# ═══════════════════════════════════════════════════════════════════════════
#  PREPROCESSING — DETREND & STANDARDISATION
# ═══════════════════════════════════════════════════════════════════════════

PREPROCESSING = {

    # ── Detrending ──────────────────────────────────────────────────────────

    # LOESS detrend on log(lmp) before computing log_return.
    # Removes the slow gas-price-driven multi-year drift from the global series.
    # NOTE: Chronos-2 normalises per-window internally, so this detrend is
    # largely redundant when the model is used as encoder. log_return is
    # already stationary by construction regardless.
    # Recommended: False (Chronos-2 handles it); set True only for ablation
    # or for pipelines that do NOT use Chronos-2.
    "loess_detrend_log_lmp":  False,
    "loess_frac":             0.10,

    # Seasonal detrend on total_mw: subtract expected hourly mean per month.
    # NOTE: same reasoning — Chronos-2 per-window normalisation removes the
    # local seasonal level automatically.
    # Recommended: False.
    "seasonal_detrend_total_mw": False,

    # ── Standardisation ─────────────────────────────────────────────────────

    # Master switch: set to False to skip standardisation entirely.
    # When False, raw values enter Chronos-2 directly.
    # NOTE: Chronos-2 has its own internal per-window normalisation, so
    # external standardisation is also largely redundant when used as encoder.
    # Both detrend and standardise can be set to False together for a clean
    # zero-preprocessing baseline. Set to True only for ablation studies or
    # for the non-Chronos-2 pipeline branch.
    # Recommended: False.
    "standardise": False,

    # Method applied to ALL active features when standardise=True.
    # "robust" : (x - median) / IQR  — handles heavy tails and spikes
    # "zscore"  : (x - mean)  / std  — assumes near-Gaussian distribution
    # Recommended: "robust" (electricity prices have heavy tails).
    "standardise_method": "robust",   # "robust" | "zscore"

    # Statistics are computed on training data only (train_end below)
    # and applied to the full dataset. Prevents data leakage into test set.
    "train_end": "2024-12-31",

    # Winsorise after standardisation: clip values to [-k, +k] IQR units.
    # Reduces the influence of residual extreme outliers on Chronos-2 input.
    # Set to None to disable winsorisation.
    # Recommended: None (keep all information; Chronos-2 handles extremes).
    "winsorise_k": None,   # e.g. 5.0 to clip at ±5 IQR units, or None
}

# ═══════════════════════════════════════════════════════════════════════════
#  ILR ZERO REPLACEMENT
# ═══════════════════════════════════════════════════════════════════════════

ILR = {
    # Multiplicative zero replacement delta (Martín-Fernández et al., 2003).
    # Replaces zeros with this value before ILR transform.
    # Must be << min(nonzero share). Typical range: 1e-4 to 1e-3.
    "zero_replacement_delta": 0.0001,

    # Fuel columns subject to zero replacement (in order of SBP).
    "fuel_cols": [
        "natural_gas_share",
        "nuclear_share",
        "hydro_share",
        "wind_share",
        "solar_share",
        "coal_share",
        "oil_share",
        "other_share",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
#  UMAP HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

UMAP_PARAMS = {
    # n_neighbors: size of local neighbourhood.
    # Low (5-15): fine local structure, risk of fragmentation.
    # High (30-100): smoother global structure, risk of merging distinct clusters.
    # Grid search range below; best value selected by DBCV.
    "n_neighbors_grid": [10, 20, 30, 50, 75, 100],
    "n_neighbors_best": 30,          # updated after grid search

    # min_dist: minimum distance between points in 2D embedding.
    # 0.0: maximally compact clusters (best for HDBSCAN input).
    # >0.1: more spread-out layout (better for visualisation only).
    "min_dist_grid": [0.0, 0.05, 0.10, 0.20],
    "min_dist_best": 0.0,            # updated after grid search

    # Output dimensionality (2 for clustering + visualisation).
    "n_components": 2,

    # Metric used in the original feature space.
    # "euclidean" is appropriate after ILR transform (isometric space).
    "metric": "euclidean",
}

# ═══════════════════════════════════════════════════════════════════════════
#  HDBSCAN HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

HDBSCAN_PARAMS = {
    # min_cluster_size: minimum number of points to form a cluster.
    # Too low → many micro-clusters (noise).
    # Too high → few large clusters (loss of detail).
    "min_cluster_size_grid": [50, 100, 200, 500],
    "min_cluster_size_best": 500,    # updated after grid search

    # min_samples: controls conservativeness of noise labelling.
    # Higher → more points labelled as noise (stricter).
    # Lower  → fewer noise points (more inclusive).
    "min_samples_grid": [10, 25, 50, 100],
    "min_samples_best": 50,          # updated after grid search

    # Cluster selection method.
    # "eom" (Excess of Mass): favours larger, stable clusters.
    # "leaf": finds smaller, more homogeneous clusters.
    "cluster_selection_method": "eom",
}

# ═══════════════════════════════════════════════════════════════════════════
#  MEAN REVERSION FIT (Step 04)
# ═══════════════════════════════════════════════════════════════════════════

MEAN_REVERSION = {
    # Forecasting horizons (hours) for drift estimation via Chronos-2.
    "horizons_h": [1, 6, 24],

    # Chronos-2 context window (hours fed into the encoder).
    "context_len": 720,   # 30 days

    # Minimum R² to consider the OU fit valid for a regime.
    "min_r2": 0.05,

    # Plausible half-life range (hours). Outside this → flag as invalid.
    "t_half_min_h": 1,
    "t_half_max_h": 720,
}

# ═══════════════════════════════════════════════════════════════════════════
#  MONTE CARLO SIMULATION (Step 11)
# ═══════════════════════════════════════════════════════════════════════════

MONTE_CARLO = {
    "n_trajectories":   1000,   # number of simulated paths
    "horizon_h":         720,   # simulation horizon (hours = 30 days)
    "regime_switch_every_h": 6, # regime transition check interval (hours)
}

# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY: build active feature list from flags above
# ═══════════════════════════════════════════════════════════════════════════

def get_active_features() -> list[str]:
    """
    Returns the ordered list of active feature names based on the flags above.
    This is the canonical feature vector E_t fed into UMAP.
    """
    active = []

    # Group A — Price dynamics
    for name, enabled in FEATURES_PRICE.items():
        if enabled:
            # hour_cyclical expands to two columns
            if name == "hour_cyclical":
                active += ["hour_sin", "hour_cos"]
            else:
                active.append(name)

    # Group B — ILR coordinates (always in fixed order ILR1..ILR7)
    for name, enabled in FEATURES_ILR.items():
        if enabled:
            active.append(name)

    # Group C — Demand & grid (optional)
    for name, enabled in FEATURES_DEMAND.items():
        if enabled:
            if name == "hour_cyclical":
                active += ["hour_sin", "hour_cos"]
            else:
                active.append(name)

    return active


if __name__ == "__main__":
    features = get_active_features()
    print(f"Active features ({len(features)}):")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")
