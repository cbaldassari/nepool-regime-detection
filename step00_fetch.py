"""
step00_fetch.py
===============
NEPOOL Regime Detection Pipeline — Step 00: Data Acquisition

Input  : nessuno  (scarica da API pubbliche)
Output : isone_dataset.parquet  (43,789 rows × 22 cols, 2021–2025)
Plots  : results/step00/

Fonti
-----
  LMP      : ISO-NE static CSV  (no autenticazione)
             https://www.iso-ne.com/static-transform/csv/histRpts/da-lmp
             Nodo: .H.INTERNAL_HUB  (Massachusetts Hub)
             Risoluzione: oraria, convenzione hour-ending (convertita in hour-beginning)

  Fuel mix : EIA API v2  (API key gratuita → https://www.eia.gov/opendata/register.php)
             https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/
             Balancing Authority: ISNE, 8 fuel type: NG, NUC, WAT, WND, SUN, COL, OIL, OTH
             Periodo: UTC → convertito in Eastern Time (US/Eastern)

Output columns (22)
-------------------
  datetime        — timestamp hour-beginning, Eastern Time (no timezone)
  lmp             — Locational Marginal Price $/MWh
  energy          — energy component $/MWh
  congestion      — congestion component $/MWh
  losses          — marginal loss component $/MWh
  natural_gas … other  — MW generati per fonte (8 colonne)
  total_mw        — MW totali (somma 8 fonti)
  natural_gas_share … other_share  — quota sul totale (8 colonne, somma=1)

Cache / Resume
--------------
  I CSV LMP vengono salvati 1 file/giorno in results/step00/lmp_cache/.
  La cache EIA è 1 parquet per fuel type in results/step00/eia_cache/.
  In caso di interruzione il fetch riprende automaticamente dai file mancanti.
  Per forzare un re-download completo: elimina la directory results/step00/.

Dependencies
------------
  pip install pandas numpy requests matplotlib seaborn tqdm pyarrow
"""

import sys
import os
import csv
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

def _progress(msg: str) -> None:
    """Stampa una riga di progresso con flush immediato."""
    print(msg, flush=True)
import requests

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ═══════════════════════════════════════════════════════════════════════════
#  Costanti (lette da config.py)
# ═══════════════════════════════════════════════════════════════════════════

BASE_LMP      = "https://www.iso-ne.com/static-transform/csv/histRpts/da-lmp"
EIA_URL       = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"

HUB_NAME      = C.FETCH["isone_hub"]         # ".H.INTERNAL_HUB"
EIA_BA        = C.FETCH["eia_ba"]            # "ISNE"
LMP_SLEEP     = C.FETCH["lmp_sleep_s"]
EIA_SLEEP     = C.FETCH["eia_sleep_s"]
MAX_RETRIES   = C.FETCH["max_retries"]
EIA_PAGE_SIZE = C.FETCH["eia_page_size"]

# Mappa codice EIA → nome colonna nel dataset
EIA_FUELTYPES = ["NG", "NUC", "WAT", "WND", "SUN", "COL", "OIL", "OTH"]
EIA_FUELNAMES = {
    "NG":  "natural_gas",
    "NUC": "nuclear",
    "WAT": "hydro",
    "WND": "wind",
    "SUN": "solar",
    "COL": "coal",
    "OIL": "oil",
    "OTH": "other",
}
FUEL_COLS = [EIA_FUELNAMES[f] for f in EIA_FUELTYPES]   # ordine fisso

FUEL_COLORS = {
    "natural_gas": "#e07b39", "nuclear": "#7b52ab",
    "hydro":       "#4da6e8", "wind":    "#6abf69",
    "solar":       "#f7c948", "coal":    "#5a5a5a",
    "oil":         "#b04040", "other":   "#aaaaaa",
}

PLOT_DIR  = Path(C.RESULTS_DIR) / "step00"
LMP_CACHE = PLOT_DIR / "lmp_cache"
EIA_CACHE = PLOT_DIR / "eia_cache"


# ═══════════════════════════════════════════════════════════════════════════
#  LMP — fetch e parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_isone_csv(text: str) -> pd.DataFrame:
    """
    Legge CSV ISO-NE con righe prefissate "C","H","D".
    Due righe H: la prima contiene i nomi colonna, la seconda i tipi → saltata.
    Usa csv.reader per gestire correttamente i campi quotati con virgole.
    """
    reader = csv.reader(text.splitlines())
    header = None
    rows   = []
    for parts in reader:
        if not parts:
            continue
        prefix = parts[0].strip()
        if prefix == "H" and header is None:
            header = [p.strip() for p in parts[1:]]
        elif prefix == "D" and header is not None:
            row = [p.strip() for p in parts[1:]]
            row += [""] * (len(header) - len(row))
            rows.append(row[:len(header)])
    if not header or not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=header)


def fetch_lmp_day(date_str: str) -> pd.DataFrame:
    """
    Scarica LMP per un singolo giorno (formato YYYYMMDD).
    Ritorna DataFrame filtrato sul nodo HUB_NAME.
    Ritorna DataFrame vuoto su 404 (giorno non disponibile) o errore persistente.
    """
    url = f"{BASE_LMP}/WW_DALMP_ISO_{date_str}.csv"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                return pd.DataFrame()   # giorno non ancora pubblicato
            if resp.status_code != 200:
                raise requests.HTTPError(f"HTTP {resp.status_code}")

            df = parse_isone_csv(resp.text)
            if df.empty or "Location Name" not in df.columns:
                return pd.DataFrame()

            # Cerca prima per nome esatto, poi per tipo HUB (fallback)
            df_hub = df[df["Location Name"] == HUB_NAME].copy()
            if df_hub.empty:
                df_hub = df[df.get("Location Type", "") == "HUB"].copy()
            return df_hub

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)   # backoff esponenziale
            else:
                return pd.DataFrame()   # fallito dopo tutti i tentativi
    return pd.DataFrame()


def fetch_lmp_range(start: str, end: str) -> pd.DataFrame:
    """
    Scarica tutti i giorni LMP nell'intervallo [start, end].
    Cache giornaliera: results/step00/lmp_cache/lmp_YYYYMMDD.parquet.
    Riprende automaticamente dai giorni mancanti in caso di interruzione.
    """
    LMP_CACHE.mkdir(parents=True, exist_ok=True)
    current  = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    n_days   = (end_dt - current).days + 1
    all_data = []
    n_cached, n_fetched, n_missing = 0, 0, 0

    _progress(f"\n  {'─'*55}")
    _progress(f"  LMP: 0/{n_days} (0%)  ···")
    day_idx = 0
    while current <= end_dt:
        date_str   = current.strftime("%Y%m%d")
        cache_file = LMP_CACHE / f"lmp_{date_str}.parquet"

        if cache_file.exists():
            df_day = pd.read_parquet(cache_file)
            n_cached += 1
        else:
            df_day = fetch_lmp_day(date_str)
            if not df_day.empty:
                df_day.to_parquet(cache_file, index=False)
                n_fetched += 1
            else:
                n_missing += 1
            time.sleep(LMP_SLEEP)

        if not df_day.empty:
            all_data.append(df_day)

        day_idx += 1
        if day_idx % 50 == 0 or day_idx == n_days:
            pct  = day_idx / n_days * 100
            bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            flag = "✓" if day_idx == n_days else "·"
            _progress(f"  LMP [{bar}] {day_idx:>4}/{n_days}  ({pct:>5.1f}%)  "
                      f"{current.strftime('%Y-%m-%d')}  {flag}")
        current += timedelta(days=1)

    print(f"  LMP: {n_cached} cached  |  {n_fetched} fetched  |  {n_missing} missing")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
#  EIA — fetch fuel mix
# ═══════════════════════════════════════════════════════════════════════════

def fetch_eia_fuel(fuel: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """
    Scarica generazione oraria per un singolo fuel type da EIA API v2.
    Paginazione automatica: continua a richiedere fino a che i dati finiscono.
    """
    all_data = []
    offset   = 0

    while True:
        params = {
            "api_key":               "0aa6ecc4e3694a179e72234add512fcc",
            "frequency":             "hourly",
            "data[0]":               "value",
            "facets[respondent][]":  EIA_BA,
            "facets[fueltype][]":    fuel,
            "start":                 start + "T00",
            "end":                   end   + "T23",
            "sort[0][column]":       "period",
            "sort[0][direction]":    "asc",
            "length":                EIA_PAGE_SIZE,
            "offset":                offset,
        }
        try:
            resp = requests.get(EIA_URL, params=params, timeout=60)
            if resp.status_code != 200:
                break
            data = resp.json().get("response", {}).get("data", [])
            if not data:
                break
            df = pd.DataFrame(data)
            df["fueltype"] = EIA_FUELNAMES.get(fuel, fuel.lower())
            all_data.append(df)
            if len(data) < EIA_PAGE_SIZE:
                break
            offset += EIA_PAGE_SIZE
            time.sleep(EIA_SLEEP)
        except Exception:
            break

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def fetch_eia_range(start: str, end: str, api_key: str) -> pd.DataFrame:
    """
    Scarica tutti gli 8 fuel type con cache per tipo.
    Cache: results/step00/eia_cache/eia_{FUEL}_{start}_{end}.parquet.
    """
    EIA_CACHE.mkdir(parents=True, exist_ok=True)
    all_data = []

    n_fuels = len(EIA_FUELTYPES)
    _progress(f"\n  {'─'*55}")
    for i, fuel in enumerate(EIA_FUELTYPES, 1):
        cache_file = EIA_CACHE / f"eia_{fuel}_{start}_{end}.parquet"
        src = "cache" if cache_file.exists() else "API  "
        _progress(f"  EIA [{i}/{n_fuels}]  {EIA_FUELNAMES.get(fuel, fuel):<12}  ← {src} ···")

        if cache_file.exists():
            df_fuel = pd.read_parquet(cache_file)
        else:
            df_fuel = fetch_eia_fuel(fuel, start, end, api_key)
            if not df_fuel.empty:
                df_fuel.to_parquet(cache_file, index=False)

        rows = len(df_fuel) if not df_fuel.empty else 0
        _progress(f"  EIA [{i}/{n_fuels}]  {EIA_FUELNAMES.get(fuel, fuel):<12}  ✓  {rows:>6,} righe")

        if not df_fuel.empty:
            all_data.append(df_fuel)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
#  Processing — LMP
# ═══════════════════════════════════════════════════════════════════════════

def process_lmp(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converte il raw CSV ISO-NE in serie oraria pulita.
      Date (MM/DD/YYYY) + HE (01–24) → datetime hour-beginning Eastern.
      Rimuove i duplicati DST (fall-back: vengono generate due righe per la stessa ora).
    """
    df = df_raw.copy()

    # HE 01–24 → offset 0–23h (hour-beginning)
    df["HE"] = pd.to_numeric(df["Hour Ending"], errors="coerce").fillna(0).astype(int)
    df["datetime"] = (
        pd.to_datetime(df["Date"], format="%m/%d/%Y")
        + pd.to_timedelta(df["HE"] - 1, unit="h")
    )

    for col in ["Locational Marginal Price", "Energy Component",
                "Congestion Component", "Marginal Loss Component"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={
        "Locational Marginal Price": "lmp",
        "Energy Component":         "energy",
        "Congestion Component":     "congestion",
        "Marginal Loss Component":  "losses",
    })

    return (df[["datetime", "lmp", "energy", "congestion", "losses"]]
            .sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="first")   # rimuove duplicati DST
            .reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════
#  Processing — EIA fuel mix
# ═══════════════════════════════════════════════════════════════════════════

def process_fuelmix(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converte il raw EIA in serie oraria con MW, total_mw e fuel shares.
      Periodo EIA: "2021-01-15T01" (UTC) → conversione in Eastern (US/Eastern).
      Pivot: long (period, fueltype, value) → wide (datetime, fuel_1, ..., fuel_8).
      Calcola total_mw e quota di ciascuna fonte.
    """
    df = df_raw.copy()

    # UTC → Eastern Time → rimuovi timezone (naive Eastern)
    df["datetime"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    df["value"]    = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    # Pivot: una colonna per fuel type
    df_pivot = df.pivot_table(
        index="datetime", columns="fueltype", values="value", aggfunc="sum"
    ).fillna(0.0)
    df_pivot.columns.name = None

    # Garantisce che tutte le 8 colonne esistano (anche se EIA non ha dati per quel fuel)
    for col in FUEL_COLS:
        if col not in df_pivot.columns:
            df_pivot[col] = 0.0

    df_pivot = df_pivot[FUEL_COLS]   # ordine fisso

    # total_mw e shares
    df_pivot["total_mw"] = df_pivot[FUEL_COLS].sum(axis=1)
    for col in FUEL_COLS:
        df_pivot[f"{col}_share"] = (
            df_pivot[col] / df_pivot["total_mw"].replace(0.0, float("nan"))
        )

    return (df_pivot.reset_index()
            .sort_values("datetime")
            .reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════
#  Validation checks
# ═══════════════════════════════════════════════════════════════════════════

def run_checks(df: pd.DataFrame) -> list:
    """
    Controlli di integrità sul dataset merged.
    Ritorna lista di stringhe (una per check) con prefisso ✓ o ⚠.
    """
    msgs = []

    # ── Copertura temporale ──────────────────────────────────────────────
    n_expected_hours = int(
        (df["datetime"].max() - df["datetime"].min()).total_seconds() / 3600
    ) + 1
    coverage = len(df) / n_expected_hours * 100
    msgs.append(
        f"{'✓' if coverage > 99 else '⚠'}  Copertura: {len(df):,} / {n_expected_hours:,} "
        f"ore ({coverage:.1f}%)"
    )

    # ── Gap temporali ────────────────────────────────────────────────────
    diffs = df["datetime"].diff().dropna()
    gaps  = diffs[diffs > pd.Timedelta("2h")]
    if len(gaps):
        msgs.append(
            f"⚠  Gap temporali > 2h: {len(gaps)} "
            f"(max {gaps.max()})  ← atteso 1 gap da ~25h (2023-06-15)"
        )
    else:
        msgs.append("✓  Nessun gap > 2h (dataset continuo)")

    # ── LMP = Energy + Congestion + Losses ───────────────────────────────
    residual = (df["lmp"] - df["energy"] - df["congestion"] - df["losses"]).abs()
    n_bad    = (residual > 0.01).sum()
    msgs.append(
        f"{'✓' if n_bad == 0 else '⚠'}  LMP = E+C+L: "
        f"{n_bad} righe con |residuo| > 0.01 $/MWh"
    )

    # ── Fuel shares somma ────────────────────────────────────────────────
    share_cols = [f"{c}_share" for c in FUEL_COLS]
    row_sums   = df[share_cols].sum(axis=1)
    bad_sum    = ((row_sums - 1.0).abs() > 0.01).sum()
    msgs.append(
        f"{'✓' if bad_sum == 0 else '⚠'}  Fuel shares sum ≈ 1: "
        f"{bad_sum} righe con errore > 0.01"
    )

    # ── NaN totali per colonna ───────────────────────────────────────────
    nan_all_shares = df[share_cols].isnull().all(axis=1).sum()
    if nan_all_shares:
        msgs.append(
            f"⚠  Righe con tutte le share NaN: {nan_all_shares} "
            f"(saranno rimosse in step01)"
        )
    else:
        msgs.append("✓  Nessuna riga con tutte le share NaN")

    # ── Statistiche LMP ──────────────────────────────────────────────────
    msgs.append(
        f"✓  LMP: min={df['lmp'].min():.1f}  "
        f"mean={df['lmp'].mean():.1f}  "
        f"max={df['lmp'].max():.1f}  $/MWh"
    )

    return msgs


# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(df: pd.DataFrame) -> None:
    """
    Genera 4 figure diagnostiche in results/step00/.

    01_lmp_coverage.png       — LMP serie temporale con componenti
    02_fuelmix_coverage.png   — MW per fonte, stacked area mensile
    03_missing_heatmap.png    — % missing per colonna × mese
    04_lmp_decomposition.png  — Energy + Congestion + Losses separati
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    dt = pd.to_datetime(df["datetime"])

    # ── 01: LMP time series ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dt, df["lmp"], lw=0.4, color="#2c6fad", alpha=0.8, label="LMP")
    pct99 = df["lmp"].quantile(0.99)
    ax.axhline(pct99, color="#d62728", lw=0.8, ls="--",
               label=f"99th pct  ${pct99:.0f}/MWh")
    ax.set_title("ISONE Mass Hub LMP — copertura temporale completa",
                 fontweight="bold")
    ax.set_ylabel("$/MWh")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_lmp_coverage.png")
    plt.close(fig)
    print("  [plot] 01_lmp_coverage.png")

    # ── 02: Fuel mix stacked area (media mensile) ────────────────────────
    df_m = (df.set_index("datetime")[FUEL_COLS]
              .resample("ME").mean()
              .fillna(0))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(
        df_m.index,
        [df_m[c] for c in FUEL_COLS],
        labels=FUEL_COLS,
        colors=[FUEL_COLORS[c] for c in FUEL_COLS],
        alpha=0.85,
    )
    ax.set_title("ISONE Fuel Mix — media mensile MW per fonte", fontweight="bold")
    ax.set_ylabel("MW")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_fuelmix_coverage.png")
    plt.close(fig)
    print("  [plot] 02_fuelmix_coverage.png")

    # ── 03: Missing data heatmap ─────────────────────────────────────────
    key_cols = (["lmp", "energy", "congestion", "losses"] +
                FUEL_COLS + ["total_mw"])
    df_miss         = df[["datetime"] + key_cols].copy()
    df_miss["month"] = pd.to_datetime(df_miss["datetime"]).dt.to_period("M").astype(str)
    miss_pct = (df_miss.groupby("month")[key_cols]
                       .apply(lambda x: x.isnull().mean() * 100))

    if not miss_pct.empty and miss_pct.values.max() > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(miss_pct.T, ax=ax, cmap="YlOrRd",
                    linewidths=0.3, cbar_kws={"label": "% missing"})
        ax.set_title("Missing data % per colonna × mese", fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "03_missing_heatmap.png")
        plt.close(fig)
        print("  [plot] 03_missing_heatmap.png")
    else:
        print("  [plot] 03_missing_heatmap.png  SKIPPED (no missing data)")

    # ── 04: LMP components decomposition ────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, col, color, ylabel in zip(
        axes,
        ["energy", "congestion", "losses"],
        ["#2c6fad", "#d62728", "#5a9e6f"],
        ["Energy $/MWh", "Congestion $/MWh", "Loss $/MWh"],
    ):
        ax.plot(dt, df[col], lw=0.3, color=color, alpha=0.7)
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=8)

    axes[0].set_title("LMP decomposition — Energy + Congestion + Losses",
                      fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_lmp_decomposition.png")
    plt.close(fig)
    print("  [plot] 04_lmp_decomposition.png")

    print(f"\n  4 figure salvate → {PLOT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> pd.DataFrame:
    import time as _time
    t0 = _time.time()

    # ── EIA API key: env var → config.py ─────────────────────────────────
    eia_key = os.environ.get("EIA_API_KEY") or C.FETCH.get("eia_api_key", "")
    if not eia_key or eia_key == "YOUR_EIA_API_KEY_HERE":
        print("\n  ERRORE: EIA_API_KEY non configurata.")
        print("  Opzione 1: variabile d'ambiente  EIA_API_KEY=<key>")
        print("  Opzione 2: FETCH['eia_api_key'] in config.py")
        print("  Registrazione gratuita: https://www.eia.gov/opendata/register.php")
        sys.exit(1)

    start = C.DATASET["start_date"]
    end   = C.DATASET["end_date"]

    print("\n" + "═" * 65)
    print("  NEPOOL Regime Detection  —  Step 00: Data Acquisition")
    print("═" * 65)
    print(f"  Periodo  : {start}  →  {end}")
    print(f"  LMP      : ISO-NE static CSV  ({HUB_NAME})")
    print(f"  Fuel mix : EIA API v2  (BA={EIA_BA},  8 fuel types)")
    print(f"  Output   : {C.DATA_PATH}")
    print(f"  Cache    : {PLOT_DIR}/lmp_cache/  +  eia_cache/")
    print(f"  Plots    : {PLOT_DIR}/")
    print()
    print("  Il download riprende automaticamente in caso di interruzione.")
    print("  Per un re-download completo eliminare results/step00/.")
    print("─" * 65)

    def _step(n: int, label: str) -> None:
        _progress(f"\n  ▶ [{n}/6] {label} ···")

    def _done(n: int, label: str, detail: str = "") -> None:
        suffix = f"  ({detail})" if detail else ""
        _progress(f"  ✓ [{n}/6] {label}{suffix}")

    # ── 1. LMP ────────────────────────────────────────────────────────────
    _step(1, "Fetch LMP  (ISO-NE static CSV)")
    df_lmp_raw = fetch_lmp_range(start, end)
    if df_lmp_raw.empty:
        _progress("  ERRORE: nessun dato LMP scaricato. Controlla la connessione.")
        sys.exit(1)
    _done(1, "Fetch LMP", f"{len(df_lmp_raw):,} righe raw")

    # ── 2. EIA ────────────────────────────────────────────────────────────
    _step(2, "Fetch fuel mix  (EIA API v2)")
    df_eia_raw = fetch_eia_range(start, end, eia_key)
    if df_eia_raw.empty:
        _progress("  ERRORE: nessun dato EIA scaricato. Controlla la API key.")
        sys.exit(1)
    _done(2, "Fetch fuel mix", f"{len(df_eia_raw):,} righe raw")

    # ── 3. Process & merge ────────────────────────────────────────────────
    _step(3, "Process & merge")
    df_lmp = process_lmp(df_lmp_raw)
    df_mix = process_fuelmix(df_eia_raw)
    df     = df_lmp.merge(df_mix, on="datetime", how="inner")
    _done(3, "Process & merge", f"{len(df):,} righe  ×  {df.shape[1]} colonne")

    # ── 4. Validation ─────────────────────────────────────────────────────
    _step(4, "Validation checks")
    check_msgs = run_checks(df)
    _done(4, "Validation checks")

    # ── 5. Save ───────────────────────────────────────────────────────────
    _step(5, "Save output")
    out_path = Path(C.DATA_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    _done(5, "Save output", f"{out_path.stat().st_size / 1e6:.1f} MB  →  {out_path}")

    # ── 6. Plots ──────────────────────────────────────────────────────────
    _step(6, "Diagnostic plots")
    make_plots(df)
    _done(6, "Diagnostic plots", f"→ {PLOT_DIR}/")

    # ── Report ────────────────────────────────────────────────────────────
    elapsed = _time.time() - t0
    n_days  = (datetime.strptime(end, "%Y-%m-%d") -
               datetime.strptime(start, "%Y-%m-%d")).days + 1

    print("\n" + "─" * 65)
    print("  DATA REPORT")
    print("─" * 65)
    print(f"  Periodo         : {start}  →  {end}  ({n_days} giorni)")
    print(f"  Ore teoriche    : {n_days * 24:>10,}")
    print(f"  Righe LMP       : {len(df_lmp):>10,}")
    print(f"  Righe EIA pivot : {len(df_mix):>10,}")
    print(f"  Righe merged    : {len(df):>10,}  (inner join)")
    print(f"  Colonne output  : {df.shape[1]:>10}")
    print(f"  Range datetime  : {df['datetime'].min().date()}"
          f"  →  {df['datetime'].max().date()}")
    print(f"  File size       : {out_path.stat().st_size / 1e6:>9.1f} MB")
    print(f"  Elapsed         : {elapsed:>9.1f}s")

    print("\n" + "─" * 65)
    print("  VALIDATION CHECKS")
    print("─" * 65)
    for msg in check_msgs:
        print(f"    {msg}")

    print("\n" + "─" * 65)
    print("  FUEL MIX SUMMARY  (mean share 2021–2025)")
    print("─" * 65)
    for col in FUEL_COLS:
        mean_s = df[f"{col}_share"].mean()
        mean_mw = df[col].mean()
        bar = "█" * int(mean_s * 30)
        print(f"    {col:<14}  {mean_s:>5.1%}  ({mean_mw:>6.0f} MW)  {bar}")

    print("\n" + "─" * 65)
    print(f"  Output  → {out_path}")
    print(f"  Cache   → {PLOT_DIR}/")
    print(f"  Plots   → {PLOT_DIR}/")
    print("  → Step 01: preprocessing e feature engineering")
    print("═" * 65 + "\n")

    return df


if __name__ == "__main__":
    main()
