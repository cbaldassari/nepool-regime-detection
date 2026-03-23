"""
ISO-NE Massachusetts Hub — LMP + Fuel Mix
==========================================
LMP:      ISO-NE static CSV (no auth)
Fuel mix: EIA API hourly generation per BA=ISNE (API key gratuita)

EIA key gratuita: https://www.eia.gov/opendata/register.php
"""
import os
import csv
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_LMP    = "https://www.iso-ne.com/static-transform/csv/histRpts/da-lmp"
EIA_URL     = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"

HUB_NAME    = ".H.INTERNAL_HUB"   # Massachusetts Hub

# EIA fuel type codes per ISNE
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


# ─────────────────────────────────────────────────────────────
# Parser CSV ISO-NE (formato C/H/D con doppio header)
# ─────────────────────────────────────────────────────────────
def parse_isone_csv(text):
    """
    Legge CSV ISO-NE con righe prefissate "C","H","D".
    Ci sono DUE righe H: la prima ha i nomi, la seconda i tipi → la seconda si salta.
    Usa csv.reader per gestire campi quotati con virgole.
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


# ─────────────────────────────────────────────────────────────
# LMP  (ISO-NE static CSV, giorno per giorno)
# ─────────────────────────────────────────────────────────────
def fetch_isone_lmp(start_date, end_date, hub=HUB_NAME):
    """
    Colonne raw: Date, Hour Ending, Location ID, Location Name,
                 Location Type, Locational Marginal Price,
                 Energy Component, Congestion Component, Marginal Loss Component
    """
    all_data = []
    current  = datetime.strptime(start_date, "%Y-%m-%d")
    end      = datetime.strptime(end_date,   "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        url = f"{BASE_LMP}/WW_DALMP_ISO_{date_str}.csv"

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = parse_isone_csv(resp.text)
                if not df.empty and "Location Name" in df.columns:
                    df_hub = df[df["Location Name"] == hub].copy()
                    if not df_hub.empty:
                        all_data.append(df_hub)
                        print(f"OK LMP: {date_str} → {len(df_hub)} righe")
                    else:
                        # fallback: cerca per Location Type == HUB
                        df_hub = df[df["Location Type"] == "HUB"].copy()
                        if not df_hub.empty:
                            all_data.append(df_hub)
                            print(f"OK LMP (HUB fallback): {date_str} → {len(df_hub)} righe")
                        else:
                            print(f"WARN LMP: {date_str} — hub non trovato. "
                                  f"Location Types: {df['Location Type'].unique().tolist()[:5]}")
                else:
                    print(f"WARN LMP: {date_str} — colonne: {df.columns.tolist()[:6]}")
            else:
                print(f"HTTP {resp.status_code} LMP per {date_str}")
        except Exception as e:
            print(f"ERRORE LMP {date_str}: {e}")

        current += timedelta(days=1)
        time.sleep(0.3)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Fuel Mix  (EIA API, BA=ISNE)
# ─────────────────────────────────────────────────────────────
def fetch_eia_fuelmix(start_date, end_date, api_key, ba="ISNE"):
    """
    Scarica generazione oraria per tipo combustibile da EIA API v2.
    Ritorna DataFrame long: period (UTC), fueltype, value (MWh)
    """
    all_data = []

    for fuel in EIA_FUELTYPES:
        offset = 0
        length = 5000
        print(f"  EIA fetch: {ba} / {fuel} ...", end="", flush=True)

        while True:
            params = {
                "api_key":               api_key,
                "frequency":             "hourly",
                "data[0]":               "value",
                "facets[respondent][]":  ba,
                "facets[fueltype][]":    fuel,
                "start":                 start_date + "T00",
                "end":                   end_date   + "T23",
                "sort[0][column]":       "period",
                "sort[0][direction]":    "asc",
                "length":                length,
                "offset":                offset,
            }

            try:
                resp = requests.get(EIA_URL, params=params, timeout=60)
                if resp.status_code != 200:
                    print(f" HTTP {resp.status_code}")
                    break

                data = resp.json().get("response", {}).get("data", [])
                if not data:
                    print(f" 0 righe")
                    break

                df = pd.DataFrame(data)
                df["fueltype"] = EIA_FUELNAMES.get(fuel, fuel.lower())
                all_data.append(df)

                if len(data) < length:
                    print(f" {offset + len(data)} righe")
                    break
                offset += length

            except Exception as e:
                print(f" ERRORE: {e}")
                break

        time.sleep(0.5)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Processing LMP
# ─────────────────────────────────────────────────────────────
def process_lmp(df_raw):
    """
    Date (MM/DD/YYYY) + HE (01-24) → datetime orario (inizio ora).
    """
    df = df_raw.copy()
    df["HE"] = pd.to_numeric(df["Hour Ending"], errors="coerce").fillna(0).astype(int)
    df["datetime"] = (
        pd.to_datetime(df["Date"], format="%m/%d/%Y")
        + pd.to_timedelta(df["HE"] - 1, unit="h")
    )
    for col in ["Locational Marginal Price", "Energy Component",
                "Congestion Component", "Marginal Loss Component"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={
        "Locational Marginal Price":  "lmp",
        "Energy Component":           "energy",
        "Congestion Component":       "congestion",
        "Marginal Loss Component":    "losses",
    })
    return df[["datetime", "lmp", "energy", "congestion", "losses"]].sort_values("datetime").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Processing Fuel Mix (EIA)
# ─────────────────────────────────────────────────────────────
def process_fuelmix(df_raw):
    """
    EIA periodo: "2021-01-15T01" (UTC) → converte in Eastern → pivot → fuel share.
    """
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    df["value"]    = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    df_pivot = df.pivot_table(
        index="datetime",
        columns="fueltype",
        values="value",
        aggfunc="sum"
    ).fillna(0)

    df_pivot.columns.name = None
    fuel_cols = df_pivot.columns.tolist()

    df_pivot["total_mw"] = df_pivot[fuel_cols].sum(axis=1)
    for col in fuel_cols:
        df_pivot[f"{col}_share"] = df_pivot[col] / df_pivot["total_mw"].replace(0, float("nan"))

    return df_pivot.reset_index()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── CONFIGURA QUI ────────────────────────────────────────
    EIA_API_KEY = "0aa6ecc4e3694a179e72234add512fcc"   # https://www.eia.gov/opendata/register.php
    START = "2021-01-01"
    END   = "2025-12-31"
    # ─────────────────────────────────────────────────────────

    print("=" * 55)
    print(f"Scarico LMP ISO-NE Mass Hub — {START} → {END}")
    print("=" * 55)
    df_lmp_raw = fetch_isone_lmp(START, END)

    print("\n" + "=" * 55)
    print(f"Scarico fuel mix EIA (ISNE) — {START} → {END}")
    print("=" * 55)
    df_mix_raw = fetch_eia_fuelmix(START, END, api_key=EIA_API_KEY)

    # processing
    df_lmp = process_lmp(df_lmp_raw)
    df_mix = process_fuelmix(df_mix_raw)

    # join su ora
    df = df_lmp.merge(df_mix, on="datetime", how="inner")

    # salva
    lmp_path  = os.path.join(SCRIPT_DIR, "isone_masshub_lmp.parquet")
    mix_path  = os.path.join(SCRIPT_DIR, "isone_fuelmix.parquet")
    join_path = os.path.join(SCRIPT_DIR, "isone_dataset.parquet")

    df_lmp.to_parquet(lmp_path)
    df_mix.to_parquet(mix_path)
    df.to_parquet(join_path)

    print("\n" + "=" * 55)
    print("DONE")
    print(f"LMP shape:     {df_lmp.shape}")
    print(f"Mix shape:     {df_mix.shape}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColonne dataset:\n{df.columns.tolist()}")
    print(f"\nRange temporale: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"\nFile salvati in: {SCRIPT_DIR}")