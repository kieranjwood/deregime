#!/usr/bin/env python3
"""Prepare the CSV files expected by the public DeRegiME settings.

The training code expects long-format CSV files with columns
``date``, ``cols`` and ``data``.  This script converts common
Autoformer/TFB-style wide CSVs into that format and applies the preprocessing
used in the paper:

* Exchange: remove the discontinuous CNY channel.
* Electricity and Traffic: sum all channels into one aggregate ``total``
  series.
* Weather: keep continuous meteorological channels and remove sparse /
  intermittent measurements such as rain, PAR and SWDR.

Place raw CSV files in ``raw_data/`` by default.  The script searches
recursively, so it can be pointed directly at the extracted TFB dataset bundle.
Recognised filenames include ``ETTh1.csv``, ``ETTh2.csv``, ``ETTm1.csv``,
``ETTm2.csv``, ``Exchange.csv`` / ``exchange_rate.csv``, ``Electricity.csv``,
``Traffic.csv``, ``ILI.csv`` / ``national_illness.csv``, ``Weather.csv`` /
``weather.csv`` and ``NASDAQ_1990_2025.csv``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATASETS = {
    "etth1": ("ETTh1.csv", "ETTh1.csv"),
    "etth2": ("ETTh2.csv", "ETTh2.csv"),
    "ettm1": ("ETTm1.csv", "ETTm1.csv"),
    "ettm2": ("ETTm2.csv", "ETTm2.csv"),
    "exchange": ("exchange_rate.csv", "Exchange.csv"),
    "electricity": ("electricity.csv", "Electricity.csv"),
    "traffic": ("traffic.csv", "Traffic.csv"),
    "illness": ("national_illness.csv", "Illness.csv"),
    "weatherc": ("weather.csv", "WeatherC.csv"),
    "nasdaq": ("NASDAQ_1990_2025.csv", "NASDAQ_1990_2025.csv"),
}


ALIASES = {
    "exchange": ["Exchange.csv", "exchange.csv", "exchange_rate.csv"],
    "electricity": ["Electricity.csv", "electricity.csv"],
    "traffic": ["Traffic.csv", "traffic.csv"],
    "illness": ["ILI.csv", "Illness.csv", "national_illness.csv"],
    "weatherc": ["weather.csv", "Weather.csv", "WeatherC.csv"],
    "nasdaq": ["NASDAQ_1990_2025.csv", "nasdaq_1990_2025.csv"],
}


WEATHER_RENAME = {
    "Date Time": "date",
    "p (mbar)": "p_mbar",
    "T (degC)": "T_degC",
    "Tpot (K)": "Tpot_K",
    "Tdew (degC)": "Tdew_degC",
    "rh (%)": "rh_pct",
    "VPmax (mbar)": "VPmax_mbar",
    "VPact (mbar)": "VPact_mbar",
    "VPdef (mbar)": "VPdef_mbar",
    "sh (g/kg)": "sh_g_per_kg",
    "H2OC (mmol/mol)": "H2OC_mmol_per_mol",
    "rho (g/m**3)": "rho_g_per_m3",
    "wv (m/s)": "wv_m_per_s",
    "max. wv (m/s)": "max_wv_m_per_s",
    "wd (deg)": "wd_deg",
    "rain (mm)": "rain_mm",
    "raining (s)": "raining_s",
    "SWDR (W/m\xb2)": "SWDR_W_per_m2",
    "PAR (\xb5mol/m\xb2/s)": "PAR_umol_per_m2_s",
    "max. PAR (\xb5mol/m\xb2/s)": "max_PAR_umol_per_m2_s",
    "Tlog (degC)": "Tlog_degC",
    "OT": "CO2_ppm",
    "CO2 (ppm)": "CO2_ppm",
}

WEATHER_DROP_COLS = {
    "rain_mm",
    "raining_s",
    "SWDR_W_per_m2",
    "PAR_umol_per_m2_s",
    "max_PAR_umol_per_m2_s",
}

WEATHER_DROP_PATTERNS = (
    "rain",
    "raining",
    "par",
    "swdr",
)

EXCHANGE_CURRENCY_MAP = {
    "0": "AUD",
    "1": "GBP",
    "2": "CAD",
    "3": "CHF",
    "4": "CNY",
    "5": "JPY",
    "6": "NZD",
    "OT": "SGD",
}


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_raw(raw_dir: Path, dataset: str, default_name: str) -> Path | None:
    names = ALIASES.get(dataset, [default_name])
    for name in names:
        path = raw_dir / name
        if path.exists():
            return path
    wanted = {name.lower() for name in names}
    for path in raw_dir.rglob("*.csv"):
        if path.name.lower() in wanted:
            return path
    return None


def _date_column(df: pd.DataFrame) -> str:
    for candidate in ("date", "Date", "datetime", "time", "ds"):
        if candidate in df.columns:
            return candidate
    return df.columns[0]


def _to_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if {"date", "cols", "data"}.issubset(df.columns):
        date_order = list(dict.fromkeys(df["date"]))
        col_order = list(dict.fromkeys(df["cols"].astype(str)))
        df = df.copy()
        df["cols"] = df["cols"].astype(str)
        wide = df.pivot(index="date", columns="cols", values="data")
        wide = wide.reindex(date_order).reset_index()
        keep_order = ["date"] + [c for c in col_order if c in wide.columns]
        wide = wide[keep_order]
        return wide, "date"
    date_col = _date_column(df)
    return df.copy(), date_col


def _to_long(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if {"date", "cols", "data"}.issubset(df.columns):
        return df[["date", "cols", "data"]].copy()
    value_cols = [c for c in df.columns if c != date_col]
    out = df.melt(id_vars=[date_col], value_vars=value_cols, var_name="cols", value_name="data")
    out = out.rename(columns={date_col: "date"})
    return out[["date", "cols", "data"]]


def _drop_channels(wide: pd.DataFrame, date_col: str, channels: Iterable[str]) -> pd.DataFrame:
    drop = {_normalise_name(c) for c in channels}
    keep_cols = [
        c for c in wide.columns
        if c == date_col or _normalise_name(c) not in drop
    ]
    return wide[keep_cols]


def _drop_weather_noncontinuous(wide: pd.DataFrame, date_col: str) -> pd.DataFrame:
    keep_cols = [date_col]
    for col in wide.columns:
        if col == date_col:
            continue
        norm = _normalise_name(col)
        if col in WEATHER_DROP_COLS or any(pat in norm for pat in WEATHER_DROP_PATTERNS):
            continue
        keep_cols.append(col)
    return wide[keep_cols]


def _sum_channels(wide: pd.DataFrame, date_col: str) -> pd.DataFrame:
    numeric = wide.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({"date": wide[date_col], "cols": "total", "data": numeric.sum(axis=1)})
    return out


def _set_hourly_paper_dates(out: pd.DataFrame) -> pd.DataFrame:
    """Match the aggregate Electricity/Traffic date labels used in the paper."""
    out = out.copy()
    out["date"] = pd.date_range(
        start="2016-07-01 02:00:00",
        periods=len(out),
        freq="h",
    ).strftime("%Y-%m-%d %H:%M:%S")
    return out


def _clean_weather_wide(wide: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, str]:
    wide = wide.rename(columns=WEATHER_RENAME)
    date_col = WEATHER_RENAME.get(date_col, date_col)
    wide = wide.drop_duplicates(subset=[date_col], keep="first")
    value_cols = [c for c in wide.columns if c != date_col]
    wide[value_cols] = wide[value_cols].apply(pd.to_numeric, errors="coerce")
    wide[value_cols] = wide[value_cols].replace(-9999.0, np.nan)
    wide[value_cols] = wide[value_cols].interpolate(method="linear", limit_direction="both")
    wide = _apply_weather_tfb_timestamp_fix(wide, date_col)
    desired = [date_col]
    for v in WEATHER_RENAME.values():
        if v != "date" and v in wide.columns and v not in desired:
            desired.append(v)
    wide = wide[desired]
    return wide, date_col


def _apply_weather_tfb_timestamp_fix(wide: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Align TFB Weather timestamps with the paper-format WeatherC file.

    The TFB bundle version used here contains a duplicated timestamp value near
    2020-05-12 and a short shifted segment near 2020-05-29.  The original
    paper-format file keeps the same values but assigns the corrected timestamp
    sequence.  Apply this fix only when the characteristic TFB endpoint is
    present, so already-correct Weather files pass through unchanged.
    """
    dt = pd.to_datetime(wide[date_col])
    if dt.max() > pd.Timestamp("2020-12-31 22:40:00"):
        return wide
    if not (dt == pd.Timestamp("2020-05-12 06:10:00")).any():
        return wide

    out = wide.loc[dt != pd.Timestamp("2020-05-12 06:10:00")].copy()
    dt = pd.to_datetime(out[date_col])
    after_dup = dt > pd.Timestamp("2020-05-12 06:10:00")
    dt.loc[after_dup] = dt.loc[after_dup] - pd.Timedelta(minutes=10)
    after_gap = dt >= pd.Timestamp("2020-05-29 09:40:00")
    dt.loc[after_gap] = dt.loc[after_gap] + pd.Timedelta(minutes=90)
    out[date_col] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def prepare_one(dataset: str, raw_dir: Path, out_dir: Path, sum_weather: bool = False) -> Path | None:
    raw_name, out_name = DATASETS[dataset]
    src = _find_raw(raw_dir, dataset, raw_name)
    if src is None:
        print(f"[skip] {dataset}: no raw CSV found in {raw_dir}")
        return None

    df = pd.read_csv(src)
    wide, date_col = _to_wide(df)

    if dataset == "exchange":
        wide = wide.rename(columns=EXCHANGE_CURRENCY_MAP)
        wide = _drop_channels(wide, date_col, ["CNY"])
        out = _to_long(wide, date_col)
    elif dataset in {"electricity", "traffic"}:
        out = _set_hourly_paper_dates(_sum_channels(wide, date_col))
    elif dataset == "weatherc":
        wide, date_col = _clean_weather_wide(wide, date_col)
        wide = _drop_weather_noncontinuous(wide, date_col)
        out = _sum_channels(wide, date_col) if sum_weather else _to_long(wide, date_col)
    elif dataset == "nasdaq":
        close_col = None
        for candidate in ("Close", "close", "Adj Close", "adj_close", "AdjClose"):
            if candidate in wide.columns:
                close_col = candidate
                break
        if close_col is not None:
            out = pd.DataFrame({"date": wide[date_col], "cols": "close", "data": wide[close_col]})
        else:
            out = _to_long(wide, date_col)
    elif dataset == "illness":
        wide = wide.rename(columns={"OT": "TOTAL PATIENTS"})
        out = _to_long(wide, date_col)
    else:
        out = _to_long(wide, date_col)

    if dataset in {"illness", "weatherc"}:
        out = out[["date", "cols", "data"]]
    else:
        out = out[["date", "data", "cols"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / out_name
    out.to_csv(dest, index=False)
    print(f"[ok] {dataset}: {src} -> {dest} ({out['cols'].nunique()} channel(s))")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--out_dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=sorted(DATASETS),
        help="Datasets to prepare.",
    )
    parser.add_argument(
        "--sum_weather",
        action="store_true",
        help="Also aggregate WeatherC to one total series. The paper setting leaves WeatherC multichannel.",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        prepare_one(dataset, args.raw_dir, args.out_dir, sum_weather=args.sum_weather)


if __name__ == "__main__":
    main()
