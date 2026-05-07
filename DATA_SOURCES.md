# Data Sources and Preprocessing

The training code expects processed long-format CSV files under `data/` with
columns:

```text
date,cols,data
```

The processed files used by the paper are generated from public sources by the
scripts in this repository.  They are not committed to git because source
redistribution rights vary across datasets; the recommended workflow is to
download the public sources and recreate the processed CSVs locally.

## Standard Forecasting Datasets

The ETT, Exchange, Electricity, Traffic, Illness and Weather datasets are
standard public long-horizon forecasting benchmarks.  In the experiments, the
starting point for these datasets was the preprocessed dataset bundle provided
by the TFB benchmark:

- TFB benchmark repository: <https://github.com/decisionintelligence/TFB>
- TFB preprocessed dataset bundle:
  <https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link>

Common upstream mirrors for overlapping datasets include:

- Autoformer-style dataset mirror:
  <https://huggingface.co/datasets/AutonLab/Timeseries-PILE/tree/main/forecasting/autoformer>
- ETT dataset repository: <https://github.com/zhouhaoyi/ETDataset>

Paper preprocessing:

- `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`: converted to long format
  if supplied in wide format.
- `Exchange.csv`: converts the standard exchange-rate dataset to long format
  and removes CNY because it is discontinuous over the full period used here.
- `Electricity.csv`: sums all client/channel columns into one aggregate
  `total` series.
- `Traffic.csv`: sums all sensor/channel columns into one aggregate `total`
  series.
- `Illness.csv`: converts `national_illness.csv` to long format.
- `WeatherC.csv`: removes sparse or discontinuous meteorological measurements
  such as rain, raining duration, PAR and SWDR; the remaining continuous
  channels are kept.  The preparation script also applies the timestamp
  compatibility correction needed for the TFB Weather file used in the paper,
  so the generated `WeatherC.csv` matches the paper-format file.

The TFB repository is MIT-licensed, but dataset redistribution rights can be
inherited from the original dataset providers.  This repository therefore cites
the TFB bundle and upstream sources and provides scripts to recreate the
processed files locally; the code license should not be interpreted as covering
all dataset contents.

Run the TFB download helper and preparation script:

```bash
uv pip install --python .venv/bin/python -e ".[data]"
.venv/bin/python scripts/download_tfb_data.py --extract_dir raw_data/tfb
.venv/bin/python scripts/prepare_datasets.py --raw_dir raw_data/tfb --out_dir data
```

The preparation script searches recursively inside the extracted bundle, so it
does not depend on a particular directory nesting.

## Nasdaq Composite

The extended Nasdaq series used in the paper is the Nasdaq Composite index
close, ticker `^IXIC`, fetched from Yahoo Finance via `yfinance`:

```python
import yfinance as yf

TICKER = "^IXIC"
START_DATE = "1990-01-01"
END_DATE = "2024-12-31"
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
```

The processed file is `NASDAQ_1990_2025.csv`, containing the `Close` series in
long format as channel `close`.  The local processed file has 8,816 daily rows
from 1990-01-02 to 2024-12-30.

To reproduce this file, install the optional data dependency and run:

```bash
uv pip install --python .venv/bin/python -e ".[data]"
.venv/bin/python scripts/fetch_nasdaq_yfinance.py --out data/NASDAQ_1990_2025.csv
```

`yfinance` is an open-source downloader for Yahoo Finance data, but the data
it retrieves are governed by Yahoo Finance and its data-provider terms.  The
helper script is provided for reproducibility; users should verify their own
rights to store or redistribute downloaded market data.
