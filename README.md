# DeRegiME

**DeRegiME** is the release code for *Deep Regime Mixtures for
Probabilistic Forecasting under Distribution Shift*.  The repository is
trimmed to the code paths used for the paper:

- DeRegiME with finite stick-breaking gates.
- Matched Gaussian, Student-t, quantile, DKL-RBF and DKL-RQ heads.
- PatchTST, DLinear and TimeMixer backbones.
- PatchTST ablations used in the appendix.

The default configuration in `deregime/config.py` is the final DeRegiME
Student-t setting.  Gate regularisers used in the experiments remain exposed,
including point entropy, batch entropy and the stick-breaking/simplex penalty.

## Code Completeness

This release is organised around the Papers with Code / NeurIPS research-code
checklist:

- **Dependencies:** `pyproject.toml` defines the installable package, and
  `requirements-tested.txt` records the versions used in the release-check
  environment.
- **Training code:** `deregime/run.py` is the main entry point, with exact
  experiment settings under `run_settings/`.
- **Evaluation code:** evaluation is run automatically at the end of training;
  per-seed metrics and seed aggregates are written under the selected
  `--out_dir`.
- **Result commands:** the commands below reproduce the PatchTST, DLinear,
  TimeMixer and PatchTST-ablation grids from the paper.
- **Data instructions:** `DATA_SOURCES.md` and `scripts/prepare_datasets.py`
  describe the public sources and preprocessing steps.

## Quick Start

All commands below assume you are working from the repository root.  If you are
elsewhere, first enter the unpacked project directory:

```bash
cd <PROJECT_DIR>
```

Create a local environment with `uv` and install the package:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

For the closest match to the release-check environment, install the recorded
versions first and then install the local package:

```bash
uv pip install --python .venv/bin/python -r requirements-tested.txt
uv pip install --python .venv/bin/python -e .
```

The equivalent non-`uv` setup is:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

Optional data-download helpers are installed separately:

```bash
uv pip install --python .venv/bin/python -e ".[data]"
```

Check the Python environment and data layout before launching training:

```bash
.venv/bin/python scripts/check_setup.py
```

If this reports missing dependencies, rerun the install command above.  If it
reports missing data files, follow the data section below.

The package intentionally avoids non-essential compiled dependencies.  The
released backbones are PatchTST, DLinear and TimeMixer.

## Data

The runner expects processed long-format CSV files under `data/`, with columns:

```text
date,cols,data
```

For example, the Electricity setting reads `data/Electricity.csv`.  A
`FileNotFoundError: data/Electricity.csv` means the Python environment is now
working and the processed data have not yet been created or linked.

### Prepare from public sources

Install the optional data helpers:

```bash
uv pip install --python .venv/bin/python -e ".[data]"
```

Download and extract the TFB preprocessed dataset bundle:

```bash
.venv/bin/python scripts/download_tfb_data.py --extract_dir raw_data/tfb
```

Then prepare the paper-format CSVs:

```bash
.venv/bin/python scripts/prepare_datasets.py --raw_dir raw_data/tfb --out_dir data
.venv/bin/python scripts/fetch_nasdaq_yfinance.py --out data/NASDAQ_1990_2025.csv
.venv/bin/python scripts/check_setup.py
```

The equivalent one-command helper is:

```bash
PYTHON_BIN=.venv/bin/python scripts/build_paper_data.sh
```

The generated `data/` and `raw_data/` directories are intentionally ignored by
git.  Recreate them with the commands above on a new machine.

If you already downloaded the raw/preprocessed CSVs manually, place them in
`raw_data/` and run the same preparation command with `--raw_dir raw_data`.
The preparation script searches recursively and recognises:

```text
ETTh1.csv
ETTh2.csv
ETTm1.csv
ETTm2.csv
exchange_rate.csv
electricity.csv
traffic.csv
national_illness.csv
ILI.csv
weather.csv
NASDAQ_1990_2025.csv
```

Paper preprocessing:

- Exchange: removes `CNY`, which is not continuous over the full period.
- Electricity and Traffic: sums all channels into one aggregate `total` series.
- WeatherC: removes sparse or discontinuous meteorological measurements
  (`rain`, `raining`, `PAR`, `SWDR`) and keeps the remaining continuous
  channels.
- Nasdaq: uses a longer 1990--2025 Nasdaq Composite close series.

The standard benchmark datasets were taken from the TFB preprocessed dataset
bundle and then transformed as described below:

- TFB benchmark repository: <https://github.com/decisionintelligence/TFB>
- TFB preprocessed dataset bundle:
  <https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link>
- Autoformer-style dataset mirror:
  <https://huggingface.co/datasets/AutonLab/Timeseries-PILE/tree/main/forecasting/autoformer>
- ETT dataset repository: <https://github.com/zhouhaoyi/ETDataset>
- Nasdaq Composite close data, ticker `^IXIC`, downloaded via `yfinance`
  from Yahoo Finance.  Recreate it with
  `scripts/fetch_nasdaq_yfinance.py`.

See `DATA_SOURCES.md` for dataset provenance, preprocessing details and the
Nasdaq/yfinance redistribution caveat.

### Use an existing processed data directory

If you already have the processed paper-format CSVs, either copy or symlink
them into this repository:

```bash
ln -s /absolute/path/to/processed/data data
.venv/bin/python scripts/check_setup.py
```

Expected processed filenames are:

```text
ETTh1.csv
ETTh2.csv
ETTm1.csv
ETTm2.csv
Exchange.csv
Electricity.csv
Traffic.csv
Illness.csv
WeatherC.csv
NASDAQ_1990_2025.csv
```

## Run Experiments

Run settings are grouped by backbone and dataset.  Each JSON file is a
sequential experiment list for one backbone/dataset pair, so it can be run on a
single GPU without a multi-GPU scheduler:

```bash
.venv/bin/python -m deregime.run \
  --experiments_json run_settings/patchtst/etth1.json \
  --out_dir runs/release
```

Equivalent shell helper:

```bash
PYTHON_BIN=.venv/bin/python \
scripts/run_sequential.sh run_settings/patchtst/etth1.json runs/release
```

Settings folders:

- `run_settings/patchtst/`: main PatchTST table.
- `run_settings/dlinear/`: appendix DLinear table.
- `run_settings/timemixer/`: appendix TimeMixer table.
- `run_settings/patchtst_ablations/`: PatchTST ablations.

Each experiment uses seeds `[42, 123, 456]` unless changed in the JSON file.

### Reproduce table grids

Run the full PatchTST grid:

```bash
for cfg in run_settings/patchtst/*.json; do
  PYTHON_BIN=.venv/bin/python scripts/run_sequential.sh "$cfg" runs/release_patchtst
done
```

Run the full DLinear appendix grid:

```bash
for cfg in run_settings/dlinear/*.json; do
  PYTHON_BIN=.venv/bin/python scripts/run_sequential.sh "$cfg" runs/release_dlinear
done
```

Run the full TimeMixer appendix grid:

```bash
for cfg in run_settings/timemixer/*.json; do
  PYTHON_BIN=.venv/bin/python scripts/run_sequential.sh "$cfg" runs/release_timemixer
done
```

Run the PatchTST ablation grid:

```bash
for cfg in run_settings/patchtst_ablations/*.json; do
  PYTHON_BIN=.venv/bin/python scripts/run_sequential.sh "$cfg" runs/release_ablations
done
```

These commands are intentionally sequential.  On a multi-GPU machine, different
dataset JSON files can be launched in parallel by setting `CUDA_VISIBLE_DEVICES`
per process, but this is not required for reproducibility.

### Batch size and gradient accumulation

The paper settings use effective `batch_size=512`.  Some DeRegiME and DKL
settings are memory intensive.  If a run does not fit on a smaller GPU,
increase `gradient_accumulation_steps` in the corresponding JSON file.  This
reduces the micro-batch size while keeping the effective batch size and
reported setting fixed.  See the paper's compute appendix for dataset-specific
guidance.

For a quick four-GPU release check:

```bash
./scripts/launch_patchtst_release_check_4gpu.sh
```

This launches Nasdaq, Electricity, Traffic and Exchange one dataset per GPU.

## Outputs

Runs write to the `--out_dir` directory.  Important files include:

- `history.csv`: training/validation trajectory.
- `best_val_metrics.json`: best validation checkpoint metrics.
- `metrics/metrics_scaled_macro__ALL_HORIZONS.json`: test metrics for a seed.
- `seed_aggregate_ALL_HORIZONS_SCALED.csv`: seed aggregate for a model.
- `diagnostics/runtime.json`: runtime, memory and early-stopping diagnostics.

There is no separate `eval.py` entry point: the runner trains, selects the best
validation checkpoint, evaluates the test set, and writes the seed aggregate in
one command.  The paper tables are computed from the `seed_aggregate_*` files.

Expected metrics include MSE, RMSE, MAE, CRPS, NLPD and interval coverage.  The
main paper reports the scaled all-horizon macro aggregates.

## Repository Hygiene

Generated data, raw downloads, run folders, logs, checkpoints, virtual
environments and local caches are ignored by git.  To check a fresh checkout,
install the package, build or link `data/`, and run:

```bash
.venv/bin/python scripts/check_setup.py
```

See `REPRODUCIBILITY.md` for a longer checklist and command examples.
