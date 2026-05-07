# Reproducibility Checklist

This file records the intended workflow for reproducing the DeRegiME
experiments.  It is written to make the training/evaluation code, dependency
specifications, data preparation steps and expected outputs easy to audit.

## 1. Repository Contents

The repository contains:

- `deregime/`: model, data, training and evaluation code.
- `run_settings/`: exact experiment settings used for the reported model
  families.
- `scripts/prepare_datasets.py`: conversion from common public raw CSV formats
  to the long-format files used by the runner.
- `scripts/download_tfb_data.py`: optional download/extraction helper for the
  TFB preprocessed dataset bundle.
- `scripts/fetch_nasdaq_yfinance.py`: optional recreation of the extended
  Nasdaq Composite close series.
- `scripts/check_setup.py`: dependency and data-layout sanity check.
- `DATA_SOURCES.md`: source links, transformations and redistribution notes.
- `pyproject.toml`: installable package and dependency specification.
- `requirements-tested.txt`: versions used in the release-check environment.
- `README.md` and this file.

Generated `data/`, `raw_data/`, `runs/`, logs, checkpoints, virtual
environments and local caches are ignored by git.

## 2. Environment

From a clean checkout:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python scripts/check_setup.py
```

For a pinned release-check environment:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-tested.txt
uv pip install --python .venv/bin/python -e .
.venv/bin/python scripts/check_setup.py
```

The equivalent `venv`/`pip` commands are:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/check_setup.py
```

Optional data-download dependencies are installed with:

```bash
uv pip install --python .venv/bin/python -e ".[data]"
```

The package intentionally avoids optional compiled Mamba dependencies and
classical statistical baseline dependencies.  The released backbones are
PatchTST, DLinear and TimeMixer.

The tested environment was:

```text
Python 3.12
numpy 2.4.4
pandas 3.0.2
torch 2.11.0
gpytorch 1.15.2
scikit-learn 1.8.0
matplotlib 3.10.8
tqdm 4.67.3
scipy 1.17.1
```

Exact package resolution can vary by CUDA/PyTorch wheel availability.  With a
different CUDA wheel, the expected result is small numerical variation rather
than a change in the experiment specification.

## 3. Data

The paper uses public long-horizon forecasting datasets from the TFB
preprocessed dataset bundle plus an extended Nasdaq Composite close series
fetched from Yahoo Finance via `yfinance`.  Generated datasets are not
committed to git; recreate them with the scripts below.

See `DATA_SOURCES.md` for source links and preprocessing details.  A complete
data-preparation path is:

```bash
uv pip install --python .venv/bin/python -e ".[data]"
.venv/bin/python scripts/download_tfb_data.py --extract_dir raw_data/tfb
.venv/bin/python scripts/prepare_datasets.py --raw_dir raw_data/tfb --out_dir data
.venv/bin/python scripts/fetch_nasdaq_yfinance.py --out data/NASDAQ_1990_2025.csv
```

Processed files expected by all run settings:

```text
data/ETTh1.csv
data/ETTh2.csv
data/ETTm1.csv
data/ETTm2.csv
data/Exchange.csv
data/Electricity.csv
data/Traffic.csv
data/Illness.csv
data/WeatherC.csv
data/NASDAQ_1990_2025.csv
```

After placing or preparing data, run:

```bash
.venv/bin/python scripts/check_setup.py
```

## 4. Main Reproduction Commands

Each command below trains the requested experiment list, evaluates the best
validation checkpoint on the test set and writes per-seed plus aggregate
metrics.  There is no separate evaluation script.

Single PatchTST dataset/backbone file:

```bash
.venv/bin/python -m deregime.run \
  --experiments_json run_settings/patchtst/etth1.json \
  --out_dir runs/release_patchtst
```

Equivalent shell helper:

```bash
PYTHON_BIN=.venv/bin/python \
scripts/run_sequential.sh run_settings/patchtst/etth1.json runs/release_patchtst
```

Full PatchTST main grid:

```bash
for cfg in run_settings/patchtst/*.json; do
  PYTHON_BIN=.venv/bin/python scripts/run_sequential.sh "$cfg" runs/release_patchtst
done
```

Full DLinear appendix grid:

```bash
for cfg in run_settings/dlinear/*.json; do
  PYTHON_BIN=.venv/bin/python scripts/run_sequential.sh "$cfg" runs/release_dlinear
done
```

Full TimeMixer appendix grid:

```bash
for cfg in run_settings/timemixer/*.json; do
  PYTHON_BIN=.venv/bin/python scripts/run_sequential.sh "$cfg" runs/release_timemixer
done
```

Full PatchTST ablation grid:

```bash
for cfg in run_settings/patchtst_ablations/*.json; do
  PYTHON_BIN=.venv/bin/python scripts/run_sequential.sh "$cfg" runs/release_ablations
done
```

Each JSON file runs a sequential list of experiments and seeds by default.
The output file used most directly for paper-table reproduction is
`seed_aggregate_ALL_HORIZONS_SCALED.csv`.

## 5. Hardware and Runtime

Each seed uses one GPU.  DeRegiME and DKL baselines are the expensive model
families because they train sparse variational Gaussian-process components with
512 inducing points.  Non-GP Gaussian, Student-t, MDN and quantile heads are
substantially cheaper.

The paper reports A100-equivalent estimates for most datasets and measured
B200 diagnostics for Weather.  The original hardware is not required for small
sanity checks, but full reproduction of all seeds and all datasets requires
substantial GPU time.

The released settings use effective `batch_size=512`.  If a DeRegiME or DKL
run does not fit on a smaller GPU, increase `gradient_accumulation_steps` in
the relevant JSON file.  This splits the effective batch into more
micro-batches and is the intended memory knob; it should not otherwise change
the experiment definition.  The paper's compute appendix gives the
dataset-specific settings used for the reported runs.

## 6. Expected Failure Modes

- `ModuleNotFoundError: No module named 'numpy'`: the code was launched with a
  system Python rather than the uv environment.  Use `.venv/bin/python` or set
  `PYTHON_BIN=.venv/bin/python`.
- `FileNotFoundError: data/<dataset>.csv`: dependencies are installed, but the
  processed data directory is missing.  Prepare or symlink `data/`.
- CUDA out-of-memory: increase `gradient_accumulation_steps` in the relevant
  settings file.  This keeps the effective batch size fixed while reducing the
  micro-batch size.
