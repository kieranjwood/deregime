# Run Settings

Each JSON file is a sequential experiment list for one backbone/dataset pair.
Run it with:

```bash
.venv/bin/python -m deregime.run \
  --experiments_json run_settings/patchtst/etth1.json \
  --out_dir runs/release
```

Folders:

- `patchtst/`: main PatchTST comparison table.
- `dlinear/`: DLinear appendix comparison table.
- `timemixer/`: TimeMixer appendix comparison table.
- `patchtst_ablations/`: PatchTST ablations on ETTh1, ETTh2, Exchange,
  Electricity, Traffic, Nasdaq and Illness.

The point-entropy, batch-entropy and stick-breaking penalty schedules are
retained in the settings so the paper runs can be reproduced.

Before launching, run:

```bash
.venv/bin/python scripts/check_setup.py
```

This catches missing dependencies or missing processed `data/*.csv` files
before starting GPU jobs.
