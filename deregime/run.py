import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch

import gc

from .config import CONFIGS, ConfigDict
from .data import load_long_to_wide, make_splits_and_loaders, forecasting_data_provider
from .models import (
    ProbabilisticRegimeForecaster,
    SingleEncoderForecaster,
    QuantileForecaster,
    MDNForecaster,
)
from .gp import HeteroskedasticNoiseLikelihood, RegimeStudentTLikelihood
from .training import train, evaluate_test, train_mle, train_mdn, train_quantile
from .evaluation import finalize_eval_artifacts


def _trial_forward_backward(model, batch, config, likelihood=None):
    """Run one forward+backward pass; returns True if it fits in memory."""
    device = config.device
    if len(batch) == 8:
        seq_x, seq_y, seq_x_mark, seq_y_mark, _, _mh_y, _mh_m, _ = batch
    else:
        raise NotImplementedError
    seq_x, seq_y, seq_x_mark, seq_y_mark = (
        t.to(device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
    )
    _mh_m = _mh_m.to(device) if _mh_m is not None and hasattr(_mh_m, "to") else None

    try:
        if hasattr(model, "forward_features"):
            out = model.forward_features(
                seq_x, seq_x_mark,
                seq_y_mark[:, -config.pred_len :, :],
                _mh_m, config.gp_label_len, config.pred_len,
            )
        else:
            out = model(
                seq_x, seq_x_mark,
                seq_y_mark[:, -config.pred_len :, :],
                _mh_m, config.gp_label_len, config.pred_len,
            )
        dummy_loss = sum(
            v.sum() for v in (out if isinstance(out, tuple) else out.values())
            if isinstance(v, torch.Tensor) and v.requires_grad
        )
        dummy_loss.backward()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        model.zero_grad(set_to_none=True)
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        gc.collect()


def auto_detect_accumulation(
    model, train_dataset, config, likelihood=None,
):
    """Find the largest micro-batch that fits, return (micro_batch_size, accumulation_steps)."""
    total_bs = config.batch_size
    micro_bs = total_bs
    device = config.device

    print(f"Auto-detecting gradient accumulation (target batch_size={total_bs})...")

    while micro_bs >= 1:
        dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=micro_bs, shuffle=False, drop_last=True,
            num_workers=0,
        )
        batch = next(iter(dl))
        model.train()
        if likelihood is not None:
            likelihood.train()

        fits = _trial_forward_backward(model, batch, config, likelihood)
        del dl

        if fits:
            accum = max(1, total_bs // micro_bs)
            print(
                f"  -> micro_batch_size={micro_bs} fits | "
                f"accumulation_steps={accum} | "
                f"effective_batch_size={micro_bs * accum}"
            )
            return micro_bs, accum

        print(f"  -> micro_batch_size={micro_bs} OOM, halving...")
        micro_bs = micro_bs // 2

    raise RuntimeError(
        f"Cannot fit even micro_batch_size=1 in GPU memory for batch_size={total_bs}"
    )


# ========================== RUN MANAGEMENT =======================================================
def dataset_slug(file_path: str) -> str:
    return Path(file_path).stem.lower().replace(" ", "_")


def _bytes_to_mib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 ** 2)


def _cuda_memory_diagnostics(device: str) -> Dict[str, Any]:
    """Return CUDA memory diagnostics for this process, if CUDA is active."""
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return {
            "available": torch.cuda.is_available(),
            "device": str(device),
        }

    try:
        cuda_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(cuda_idx)
        return {
            "available": True,
            "device": str(device),
            "visible_device_index": cuda_idx,
            "name": props.name,
            "total_memory_mib": _bytes_to_mib(props.total_memory),
            "memory_allocated_mib": _bytes_to_mib(torch.cuda.memory_allocated(cuda_idx)),
            "memory_reserved_mib": _bytes_to_mib(torch.cuda.memory_reserved(cuda_idx)),
            "max_memory_allocated_mib": _bytes_to_mib(
                torch.cuda.max_memory_allocated(cuda_idx)
            ),
            "max_memory_reserved_mib": _bytes_to_mib(
                torch.cuda.max_memory_reserved(cuda_idx)
            ),
        }
    except Exception as exc:  # pragma: no cover - diagnostics must never kill runs
        return {
            "available": True,
            "device": str(device),
            "error": repr(exc),
        }


def _simulate_early_stopping(
    history_df: pd.DataFrame,
    *,
    patience_limit: int,
    min_epochs: int,
) -> Dict[str, Any]:
    """Replay the training early-stopping rule for a candidate patience."""
    best_val = float("inf")
    best_epoch = None
    patience_count = 0
    stop_epoch = None

    for _, row in history_df.iterrows():
        epoch = int(row["epoch"])
        val = float(row["val_loss"])
        if not np.isfinite(val):
            val = float("inf")

        if val < best_val:
            best_val = val
            best_epoch = epoch
            patience_count = 0
        else:
            if epoch < min_epochs:
                patience_count = 0
            patience_count += 1
            if patience_count >= patience_limit:
                stop_epoch = epoch
                break

    if stop_epoch is None and len(history_df) > 0:
        stop_epoch = int(history_df["epoch"].iloc[-1])

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val if np.isfinite(best_val) else None,
        "stop_epoch": stop_epoch,
    }


def _early_stopping_patience_diagnostics(
    run_dir: Path, config: ConfigDict
) -> Dict[str, Any]:
    """Compute the smallest patience that preserves the selected checkpoint."""
    history_path = run_dir / "history.csv"
    best_path = run_dir / "best_val_metrics.json"
    if not history_path.exists():
        return {"available": False, "reason": "history.csv not found"}

    try:
        history_df = pd.read_csv(history_path)
    except Exception as exc:  # pragma: no cover - diagnostics must never kill runs
        return {"available": False, "reason": f"failed to read history.csv: {exc!r}"}

    if history_df.empty or "epoch" not in history_df or "val_loss" not in history_df:
        return {
            "available": False,
            "reason": "history.csv missing epoch/val_loss columns",
        }

    history_df = history_df[["epoch", "val_loss"]].dropna(subset=["epoch"])
    history_df = history_df.sort_values("epoch").reset_index(drop=True)
    final_epoch = int(history_df["epoch"].iloc[-1])

    actual_best_epoch = None
    actual_best_val = None
    if best_path.exists():
        try:
            with open(best_path, "r") as f:
                best_metrics = json.load(f)
            actual_best_epoch = int(best_metrics["epoch"])
            actual_best_val = float(best_metrics.get("val_loss", np.nan))
        except Exception:
            actual_best_epoch = None

    if actual_best_epoch is None:
        best_idx = history_df["val_loss"].astype(float).idxmin()
        actual_best_epoch = int(history_df.loc[best_idx, "epoch"])
        actual_best_val = float(history_df.loc[best_idx, "val_loss"])

    configured_patience = max(1, int(config.get("patience", 1)))
    min_epochs = int(config.get("min_epochs", 0))
    same_best = None
    for candidate_patience in range(1, configured_patience + 1):
        replay = _simulate_early_stopping(
            history_df,
            patience_limit=candidate_patience,
            min_epochs=min_epochs,
        )
        if replay["best_epoch"] == actual_best_epoch:
            same_best = {
                "patience": candidate_patience,
                "stop_epoch": replay["stop_epoch"],
            }
            break

    if same_best is None:
        same_best = {
            "patience": configured_patience,
            "stop_epoch": final_epoch,
        }

    post_best_epochs = max(0, final_epoch - actual_best_epoch)
    stop_epoch_at_min_patience = int(same_best["stop_epoch"])
    return {
        "available": True,
        "configured_patience": configured_patience,
        "min_epochs": min_epochs,
        "best_epoch": actual_best_epoch,
        "best_val_loss": actual_best_val,
        "final_epoch": final_epoch,
        "post_best_epochs_observed": post_best_epochs,
        "minimum_patience_same_best": int(same_best["patience"]),
        "stop_epoch_at_minimum_patience": stop_epoch_at_min_patience,
        "epochs_saved_at_minimum_patience": max(0, final_epoch - stop_epoch_at_min_patience),
        "patience_reduction_without_changing_best": max(
            0, configured_patience - int(same_best["patience"])
        ),
    }


def _write_runtime_diagnostics(
    run_dir: Path,
    *,
    exp_name: str,
    config: ConfigDict,
    start_perf: float,
    start_time_utc: str,
    status: str,
):
    """Persist per-seed runtime and VRAM diagnostics."""
    if torch.cuda.is_available() and str(config.device).startswith("cuda"):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    elapsed_seconds = time.perf_counter() - start_perf
    accumulation_steps = max(1, int(config.gradient_accumulation_steps))
    configured_batch_size = int(config.batch_size)
    micro_batch_size = max(1, configured_batch_size // accumulation_steps)
    diagnostics = {
        "status": status,
        "experiment": exp_name,
        "seed": int(config.seed),
        "dataset": dataset_slug(config.file),
        "model_type": config.model_type,
        "started_at_utc": start_time_utc,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hours": elapsed_seconds / 3600.0,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": configured_batch_size,
        "eval_batch_size": int(config.eval_batch_size),
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "configured_effective_batch_size": configured_batch_size,
        "micro_batch_size": micro_batch_size,
        "actual_effective_batch_size": micro_batch_size * accumulation_steps,
        "device": str(config.device),
        "cuda": _cuda_memory_diagnostics(config.device),
        "early_stopping": _early_stopping_patience_diagnostics(run_dir, config),
    }

    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    with open(diag_dir / "runtime.json", "w") as f:
        json.dump(diagnostics, f, indent=2)


def build_run_dir(base_out: Path, cfg: ConfigDict, exp_name: str) -> Path:
    ds = dataset_slug(cfg.file)
    run_dir = base_out / ds / exp_name
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, cfg: Dict[str, Any]):
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out.update(overrides or {})
    return out


def _load_best_ckpt(
    model,
    likelihood,
    run_dir: Path,
    ckpt_name: str,
    config: ConfigDict,
) -> bool:
    """Load the best checkpoint produced by a prior training run into the
    given model (and likelihood, for regime-GP variants). Used by
    --test_only mode to evaluate previously trained checkpoints without
    re-training. Returns True on success, False if the checkpoint is
    missing or incompatible.
    """
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        print(f"[test_only] ERROR: checkpoint not found at {ckpt_path}; "
              "cannot evaluate this run.")
        return False
    try:
        ckpt = torch.load(ckpt_path, map_location=config.device)
        model.load_state_dict(ckpt["model_state"])
        if likelihood is not None and "likelihood_state" in ckpt:
            likelihood.load_state_dict(ckpt["likelihood_state"])
        best_temp = ckpt.get("best_temperature")
        if best_temp is not None and hasattr(model, "set_temperature"):
            model.set_temperature(float(best_temp))
        print(f"[test_only] Loaded {ckpt_name} from epoch "
              f"{ckpt.get('epoch', 'N/A')}.")
        return True
    except Exception as e:
        print(f"[test_only] ERROR: failed to load {ckpt_path}: {e}")
        return False


def run_single_experiment(
    exp_name: str,
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
    base_out: Path,
):
    cfg_dict = apply_overrides(base_config, overrides or {})
    config = ConfigDict(cfg_dict)

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create the run directory and save the config
    run_dir = build_run_dir(base_out, config, exp_name)
    save_config(run_dir, dict(config))
    test_only = bool(config.get("test_only", False))
    mode_banner = " (TEST-ONLY)" if test_only else ""
    print(f"\n=== Running experiment: {exp_name}{mode_banner} | "
          f"run_dir={run_dir} ===")
    diagnostics_start_perf = time.perf_counter()
    diagnostics_start_utc = datetime.now(timezone.utc).isoformat()
    if torch.cuda.is_available() and str(config.device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    _write_runtime_diagnostics(
        run_dir,
        exp_name=exp_name,
        config=config,
        start_perf=diagnostics_start_perf,
        start_time_utc=diagnostics_start_utc,
        status="running",
    )

    # --- 1. Data Loading and Preparation ---
    wide = load_long_to_wide(
        config.file, config.date_col, config.series_id_col, config.value_col
    )
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        targ_scaler,
        train_dataset,
        train_loader,
        _,
        valid_loader,
        _,
        test_loader,
        tmark_dim,
        series_dim,
        target_dim,
    ) = make_splits_and_loaders(wide, config)

    # --- 2. Model Training based on model_type ---
    if config.model_type in ["regime_gp", "rq_gp"]:
        ## ✨ Set up the Regime GP model and likelihood ##
        model = ProbabilisticRegimeForecaster(
            series_dim, target_dim, tmark_dim, config
        ).to(config.device)

        # Pass KMeans config attributes to the model instance
        setattr(model, "kmeans_n_init", int(config.get("kmeans_n_init", 10)))
        setattr(model, "kmeans_max_iter", int(config.get("kmeans_max_iter", 100)))

        # CHECK FOR OVERRIDE
        use_student_t_likelihood = config.get("use_student_t_likelihood", False)
        gaussian_regime_mixture_likelihood = config.get(
            "gaussian_regime_mixture_likelihood", False
        )
        if (
            (use_student_t_likelihood or gaussian_regime_mixture_likelihood)
            and config.use_kernel_heteroskedastic_noise
        ):
            if use_student_t_likelihood:
                print("Initializing Regime-Switching Student-t Likelihood...")
            else:
                print("Initializing Regime-Switching Gaussian Mixture Likelihood...")
            likelihood = RegimeStudentTLikelihood(
                kernel=model.gp.covar_module,
                num_regimes=model.R,
                num_tasks=model.D,
                regime_scale_init=config.regime_noise_init,
                mc_samples=config.student_t_mc_samples,
                use_quadrature=config.student_t_use_quadrature,
                gh_points=config.student_t_gh_points,
                df_init=config.student_t_df_init,
                learn_df=config.student_t_learn_df,
                df_min=config.student_t_df_min,
                df_max=config.student_t_df_max,
                likelihood_noise_floor=config.likelihood_noise_floor,
                regime_mixture_likelihood=(
                    config.get("student_t_regime_mixture_likelihood", False)
                    or gaussian_regime_mixture_likelihood
                ),
                use_student_t=use_student_t_likelihood,
                tau_init_jitter=float(
                    config.get("regime_tau_init_jitter", 0.0)
                ),
                df_init_jitter=float(
                    config.get("regime_df_init_jitter", 0.0)
                ),
            ).to(config.device)

            if bool(config.get("tie_regime_likelihood_params", False)):
                if getattr(likelihood, "raw_log_tau_main", None) is not None:
                    likelihood.raw_log_tau_main.requires_grad_(False)
                if hasattr(likelihood, "raw_df_regime"):
                    likelihood.raw_df_regime.requires_grad_(False)

        elif (
            config.use_kernel_heteroskedastic_noise
        ):
            likelihood = HeteroskedasticNoiseLikelihood(
                kernel=model.gp.covar_module,
                num_regimes=model.R,
                num_tasks=model.D,
                regime_noise_init=config.regime_noise_init,
                likelihood_noise_floor=config.likelihood_noise_floor,
            ).to(config.device)
        else:
            raise NotImplementedError(
                "Unsupported likelihood configuration for the release settings."
            )

        if test_only:
            # In test_only mode we skip all of: gradient-accumulation
            # autodetection, inducing-point initialization, and training.
            # The trained inducing points are baked into the checkpoint's
            # state dict.
            if not _load_best_ckpt(model, likelihood, run_dir, "best.ckpt", config):
                _write_runtime_diagnostics(
                    run_dir,
                    exp_name=exp_name,
                    config=config,
                    start_perf=diagnostics_start_perf,
                    start_time_utc=diagnostics_start_utc,
                    status="skipped_missing_checkpoint",
                )
                return
            final_model = model
            final_likelihood = likelihood
            print("\nRunning final test evaluation...")
            evaluate_test(
                config=config,
                model=final_model,
                test_loader=test_loader,
                target_scaler=targ_scaler,
                run_dir=run_dir,
                likelihood=final_likelihood,
            )
            _write_runtime_diagnostics(
                run_dir,
                exp_name=exp_name,
                config=config,
                start_perf=diagnostics_start_perf,
                start_time_utc=diagnostics_start_utc,
                status="completed",
            )
            return

        # --- Auto-detect gradient accumulation if requested ---
        if config.gradient_accumulation_steps == 0:
            micro_bs, accum = auto_detect_accumulation(
                model, train_dataset, config, likelihood
            )
            config.gradient_accumulation_steps = accum
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=micro_bs, shuffle=True,
                num_workers=config.num_workers, drop_last=config.train_drop_last,
            )

        ## ✨ RESTORED: Initialize inducing points using KMeans or random sampling ##
        try:
            if str(config.get("inducing_init_method", "kmeans")).lower() == "kmeans":
                print("Initializing inducing points with KMeans...")
                model.initialize_inducing_from_loader(
                    train_loader=train_loader,
                    gp_label_len=config.gp_label_len,
                    pred_len=config.pred_len,
                    num_inducing=config.num_inducing_points,
                    pool_multiplier=int(config.get("inducing_pool_multiplier", 6)),
                    warm_batches=int(config.get("inducing_warm_batches", 8)),
                    device=config.device,
                    seed=int(config.seed),
                )
            else:  # Fallback to random initialization from one batch
                print("Initializing inducing points with random sampling...")
                batch0 = next(iter(train_loader))
                seq_x, _, seq_x_mark, seq_y_mark, _, _, _mh_m, _ = batch0
                model.initialize_inducing_from_batch(
                    seq_x.to(config.device),
                    seq_x_mark.to(config.device),
                    seq_y_mark.to(config.device),
                    _mh_m.to(config.device) if _mh_m is not None else None,
                    config.gp_label_len,
                    config.pred_len,
                    config.num_inducing_points,
                )
        except StopIteration:
            print(
                "Warning: Empty train loader; skipping inducing point initialization."
            )

        ## ✨ Train the model ##
        train(
            config,
            model,
            likelihood,
            train_loader,
            valid_loader,
            config.gp_label_len,
            target_scaler=targ_scaler,
            run_dir=run_dir,
        )

        final_model = model
        final_likelihood = likelihood

    elif config.model_type == "single_encoder_mle":
        ## ✨ Set up and train the Single Encoder MLE model ##
        model = SingleEncoderForecaster(series_dim, target_dim, tmark_dim, config).to(
            config.device
        )

        if test_only:
            if not _load_best_ckpt(model, None, run_dir, "best_mle.ckpt", config):
                _write_runtime_diagnostics(
                    run_dir,
                    exp_name=exp_name,
                    config=config,
                    start_perf=diagnostics_start_perf,
                    start_time_utc=diagnostics_start_utc,
                    status="skipped_missing_checkpoint",
                )
                return
            print("\nRunning final test evaluation...")
            evaluate_test(
                config=config,
                model=model,
                test_loader=test_loader,
                target_scaler=targ_scaler,
                run_dir=run_dir,
                likelihood=None,
            )
            _write_runtime_diagnostics(
                run_dir,
                exp_name=exp_name,
                config=config,
                start_perf=diagnostics_start_perf,
                start_time_utc=diagnostics_start_utc,
                status="completed",
            )
            return

        if config.gradient_accumulation_steps == 0:
            micro_bs, accum = auto_detect_accumulation(
                model, train_dataset, config,
            )
            config.gradient_accumulation_steps = accum
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=micro_bs, shuffle=True,
                num_workers=config.num_workers, drop_last=config.train_drop_last,
            )

        train_mle(
            config,
            model,
            train_loader,
            valid_loader,
            config.gp_label_len,
            target_scaler=targ_scaler,
            run_dir=run_dir,
        )

        final_model = model
        final_likelihood = None

    elif config.model_type == "single_encoder_quantile":
        model = QuantileForecaster(series_dim, target_dim, tmark_dim, config).to(
            config.device
        )

        if test_only:
            if not _load_best_ckpt(model, None, run_dir, "best_quantile.ckpt", config):
                _write_runtime_diagnostics(
                    run_dir,
                    exp_name=exp_name,
                    config=config,
                    start_perf=diagnostics_start_perf,
                    start_time_utc=diagnostics_start_utc,
                    status="skipped_missing_checkpoint",
                )
                return
            print("\nRunning final test evaluation...")
            evaluate_test(
                config=config,
                model=model,
                test_loader=test_loader,
                target_scaler=targ_scaler,
                run_dir=run_dir,
                likelihood=None,
            )
            _write_runtime_diagnostics(
                run_dir,
                exp_name=exp_name,
                config=config,
                start_perf=diagnostics_start_perf,
                start_time_utc=diagnostics_start_utc,
                status="completed",
            )
            return

        if config.gradient_accumulation_steps == 0:
            micro_bs, accum = auto_detect_accumulation(
                model, train_dataset, config,
            )
            config.gradient_accumulation_steps = accum
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=micro_bs, shuffle=True,
                num_workers=config.num_workers, drop_last=config.train_drop_last,
            )

        train_quantile(
            config,
            model,
            train_loader,
            valid_loader,
            config.gp_label_len,
            target_scaler=targ_scaler,
            run_dir=run_dir,
        )

        final_model = model
        final_likelihood = None

    elif config.model_type == "single_encoder_mdn":
        ## ✨ Set up and train the Mixture Density Network baseline ##
        model = MDNForecaster(series_dim, target_dim, tmark_dim, config).to(
            config.device
        )

        if test_only:
            if not _load_best_ckpt(model, None, run_dir, "best_mdn.ckpt", config):
                _write_runtime_diagnostics(
                    run_dir,
                    exp_name=exp_name,
                    config=config,
                    start_perf=diagnostics_start_perf,
                    start_time_utc=diagnostics_start_utc,
                    status="skipped_missing_checkpoint",
                )
                return
            print("\nRunning final test evaluation...")
            evaluate_test(
                config=config,
                model=model,
                test_loader=test_loader,
                target_scaler=targ_scaler,
                run_dir=run_dir,
                likelihood=None,
            )
            _write_runtime_diagnostics(
                run_dir,
                exp_name=exp_name,
                config=config,
                start_perf=diagnostics_start_perf,
                start_time_utc=diagnostics_start_utc,
                status="completed",
            )
            return

        if config.gradient_accumulation_steps == 0:
            micro_bs, accum = auto_detect_accumulation(
                model, train_dataset, config,
            )
            config.gradient_accumulation_steps = accum
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=micro_bs, shuffle=True,
                num_workers=config.num_workers, drop_last=config.train_drop_last,
            )

        train_mdn(
            config,
            model,
            train_loader,
            valid_loader,
            config.gp_label_len,
            target_scaler=targ_scaler,
            run_dir=run_dir,
        )

        final_model = model
        final_likelihood = None

    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    # --- 3. Final Evaluation ---
    ## ✨ Call the single, unified test evaluation function ##
    print("\nRunning final test evaluation...")
    evaluate_test(
        config=config,
        model=final_model,
        test_loader=test_loader,
        target_scaler=targ_scaler,
        run_dir=run_dir,
        likelihood=final_likelihood,
    )
    _write_runtime_diagnostics(
        run_dir,
        exp_name=exp_name,
        config=config,
        start_perf=diagnostics_start_perf,
        start_time_utc=diagnostics_start_utc,
        status="completed",
    )


def seed_is_complete(base_out: Path, cfg_for_seeds: Dict[str, Any], exp_name: str, seed: int) -> bool:
    """Check whether a seed run already has final test metrics."""
    ds = dataset_slug(cfg_for_seeds["file"])
    seed_dir = base_out / ds / exp_name / f"seed_{seed}"
    marker = seed_dir / "metrics" / "metrics_scaled_macro_summary.csv"
    return marker.exists()


def run_all_seeds_for_exp(
    exp_name: str,
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
    base_out: Path,
):
    """
    Runs a given experiment configuration for multiple seeds and aggregates the results.
    Skips seeds that already have completed metrics.
    """
    print(f"\n{'='*20} Starting Experiment: {exp_name} {'='*20}")

    cfg_for_seeds = apply_overrides(base_config, overrides)
    seeds = cfg_for_seeds.get("seeds", [base_config.get("seed", 42)])
    test_only = bool(cfg_for_seeds.get("test_only", False))

    for seed in seeds:
        # In test_only mode we always want to re-evaluate, even if a prior
        # metrics summary exists (the point is typically to regenerate it
        # or to produce it for runs that never reached test eval).
        if not test_only and seed_is_complete(base_out, cfg_for_seeds, exp_name, seed):
            print(f"\n--- Seed {seed} already complete, skipping ---")
            continue

        seed_exp_name = f"{exp_name}/seed_{seed}"
        seed_overrides = {**overrides, "seed": seed}

        print(f"\n--- Running seed {seed} ---")
        run_single_experiment(seed_exp_name, base_config, seed_overrides, base_out)

    # Always re-aggregate (picks up any newly completed seeds)
    print(f"\n--- Aggregating results for {exp_name} ---")
    exp_base_dir = base_out / dataset_slug(cfg_for_seeds["file"]) / exp_name
    aggregate_seed_results(
        exp_base_dir=exp_base_dir, seeds=seeds, config=ConfigDict(cfg_for_seeds)
    )


def aggregate_seed_results(exp_base_dir: Path, seeds: List[int], config: ConfigDict):
    """
    Finds test metrics from each seed directory, aggregates them, and saves summaries.
    1. Saves a per-horizon summary (mean/std across seeds).
    2. Saves an overall summary (mean/std across all seeds AND all horizons).
    """
    all_metrics = []

    for seed in seeds:
        # --- MODIFICATION HERE ---
        # Point to the 'scaled' metrics file instead of the 'raw' one.
        metrics_file = (
            exp_base_dir
            / f"seed_{seed}"
            / "metrics"
            / "metrics_scaled_macro_summary.csv"
        )
        # --- END MODIFICATION ---

        if metrics_file.exists():
            df = pd.read_csv(metrics_file)  # <-- Use pandas to read the CSV
            df["seed"] = seed
            all_metrics.append(df)
        else:
            print(f"Warning: Metrics file not found for seed {seed} at {metrics_file}")

    if not all_metrics:
        print("No metrics files found to aggregate. Skipping.")
        return

    # Combine all dataframes into one
    combined_df = pd.concat(all_metrics, ignore_index=True)

    preferred_metric_cols = [
        "mse",
        "rmse",
        "mae",
        "crps_proper",
        "nlpd_proper",
        "picp_proper",
        "crps_gauss_proxy",
        "nlpd_gauss_proxy",
        "picp_gauss_proxy",
    ]
    legacy_metric_cols = ["crps", "nlpd", "picp"]
    available_metrics = [
        col for col in preferred_metric_cols if col in combined_df.columns
    ]
    if not available_metrics:
        available_metrics = [col for col in legacy_metric_cols if col in combined_df.columns]
    for col in ["mse", "rmse", "mae"]:
        if col in combined_df.columns and col not in available_metrics:
            available_metrics.insert(["mse", "rmse", "mae"].index(col), col)

    if not available_metrics:
        print("No metrics found to aggregate. Skipping.")
        return

    agg_spec = {}
    for col in available_metrics:
        agg_spec[f"{col}_mean"] = (col, "mean")
        agg_spec[f"{col}_std"] = (col, "std")

    agg_df = combined_df.groupby("horizon").agg(**agg_spec).reset_index()

    # Save the per-horizon aggregated results
    agg_filename = config.get("seed_aggregate_name", "seed_aggregate")

    # --- MODIFICATION HERE ---
    # Save to a new file to indicate these are scaled results.
    output_path = exp_base_dir / f"{agg_filename}_SCALED.csv"
    # --- END MODIFICATION ---

    agg_df.to_csv(output_path, index=False)

    print(f"✅ Aggregated per-horizon (SCALED) results saved to: {output_path}")
    print("Aggregated Per-Horizon Metrics (SCALED, Mean ± Std):")
    for _, row in agg_df.iterrows():
        summary_parts = [
            f"MSE {row['mse_mean']:.4f} ± {row['mse_std']:.4f}"
            if "mse_mean" in row
            else None,
            f"RMSE {row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}"
            if "rmse_mean" in row
            else None,
            f"MAE {row['mae_mean']:.4f} ± {row['mae_std']:.4f}"
            if "mae_mean" in row
            else None,
            f"CRPS_p {row['crps_proper_mean']:.4f} ± {row['crps_proper_std']:.4f}"
            if "crps_proper_mean" in row
            else (
                f"CRPS {row['crps_mean']:.4f} ± {row['crps_std']:.4f}"
                if "crps_mean" in row
                else None
            ),
            f"NLPD_p {row['nlpd_proper_mean']:.4f} ± {row['nlpd_proper_std']:.4f}"
            if "nlpd_proper_mean" in row
            else (
                f"NLPD {row['nlpd_mean']:.4f} ± {row['nlpd_std']:.4f}"
                if "nlpd_mean" in row
                else None
            ),
            f"PICP_p {row['picp_proper_mean']:.3f} ± {row['picp_proper_std']:.3f}"
            if "picp_proper_mean" in row
            else (
                f"PICP {row['picp_mean']:.3f} ± {row['picp_std']:.3f}"
                if "picp_mean" in row
                else None
            ),
            f"CRPS_g {row['crps_gauss_proxy_mean']:.4f} ± {row['crps_gauss_proxy_std']:.4f}"
            if "crps_gauss_proxy_mean" in row
            else None,
            f"NLPD_g {row['nlpd_gauss_proxy_mean']:.4f} ± {row['nlpd_gauss_proxy_std']:.4f}"
            if "nlpd_gauss_proxy_mean" in row
            else None,
            f"PICP_g {row['picp_gauss_proxy_mean']:.3f} ± {row['picp_gauss_proxy_std']:.3f}"
            if "picp_gauss_proxy_mean" in row
            else None,
        ]
        print(
            f"  H={int(row['horizon']):>2}: "
            + " | ".join(part for part in summary_parts if part is not None)
        )

    # --- 2. Overall Aggregation (This logic is unchanged) ---
    overall_agg = combined_df[available_metrics].agg(["mean", "std"])

    # --- MODIFICATION HERE ---
    # Save to a new file to indicate these are scaled results.
    overall_output_path = exp_base_dir / f"{agg_filename}_ALL_HORIZONS_SCALED.csv"
    # --- END MODIFICATION ---

    overall_agg.to_csv(
        overall_output_path, index=True
    )  # index=True to keep 'mean'/'std' rows

    print(f"\n✅ Aggregated OVERALL (SCALED) results saved to: {overall_output_path}")
    print("Aggregated Overall Metrics (SCALED, Mean ± Std across all seeds/horizons):")
    for col in overall_agg.columns:
        mean = overall_agg.loc["mean", col]
        std = overall_agg.loc["std", col]
        print(f"  {col.upper()}: {mean:.4f} ± {std:.4f}")


# ========================== PARALLEL SEED RUNNER ==================================================

def run_all_seeds_parallel(
    exp_name: str,
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
    base_out: Path,
    gpu_ids: List[int],
    experiments_json: str,
):
    """
    Runs seeds in parallel across multiple GPUs using subprocess.Popen.
    Each seed gets its own OS process with CUDA_VISIBLE_DEVICES set in the
    environment *before* Python starts, guaranteeing true GPU isolation.
    """
    cfg_for_seeds = apply_overrides(base_config, overrides)
    seeds = cfg_for_seeds.get("seeds", [base_config.get("seed", 42)])
    test_only = bool(cfg_for_seeds.get("test_only", False))

    if test_only:
        # Always re-evaluate every seed in test_only mode.
        pending_seeds = list(seeds)
    else:
        pending_seeds = [
            s for s in seeds
            if not seed_is_complete(base_out, cfg_for_seeds, exp_name, s)
        ]

    if not pending_seeds:
        print(f"All {len(seeds)} seeds already complete for {exp_name}.")
    else:
        print(f"\n{'='*20} Parallel: {exp_name} | {len(pending_seeds)} seeds on GPUs {gpu_ids} {'='*20}")

        parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if parent_visible is not None:
            parent_gpu_list = [g.strip() for g in parent_visible.split(",")]
        else:
            parent_gpu_list = None

        active_procs: Dict[subprocess.Popen, tuple] = {}
        seed_queue = list(pending_seeds)
        available_gpus = list(gpu_ids)

        while seed_queue or active_procs:
            # Launch new processes on available GPUs
            while seed_queue and available_gpus:
                seed = seed_queue.pop(0)
                gpu = available_gpus.pop(0)
                env = os.environ.copy()
                if parent_gpu_list is not None:
                    physical_gpu = parent_gpu_list[gpu] if gpu < len(parent_gpu_list) else str(gpu)
                else:
                    physical_gpu = str(gpu)
                env["CUDA_VISIBLE_DEVICES"] = physical_gpu
                cmd = [
                    sys.executable, "-u", "-m", "deregime.run",
                    "--experiments_json", experiments_json,
                    "--out_dir", str(base_out),
                    "--run_one", f"{exp_name}:{seed}",
                ]
                if test_only:
                    cmd.append("--test_only")
                print(f"[GPU {physical_gpu}] Launching seed {seed} for {exp_name}")
                proc = subprocess.Popen(cmd, env=env)
                active_procs[proc] = (seed, gpu)

            # Poll for completed processes
            finished = []
            for proc, (seed, gpu) in active_procs.items():
                ret = proc.poll()
                if ret is not None:
                    finished.append(proc)
                    if ret == 0:
                        print(f"[GPU {gpu}] Seed {seed} finished successfully")
                    else:
                        print(f"[GPU {gpu}] Seed {seed} FAILED (exit code {ret})")
                    available_gpus.append(gpu)

            for proc in finished:
                del active_procs[proc]

            # Brief sleep to avoid busy-waiting
            if active_procs and not finished:
                import time
                time.sleep(5)

    # Always re-aggregate
    print(f"\n--- Aggregating results for {exp_name} ---")
    exp_base_dir = base_out / dataset_slug(cfg_for_seeds["file"]) / exp_name
    aggregate_seed_results(
        exp_base_dir=exp_base_dir, seeds=seeds, config=ConfigDict(cfg_for_seeds)
    )


# ========================== MAIN (experiments JSON supported) ====================================
def main():
    warnings.filterwarnings("ignore")
    # NB: deliberately do NOT touch torch.backends.cuda.matmul.allow_tf32 or
    # torch.backends.cudnn.allow_tf32 here. TF32 lowers matmul precision and
    # we've seen Cholesky failures on poorly-conditioned K_uu with it on.
    # Keep PyTorch's defaults (matches pre-v12 behaviour). To opt back in,
    # set the flags in your launcher env, not here.
    parser = argparse.ArgumentParser(
        description="Regime GP multi-seed runner with result aggregation."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=CONFIGS["file"],
        help="Path to long-format CSV (date, cols, data)",
    )
    parser.add_argument(
        "--experiments_json",
        type=str,
        default="settings_jan3.json",
        help='Path to JSON: [{"name":..., "overrides":{...}}, ...]',
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="baseline",
        help="Experiment name when not using --experiments_json",
    )
    parser.add_argument(
        "--out_dir", type=str, default="runs", help="Base output directory"
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index to use (e.g. --gpu 1 for second GPU). Defaults to CUDA_VISIBLE_DEVICES or 0.",
    )
    parser.add_argument(
        "--parallel_seeds", type=str, default=None,
        help="Comma-separated GPU IDs for parallel seed execution (e.g. '0,1,2,3'). "
             "Each seed runs on a separate GPU in its own process.",
    )
    parser.add_argument(
        "--run_one", type=str, default=None,
        help="Run a single seed of a single experiment: 'EXP_NAME:SEED'. "
             "Used by --parallel_seeds.",
    )
    parser.add_argument(
        "--test_only", action="store_true",
        help="Skip training; load the best checkpoint saved from a prior "
             "run and run only the final test evaluation. Useful for "
             "regenerating test metrics for runs that were interrupted "
             "before the final evaluation step.",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    base_out = Path(args.out_dir)

    # --run_one mode: run exactly one seed then exit (used by parallel launcher)
    if args.run_one:
        exp_name, seed_str = args.run_one.rsplit(":", 1)
        seed = int(seed_str)
        with open(args.experiments_json, "r") as f:
            experiments = json.load(f)
        target_exp = None
        for exp in experiments:
            if exp.get("name") == exp_name:
                target_exp = exp
                break
        if target_exp is None:
            print(f"Error: experiment '{exp_name}' not found in {args.experiments_json}")
            sys.exit(1)
        overrides = target_exp.get("overrides", {})
        seed_overrides = {**overrides, "seed": seed}
        if args.test_only:
            seed_overrides["test_only"] = True
        seed_exp_name = f"{exp_name}/seed_{seed}"
        run_single_experiment(seed_exp_name, CONFIGS, seed_overrides, base_out)
        return

    gpu_ids = None
    if args.parallel_seeds:
        gpu_ids = [int(g.strip()) for g in args.parallel_seeds.split(",")]

    if args.experiments_json:
        with open(args.experiments_json, "r") as f:
            experiments = json.load(f)
        for exp in experiments:
            name = exp.get("name")
            overrides = exp.get("overrides", {})
            if overrides.get("done", False):
                print(f"Skipping experiment '{name}' (marked done).")
                continue
            if args.test_only:
                overrides = {**overrides, "test_only": True}

            if gpu_ids:
                run_all_seeds_parallel(
                    name, CONFIGS, overrides, base_out, gpu_ids, args.experiments_json
                )
            else:
                run_all_seeds_for_exp(name, CONFIGS, overrides, base_out)
    else:
        overrides = {"file": args.file} if args.file else {}
        if args.test_only:
            overrides["test_only"] = True
        if gpu_ids:
            print("Error: --parallel_seeds requires --experiments_json")
            sys.exit(1)
        run_all_seeds_for_exp(args.exp_name, CONFIGS, overrides, base_out)


if __name__ == "__main__":
    main()
