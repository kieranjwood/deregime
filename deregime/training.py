import json
import sys
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import gpytorch
from tqdm import tqdm

from .config import ConfigDict
from .models import ProbabilisticRegimeForecaster, SingleEncoderForecaster
from .evaluation import (
    _build_pred_targets,
    _build_pred_times_flat,
    _metric_options_from_config,
    compute_gaussian_metrics,
    compute_mdn_metrics,
    compute_regime_student_t_mixture_metrics,
    compute_student_t_metrics,
    compute_student_t_mixture_metrics,
    finalize_eval_artifacts,
)

is_interactive = sys.stdout.isatty()


# ========================== TRAIN / EVAL =========================================================
def ramp(epoch: int, init: float, final: float, steps: int) -> float:
    if steps <= 0:
        return final
    if epoch >= steps:
        return final
    return init + (final - init) * (epoch / steps)


def current_noise(likelihood) -> float:
    try:
        # Standard GPyTorch or our wrapper property
        t = likelihood.noise
    except AttributeError:
        t = likelihood.noise_covar.noise

    return float(t.mean().detach().cpu().item())


def _student_t_metric_options(config: ConfigDict, likelihood=None) -> dict:
    opts = _metric_options_from_config(config)
    if likelihood is not None and hasattr(likelihood, "gh_points"):
        opts["gh_points"] = int(getattr(likelihood, "gh_points"))
    return opts


def _revin_output_scale(model, batch_std: torch.Tensor) -> torch.Tensor:
    scale = batch_std
    if getattr(model.revin, "affine", False):
        w = model.revin.affine_weight.view(1, 1, -1)
        eps2 = float(getattr(model.revin, "eps", 0.0)) ** 2
        scale = scale / (w + eps2).abs()
    return scale


def _normalize_targets_with_current_revin(
    model,
    y_target: torch.Tensor,
    batch_mean: torch.Tensor,
) -> torch.Tensor:
    B = batch_mean.shape[0]
    D = getattr(model, "D", batch_mean.size(-1))
    if batch_mean.size(-1) != D:
        raise NotImplementedError(
            "RevIN target normalization expects input/output channel counts to "
            f"match; got stats D={batch_mean.size(-1)} and model D={D}."
        )
    y_3d = y_target.reshape(B, D, -1).transpose(1, 2)
    y_norm = model.revin._normalize(y_3d)
    return y_norm.transpose(1, 2).reshape(B * D, -1)


def _uses_latent_mean_shift(model) -> bool:
    return bool(
        getattr(model, "dm_mode", "none") != "none"
        or getattr(model, "use_residual_mle_backbone", False)
    )


def _prediction_likelihood_overrides(aux: dict | None) -> dict:
    if not isinstance(aux, dict):
        return {
            "residual_obs_var": None,
            "backbone_obs_scale": None,
            "backbone_df": None,
        }
    return {
        "residual_obs_var": aux.get("residual_obs_var_pred"),
        "backbone_obs_scale": aux.get("backbone_obs_scale_pred"),
        "backbone_df": aux.get("backbone_df_pred"),
    }


def _uses_distributional_gp_likelihood(config: ConfigDict, likelihood) -> bool:
    """True for Student-t DeRegime and proper Gaussian regime mixtures."""
    return bool(
        (config.get("use_student_t_likelihood", False) and hasattr(likelihood, "local_df"))
        or (
            getattr(likelihood, "regime_mixture_likelihood", False)
            and hasattr(likelihood, "regime_component_params")
        )
    )


def _gp_student_t_predictive_tensors(
    model,
    likelihood,
    latent_pred_mvn,
    feats_pred_only: torch.Tensor,
    batch_std: torch.Tensor,
    *,
    kernel,
    residual_obs_var: torch.Tensor | None = None,
    backbone_obs_scale: torch.Tensor | None = None,
    backbone_df: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    pm_norm = latent_pred_mvn.mean
    latent_std_norm = latent_pred_mvn.stddev
    B_curr = batch_std.shape[0]
    pm_3d = pm_norm.reshape(B_curr, model.D, -1).transpose(1, 2)
    latent_std_3d = latent_std_norm.reshape(B_curr, model.D, -1).transpose(1, 2)
    pm_denorm = model.revin._denormalize(pm_3d)
    out_scale = _revin_output_scale(model, batch_std)
    latent_std_denorm = latent_std_3d * out_scale

    if getattr(likelihood, "regime_mixture_likelihood", False) and hasattr(
        likelihood, "regime_component_params"
    ):
        weights_norm, comp_scale_norm, comp_df = likelihood.regime_component_params(
            feats_pred_only,
            kernel=kernel,
            residual_obs_var=residual_obs_var,
            backbone_obs_scale=backbone_obs_scale,
            backbone_df=backbone_df,
        )
        R = weights_norm.size(-1)
        weights_4d = weights_norm.reshape(B_curr, model.D, -1, R).transpose(1, 2)
        comp_scale_4d = comp_scale_norm.reshape(B_curr, model.D, -1, R).transpose(
            1, 2
        )
        comp_scale_denorm = comp_scale_4d * out_scale.unsqueeze(-1)
        if comp_df is not None:
            comp_df_4d = comp_df.reshape(B_curr, model.D, -1, R).transpose(1, 2)
            comp_var_denorm = (
                comp_df_4d
                / (comp_df_4d - 2.0).clamp_min(1e-6)
                * comp_scale_denorm.pow(2)
            )
        else:
            comp_df_4d = None
            comp_var_denorm = comp_scale_denorm.pow(2)
        obs_var_denorm = (weights_4d * comp_var_denorm).sum(dim=-1)
        moment_std_denorm = torch.sqrt(latent_std_denorm.pow(2) + obs_var_denorm)
        residual_obs_var_denorm = None
        if residual_obs_var is not None:
            residual_obs_var_denorm = (
                residual_obs_var.to(device=pm_norm.device, dtype=pm_norm.dtype)
                .reshape(B_curr, model.D, -1)
                .transpose(1, 2)
                * out_scale.pow(2)
            )
        result = {
            "mu": pm_denorm,
            "latent_std": latent_std_denorm,
            "obs_scale": torch.sqrt(obs_var_denorm.clamp_min(1e-12)),
            "sd": moment_std_denorm,
            "residual_obs_var": residual_obs_var_denorm,
            "mixture_weights": weights_4d,
            "obs_scale_components": comp_scale_denorm,
        }
        if comp_df_4d is not None:
            result["df"] = (weights_4d * comp_df_4d).sum(dim=-1)
            result["df_components"] = comp_df_4d
        return result

    if backbone_obs_scale is not None:
        obs_scale_norm = backbone_obs_scale.to(device=pm_norm.device, dtype=pm_norm.dtype)
    else:
        obs_scale_norm = likelihood.local_scale(
            feats_pred_only, kernel=kernel, residual_obs_var=residual_obs_var
        ).to(device=pm_norm.device, dtype=pm_norm.dtype)
    if backbone_df is not None:
        local_df = backbone_df.to(device=pm_norm.device, dtype=pm_norm.dtype)
    else:
        local_df = likelihood.local_df(feats_pred_only, kernel=kernel).to(
            device=pm_norm.device, dtype=pm_norm.dtype
        )
    if backbone_obs_scale is not None and residual_obs_var is not None:
        residual = residual_obs_var.to(device=pm_norm.device, dtype=pm_norm.dtype)
        obs_var = (
            obs_scale_norm.pow(2)
            * local_df
            / (local_df - 2.0).clamp_min(1e-6)
        )
        obs_var = obs_var + residual.clamp_min(0.0)
        obs_scale_norm = torch.sqrt(
            (obs_var * (local_df - 2.0) / local_df).clamp_min(1e-12)
        )

    obs_scale_3d = obs_scale_norm.reshape(B_curr, model.D, -1).transpose(1, 2)
    df_3d = local_df.reshape(B_curr, model.D, -1).transpose(1, 2)
    residual_obs_var_3d = None
    if residual_obs_var is not None:
        residual_obs_var_3d = (
            residual_obs_var.to(device=pm_norm.device, dtype=pm_norm.dtype)
            .reshape(B_curr, model.D, -1)
            .transpose(1, 2)
        )

    obs_scale_denorm = obs_scale_3d * out_scale
    residual_obs_var_denorm = (
        residual_obs_var_3d * out_scale.pow(2)
        if residual_obs_var_3d is not None
        else None
    )
    moment_std_denorm = torch.sqrt(
        latent_std_denorm.pow(2)
        + (df_3d / (df_3d - 2.0).clamp_min(1e-6)) * obs_scale_denorm.pow(2)
    )

    return {
        "mu": pm_denorm,
        "latent_std": latent_std_denorm,
        "obs_scale": obs_scale_denorm,
        "df": df_3d,
        "sd": moment_std_denorm,
        "residual_obs_var": residual_obs_var_denorm,
    }


def _append_metric_arrays(dest: dict, values: dict) -> None:
    for key, value in values.items():
        dest.setdefault(key, []).append(value)


def _empty_metric_result() -> dict:
    return {
        "mse": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "crps": np.nan,
        "nlpd": np.nan,
        "picp": np.nan,
        "crps_proper": np.nan,
        "nlpd_proper": np.nan,
        "picp_proper": np.nan,
        "crps_gauss_proxy": np.nan,
        "nlpd_gauss_proxy": np.nan,
        "picp_gauss_proxy": np.nan,
        "n": 0,
    }


def _metric_history_fields(metrics: dict, *, prefix: str = "val_") -> dict:
    return {
        f"{prefix}mse": float(metrics["mse"]),
        f"{prefix}rmse": float(metrics["rmse"]),
        f"{prefix}mae": float(metrics["mae"]),
        f"{prefix}crps": float(metrics["crps"]),
        f"{prefix}nlpd": float(metrics["nlpd"]),
        f"{prefix}picp": float(metrics["picp"]),
        f"{prefix}crps_proper": float(metrics["crps_proper"]),
        f"{prefix}nlpd_proper": float(metrics["nlpd_proper"]),
        f"{prefix}picp_proper": float(metrics["picp_proper"]),
        f"{prefix}crps_gauss_proxy": float(metrics["crps_gauss_proxy"]),
        f"{prefix}nlpd_gauss_proxy": float(metrics["nlpd_gauss_proxy"]),
        f"{prefix}picp_gauss_proxy": float(metrics["picp_gauss_proxy"]),
    }


def _metric_json_fields(metrics: dict) -> dict:
    return {
        "mse": float(metrics["mse"]),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "crps": float(metrics["crps"]),
        "nlpd": float(metrics["nlpd"]),
        "picp": float(metrics["picp"]),
        "crps_proper": float(metrics["crps_proper"]),
        "nlpd_proper": float(metrics["nlpd_proper"]),
        "picp_proper": float(metrics["picp_proper"]),
        "crps_gauss_proxy": float(metrics["crps_gauss_proxy"]),
        "nlpd_gauss_proxy": float(metrics["nlpd_gauss_proxy"]),
        "picp_gauss_proxy": float(metrics["picp_gauss_proxy"]),
        "n": int(metrics["n"]),
    }


def _metric_console_summary(metrics: dict) -> str:
    return (
        f"RMSE {metrics['rmse']:.4f} | MAE {metrics['mae']:.4f} | "
        f"NLPD_p {metrics['nlpd_proper']:.4f} | CRPS_p {metrics['crps_proper']:.4f} | "
        f"PICP_p {metrics['picp_proper']:.3f} | "
        f"NLPD_g {metrics['nlpd_gauss_proxy']:.4f} | "
        f"CRPS_g {metrics['crps_gauss_proxy']:.4f} | "
        f"PICP_g {metrics['picp_gauss_proxy']:.3f}"
    )


def run_mean_warmup(
    config: ConfigDict,
    model: ProbabilisticRegimeForecaster,
    likelihood,
    train_loader,
    optimizer,
    mll_pred,
    gp_label_len: int,
    num_train_points_pred: int,
    warmup_epochs: int = 50,
):
    device = config.device
    print(f"Starting Mean Warmup for {warmup_epochs} epochs...")

    # 1. Freeze Kernel & Likelihood (Force Mean to learn trend)
    model.gp.covar_module.requires_grad_(False)
    likelihood.requires_grad_(False)

    for w_epoch in range(1, warmup_epochs + 1):
        model.train()
        likelihood.train()
        w_loss_sum = 0.0

        # Using the same loader as main training
        for i, batch in enumerate(train_loader):
            # --- Data Loading ---
            if len(batch) == 8:
                seq_x, seq_y, seq_x_mark, seq_y_mark, _time_idx, _mh_y, _mh_m, _ = batch
            else:
                continue

            # Move to device
            seq_x = seq_x.to(device)
            seq_x_mark = seq_x_mark.to(device)
            seq_y_mark = seq_y_mark.to(device)
            _mh_m = _mh_m.to(device) if _mh_m is not None else None
            _mh_y = _mh_y.to(device) if _mh_y is not None else None

            optimizer.zero_grad()

            # --- Forward Pass ---
            (
                feats_fit,
                feats_pred,
                _,
                aux_warm,
                _,
                _,
                batch_mean,
                batch_std,
                deep_mean_fit,
                deep_mean_pred,
            ) = model.forward_features(
                seq_x, seq_x_mark, seq_y_mark, _mh_m, gp_label_len, config.pred_len
            )

            # --- Target Prep (RevIN) ---
            # We must normalize targets to match the output of the mean
            y_pred_all = _build_pred_targets(seq_y.to(device), _mh_y, config.pred_len)
            y_target_norm = _normalize_targets_with_current_revin(
                model, y_pred_all, batch_mean
            )

            # --- Deep Mean Injection ---
            feats_all = torch.cat([feats_fit, feats_pred], dim=1)
            deep_mean_all = None
            if _uses_latent_mean_shift(model):
                deep_mean_all = torch.cat([deep_mean_fit, deep_mean_pred], dim=1)

            # --- Loss Calculation ---
            # Since kernel variance is frozen, maximizing ELBO == minimizing MSE
            with gpytorch.settings.cholesky_jitter(1e-6):
                mvn_all = model.gp_mvn(feats_all)

                # Shift GP mean by Deep Mean
                if deep_mean_all is not None:
                    mvn_all = gpytorch.distributions.MultivariateNormal(
                        mvn_all.mean + deep_mean_all, mvn_all.covariance_matrix
                    )

                feats_pred_only = feats_all[..., -config.pred_len :, :]

                # We use the same ELBO loss, but gradients only flow to the Mean
                elbo_scale = 1.0 / max(1, num_train_points_pred)
                raw_elbo = -mll_pred(
                    mvn_all[..., -config.pred_len :],
                    y_target_norm,
                    feats_all=feats_pred_only,
                    kernel=model.gp.covar_module,
                    **_prediction_likelihood_overrides(aux_warm),
                )

                if raw_elbo.dim() > 0:
                    raw_elbo = raw_elbo.sum()
                loss = raw_elbo * elbo_scale

                # Gradient Step
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps

                loss.backward()

                if (i + 1) % config.gradient_accumulation_steps == 0 or (i + 1) == len(
                    train_loader
                ):
                    optimizer.step()
                    optimizer.zero_grad()

                w_loss_sum += loss.item() * config.gradient_accumulation_steps

        # Simple logging
        if w_epoch % 10 == 0 or w_epoch == 1:
            print(
                f"   Warmup Epoch {w_epoch}/{warmup_epochs} | Loss (MSE-proxy): {w_loss_sum/max(1, len(train_loader)):.4f}"
            )

    # 2. Unfreeze Everything
    model.gp.covar_module.requires_grad_(True)
    likelihood.requires_grad_(True)
    print("🔥 Warmup Complete. Kernels unfrozen. Starting main training loop...\n")


def train(
    config: ConfigDict,
    model: ProbabilisticRegimeForecaster,
    likelihood,
    train_loader,
    valid_loader,
    gp_label_len: int,
    target_scaler=None,
    run_dir: Path = Path("."),
) -> Dict[str, Any]:
    device = config.device
    # # kernel_params = model.gp.covar_module.parameters()
    # # other_params = [p for n, p in model.named_parameters() if "covar_module" not in n] + list(likelihood.parameters())
    # # opt = torch.optim.Adam([{"params": other_params}, {"params": kernel_params, "lr": config.lr_kernel}], lr=config.lr)
    # # 1. Get all parameters from the kernel module and store their unique memory IDs.
    # #    This is your single source of truth for what a "kernel parameter" is.
    # kernel_params = list(model.gp.covar_module.parameters())
    # kernel_param_ids = {id(p) for p in kernel_params}

    # # 2. Get all other parameters from the entire model (`model.parameters()`).
    # #    Crucially, this loop will also see the parameters inside `model.likelihood.kernel`.
    # #    However, the `if` condition will filter them out because their IDs have already
    # #    been added to `kernel_param_ids`.
    # other_params = []
    # for param in model.parameters():
    #     if id(param) not in kernel_param_ids:
    #         other_params.append(param)

    # # 3. Now, the `other_params` list contains everything BUT the kernel parameters.
    # #    The two lists are guaranteed to be distinct, so it's safe to create the optimizer.
    # opt = torch.optim.Adam([
    #     {"params": other_params},
    #     {"params": kernel_params, "lr": config.lr_kernel}
    # ], lr=config.lr)
    # Build groups
    kernel_params = list(model.gp.covar_module.parameters())
    kernel_param_ids = {id(p) for p in kernel_params}
    lr_inducing = getattr(config, "lr_inducing", None)
    inducing_params = []
    if lr_inducing is not None:
        variational_strategy = getattr(model.gp, "variational_strategy", None)
        inducing_points = getattr(variational_strategy, "inducing_points", None)
        if isinstance(inducing_points, torch.nn.Parameter) and inducing_points.requires_grad:
            inducing_params = [inducing_points]
        elif isinstance(inducing_points, torch.Tensor) and inducing_points.requires_grad:
            inducing_params = [inducing_points]
    inducing_param_ids = {id(p) for p in inducing_params}
    model_non_kernel = [
        p
        for p in model.parameters()
        if id(p) not in kernel_param_ids and id(p) not in inducing_param_ids
    ]
    likelihood_trainable = [
        p
        for p in likelihood.parameters()
        if p.requires_grad and id(p) not in kernel_param_ids
    ]
    param_groups = [
        {"params": model_non_kernel, "lr": config.lr},
        {"params": kernel_params + likelihood_trainable, "lr": config.lr_kernel},
    ]
    if inducing_params:
        print(f"INFO: Using lr_inducing={float(lr_inducing):.3g} for GP inducing locations.")
        param_groups.append({"params": inducing_params, "lr": float(lr_inducing)})
    opt = torch.optim.Adam(param_groups, lr=config.lr)

    best_val, patience = float("inf"), 0
    best_temperature = None
    best_epoch = None
    best_val_metrics: Dict[str, float] = {}

    num_train_points_pred = len(train_loader.dataset) * config.pred_len * model.D
    mll_pred = gpytorch.mlls.VariationalELBO(
        likelihood, model.gp, num_data=max(1, num_train_points_pred)
    )
    metric_options = _student_t_metric_options(config, likelihood)

    history_rows = []

    # --- GRADIENT ACCUMULATION SETUP ---
    accumulation_steps = config.gradient_accumulation_steps

    # --- RESUME FROM CHECKPOINT ---
    start_epoch = 1
    resume_ckpt_path = run_dir / "resume.ckpt"
    if resume_ckpt_path.exists():
        print(f"Found resume checkpoint at {resume_ckpt_path}, loading...")
        rckpt = torch.load(resume_ckpt_path, map_location=config.device)
        try:
            model.load_state_dict(rckpt["model_state"])
            likelihood.load_state_dict(rckpt["likelihood_state"])
            opt.load_state_dict(rckpt["optimizer_state"])
            start_epoch = rckpt["epoch"] + 1
            best_val = rckpt.get("best_val", float("inf"))
            best_epoch = rckpt.get("best_epoch", None)
            best_temperature = rckpt.get("best_temperature", None)
            best_val_metrics = rckpt.get("best_val_metrics", {})
            patience = rckpt.get("patience", 0)
            history_rows = rckpt.get("history_rows", [])
            print(f"Resumed from epoch {start_epoch - 1} (best_val={best_val:.4f}, patience={patience})")
        except Exception as e:
            print(f"Warning: Failed to resume from checkpoint - {e}. Starting fresh.")
            start_epoch = 1

    # 1. RUN WARMUP (skip if resuming past warmup)
    # ==========================================
    if start_epoch == 1 and config.get("mean_warmup_epochs", 0) > 0:
        run_mean_warmup(
            config=config,
            model=model,
            likelihood=likelihood,
            train_loader=train_loader,
            optimizer=opt,
            mll_pred=mll_pred,
            gp_label_len=gp_label_len,
            num_train_points_pred=num_train_points_pred,
            warmup_epochs=config["mean_warmup_epochs"],
        )

    checkpoint_every = config.get("checkpoint_every", 25)

    for epoch in range(start_epoch, config.training_iterations + 1):
        # --- Annealing Logic ---
        sb_lambda_nominal = (
            ramp(
                epoch,
                config.sb_lambda_init,
                config.sb_lambda_final,
                config.anneal_steps,
            )
            if config.gating_method == "stick_breaking"
            else 0.0
        )
        # KL weight for variational stick-breaking. Default 1.0 → true ELBO;
        # ramping is exposed to allow β-VAE-style warmups if desired.
        vsb_kl_weight = (
            ramp(
                epoch,
                float(config.get("vsb_kl_weight_init", 1.0)),
                float(config.get("vsb_kl_weight_final", 1.0)),
                config.anneal_steps,
            )
            if config.gating_method == "vsb"
            else 0.0
        )
        w_point = ramp(
            epoch,
            config.point_entropy_weight_init,
            config.point_entropy_weight_final,
            config.anneal_steps,
        )
        w_batch = ramp(
            epoch,
            config.batch_entropy_weight_init,
            config.batch_entropy_weight_final,
            config.anneal_steps,
        )
        decay = (config.anneal_end_temp / config.anneal_start_temp) ** (
            1 / max(config.anneal_epochs, 1)
        )
        temp = max(config.anneal_start_temp * (decay**epoch), config.anneal_end_temp)
        T_ref = 1.0
        sb_lambda = sb_lambda_nominal * (temp / T_ref)
        model.set_temperature(temp)

        # --- NEW: Alpha Annealing ---
        # Define start/end in config (or hardcode for testing)
        a_start = config["sb_alpha_init"]
        a_end = config.get("sb_alpha_final", config["sb_alpha_init"])
        model.set_alpha(epoch, config.anneal_epochs, a_start, a_end)
        # --- End Annealing Logic ---

        # --- Training Loop ---
        model.train()
        likelihood.train()
        elbo_sum = ent_sum = bent_sum = beta_sum = loss_sum = dp_reg_sum = 0.0
        resid_pen_sum = resid_var_sum = 0.0
        nbatches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.training_iterations}",
            disable=not is_interactive,
        )

        opt.zero_grad()  # Zero gradients at the start of the epoch

        warning_file_path = run_dir / "CHOLESKY_WARNINGS.txt"

        for i, batch in enumerate(pbar):
            # --- Data Loading ---
            if len(batch) == 8:
                seq_x, seq_y, seq_x_mark, seq_y_mark, _time_idx, _mh_y, _mh_m, _ = batch
            else:
                raise NotImplementedError("Multi-horizon inputs required for training.")
            seq_x, seq_y, seq_x_mark, seq_y_mark = (
                t.to(device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
            )
            _mh_y = (
                _mh_y.to(device) if _mh_y is not None and hasattr(_mh_y, "to") else None
            )
            _mh_m = (
                _mh_m.to(device) if _mh_m is not None and hasattr(_mh_m, "to") else None
            )
            # --- End Data Loading ---

            try:
                # --- Forward Pass ---
                (
                    feats_fit,
                    feats_pred,
                    g_pred,
                    aux,
                    g_pred_logits,
                    g_fit,
                    batch_mean,
                    batch_std,
                    deep_mean_fit,
                    deep_mean_pred,
                ) = model.forward_features(
                    seq_x,
                    seq_x_mark,
                    seq_y_mark,
                    _mh_m,
                    gp_label_len,
                    config.pred_len,
                )
                likelihood_overrides = _prediction_likelihood_overrides(aux)
                residual_obs_var_pred = likelihood_overrides["residual_obs_var"]
                # --- End Forward Pass ---

                # --- Loss Calculation ---
                y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)

                y_target_norm = _normalize_targets_with_current_revin(
                    model, y_pred_all, batch_mean
                )

                feats_all = torch.cat([feats_fit, feats_pred], dim=1)
                # --- NEW: Deep Mean Logic ---
                # Check string attribute 'dm_mode' instead of boolean
                if _uses_latent_mean_shift(model):
                    deep_mean_all = torch.cat([deep_mean_fit, deep_mean_pred], dim=1)
                else:
                    deep_mean_all = None
                # ----------------------------

                jitter_value = 1e-6
                with gpytorch.settings.cholesky_jitter(jitter_value):
                    # Get the base GP distribution
                    mvn_all = model.gp_mvn(feats_all)

                    # --- NEW: Shift Mean if Enabled ---
                    if deep_mean_all is not None:
                        mvn_all = gpytorch.distributions.MultivariateNormal(
                            mvn_all.mean + deep_mean_all, mvn_all.covariance_matrix
                        )
                    # ----------------------------------
                    elbo_scale = 1.0 / max(
                        1, num_train_points_pred
                    )  # Avoid division by zero
                    feats_pred_only = feats_all[..., -config.pred_len :, :]
                    raw_elbo = -mll_pred(
                        mvn_all[..., -config.pred_len :],
                        y_target_norm,  # <--- CHANGED from y_pred_all
                        feats_all=feats_pred_only,
                        kernel=model.gp.covar_module,
                        **likelihood_overrides,
                    )
                    # --- FIX: Force Scalar Reduction ---
                    # The Student-T likelihood returns [B*D, T], preventing implicit gradients.
                    # We sum it up to get the total ELBO for the batch.
                    if raw_elbo.dim() > 0:
                        raw_elbo = raw_elbo.sum()
                    # -----------------------------------

                    loss_elbo = raw_elbo * elbo_scale
                    loss_resid = torch.tensor(0.0, device=device)
                    resid_mean = torch.tensor(float("nan"), device=device)
                    if residual_obs_var_pred is not None:
                        resid_mean = residual_obs_var_pred.mean()
                        penalty = float(
                            config.get(
                                "residual_observation_variance_penalty", 0.0
                            )
                            or 0.0
                        )
                        if penalty > 0.0:
                            loss_resid = penalty * resid_mean

                    loss_ent = torch.tensor(0.0, device=device)
                    loss_bent = torch.tensor(0.0, device=device)
                    if g_pred is not None:
                        acts = g_pred
                        point_entropy = -torch.sum(
                            acts * torch.log(acts.clamp(min=1e-9)), dim=-1
                        )
                        loss_ent = w_point * point_entropy.mean()

                        avg_acts = acts.mean(dim=(0, 1))
                        batch_entropy = -torch.sum(
                            avg_acts * torch.log(avg_acts.clamp(min=1e-9))
                        )
                        loss_bent = w_batch * (-batch_entropy)

                    loss_beta = torch.tensor(0.0, device=device)
                    if (
                        config.gating_method == "stick_breaking"
                        and sb_lambda > 0.0
                        and g_pred is not None
                    ):
                        alpha = model.alpha_value()
                        if model.sb_mode == "renorm":
                            R = g_pred.size(-1)
                            sum_log_pi = torch.log(g_pred.clamp_min(1e-8)).sum(dim=-1)
                            nll_dir = -(alpha - 1.0) * sum_log_pi + (
                                R * torch.lgamma(alpha) - torch.lgamma(R * alpha)
                            )
                            loss_beta = sb_lambda * nll_dir.mean()
                        elif (
                            model.sb_mode == "residual"
                            and aux is not None
                            and "log_one_minus_sig" in aux
                        ):
                            log1mv = aux["log_one_minus_sig"]
                            penalty_elems = -torch.log(alpha) - (alpha - 1.0) * log1mv
                            loss_beta = sb_lambda * penalty_elems.sum(dim=-1).mean()
                    elif (
                        config.gating_method == "vsb"
                        and aux is not None
                        and aux.get("kl_elem") is not None
                    ):
                        # Closed-form KL[Kuma(a,b) || Beta(1, alpha)] per stick.
                        # Sum across sticks (R-1), mean across batch+time, and
                        # apply the (typically constant) ELBO weight.
                        kl_per_sample = aux["kl_elem"].sum(dim=-1)
                        loss_beta = vsb_kl_weight * kl_per_sample.mean()

                    loss_dp = torch.tensor(0.0, device=device)
                    if (
                        config.use_dp_reg
                        and config.gating_method == "stick_breaking"
                        and sb_lambda > 0.0
                        and aux is not None
                    ):
                        loss_dp = model.dp_regularizer(aux["z"])

                loss = (
                    loss_elbo
                    + loss_ent
                    + loss_bent
                    + loss_beta
                    + loss_dp
                    + loss_resid
                )
                # --- End Loss Calculation ---

                # --- Gradient Accumulation Backward/Step ---
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

                loss.backward()
            # --- CATCH THE SPECIFIC ERROR ---
            except torch._C._LinAlgError as e:
                warning_msg = f"Epoch {epoch}, Batch {i}: Cholesky failed! Skipping batch. Error: {e}"
                print(f"\n⚠️ WARNING: {warning_msg}")
                # Log to warning file
                try:
                    with open(warning_file_path, "a") as f:
                        f.write(warning_msg + "\n")
                except Exception as log_e:
                    print(f"  (Failed to write to warning file: {log_e})")

                # Crucially, skip the optimizer step for this accumulation cycle
                # If we are accumulating, we need to make sure gradients are cleared
                # if the error happens mid-accumulation. The safest is often to
                # clear gradients here and let the next successful batch start fresh.
                opt.zero_grad()  # Clear any potentially corrupted gradients from this failed step

                # Skip the rest of this loop iteration (including opt.step)
                continue

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                max_grad_norm = float(config.get("max_grad_norm", 0.0) or 0.0)
                if max_grad_norm > 0.0:
                    params_with_grad = [
                        p
                        for group in opt.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params_with_grad,
                        max_grad_norm,
                        error_if_nonfinite=False,
                    )
                    if not torch.isfinite(grad_norm):
                        print(
                            f"Warning: non-finite gradient norm at epoch {epoch}, "
                            f"batch {i}; skipping optimizer step.",
                            flush=True,
                        )
                        opt.zero_grad()
                        continue
                opt.step()
                opt.zero_grad()
            # --- End Accumulation ---

            # --- Logging (inside batch loop) ---
            loss_value_for_logging = float(loss.item()) * accumulation_steps  # Rescale
            elbo_sum += float(loss_elbo.item())
            ent_sum += float(loss_ent.item())
            bent_sum += float(loss_bent.item())
            beta_sum += float(loss_beta.item())
            loss_sum += loss_value_for_logging
            dp_reg_sum += float(loss_dp.item())
            resid_pen_sum += float(loss_resid.item())
            if residual_obs_var_pred is not None:
                resid_var_sum += float(resid_mean.detach().cpu().item())
            nbatches += 1

            nz = current_noise(likelihood)
            pbar.set_postfix(
                {
                    "Loss": f"{loss_value_for_logging:.2e}",
                    "ELBO": f"{loss_elbo.item():.2e}",
                    "Ent": f"{loss_ent.item():.2e}",
                    "BatchEnt": f"{loss_bent.item():.2e}",
                    "SB": f"{loss_beta.item():.2e}",
                    "Noise": f"{nz:.4f}",
                    "DP": f"{loss_dp.item():.4f}",
                    "Vres": (
                        f"{resid_mean.item():.2e}"
                        if residual_obs_var_pred is not None
                        else "NA"
                    ),
                }
            )
            # --- End Logging ---
        # --- End Training Batch Loop ---

        # --- Aggregate Training Metrics ---
        train_elbo = elbo_sum / nbatches if nbatches else float("nan")
        train_ent = ent_sum / nbatches if nbatches else float("nan")
        train_bent = bent_sum / nbatches if nbatches else float("nan")
        train_beta = beta_sum / nbatches if nbatches else float("nan")
        train_dp = dp_reg_sum / nbatches if nbatches else float("nan")
        train_resid_pen = resid_pen_sum / nbatches if nbatches else float("nan")
        train_resid_var = resid_var_sum / nbatches if nbatches else float("nan")
        train_loss = loss_sum / nbatches if nbatches else float("nan")
        # --- End Aggregation ---

        # --- Validation Loop (with standardized metrics logic) ---
        model.eval()
        likelihood.eval()
        val_loss = 0.0
        val_metric_arrays = {"y": [], "mu": []}
        reff_vals = []
        with torch.no_grad():
            for batch in valid_loader:
                # ... (validation data loading) ...
                if len(batch) == 8:
                    seq_x, seq_y, seq_x_mark, seq_y_mark, _time_idx, _mh_y, _mh_m, _ = (
                        batch
                    )
                else:
                    raise NotImplementedError(
                        "Multi-horizon inputs required for validation."
                    )
                seq_x, seq_y, seq_x_mark, seq_y_mark = (
                    t.to(device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
                )
                _mh_y = (
                    _mh_y.to(device)
                    if _mh_y is not None and hasattr(_mh_y, "to")
                    else None
                )
                _mh_m = (
                    _mh_m.to(device)
                    if _mh_m is not None and hasattr(_mh_m, "to")
                    else None
                )
                try:
                    (
                        feats_fit,
                        feats_pred,
                        g_pred,
                        aux_val,
                        _,
                        _,
                        batch_mean,
                        batch_std,
                        deep_mean_fit,
                        deep_mean_pred,
                    ) = model.forward_features(
                        seq_x,
                        seq_x_mark,
                        seq_y_mark,
                        _mh_m,
                        gp_label_len,
                        config.pred_len,
                    )
                    likelihood_overrides = _prediction_likelihood_overrides(aux_val)
                    residual_obs_var_pred = likelihood_overrides["residual_obs_var"]

                    y_pred_all = _build_pred_targets(
                        seq_y, _mh_y, config.pred_len
                    )  # (B*D, H)

                    y_target_norm = _normalize_targets_with_current_revin(
                        model, y_pred_all, batch_mean
                    )
                    feats_all = torch.cat([feats_fit, feats_pred], dim=1)

                    # --- NEW: Deep Mean Logic ---
                    # Check string attribute 'dm_mode' instead of boolean
                    if _uses_latent_mean_shift(model):
                        deep_mean_all = torch.cat(
                            [deep_mean_fit, deep_mean_pred], dim=1
                        )
                    else:
                        deep_mean_all = None
                    # ----------------------------

                    jitter_value = 1e-6
                    with gpytorch.settings.cholesky_jitter(jitter_value):
                        # Get the base GP distribution
                        mvn_all = model.gp_mvn(feats_all)

                        # --- NEW: Shift Mean if Enabled ---
                        if deep_mean_all is not None:
                            mvn_all = gpytorch.distributions.MultivariateNormal(
                                mvn_all.mean + deep_mean_all, mvn_all.covariance_matrix
                            )
                        # ----------------------------------
                        feats_pred_only = feats_all[..., -config.pred_len :, :]
                        raw_val_elbo = -mll_pred(
                            mvn_all[..., -config.pred_len :],
                            y_target_norm,
                            feats_all=feats_pred_only,
                            kernel=model.gp.covar_module,
                            **likelihood_overrides,
                        )
                        # --- FIX: Force Scalar Reduction ---
                        if raw_val_elbo.dim() > 0:
                            raw_val_elbo = raw_val_elbo.sum()
                        # -----------------------------------

                        # 2. Scale and Store
                        val_loss_elbo = raw_val_elbo * elbo_scale
                        val_loss += val_loss_elbo.item()

                        latent_pred = mvn_all[..., -config.pred_len :]
                        yy = y_pred_all.detach().cpu().numpy().reshape(-1)

                        if _uses_distributional_gp_likelihood(config, likelihood):
                            pred_stats = _gp_student_t_predictive_tensors(
                                model,
                                likelihood,
                                latent_pred,
                                feats_pred_only,
                                batch_std,
                                kernel=model.gp.covar_module,
                                **likelihood_overrides,
                            )
                            _append_metric_arrays(
                                val_metric_arrays,
                                {
                                    "y": yy,
                                    "mu": pred_stats["mu"]
                                    .transpose(1, 2)
                                    .reshape(-1)
                                    .detach()
                                    .cpu()
                                    .numpy(),
                                    "latent_std": pred_stats["latent_std"]
                                    .transpose(1, 2)
                                    .reshape(-1)
                                    .detach()
                                    .cpu()
                                    .numpy(),
                                    "obs_scale": pred_stats["obs_scale"]
                                    .transpose(1, 2)
                                    .reshape(-1)
                                    .detach()
                                    .cpu()
                                    .numpy(),
                                    **(
                                        {
                                            "mixture_weights": pred_stats[
                                                "mixture_weights"
                                            ]
                                            .transpose(1, 2)
                                            .reshape(-1, pred_stats["mixture_weights"].size(-1))
                                            .detach()
                                            .cpu()
                                            .numpy(),
                                            "obs_scale_components": pred_stats[
                                                "obs_scale_components"
                                            ]
                                            .transpose(1, 2)
                                            .reshape(
                                                -1,
                                                pred_stats[
                                                    "obs_scale_components"
                                                ].size(-1),
                                            )
                                            .detach()
                                            .cpu()
                                            .numpy(),
                                        }
                                        if "mixture_weights" in pred_stats
                                        else {}
                                    ),
                                    **(
                                        {
                                            "df": pred_stats["df"]
                                            .transpose(1, 2)
                                            .reshape(-1)
                                            .detach()
                                            .cpu()
                                            .numpy(),
                                        }
                                        if "df" in pred_stats
                                        else {}
                                    ),
                                    **(
                                        {
                                            "df_components": pred_stats[
                                                "df_components"
                                            ]
                                            .transpose(1, 2)
                                            .reshape(
                                                -1,
                                                pred_stats["df_components"].size(-1),
                                            )
                                            .detach()
                                            .cpu()
                                            .numpy(),
                                        }
                                        if "df_components" in pred_stats
                                        else {}
                                    ),
                                    **(
                                        {
                                            "residual_obs_var": pred_stats[
                                                "residual_obs_var"
                                            ]
                                            .transpose(1, 2)
                                            .reshape(-1)
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        }
                                        if pred_stats["residual_obs_var"] is not None
                                        else {}
                                    ),
                                },
                            )
                        else:
                            pred_mvn = likelihood.marginal(
                                mvn_all[..., -config.pred_len :],
                                feats_all=feats_pred_only,
                                kernel=model.gp.covar_module,
                                **likelihood_overrides,
                            )
                            pm_norm = pred_mvn.mean  # (B*D, H)
                            ps_norm = pred_mvn.stddev  # (B*D, H)

                            # Reshape to (B, H, D) for the RevIN method
                            B_curr = batch_mean.shape[0]
                            pm_3d = pm_norm.reshape(B_curr, model.D, -1).transpose(
                                1, 2
                            )  # (B, H, D)
                            ps_3d = ps_norm.reshape(B_curr, model.D, -1).transpose(
                                1, 2
                            )

                            # Apply Denormalize (handles Affine - bias / weight * std + mean)
                            pm_denorm = model.revin._denormalize(pm_3d)

                            # Apply Sigma Denorm (sigma / weight * std)
                            ps_denorm = ps_3d * _revin_output_scale(
                                model, batch_std
                            )

                            _append_metric_arrays(
                                val_metric_arrays,
                                {
                                    "y": yy,
                                    "mu": pm_denorm.transpose(1, 2)
                                    .reshape(-1)
                                    .detach()
                                    .cpu()
                                    .numpy(),
                                    "sd": ps_denorm.transpose(1, 2)
                                    .reshape(-1)
                                    .detach()
                                    .cpu()
                                    .numpy(),
                                },
                            )

                    if g_pred is not None:
                        gbar = g_pred.mean(dim=(0, 1)).detach().cpu().numpy()
                        reff = int((gbar > config.remainder_epsilon).sum())
                        reff_vals.append(reff)
                # --- CATCH THE SPECIFIC ERROR ---
                except torch._C._LinAlgError as e:
                    warning_msg = f"Epoch {epoch}, Validation Batch: Cholesky failed! Skipping batch. Error: {e}"
                    print(f"\n⚠️ WARNING: {warning_msg}")
                    # Log to warning file
                    try:
                        with open(warning_file_path, "a") as f:
                            f.write(warning_msg + "\n")
                    except Exception as log_e:
                        print(f"  (Failed to write to warning file: {log_e})")

                    # Skip accumulating results for this batch
                    continue
        # --- END CATCH BLOCK ---
        avg_val_loss = val_loss / max(1, len(valid_loader))  # This is ELBO based
        if val_metric_arrays["y"]:
            y_all = np.concatenate(val_metric_arrays["y"])
            mu_all = np.concatenate(val_metric_arrays["mu"])
            if "mixture_weights" in val_metric_arrays:
                val_metrics = compute_regime_student_t_mixture_metrics(
                    y_all,
                    mu_all,
                    np.concatenate(val_metric_arrays["latent_std"]),
                    np.concatenate(val_metric_arrays["mixture_weights"]),
                    np.concatenate(val_metric_arrays["obs_scale_components"]),
                    (
                        np.concatenate(val_metric_arrays["df_components"])
                        if "df_components" in val_metric_arrays
                        else None
                    ),
                    interval_level=metric_options["interval_level"],
                    sample_size=metric_options["sample_size"],
                    chunk_size=metric_options["chunk_size"],
                    gh_points=metric_options["gh_points"],
                    seed=metric_options["seed"],
                )
            elif "latent_std" in val_metric_arrays:
                val_metrics = compute_student_t_mixture_metrics(
                    y_all,
                    mu_all,
                    np.concatenate(val_metric_arrays["latent_std"]),
                    np.concatenate(val_metric_arrays["obs_scale"]),
                    np.concatenate(val_metric_arrays["df"]),
                    interval_level=metric_options["interval_level"],
                    sample_size=metric_options["sample_size"],
                    chunk_size=metric_options["chunk_size"],
                    gh_points=metric_options["gh_points"],
                    seed=metric_options["seed"],
                )
            else:
                std_all = np.concatenate(val_metric_arrays["sd"])
                val_metrics = compute_gaussian_metrics(
                    y_all,
                    mu_all,
                    std_all,
                    interval_level=metric_options["interval_level"],
                )
            val_objective = float(val_metrics["nlpd_proper"])
        else:
            val_metrics = _empty_metric_result()
            val_objective = float("inf")  # Assign inf if validation fails

        alpha_val = (
            float(model.alpha_value().detach().cpu().item())
            if config.gating_method in ("stick_breaking", "vsb")
            and hasattr(model, "alpha_value")
            and model.alpha_value() is not None
            else None
        )
        reff_mean = float(np.mean(reff_vals)) if reff_vals else np.nan
        noise_val = current_noise(likelihood)
        # --- End Validation ---

        # --- >>> RESTORED THIS BLOCK <<< ---
        # --- Logging, Early Stopping & Checkpointing ---
        print(
            f"Epoch {epoch}: Train {train_loss:.2e} (ELBO {train_elbo:.2e} | Ent {train_ent:.2e} | "
            f"BatchEnt {train_bent:.2e} | SB {train_beta:.2e}) | DP {train_dp:.2e} | "
            f"Vres {train_resid_var:.2e} | "
            f"Val ELBO {avg_val_loss:.2e} | {_metric_console_summary(val_metrics)} | "
            f"Noise {noise_val:.4f} | R_eff {reff_mean:.2f} | α {alpha_val if alpha_val is not None else 'NA'} | "
            f"w_point {w_point:.4g} | w_batch {w_batch:.4g} | λ_SB {sb_lambda:.4g}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_elbo": train_elbo,
            "train_ent": train_ent,
            "train_batchent": train_bent,
            "train_sb": train_beta,
            "train_dp": train_dp,
            "train_resid_obs_var": train_resid_var,
            "train_resid_pen": train_resid_pen,
            "val_loss": float(
                val_objective
            ),  # Using NLPD as the primary validation metric now
            "val_elbo": float(avg_val_loss),  # Still log ELBO
            "val_mse": float(val_metrics["mse"]),  # Standardized
            "val_rmse": float(val_metrics["rmse"]),  # Standardized
            "val_mae": float(val_metrics["mae"]),  # Standardized
            "val_crps": float(val_metrics["crps"]),  # Standardized
            "val_nlpd": float(val_metrics["nlpd"]),  # Standardized
            "val_picp": float(val_metrics["picp"]),
            "R_eff": reff_mean,
            "alpha": alpha_val,
            "w_point": w_point,
            "w_batch": w_batch,
            "lambda_SB": sb_lambda,
            "noise": noise_val,
        }
        row.update(_metric_history_fields(val_metrics))
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

        # Early stopping + checkpoint on val objective (NLPD)
        if val_objective < best_val:
            best_val = val_objective  # Use NLPD
            patience = 0
            best_epoch = epoch
            best_temperature = float(model.current_temp.detach().cpu().item())
            best_val_metrics = {
                "epoch": int(epoch),
                "val_loss": float(val_objective),  # NLPD
                "val_elbo": float(avg_val_loss),
                **_metric_json_fields(val_metrics),
                "R_eff": reff_mean,
                "alpha": alpha_val,
                "w_point": w_point,
                "w_batch": w_batch,
                "lambda_SB": sb_lambda,
                "noise": noise_val,
            }
            ckpt = {
                "model_state": model.state_dict(),
                "likelihood_state": likelihood.state_dict(),
                "best_temperature": best_temperature,
                "epoch": epoch,
                "val_loss": float(val_objective),  # NLPD
                "config": dict(config),
            }
            torch.save(ckpt, run_dir / "best.ckpt")
            with open(run_dir / "best_val_metrics.json", "w") as f:
                json.dump(best_val_metrics, f, indent=2)
        else:
            if epoch < config.min_epochs:
                patience = 0  # don't early stop yet
            patience += 1
            if patience >= config.patience:
                print("Early stopping.")
                break

        # --- Periodic resume checkpoint ---
        if epoch % checkpoint_every == 0:
            torch.save({
                "model_state": model.state_dict(),
                "likelihood_state": likelihood.state_dict(),
                "optimizer_state": opt.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "best_temperature": best_temperature,
                "best_val_metrics": best_val_metrics,
                "patience": patience,
                "history_rows": history_rows,
                "config": dict(config),
            }, run_dir / "resume.ckpt")

    # Clean up resume checkpoint on successful completion
    resume_path = run_dir / "resume.ckpt"
    if resume_path.exists():
        resume_path.unlink()

    # --- Load Best Checkpoint ---
    best_ckpt = run_dir / "best.ckpt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=config.device)
        try:
            model.load_state_dict(ckpt["model_state"])
            likelihood.load_state_dict(ckpt["likelihood_state"])
            if "best_temperature" in ckpt and ckpt["best_temperature"] is not None:
                model.set_temperature(float(ckpt["best_temperature"]))
            print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', 'N/A')}.")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint states - {e}")

    return {
        "best_temperature": best_temperature,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val) if best_val < float("inf") else None,
        "best_val_metrics": best_val_metrics,
        "history": history_rows,
    }


@torch.no_grad()
def evaluate_test(
    config: ConfigDict,
    model,
    test_loader,
    target_scaler,
    run_dir: Path,
    likelihood=None,
):
    """
    Collects per-horizon predictions in both SCALED and RAW space,
    then calls artifact finalizer.
    """
    device = config.device
    model.eval()
    if likelihood is not None:
        likelihood.eval()

    # ✨ Updated collector to hold both raw and scaled predictions ✨
    collect = {
        h: {
            "t": [],
            "acts": [],
            "raw": {"y": [], "mu": [], "sd": []},
            "scaled": {"y": [], "mu": [], "sd": []},
        }
        for h in range(1, config.pred_len + 1)
    }

    for batch in tqdm(test_loader, desc="Testing"):
        (
            seq_x,
            seq_y,
            seq_x_mark,
            seq_y_mark,
            seq_y_time_idx,
            _mh_y,
            _mh_m,
            seq_mh_time_idx,
        ) = batch

        seq_x, seq_y, seq_x_mark, seq_y_mark = (
            t.to(device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
        )
        _mh_y = _mh_y.to(device) if _mh_y is not None and hasattr(_mh_y, "to") else None
        _mh_m = _mh_m.to(device) if _mh_m is not None and hasattr(_mh_m, "to") else None

        if config.model_type in ["regime_gp", "rq_gp"]:
            (
                feats_fit,
                feats_pred,
                g_pred,
                aux_eval,
                _,
                _,
                batch_mean,
                batch_std,
                deep_mean_fit,
                deep_mean_pred,
            ) = model.forward_features(
                seq_x,
                seq_x_mark,
                seq_y_mark,
                _mh_m,
                config.gp_label_len,
                config.pred_len,
            )
            likelihood_overrides = _prediction_likelihood_overrides(aux_eval)
            y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)
            feats_all = torch.cat([feats_fit, feats_pred], dim=1)

            # --- NEW: Deep Mean Logic ---
            if _uses_latent_mean_shift(model):
                deep_mean_all = torch.cat([deep_mean_fit, deep_mean_pred], dim=1)
            else:
                deep_mean_all = None
            # ----------------------------

            mvn_all = model.gp_mvn(feats_all)

            # --- NEW: Shift Mean if Enabled ---
            if deep_mean_all is not None:
                mvn_all = gpytorch.distributions.MultivariateNormal(
                    mvn_all.mean + deep_mean_all, mvn_all.covariance_matrix
                )
            latent_pred = mvn_all[..., -config.pred_len :]
            yy = y_pred_all.detach().cpu().numpy()
            scaled_metric_extras_flat = {}

            if _uses_distributional_gp_likelihood(config, likelihood):
                pred_stats = _gp_student_t_predictive_tensors(
                    model,
                    likelihood,
                    latent_pred,
                    feats_all[..., -config.pred_len :, :],
                    batch_std,
                    kernel=model.gp.covar_module,
                    **likelihood_overrides,
                )
                B, H, _ = pred_stats["mu"].shape
                BD = B * model.D
                pm = (
                    pred_stats["mu"]
                    .transpose(1, 2)
                    .reshape(BD, H)
                    .detach()
                    .cpu()
                    .numpy()
                )
                ps = (
                    pred_stats["sd"]
                    .transpose(1, 2)
                    .reshape(BD, H)
                    .detach()
                    .cpu()
                    .numpy()
                )
                scaled_metric_extras_flat = {
                    "latent_std": pred_stats["latent_std"]
                    .transpose(1, 2)
                    .reshape(BD, H)
                    .detach()
                    .cpu()
                    .numpy(),
                    "obs_scale": pred_stats["obs_scale"]
                    .transpose(1, 2)
                    .reshape(BD, H)
                    .detach()
                    .cpu()
                    .numpy(),
                }
                if "df" in pred_stats:
                    scaled_metric_extras_flat["df"] = (
                        pred_stats["df"]
                        .transpose(1, 2)
                        .reshape(BD, H)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                if "mixture_weights" in pred_stats:
                    scaled_metric_extras_flat.update(
                        {
                            "mixture_weights": pred_stats["mixture_weights"]
                            .transpose(1, 2)
                            .reshape(BD, H, pred_stats["mixture_weights"].size(-1))
                            .detach()
                            .cpu()
                            .numpy(),
                            "obs_scale_components": pred_stats[
                                "obs_scale_components"
                            ]
                            .transpose(1, 2)
                            .reshape(
                                BD,
                                H,
                                pred_stats["obs_scale_components"].size(-1),
                            )
                            .detach()
                            .cpu()
                            .numpy(),
                        }
                    )
                    if "df_components" in pred_stats:
                        scaled_metric_extras_flat["df_components"] = (
                            pred_stats["df_components"]
                            .transpose(1, 2)
                            .reshape(BD, H, pred_stats["df_components"].size(-1))
                            .detach()
                            .cpu()
                            .numpy()
                        )
                if pred_stats["residual_obs_var"] is not None:
                    scaled_metric_extras_flat["residual_obs_var"] = (
                        pred_stats["residual_obs_var"]
                        .transpose(1, 2)
                        .reshape(BD, H)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            else:
                pred_mvn = likelihood.marginal(
                    latent_pred,
                    feats_all=feats_all[..., -config.pred_len :, :],
                    kernel=model.gp.covar_module,
                    **likelihood_overrides,
                )

                pm_tensor = pred_mvn.mean  # (B*D, H)
                ps_tensor = pred_mvn.stddev  # (B*D, H)

                BD, H = pm_tensor.shape
                B = BD // model.D

                pm_3d = pm_tensor.reshape(B, model.D, H).transpose(1, 2)
                ps_3d = ps_tensor.reshape(B, model.D, H).transpose(1, 2)

                pm_denorm_tensor = model.revin._denormalize(pm_3d)
                ps_denorm_tensor = ps_3d * _revin_output_scale(model, batch_std)

                pm = (
                    pm_denorm_tensor.transpose(1, 2)
                    .reshape(BD, H)
                    .detach()
                    .cpu()
                    .numpy()
                )
                ps = (
                    ps_denorm_tensor.transpose(1, 2)
                    .reshape(BD, H)
                    .detach()
                    .cpu()
                    .numpy()
                )

            acts_bhdr = None
            # Corrected code
            if g_pred is not None and not config.single_kernel_mode:
                BD, H, R = g_pred.shape
                B = BD // model.D
                acts_bhdr = (
                    g_pred.cpu()
                    .numpy()
                    .reshape(B, model.D, H, R)
                    .transpose(0, 2, 1, 3)  # Transpose to (B, H, D, R)
                )

        elif config.model_type == "single_encoder_mle":
            # Unpack 3 values
            mu, sigma, nu = model(
                seq_x,
                seq_x_mark,
                seq_y_mark,
                _mh_m,
                config.gp_label_len,
                config.pred_len,
            )

            # --- Handle Student's t Variance Scaling ---
            if nu is not None:
                # Std = scale * sqrt(nu / (nu - 2))
                var_factor = nu / (nu - 2.0)
                final_std = sigma * torch.sqrt(var_factor)
            else:
                final_std = sigma

            y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)
            B, H, D = mu.shape

            # Use final_std for ps (predicted std)
            pm = mu.cpu().numpy().transpose(0, 2, 1).reshape(B * D, H)
            ps = final_std.cpu().numpy().transpose(0, 2, 1).reshape(B * D, H)
            scaled_metric_extras_flat = {}
            if nu is not None:
                scaled_metric_extras_flat = {
                    "scale": sigma.cpu().numpy().transpose(0, 2, 1).reshape(B * D, H),
                    "df": nu.cpu().numpy().transpose(0, 2, 1).reshape(B * D, H),
                }

            yy = y_pred_all.cpu().numpy()
            acts_bhdr = None

        elif config.model_type == "single_encoder_quantile":
            quantiles_np = np.asarray(
                config.get("quantiles", [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
            )
            q_pred = model(
                seq_x, seq_x_mark, seq_y_mark, _mh_m,
                config.gp_label_len, config.pred_len,
            )  # (B, H, D, Q)
            B, H, D, Q = q_pred.shape

            q_np = q_pred.cpu().numpy()  # (B, H, D, Q)
            q_flat = q_np.transpose(0, 2, 1, 3).reshape(B * D, H, Q)
            mu_flat, sd_flat = _quantiles_to_gaussian(q_flat, quantiles_np)

            pm = mu_flat  # (B*D, H)
            ps = sd_flat  # (B*D, H)
            scaled_metric_extras_flat = {}

            y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)
            yy = y_pred_all.cpu().numpy()
            acts_bhdr = None

        elif config.model_type == "single_encoder_mdn":
            weights, mus, scales, dfs = model(
                seq_x, seq_x_mark, seq_y_mark, _mh_m,
                config.gp_label_len, config.pred_len,
            )  # each (B, H, D, K), dfs may be None
            B, H, D, K = mus.shape

            # Mixture mean and std (for point metrics + Gaussian-proxy)
            mean_mix = (weights * mus).sum(dim=-1)  # (B, H, D)
            if dfs is not None:
                comp_var = scales.pow(2) * dfs / (dfs - 2.0).clamp(min=1e-6)
            else:
                comp_var = scales.pow(2)
            var_mix = (
                weights * (comp_var + (mus - mean_mix.unsqueeze(-1)).pow(2))
            ).sum(dim=-1)
            std_mix = var_mix.clamp(min=1e-12).sqrt()

            pm = mean_mix.cpu().numpy().transpose(0, 2, 1).reshape(B * D, H)
            ps = std_mix.cpu().numpy().transpose(0, 2, 1).reshape(B * D, H)
            # Stash full mixture params for the MDN-aware metric
            scaled_metric_extras_flat = {
                "mdn_weights": weights.cpu().numpy().transpose(0, 2, 1, 3).reshape(B * D, H, K),
                "mdn_mus": mus.cpu().numpy().transpose(0, 2, 1, 3).reshape(B * D, H, K),
                "mdn_scales": scales.cpu().numpy().transpose(0, 2, 1, 3).reshape(B * D, H, K),
            }
            if dfs is not None:
                scaled_metric_extras_flat["mdn_dfs"] = (
                    dfs.cpu().numpy().transpose(0, 2, 1, 3).reshape(B * D, H, K)
                )

            y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)
            yy = y_pred_all.cpu().numpy()
            acts_bhdr = None

        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

        # time grid of shape (B*D, H)
        tt = (
            _build_pred_times_flat(
                seq_y_time_idx, seq_mh_time_idx, config.pred_len, D=model.D
            )
            .cpu()
            .numpy()
        )

        BD, H = pm.shape
        B = BD // model.D

        # Reshape scaled predictions to (B, H, D)
        pm_bhd = pm.reshape(B, model.D, H).transpose(0, 2, 1)
        ps_bhd = ps.reshape(B, model.D, H).transpose(0, 2, 1)
        yy_bhd = yy.reshape(B, model.D, H).transpose(0, 2, 1)
        tt_bh = tt.reshape(B, model.D, H)[:, 0, :].reshape(B, H, 1)
        scaled_metric_extras = {}
        for key, value in scaled_metric_extras_flat.items():
            if value.ndim == 3:
                R = value.shape[-1]
                scaled_metric_extras[key] = value.reshape(B, model.D, H, R).transpose(
                    0, 2, 1, 3
                )
            else:
                scaled_metric_extras[key] = value.reshape(B, model.D, H).transpose(
                    0, 2, 1
                )

        # Inverse-transform to get raw predictions
        if target_scaler is not None:
            raw_scale = np.asarray(target_scaler.scale_).reshape(1, 1, -1)
            mu_raw = target_scaler.inverse_transform(
                pm_bhd.reshape(-1, model.D)
            ).reshape(B, H, model.D)
            y_raw = target_scaler.inverse_transform(
                yy_bhd.reshape(-1, model.D)
            ).reshape(B, H, model.D)
            sd_raw = ps_bhd * raw_scale
            raw_metric_extras = {}
            for key, value in scaled_metric_extras.items():
                if key in {"latent_std", "obs_scale", "scale"}:
                    raw_metric_extras[key] = value * raw_scale
                elif key in {"obs_scale_components"}:
                    raw_metric_extras[key] = value * raw_scale[..., None]
                elif key in {"residual_obs_var"}:
                    raw_metric_extras[key] = value * raw_scale.pow(2) if hasattr(raw_scale, "pow") else value * (raw_scale ** 2)
                else:
                    raw_metric_extras[key] = value
        else:
            mu_raw, y_raw, sd_raw = pm_bhd, yy_bhd, ps_bhd
            raw_metric_extras = dict(scaled_metric_extras)

        # ✨ Append both raw and scaled values for each horizon ✨
        for j in range(H):
            h = j + 1
            collect[h]["t"].append(tt_bh[:, j, 0])

            # Raw
            _append_metric_arrays(
                collect[h]["raw"],
                {
                    "y": y_raw[:, j, :],
                    "mu": mu_raw[:, j, :],
                    "sd": sd_raw[:, j, :],
                    **{k: v[:, j, :] for k, v in raw_metric_extras.items()},
                },
            )

            # Scaled
            _append_metric_arrays(
                collect[h]["scaled"],
                {
                    "y": yy_bhd[:, j, :],
                    "mu": pm_bhd[:, j, :],
                    "sd": ps_bhd[:, j, :],
                    **{k: v[:, j, :] for k, v in scaled_metric_extras.items()},
                },
            )

            if acts_bhdr is not None:
                collect[h]["acts"].append(
                    acts_bhdr[:, j, :, :].reshape(B * model.D, -1)
                )

    # Stack results
    all_h_raw, all_h_scaled, acts_per_h = {}, {}, {}
    for h, buf in collect.items():
        if buf["t"]:
            t = np.concatenate(buf["t"])
            acts_per_h[h] = np.concatenate(buf["acts"]) if buf["acts"] else None

            all_h_raw[h] = {k: np.concatenate(v) for k, v in buf["raw"].items()}
            all_h_raw[h]["t"] = t

            all_h_scaled[h] = {k: np.concatenate(v) for k, v in buf["scaled"].items()}
            all_h_scaled[h]["t"] = t

    # ✨ Pass both raw and scaled dictionaries to the finalizer ✨
    results_dir = run_dir
    finalize_eval_artifacts(
        all_h_raw=all_h_raw,
        all_h_scaled=all_h_scaled,
        results_dir=results_dir,
        config=config,
        D=model.D,
        acts_per_h=(acts_per_h if acts_per_h else None),
        series_names=test_loader.dataset.target.columns.tolist(),
    )


def train_mle(
    config: ConfigDict,
    model: SingleEncoderForecaster,
    train_loader,
    valid_loader,
    gp_label_len: int,
    target_scaler=None,
    run_dir: Path = Path("."),
):
    device = config.device
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    metric_options = _student_t_metric_options(config)

    best_val = float("inf")  # Will store the best NLPD
    patience = 0
    history_rows = []
    best_epoch = None

    def _calc_mle_loss(y_true, mu, sigma, nu=None):
        """Calculates NLPD based on density type."""
        if nu is not None:
            dist = torch.distributions.StudentT(
                df=nu, loc=mu, scale=sigma.clamp(min=1e-6)
            )
            nll = -dist.log_prob(y_true)
            return nll.mean()
        else:
            nll = 0.5 * (
                ((y_true - mu) ** 2) / (sigma.clamp(min=1e-9) ** 2)
                + 2.0 * torch.log(sigma.clamp(min=1e-9))
                + math.log(2.0 * math.pi)
            )
            return nll.mean()

    # --- GRADIENT ACCUMULATION SETUP ---
    accumulation_steps = config.get("gradient_accumulation_steps", 1)

    # --- RESUME FROM CHECKPOINT ---
    start_epoch = 1
    resume_ckpt_path = run_dir / "resume_mle.ckpt"
    if resume_ckpt_path.exists():
        print(f"Found MLE resume checkpoint at {resume_ckpt_path}, loading...")
        rckpt = torch.load(resume_ckpt_path, map_location=device)
        try:
            model.load_state_dict(rckpt["model_state"])
            opt.load_state_dict(rckpt["optimizer_state"])
            start_epoch = rckpt["epoch"] + 1
            best_val = rckpt.get("best_val", float("inf"))
            best_epoch = rckpt.get("best_epoch", None)
            patience = rckpt.get("patience", 0)
            history_rows = rckpt.get("history_rows", [])
            print(f"Resumed MLE from epoch {start_epoch - 1} (best_val={best_val:.4f}, patience={patience})")
        except Exception as e:
            print(f"Warning: Failed to resume MLE checkpoint - {e}. Starting fresh.")
            start_epoch = 1

    checkpoint_every = config.get("checkpoint_every", 25)

    for epoch in range(start_epoch, config.training_iterations + 1):
        # --- TRAINING LOOP ---
        model.train()
        loss_sum, nbatches = 0.0, 0

        pbar = tqdm(
            train_loader,
            desc=f"MLE Epoch {epoch}/{config.training_iterations}",
            disable=not is_interactive,
        )

        opt.zero_grad()  # Zero gradients at the start of the epoch

        for i, batch in enumerate(pbar):
            # ... (Data loading to device - unchanged) ...
            if len(batch) != 8:
                raise NotImplementedError(
                    "Multi-horizon inputs required for training (MLE)."
                )
            (
                seq_x,
                seq_y,
                seq_x_mark,
                seq_y_mark,
                _time_idx,
                _mh_y,
                _mh_m,
                _mh_time_idx,
            ) = batch
            seq_x, seq_y, seq_x_mark, seq_y_mark = (
                t.to(device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
            )
            _mh_y = (
                _mh_y.to(device) if _mh_y is not None and hasattr(_mh_y, "to") else None
            )
            _mh_m = (
                _mh_m.to(device) if _mh_m is not None and hasattr(_mh_m, "to") else None
            )

            # --- Forward Pass ---
            mu, sigma, nu = model(
                seq_x,
                seq_x_mark,
                seq_y_mark,
                _mh_m,
                gp_label_len=gp_label_len,
                pred_len=config.pred_len,
            )  # (B, H, D)

            # --- Loss Calculation ---
            y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)  # (B*D, H)
            B, H, D = mu.shape
            y_pred_all = y_pred_all.view(B, D, H).permute(0, 2, 1)  # (B, H, D)

            loss = _calc_mle_loss(y_pred_all, mu, sigma, nu)

            # --- Backward Pass & Step ---
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                opt.step()
                opt.zero_grad()
            # --- End Backward ---

            # --- Logging ---
            loss_value_for_logging = float(loss.item()) * accumulation_steps  # Rescale
            loss_sum += loss_value_for_logging
            nbatches += 1
            pbar.set_postfix({"Loss": f"{loss_value_for_logging:.4f}"})
            # --- End Logging ---
        # --- End Training Batch Loop ---

        train_loss = loss_sum / max(1, nbatches)  # Average training NLL

        # --- VALIDATION LOOP ---
        model.eval()
        val_metric_arrays = {"y": [], "mu": []}
        with torch.no_grad():
            for batch in valid_loader:
                # ... (Data loading for validation - unchanged) ...
                if len(batch) != 8:
                    raise NotImplementedError(
                        "Multi-horizon inputs required for validation (MLE)."
                    )
                (
                    seq_x,
                    seq_y,
                    seq_x_mark,
                    seq_y_mark,
                    _time_idx,
                    _mh_y,
                    _mh_m,
                    _mh_time_idx,
                ) = batch
                seq_x, seq_y, seq_x_mark, seq_y_mark = (
                    t.to(device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
                )
                _mh_y = (
                    _mh_y.to(device)
                    if _mh_y is not None and hasattr(_mh_y, "to")
                    else None
                )
                _mh_m = (
                    _mh_m.to(device)
                    if _mh_m is not None and hasattr(_mh_m, "to")
                    else None
                )

                mu, sigma, nu = model(
                    seq_x,
                    seq_x_mark,
                    seq_y_mark,
                    _mh_m,
                    gp_label_len=gp_label_len,
                    pred_len=config.pred_len,
                )  # (B, H, D)

                B, H, D = mu.shape

                y_pred_all = _build_pred_targets(
                    seq_y, _mh_y, config.pred_len
                )  # (B*D, H)
                y_pred_all = y_pred_all.view(B, D, H).permute(0, 2, 1)  # (B, H, D)

                if nu is not None:
                    _append_metric_arrays(
                        val_metric_arrays,
                        {
                            "y": y_pred_all.cpu().numpy().reshape(-1),
                            "mu": mu.cpu().numpy().reshape(-1),
                            "scale": sigma.cpu().numpy().reshape(-1),
                            "df": nu.cpu().numpy().reshape(-1),
                        },
                    )
                else:
                    _append_metric_arrays(
                        val_metric_arrays,
                        {
                            "y": y_pred_all.cpu().numpy().reshape(-1),
                            "mu": mu.cpu().numpy().reshape(-1),
                            "sd": sigma.cpu().numpy().reshape(-1),
                        },
                    )

        if val_metric_arrays["y"]:
            y_all = np.concatenate(val_metric_arrays["y"])
            mu_all = np.concatenate(val_metric_arrays["mu"])
            if "scale" in val_metric_arrays:
                val_metrics = compute_student_t_metrics(
                    y_all,
                    mu_all,
                    np.concatenate(val_metric_arrays["scale"]),
                    np.concatenate(val_metric_arrays["df"]),
                    interval_level=metric_options["interval_level"],
                    sample_size=metric_options["sample_size"],
                    chunk_size=metric_options["chunk_size"],
                    seed=metric_options["seed"],
                )
            else:
                sd_all = np.concatenate(val_metric_arrays["sd"])
                val_metrics = compute_gaussian_metrics(
                    y_all,
                    mu_all,
                    sd_all,
                    interval_level=metric_options["interval_level"],
                )
            val_objective = float(val_metrics["nlpd_proper"])
        else:
            val_metrics = _empty_metric_result()
            val_objective = float("inf")
        # --- End Validation Loop ---

        # --- LOGGING / CHECKPOINT ---
        row = {
            "epoch": epoch,
            "train_loss": train_loss,  # Average NLL over training batches
            "val_loss": val_objective,  # NLPD on validation set
        }
        row.update(_metric_history_fields(val_metrics))
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch}: Train NLL {train_loss:.4f} | {_metric_console_summary(val_metrics)}"
        )

        # --- Early stopping uses NLPD (val_objective) ---
        if val_objective < best_val:
            best_val = val_objective
            patience = 0
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": float(best_val),  # Save the best NLPD
                    "config": dict(config),
                },
                run_dir / "best_mle.ckpt",
            )
            with open(run_dir / "best_val_metrics.json", "w") as f:
                json.dump(
                    {
                        "epoch": int(epoch),
                        "val_loss": float(best_val),  # Best NLPD
                        **_metric_json_fields(val_metrics),
                    },
                    f,
                    indent=2,
                )
        else:
            if epoch < config.min_epochs:
                patience = 0  # don't early stop yet
            patience += 1
            if patience >= config.patience:
                print("MLE Early stopping.")
                break

        # --- Periodic resume checkpoint ---
        if epoch % checkpoint_every == 0:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "patience": patience,
                "history_rows": history_rows,
                "config": dict(config),
            }, run_dir / "resume_mle.ckpt")

    # Clean up resume checkpoint on successful completion
    resume_mle_path = run_dir / "resume_mle.ckpt"
    if resume_mle_path.exists():
        resume_mle_path.unlink()

    # --- Load Best ---
    best_ckpt = run_dir / "best_mle.ckpt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=config.device)
        try:
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded best MLE checkpoint from epoch {ckpt.get('epoch', 'N/A')}.")
        except Exception as e:
            print(f"Warning: Failed to load MLE checkpoint states - {e}")
    # --- End Load Best ---

    return {
        "best_epoch": best_epoch,
        "best_val_loss": (
            float(best_val) if best_val < float("inf") else None
        ),  # Best NLPD
        "history": history_rows,
    }


# =========================================================================
# Mixture Density Network Training
# =========================================================================

def _mdn_loss(y_true, weights, mus, scales, dfs=None, eps=1e-12):
    """Negative log mixture likelihood.

    y_true:  (B, H, D)
    weights: (B, H, D, K) — sum to 1 over K
    mus:     (B, H, D, K)
    scales:  (B, H, D, K), positive
    dfs:     (B, H, D, K) Student-t df, or None for Gaussian
    """
    log_pi = torch.log(weights.clamp_min(eps))  # (B, H, D, K)
    target = y_true.unsqueeze(-1)  # (B, H, D, 1) broadcasting against K
    if dfs is not None:
        comp = torch.distributions.StudentT(
            df=dfs, loc=mus, scale=scales.clamp_min(1e-8)
        )
    else:
        comp = torch.distributions.Normal(loc=mus, scale=scales.clamp_min(1e-8))
    log_p = comp.log_prob(target)  # (B, H, D, K)
    log_mix = torch.logsumexp(log_pi + log_p, dim=-1)  # (B, H, D)
    return -log_mix.mean()


def train_mdn(
    config: ConfigDict,
    model,
    train_loader,
    valid_loader,
    gp_label_len: int,
    target_scaler=None,
    run_dir: Path = Path("."),
):
    """Train a Mixture Density Network forecaster (MDNForecaster).

    Mirrors train_mle but with mixture NLL and (weights, mus, scales, dfs)
    forward output. Validation NLPD is computed via the analytical mixture
    log-likelihood; CRPS / PICP via mixture sampling.
    """
    device = config.device
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    metric_options = _student_t_metric_options(config)

    best_val = float("inf")
    patience = 0
    history_rows = []
    best_epoch = None

    accumulation_steps = config.get("gradient_accumulation_steps", 1)

    start_epoch = 1
    resume_ckpt_path = run_dir / "resume_mdn.ckpt"
    if resume_ckpt_path.exists():
        rckpt = torch.load(resume_ckpt_path, map_location=device)
        try:
            model.load_state_dict(rckpt["model_state"])
            opt.load_state_dict(rckpt["optimizer_state"])
            start_epoch = rckpt["epoch"] + 1
            best_val = rckpt.get("best_val", float("inf"))
            best_epoch = rckpt.get("best_epoch", None)
            patience = rckpt.get("patience", 0)
            history_rows = rckpt.get("history_rows", [])
            print(
                f"Resumed MDN from epoch {start_epoch - 1} "
                f"(best_val={best_val:.4f}, patience={patience})"
            )
        except Exception as e:
            print(f"Warning: Failed to resume MDN checkpoint - {e}. Starting fresh.")
            start_epoch = 1

    checkpoint_every = config.get("checkpoint_every", 25)

    for epoch in range(start_epoch, config.training_iterations + 1):
        # --- Training loop ---
        model.train()
        loss_sum, nbatches = 0.0, 0

        pbar = tqdm(
            train_loader,
            desc=f"MDN Epoch {epoch}/{config.training_iterations}",
            disable=not is_interactive,
        )
        opt.zero_grad()

        for i, batch in enumerate(pbar):
            if len(batch) != 8:
                raise NotImplementedError("Multi-horizon inputs required for MDN.")
            (
                seq_x, seq_y, seq_x_mark, seq_y_mark,
                _time_idx, _mh_y, _mh_m, _mh_time_idx,
            ) = batch
            seq_x, seq_y, seq_x_mark, seq_y_mark = (
                t.to(device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
            )
            _mh_y = _mh_y.to(device) if _mh_y is not None and hasattr(_mh_y, "to") else None
            _mh_m = _mh_m.to(device) if _mh_m is not None and hasattr(_mh_m, "to") else None

            weights, mus, scales, dfs = model(
                seq_x, seq_x_mark, seq_y_mark, _mh_m,
                gp_label_len=gp_label_len, pred_len=config.pred_len,
            )  # each (B, H, D, K) [dfs may be None]

            y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)  # (B*D, H)
            B, H, D, K = mus.shape
            y_pred_all = y_pred_all.view(B, D, H).permute(0, 2, 1)  # (B, H, D)

            loss = _mdn_loss(y_pred_all, weights, mus, scales, dfs)
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                opt.step()
                opt.zero_grad()

            loss_value_for_logging = float(loss.item()) * accumulation_steps
            loss_sum += loss_value_for_logging
            nbatches += 1
            pbar.set_postfix({"Loss": f"{loss_value_for_logging:.4f}"})

        train_loss = loss_sum / max(1, nbatches)

        # --- Validation loop ---
        model.eval()
        y_buf, w_buf, mu_buf, sc_buf, df_buf = [], [], [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                if len(batch) != 8:
                    raise NotImplementedError("Multi-horizon inputs required (MDN val).")
                (
                    seq_x, seq_y, seq_x_mark, seq_y_mark,
                    _time_idx, _mh_y, _mh_m, _mh_time_idx,
                ) = batch
                seq_x, seq_y, seq_x_mark, seq_y_mark = (
                    t.to(device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
                )
                _mh_y = _mh_y.to(device) if _mh_y is not None and hasattr(_mh_y, "to") else None
                _mh_m = _mh_m.to(device) if _mh_m is not None and hasattr(_mh_m, "to") else None

                weights, mus, scales, dfs = model(
                    seq_x, seq_x_mark, seq_y_mark, _mh_m,
                    gp_label_len=gp_label_len, pred_len=config.pred_len,
                )
                B, H, D, K = mus.shape
                y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)
                y_pred_all = y_pred_all.view(B, D, H).permute(0, 2, 1)  # (B, H, D)

                y_buf.append(y_pred_all.cpu().numpy().reshape(-1))
                # Collapse (B, H, D, K) → (N, K) where N = B*H*D
                w_buf.append(weights.cpu().numpy().reshape(-1, K))
                mu_buf.append(mus.cpu().numpy().reshape(-1, K))
                sc_buf.append(scales.cpu().numpy().reshape(-1, K))
                if dfs is not None:
                    df_buf.append(dfs.cpu().numpy().reshape(-1, K))

        if y_buf:
            y_all = np.concatenate(y_buf)
            w_all = np.concatenate(w_buf, axis=0)
            mu_all = np.concatenate(mu_buf, axis=0)
            sc_all = np.concatenate(sc_buf, axis=0)
            df_all = np.concatenate(df_buf, axis=0) if df_buf else None
            val_metrics = compute_mdn_metrics(
                y_all, w_all, mu_all, sc_all, dfs=df_all,
                interval_level=metric_options["interval_level"],
                sample_size=metric_options["sample_size"],
                chunk_size=metric_options["chunk_size"],
                seed=metric_options["seed"],
            )
            val_objective = float(val_metrics["nlpd_proper"])
        else:
            val_metrics = _empty_metric_result()
            val_objective = float("inf")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_objective,
        }
        row.update(_metric_history_fields(val_metrics))
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch}: Train NLL {train_loss:.4f} | "
            f"{_metric_console_summary(val_metrics)}"
        )

        if val_objective < best_val:
            best_val = val_objective
            patience = 0
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": float(best_val),
                    "config": dict(config),
                },
                run_dir / "best_mdn.ckpt",
            )
            with open(run_dir / "best_val_metrics.json", "w") as f:
                json.dump(
                    {
                        "epoch": int(epoch),
                        "val_loss": float(best_val),
                        **_metric_json_fields(val_metrics),
                    },
                    f, indent=2,
                )
        else:
            if epoch < config.min_epochs:
                patience = 0
            patience += 1
            if patience >= config.patience:
                print("MDN Early stopping.")
                break

        if epoch % checkpoint_every == 0:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "patience": patience,
                "history_rows": history_rows,
                "config": dict(config),
            }, run_dir / "resume_mdn.ckpt")

    resume_mdn_path = run_dir / "resume_mdn.ckpt"
    if resume_mdn_path.exists():
        resume_mdn_path.unlink()

    best_ckpt = run_dir / "best_mdn.ckpt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=config.device)
        try:
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded best MDN checkpoint from epoch {ckpt.get('epoch', 'N/A')}.")
        except Exception as e:
            print(f"Warning: Failed to load MDN checkpoint - {e}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": (
            float(best_val) if best_val < float("inf") else None
        ),
        "history": history_rows,
    }


# =========================================================================
# Quantile Regression Training
# =========================================================================

def _pinball_loss(y_true, q_pred, quantiles):
    """Pinball / quantile loss.
    y_true:   (B, H, D)
    q_pred:   (B, H, D, Q)
    quantiles: (Q,) tensor
    """
    errors = y_true.unsqueeze(-1) - q_pred  # (B, H, D, Q)
    tau = quantiles.view(1, 1, 1, -1)
    loss = torch.max(tau * errors, (tau - 1.0) * errors)
    return loss.mean()


def _quantiles_to_gaussian(q_pred, quantiles):
    """Derive mu (median) and sigma from quantile predictions for metric compat.
    q_pred: ndarray (*, Q)
    quantiles: 1-D array (Q,)
    Returns mu, sigma ndarrays of shape (*,).
    """
    q_arr = np.asarray(quantiles)
    median_idx = np.argmin(np.abs(q_arr - 0.5))
    mu = q_pred[..., median_idx]

    idx_lo = np.argmin(np.abs(q_arr - 0.025))
    idx_hi = np.argmin(np.abs(q_arr - 0.975))
    iqr_95 = q_pred[..., idx_hi] - q_pred[..., idx_lo]
    sigma = np.maximum(iqr_95 / (2.0 * 1.96), 1e-8)
    return mu, sigma


def train_quantile(
    config: ConfigDict,
    model,
    train_loader,
    valid_loader,
    gp_label_len: int,
    target_scaler=None,
    run_dir: Path = Path("."),
):
    from .models import QuantileForecaster
    device = config.device
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    quantiles_np = np.asarray(config.get("quantiles", [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]))
    interval_level = config.get("student_t_interval_level", 0.95)

    best_val = float("inf")
    patience = 0
    history_rows = []
    best_epoch = None

    accumulation_steps = config.get("gradient_accumulation_steps", 1)

    start_epoch = 1
    resume_ckpt_path = run_dir / "resume_quantile.ckpt"
    if resume_ckpt_path.exists():
        print(f"Found quantile resume checkpoint at {resume_ckpt_path}, loading...")
        rckpt = torch.load(resume_ckpt_path, map_location=device)
        try:
            model.load_state_dict(rckpt["model_state"])
            opt.load_state_dict(rckpt["optimizer_state"])
            start_epoch = rckpt["epoch"] + 1
            best_val = rckpt.get("best_val", float("inf"))
            best_epoch = rckpt.get("best_epoch", None)
            patience = rckpt.get("patience", 0)
            history_rows = rckpt.get("history_rows", [])
            print(f"Resumed quantile from epoch {start_epoch - 1} (best_val={best_val:.4f}, patience={patience})")
        except Exception as e:
            print(f"Warning: Failed to resume quantile checkpoint - {e}. Starting fresh.")
            start_epoch = 1

    checkpoint_every = config.get("checkpoint_every", 25)

    for epoch in range(start_epoch, config.training_iterations + 1):
        model.train()
        loss_sum, nbatches = 0.0, 0

        pbar = tqdm(
            train_loader,
            desc=f"QR Epoch {epoch}/{config.training_iterations}",
            disable=not is_interactive,
        )

        opt.zero_grad()

        for i, batch in enumerate(pbar):
            if len(batch) != 8:
                raise NotImplementedError("Multi-horizon inputs required for training (QR).")
            (seq_x, seq_y, seq_x_mark, seq_y_mark,
             _time_idx, _mh_y, _mh_m, _mh_time_idx) = batch
            seq_x, seq_y, seq_x_mark, seq_y_mark = (
                t.to(device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
            )
            _mh_y = _mh_y.to(device) if _mh_y is not None and hasattr(_mh_y, "to") else None
            _mh_m = _mh_m.to(device) if _mh_m is not None and hasattr(_mh_m, "to") else None

            q_pred = model(
                seq_x, seq_x_mark, seq_y_mark, _mh_m,
                gp_label_len=gp_label_len, pred_len=config.pred_len,
            )  # (B, H, D, Q)

            y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)
            B, H, D, Q = q_pred.shape
            y_pred_all = y_pred_all.view(B, D, H).permute(0, 2, 1)  # (B, H, D)

            loss = _pinball_loss(y_pred_all, q_pred, model.quantiles)

            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                opt.step()
                opt.zero_grad()

            loss_value = float(loss.item()) * accumulation_steps
            loss_sum += loss_value
            nbatches += 1
            pbar.set_postfix({"Loss": f"{loss_value:.4f}"})

        train_loss = loss_sum / max(1, nbatches)

        # --- Validation ---
        model.eval()
        val_y_list, val_mu_list, val_sd_list = [], [], []
        val_ql_sum, val_ql_n = 0.0, 0
        with torch.no_grad():
            for batch in valid_loader:
                if len(batch) != 8:
                    raise NotImplementedError("Multi-horizon inputs required for validation (QR).")
                (seq_x, seq_y, seq_x_mark, seq_y_mark,
                 _time_idx, _mh_y, _mh_m, _mh_time_idx) = batch
                seq_x, seq_y, seq_x_mark, seq_y_mark = (
                    t.to(device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
                )
                _mh_y = _mh_y.to(device) if _mh_y is not None and hasattr(_mh_y, "to") else None
                _mh_m = _mh_m.to(device) if _mh_m is not None and hasattr(_mh_m, "to") else None

                q_pred = model(
                    seq_x, seq_x_mark, seq_y_mark, _mh_m,
                    gp_label_len=gp_label_len, pred_len=config.pred_len,
                )
                y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)
                B, H, D, Q = q_pred.shape
                y_true = y_pred_all.view(B, D, H).permute(0, 2, 1)

                val_ql_sum += float(_pinball_loss(y_true, q_pred, model.quantiles).item()) * B
                val_ql_n += B

                q_np = q_pred.cpu().numpy().reshape(-1, Q)
                mu_v, sd_v = _quantiles_to_gaussian(q_np, quantiles_np)
                val_y_list.append(y_true.cpu().numpy().reshape(-1))
                val_mu_list.append(mu_v)
                val_sd_list.append(sd_v)

        if val_y_list:
            y_all = np.concatenate(val_y_list)
            mu_all = np.concatenate(val_mu_list)
            sd_all = np.concatenate(val_sd_list)
            val_metrics = compute_gaussian_metrics(
                y_all, mu_all, sd_all, interval_level=interval_level,
            )
            val_ql = val_ql_sum / max(val_ql_n, 1)
            val_objective = val_ql
        else:
            val_metrics = _empty_metric_result()
            val_objective = float("inf")
            val_ql = float("inf")

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_objective, "val_ql": val_ql}
        row.update(_metric_history_fields(val_metrics))
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch}: Train QL {train_loss:.4f} | Val QL {val_ql:.4f} | {_metric_console_summary(val_metrics)}"
        )

        if val_objective < best_val:
            best_val = val_objective
            patience = 0
            best_epoch = epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch,
                 "val_loss": float(best_val), "config": dict(config)},
                run_dir / "best_quantile.ckpt",
            )
            with open(run_dir / "best_val_metrics.json", "w") as f:
                json.dump(
                    {"epoch": int(epoch), "val_loss": float(best_val),
                     **_metric_json_fields(val_metrics)},
                    f, indent=2,
                )
        else:
            if epoch < config.min_epochs:
                patience = 0
            patience += 1
            if patience >= config.patience:
                print("Quantile early stopping.")
                break

        if epoch % checkpoint_every == 0:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "epoch": epoch, "best_val": best_val,
                "best_epoch": best_epoch, "patience": patience,
                "history_rows": history_rows, "config": dict(config),
            }, run_dir / "resume_quantile.ckpt")

    resume_qr_path = run_dir / "resume_quantile.ckpt"
    if resume_qr_path.exists():
        resume_qr_path.unlink()

    best_ckpt = run_dir / "best_quantile.ckpt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=config.device)
        try:
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded best quantile checkpoint from epoch {ckpt.get('epoch', 'N/A')}.")
        except Exception as e:
            print(f"Warning: Failed to load quantile checkpoint states - {e}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val) if best_val < float("inf") else None,
        "history": history_rows,
    }
