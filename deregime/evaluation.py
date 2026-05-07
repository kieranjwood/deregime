import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import special as spspecial
from scipy import stats as spstats

from .config import ConfigDict


def _flatten_last_is_D(x: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B*D, T) in the same (B,D,…) ordering used by _split_last_dim_by_D."""
    B, T, D = x.shape
    return x.permute(0, 2, 1).reshape(B * D, T)


def _build_pred_targets(
    seq_y: torch.Tensor, mh_y: torch.Tensor | None, pred_len: int
) -> torch.Tensor:
    """
    Build prediction targets aligned with (B*D, H) GP shapes.

    Args
    ----
    seq_y   : (B, L_y, D)   where L_y = gp_label_len + 1  (decoder-start is the last step)
    mh_y    : (B, H-1, D) or None
    pred_len: H

    Returns
    -------
    y_pred_all : (B*D, H)
    """
    if pred_len == 1:
        y_bt = seq_y[:, -1:, :]  # (B, 1, D)
    else:
        if (mh_y is None) or (mh_y.numel() == 0):
            raise ValueError(
                "pred_len>1 requires non-empty multi-horizon targets mh_y."
            )
        # concat [decoder-start] + [future H-1] then keep last H along time
        y_bt = torch.cat([seq_y, mh_y], dim=1)  # (B, L_y + (H-1), D)
        y_bt = y_bt[:, -pred_len:, :]  # (B, H, D)

    # fold D into batch to match (B*D, ·) pipeline
    return y_bt.permute(0, 2, 1).reshape(-1, y_bt.size(1))  # (B*D, H)


def _build_pred_times(
    seq_y_time_idx: torch.Tensor, mh_time_idx: torch.Tensor | None, pred_len: int
) -> torch.Tensor:
    """
    seq_y_time_idx : (B, L_y, D)
    mh_time_idx    : (B, H-1, D) or None
    -> (B*D, H) int64
    """
    if pred_len == 1:
        t_bt = seq_y_time_idx[:, -1:, :]  # (B, 1, D)
    else:
        if (mh_time_idx is None) or (mh_time_idx.numel() == 0):
            raise ValueError("pred_len>1 requires non-empty mh_time_idx.")
        t_bt = torch.cat([seq_y_time_idx, mh_time_idx], dim=1)  # (B, L_y+(H-1), D)
        t_bt = t_bt[:, -pred_len:, :]  # (B, H, D)
    return t_bt.permute(0, 2, 1).reshape(-1, t_bt.size(1))


def _build_pred_times_flat(
    seq_y_t: torch.Tensor, mh_t: torch.Tensor | None, pred_len: int, D: int
) -> torch.Tensor:
    if pred_len == 1:
        t = seq_y_t[:, -1:].unsqueeze(-1)  # (B, 1, 1)
    else:
        if (mh_t is None) or (mh_t.numel() == 0):
            raise ValueError("pred_len>1 requires non-empty mh time idx.")
        t = torch.cat([seq_y_t, mh_t], dim=-1)[..., -pred_len:]  # (B, H)
        t = t.unsqueeze(-1)  # (B, H, 1)
    t = t.repeat_interleave(D, dim=0)  # (B*D, H, 1)
    return t.squeeze(-1)


def _metric_options_from_config(config: ConfigDict) -> dict:
    return {
        "interval_level": float(config.get("student_t_interval_level", 0.95)),
        "sample_size": int(config.get("student_t_eval_samples", 128)),
        "chunk_size": int(config.get("student_t_metric_chunk_size", 4096)),
        "gh_points": int(config.get("student_t_gh_points", 20)),
        "seed": int(config.get("seed", 0)),
    }


def _metric_pack(pack: dict) -> dict:
    return {k: v for k, v in pack.items() if k != "t"}


def _concat_metric_packs(packs: list[dict]) -> dict:
    keys = packs[0].keys()
    return {k: np.concatenate([pack[k] for pack in packs], axis=0) for k in keys}


def _slice_metric_pack_dim(pack: dict, dim: int) -> dict:
    y = pack["y"]
    sliced = {}
    for k, v in pack.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == y.shape[1]:
            sliced[k] = v[:, dim]
        elif isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[1] == y.shape[1]:
            sliced[k] = v[:, dim, :]
        else:
            sliced[k] = v
    return sliced


def _compute_metrics_from_pack(pack: dict, metric_options: dict | None = None) -> dict:
    metric_options = metric_options or {}
    interval_level = float(metric_options.get("interval_level", 0.95))
    sample_size = int(metric_options.get("sample_size", 128))
    chunk_size = int(metric_options.get("chunk_size", 4096))
    gh_points = int(metric_options.get("gh_points", 20))
    seed = int(metric_options.get("seed", 0))

    if {"mdn_weights", "mdn_mus", "mdn_scales"} <= set(pack):
        return compute_mdn_metrics(
            pack["y"],
            pack["mdn_weights"],
            pack["mdn_mus"],
            pack["mdn_scales"],
            dfs=pack.get("mdn_dfs"),
            interval_level=interval_level,
            sample_size=sample_size,
            chunk_size=chunk_size,
            seed=seed,
        )
    if {"latent_std", "mixture_weights", "obs_scale_components"} <= set(pack):
        return compute_regime_student_t_mixture_metrics(
            pack["y"],
            pack["mu"],
            pack["latent_std"],
            pack["mixture_weights"],
            pack["obs_scale_components"],
            pack.get("df_components"),
            interval_level=interval_level,
            sample_size=sample_size,
            chunk_size=chunk_size,
            gh_points=gh_points,
            seed=seed,
        )
    if {"latent_std", "obs_scale", "df"} <= set(pack):
        return compute_student_t_mixture_metrics(
            pack["y"],
            pack["mu"],
            pack["latent_std"],
            pack["obs_scale"],
            pack["df"],
            interval_level=interval_level,
            sample_size=sample_size,
            chunk_size=chunk_size,
            gh_points=gh_points,
            seed=seed,
        )
    if {"scale", "df"} <= set(pack):
        return compute_student_t_metrics(
            pack["y"],
            pack["mu"],
            pack["scale"],
            pack["df"],
            interval_level=interval_level,
            sample_size=sample_size,
            chunk_size=chunk_size,
            seed=seed,
        )
    return compute_gaussian_metrics(
        pack["y"], pack["mu"], pack["sd"], interval_level=interval_level
    )


def _interval_bounds_from_pack(
    pack: dict, metric_options: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    metric_options = metric_options or {}
    interval_level = float(metric_options.get("interval_level", 0.95))
    sample_size = int(metric_options.get("sample_size", 128))
    chunk_size = int(metric_options.get("chunk_size", 4096))
    seed = int(metric_options.get("seed", 0))

    if {"mdn_weights", "mdn_mus", "mdn_scales"} <= set(pack):
        return mdn_interval_bounds(
            pack["mdn_weights"],
            pack["mdn_mus"],
            pack["mdn_scales"],
            dfs=pack.get("mdn_dfs"),
            interval_level=interval_level,
            sample_size=sample_size,
            chunk_size=chunk_size,
            seed=seed,
        )
    if {"latent_std", "mixture_weights", "obs_scale_components"} <= set(pack):
        return regime_student_t_mixture_interval_bounds(
            pack["mu"],
            pack["latent_std"],
            pack["mixture_weights"],
            pack["obs_scale_components"],
            pack.get("df_components"),
            interval_level=interval_level,
            sample_size=sample_size,
            chunk_size=chunk_size,
            seed=seed,
        )
    if {"latent_std", "obs_scale", "df"} <= set(pack):
        return student_t_mixture_interval_bounds(
            pack["mu"],
            pack["latent_std"],
            pack["obs_scale"],
            pack["df"],
            interval_level=interval_level,
            sample_size=sample_size,
            chunk_size=chunk_size,
            seed=seed,
        )
    if {"scale", "df"} <= set(pack):
        return student_t_interval_bounds(
            pack["mu"],
            pack["scale"],
            pack["df"],
            interval_level=interval_level,
        )

    z = float(spstats.norm.ppf(0.5 + interval_level / 2.0))
    return pack["mu"] - z * pack["sd"], pack["mu"] + z * pack["sd"]


def _per_dim_metrics_from_pack(
    pack: dict,
    series_names: Optional[list[str]] = None,
    metric_options: dict | None = None,
) -> pd.DataFrame:
    y = pack["y"]
    if series_names is None:
        series_names = [f"dim_{d}" for d in range(y.shape[1])]

    rows = []
    for d in range(y.shape[1]):
        m = _compute_metrics_from_pack(
            _slice_metric_pack_dim(pack, d), metric_options=metric_options
        )
        m["dim"] = d
        m["channel_name"] = series_names[d]
        rows.append(m)
    return pd.DataFrame(
        rows,
        columns=[
            "dim",
            "channel_name",
            "mse",
            "rmse",
            "mae",
            "crps",
            "nlpd",
            "picp",
            "crps_proper",
            "nlpd_proper",
            "picp_proper",
            "crps_gauss_proxy",
            "nlpd_gauss_proxy",
            "picp_gauss_proxy",
            "n",
        ],
    )


def _macro_from_per_dim(df: pd.DataFrame) -> dict:
    """
    Macro = mean of per-dim metrics (rmse/mae/crps/nlpd/picp), weighted n for "n".
    """
    if len(df) == 0:
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
    w = df["n"].values
    w = np.maximum(w, 1e-12)

    def wmean(col):
        return float(np.average(df[col].values, weights=w))

    return {
        "mse": wmean("mse"),  # <-- ADD THIS
        "rmse": wmean("rmse"),
        "mae": wmean("mae"),
        "crps": wmean("crps"),
        "nlpd": wmean("nlpd"),
        "picp": wmean("picp"),
        "crps_proper": wmean("crps_proper"),
        "nlpd_proper": wmean("nlpd_proper"),
        "picp_proper": wmean("picp_proper"),
        "crps_gauss_proxy": wmean("crps_gauss_proxy"),
        "nlpd_gauss_proxy": wmean("nlpd_gauss_proxy"),
        "picp_gauss_proxy": wmean("picp_gauss_proxy"),
        "n": int(np.sum(df["n"].values)),
    }


def _save_predictions_table(
    path: Path,
    times: np.ndarray,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    lower_95: Optional[np.ndarray] = None,
    upper_95: Optional[np.ndarray] = None,
    dims_to_include: Optional[list[int]] = None,
    series_names: Optional[list[str]] = None,
):
    """
    Saves predictions to a long-format CSV file.

    The output includes columns for time, dimension index, channel name,
    true values, predicted mean, standard deviation, and a 95% prediction interval.
    """
    if dims_to_include is None:
        dims_to_include = list(range(y_true.shape[1]))

    if series_names is None:
        # Create default names like "dim_0", "dim_1" if none are provided
        series_names = [f"dim_{i}" for i in range(y_true.shape[1])]

    if lower_95 is None or upper_95 is None:
        lower_95 = y_mean - 1.96 * y_std
        upper_95 = y_mean + 1.96 * y_std

    rows = []
    for d in dims_to_include:
        rows.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(times),
                    "dim": d,
                    "channel_name": series_names[d],  # Adds the channel name column
                    "y_true": y_true[:, d],
                    "y_mean": y_mean[:, d],
                    "y_std": y_std[:, d],
                    "y_lower_95": lower_95[:, d],
                    "y_upper_95": upper_95[:, d],
                }
            )
        )

    out = pd.concat(rows, axis=0, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def _save_activations_table(
    path: Path,
    times: np.ndarray,
    acts: np.ndarray,
    D: int,
    R: int,
    series_names: list[str],
):
    """
    Saves regime activations to a long-format CSV file.

    The output includes columns for time, dimension, channel name, and one column for each regime's activation (R1, R2, ...).
    """
    # acts is shape (num_samples * D, R), times is shape (num_samples,)
    num_samples = len(times)

    # Create columns that align with the flattened activation array
    times_col = np.repeat(times, D)
    dim_col = np.tile(np.arange(D), num_samples)
    channel_col = np.tile(np.asarray(series_names), num_samples)

    df_data = {
        "time": pd.to_datetime(times_col),
        "dim": dim_col,
        "channel_name": channel_col,
    }

    # Add a column for each regime's activation
    for r in range(R):
        df_data[f"R{r+1}"] = acts[:, r]

    df = pd.DataFrame(df_data)
    df.to_csv(path, index=False)


def _save_metrics_bundles(
    out_dir: Path,
    tag: str,  # e.g., "h1" / "h3" / "macro_all"
    scaled_per_dim: pd.DataFrame,
    raw_per_dim: pd.DataFrame,
    scaled_macro: dict,
    raw_macro: dict,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    scaled_per_dim.to_csv(out_dir / f"metrics_scaled_per_dim__{tag}.csv", index=False)
    raw_per_dim.to_csv(out_dir / f"metrics_raw_per_dim__{tag}.csv", index=False)
    with open(out_dir / f"metrics_scaled_macro__{tag}.json", "w") as f:
        json.dump(scaled_macro, f, indent=2)
    with open(out_dir / f"metrics_raw_macro__{tag}.json", "w") as f:
        json.dump(raw_macro, f, indent=2)


def plot_predictions_per_dim(
    *,
    time_seg: np.ndarray,
    y_true_seg: np.ndarray,
    y_mean_seg: np.ndarray,
    y_std_seg: np.ndarray,
    acts_seg: Optional[np.ndarray],
    out_path: Path,
    title: str,
    reff_info: Optional[str],
    effective_regimes: Optional[set[int]] = None,
):
    """
    Generates a single plot with predictions and (optional) activations.
    This function no longer performs any data reshaping or slicing.
    """
    has_regimes = acts_seg is not None and acts_seg.size > 0

    if has_regimes:
        fig, (ax_pred, ax_acts) = plt.subplots(
            2, 1, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_pred = plt.subplots(figsize=(15, 5))
        ax_acts = None

    td = pd.to_datetime(time_seg)

    # --- Top Plot: Predictions ---
    ax_pred.plot(td, y_true_seg, label="True", color="blue")
    ax_pred.plot(td, y_mean_seg, label="Mean", color="orange")
    ax_pred.fill_between(
        td,
        y_mean_seg - 1.96 * y_std_seg,
        y_mean_seg + 1.96 * y_std_seg,
        color="orange",
        alpha=0.2,
        label="95% PI",
    )
    ax_pred.set_title(title)
    ax_pred.set_ylabel("Value")
    ax_pred.legend(loc="upper left")
    ax_pred.grid(True, linestyle="--", alpha=0.6)

    if reff_info:
        ax_pred.text(
            0.99,
            0.95,
            reff_info,
            transform=ax_pred.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        )

    # --- Bottom Plot: Regime Activations ---
    if has_regimes and ax_acts:
        num_regimes = acts_seg.shape[1]

        if num_regimes <= 20:
            color_map = plt.get_cmap("tab20")
            colors = [color_map(i / 20.0) for i in range(num_regimes)]
        else:
            # Fallback for larger regime truncations.
            color_map = plt.get_cmap("hsv")
            colors = [color_map(i) for i in np.linspace(0, 0.95, num_regimes)]

        for r in range(num_regimes):
            label = None
            if effective_regimes is not None and r in effective_regimes:
                label = f"R{r+1}"

            ax_acts.plot(td, acts_seg[:, r], label=label, lw=1.5, color=colors[r])

        ax_acts.set_ylim(0.0, 1.05)
        ax_acts.set_ylabel("Activation")

        handles, labels = ax_acts.get_legend_handles_labels()
        if labels and len(labels) <= 16:
            ax_acts.legend(
                ncol=len(labels),
                fontsize=8,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.4),
            )

        ax_acts.set_xlabel("Time")
        ax_acts.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_all_horizons(
    D: int,
    results_dir: Path,
    config: ConfigDict,
    all_h_data: Dict[int, Dict[str, np.ndarray]],
    acts_per_h: Optional[Dict[int, np.ndarray]] = None,
    series_names: Optional[List[str]] = None,
    subdir: str = "plots",
):
    plots_root_dir = results_dir / subdir

    if series_names is None or len(series_names) != D:
        series_names = [f"dim_{i}" for i in range(D)]

    horizons_to_plot = [h for h in config.plot_multi_horizons if h in all_h_data]

    # --- NEW: User-added default horizon logic ---
    if not horizons_to_plot and 1 in all_h_data:
        print(
            "Warning: No horizons specified in config match results. Defaulting to H=1 for plotting."
        )
        horizons_to_plot = [1]
    # --- END NEW ---

    segment_len = config.test_segment_plot_len

    for h in horizons_to_plot:
        h_data = all_h_data[h]
        h_acts = acts_per_h.get(h) if acts_per_h else None
        total_timesteps = h_data["t"].shape[0]

        for d in range(D):
            channel_name = series_names[d]
            output_dir = plots_root_dir / f"H{h}" / channel_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # This loop creates each individual plot (segment)
            for i, start in enumerate(range(0, total_timesteps, segment_len)):
                end = min(start + segment_len, total_timesteps)

                t_seg = h_data["t"][start:end]
                y_seg = h_data["y"][start:end, d]
                mu_seg = h_data["mu"][start:end, d]
                sd_seg = h_data["sd"][start:end, d]

                acts_seg = None
                reff_string = None
                effective_regimes = None
                if h_acts is not None:
                    acts_reshaped = h_acts.reshape(total_timesteps, D, -1)  # (T, D, R)
                    acts_seg = acts_reshaped[start:end, d, :]  # (seg, R)

                    # --- THIS CALCULATION RUNS PER-WINDOW ---
                    # It uses acts_seg, which is the slice for the current plot
                    gbar = acts_seg.mean(axis=0)
                    is_effective = gbar > config.remainder_epsilon
                    effective_regimes = set(np.where(is_effective)[0])
                    reff = len(effective_regimes)
                    reff_string = f"$R_{{eff}}$ = {reff}"
                    # --- END PER-WINDOW CALCULATION ---

                filename = f"H{h}_{channel_name}_segment_{i+1}.png"
                plot_predictions_per_dim(
                    time_seg=t_seg,
                    y_true_seg=y_seg,
                    y_mean_seg=mu_seg,
                    y_std_seg=sd_seg,
                    acts_seg=acts_seg,
                    out_path=output_dir / filename,
                    title=f"H{h} Forecast — {channel_name} (Segment {i+1})",
                    reff_info=reff_string,
                    effective_regimes=effective_regimes,
                )


def finalize_eval_artifacts(
    all_h_raw: dict,
    all_h_scaled: dict,
    *,
    results_dir: Path,
    config: ConfigDict,
    D: int,
    acts_per_h: dict | None = None,
    series_names: list[str] | None = None,
):
    # Define and create the required subdirectories
    pred_dir = results_dir / "predictions"
    metrics_dir = results_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    macro_rows_raw, macro_rows_scaled = [], []
    dims_to_include = (
        None if config.save_all_dims else list(range(int(config.plot_first_n_dims)))
    )
    metric_options = _metric_options_from_config(config)

    all_packs_raw, all_packs_scl = [], []

    for h in sorted(all_h_raw.keys()):
        # Unpack raw and scaled data
        pack_raw = all_h_raw[h]
        pack_scl = all_h_scaled[h]

        t, y_raw, mu_raw, sd_raw = (
            pack_raw["t"],
            pack_raw["y"],
            pack_raw["mu"],
            pack_raw["sd"],
        )
        y_scl, mu_scl, sd_scl = pack_scl["y"], pack_scl["mu"], pack_scl["sd"]

        metric_pack_raw = _metric_pack(pack_raw)
        metric_pack_scl = _metric_pack(pack_scl)
        all_packs_raw.append(metric_pack_raw)
        all_packs_scl.append(metric_pack_scl)

        lower_raw, upper_raw = _interval_bounds_from_pack(
            metric_pack_raw, metric_options=metric_options
        )
        lower_scl, upper_scl = _interval_bounds_from_pack(
            metric_pack_scl, metric_options=metric_options
        )

        _save_predictions_table(
            pred_dir / f"predictions__h{h}.csv",
            times=t,
            y_true=y_raw,
            y_mean=mu_raw,
            y_std=sd_raw,
            lower_95=lower_raw,
            upper_95=upper_raw,
            dims_to_include=dims_to_include,
            series_names=series_names,
        )

        if acts_per_h and h in acts_per_h and acts_per_h[h] is not None:
            # ... (code to save activations) ...
            activations = acts_per_h[h]
            if activations.size > 0:
                _save_activations_table(
                    path=pred_dir / f"activations__h{h}.csv",
                    times=t,
                    acts=activations,
                    D=D,
                    R=activations.shape[1],
                    series_names=series_names,
                )

        _save_predictions_table(
            pred_dir / f"predictions_scaled__h{h}.csv",
            times=t,
            y_true=y_scl,
            y_mean=mu_scl,
            y_std=sd_scl,
            lower_95=lower_scl,
            upper_95=upper_scl,
            dims_to_include=dims_to_include,
            series_names=series_names,
        )
        per_dim_raw = _per_dim_metrics_from_pack(
            metric_pack_raw,
            series_names=series_names,
            metric_options=metric_options,
        )
        macro_raw = _macro_from_per_dim(per_dim_raw)
        per_dim_scl = _per_dim_metrics_from_pack(
            metric_pack_scl,
            series_names=series_names,
            metric_options=metric_options,
        )
        macro_scl = _macro_from_per_dim(per_dim_scl)
        _save_metrics_bundles(
            out_dir=metrics_dir,
            tag=f"h{h}",
            raw_per_dim=per_dim_raw,
            raw_macro=macro_raw,
            scaled_per_dim=per_dim_scl,
            scaled_macro=macro_scl,
        )
        macro_rows_raw.append({"horizon": h, **macro_raw})
        macro_rows_scaled.append({"horizon": h, **macro_scl})

    # ... (saving macro summaries is also correct) ...
    # (This part saves the per-horizon summary CSVs)
    macro_summary_raw = pd.DataFrame(macro_rows_raw)
    macro_summary_scl = pd.DataFrame(macro_rows_scaled)
    macro_summary_raw.to_csv(metrics_dir / "metrics_raw_macro_summary.csv", index=False)
    macro_summary_scl.to_csv(
        metrics_dir / "metrics_scaled_macro_summary.csv", index=False
    )

    if all_packs_raw:
        print("Computing final aggregation across all horizons...")

        agg_pack_raw = _concat_metric_packs(all_packs_raw)
        agg_pack_scl = _concat_metric_packs(all_packs_scl)

        per_dim_agg_raw = _per_dim_metrics_from_pack(
            agg_pack_raw,
            series_names=series_names,
            metric_options=metric_options,
        )
        per_dim_agg_scl = _per_dim_metrics_from_pack(
            agg_pack_scl,
            series_names=series_names,
            metric_options=metric_options,
        )

        macro_agg_raw = _macro_from_per_dim(per_dim_agg_raw)
        macro_agg_scl = _macro_from_per_dim(per_dim_agg_scl)

        _save_metrics_bundles(
            out_dir=metrics_dir,
            tag="ALL_HORIZONS",
            raw_per_dim=per_dim_agg_raw,
            raw_macro=macro_agg_raw,
            scaled_per_dim=per_dim_agg_scl,
            scaled_macro=macro_agg_scl,
        )

    # ✨ CORRECTED PLOTTING LOGIC ✨
    if config.plot_multi_horizons and all_h_raw:
        plot_all_horizons(
            D=D,
            results_dir=results_dir,
            config=config,
            all_h_data=all_h_raw,
            acts_per_h=acts_per_h,
            series_names=series_names,
            subdir="plots",
        )

    return pd.DataFrame(macro_rows_raw), pd.DataFrame(macro_rows_scaled)


# Metrics
def _phi(z):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)


def _Phi(z):
    return 0.5 * (1.0 + spspecial.erf(z / np.sqrt(2.0)))


def gaussian_crps(mu, sigma, y, eps=1e-8):
    sigma = np.maximum(sigma, eps)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - 1.0 / np.sqrt(np.pi))


_GH_CACHE: Dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss_hermite_nodes_weights(points: int) -> tuple[np.ndarray, np.ndarray]:
    if points not in _GH_CACHE:
        nodes, weights = np.polynomial.hermite.hermgauss(points)
        weights = weights / np.sqrt(np.pi)
        _GH_CACHE[points] = (nodes.astype(np.float64), weights.astype(np.float64))
    return _GH_CACHE[points]


def _point_metrics(y, mu) -> dict:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    mse = np.mean((y - mu) ** 2)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(y - mu))),
        "n": int(y.shape[0]),
    }


def _prob_metrics_dict(*, crps: float, nlpd: float, picp: float) -> dict:
    return {
        "crps": float(crps),
        "nlpd": float(nlpd),
        "picp": float(picp),
    }


def _compose_metric_views(point: dict, proper: dict, gauss_proxy: dict) -> dict:
    return {
        "mse": float(point["mse"]),
        "rmse": float(point["rmse"]),
        "mae": float(point["mae"]),
        "crps": float(proper["crps"]),
        "nlpd": float(proper["nlpd"]),
        "picp": float(proper["picp"]),
        "crps_proper": float(proper["crps"]),
        "nlpd_proper": float(proper["nlpd"]),
        "picp_proper": float(proper["picp"]),
        "crps_gauss_proxy": float(gauss_proxy["crps"]),
        "nlpd_gauss_proxy": float(gauss_proxy["nlpd"]),
        "picp_gauss_proxy": float(gauss_proxy["picp"]),
        "n": int(point["n"]),
    }


def _student_t_samples(
    mu: np.ndarray,
    scale: np.ndarray,
    df: np.ndarray,
    *,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    scale = np.maximum(np.asarray(scale, dtype=np.float64).reshape(-1), 1e-8)
    df = np.maximum(np.asarray(df, dtype=np.float64).reshape(-1), 2.0 + 1e-8)
    return mu[None, :] + scale[None, :] * rng.standard_t(
        df[None, :], size=(sample_size, mu.shape[0])
    )


def _student_t_mixture_samples(
    mu: np.ndarray,
    latent_std: np.ndarray,
    obs_scale: np.ndarray,
    df: np.ndarray,
    *,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    latent_std = np.maximum(np.asarray(latent_std, dtype=np.float64).reshape(-1), 0.0)
    obs_scale = np.maximum(np.asarray(obs_scale, dtype=np.float64).reshape(-1), 1e-8)
    df = np.maximum(np.asarray(df, dtype=np.float64).reshape(-1), 2.0 + 1e-8)
    latent = mu[None, :] + latent_std[None, :] * rng.standard_normal(
        size=(sample_size, mu.shape[0])
    )
    noise = obs_scale[None, :] * rng.standard_t(
        df[None, :], size=(sample_size, mu.shape[0])
    )
    return latent + noise


def _regime_student_t_mixture_samples(
    mu: np.ndarray,
    latent_std: np.ndarray,
    weights: np.ndarray,
    obs_scale: np.ndarray,
    df: np.ndarray | None,
    *,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    latent_std = np.maximum(np.asarray(latent_std, dtype=np.float64).reshape(-1), 0.0)
    weights = np.asarray(weights, dtype=np.float64).reshape(mu.shape[0], -1)
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    obs_scale = np.maximum(
        np.asarray(obs_scale, dtype=np.float64).reshape(mu.shape[0], -1), 1e-8
    )
    if df is not None:
        df = np.maximum(
            np.asarray(df, dtype=np.float64).reshape(mu.shape[0], -1), 2.0 + 1e-8
        )

    latent = mu[None, :] + latent_std[None, :] * rng.standard_normal(
        size=(sample_size, mu.shape[0])
    )
    u = rng.random(size=(sample_size, mu.shape[0]))
    cdf = np.cumsum(weights, axis=-1)
    comp = (u[..., None] > cdf[None, :, :]).sum(axis=-1)
    comp = np.minimum(comp, weights.shape[-1] - 1)
    scale_chosen = np.take_along_axis(
        obs_scale[None, :, :], comp[..., None], axis=-1
    ).squeeze(-1)
    if df is None:
        noise = scale_chosen * rng.standard_normal(size=scale_chosen.shape)
    else:
        df_chosen = np.take_along_axis(
            df[None, :, :], comp[..., None], axis=-1
        ).squeeze(-1)
        noise = scale_chosen * rng.standard_t(df_chosen)
    return latent + noise


def _crps_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(1, -1)
    term1 = np.mean(np.abs(samples - y), axis=0)
    xs = np.sort(samples, axis=0)
    ns = xs.shape[0]
    coeff = (2.0 * np.arange(1, ns + 1) - ns - 1.0).reshape(ns, 1)
    term2 = np.sum(coeff * xs, axis=0) / float(ns * ns)
    return term1 - term2


def student_t_interval_bounds(
    mu,
    scale,
    df,
    *,
    interval_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - interval_level
    mu = np.asarray(mu, dtype=np.float64)
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-8)
    df = np.maximum(np.asarray(df, dtype=np.float64), 2.0 + 1e-8)
    lower = spstats.t.ppf(alpha / 2.0, df=df, loc=mu, scale=scale)
    upper = spstats.t.ppf(1.0 - alpha / 2.0, df=df, loc=mu, scale=scale)
    return lower, upper


def student_t_mixture_interval_bounds(
    mu,
    latent_std,
    obs_scale,
    df,
    *,
    interval_level: float = 0.95,
    sample_size: int = 128,
    chunk_size: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    orig_shape = np.asarray(mu).shape
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    latent_std = np.asarray(latent_std, dtype=np.float64).reshape(-1)
    obs_scale = np.asarray(obs_scale, dtype=np.float64).reshape(-1)
    df = np.asarray(df, dtype=np.float64).reshape(-1)

    lower = np.empty_like(mu)
    upper = np.empty_like(mu)
    alpha = 1.0 - interval_level
    rng = np.random.default_rng(seed)

    for start in range(0, mu.shape[0], chunk_size):
        end = min(start + chunk_size, mu.shape[0])
        samples = _student_t_mixture_samples(
            mu[start:end],
            latent_std[start:end],
            obs_scale[start:end],
            df[start:end],
            sample_size=sample_size,
            rng=rng,
        )
        q = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        lower[start:end] = q[0]
        upper[start:end] = q[1]

    return lower.reshape(orig_shape), upper.reshape(orig_shape)


def regime_student_t_mixture_interval_bounds(
    mu,
    latent_std,
    mixture_weights,
    obs_scale_components,
    df_components=None,
    *,
    interval_level: float = 0.95,
    sample_size: int = 128,
    chunk_size: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    orig_shape = np.asarray(mu).shape
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    latent_std = np.asarray(latent_std, dtype=np.float64).reshape(-1)
    weights = np.asarray(mixture_weights, dtype=np.float64).reshape(mu.shape[0], -1)
    obs_scale = np.asarray(obs_scale_components, dtype=np.float64).reshape(
        mu.shape[0], -1
    )
    df = (
        None
        if df_components is None
        else np.asarray(df_components, dtype=np.float64).reshape(mu.shape[0], -1)
    )

    lower = np.empty_like(mu)
    upper = np.empty_like(mu)
    alpha = 1.0 - interval_level
    rng = np.random.default_rng(seed)

    for start in range(0, mu.shape[0], chunk_size):
        end = min(start + chunk_size, mu.shape[0])
        df_chunk = None if df is None else df[start:end]
        samples = _regime_student_t_mixture_samples(
            mu[start:end],
            latent_std[start:end],
            weights[start:end],
            obs_scale[start:end],
            df_chunk,
            sample_size=sample_size,
            rng=rng,
        )
        q = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        lower[start:end] = q[0]
        upper[start:end] = q[1]

    return lower.reshape(orig_shape), upper.reshape(orig_shape)


def mdn_interval_bounds(
    weights,
    mus,
    scales,
    dfs=None,
    *,
    interval_level: float = 0.95,
    sample_size: int = 128,
    chunk_size: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical interval bounds for a flat MDN via mixture sampling."""
    orig_shape = np.asarray(mus).shape[:-1]
    weights = np.asarray(weights, dtype=np.float64).reshape(-1, weights.shape[-1])
    mus = np.asarray(mus, dtype=np.float64).reshape(-1, mus.shape[-1])
    scales = np.maximum(
        np.asarray(scales, dtype=np.float64).reshape(-1, scales.shape[-1]), 1e-8
    )
    has_t = dfs is not None
    if has_t:
        dfs = np.maximum(
            np.asarray(dfs, dtype=np.float64).reshape(-1, dfs.shape[-1]), 2.0 + 1e-8
        )

    N, K = mus.shape
    alpha = 1.0 - interval_level
    rng = np.random.default_rng(seed)
    lower = np.empty(N)
    upper = np.empty(N)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        w_c = weights[start:end]
        mu_c = mus[start:end]
        sc_c = scales[start:end]
        cdf = np.cumsum(w_c, axis=-1)
        u = rng.random((end - start, sample_size))
        comp_idx = (u[..., None] > cdf[:, None, :]).sum(axis=-1)
        comp_idx = np.clip(comp_idx, 0, K - 1)
        rows = np.arange(end - start)[:, None]
        mu_pick = mu_c[rows, comp_idx]
        sc_pick = sc_c[rows, comp_idx]
        if has_t:
            df_pick = dfs[start:end][rows, comp_idx]
            samp = mu_pick + sc_pick * rng.standard_t(df_pick)
        else:
            samp = rng.normal(loc=mu_pick, scale=sc_pick)
        # samp shape (n, S) — quantile over axis 1 (samples)
        q = np.quantile(samp, [alpha / 2.0, 1.0 - alpha / 2.0], axis=1)
        lower[start:end] = q[0]
        upper[start:end] = q[1]

    return lower.reshape(orig_shape), upper.reshape(orig_shape)


def _compute_gaussian_metric_core(y, mu, sigma, eps=1e-8, interval_level: float = 0.95):
    y = np.asarray(y).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    sigma = np.maximum(np.asarray(sigma).reshape(-1), eps)
    crps = np.mean(gaussian_crps(mu, sigma, y, eps=eps))
    nlpd = float(
        np.mean(
            0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * ((y - mu) ** 2) / (sigma**2)
        )
    )
    z = float(spstats.norm.ppf(0.5 + interval_level / 2.0))
    lower, upper = mu - z * sigma, mu + z * sigma
    picp = float(np.mean((y >= lower) & (y <= upper)))
    return {
        **_point_metrics(y, mu),
        "crps": float(crps),
        "nlpd": nlpd,
        "picp": picp,
    }


def compute_gaussian_metrics(y, mu, sigma, eps=1e-8, interval_level: float = 0.95):
    core = _compute_gaussian_metric_core(
        y, mu, sigma, eps=eps, interval_level=interval_level
    )
    proper = _prob_metrics_dict(
        crps=core["crps"],
        nlpd=core["nlpd"],
        picp=core["picp"],
    )
    return _compose_metric_views(core, proper, proper)


def compute_student_t_metrics(
    y,
    mu,
    scale,
    df,
    *,
    interval_level: float = 0.95,
    sample_size: int = 128,
    chunk_size: int = 4096,
    seed: int = 0,
) -> dict:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    scale = np.maximum(np.asarray(scale, dtype=np.float64).reshape(-1), 1e-8)
    df = np.maximum(np.asarray(df, dtype=np.float64).reshape(-1), 2.0 + 1e-8)

    point = _point_metrics(y, mu)
    nlpd = float(-np.mean(spstats.t.logpdf(y, df=df, loc=mu, scale=scale)))

    lower, upper = student_t_interval_bounds(
        mu, scale, df, interval_level=interval_level
    )
    picp = float(np.mean((y >= lower) & (y <= upper)))

    rng = np.random.default_rng(seed)
    crps_sum = 0.0
    n_total = 0
    for start in range(0, y.shape[0], chunk_size):
        end = min(start + chunk_size, y.shape[0])
        samples = _student_t_samples(
            mu[start:end],
            scale[start:end],
            df[start:end],
            sample_size=sample_size,
            rng=rng,
        )
        crps_sum += float(_crps_from_samples(samples, y[start:end]).sum())
        n_total += end - start

    proper = _prob_metrics_dict(
        crps=float(crps_sum / max(1, n_total)),
        nlpd=nlpd,
        picp=picp,
    )
    proxy_std = scale * np.sqrt(df / np.maximum(df - 2.0, 1e-8))
    proxy_core = _compute_gaussian_metric_core(
        y, mu, proxy_std, interval_level=interval_level
    )
    gauss_proxy = _prob_metrics_dict(
        crps=proxy_core["crps"],
        nlpd=proxy_core["nlpd"],
        picp=proxy_core["picp"],
    )
    return _compose_metric_views(point, proper, gauss_proxy)


def compute_student_t_mixture_metrics(
    y,
    mu,
    latent_std,
    obs_scale,
    df,
    *,
    interval_level: float = 0.95,
    sample_size: int = 128,
    chunk_size: int = 4096,
    gh_points: int = 20,
    seed: int = 0,
) -> dict:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    latent_std = np.maximum(np.asarray(latent_std, dtype=np.float64).reshape(-1), 0.0)
    obs_scale = np.maximum(np.asarray(obs_scale, dtype=np.float64).reshape(-1), 1e-8)
    df = np.maximum(np.asarray(df, dtype=np.float64).reshape(-1), 2.0 + 1e-8)

    point = _point_metrics(y, mu)

    gh_x, gh_w = _gauss_hermite_nodes_weights(gh_points)
    log_w = np.log(np.maximum(gh_w, 1e-300))
    nlpd_sum = 0.0

    rng = np.random.default_rng(seed)
    crps_sum = 0.0
    picp_hits = 0
    n_total = 0
    alpha = 1.0 - interval_level

    for start in range(0, y.shape[0], chunk_size):
        end = min(start + chunk_size, y.shape[0])
        y_c = y[start:end]
        mu_c = mu[start:end]
        latent_std_c = latent_std[start:end]
        obs_scale_c = obs_scale[start:end]
        df_c = df[start:end]

        f_points = (
            mu_c[None, :]
            + np.sqrt(2.0) * latent_std_c[None, :] * gh_x[:, None]
        )
        log_pdf = spstats.t.logpdf(
            y_c[None, :], df=df_c[None, :], loc=f_points, scale=obs_scale_c[None, :]
        )
        log_mix = spspecial.logsumexp(log_w[:, None] + log_pdf, axis=0)
        nlpd_sum += float((-log_mix).sum())

        samples = _student_t_mixture_samples(
            mu_c,
            latent_std_c,
            obs_scale_c,
            df_c,
            sample_size=sample_size,
            rng=rng,
        )
        crps_sum += float(_crps_from_samples(samples, y_c).sum())
        q = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        picp_hits += int(np.sum((y_c >= q[0]) & (y_c <= q[1])))
        n_total += end - start

    proper = _prob_metrics_dict(
        crps=float(crps_sum / max(1, n_total)),
        nlpd=float(nlpd_sum / max(1, n_total)),
        picp=float(picp_hits / max(1, n_total)),
    )
    proxy_std = np.sqrt(
        latent_std**2 + (df / np.maximum(df - 2.0, 1e-8)) * obs_scale**2
    )
    proxy_core = _compute_gaussian_metric_core(
        y, mu, proxy_std, interval_level=interval_level
    )
    gauss_proxy = _prob_metrics_dict(
        crps=proxy_core["crps"],
        nlpd=proxy_core["nlpd"],
        picp=proxy_core["picp"],
    )
    return _compose_metric_views(point, proper, gauss_proxy)


def compute_mdn_metrics(
    y,
    weights,
    mus,
    scales,
    dfs=None,
    *,
    interval_level: float = 0.95,
    sample_size: int = 128,
    chunk_size: int = 4096,
    seed: int = 0,
) -> dict:
    """Metrics for a flat finite mixture: y ~ sum_k pi_k * Comp_k(mu_k, sigma_k, [nu_k]).

    Components are Gaussian when `dfs is None`, else Student-t.
    Inputs (each shape (N, K)):
      weights: mixture weights (sum to 1 over K, last axis)
      mus:     per-component means
      scales:  per-component scales (>0)
      dfs:     per-component degrees-of-freedom (>2) for Student-t, or None
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1, weights.shape[-1])
    mus = np.asarray(mus, dtype=np.float64).reshape(-1, mus.shape[-1])
    scales = np.maximum(np.asarray(scales, dtype=np.float64).reshape(-1, scales.shape[-1]), 1e-8)
    has_t = dfs is not None
    if has_t:
        dfs = np.maximum(np.asarray(dfs, dtype=np.float64).reshape(-1, dfs.shape[-1]), 2.0 + 1e-8)

    N, K = mus.shape
    log_w = np.log(np.maximum(weights, 1e-12))

    # Mixture mean and variance for point metrics + Gaussian-proxy
    mean_mix = np.sum(weights * mus, axis=-1)
    if has_t:
        comp_var = scales**2 * dfs / np.maximum(dfs - 2.0, 1e-8)
    else:
        comp_var = scales**2
    var_mix = np.sum(weights * (comp_var + (mus - mean_mix[:, None]) ** 2), axis=-1)
    std_mix = np.sqrt(np.maximum(var_mix, 1e-16))

    point = _point_metrics(y, mean_mix)

    # Analytic NLPD via logsumexp(log pi_k + log Comp_k(y))
    nlpd_sum = 0.0
    rng = np.random.default_rng(seed)
    crps_sum = 0.0
    picp_hits = 0
    n_total = 0
    alpha = 1.0 - interval_level

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        y_c = y[start:end]
        w_c = weights[start:end]
        log_w_c = log_w[start:end]
        mu_c = mus[start:end]
        sc_c = scales[start:end]

        if has_t:
            df_c = dfs[start:end]
            log_p_k = spstats.t.logpdf(
                y_c[:, None], df=df_c, loc=mu_c, scale=sc_c
            )
        else:
            z = (y_c[:, None] - mu_c) / sc_c
            log_p_k = -0.5 * z**2 - np.log(sc_c) - 0.5 * np.log(2.0 * np.pi)
        log_mix = spspecial.logsumexp(log_w_c + log_p_k, axis=-1)
        nlpd_sum += float((-log_mix).sum())

        # Sample-based CRPS / PICP: draw S samples from the mixture per row
        S = int(sample_size)
        # 1) sample component index ~ Cat(pi_c)
        cdf = np.cumsum(w_c, axis=-1)
        u = rng.random((end - start, S))
        comp_idx = (u[..., None] > cdf[:, None, :]).sum(axis=-1)  # (n, S) in [0, K)
        comp_idx = np.clip(comp_idx, 0, K - 1)
        # 2) sample component-conditional values
        rows = np.arange(end - start)[:, None]
        mu_pick = mu_c[rows, comp_idx]  # (n, S)
        sc_pick = sc_c[rows, comp_idx]
        if has_t:
            df_pick = df_c[rows, comp_idx]
            t_samp = rng.standard_t(df_pick)
            samples = mu_pick + sc_pick * t_samp
        else:
            samples = rng.normal(loc=mu_pick, scale=sc_pick)
        # samples shape (n, S); _crps_from_samples expects (S, n)
        samples = samples.T
        crps_sum += float(_crps_from_samples(samples, y_c).sum())
        q = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        picp_hits += int(np.sum((y_c >= q[0]) & (y_c <= q[1])))
        n_total += end - start

    proper = _prob_metrics_dict(
        crps=float(crps_sum / max(1, n_total)),
        nlpd=float(nlpd_sum / max(1, n_total)),
        picp=float(picp_hits / max(1, n_total)),
    )
    proxy_core = _compute_gaussian_metric_core(
        y, mean_mix, std_mix, interval_level=interval_level
    )
    gauss_proxy = _prob_metrics_dict(
        crps=proxy_core["crps"],
        nlpd=proxy_core["nlpd"],
        picp=proxy_core["picp"],
    )
    return _compose_metric_views(point, proper, gauss_proxy)


def compute_regime_student_t_mixture_metrics(
    y,
    mu,
    latent_std,
    mixture_weights,
    obs_scale_components,
    df_components=None,
    *,
    interval_level: float = 0.95,
    sample_size: int = 128,
    chunk_size: int = 4096,
    gh_points: int = 20,
    seed: int = 0,
) -> dict:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    latent_std = np.maximum(np.asarray(latent_std, dtype=np.float64).reshape(-1), 0.0)
    weights = np.asarray(mixture_weights, dtype=np.float64).reshape(y.shape[0], -1)
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    obs_scale = np.maximum(
        np.asarray(obs_scale_components, dtype=np.float64).reshape(y.shape[0], -1),
        1e-8,
    )
    df = (
        None
        if df_components is None
        else np.maximum(
            np.asarray(df_components, dtype=np.float64).reshape(y.shape[0], -1),
            2.0 + 1e-8,
        )
    )

    point = _point_metrics(y, mu)
    gh_x, gh_w = _gauss_hermite_nodes_weights(gh_points)
    log_w_gh = np.log(np.maximum(gh_w, 1e-300))

    rng = np.random.default_rng(seed)
    crps_sum = 0.0
    nlpd_sum = 0.0
    picp_hits = 0
    n_total = 0
    alpha = 1.0 - interval_level

    for start in range(0, y.shape[0], chunk_size):
        end = min(start + chunk_size, y.shape[0])
        y_c = y[start:end]
        mu_c = mu[start:end]
        latent_std_c = latent_std[start:end]
        weights_c = weights[start:end]
        obs_scale_c = obs_scale[start:end]
        df_c = None if df is None else df[start:end]

        f_points = (
            mu_c[None, :]
            + np.sqrt(2.0) * latent_std_c[None, :] * gh_x[:, None]
        )
        if df_c is None:
            log_pdf = spstats.norm.logpdf(
                y_c[None, :, None],
                loc=f_points[:, :, None],
                scale=obs_scale_c[None, :, :],
            )
        else:
            log_pdf = spstats.t.logpdf(
                y_c[None, :, None],
                df=df_c[None, :, :],
                loc=f_points[:, :, None],
                scale=obs_scale_c[None, :, :],
            )
        log_r = spspecial.logsumexp(
            np.log(weights_c)[None, :, :] + log_pdf, axis=-1
        )
        log_pred = spspecial.logsumexp(log_w_gh[:, None] + log_r, axis=0)
        nlpd_sum += float((-log_pred).sum())

        samples = _regime_student_t_mixture_samples(
            mu_c,
            latent_std_c,
            weights_c,
            obs_scale_c,
            df_c,
            sample_size=sample_size,
            rng=rng,
        )
        crps_sum += float(_crps_from_samples(samples, y_c).sum())
        q = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        picp_hits += int(np.sum((y_c >= q[0]) & (y_c <= q[1])))
        n_total += end - start

    proper = _prob_metrics_dict(
        crps=float(crps_sum / max(1, n_total)),
        nlpd=float(nlpd_sum / max(1, n_total)),
        picp=float(picp_hits / max(1, n_total)),
    )
    if df is None:
        comp_var = obs_scale**2
    else:
        comp_var = (df / np.maximum(df - 2.0, 1e-8)) * obs_scale**2
    proxy_std = np.sqrt(latent_std**2 + np.sum(weights * comp_var, axis=-1))
    proxy_core = _compute_gaussian_metric_core(
        y, mu, proxy_std, interval_level=interval_level
    )
    gauss_proxy = _prob_metrics_dict(
        crps=proxy_core["crps"],
        nlpd=proxy_core["nlpd"],
        picp=proxy_core["picp"],
    )
    return _compose_metric_views(point, proper, gauss_proxy)
