import torch

# ========================== Public Release Defaults =============================================
CONFIGS = {
    # Data
    "file": "dataset/forecasting/ETTh1.csv",
    # --- NEW TRANSFORM OPTIONS ---
    "input_transform": "none",  # Options: "none", "log_returns", "returns"
    "target_transform": "none",  # Options: "none", "log_returns", "returns"
    "add_volatility_feature": False,  # append rolling-std channel per input col
    "volatility_window": 24,  # lookback window for rolling std (steps)
    # --- MODULAR ENCODER CONFIG ---
    "encoder_type": "patchtst",  # "patchtst" | "dlinear" | "timemixer"
    "encoder_args": {
        # PatchTST-specific args
        "patch_len": 16,
        "stride": 8,
        # Shared Transformer args
        "n_layers": 3,
        "n_head": 4,
        # "d_ff": 256,
        "d_ff_mult_hidden": 2,
        # DLinear-specific args
        "dlinear_kernel_size": 25,  # Must be odd
        # "forecast_tokens": DLinear-style full-lookback seasonal/trend
        # temporal maps into pred_len hidden tokens (recommended).
        # "decomp_tokens": old pointwise [seasonal, trend] -> hidden adapter.
        "dlinear_output_mode": "forecast_tokens",
        "dlinear_individual": False,  # True: per-channel temporal linears
        # TimeMixer-specific args
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "down_sampling_method": "avg",
        "moving_avg_kernel": 25,
        "d_ff_mult_hidden": 2,
    },
    # The public release uses positional marks only; they are not fed through
    # FiLM/concat conditioning paths.
    "date_col": "date",
    "series_id_col": "cols",
    "value_col": "data",
    "freq": "h",
    # Splits
    "test_ratio": 0.2,
    "valid_ratio": 0.2,
    # Sequence windows. gp_label_len is retained for dataloader/window
    # compatibility and for non-flatten decoder paths. In the final
    # sequence_flatten settings, the GP residual is trained and evaluated on
    # direct future features, so the old pre-horizon GP fitting role is inactive.
    "seq_len": 336,
    "gp_label_len": 48,
    "pred_len": 24,
    # Batching
    "batch_size": 512,
    "gradient_accumulation_steps": 1,  # 0 = auto-detect largest micro-batch that fits
    "eval_batch_size": 512,  # test/eval batch size
    "num_workers": 0,
    "train_drop_last": True,
    "enable_input_aggregation": False,  # don't collapse inputs
    "enable_target_aggregation": False,  # don't collapse targets
    "input_aggregation": None,  # explicit None
    "target_aggregation": None,  # explicit None (means: use all columns)
    # Target / Input aggregation
    # "target_aggregation": "sum",  # "sum" | "mean" | <col>
    # "enable_input_aggregation": True,
    # "input_aggregation": "sum",  # "sum" | "mean" | <col>
    # Model core
    "num_regimes": 4,  # used only for legacy softmax; SB uses Rmax
    "Rmax": 16,  # truncation for stick-breaking (max regimes)
    "expert_hidden_dim": 128,
    "gating_hidden_dim": 128,
    "mean_hidden_dim": 128,
    "horizon_mlp_hidden_dim": 128,
    "gp_input_dim": 4,
    "dropout_rate": 0.1,
    "num_inducing_points": 512,
    # Inducing point initialization
    "inducing_init_method": "kmeans",  # "kmeans" | "random"
    "inducing_warm_batches": 8,  # how many early batches to pool
    "inducing_pool_multiplier": 6,  # pool ≈ multiplier * num_inducing_points
    "kmeans_n_init": 10,
    "kmeans_max_iter": 100,
    # Gating selection
    "gating_method": "stick_breaking",  # "stick_breaking" | "softmax"
    "sb_mode": "renorm",  # "renorm" | "residual"
    "remainder_epsilon": 1e-2,  # threshold for R_eff (updated)
    "sb_alpha_init": 2.0,
    "sb_alpha_final": 0.9,
    # Training
    "training_iterations": 1000,  # updated
    "patience": 50,
    "min_epochs": 50,
    "checkpoint_every": 5,
    "lr": 1e-4,
    "lr_kernel": 1e-4,
    # Optional GP inducing-point LR. None preserves the historical behaviour:
    # inducing locations train with the main model LR, while kernel/likelihood
    # parameters use lr_kernel.
    "lr_inducing": 1e-4,
    # "noise_initial": 1e-1,  # initial homoskedastic noise # OBSOLETE
    "regime_noise_init": 0.5,  # None is no per-regime nugget
    "p_mode": "fixed",  # "fixed" | "nn"
    "p_fixed": 1.0,  # used if p_mode="fixed"
    "use_kernel_heteroskedastic_noise": True,  # keep v(t) inside the kernel
    "likelihood_noise_floor": 1e-4,  # tiny floor only
    # "tau_lower": 1e-4, # job replaced
    # kernel init
    "rbf_ls_range": [0.5, 5.0],
    "rbf_os_range": [0.5, 1.5],
    "rbf_ls_isotropic": True,
    "rbf_init": "rand_loguniform",  # "fixed" | "logspace" | "rand_loguniform" | "empirical"
    "rbf_empirical_jitter": [1.0, 1.0],  # multiplicative jitter for empirical init
    "rbf_randomize_order": True,
    "rbf_use_priors": False,
    "rbf_ls_prior_logstd": 0.35,
    "rbf_os_prior_logstd": 0.35,
    # Annealing (SB-related)
    "anneal_steps": 50,  # epochs to reach final values (linear ramp)
    # λ_SB (Beta penalty) ramp up
    "sb_lambda_init": 1e-6,
    "sb_lambda_final": 1e-6,
    "use_dp_reg": False,
    "lambda_dp": 0.0,
    # Gate regularizers used for the paper schedules. Keep these exposed:
    # point entropy sharpens per-location assignments; batch entropy keeps
    # multiple regimes active early; sb_lambda applies the stick-breaking
    # simplex penalty.
    "point_entropy_weight_init": 0.0,
    "point_entropy_weight_final": 0.0,
    "batch_entropy_weight_init": 3e-4,
    "batch_entropy_weight_final": 1e-6,
    # Init-time per-regime symmetry breaking on observation-likelihood params.
    #   0.0  -> identical across regimes (current/legacy behaviour, exact bit-compat)
    #   >0.0 -> log-space std of multiplicative jitter applied at init
    # Example: 0.5 gives ~exp(±0.5) ≈ 0.6×–1.6× spread across regimes.
    "regime_tau_init_jitter": 0.5,
    "regime_df_init_jitter": 0.3,
    # Ablation: tie observation-variance / Student-t df across regimes (kept
    # learnable per channel via c_d). Freezes raw_log_tau_main at zeros and
    # raw_df_regime at its init value, so v_d(t) is regime-invariant.
    "tie_regime_likelihood_params": False,
    # Gate temperature schedule.
    "anneal_start_temp": 1.0,
    "anneal_end_temp": 0.2,
    "anneal_epochs": 50,
    # Plotting
    "test_segment_plot_len": 10000,
    # "test_segment_plot_len": 7 * 24,
    "plot_multi_horizons": [1, 3, 4, 6, 8, 12, 24],  # e.g., [1, 3, 6, 24, 48]
    # ---------------- new for single-encoder MLE ----------------
    "model_type": "regime_gp",  # "regime_gp" | "rq_gp" | "single_encoder_mle" | "single_encoder_quantile" | "single_encoder_mdn"
    # ---- Mixture Density Network baseline (model_type=single_encoder_mdn) ----
    # Number of mixture components in the flat MDN head. Each component is
    # Gaussian if use_student_t_likelihood=False, Student-t otherwise.
    # Mixture weights are produced by a softmax over K logits; per-component
    # (mu, sigma, [nu]) are produced by a single Linear head over fut_dim.
    "mdn_num_components": 4,
    "use_mixed_linear": False,
    # "use_deep_mean": True, # <--- Add this flag
    "deep_mean_mode": "two_stream",  # Options: "none", "global", "regime", "both", OR "two_stream"
    "use_revin": True,
    "use_student_t_likelihood": True,
    "student_t_df_init": 10.0,
    "student_t_learn_df": True,
    "student_t_df_min": 4.0,
    "student_t_df_max": 100.0,
    # If True, Student-t DeRegime scores a proper mixture over regimes:
    #   log sum_r pi_r(x,h) StudentT(y | f(x,h), sigma_r, nu_r).
    # Default False preserves the historical path, which first blends
    # regime scale/df into a single local Student-t distribution.
    "student_t_regime_mixture_likelihood": True,
    # If True with use_student_t_likelihood=False, Gaussian DeRegime scores a
    # proper finite mixture over regimes instead of moment-matching regime
    # variance into one Gaussian. Default False preserves the historical
    # Gaussian heteroskedastic path.
    "gaussian_regime_mixture_likelihood": False,
    # Optional residual observation-variance calibration head for DeRegime.
    # Adds v_res(h,d) to the regime likelihood observation variance before
    # converting to Gaussian/Student-t scale. Disabled by default.
    "use_residual_observation_variance": True,
    "residual_observation_variance_source": "gp_features",  # "deep_mean" | "gp_features"
    "residual_observation_scale_init": 0.03,
    "residual_observation_scale_floor": 1e-6,
    "residual_observation_variance_cap": None,
    "residual_observation_variance_penalty": 0.0,
    # MLE-style deterministic mean backbone. In the paper settings only its
    # mean is used; scale and degrees of freedom stay in the DeRegime head.
    "use_residual_mle_backbone": True,
    "residual_mle_backbone_use_mean": True,
    "residual_mle_backbone_replace_deep_mean": True,
    "student_t_use_quadrature": True,
    "student_t_gh_points": 20,
    "student_t_mc_samples": 8,
    "student_t_eval_samples": 128,
    "student_t_metric_chunk_size": 4096,
    "student_t_interval_level": 0.95,
    "single_kernel_mode": False,
    # GP residual kernel gate usage:
    #   "regime" = full DeRegiME K_mix = sum_r pi_r pi'_r K_r(z_r,z'_r)
    #   "shared" = NoKernelGate ablation: keep learned regime gates/likelihood,
    #              but score residual similarity with one shared kernel over the
    #              average expert feature block.
    "kernel_gate_mode": "regime",
    "share_expert_encoder": True,  # True: one shared encoder + R projection heads
    "expert_head_hidden_dim": 0,  # >0: per-regime MLP head (Linear->GELU->Linear) instead of linear
    "unified_pred_features": False,
    "future_decoder_type": "sequence_flatten",  # "sequence_flatten" for DeRegime; "linear" for simple heads
    "quantiles": [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975],
    "sigma_floor": 1e-6,
    "sigma_cap": None,  # or a float (in scaled space); None disables
    # "var_temp_calibrate": False,           # we're NOT using this now
    # "use_student_t": False,                # keep Gaussian NLL
    # "student_t_nu_init": 6.0,              # unused while use_student_t=False
    # Repro / device
    "plot_first_n_dims": 10,  # plot only the first N output channels; None = all
    "save_all_dims": True,  # always save all predictions/metrics even if we plot only N
    "seeds": [42, 123, 456],
    "seed_aggregate_name": "seed_aggregate",  # file stem for aggregated CSV/JSON
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
