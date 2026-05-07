import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

from .config import ConfigDict
from .encoders import get_encoder, MarkAggregator
from .gp import (
    RegimeMixingKernel,
    HeteroskedasticNoiseLikelihood,
    RegimeStudentTLikelihood,
    GatedConstMean,
    _GPModel,
    stick_break_from_logits,
    vsb_gates_from_ab,
    _mean_over_time_batch,
)
from .modules import (
    HorizonMLPEncoder,
    PELinearProjection,
    DirectMLPProjection,
    SequenceFlattenProjection,
    BasisLinearProjection,
    BasisMLPProjection,
    RevIN,
    IdentityRevIN,
)


_SEQUENCE_FLATTEN_DECODERS = {
    "sequence_flatten",
    "seq_flatten",
    "flatten_sequence",
    "patchtst_flatten",
    "patch_flatten",
}


def _is_sequence_flatten_decoder(decoder_type) -> bool:
    return str(decoder_type or "").lower() in _SEQUENCE_FLATTEN_DECODERS


def _canonical_future_decoder(decoder_type):
    decoder_type = str(decoder_type or "mlp").lower()
    return "sequence_flatten" if decoder_type in _SEQUENCE_FLATTEN_DECODERS else decoder_type


def _encoder_token_count(enc, config) -> int:
    # PatchTST exposes encoded patch count as num_patches. Encoders that emit
    # a non-input-length token grid (e.g. DLinear forecast-token mode) expose
    # num_tokens. Most other encoders emit one token per input step.
    return int(
        getattr(enc, "num_tokens", getattr(enc, "num_patches", int(config.seq_len)))
    )


def _make_future_branch(decoder_type, **kwargs):
    """Factory that returns a future-decoder module honouring the
    HorizonMLPEncoder API for the chosen `future_decoder_type`.

    Used by all sites that previously instantiated HorizonMLPEncoder directly
    so that adding a new future-decoder mode is a one-line edit.
    """
    decoder_type = _canonical_future_decoder(decoder_type)
    if decoder_type in ("basis_mlp", "mlp_basis"):
        return BasisMLPProjection(**kwargs)

    if decoder_type in ("basis_linear", "lowrank_linear"):
        return BasisLinearProjection(**kwargs)

    kwargs.pop("basis_rank", None)
    kwargs.pop("basis_type", None)
    kwargs.pop("learnable_basis", None)

    if decoder_type in ("direct_mlp", "ctx_mlp", "mlp_direct"):
        return DirectMLPProjection(**kwargs)

    if decoder_type == "pe_linear":
        return PELinearProjection(**kwargs)

    kwargs.pop("max_horizon", None)
    # default ("mlp") and any unknown -> MLP encoder.
    return HorizonMLPEncoder(**kwargs)


def _future_decoder_branch_roles(decoder_type):
    """Resolve composite future-decoder modes into branch-specific roles.

    Returns (gating_decoder, expert_decoder, deep_mean_decoder).
    """
    decoder_type = _canonical_future_decoder(decoder_type)
    if decoder_type in ("basis_mlp_dm_linear", "basis_mlp_trend_linear"):
        return "basis_mlp", "basis_mlp", "linear"
    if decoder_type in ("basis_linear_dm_linear", "basis_linear_trend_linear"):
        return "basis_linear", "basis_linear", "linear"
    if decoder_type in ("mlp_dm_linear", "mlp_trend_linear"):
        return "mlp", "mlp", "linear"
    if decoder_type in (
        "direct_mlp_dm_linear",
        "ctx_mlp_dm_linear",
        "mlp_direct_dm_linear",
    ):
        return "direct_mlp", "direct_mlp", "linear"
    if decoder_type in (
        "gate_mlp_expert_basis_dm_linear",
        "gatemlp_expertbasis_dmlinear",
    ):
        return "mlp", "basis_linear", "linear"
    if decoder_type in (
        "gate_mlp_expert_pe_dm_linear",
        "gatemlp_expertpe_dmlinear",
    ):
        return "mlp", "pe_linear", "linear"
    return decoder_type, decoder_type, decoder_type


def _future_decoder_roles(decoder_type):
    """Backward-compatible two-role view: regime branch and deep mean."""
    gating_decoder, expert_decoder, deep_mean_decoder = _future_decoder_branch_roles(
        decoder_type
    )
    if gating_decoder != expert_decoder:
        return f"gate:{gating_decoder}|expert:{expert_decoder}", deep_mean_decoder
    return expert_decoder, deep_mean_decoder


def _future_branch_extra_kwargs(config):
    return {"max_horizon": int(config.pred_len)}


class ProbabilisticRegimeForecaster(nn.Module):
    def __init__(
        self, series_dim: int, target_dim: int, tmark_dim: int, config: ConfigDict
    ):
        super().__init__()
        self.cfg = config
        self.series_dim = series_dim
        self.D_in = int(series_dim)  # <-- Input dimensions
        self.D = int(target_dim)  # <-- D is now Output dimensions
        self.tmark_dim = tmark_dim
        self.device = config.device
        self.gp_input_dim = config.gp_input_dim
        self.use_channel_embed = False
        self.chan_dim = 0
        self.channel_fusion_mode = "none"
        # Original per-expert feature size G
        self.G = int(self.gp_input_dim)
        self.unified_pred_features = bool(
            getattr(config, "unified_pred_features", False)
        )
        _fdt = config.get("future_decoder_type", "mlp")
        _fgdt, _fedt, _ = _future_decoder_branch_roles(_fdt)
        self._hi_flag = False
        self.expert_in_dim = self._compute_expert_in_dim(self.channel_fusion_mode)

        # does the encoder return per-channel rows (B*D, L, H)?
        def _per_channel(enc) -> bool:
            return getattr(enc, "outputs_per_channel", False)

        # --- SINGLE KERNEL MODE SWITCH ---
        self.single_kernel_mode = getattr(config, "single_kernel_mode", False)
        # if config.model_type == "rq_gp":
        #     self.single_kernel_mode = True
        self.sb_mode = config.get("sb_mode", "renorm")
        if self.single_kernel_mode:
            print("INFO: Running in single-kernel mode. Gating network is disabled.")
            self.R = 1
            self.gating_method = "none"  # Override to simplify downstream logic
            pass
        else:
            self.gating_method = config.gating_method
            self.sb_mode = config.sb_mode
            # Both stick-breaking variants ("stick_breaking", "vsb") use Rmax as the
            # effective regime count; "softmax" uses the legacy fixed num_regimes.
            self.R = (
                config.Rmax
                if self.gating_method in ("stick_breaking", "vsb")
                else config.num_regimes
            )

        self.gate_ln = nn.Identity()
        self.db_cov_weight = 0.0
        self.past_cal_mode = "none"
        self.past_in_dim = series_dim
        self.past_film_gating = None
        self.past_film_expert = None

        # --- MODULAR ENCODER INITIALIZATION ---
        EncoderClass = get_encoder(config.encoder_type)
        # Initialize RevIN
        # We use input_dim if we are norming inputs, or target_dim if distinct.
        # usually series_dim == target_dim for auto-regressive tasks.

        # ... (RevIN Init) ...
        self.use_revin = config.get("use_revin", True)
        gating_input_dim = self.past_in_dim  # + self.added_context_dim

        if not self.single_kernel_mode:
            self.gating_enc = EncoderClass(
                input_dim=gating_input_dim,
                output_dim=config.gating_hidden_dim,
                config=config,
                hidden_dim=config.gating_hidden_dim,
            )
            self.gating_head = nn.Linear(config.gating_hidden_dim, self.R * self.D)
        else:
            self.gating_enc = None
            self.gating_head = None

        self.g_enc_per_channel = (
            (not self.single_kernel_mode) and _per_channel(self.gating_enc)
            if self.gating_enc
            else False
        )
        self.share_expert_encoder = bool(
            config.get("share_expert_encoder", False)
        ) and not self.single_kernel_mode

        if self.share_expert_encoder:
            _shared_enc = EncoderClass(
                input_dim=self.past_in_dim,
                output_dim=config.expert_hidden_dim,
                config=config,
                hidden_dim=config.expert_hidden_dim,
            )
            self.expert_encs = nn.ModuleList([_shared_enc])
            self.e_enc_per_channel = [_per_channel(_shared_enc)] * self.R
            print(
                f"INFO: Sharing ONE expert encoder across {self.R} regimes "
                f"(hidden_dim={config.expert_hidden_dim})"
            )
        else:
            self.expert_encs = nn.ModuleList(
                [
                    EncoderClass(
                        input_dim=self.past_in_dim,
                        output_dim=config.expert_hidden_dim,
                        config=config,
                        hidden_dim=config.expert_hidden_dim,
                    )
                    for _ in range(self.R)
                ]
            )
            self.e_enc_per_channel = [_per_channel(e) for e in self.expert_encs]

        # Channel embeddings are not part of the release configuration.
        self.channel_fusion_mode = "none"
        self.expert_in_dim = self._compute_expert_in_dim(self.channel_fusion_mode)
        self._chan_proj_gating = None
        self._chan_proj_expert = None
        self._chan_proj_dm = None
        self.gating_feature_dim = config.gating_hidden_dim

        if self.use_revin:
            self.revin = RevIN(series_dim)
            self.stat_enc = None
            self.stat_film = None
        else:
            print("INFO: RevIN is DISABLED (Using Identity).")
            self.revin = IdentityRevIN(series_dim)
            self.stat_enc = None
            self.stat_film = None

        # heads: if per-channel, map H -> R (not H -> D*R), and H -> gp_in (not H -> D*gp_in)
        if not self.single_kernel_mode:
            if self.g_enc_per_channel:
                self.gating_head = nn.Linear(
                    self.gating_feature_dim, self.R
                )  # (B*D,L,H(+S)) -> (B*D,L,R)
            else:
                self.gating_head = nn.Linear(self.gating_feature_dim, self.R * self.D)
        # experts -- MLP head (nonlinear) or linear projection per regime
        _head_hid = int(config.get("expert_head_hidden_dim", 0))
        self._expert_head_type = "mlp" if _head_hid > 0 else "linear"

        def _make_expert_head(out_dim):
            if _head_hid > 0:
                return nn.Sequential(
                    nn.Linear(config.expert_hidden_dim, _head_hid),
                    nn.GELU(),
                    nn.Linear(_head_hid, out_dim),
                )
            return nn.Linear(config.expert_hidden_dim, out_dim)

        self.expert_heads = nn.ModuleList(
            [
                _make_expert_head(
                    self.gp_input_dim
                    if self.e_enc_per_channel[i]
                    else self.D * self.gp_input_dim
                )
                for i in range(self.R)
            ]
        )
        if _head_hid > 0:
            print(
                f"INFO: Using MLP expert heads "
                f"({config.expert_hidden_dim} -> {_head_hid} -> gp_input_dim)"
            )

        self.future_decoder_type = str(config.get("future_decoder_type", "mlp")).lower()
        (
            self.future_gating_decoder_type,
            self.future_expert_decoder_type,
            self.deep_mean_future_decoder_type,
        ) = _future_decoder_branch_roles(self.future_decoder_type)
        # Older code paths/logging refer to a single "regime" decoder. Keep the
        # alias when gate and expert branches share a type; otherwise make the
        # split explicit for debugging.
        if self.future_gating_decoder_type == self.future_expert_decoder_type:
            self.future_regime_decoder_type = self.future_expert_decoder_type
        else:
            self.future_regime_decoder_type = (
                f"gate:{self.future_gating_decoder_type}|"
                f"expert:{self.future_expert_decoder_type}"
            )

        gate_linear = self.future_gating_decoder_type == "linear"
        expert_linear = self.future_expert_decoder_type == "linear"
        gate_seq_flatten = _is_sequence_flatten_decoder(self.future_gating_decoder_type)
        expert_seq_flatten = _is_sequence_flatten_decoder(self.future_expert_decoder_type)
        if gate_linear != expert_linear:
            raise NotImplementedError(
                "Mixed linear/nonlinear gate-expert future decoders are not supported. "
                f"Got gate={self.future_gating_decoder_type}, "
                f"expert={self.future_expert_decoder_type}."
            )
        if gate_seq_flatten != expert_seq_flatten:
            raise NotImplementedError(
                "Mixed sequence-flatten/non-sequence-flatten gate-expert future "
                "decoders are not supported. "
                f"Got gate={self.future_gating_decoder_type}, "
                f"expert={self.future_expert_decoder_type}."
            )

        if gate_linear and expert_linear:
            self._setup_future_linear(config)
        elif gate_seq_flatten and expert_seq_flatten:
            self._setup_future_sequence_flatten(config)
        elif not self.single_kernel_mode:
            self.future_gating_enc, self.future_expert_encs = (
                self.setup_future_encoders(config, tmark_dim)
            )
        else:
            self.future_gating_enc = None
            e_out_dim = (
                self.gp_input_dim
                if self.e_enc_per_channel[0]
                else (self.D * self.gp_input_dim)
            )
            self.future_expert_encs = nn.ModuleList(
                [
                    _make_future_branch(
                        self.future_expert_decoder_type,
                        in_dim=tmark_dim,
                        out_dim=e_out_dim,
                        ctx_dim=config.expert_hidden_dim,
                        hidden=config.horizon_mlp_hidden_dim,
                        dropout=config.dropout_rate,
                        zero_init=False,
                        **_future_branch_extra_kwargs(config),
                    )
                ]
            )

        self.mark_agg = None

        gp_total_in = self.R + self.R * self.expert_in_dim

        # =====================================================================
        # TWO-STREAM ARCHITECTURE SETUP
        # =====================================================================
        self.dm_mode = config.get("deep_mean_mode", "none").lower()

        # Select Dimension: Default to expert_dim if mean_dim is missing
        self.mean_dim = config.get("mean_hidden_dim", config.expert_hidden_dim)

        # In single kernel mode, if not explicitly "two_stream", force legacy "regime"
        if (
            self.single_kernel_mode
            and self.dm_mode != "none"
            and self.dm_mode != "two_stream"
        ):
            self.dm_mode = "regime"

        self.deep_mean_enc = None
        self.deep_mean_future_enc = None
        self.deep_mean_head = None
        self.dm_enc_per_channel = False
        self.future_dm_proj = None
        self.future_dm_seq_proj = None
        self.use_residual_observation_variance = bool(
            config.get("use_residual_observation_variance", False)
        )
        self.residual_obs_var_source = str(
            config.get("residual_observation_variance_source", "deep_mean")
        ).lower()
        if self.residual_obs_var_source not in ("deep_mean", "gp_features"):
            raise ValueError(
                "residual_observation_variance_source must be 'deep_mean' "
                f"or 'gp_features', got {self.residual_obs_var_source!r}"
            )
        self.residual_obs_scale_floor = float(
            config.get("residual_observation_scale_floor", 1e-6)
        )
        scale_init = config.get("residual_observation_scale_init", None)
        if scale_init is None:
            var_init = float(config.get("residual_observation_variance_init", 9e-4))
            scale_init = math.sqrt(max(var_init, 0.0))
        self.residual_obs_scale_init = float(scale_init)
        self.residual_obs_var_cap = config.get(
            "residual_observation_variance_cap", None
        )
        self.residual_obs_var_head = None
        self.future_residual_obs_var_proj = None
        self.future_residual_obs_var_seq_proj = None
        self.residual_obs_var_future_enc = None
        self.residual_obs_var_gp_head = None
        self.dm_out_dim = None

        if self.dm_mode == "two_stream":
            print(f"INFO: Initializing Two-Stream Trend Encoder (Dim={self.mean_dim})")

            # 1. Stream 1 Encoder
            EncoderClass = get_encoder(config.encoder_type)
            self.deep_mean_enc = EncoderClass(
                input_dim=self.past_in_dim,
                output_dim=self.mean_dim,
                config=config,
                hidden_dim=self.mean_dim,
            )
            self.dm_enc_per_channel = _per_channel(self.deep_mean_enc)

            self._chan_proj_dm = None

            # 2. Past Projection Head
            dm_head_dim = 1 if self.dm_enc_per_channel else self.D
            self.deep_mean_head = nn.Linear(self.mean_dim, dm_head_dim)
            with torch.no_grad():
                self.deep_mean_head.weight.fill_(0.0)
                # self.deep_mean_head.weight.normal_(0.02)
                self.deep_mean_head.bias.fill_(0.0)

            # 3. Future Projection Encoder
            dm_out_dim = 1 if self.dm_enc_per_channel else self.D
            self.dm_out_dim = dm_out_dim
            if _is_sequence_flatten_decoder(self.deep_mean_future_decoder_type):
                dm_out_dim = 1 if self.dm_enc_per_channel else self.D
                self.future_dm_seq_proj = SequenceFlattenProjection(
                    token_count=_encoder_token_count(self.deep_mean_enc, config),
                    hidden_dim=self.mean_dim,
                    out_dim=dm_out_dim,
                    max_horizon=config.pred_len,
                    dropout=config.dropout_rate,
                    zero_init=True,
                )
            elif self.deep_mean_future_decoder_type == "linear":
                H = config.pred_len
                self.future_dm_proj = nn.Linear(self.mean_dim, H * dm_out_dim)
                with torch.no_grad():
                    self.future_dm_proj.weight.fill_(0.0)
                    self.future_dm_proj.bias.fill_(0.0)
            else:
                self.deep_mean_future_enc = _make_future_branch(
                    self.deep_mean_future_decoder_type,
                    in_dim=tmark_dim,
                    out_dim=dm_out_dim,
                    ctx_dim=self.mean_dim,
                    hidden=config.horizon_mlp_hidden_dim,
                    dropout=config.dropout_rate,
                    zero_init=True,
                    **_future_branch_extra_kwargs(config),
                )

            if (
                self.use_residual_observation_variance
                and self.residual_obs_var_source == "deep_mean"
            ):
                print(
                    "INFO: Residual observation-variance head uses the "
                    "two-stream deep-mean representation."
                )
                raw_bias = self._inverse_softplus_scalar(
                    max(
                        self.residual_obs_scale_init
                        - self.residual_obs_scale_floor,
                        1e-12,
                    )
                )
                self.residual_obs_var_head = nn.Linear(self.mean_dim, dm_head_dim)
                self._init_residual_obs_raw_module(self.residual_obs_var_head, raw_bias)

                if _is_sequence_flatten_decoder(self.deep_mean_future_decoder_type):
                    self.future_residual_obs_var_seq_proj = SequenceFlattenProjection(
                        token_count=_encoder_token_count(self.deep_mean_enc, config),
                        hidden_dim=self.mean_dim,
                        out_dim=dm_out_dim,
                        max_horizon=config.pred_len,
                        dropout=config.dropout_rate,
                        zero_init=False,
                    )
                    self._init_residual_obs_raw_module(
                        self.future_residual_obs_var_seq_proj, raw_bias
                    )
                elif self.deep_mean_future_decoder_type == "linear":
                    H = config.pred_len
                    self.future_residual_obs_var_proj = nn.Linear(
                        self.mean_dim, H * dm_out_dim
                    )
                    self._init_residual_obs_raw_module(
                        self.future_residual_obs_var_proj, raw_bias
                    )
                else:
                    self.residual_obs_var_future_enc = _make_future_branch(
                        self.deep_mean_future_decoder_type,
                        in_dim=tmark_dim,
                        out_dim=dm_out_dim,
                        ctx_dim=self.mean_dim,
                        hidden=config.horizon_mlp_hidden_dim,
                        dropout=config.dropout_rate,
                        zero_init=False,
                        **_future_branch_extra_kwargs(config),
                    )
                    self._init_residual_obs_raw_module(
                        self.residual_obs_var_future_enc, raw_bias
                    )

        # ---------------------------------------------------
        # LEGACY DEEP MEAN HEADS
        # ---------------------------------------------------
        self.deep_mean_head_global = None
        self.deep_mean_heads_regime = None

        if self.dm_mode != "two_stream":
            if self.dm_mode in ["global", "both"]:
                self.deep_mean_head_global = nn.Linear(gp_total_in, 1)
                with torch.no_grad():
                    self.deep_mean_head_global.weight.normal_(0.02)
                    self.deep_mean_head_global.bias.fill_(0.0)

            if self.dm_mode in ["regime", "both"]:
                self.deep_mean_heads_regime = nn.ModuleList(
                    [nn.Linear(self.expert_in_dim, 1) for _ in range(self.R)]
                )
                for head in self.deep_mean_heads_regime:
                    with torch.no_grad():
                        head.weight.data.normal_(0.02)
                        head.bias.data.fill_(0.0)

        if self.use_residual_observation_variance and (
            self.residual_obs_var_source == "gp_features"
            or self.residual_obs_var_head is None
        ):
            if self.residual_obs_var_source == "deep_mean":
                print(
                    "WARN: residual observation-variance source='deep_mean' "
                    "requires deep_mean_mode='two_stream'; falling back to "
                    "GP feature state."
                )
            else:
                print(
                    "INFO: Residual observation-variance head uses the GP "
                    "prediction feature state."
                )
            raw_bias = self._inverse_softplus_scalar(
                max(
                    self.residual_obs_scale_init - self.residual_obs_scale_floor,
                    1e-12,
                )
            )
            self.residual_obs_var_source = "gp_features"
            self.residual_obs_var_gp_head = nn.Linear(gp_total_in, 1)
            self._init_residual_obs_raw_module(
                self.residual_obs_var_gp_head, raw_bias
            )

        self.use_residual_mle_backbone = bool(
            config.get("use_residual_mle_backbone", False)
        )
        self.residual_mle_backbone_use_mean = bool(
            config.get("residual_mle_backbone_use_mean", True)
        )
        self.residual_mle_backbone_replace_deep_mean = bool(
            config.get("residual_mle_backbone_replace_deep_mean", True)
        )
        self.residual_mle_backbone = None
        if self.use_residual_mle_backbone:
            _, _, backbone_decoder = _future_decoder_branch_roles(
                self.future_decoder_type
            )
            backbone_cfg = ConfigDict(dict(config))
            backbone_cfg.model_type = "single_encoder_mle"
            backbone_cfg.future_decoder_type = backbone_decoder
            backbone_cfg.use_residual_mle_backbone = False
            self.residual_mle_backbone = SingleEncoderForecaster(
                series_dim, target_dim, tmark_dim, backbone_cfg
            )
            print(
                "INFO: Residual-MLE backbone ENABLED "
                f"(decoder={backbone_decoder!r}, "
                f"mean={self.residual_mle_backbone_use_mean}). "
                "DeRegime GP models residual mean around this base forecast."
            )

        inducing = torch.empty(
            config.num_inducing_points, gp_total_in, device=self.device
        )

        self.gp = _GPModel(
            inducing,
            GatedConstMean(self.R),
            RegimeMixingKernel(
                self.R,
                self.expert_in_dim,  # <<< important: pass G+C here
                expert_kernel_type="rq" if config.model_type == "rq_gp" else "rbf",
                phi_hidden=config.expert_hidden_dim,
                rbf_init=config.rbf_init,
                rbf_ls_range=tuple(config.rbf_ls_range),
                rbf_os_range=tuple(config.rbf_os_range),
                rbf_ls_isotropic=config.rbf_ls_isotropic,
                rbf_empirical_jitter=tuple(
                    config.get("rbf_empirical_jitter", [1.0, 1.0])
                ),
                rbf_randomize_order=config.rbf_randomize_order,
                rbf_use_priors=config.rbf_use_priors,
                rbf_ls_prior_logstd=config.rbf_ls_prior_logstd,
                rbf_os_prior_logstd=config.rbf_os_prior_logstd,
                p_mode=config.p_mode,
                p_fixed=config.p_fixed,
                use_mixed_linear=config.use_mixed_linear,
                regime_checkpointing=False,
                kernel_gate_mode=config.get("kernel_gate_mode", "regime"),
            ),
        )

        self.register_buffer("current_temp", torch.tensor(config.anneal_start_temp))

        self.sb_alpha_init = None
        # Both stick-breaking variants share the fixed alpha buffer / annealing pathway.
        if self.gating_method in ("stick_breaking", "vsb"):
            alpha0 = max(1e-8, float(config.sb_alpha_init))
            # --- Register as Buffer so it saves with State Dict ---
            self.register_buffer(
                "sb_alpha_fixed", torch.tensor(alpha0, device=self.device)
            )
            self.lambda_dp = config.lambda_dp
            self.lambda_alpha = 1e-6
            self.alpha_min = 1e-3
            self.mu_log_alpha = 0.0
            self.alpha_raw = None

        if self.gating_method == "stick_breaking":
            bias = -np.log(alpha0)
            self.gating_head_sb = torch.nn.Linear(self.R, self.R, bias=True)
            with torch.no_grad():
                self.gating_head_sb.weight.copy_(torch.eye(self.R))
                self.gating_head_sb.bias.fill_(bias)
        elif self.gating_method == "vsb":
            # Variational stick-breaking head: project the R-dim "logits" to the
            # two Kumaraswamy parameters per stick (a_k, b_k for k=1..R-1).
            self.vsb_a_min = float(config.get("vsb_a_min", 0.1))
            self.vsb_b_min = float(config.get("vsb_b_min", 0.1))
            self.vsb_kl_truncation = int(config.get("vsb_kl_truncation", 11))
            self.vsb_eval_use_mean = bool(config.get("vsb_eval_use_mean", True))
            self.gating_head_vsb = torch.nn.Linear(
                self.R, 2 * (self.R - 1), bias=True
            )
            with torch.no_grad():
                # Initialise q(v)=Kuma(a,b) independently of the sparse prior
                # when requested. By default, b follows alpha0.
                self.gating_head_vsb.weight.zero_()
                a_init = config.get("vsb_init_a", 1.0)
                b_init = config.get("vsb_init_b", None)
                a_target = 1.0 if a_init is None else float(a_init)
                b_target = alpha0 if b_init is None else float(b_init)
                a_target = max(a_target, self.vsb_a_min + 1e-2)
                b_target = max(b_target, self.vsb_b_min + 1e-2)

                def _inverse_softplus_offset(target, lo):
                    # Solve softplus(x) + lo = target  ⇒  x = log(expm1(target - lo)).
                    # Fall back to a small negative bias if target ≤ lo.
                    delta = max(target - lo, 1e-3)
                    return float(math.log(math.expm1(delta)))

                bias_a = _inverse_softplus_offset(a_target, self.vsb_a_min)
                bias_b = _inverse_softplus_offset(b_target, self.vsb_b_min)
                bias = self.gating_head_vsb.bias
                bias[: self.R - 1].fill_(bias_a)
                bias[self.R - 1 :].fill_(bias_b)

        self.fusion_method = "none"

        # Check if we SHOULD fuse.
        # 1. Necessity: Input != Output count (MISO/Bottleneck)
        # 2. Choice: User selected "attention" or "linear" explicitly
        self.use_fusion = (self.D != self.D_in) or (self.fusion_method != "none")

        # Containers
        self.g_fusion_layer = None
        self.expert_fusion_layers = nn.ModuleList()
        self.dropout_layer = nn.Dropout(config.dropout_rate)

        # --- FUSION SETUP ---
        if self.use_fusion:

            # Default to 'mean' if mismatch exists but 'none' was selected
            if self.fusion_method == "none" and self.D != self.D_in:
                self.fusion_method = "mean"

            # === METHOD A: CROSS-ATTENTION ===
            if self.fusion_method == "attention":
                # 1. Gating Identity
                self.g_input_pos = nn.Parameter(
                    torch.randn(1, self.D_in, config.gating_hidden_dim)
                )

                self.g_attn = nn.MultiheadAttention(
                    embed_dim=config.gating_hidden_dim,
                    num_heads=4,
                    batch_first=True,
                    dropout=config.dropout_rate,
                )

                # 2. Expert Identities
                for _ in range(self.R):
                    # Create container
                    expert_mod = nn.Module()

                    # Assign parameter as attribute (PyTorch registers it automatically)
                    expert_mod.k_pos = nn.Parameter(
                        torch.randn(1, self.D_in, config.expert_hidden_dim)
                    )

                    expert_mod.attn = nn.MultiheadAttention(
                        embed_dim=config.expert_hidden_dim,
                        num_heads=4,
                        batch_first=True,
                        dropout=config.dropout_rate,
                    )

                    # Append container to list
                    self.expert_fusion_layers.append(expert_mod)
                # # --- GATING SETUP ---
                # # 1. Queries (D_out): What we want to produce
                # self.g_query = nn.Parameter(torch.randn(1, self.D, config.gating_hidden_dim))

                # # 2. Input ID Embedding (D_in): Who the inputs are
                # # >>> NEW ADDITION <<<
                # self.g_input_pos = nn.Parameter(torch.randn(1, self.D_in, config.gating_hidden_dim))

                # self.g_attn = nn.MultiheadAttention(
                #     embed_dim=config.gating_hidden_dim,
                #     num_heads=4,
                #     batch_first=True,
                #     dropout=config.dropout_rate
                # )

                # # --- EXPERT SETUP ---
                # for _ in range(self.R):
                #     # Create a container module to hold parameters + layer
                #     expert_mod = nn.Module()

                #     # Assign parameters as attributes (PyTorch automatically registers them)
                #     expert_mod.q = nn.Parameter(torch.randn(1, self.D, config.expert_hidden_dim))
                #     expert_mod.k_pos = nn.Parameter(torch.randn(1, self.D_in, config.expert_hidden_dim))

                #     expert_mod.attn = nn.MultiheadAttention(
                #         embed_dim=config.expert_hidden_dim,
                #         num_heads=4,
                #         batch_first=True,
                #         dropout=config.dropout_rate
                #     )

                #     self.expert_fusion_layers.append(expert_mod)
            # === METHOD B: BOTTLENECK LINEAR ===
            elif self.fusion_method == "linear":
                raise NotImplementedError("Linear fusion disabled for now.")
                pass

    def setup_future_encoders(self, config, tmark_dim):
        zi = False
        # gating: R if per-channel; else D*R
        g_out = self.R if self.g_enc_per_channel else (self.D * self.R)
        future_gating_enc = _make_future_branch(
            self.future_gating_decoder_type,
            in_dim=tmark_dim,
            out_dim=g_out,
            ctx_dim=self.gating_feature_dim,
            hidden=config.horizon_mlp_hidden_dim,
            dropout=config.dropout_rate,
            zero_init=zi,
            **_future_branch_extra_kwargs(config),
        )
        future_expert_encs = nn.ModuleList()
        for i in range(self.R):
            # experts: gp_in if per-channel; else D*gp_in
            e_out = (
                self.gp_input_dim
                if self.e_enc_per_channel[i]
                else (self.D * self.gp_input_dim)
            )
            future_expert_encs.append(
                _make_future_branch(
                    self.future_expert_decoder_type,
                    in_dim=tmark_dim,
                    out_dim=e_out,
                    ctx_dim=config.expert_hidden_dim,
                    hidden=config.horizon_mlp_hidden_dim,
                    dropout=config.dropout_rate,
                    zero_init=zi,
                    **_future_branch_extra_kwargs(config),
                )
            )
        return future_gating_enc, future_expert_encs

    def _setup_future_linear(self, config):
        """Replace HorizonMLPEncoders with direct linear projections from ctx."""
        H = config.pred_len
        self.future_gating_enc = None
        self.future_expert_encs = None

        # Expert projections (one per regime, or one for single-kernel)
        n_experts = 1 if self.single_kernel_mode else self.R
        self.future_expert_projs = nn.ModuleList()
        for i in range(n_experts):
            idx = min(i, len(self.e_enc_per_channel) - 1)
            e_out = (
                self.gp_input_dim
                if self.e_enc_per_channel[idx]
                else (self.D * self.gp_input_dim)
            )
            self.future_expert_projs.append(
                nn.Linear(config.expert_hidden_dim, H * e_out)
            )

        # Gating projection (only for multi-regime)
        self.future_gating_proj = None
        if not self.single_kernel_mode:
            g_out = self.R if self.g_enc_per_channel else (self.D * self.R)
            self.future_gating_proj = nn.Linear(self.gating_feature_dim, H * g_out)

        self.future_dm_proj = None

    def _setup_future_sequence_flatten(self, config):
        """PatchTST-style direct projection from the full encoder token sequence."""
        self.future_gating_enc = None
        self.future_expert_encs = None
        self.future_gating_seq_proj = None
        self.future_expert_seq_projs = nn.ModuleList()

        if not self.single_kernel_mode:
            g_out = self.R if self.g_enc_per_channel else (self.D * self.R)
            self.future_gating_seq_proj = SequenceFlattenProjection(
                token_count=_encoder_token_count(self.gating_enc, config),
                hidden_dim=self.gating_feature_dim,
                out_dim=g_out,
                max_horizon=config.pred_len,
                dropout=config.dropout_rate,
                zero_init=False,
            )

        n_experts = 1 if self.single_kernel_mode else self.R
        for i in range(n_experts):
            idx = min(i, len(self.e_enc_per_channel) - 1)
            enc_idx = 0 if self.share_expert_encoder else idx
            e_out = (
                self.gp_input_dim
                if self.e_enc_per_channel[idx]
                else (self.D * self.gp_input_dim)
            )
            self.future_expert_seq_projs.append(
                SequenceFlattenProjection(
                    token_count=_encoder_token_count(self.expert_encs[enc_idx], config),
                    hidden_dim=config.expert_hidden_dim,
                    out_dim=e_out,
                    max_horizon=config.pred_len,
                    dropout=config.dropout_rate,
                    zero_init=False,
                )
            )

        self.future_gating_proj = None
        self.future_expert_projs = None
        self.future_dm_proj = None

    def set_alpha(
        self, epoch: int, total_epochs: int, start_alpha: float, end_alpha: float
    ):
        """
        Anneals alpha using an Exponential (Log) schedule.
        """
        # Guard: If we are in single kernel mode (or not using SB), this buffer won't exist.
        if not hasattr(self, "sb_alpha_fixed"):
            return

        # Avoid division by zero or log(0)
        eps = 1e-6
        start_alpha = max(start_alpha, eps)
        end_alpha = max(end_alpha, eps)

        # Calculate progress (0.0 to 1.0)
        frac = min(1.0, epoch / total_epochs)

        # Exponential Decay Formula: y = start * (end/start)^x
        curr_alpha = start_alpha * ((end_alpha / start_alpha) ** frac)

        # Update the buffer
        self.sb_alpha_fixed.fill_(curr_alpha)

        # Sync alpha_raw if it exists (for logging/vis)
        if hasattr(self, "alpha_raw") and self.alpha_raw is not None:
            val_to_encode = curr_alpha - self.alpha_min
            new_raw = torch.log(torch.expm1(torch.tensor(max(1e-6, val_to_encode))))
            with torch.no_grad():
                self.alpha_raw.copy_(new_raw.to(self.device))

    def alpha_value(self):
        if hasattr(self, "sb_alpha_fixed"):
            return self.sb_alpha_fixed

        # Safe fallback so training loop doesn't crash in single-kernel mode.
        return torch.tensor(1.0, device=self.device)
        # if self.alpha_raw is None:
        #     return None
        # a = torch.nn.functional.softplus(self.alpha_raw) + self.alpha_eps
        # low, high = self.alpha_bounds
        # return torch.clamp(a, min=low, max=high)

    @staticmethod
    def _inverse_softplus_scalar(y: float) -> float:
        y = float(y)
        if y > 20.0:
            return y
        return math.log(math.expm1(max(y, 1e-12)))

    @staticmethod
    def _init_residual_obs_raw_module(module: nn.Module, raw_bias: float) -> None:
        """Initialize a residual-scale raw-output module near a constant value."""
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                module.weight.fill_(0.0)
                module.bias.fill_(float(raw_bias))
            elif isinstance(module, SequenceFlattenProjection):
                module.linear.weight.fill_(0.0)
                module.linear.bias.fill_(float(raw_bias))
            elif isinstance(module, DirectMLPProjection):
                final = module.net[-1]
                final.weight.fill_(0.0)
                final.bias.fill_(float(raw_bias))
            elif isinstance(module, PELinearProjection):
                module.linear.weight.fill_(0.0)
                module.linear.bias.fill_(float(raw_bias))
            elif isinstance(module, HorizonMLPEncoder):
                module.output_layer.weight.fill_(0.0)
                module.output_layer.bias.fill_(float(raw_bias))
                module.film_gamma.weight.fill_(0.0)
                module.film_gamma.bias.fill_(0.0)
                module.film_beta.weight.fill_(0.0)
                module.film_beta.bias.fill_(0.0)
            elif isinstance(module, BasisLinearProjection):
                module.coeff_proj.weight.fill_(0.0)
                module.coeff_proj.bias.fill_(float(raw_bias))
            elif isinstance(module, BasisMLPProjection):
                final = module.coeff_net[-1]
                final.weight.fill_(0.0)
                final.bias.fill_(float(raw_bias))

    def _scalar_future_out_to_bd_time(self, x: torch.Tensor) -> torch.Tensor:
        if self.dm_enc_per_channel:
            return x.squeeze(-1)
        return self._split_last_dim_by_D(x, block=1).squeeze(-1)

    def _residual_obs_var_from_raw(self, raw: torch.Tensor | None) -> torch.Tensor | None:
        if raw is None:
            return None
        scale = F.softplus(raw) + self.residual_obs_scale_floor
        var = scale.pow(2)
        if self.residual_obs_var_cap is not None:
            var = torch.clamp(var, max=float(self.residual_obs_var_cap))
        return var

    def _revin_output_scale_from_stats(
        self, batch_std: torch.Tensor
    ) -> torch.Tensor:
        scale = batch_std
        if getattr(self.revin, "affine", False):
            w = self.revin.affine_weight.view(1, 1, -1)
            eps2 = float(getattr(self.revin, "eps", 0.0)) ** 2
            scale = scale / (w + eps2).abs()
        return scale

    def _normalize_output_with_current_revin(
        self, y: torch.Tensor, batch_mean: torch.Tensor, batch_std: torch.Tensor
    ) -> torch.Tensor:
        if batch_mean.size(-1) != self.D:
            raise NotImplementedError(
                "use_residual_mle_backbone currently requires D_in == D so "
                "the MLE backbone forecast can be expressed in the same RevIN "
                f"space as DeRegime; got D_in={batch_mean.size(-1)} and D={self.D}."
            )
        del batch_std
        return self.revin._normalize(y)

    def _bhd_to_bd_time(self, x: torch.Tensor) -> torch.Tensor:
        B, H, D = x.shape
        if D != self.D:
            raise ValueError(f"Expected D={self.D}, got tensor shape={tuple(x.shape)}")
        return x.transpose(1, 2).reshape(B * D, H)

    def _residual_mle_backbone_prediction(
        self,
        seq_x: torch.Tensor,
        seq_x_mark: torch.Tensor,
        seq_y_mark: torch.Tensor,
        seq_mh_mark: torch.Tensor | None,
        gp_label_len: int,
        pred_len: int,
        batch_mean: torch.Tensor,
        batch_std: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if not self.use_residual_mle_backbone or self.residual_mle_backbone is None:
            return {}

        mu, _, _ = self.residual_mle_backbone(
            seq_x,
            seq_x_mark,
            seq_y_mark,
            seq_mh_mark,
            gp_label_len=gp_label_len,
            pred_len=pred_len,
        )

        out = {}
        if self.residual_mle_backbone_use_mean:
            mu_norm = self._normalize_output_with_current_revin(
                mu, batch_mean, batch_std
            )
            out["mean"] = self._bhd_to_bd_time(mu_norm)

        return out

    def _apply_residual_mle_backbone(
        self,
        *,
        seq_x: torch.Tensor,
        seq_x_mark: torch.Tensor,
        seq_y_mark: torch.Tensor,
        seq_mh_mark: torch.Tensor | None,
        gp_label_len: int,
        pred_len: int,
        batch_mean: torch.Tensor,
        batch_std: torch.Tensor,
        feats_fit: torch.Tensor,
        feats_pred: torch.Tensor,
        deep_mean_fit,
        deep_mean_pred,
        aux,
    ):
        bb = self._residual_mle_backbone_prediction(
            seq_x,
            seq_x_mark,
            seq_y_mark,
            seq_mh_mark,
            gp_label_len,
            pred_len,
            batch_mean,
            batch_std,
        )
        if not bb:
            return deep_mean_fit, deep_mean_pred, aux

        aux = dict(aux or {})
        if self.residual_mle_backbone_replace_deep_mean:
            deep_mean_fit = torch.zeros(
                feats_fit.shape[:-1], device=feats_fit.device, dtype=feats_fit.dtype
            )
            deep_mean_pred = torch.zeros(
                feats_pred.shape[:-1], device=feats_pred.device, dtype=feats_pred.dtype
            )
        else:
            if not torch.is_tensor(deep_mean_fit):
                deep_mean_fit = torch.zeros(
                    feats_fit.shape[:-1],
                    device=feats_fit.device,
                    dtype=feats_fit.dtype,
                )
            if not torch.is_tensor(deep_mean_pred):
                deep_mean_pred = torch.zeros(
                    feats_pred.shape[:-1],
                    device=feats_pred.device,
                    dtype=feats_pred.dtype,
                )

        if bb.get("mean") is not None:
            deep_mean_pred = deep_mean_pred + bb["mean"].to(
                device=feats_pred.device, dtype=feats_pred.dtype
            )
        if bb.get("obs_scale") is not None:
            aux["backbone_obs_scale_pred"] = bb["obs_scale"].to(
                device=feats_pred.device, dtype=feats_pred.dtype
            )
        if bb.get("df") is not None:
            aux["backbone_df_pred"] = bb["df"].to(
                device=feats_pred.device, dtype=feats_pred.dtype
            )

        return deep_mean_fit, deep_mean_pred, aux

    def dp_regularizer(self, z):  # z are the pre-sigmoid breaks [..., R-1]
        # log(1 - sigmoid(z)) = -softplus(z)
        log1m_v = -F.softplus(z)
        a = self.alpha_value()
        dp_term = (a - 1.0) * log1m_v.sum()
        # mild prior on log(alpha)
        loga = torch.log(a)
        prior_a = self.lambda_alpha * (loga - self.mu_log_alpha).pow(2)
        reg = -self.lambda_dp * dp_term + prior_a
        return reg

    # def clamp_alpha_(self):
    #     if self.alpha_raw is None: return
    #     with torch.no_grad():
    #         a = torch.nn.functional.softplus(self.alpha_raw) + self.alpha_eps
    #         low, high = self.alpha_bounds
    #         a = torch.clamp(a, min=low, max=high)
    #         self.alpha_raw.copy_(torch.log(torch.expm1(a - self.alpha_eps).clamp(min=1e-8)))

    def set_temperature(self, temp: float):
        self.current_temp = torch.tensor(temp, device=self.device)

    def _gates_from_logits(self, logits: torch.Tensor, compute_kl: bool = True):
        """Build gates from raw logits.

        Parameters
        ----------
        logits : torch.Tensor
            Pre-head logits of shape (..., R).
        compute_kl : bool
            VSB-only: when False, the closed-form KL is *not* computed.
            Use this on the discarded `g_fit` path so we don't pay for
            (and retain autograd activations for) a KL we never add to
            the loss. Stick-breaking and softmax paths ignore this flag.
        """
        if self.gating_method == "stick_breaking":
            logits = self.gating_head_sb(logits)  # (B,T,R) identity+bias as before

            # >>> CRITICAL: use only the first R-1 "breaks" for stick-breaking
            sb_logits = logits[..., : self.R - 1]  # (B,T,R-1)

            gates, aux = stick_break_from_logits(
                logits=sb_logits,
                temp=self.current_temp,
                sb_mode=self.sb_mode,
                dustbin_idx=None,
            )
            return gates, aux
        elif self.gating_method == "vsb":
            return self._gates_from_vsb(logits, compute_kl=compute_kl)
        else:
            g = torch.softmax(logits / torch.clamp(self.current_temp, min=1e-8), dim=-1)
            return g, None

    def _gates_from_vsb(self, logits: torch.Tensor, compute_kl: bool = True):
        """Variational stick-breaking gates with Kumaraswamy posteriors.

        Uses the same input layout as the SB path (logits of shape (..., R));
        projects them to 2*(R-1) Kumaraswamy parameters, samples sticks via
        the Kuma reparameterisation in train mode (or the analytical mean in
        eval mode if `vsb_eval_use_mean` is True), and stashes the closed-form
        KL[Kuma(a,b) || Beta(1, alpha)] in `aux["kl_elem"]` for the loss.
        """
        raw = self.gating_head_vsb(logits)  # (..., 2*(R-1))
        a_raw, b_raw = raw.chunk(2, dim=-1)
        a = F.softplus(a_raw) + self.vsb_a_min
        b = F.softplus(b_raw) + self.vsb_b_min

        # Eval-time switch: deterministic mean keeps gates stable for metrics
        # / plots; train-time sampling provides the standard VI Monte-Carlo
        # gradient estimator for the ELBO.
        sample = self.training or (not self.vsb_eval_use_mean)

        gates, aux = vsb_gates_from_ab(
            a=a,
            b=b,
            alpha_prior=self.alpha_value(),
            sample=sample,
            sb_mode=self.sb_mode,
            dustbin_idx=None,
            num_kl_terms=self.vsb_kl_truncation,
            compute_kl=compute_kl,
        )
        return gates, aux

    def _compute_expert_in_dim(self, mode: str) -> int:
        """Return GP kernel-input per-regime feature dimension for a given fusion mode.

        - "late": legacy behaviour, chan_dim is concatenated at the GP
           kernel-input stage, so expert_in_dim = G + chan_dim (+1 if horizon_idx).
        - "early": chan identity is fused into encoder outputs; the GP kernel
           input stays at G (+1 if horizon_idx), i.e. no extra chan_dim.
        - "both": early additive fusion AND late concat; expert_in_dim
           includes chan_dim like "late".
        - "none": no channel embedding (univariate or disabled).
        """
        C = self.chan_dim if (self.use_channel_embed and self.D > 1) else 0
        late_extra = C if mode in ("late", "both") else 0
        return self.G + late_extra + (1 if self._hi_flag else 0)

    def _append_channel_emb(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*D, T, G). If channel embeddings are enabled AND fusion mode is
        "late" or "both", append e_d (size C) to last dim.
        Returns: (B*D, T, G+C) in "late"/"both" mode, or x unchanged otherwise.
        """
        if not (self.use_channel_embed and self.D > 1):
            return x
        if self.channel_fusion_mode not in ("late", "both"):
            # "early" fuses channel identity into encoder outputs only; "none"
            # disables channel embeddings entirely — no late concat in either.
            return x
        BD, T, G = x.shape
        B = BD // self.D
        device = x.device
        # Build [0..D-1] for each batch and flatten to (B*D,)
        d_ids = (
            torch.arange(self.D, device=device).unsqueeze(0).repeat(B, 1).reshape(-1)
        )
        emb = self.channel_emb(d_ids)  # (B*D, C)
        emb = emb.unsqueeze(1).expand(BD, T, self.chan_dim)  # (B*D, T, C)
        return torch.cat([x, emb], dim=-1)

    def _chan_emb_per_channel_rows(self, BD: int, device) -> torch.Tensor:
        """Return channel embeddings of shape (B*D, chan_dim) laid out so row i
        corresponds to channel (i % D). Used for per-channel encoder outputs."""
        B = BD // self.D
        d_ids = (
            torch.arange(self.D, device=device).unsqueeze(0).repeat(B, 1).reshape(-1)
        )
        return self.channel_emb(d_ids)  # (B*D, C)

    def _fuse_chan_emb_seq(
        self, x: torch.Tensor, proj: "nn.Linear | None"
    ) -> torch.Tensor:
        """Additively fuse learned channel emb into a per-channel sequence (B*D, T, H).

        Active in "early" and "both" modes (when proj is not None).
        """
        if self.channel_fusion_mode not in ("early", "both") or proj is None:
            return x
        BD, T, H = x.shape
        emb = self._chan_emb_per_channel_rows(BD, x.device)  # (B*D, C)
        return x + proj(emb).unsqueeze(1)  # broadcast over T

    def _fuse_chan_emb_ctx(
        self, x: torch.Tensor, proj: "nn.Linear | None"
    ) -> torch.Tensor:
        """Additively fuse learned channel emb into a per-channel context (B*D, H).

        Active in "early" and "both" modes (when proj is not None).
        """
        if self.channel_fusion_mode not in ("early", "both") or proj is None:
            return x
        BD = x.shape[0]
        emb = self._chan_emb_per_channel_rows(BD, x.device)  # (B*D, C)
        return x + proj(emb)

    def _repeat_marks_if_per_channel(
        self, marks: torch.Tensor, per_channel: bool
    ) -> torch.Tensor:
        # marks: (B, H, M) -> (B*D, H, M) if per_channel=True
        if per_channel and marks is not None and marks.numel() > 0:
            return marks.repeat_interleave(self.D, dim=0)
        return marks

    def forward_features(
        self,
        seq_x,
        seq_x_mark,
        seq_y_mark,
        seq_mh_mark,
        gp_label_len: int,
        pred_len: int,
    ):
        # 1. RevIN: Calculate statistics and normalize the main input
        self.revin._get_statistics(seq_x)
        seq_x_norm = self.revin._normalize(seq_x)

        # Capture stats for return
        curr_mean = self.revin.mean.clone()
        curr_std = self.revin.stdev.clone()

        # # 2. Prepare RevIN Context (Scale and Level)
        # sigma_ctx = self.revin.stdev.expand(-1, seq_x.size(1), -1)
        # mu_ctx = self.revin.mean.expand(-1, seq_x.size(1), -1)

        # The release model feeds the normalized past directly to both gate
        # and expert encoders.
        past_gate = seq_x_norm
        past_for_experts = seq_x_norm

        # 4. Inject RevIN Context into Gating Input
        # --- MODIFIED: Conditional Context Injection ---
        # if self.use_revin:
        #     # Expand stats to match sequence length
        #     sigma_ctx = self.revin.stdev.expand(-1, seq_x.size(1), -1)
        #     mu_ctx = self.revin.mean.expand(-1, seq_x.size(1), -1)

        #     # Concatenate: [Data, Sigma, Mu]
        #     gating_input = torch.cat([past_gate, sigma_ctx, mu_ctx], dim=-1)
        # else:
        #     # Pure Data only. No constants.
        #     gating_input = past_gate

        # Now we only pass the normalized shape (past_gate).
        # We DO NOT concatenate sigma/mu here anymore.
        gating_input = past_gate

        # Helper: build FUTURE marks of length (pred_len-1)
        def _future_marks_predlen_minus_1(B_real: int, per_channel: bool):
            """
            Returns marks shaped:
                if per_channel False: (B_real, pred_len-1, tmark_dim)
                if per_channel True:  (B_real*D, pred_len-1, tmark_dim)
            """
            if pred_len <= 1:
                return None

            if seq_mh_mark is None or seq_mh_mark.numel() == 0:
                m = torch.zeros(
                    B_real,
                    pred_len - 1,
                    self.tmark_dim,
                    device=seq_x.device,
                    dtype=seq_x.dtype,
                )
            else:
                # expected: (B, pred_len-1, tmark_dim)
                m = seq_mh_mark

            return self._repeat_marks_if_per_channel(m, per_channel)

        _use_linear = (
            self.future_gating_decoder_type == "linear"
            and self.future_expert_decoder_type == "linear"
        )
        _use_seq_flatten = (
            _is_sequence_flatten_decoder(self.future_gating_decoder_type)
            and _is_sequence_flatten_decoder(self.future_expert_decoder_type)
        )
        _use_dm_linear = self.deep_mean_future_decoder_type == "linear"
        _use_dm_seq_flatten = _is_sequence_flatten_decoder(
            self.deep_mean_future_decoder_type
        )
        residual_obs_var_pred = None

        # --- SINGLE KERNEL MODE PATH ---
        if self.single_kernel_mode:
            enc = self.expert_encs[0]
            head = self.expert_heads[0]
            e_past_features, e_ctx = enc(past_for_experts)
            # Early channel-embedding fusion (no-op in non-"early" mode).
            e_past_features = self._fuse_chan_emb_seq(
                e_past_features, self._chan_proj_expert
            )
            e_ctx = self._fuse_chan_emb_ctx(e_ctx, self._chan_proj_expert)

            e_past = head(e_past_features)
            if not self.e_enc_per_channel[0]:
                e_past = self._split_last_dim_by_D(e_past, self.gp_input_dim)

            e_past = self._append_channel_emb(e_past)

            enc_len = e_past.size(1)
            fit_len = min(gp_label_len, enc_len)
            exp_fit_list = [e_past[:, -fit_len:, :]]

            if _use_seq_flatten:
                e_fut = self.future_expert_seq_projs[0](
                    e_past_features, horizon=pred_len
                )
                if not self.e_enc_per_channel[0]:
                    e_fut = self._split_last_dim_by_D(e_fut, self.gp_input_dim)
                e_fut = self._append_channel_emb(e_fut)
                exp_pred_list = [e_fut]
                # Sequence-flatten mode uses the past tokens only as a forecasting
                # head input. The GP itself is trained/evaluated on future points.
                exp_fit_list = [e_fut[:, :0, :]]
                fit_len = 0
            elif _use_linear:
                e_fut_flat = self.future_expert_projs[0](e_ctx)
                e_out = (
                    self.gp_input_dim
                    if self.e_enc_per_channel[0]
                    else (self.D * self.gp_input_dim)
                )
                e_fut = e_fut_flat.view(e_ctx.size(0), pred_len, e_out)
                if not self.e_enc_per_channel[0]:
                    e_fut = self._split_last_dim_by_D(e_fut, self.gp_input_dim)
                e_fut = self._append_channel_emb(e_fut)
                exp_pred_list = [e_fut]
            else:
                future_enc = self.future_expert_encs[0]
                h1_anchor = None
                exp_h1_list = [e_past[:, -1:, :]]

                _upf = self.unified_pred_features
                fut_horizon_len = pred_len if _upf else max(pred_len - 1, 0)

                if fut_horizon_len > 0:
                    B = seq_x.size(0)
                    if (
                        (self.past_cal_mode == "none")
                        or (seq_mh_mark is None)
                        or (seq_mh_mark.numel() == 0)
                    ):
                        marks_H = torch.zeros(
                            B,
                            fut_horizon_len,
                            self.tmark_dim,
                            device=seq_x.device,
                            dtype=seq_x.dtype,
                        )
                    else:
                        if _upf:
                            h1_mark = seq_y_mark[:, -1:, :]
                            if seq_mh_mark.size(1) == pred_len - 1:
                                marks_H = torch.cat(
                                    [h1_mark, seq_mh_mark], dim=1
                                )
                            elif seq_mh_mark.size(1) >= pred_len:
                                marks_H = seq_mh_mark[:, :pred_len, :]
                            else:
                                marks_H = torch.cat(
                                    [h1_mark, seq_mh_mark], dim=1
                                )
                        else:
                            if seq_mh_mark.size(1) == pred_len - 1:
                                marks_H = seq_mh_mark
                            elif seq_mh_mark.size(1) == pred_len:
                                marks_H = seq_mh_mark[:, 1:, :]
                            else:
                                marks_H = seq_mh_mark

                    marks_H = self._repeat_marks_if_per_channel(
                        marks_H, self.e_enc_per_channel[0]
                    )
                    e_fut = future_enc(marks_H, e_ctx)
                    if not self.e_enc_per_channel[0]:
                        e_fut = self._split_last_dim_by_D(
                            e_fut, self.gp_input_dim
                        )

                    if h1_anchor is not None:
                        e_fut = h1_anchor.expand_as(e_fut) + e_fut

                    e_fut = self._append_channel_emb(e_fut)
                    if _upf:
                        exp_pred_list = [e_fut]
                    else:
                        exp_pred_list = [
                            torch.cat([exp_h1_list[0], e_fut], dim=1)
                        ]
                else:
                    exp_pred_list = exp_h1_list

            e_past_fit = exp_fit_list[0]
            e_pred = exp_pred_list[0]

            g_fit = torch.ones(
                e_past_fit.shape[0],
                e_past_fit.shape[1],
                1,
                device=seq_x.device,
                dtype=seq_x.dtype,
            )
            g_pred = torch.ones(
                e_pred.shape[0],
                e_pred.shape[1],
                1,
                device=seq_x.device,
                dtype=seq_x.dtype,
            )
            feats_fit = e_past_fit
            feats_pred = e_pred

            # --- DEEP MEAN CALCULATION (SINGLE KERNEL) ---
            deep_mean_fit = 0.0
            deep_mean_pred = 0.0
            aux_out = None
            dm_marks = None

            # 1. Two-Stream Path
            if self.dm_mode == "two_stream":
                dm_seq, dm_ctx = self.deep_mean_enc(past_for_experts)
                # Early channel-embedding fusion (no-op in non-"early" mode).
                dm_seq = self._fuse_chan_emb_seq(dm_seq, self._chan_proj_dm)
                dm_ctx = self._fuse_chan_emb_ctx(dm_ctx, self._chan_proj_dm)

                # Detect mismatch: Encoder produced (B*Din) but we need (B*Dout)
                # This happens when using Channel-Independent encoders + Target Aggregation
                B_real = seq_x.size(0)
                if self.dm_enc_per_channel and (self.D_in != self.D):
                    # 1. Reshape from flattened (B*Din, ...) to (B, Din, ...)
                    dm_ctx = dm_ctx.view(B_real, self.D_in, -1)
                    dm_seq = dm_seq.view(B_real, self.D_in, dm_seq.shape[1], -1)

                    # 2. Mean Pool over Input Channels (Din) to get global trend context
                    dm_ctx = dm_ctx.mean(dim=1)  # (B, H_dim)
                    dm_seq = dm_seq.mean(dim=1)  # (B, L, H_dim)

                    # 3. If Multi-Output (D > 1), repeat to match B*D structure.
                    # If Single-Output (D == 1), (B, ...) is already correct.
                    if self.D > 1:
                        dm_ctx = dm_ctx.repeat_interleave(self.D, dim=0)
                        dm_seq = dm_seq.repeat_interleave(self.D, dim=0)

                dm_fit_out = self.deep_mean_head(
                    dm_seq
                )  # (B*D,L,1) if per-channel else (B,L,D)

                if self.dm_enc_per_channel:
                    deep_mean_fit = dm_fit_out.squeeze(-1)  # (B*D, L)
                else:
                    deep_mean_fit = self._split_last_dim_by_D(
                        dm_fit_out, block=1
                    ).squeeze(
                        -1
                    )  # (B*D, L)

                # CRITICAL: slice to fit_len to match feats_fit time axis.
                # `-0:` would mean the full sequence, so handle future-only mode.
                deep_mean_fit = (
                    deep_mean_fit[:, -fit_len:] if fit_len > 0 else deep_mean_fit[:, :0]
                )

                if _use_dm_seq_flatten and self.future_dm_seq_proj is not None:
                    dm_out = 1 if self.dm_enc_per_channel else self.D
                    dm_fut = self.future_dm_seq_proj(dm_seq, horizon=pred_len)
                    if self.dm_enc_per_channel:
                        deep_mean_pred = dm_fut.squeeze(-1)
                    else:
                        deep_mean_pred = self._split_last_dim_by_D(
                            dm_fut, block=1
                        ).squeeze(-1)
                elif _use_dm_linear and self.future_dm_proj is not None:
                    dm_out = 1 if self.dm_enc_per_channel else self.D
                    dm_fut_flat = self.future_dm_proj(dm_ctx)
                    dm_fut = dm_fut_flat.view(dm_ctx.size(0), pred_len, dm_out)
                    if self.dm_enc_per_channel:
                        deep_mean_pred = dm_fut.squeeze(-1)
                    else:
                        deep_mean_pred = self._split_last_dim_by_D(
                            dm_fut, block=1
                        ).squeeze(-1)
                else:
                    _upf_dm = self.unified_pred_features
                    dm_fut_len = pred_len if _upf_dm else max(pred_len - 1, 0)

                    if dm_fut_len > 0:
                        B_real = seq_x.size(0)
                        is_dm_expanded = dm_ctx.size(0) > B_real
                        if _upf_dm:
                            dm_marks = torch.zeros(
                                B_real,
                                dm_fut_len,
                                self.tmark_dim,
                                device=seq_x.device,
                                dtype=seq_x.dtype,
                            )
                            if (
                                self.past_cal_mode != "none"
                                and seq_mh_mark is not None
                                and seq_mh_mark.numel() > 0
                            ):
                                h1_mark = seq_y_mark[:, -1:, :]
                                if seq_mh_mark.size(1) == pred_len - 1:
                                    dm_marks = torch.cat(
                                        [h1_mark, seq_mh_mark], dim=1
                                    )
                                elif seq_mh_mark.size(1) >= pred_len:
                                    dm_marks = seq_mh_mark[:, :pred_len, :]
                                else:
                                    dm_marks = torch.cat(
                                        [h1_mark, seq_mh_mark], dim=1
                                    )
                            dm_marks = self._repeat_marks_if_per_channel(
                                dm_marks, is_dm_expanded
                            )
                        else:
                            dm_marks = _future_marks_predlen_minus_1(
                                B_real, per_channel=is_dm_expanded
                            )
                        dm_fut_out = self.deep_mean_future_enc(
                            dm_marks, dm_ctx
                        )

                        if self.dm_enc_per_channel:
                            dm_fut = dm_fut_out.squeeze(-1)
                        else:
                            dm_fut = self._split_last_dim_by_D(
                                dm_fut_out, block=1
                            ).squeeze(-1)

                        if _upf_dm:
                            deep_mean_pred = dm_fut
                        else:
                            deep_mean_pred = torch.cat(
                                [deep_mean_fit[:, -1:], dm_fut], dim=1
                            )
                    else:
                        deep_mean_pred = deep_mean_fit[:, -1:]

                if (
                    self.use_residual_observation_variance
                    and self.residual_obs_var_source == "deep_mean"
                ):
                    raw_resid = None
                    if (
                        _use_dm_seq_flatten
                        and self.future_residual_obs_var_seq_proj is not None
                    ):
                        raw_resid = self.future_residual_obs_var_seq_proj(
                            dm_seq, horizon=pred_len
                        )
                    elif (
                        _use_dm_linear
                        and self.future_residual_obs_var_proj is not None
                    ):
                        dm_out = 1 if self.dm_enc_per_channel else self.D
                        raw_flat = self.future_residual_obs_var_proj(dm_ctx)
                        raw_resid = raw_flat.view(dm_ctx.size(0), pred_len, dm_out)

                    if raw_resid is not None:
                        raw_resid = self._scalar_future_out_to_bd_time(raw_resid)
                        residual_obs_var_pred = self._residual_obs_var_from_raw(
                            raw_resid
                        )

            # 2. Legacy Path
            elif self.deep_mean_heads_regime is not None:
                head = self.deep_mean_heads_regime[0]
                deep_mean_fit = head(e_past_fit).squeeze(-1)
                deep_mean_pred = head(e_pred).squeeze(-1)

            # --- 2. Create Dummy Gates (Required for GP API only) ---
            g_fit = torch.ones(
                e_past_fit.shape[0],
                e_past_fit.shape[1],
                1,
                device=seq_x.device,
                dtype=seq_x.dtype,
            )
            g_pred = torch.ones(
                e_pred.shape[0],
                e_pred.shape[1],
                1,
                device=seq_x.device,
                dtype=seq_x.dtype,
            )

            # --- 3. Construct GP Features ---
            feats_fit = torch.cat([g_fit, e_past_fit], dim=-1)
            feats_pred = torch.cat([g_pred, e_pred], dim=-1)
            if (
                self.use_residual_observation_variance
                and self.residual_obs_var_source == "gp_features"
                and self.residual_obs_var_gp_head is not None
            ):
                residual_obs_var_pred = self._residual_obs_var_from_raw(
                    self.residual_obs_var_gp_head(feats_pred).squeeze(-1)
                )
            if self.use_residual_observation_variance:
                aux_out = dict(aux_out or {})
                aux_out["residual_obs_var_pred"] = residual_obs_var_pred
            deep_mean_fit, deep_mean_pred, aux_out = self._apply_residual_mle_backbone(
                seq_x=seq_x,
                seq_x_mark=seq_x_mark,
                seq_y_mark=seq_y_mark,
                seq_mh_mark=seq_mh_mark,
                gp_label_len=gp_label_len,
                pred_len=pred_len,
                batch_mean=curr_mean,
                batch_std=curr_std,
                feats_fit=feats_fit,
                feats_pred=feats_pred,
                deep_mean_fit=deep_mean_fit,
                deep_mean_pred=deep_mean_pred,
                aux=aux_out,
            )

            return (
                feats_fit,
                feats_pred,
                g_pred,
                aux_out,
                None,
                g_fit,
                curr_mean,
                curr_std,
                deep_mean_fit,
                deep_mean_pred,
            )

        # --- MULTI-KERNEL MODE PATH ---
        else:
            # Gating past logits
            g_past_features, g_ctx = self.gating_enc(gating_input)
            # Early channel-embedding fusion (no-op in non-"early" mode).
            g_past_features = self._fuse_chan_emb_seq(
                g_past_features, self._chan_proj_gating
            )
            g_ctx = self._fuse_chan_emb_ctx(g_ctx, self._chan_proj_gating)

            # --- NEW: Apply FiLM Modulation ---
            if self.use_revin and self.stat_enc is not None:

                # CASE A: Channel Independent (Local Context)
                if self.g_enc_per_channel:
                    # curr_mean is (B, 1, D).
                    # We need (B*D, 1, 1) so StatEncoder sees them as independent samples.

                    # 1. Transpose to (B, D, 1) and Flatten to (B*D, 1, 1)
                    mu_in = curr_mean.transpose(1, 2).reshape(-1, 1, 1)
                    std_in = curr_std.transpose(1, 2).reshape(-1, 1, 1)

                    # 2. Encode with weights shared across channels.
                    stat_emb = self.stat_enc(mu_in, std_in)  # Output: (B*D, 1, Emb)

                    # The batch is already B*D, aligned with g_past_features.

                # Case B: channel-dependent global context.
                else:
                    # curr_mean is (B, 1, D); StatEncoder concatenates mean/std.
                    stat_emb = self.stat_enc(curr_mean, curr_std)  # Output: (B, 1, Emb)

                    # No repeat is needed because alignment is on B.

                # Generate FiLM scale and shift.
                gamma, beta = self.stat_film(stat_emb)  # (B*D, 1, H)

                # C. Modulate Features
                # g_past_features: (B*D, L, H)
                g_past_features = g_past_features * (1.0 + gamma) + beta

                # >>> FIX: Modulate Context Vector (Future Regimes) <<<
                # g_ctx is (B*D, H). gamma/beta are (B*D, 1, H).
                # We squeeze gamma/beta to (B*D, H) to match g_ctx.
                gamma_ctx = gamma.squeeze(1)
                beta_ctx = beta.squeeze(1)

                g_ctx = g_ctx * (1.0 + gamma_ctx) + beta_ctx
                # >>> END FIX <<<
                # ----------------------------------

            # >>> FIX: Collapse Data/Mean/Std channels back to D channels <<<
            curr_dim = gating_input.shape[-1]
            tgt_dim = self.D_in
            if (
                self.g_enc_per_channel
                and (curr_dim > tgt_dim)
                and (curr_dim % tgt_dim == 0)
            ):
                ratio = curr_dim // tgt_dim
                B_real = seq_x.size(0)

                # 1. Fix Context
                H_dim = g_ctx.shape[-1]
                g_ctx = g_ctx.view(B_real, ratio, tgt_dim, H_dim).sum(dim=1)
                g_ctx = g_ctx.view(B_real * tgt_dim, H_dim)

                # 2. Fix Features
                Np = g_past_features.shape[1]
                g_past_features = g_past_features.view(
                    B_real, ratio, tgt_dim, Np, H_dim
                ).sum(dim=1)
                g_past_features = g_past_features.view(B_real * tgt_dim, Np, H_dim)
            # >>> END FIX <<<

            # Fusion Logic
            if self.g_enc_per_channel and self.use_fusion:
                B = g_past_features.size(0) // self.D_in
                Np, H = g_past_features.size(1), g_past_features.size(2)

                if self.fusion_method == "attention":
                    g_inputs = (
                        g_past_features.view(B, self.D_in, Np, H)
                        .permute(0, 2, 1, 3)
                        .reshape(B * Np, self.D_in, H)
                    )
                    g_inputs = g_inputs + self.g_input_pos
                    g_q = g_inputs[:, -self.D :, :]
                    g_agg, _ = self.g_attn(g_q, g_inputs, g_inputs)
                    g_past_features = (
                        g_agg.view(B, Np, self.D, H)
                        .permute(0, 2, 1, 3)
                        .reshape(B * self.D, Np, H)
                    )

                    # Context (Mean pool remains safest)
                    g_ctx = (
                        g_ctx.view(B, self.D_in, -1)
                        .mean(dim=1)
                        .repeat_interleave(self.D, dim=0)
                    )

                elif self.fusion_method == "linear":
                    raise NotImplementedError("Linear fusion disabled for now.")

                elif self.fusion_method == "mean":
                    g_agg = g_past_features.view(B, self.D_in, Np, H).mean(dim=1)
                    g_past_features = g_agg.repeat_interleave(self.D, dim=0)
                    g_ctx = (
                        g_ctx.view(B, self.D_in, -1)
                        .mean(dim=1)
                        .repeat_interleave(self.D, dim=0)
                    )

            g_past_logits = self.gating_head(g_past_features)
            if not self.g_enc_per_channel:
                g_past_logits = self._maybe_split_by_D(g_past_logits)

            # Experts (past)
            exp_fit_list, exp_h1_list, expert_contexts, expert_sequences = [], [], [], []
            exp_h1_anchors = []
            B = seq_x.size(0)

            _shared_e_seq = None
            _shared_e_ctx = None

            for i in range(self.R):
                if self.share_expert_encoder:
                    if _shared_e_seq is None:
                        _shared_e_seq, _shared_e_ctx = self.expert_encs[0](
                            past_for_experts
                        )
                        # Early channel-embedding fusion (no-op in non-"early" mode).
                        _shared_e_seq = self._fuse_chan_emb_seq(
                            _shared_e_seq, self._chan_proj_expert
                        )
                        _shared_e_ctx = self._fuse_chan_emb_ctx(
                            _shared_e_ctx, self._chan_proj_expert
                        )
                        _pc = self.e_enc_per_channel[0]
                        if _pc and self.use_fusion:
                            B_e = _shared_e_seq.size(0) // self.D_in
                            Np, H = _shared_e_seq.size(1), _shared_e_seq.size(2)
                            if self.fusion_method == "attention":
                                e_inputs = (
                                    _shared_e_seq.view(B_e, self.D_in, Np, H)
                                    .permute(0, 2, 1, 3)
                                    .reshape(B_e * Np, self.D_in, H)
                                )
                                expert_mod = self.expert_fusion_layers[0]
                                e_inputs = e_inputs + expert_mod.k_pos
                                e_q = e_inputs[:, -self.D :, :]
                                e_agg, _ = expert_mod.attn(
                                    e_q, e_inputs, e_inputs
                                )
                                _shared_e_seq = (
                                    e_agg.view(B_e, Np, self.D, H)
                                    .permute(0, 2, 1, 3)
                                    .reshape(B_e * self.D, Np, H)
                                )
                                _shared_e_ctx = (
                                    _shared_e_ctx.view(B_e, self.D_in, -1)
                                    .mean(dim=1)
                                    .repeat_interleave(self.D, dim=0)
                                )
                            elif self.fusion_method == "mean":
                                e_agg = _shared_e_seq.view(
                                    B_e, self.D_in, Np, H
                                ).mean(dim=1)
                                _shared_e_seq = e_agg.repeat_interleave(
                                    self.D, dim=0
                                )
                                _shared_e_ctx = (
                                    _shared_e_ctx.view(B_e, self.D_in, -1)
                                    .mean(dim=1)
                                    .repeat_interleave(self.D, dim=0)
                                )
                    e_seq = _shared_e_seq
                    e_ctx = _shared_e_ctx
                else:
                    enc = self.expert_encs[i]
                    e_seq, e_ctx = enc(past_for_experts)
                    # Early channel-embedding fusion (no-op in non-"early" mode).
                    e_seq = self._fuse_chan_emb_seq(
                        e_seq, self._chan_proj_expert
                    )
                    e_ctx = self._fuse_chan_emb_ctx(
                        e_ctx, self._chan_proj_expert
                    )

                    if self.e_enc_per_channel[i] and self.use_fusion:
                        B_e = e_seq.size(0) // self.D_in
                        Np, H = e_seq.size(1), e_seq.size(2)
                        if self.fusion_method == "attention":
                            e_inputs = (
                                e_seq.view(B_e, self.D_in, Np, H)
                                .permute(0, 2, 1, 3)
                                .reshape(B_e * Np, self.D_in, H)
                            )
                            expert_mod = self.expert_fusion_layers[i]
                            e_inputs = e_inputs + expert_mod.k_pos
                            e_q = e_inputs[:, -self.D :, :]
                            e_agg, _ = expert_mod.attn(
                                e_q, e_inputs, e_inputs
                            )
                            e_seq = (
                                e_agg.view(B_e, Np, self.D, H)
                                .permute(0, 2, 1, 3)
                                .reshape(B_e * self.D, Np, H)
                            )
                            e_ctx = (
                                e_ctx.view(B_e, self.D_in, -1)
                                .mean(dim=1)
                                .repeat_interleave(self.D, dim=0)
                            )
                        elif self.fusion_method == "mean":
                            e_agg = e_seq.view(B_e, self.D_in, Np, H).mean(
                                dim=1
                            )
                            e_seq = e_agg.repeat_interleave(self.D, dim=0)
                            e_ctx = (
                                e_ctx.view(B_e, self.D_in, -1)
                                .mean(dim=1)
                                .repeat_interleave(self.D, dim=0)
                            )

                e_past = self.expert_heads[i](e_seq)
                if not self.e_enc_per_channel[i]:
                    e_past = self._split_last_dim_by_D(e_past, self.gp_input_dim)

                e_past = self._append_channel_emb(e_past)

                enc_len = e_past.size(1)
                fit_len = min(gp_label_len, enc_len)
                exp_fit_list.append(e_past[:, -fit_len:, :])
                exp_h1_list.append(e_past[:, -1:, :])
                expert_contexts.append(e_ctx)
                expert_sequences.append(e_seq)

            # Gating slices
            enc_len = g_past_logits.size(1)
            fit_len = min(gp_label_len, enc_len)
            g_fit_logits = g_past_logits[:, -fit_len:, :]
            g_h1_logits = g_past_logits[:, -1:, :]

            # Future Paths
            if _use_seq_flatten:
                # --- PATCHTST-STYLE FLATTENED SEQUENCE HEAD ---
                # For PatchTST this consumes all encoded patch tokens; for other
                # encoders it consumes their full native sequence output.
                g_out = self.R if self.g_enc_per_channel else (self.D * self.R)
                g_fut_logits = self.future_gating_seq_proj(
                    g_past_features, horizon=pred_len
                )
                if not self.g_enc_per_channel:
                    g_fut_logits = self._maybe_split_by_D(g_fut_logits)
                g_pred_logits = g_fut_logits

                exp_pred_list = []
                for i, seq_proj in enumerate(self.future_expert_seq_projs):
                    e_fut = seq_proj(expert_sequences[i], horizon=pred_len)
                    if not self.e_enc_per_channel[i]:
                        e_fut = self._split_last_dim_by_D(e_fut, self.gp_input_dim)
                    e_fut = self._append_channel_emb(e_fut)
                    exp_pred_list.append(e_fut)

                exp_fit_list = [e[:, :0, :] for e in exp_pred_list]
                fit_len = 0
            elif _use_linear:
                # --- LINEAR PROJECTION PATH ---
                g_out = self.R if self.g_enc_per_channel else (self.D * self.R)
                g_fut_logits = self.future_gating_proj(g_ctx).view(
                    g_ctx.size(0), pred_len, g_out
                )
                if not self.g_enc_per_channel:
                    g_fut_logits = self._maybe_split_by_D(g_fut_logits)
                g_pred_logits = g_fut_logits

                exp_pred_list = []
                for i in range(self.R):
                    e_ctx = expert_contexts[i]
                    e_out = (
                        self.gp_input_dim
                        if self.e_enc_per_channel[i]
                        else (self.D * self.gp_input_dim)
                    )
                    e_fut = self.future_expert_projs[i](e_ctx).view(
                        e_ctx.size(0), pred_len, e_out
                    )
                    if not self.e_enc_per_channel[i]:
                        e_fut = self._split_last_dim_by_D(e_fut, self.gp_input_dim)
                    e_fut = self._append_channel_emb(e_fut)
                    exp_pred_list.append(e_fut)
            else:
                # --- MLP PATH (legacy) ---
                _upf = self.unified_pred_features
                _fra = False
                fut_horizon_len = pred_len if _upf else max(pred_len - 1, 0)

                if fut_horizon_len > 0:
                    B_real = seq_x.size(0)
                    is_g_expanded = g_ctx.size(0) > B_real

                    if self.past_cal_mode == "none":
                        dummy_m = torch.zeros(
                            B_real,
                            fut_horizon_len,
                            self.tmark_dim,
                            device=seq_x.device,
                            dtype=seq_x.dtype,
                        )
                        dm = self._repeat_marks_if_per_channel(dummy_m, is_g_expanded)
                        g_fut_logits = self.future_gating_enc(dm, g_ctx)
                        if not self.g_enc_per_channel:
                            g_fut_logits = self._maybe_split_by_D(g_fut_logits)
                        if _fra:
                            g_fut_logits = g_h1_logits.expand_as(g_fut_logits) + g_fut_logits

                        exp_fut_list = []
                        for i, enc_mlp in enumerate(self.future_expert_encs):
                            e_ctx = expert_contexts[i]
                            is_e_expanded = e_ctx.size(0) > B_real
                            dm_i = self._repeat_marks_if_per_channel(dummy_m, is_e_expanded)
                            e_fut = enc_mlp(dm_i, e_ctx)
                            if not self.e_enc_per_channel[i]:
                                e_fut = self._split_last_dim_by_D(e_fut, self.gp_input_dim)
                            if _fra:
                                e_fut = exp_h1_anchors[i].expand_as(e_fut) + e_fut
                            e_fut = self._append_channel_emb(e_fut)
                            exp_fut_list.append(e_fut)

                        if _upf:
                            g_pred_logits = g_fut_logits
                            exp_pred_list = exp_fut_list
                        else:
                            g_pred_logits = torch.cat([g_h1_logits, g_fut_logits], dim=1)
                            exp_pred_list = [
                                torch.cat([h1, fut], dim=1)
                                for h1, fut in zip(exp_h1_list, exp_fut_list)
                            ]
                    else:
                        if (seq_mh_mark is not None) and (seq_mh_mark.numel() > 0):
                            if _upf:
                                h1_mark = seq_y_mark[:, -1:, :]
                                if seq_mh_mark.size(1) == pred_len - 1:
                                    full_marks = torch.cat([h1_mark, seq_mh_mark], dim=1)
                                elif seq_mh_mark.size(1) >= pred_len:
                                    full_marks = seq_mh_mark[:, :pred_len, :]
                                else:
                                    full_marks = torch.cat([h1_mark, seq_mh_mark], dim=1)
                                mh_g = self._repeat_marks_if_per_channel(
                                    full_marks, is_g_expanded
                                )
                            else:
                                mh_g = self._repeat_marks_if_per_channel(
                                    seq_mh_mark, is_g_expanded
                                )
                            g_fut_logits = self.future_gating_enc(mh_g, g_ctx)
                            if not self.g_enc_per_channel:
                                g_fut_logits = self._maybe_split_by_D(g_fut_logits)
                            if _fra:
                                g_fut_logits = (
                                    g_h1_logits.expand_as(g_fut_logits) + g_fut_logits
                                )

                            exp_fut_list = []
                            for i, enc_mlp in enumerate(self.future_expert_encs):
                                e_ctx = expert_contexts[i]
                                if _upf:
                                    mh_e = self._repeat_marks_if_per_channel(
                                        full_marks, self.e_enc_per_channel[i]
                                    )
                                else:
                                    mh_e = self._repeat_marks_if_per_channel(
                                        seq_mh_mark, self.e_enc_per_channel[i]
                                    )
                                e_fut = enc_mlp(mh_e, e_ctx)
                                if not self.e_enc_per_channel[i]:
                                    e_fut = self._split_last_dim_by_D(
                                        e_fut, self.gp_input_dim
                                    )
                                if _fra:
                                    e_fut = exp_h1_anchors[i].expand_as(e_fut) + e_fut
                                e_fut = self._append_channel_emb(e_fut)
                                exp_fut_list.append(e_fut)

                            if _upf:
                                g_pred_logits = g_fut_logits
                                exp_pred_list = exp_fut_list
                            else:
                                g_pred_logits = torch.cat([g_h1_logits, g_fut_logits], dim=1)
                                exp_pred_list = [
                                    torch.cat([h1, fut], dim=1)
                                    for h1, fut in zip(exp_h1_list, exp_fut_list)
                                ]
                        else:
                            g_pred_logits = g_h1_logits
                            exp_pred_list = exp_h1_list
                else:
                    g_pred_logits = g_h1_logits
                    exp_pred_list = exp_h1_list

            # KL is consumed only on the g_pred path (the loss does not use
            # g_fit's posterior beyond producing the past-side gates), so we
            # skip its computation here to avoid retaining a second ~M-term
            # autograd graph on the past window. No-op for SB/softmax.
            g_pred, aux = self._gates_from_logits(g_pred_logits, compute_kl=True)
            if _use_seq_flatten:
                g_fit = g_pred[:, :0, :]
            else:
                g_fit, _ = self._gates_from_logits(g_fit_logits, compute_kl=False)

            feats_fit = torch.cat([g_fit] + exp_fit_list, dim=-1)
            feats_pred = torch.cat([g_pred] + exp_pred_list, dim=-1)
            # --- Calculate Deep Mean ---
            deep_mean_fit = 0.0
            deep_mean_pred = 0.0
            dm_marks = None

            # 1. Two-Stream Path
            if self.dm_mode == "two_stream":
                dm_seq, dm_ctx = self.deep_mean_enc(past_for_experts)
                # Early channel-embedding fusion (no-op in non-"early" mode).
                dm_seq = self._fuse_chan_emb_seq(dm_seq, self._chan_proj_dm)
                dm_ctx = self._fuse_chan_emb_ctx(dm_ctx, self._chan_proj_dm)
                # Detect Input/Output Mismatch (e.g. 21 inputs -> 1 target)
                B_real = seq_x.size(0)
                if self.dm_enc_per_channel and (self.D_in != self.D):
                    # 1. Reshape from flattened (B*Din, ...) -> (B, Din, ...)
                    dm_ctx = dm_ctx.view(B_real, self.D_in, -1)
                    dm_seq = dm_seq.view(B_real, self.D_in, dm_seq.shape[1], -1)

                    # 2. Mean Pool over Input Channels to get global trend context
                    dm_ctx = dm_ctx.mean(dim=1)  # (B, H_dim)
                    dm_seq = dm_seq.mean(dim=1)  # (B, L, H_dim)

                    # 3. If Multi-Output (D > 1), repeat to match B*D.
                    # If D=1 (your case), this is a no-op, keeping it (B, ...).
                    if self.D > 1:
                        dm_ctx = dm_ctx.repeat_interleave(self.D, dim=0)
                        dm_seq = dm_seq.repeat_interleave(self.D, dim=0)
                dm_fit_out = self.deep_mean_head(dm_seq)  # (B*D,L,1) or (B,L,D)

                if self.dm_enc_per_channel:
                    deep_mean_fit = dm_fit_out.squeeze(-1)  # (B*D, L)
                else:
                    deep_mean_fit = self._split_last_dim_by_D(
                        dm_fit_out, block=1
                    ).squeeze(
                        -1
                    )  # (B*D, L)

                deep_mean_fit = (
                    deep_mean_fit[:, -fit_len:] if fit_len > 0 else deep_mean_fit[:, :0]
                )  # CRITICAL: align with feats_fit

                if _use_dm_seq_flatten and self.future_dm_seq_proj is not None:
                    dm_fut = self.future_dm_seq_proj(dm_seq, horizon=pred_len)
                    if self.dm_enc_per_channel:
                        deep_mean_pred = dm_fut.squeeze(-1)
                    else:
                        deep_mean_pred = self._split_last_dim_by_D(
                            dm_fut, block=1
                        ).squeeze(-1)
                elif _use_dm_linear and self.future_dm_proj is not None:
                    dm_out = 1 if self.dm_enc_per_channel else self.D
                    dm_fut_flat = self.future_dm_proj(dm_ctx)
                    dm_fut = dm_fut_flat.view(dm_ctx.size(0), pred_len, dm_out)
                    if self.dm_enc_per_channel:
                        deep_mean_pred = dm_fut.squeeze(-1)
                    else:
                        deep_mean_pred = self._split_last_dim_by_D(
                            dm_fut, block=1
                        ).squeeze(-1)
                else:
                    _upf_dm = self.unified_pred_features
                    dm_fut_len = pred_len if _upf_dm else max(pred_len - 1, 0)

                    if dm_fut_len > 0:
                        is_dm_expanded = dm_ctx.size(0) > B_real
                        if _upf_dm:
                            dm_marks = torch.zeros(
                                B_real,
                                dm_fut_len,
                                self.tmark_dim,
                                device=seq_x.device,
                                dtype=seq_x.dtype,
                            )
                            if (
                                self.past_cal_mode != "none"
                                and seq_mh_mark is not None
                                and seq_mh_mark.numel() > 0
                            ):
                                h1_mark = seq_y_mark[:, -1:, :]
                                if seq_mh_mark.size(1) == pred_len - 1:
                                    dm_marks = torch.cat(
                                        [h1_mark, seq_mh_mark], dim=1
                                    )
                                elif seq_mh_mark.size(1) >= pred_len:
                                    dm_marks = seq_mh_mark[:, :pred_len, :]
                                else:
                                    dm_marks = torch.cat(
                                        [h1_mark, seq_mh_mark], dim=1
                                    )
                            dm_marks = self._repeat_marks_if_per_channel(
                                dm_marks, is_dm_expanded
                            )
                        else:
                            dm_marks = _future_marks_predlen_minus_1(
                                B_real, per_channel=is_dm_expanded
                            )
                        dm_fut_out = self.deep_mean_future_enc(dm_marks, dm_ctx)

                        if self.dm_enc_per_channel:
                            dm_fut = dm_fut_out.squeeze(-1)
                        else:
                            dm_fut = self._split_last_dim_by_D(
                                dm_fut_out, block=1
                            ).squeeze(-1)

                        if _upf_dm:
                            deep_mean_pred = dm_fut
                        else:
                            deep_mean_pred = torch.cat(
                                [deep_mean_fit[:, -1:], dm_fut], dim=1
                            )
                    else:
                        deep_mean_pred = deep_mean_fit[:, -1:]

                if (
                    self.use_residual_observation_variance
                    and self.residual_obs_var_source == "deep_mean"
                ):
                    raw_resid = None
                    if (
                        _use_dm_seq_flatten
                        and self.future_residual_obs_var_seq_proj is not None
                    ):
                        raw_resid = self.future_residual_obs_var_seq_proj(
                            dm_seq, horizon=pred_len
                        )
                    elif (
                        _use_dm_linear
                        and self.future_residual_obs_var_proj is not None
                    ):
                        dm_out = 1 if self.dm_enc_per_channel else self.D
                        raw_flat = self.future_residual_obs_var_proj(dm_ctx)
                        raw_resid = raw_flat.view(dm_ctx.size(0), pred_len, dm_out)
                    elif (
                        self.residual_obs_var_future_enc is not None
                        and dm_marks is not None
                    ):
                        raw_resid = self.residual_obs_var_future_enc(dm_marks, dm_ctx)

                    if raw_resid is not None:
                        raw_resid = self._scalar_future_out_to_bd_time(raw_resid)
                        residual_obs_var_pred = self._residual_obs_var_from_raw(
                            raw_resid
                        )

            # 2. Legacy Global Mean Path
            elif self.deep_mean_head_global is not None:
                deep_mean_fit = deep_mean_fit + self.deep_mean_head_global(
                    feats_fit
                ).squeeze(-1)
                deep_mean_pred = deep_mean_pred + self.deep_mean_head_global(
                    feats_pred
                ).squeeze(-1)

            # 3. Legacy Regime Means Path
            if self.deep_mean_heads_regime is not None and self.dm_mode != "two_stream":
                for r in range(self.R):
                    feat_r = exp_fit_list[r]
                    mu_r = self.deep_mean_heads_regime[r](feat_r).squeeze(-1)
                    gate_r = g_fit[..., r]
                    deep_mean_fit = deep_mean_fit + (gate_r * mu_r)

                for r in range(self.R):
                    feat_r = exp_pred_list[r]
                    mu_r = self.deep_mean_heads_regime[r](feat_r).squeeze(-1)
                    gate_r = g_pred[..., r]
                    deep_mean_pred = deep_mean_pred + (gate_r * mu_r)

            if (
                self.use_residual_observation_variance
                and self.residual_obs_var_source == "gp_features"
                and self.residual_obs_var_gp_head is not None
            ):
                residual_obs_var_pred = self._residual_obs_var_from_raw(
                    self.residual_obs_var_gp_head(feats_pred).squeeze(-1)
                )

            if self.use_residual_observation_variance:
                aux = dict(aux or {})
                aux["residual_obs_var_pred"] = residual_obs_var_pred
            deep_mean_fit, deep_mean_pred, aux = self._apply_residual_mle_backbone(
                seq_x=seq_x,
                seq_x_mark=seq_x_mark,
                seq_y_mark=seq_y_mark,
                seq_mh_mark=seq_mh_mark,
                gp_label_len=gp_label_len,
                pred_len=pred_len,
                batch_mean=curr_mean,
                batch_std=curr_std,
                feats_fit=feats_fit,
                feats_pred=feats_pred,
                deep_mean_fit=deep_mean_fit,
                deep_mean_pred=deep_mean_pred,
                aux=aux,
            )

            return (
                feats_fit,
                feats_pred,
                g_pred,
                aux,
                g_pred_logits,
                g_fit,
                curr_mean,
                curr_std,
                deep_mean_fit,
                deep_mean_pred,
            )

    def _split_last_dim_by_D(self, x: torch.Tensor, block: int) -> torch.Tensor:
        """
        x: (B, L, D*block) -> (B*D, L, block), per-output split for expert features.
        If D==1, or last dim != D*block, returns x unchanged with a warning (once).
        """
        if x.dim() != 3:
            raise ValueError(
                f"_split_last_dim_by_D expects 3D (B,L,·); got shape={tuple(x.shape)}"
            )
        if self.D <= 1:
            return x
        B, L, C = x.shape
        expected = self.D * block
        if C != expected:
            # Be tolerant: just return x if it's already per-D (e.g., block) or miswired.
            # You can switch this to an assert if you prefer hard failure:
            #   assert C == expected, f"..."
            # But a gentle warning is often nicer in long runs:
            if not hasattr(self, "_warned_split_mismatch"):
                print(
                    f"[WARN] _split_last_dim_by_D: expected last dim {expected} (=D*block), got {C}. Returning x unchanged."
                )
                self._warned_split_mismatch = True
            return x
        return (
            x.view(B, L, self.D, block)
            .permute(0, 2, 1, 3)
            .reshape(B * self.D, L, block)
        )

    def _maybe_split_by_D(self, x: torch.Tensor) -> torch.Tensor:
        """
        For gating logits shaped (B, L, D*R) -> (B*D, L, R).
        If D==1, or D does not divide last dim cleanly, returns x unchanged with a warning (once).
        """
        if x.dim() != 3:
            raise ValueError(
                f"_maybe_split_by_D expects 3D (B,L,·); got shape={tuple(x.shape)}"
            )
        if self.D <= 1:
            return x
        B, L, C = x.shape
        if C % self.D != 0:
            if not hasattr(self, "_warned_maybe_split_mismatch"):
                print(
                    f"[WARN] _maybe_split_by_D: last dim {C} not divisible by D={self.D}. Returning x unchanged."
                )
                self._warned_maybe_split_mismatch = True
            return x
        R = C // self.D
        return x.view(B, L, self.D, R).permute(0, 2, 1, 3).reshape(B * self.D, L, R)

    def gp_mvn(self, features):
        return self.gp(features)

    def initialize_inducing_from_batch(
        self,
        seq_x,
        seq_x_mark,
        seq_y_mark,
        seq_mh_mark,
        gp_label_len: int,
        pred_len: int,
        num_inducing: int,
    ):
        with torch.no_grad():
            feats_fit, feats_pred, _, _, _, _, _, _, _, _ = self.forward_features(
                seq_x, seq_x_mark, seq_y_mark, seq_mh_mark, gp_label_len, pred_len
            )
            feats_all = torch.cat([feats_fit, feats_pred], dim=1)
            flat = feats_all.reshape(-1, feats_all.size(-1))
            if str(getattr(self.gp.covar_module, "rbf_init", "")).lower() == "empirical":
                self.gp.covar_module.initialize_empirical_lengthscales_from_features(
                    flat
                )
            idx = torch.randperm(flat.size(0), device=flat.device)[:num_inducing]
            self.gp.variational_strategy.inducing_points.data.copy_(flat[idx])

    def initialize_inducing_from_loader(
        self,
        train_loader,
        gp_label_len: int,
        pred_len: int,
        num_inducing: int,
        pool_multiplier: int = 6,
        warm_batches: int = 8,
        device: str | torch.device | None = None,
        seed: int = 42,
    ):
        """
        Build a feature pool from the first few batches, run KMeans to choose inducing points.
        Falls back to random selection from the pool if KMeans is unavailable.
        """
        dev_in = device or self.gp.variational_strategy.inducing_points.device
        dev = torch.device(dev_in) if isinstance(dev_in, str) else dev_in
        dtype = self.gp.variational_strategy.inducing_points.dtype
        pool_list = []

        self.eval()
        with torch.no_grad():
            for bi, batch in enumerate(train_loader):
                if bi >= warm_batches:
                    break
                if len(batch) != 8:
                    raise NotImplementedError(
                        "Multi-horizon inputs required for inducing init."
                    )

                seq_x, seq_y, seq_x_mark, seq_y_mark, _time_idx, _mh_y, _mh_m, _ = batch
                seq_x = seq_x.to(dev)
                seq_y = seq_y.to(dev)
                seq_x_mark = seq_x_mark.to(dev)
                seq_y_mark = seq_y_mark.to(dev)
                _mh_m = (
                    _mh_m.to(dev)
                    if _mh_m is not None and hasattr(_mh_m, "to")
                    else None
                )

                (
                    feats_fit,
                    feats_pred,
                    _g_pred,
                    _,
                    _g_pred_logits,
                    _g_fit,
                    _,
                    _,
                    _,
                    _,
                ) = self.forward_features(
                    seq_x, seq_x_mark, seq_y_mark, _mh_m, gp_label_len, pred_len
                )

                feats_all = torch.cat([feats_fit, feats_pred], dim=1)  # (B, L-1+H, F)
                flat = (
                    feats_all.reshape(-1, feats_all.size(-1)).detach().cpu()
                )  # pool on CPU for sklearn
                pool_list.append(flat)

        if not pool_list:
            print(
                "Warning: empty pool for inducing init; falling back to random from a single batch."
            )
            # fallback to legacy path if no data collected
            return

        pool = torch.cat(pool_list, dim=0)  # (N_pool, F)
        pool_size_target = int(num_inducing * max(1, pool_multiplier))
        if pool_size_target < pool.size(0):
            idx = torch.randperm(pool.size(0))[:pool_size_target]
            pool = pool[idx]

        if str(getattr(self.gp.covar_module, "rbf_init", "")).lower() == "empirical":
            self.gp.covar_module.initialize_empirical_lengthscales_from_features(
                pool
            )

        # Try KMeans
        try:
            from sklearn.cluster import KMeans

            km = KMeans(
                n_clusters=num_inducing,
                n_init=(
                    int(getattr(self, "kmeans_n_init", 10))
                    if hasattr(self, "kmeans_n_init")
                    else 10
                ),
                max_iter=(
                    int(getattr(self, "kmeans_max_iter", 100))
                    if hasattr(self, "kmeans_max_iter")
                    else 100
                ),
                random_state=int(seed),
                verbose=0,
            )
            km.fit(pool.numpy())
            centers = torch.from_numpy(km.cluster_centers_)
            centers = centers.to(device=dev, dtype=dtype)
            self.gp.variational_strategy.inducing_points.data.copy_(centers)
            print(
                f"Inducing init (KMeans): pool={pool.shape[0]} → {num_inducing} centers."
            )
        except Exception as e:
            # Safe fallback: random subset from the pool
            print(f"KMeans inducing init failed ({e}); using random pool subset.")
            if pool.size(0) < num_inducing:
                # repeat some rows if pool is smaller than requested
                reps = (num_inducing + pool.size(0) - 1) // pool.size(0)
                pool = pool.repeat(reps, 1)
            ridx = torch.randperm(pool.size(0))[:num_inducing]
            centers = pool[ridx].to(device=dev, dtype=dtype)
            self.gp.variational_strategy.inducing_points.data.copy_(centers)

class SingleEncoderForecaster(nn.Module):
    def __init__(
        self, series_dim: int, target_dim: int, tmark_dim: int, config: ConfigDict
    ):
        super().__init__()
        self.cfg = config
        self.D_in = int(series_dim)
        self.D = int(target_dim)  # <-- D is now Output dimensions
        self.tmark_dim = tmark_dim
        self.device = config.device
        self.gp_input_dim = int(config.gp_input_dim)  # define this

        # Positional time marks are used by the data loader, but not passed to
        # the encoder via concat or FiLM paths in the release settings.
        self.past_cal_mode = "none"
        self.past_in_dim = series_dim

        # --- MODIFIED: RevIN Toggle ---
        self.use_revin = config.get("use_revin", True)

        if self.use_revin:
            self.revin = RevIN(series_dim)
        else:
            print("INFO: RevIN is DISABLED (Using Identity).")
            self.revin = IdentityRevIN(series_dim)
        # ------------------------------

        # Encoder
        EncoderClass = get_encoder(config.encoder_type)
        self.past_enc = EncoderClass(
            input_dim=self.past_in_dim,
            output_dim=config.expert_hidden_dim,
            config=config,
            hidden_dim=config.expert_hidden_dim,
        )
        self.enc_hidden = config.expert_hidden_dim

        # >>> ADD THIS LINE TO STORE THE ENCODER TYPE <<<
        self.enc_is_per_channel = getattr(self.past_enc, "outputs_per_channel", False)

        self.use_channel_embed = False
        self.chan_dim = 0
        self.channel_emb = None
        self._chan_proj = None

        # Future marks encoder -> ALWAYS produce (B, H, D*fut_dim); we'll just reshape
        self.fut_dim = self.gp_input_dim
        self.future_decoder_type = _canonical_future_decoder(
            config.get("future_decoder_type", "mlp")
        )
        self.use_student_t = getattr(config, "use_student_t_likelihood", False)

        # Determine the correct output dim for the future encoder
        if self.enc_is_per_channel:
            # If past_enc is per-channel, ctx is (B*D, H_enc).
            # We will call future_enc with (B*D, ...) inputs,
            # so it must only output G features per channel.
            future_out_dim = self.fut_dim  # G
        else:
            # If past_enc is batch-level, ctx is (B, H_enc).
            # We call future_enc with (B, ...) inputs,
            # so it must output D*G features.
            future_out_dim = self.D * self.fut_dim  # D * G

        self.future_param_seq_proj = None
        if _is_sequence_flatten_decoder(self.future_decoder_type):
            param_dim = 3 if self.use_student_t else 2
            seq_out_dim = param_dim if self.enc_is_per_channel else (self.D * param_dim)
            self.future_param_seq_proj = SequenceFlattenProjection(
                token_count=_encoder_token_count(self.past_enc, config),
                hidden_dim=self.enc_hidden,
                out_dim=seq_out_dim,
                max_horizon=config.pred_len,
                dropout=config.dropout_rate,
            )
            self.future_proj = None
            self.future_enc = None
        elif self.future_decoder_type == "linear":
            H = config.pred_len
            self.future_proj = nn.Linear(config.expert_hidden_dim, H * future_out_dim)
            self.future_enc = None
        else:
            self.future_enc = _make_future_branch(
                self.future_decoder_type,
                in_dim=tmark_dim,
                out_dim=future_out_dim,
                ctx_dim=config.expert_hidden_dim,
                hidden=config.horizon_mlp_hidden_dim,
                dropout=config.dropout_rate,
                **_future_branch_extra_kwargs(config),
            )
            self.future_proj = None
        # Heads over fut_dim (shared across D)
        self.mean_head = nn.Linear(self.fut_dim, 1)
        self.logvar_head = nn.Linear(self.fut_dim, 1)

        if self.use_student_t:
            self.nu_head = nn.Linear(self.fut_dim, 1)

        self.mark_agg = None

    @staticmethod
    def _softplus_floor(x, floor=1e-6, cap=None):
        s = F.softplus(x) + floor
        if cap is not None:
            s = torch.clamp(s, max=float(cap))
        return s

    def forward(
        self,
        seq_x,
        seq_x_mark,
        seq_y_mark,
        seq_mh_mark,
        gp_label_len: int,
        pred_len: int,
    ):
        # 1. Normalize Input
        self.revin._get_statistics(seq_x)
        seq_x = self.revin._normalize(seq_x)
        # Past
        past_in = seq_x
        past_seq, ctx = self.past_enc(past_in)

        # Future marks (H horizons)
        B = seq_x.size(0)
        if pred_len > 1:
            if (
                (self.past_cal_mode == "none")
                or (seq_mh_mark is None)
                or (seq_mh_mark.numel() == 0)
            ):
                marks_H = torch.zeros(
                    B, pred_len, self.tmark_dim, device=seq_x.device, dtype=seq_x.dtype
                )
            else:
                if seq_mh_mark.size(1) == pred_len - 1:
                    dec_start = (
                        seq_y_mark[:, -1:, :]
                        if (
                            self.past_cal_mode != "none"
                            and seq_y_mark is not None
                            and seq_y_mark.numel() > 0
                        )
                        else torch.zeros(
                            B, 1, self.tmark_dim, device=seq_x.device, dtype=seq_x.dtype
                        )
                    )
                    marks_H = torch.cat([dec_start, seq_mh_mark], dim=1)
                elif seq_mh_mark.size(1) == pred_len:
                    marks_H = seq_mh_mark
                else:
                    raise ValueError(
                        f"Expected future marks length {pred_len-1} or {pred_len}, got {seq_mh_mark.size(1)}"
                    )
        else:
            marks_H = (
                seq_y_mark[:, -1:, :]
                if (
                    self.past_cal_mode != "none"
                    and seq_y_mark is not None
                    and seq_y_mark.numel() > 0
                )
                # ... (logic that builds marks_H)
                else torch.zeros(
                    B, 1, self.tmark_dim, device=seq_x.device, dtype=seq_x.dtype
                )
            )

        # If the encoder (and thus ctx) is per-channel (B*D, ...),
        # we must repeat the batch-level future marks (B, ...)
        # to match the per-channel dimension.
        if self.enc_is_per_channel:
            # marks_H shape is (B, H, M) -> (64, 24, M)
            # We repeat it D times to get (B*D, H, M)
            marks_H = marks_H.repeat_interleave(self.D, dim=0)
            # marks_H shape is now (512, 24, M)

        # Future features / direct parameters
        if _is_sequence_flatten_decoder(self.future_decoder_type):
            params = self.future_param_seq_proj(past_seq, horizon=pred_len)
            param_dim = 3 if self.use_student_t else 2
            if self.enc_is_per_channel:
                BD, H, P = params.shape
                B_calc = BD // self.D
                params = params.view(B_calc, self.D, H, P).permute(0, 2, 1, 3)
            else:
                B_calc, H, _ = params.shape
                params = params.view(B_calc, H, self.D, param_dim)

            mu = params[..., 0]
            logv = params[..., 1]
            sigma = self._softplus_floor(
                logv,
                floor=float(self.cfg.sigma_floor),
                cap=(
                    float(self.cfg.sigma_cap)
                    if self.cfg.sigma_cap is not None
                    else None
                ),
            )
            mu = self.revin._denormalize(mu)

            scale = self.revin.stdev
            if getattr(self.revin, "affine", False):
                w = self.revin.affine_weight.view(1, 1, -1)
                eps2 = float(getattr(self.revin, "eps", 0.0)) ** 2
                scale = scale / (w + eps2).abs()
            sigma = sigma * scale

            if self.use_student_t:
                nu = 2.0 + F.softplus(params[..., 2])
                return mu, sigma, nu
            return mu, sigma, None

        if self.future_decoder_type == "linear":
            fut_feats = self.future_proj(ctx)
            if self.enc_is_per_channel:
                fut_feats = fut_feats.view(ctx.size(0), pred_len, self.fut_dim)
            else:
                fut_feats = fut_feats.view(ctx.size(0), pred_len, self.D * self.fut_dim)
        else:
            fut_feats = self.future_enc(marks_H, ctx)

        if self.enc_is_per_channel:
            # Input is (B*D, H, G) -> (512, 24, 4)
            BD, H, G = fut_feats.shape
            B_calc = BD // self.D  # 64

            # Reshape to (B, D, H, G)
            fut_feats = fut_feats.view(B_calc, self.D, H, G)  # (64, 8, 24, 4)
            # Permute to (B, H, D, G) to match the other path's shape
            fut_feats = fut_feats.permute(0, 2, 1, 3)  # (64, 24, 8, 4)

        else:
            # Input is (B, H, D*G)
            B_calc = fut_feats.size(0)
            H = fut_feats.size(1)
            # Reshape to (B, H, D, G)
            fut_feats = fut_feats.view(
                B_calc, H, self.D, self.fut_dim
            )  # (64, 24, 8, 4)

        # Now fut_feats is (B, H, D, G) -> (64, 24, 8, 4) in both cases

        mu = self.mean_head(fut_feats).squeeze(-1)  # (B, H, D)
        logv = self.logvar_head(fut_feats).squeeze(-1)
        sigma = self._softplus_floor(
            logv,
            floor=float(self.cfg.sigma_floor),
            cap=(float(self.cfg.sigma_cap) if self.cfg.sigma_cap is not None else None),
        )

        mu = self.revin._denormalize(mu)  # [B, H, D]

        # 3. Rescale Sigma
        # Sigma lives in the same RevIN output space as mu, so undo the
        # affine scaling before restoring the instance standard deviation.
        scale = self.revin.stdev
        if getattr(self.revin, "affine", False):
            w = self.revin.affine_weight.view(1, 1, -1)
            eps2 = float(getattr(self.revin, "eps", 0.0)) ** 2
            scale = scale / (w + eps2).abs()
        sigma = sigma * scale

        # Return Nu
        if self.use_student_t:
            nu_raw = self.nu_head(fut_feats).squeeze(-1)
            # Enforce nu > 2.0 for finite variance
            nu = 2.0 + F.softplus(nu_raw)
            return mu, sigma, nu

        return mu, sigma, None


# ---------------------------------------------------------------------------
# Quantile Regression Forecaster
# ---------------------------------------------------------------------------

_DEFAULT_QUANTILES = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)


class QuantileForecaster(nn.Module):
    """Same encoder as SingleEncoderForecaster but outputs quantiles."""

    def __init__(
        self, series_dim: int, target_dim: int, tmark_dim: int, config: ConfigDict
    ):
        super().__init__()
        self.cfg = config
        self.D_in = int(series_dim)
        self.D = int(target_dim)
        self.tmark_dim = tmark_dim
        self.device = config.device
        self.gp_input_dim = int(config.gp_input_dim)

        qs = config.get("quantiles", _DEFAULT_QUANTILES)
        self.register_buffer("quantiles", torch.tensor(qs, dtype=torch.float32))
        self.Q = len(qs)

        self.past_cal_mode = "none"
        self.past_in_dim = series_dim

        self.use_revin = config.get("use_revin", True)
        if self.use_revin:
            self.revin = RevIN(series_dim)
        else:
            self.revin = IdentityRevIN(series_dim)

        EncoderClass = get_encoder(config.encoder_type)
        self.past_enc = EncoderClass(
            input_dim=self.past_in_dim,
            output_dim=config.expert_hidden_dim,
            config=config,
            hidden_dim=config.expert_hidden_dim,
        )
        self.enc_hidden = config.expert_hidden_dim
        self.enc_is_per_channel = getattr(self.past_enc, "outputs_per_channel", False)

        self.use_channel_embed = False
        self.chan_dim = 0
        self.channel_emb = None
        self._chan_proj = None

        self.fut_dim = self.gp_input_dim
        if self.enc_is_per_channel:
            future_out_dim = self.fut_dim
        else:
            future_out_dim = self.D * self.fut_dim

        self.future_decoder_type = _canonical_future_decoder(
            config.get("future_decoder_type", "mlp")
        )
        self.future_quantile_seq_proj = None
        if _is_sequence_flatten_decoder(self.future_decoder_type):
            seq_out_dim = self.Q if self.enc_is_per_channel else (self.D * self.Q)
            self.future_quantile_seq_proj = SequenceFlattenProjection(
                token_count=_encoder_token_count(self.past_enc, config),
                hidden_dim=self.enc_hidden,
                out_dim=seq_out_dim,
                max_horizon=config.pred_len,
                dropout=config.dropout_rate,
            )
            self.future_proj = None
        elif self.future_decoder_type != "linear":
            raise NotImplementedError(
                "QuantileForecaster only supports future_decoder_type='linear' "
                "or 'sequence_flatten'."
            )
        else:
            H = config.pred_len
            self.future_proj = nn.Linear(config.expert_hidden_dim, H * future_out_dim)

        self.quantile_head = nn.Linear(self.fut_dim, self.Q)

        self.mark_agg = None

    def forward(self, seq_x, seq_x_mark, seq_y_mark, seq_mh_mark,
                gp_label_len: int, pred_len: int):
        self.revin._get_statistics(seq_x)
        seq_x = self.revin._normalize(seq_x)

        past_in = seq_x
        past_seq, ctx = self.past_enc(past_in)

        if _is_sequence_flatten_decoder(self.future_decoder_type):
            q_raw = self.future_quantile_seq_proj(past_seq, horizon=pred_len)
            if self.enc_is_per_channel:
                BD, H, Q = q_raw.shape
                B_calc = BD // self.D
                q_raw = q_raw.view(B_calc, self.D, H, Q).permute(0, 2, 1, 3)
            else:
                B_calc, H, _ = q_raw.shape
                q_raw = q_raw.view(B_calc, H, self.D, self.Q)

            scale = self.revin.stdev
            mean = self.revin.mean
            if getattr(self.revin, "affine", False):
                w = self.revin.affine_weight.view(1, 1, -1)
                b = self.revin.affine_bias.view(1, 1, -1)
                eps = float(getattr(self.revin, "eps", 1e-5))
                q_denorm = (q_raw - b.unsqueeze(-1)) / (w.unsqueeze(-1) + eps)
                q_denorm = q_denorm * scale.unsqueeze(-1) + mean.unsqueeze(-1)
            else:
                q_denorm = q_raw * scale.unsqueeze(-1) + mean.unsqueeze(-1)
            return q_denorm

        fut_feats = self.future_proj(ctx)
        if self.enc_is_per_channel:
            fut_feats = fut_feats.view(ctx.size(0), pred_len, self.fut_dim)
        else:
            fut_feats = fut_feats.view(ctx.size(0), pred_len, self.D * self.fut_dim)

        if self.enc_is_per_channel:
            BD, H, G = fut_feats.shape
            B_calc = BD // self.D
            fut_feats = fut_feats.view(B_calc, self.D, H, G)
            fut_feats = fut_feats.permute(0, 2, 1, 3)  # (B, H, D, G)
        else:
            B_calc = fut_feats.size(0)
            H = fut_feats.size(1)
            fut_feats = fut_feats.view(B_calc, H, self.D, self.fut_dim)

        q_raw = self.quantile_head(fut_feats)  # (B, H, D, Q)

        # Denormalize each quantile with RevIN
        # q_raw is in normalized space; apply the same denorm as mu
        scale = self.revin.stdev  # (B, 1, D)
        mean = self.revin.mean    # (B, 1, D)
        if getattr(self.revin, "affine", False):
            w = self.revin.affine_weight.view(1, 1, -1)
            b = self.revin.affine_bias.view(1, 1, -1)
            eps = float(getattr(self.revin, "eps", 1e-5))
            q_denorm = (q_raw - b.unsqueeze(-1)) / (w.unsqueeze(-1) + eps)
            q_denorm = q_denorm * scale.unsqueeze(-1) + mean.unsqueeze(-1)
        else:
            q_denorm = q_raw * scale.unsqueeze(-1) + mean.unsqueeze(-1)

        return q_denorm  # (B, H, D, Q)


# ---------------------------------------------------------------------------
# Mixture Density Network forecaster
# ---------------------------------------------------------------------------
class MDNForecaster(nn.Module):
    """Mixture Density Network (Bishop, 1994) over the same encoder family.

    Outputs a finite K-component mixture per (batch, horizon, channel):

        p(y_t | x) = sum_{k=1..K} pi_k(x,t) * Component_k(y_t; mu_k, sigma_k, [nu_k])

    where Component is Gaussian if `use_student_t_likelihood=False` and
    Student-t otherwise. Mixture weights pi_k come from a softmax over a
    K-way logit head, the mu_k are denormalised through RevIN per channel,
    sigma_k via softplus + RevIN scale, and (Student-t only) nu_k via
    `2 + softplus(...)`.

    Designed as a flat-mixture baseline: same encoder/training stack as
    SingleEncoderForecaster, but with a mixture output head replacing the
    single-distribution (mu, sigma, [nu]) head. Tests whether the regime /
    kernel / GP machinery in DeRegime adds value over a flexible mixture
    output of equivalent component count.
    """

    def __init__(
        self, series_dim: int, target_dim: int, tmark_dim: int, config: ConfigDict
    ):
        super().__init__()
        self.cfg = config
        self.D_in = int(series_dim)
        self.D = int(target_dim)
        self.tmark_dim = tmark_dim
        self.device = config.device
        self.gp_input_dim = int(config.gp_input_dim)

        self.K = int(config.get("mdn_num_components", 4))
        if self.K < 1:
            raise ValueError(f"mdn_num_components must be >= 1, got {self.K}")
        self.use_student_t = bool(config.get("use_student_t_likelihood", False))
        # Per-component params: mu, log_sigma_raw, [log_df_raw]
        self.params_per_comp = 3 if self.use_student_t else 2
        # Total scalars per (batch, horizon, channel): K weight logits + K * params
        self.mdn_out_dim = self.K * (self.params_per_comp + 1)

        # Past input
        self.past_cal_mode = "none"
        self.past_in_dim = series_dim

        self.use_revin = config.get("use_revin", True)
        if self.use_revin:
            self.revin = RevIN(series_dim)
        else:
            print("INFO: RevIN is DISABLED (Using Identity).")
            self.revin = IdentityRevIN(series_dim)

        EncoderClass = get_encoder(config.encoder_type)
        self.past_enc = EncoderClass(
            input_dim=self.past_in_dim,
            output_dim=config.expert_hidden_dim,
            config=config,
            hidden_dim=config.expert_hidden_dim,
        )
        self.enc_hidden = config.expert_hidden_dim
        self.enc_is_per_channel = getattr(self.past_enc, "outputs_per_channel", False)

        self.use_channel_embed = False
        self.chan_dim = 0
        self.channel_emb = None
        self._chan_proj = None

        # Future decoder type
        self.fut_dim = self.gp_input_dim
        self.future_decoder_type = _canonical_future_decoder(
            config.get("future_decoder_type", "mlp")
        )
        if self.enc_is_per_channel:
            future_out_dim = self.fut_dim
        else:
            future_out_dim = self.D * self.fut_dim

        self.future_mixture_seq_proj = None
        if _is_sequence_flatten_decoder(self.future_decoder_type):
            seq_out_dim = (
                self.mdn_out_dim
                if self.enc_is_per_channel
                else (self.D * self.mdn_out_dim)
            )
            self.future_mixture_seq_proj = SequenceFlattenProjection(
                token_count=_encoder_token_count(self.past_enc, config),
                hidden_dim=self.enc_hidden,
                out_dim=seq_out_dim,
                max_horizon=config.pred_len,
                dropout=config.dropout_rate,
            )
            self.future_proj = None
            self.future_enc = None
        elif self.future_decoder_type == "linear":
            H = config.pred_len
            self.future_proj = nn.Linear(config.expert_hidden_dim, H * future_out_dim)
            self.future_enc = None
        else:
            self.future_enc = _make_future_branch(
                self.future_decoder_type,
                in_dim=tmark_dim,
                out_dim=future_out_dim,
                ctx_dim=config.expert_hidden_dim,
                hidden=config.horizon_mlp_hidden_dim,
                dropout=config.dropout_rate,
                **_future_branch_extra_kwargs(config),
            )
            self.future_proj = None

        # Single mixture head over fut_dim → K * (params + 1)
        self.mixture_head = nn.Linear(self.fut_dim, self.mdn_out_dim)

        self.mark_agg = None

    @staticmethod
    def _softplus_floor(x, floor=1e-6, cap=None):
        s = F.softplus(x) + floor
        if cap is not None:
            s = torch.clamp(s, max=float(cap))
        return s

    def _split_mixture_params(self, raw):
        """Split last dim of raw (..., K*(P+1)) into (logits, mu_raw, logsig_raw, [logdf_raw]).

        Returns tensors of shape (..., K).
        """
        K = self.K
        P = self.params_per_comp
        # raw: (..., K*(P+1)) = (..., K + K*P) — K weight logits then K*P component params
        weight_logits = raw[..., :K]
        comp = raw[..., K:].view(*raw.shape[:-1], K, P)
        mu_raw = comp[..., 0]
        logsig_raw = comp[..., 1]
        logdf_raw = comp[..., 2] if self.use_student_t else None
        return weight_logits, mu_raw, logsig_raw, logdf_raw

    def _denormalize_mu_sigma(self, mu_raw, logsig_raw):
        """Apply RevIN inverse to per-component mu and sigma. Each is (B, H, D, K)."""
        sigma = self._softplus_floor(
            logsig_raw,
            floor=float(self.cfg.sigma_floor),
            cap=(
                float(self.cfg.sigma_cap)
                if self.cfg.sigma_cap is not None
                else None
            ),
        )
        # Manual RevIN inverse with broadcast over the K dimension.
        # revin.mean / stdev are (B, 1, D); we need (B, H, D, K) outputs.
        mean = self.revin.mean.unsqueeze(-1)   # (B, 1, D, 1)
        stdev = self.revin.stdev.unsqueeze(-1)  # (B, 1, D, 1)
        if getattr(self.revin, "affine", False):
            w = self.revin.affine_weight.view(1, 1, -1, 1)
            b = self.revin.affine_bias.view(1, 1, -1, 1)
            eps2 = float(getattr(self.revin, "eps", 0.0)) ** 2
            mu = (mu_raw - b) / (w + eps2)
            mu = mu * stdev + mean
            sigma_scale = (stdev / (w + eps2).abs()).expand_as(sigma)
        else:
            mu = mu_raw * stdev + mean
            sigma_scale = stdev.expand_as(sigma)
        sigma = sigma * sigma_scale
        return mu, sigma

    def forward(
        self,
        seq_x,
        seq_x_mark,
        seq_y_mark,
        seq_mh_mark,
        gp_label_len: int,
        pred_len: int,
    ):
        # 1. Normalize + encode past
        self.revin._get_statistics(seq_x)
        seq_x = self.revin._normalize(seq_x)
        past_in = seq_x
        past_seq, ctx = self.past_enc(past_in)

        # Future marks
        B = seq_x.size(0)
        if pred_len > 1:
            if seq_mh_mark is None or seq_mh_mark.numel() == 0:
                marks_H = torch.zeros(
                    B, pred_len, self.tmark_dim, device=seq_x.device, dtype=seq_x.dtype
                )
            else:
                if seq_mh_mark.size(1) == pred_len - 1:
                    dec_start = (
                        seq_y_mark[:, -1:, :]
                        if (
                            self.past_cal_mode != "none"
                            and seq_y_mark is not None
                            and seq_y_mark.numel() > 0
                        )
                        else torch.zeros(
                            B, 1, self.tmark_dim, device=seq_x.device, dtype=seq_x.dtype
                        )
                    )
                    marks_H = torch.cat([dec_start, seq_mh_mark], dim=1)
                elif seq_mh_mark.size(1) == pred_len:
                    marks_H = seq_mh_mark
                else:
                    raise ValueError(
                        f"Expected future marks length {pred_len-1} or {pred_len}, "
                        f"got {seq_mh_mark.size(1)}"
                    )
        else:
            marks_H = (
                seq_y_mark[:, -1:, :]
                if (
                    self.past_cal_mode != "none"
                    and seq_y_mark is not None
                    and seq_y_mark.numel() > 0
                )
                else torch.zeros(
                    B, 1, self.tmark_dim, device=seq_x.device, dtype=seq_x.dtype
                )
            )
        if self.enc_is_per_channel:
            marks_H = marks_H.repeat_interleave(self.D, dim=0)

        # 2a. Sequence-flatten path: project directly to mixture params.
        if _is_sequence_flatten_decoder(self.future_decoder_type):
            raw = self.future_mixture_seq_proj(past_seq, horizon=pred_len)
            if self.enc_is_per_channel:
                BD, H, P = raw.shape
                B_calc = BD // self.D
                raw = raw.view(B_calc, self.D, H, P).permute(0, 2, 1, 3)
            else:
                B_calc, H, _ = raw.shape
                raw = raw.view(B_calc, H, self.D, self.mdn_out_dim)
            weight_logits, mu_raw, logsig_raw, logdf_raw = self._split_mixture_params(
                raw
            )
            weights = F.softmax(weight_logits, dim=-1)  # (B, H, D, K)
            mu, sigma = self._denormalize_mu_sigma(mu_raw, logsig_raw)
            if self.use_student_t:
                nu = 2.0 + F.softplus(logdf_raw)
                return weights, mu, sigma, nu
            return weights, mu, sigma, None

        # 2b. future_enc / linear path.
        if self.future_decoder_type == "linear":
            fut_feats = self.future_proj(ctx)
            if self.enc_is_per_channel:
                fut_feats = fut_feats.view(ctx.size(0), pred_len, self.fut_dim)
            else:
                fut_feats = fut_feats.view(ctx.size(0), pred_len, self.D * self.fut_dim)
        else:
            fut_feats = self.future_enc(marks_H, ctx)

        if self.enc_is_per_channel:
            BD, H, G = fut_feats.shape
            B_calc = BD // self.D
            fut_feats = fut_feats.view(B_calc, self.D, H, G).permute(0, 2, 1, 3)
        else:
            B_calc = fut_feats.size(0)
            H = fut_feats.size(1)
            fut_feats = fut_feats.view(B_calc, H, self.D, self.fut_dim)

        raw = self.mixture_head(fut_feats)  # (B, H, D, K*(P+1))
        weight_logits, mu_raw, logsig_raw, logdf_raw = self._split_mixture_params(raw)
        weights = F.softmax(weight_logits, dim=-1)
        mu, sigma = self._denormalize_mu_sigma(mu_raw, logsig_raw)
        if self.use_student_t:
            nu = 2.0 + F.softplus(logdf_raw)
            return weights, mu, sigma, nu
        return weights, mu, sigma, None
