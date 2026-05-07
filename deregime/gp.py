import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood


class _PhiNet(nn.Module):
    """φ(t)=sigmoid(MLP(σ(t))) ∈ [0,1]; p(t)=p_min+(p_max-p_min)*φ(t)."""

    def __init__(self, R: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(R, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, g):  # g: [N,R]
        return torch.sigmoid(self.net(g)).squeeze(-1)  # [N] in [0,1]


class RegimeMixingKernel(gpytorch.kernels.Kernel):
    """
    k((t,x),(t',x')) = Σ_r [ σ_r(t) σ_r(t') * k_r(x_r, x'_r) ] + 1{i=j} * v(t)
    v(t) = Σ_r σ_r(t)^{p(t)} * τ_r^2

    Modes:
      - p_mode="fixed": p(t) ≡ p_fixed (user-specified, not learned)
      - p_mode="nn":    p(t) = p_min + (p_max - p_min) * φ(t), φ(t)=sigmoid(MLP(σ(t)))

    Notes:
      • Overall kernel is non-stationary in (t,x); each expert RBF/RQ is stationary on its slice.
      • Heteroskedastic term is diagonal-only; added for diag=True and when x1==x2.
    """

    is_stationary = False

    def __init__(
        self,
        num_regimes: int,
        expert_gp_input_dim: int,
        *,
        # ---- NEW: Kernel Type ----
        expert_kernel_type: str = "rbf",  # "rbf" or "rq"
        # ---- RBF/RQ init controls ----
        rbf_init: str = "logspace",  # "fixed" | "logspace" | "rand_loguniform" | "empirical"
        rbf_ls_range: tuple[float, float] = (0.1, 10.0),  # lengthscale range
        rbf_os_range: tuple[float, float] = (
            0.1,
            2.0,
        ),  # outputscale^0.5 (amplitude) range
        rbf_ls_isotropic: bool = True,  # if False, per-dim randomisation (ARD)
        rbf_empirical_jitter: tuple[float, float] = (1.0, 1.0),
        rbf_randomize_order: bool = True,  # shuffle regime assignment after init
        # ---- Optional priors (off by default) ----
        rbf_use_priors: bool = False,
        rbf_ls_prior_logstd: float = 0.35,
        rbf_os_prior_logstd: float = 0.35,
        # ---- Existing args ----
        p_mode: str = "nn",  # "fixed" or "nn"
        p_fixed: float = 2.0,  # used iff p_mode="fixed" (e.g., 1.0 or 2.0)
        p_min: float = 1.0,  # bounds for p(t) when p_mode="nn"
        p_max: float = 2.0,
        phi_hidden: int = 16,  # NN head width (p_mode="nn")
        detach_g_for_phi: bool = False,  # stop φ-gradients back into gates σ
        eps_min: float = 1e-8,
        use_mixed_linear=False,
        regime_checkpointing: bool = False,
        kernel_gate_mode: str = "regime",
        **kwargs,
    ):
        super().__init__()
        self.num_regimes = num_regimes
        self.expert_gp_input_dim = expert_gp_input_dim
        self.expert_kernel_type = expert_kernel_type.lower()
        self.regime_checkpointing = bool(regime_checkpointing)
        kernel_gate_mode = str(kernel_gate_mode).lower()
        if kernel_gate_mode in {
            "regime",
            "regime_mixing",
            "gated",
            "gate",
            "full",
        }:
            self.kernel_gate_mode = "regime"
        elif kernel_gate_mode in {
            "shared",
            "no_gate",
            "nogate",
            "no_kernel_gate",
            "nokernelgate",
            "none",
        }:
            self.kernel_gate_mode = "shared"
        else:
            raise ValueError(
                "kernel_gate_mode must be 'regime' or 'shared'/'no_kernel_gate'; "
                f"got {kernel_gate_mode!r}"
            )

        # --- store new options
        self.rbf_init = rbf_init
        self.rbf_ls_range = rbf_ls_range
        self.rbf_os_range = rbf_os_range
        self.rbf_ls_isotropic = rbf_ls_isotropic
        self.rbf_empirical_jitter = rbf_empirical_jitter
        self.rbf_randomize_order = rbf_randomize_order
        self.rbf_use_priors = rbf_use_priors
        self.rbf_ls_prior_logstd = rbf_ls_prior_logstd
        self.rbf_os_prior_logstd = rbf_os_prior_logstd

        # Initialize Expert kernels (ScaleKernel wrapping RBF or RQ)
        self.use_mixed_linear = use_mixed_linear
        self.kernels = nn.ModuleList()
        for _ in range(num_regimes):
            # 1. Create the Stationary Kernel (RBF or RQ)
            if self.expert_kernel_type == "rq":
                base_stat = gpytorch.kernels.RQKernel(ard_num_dims=expert_gp_input_dim)
            elif self.expert_kernel_type == "rbf":
                base_stat = gpytorch.kernels.RBFKernel(ard_num_dims=expert_gp_input_dim)
            else:
                raise ValueError(
                    f"Unknown expert_kernel_type: {self.expert_kernel_type}. Use 'rbf' or 'rq'."
                )

            # Wrap stationary in ScaleKernel (Standard Amplitude)
            k_stat = gpytorch.kernels.ScaleKernel(base_stat)

            if self.use_mixed_linear:
                # 2. Create Linear Kernel (Trend)
                # Wrap in ScaleKernel to allow learning the trend variance (slope magnitude)
                base_lin = gpytorch.kernels.LinearKernel(
                    ard_num_dims=expert_gp_input_dim
                )
                k_lin = gpytorch.kernels.ScaleKernel(base_lin)

                # 3. Combine: K_total = K_stat + K_lin
                # GPyTorch AdditiveKernel
                k_combined = k_stat + k_lin
                self.kernels.append(k_combined)
            else:
                # Just the stationary kernel
                self.kernels.append(k_stat)

        self.base_kernels = self.kernels

        # ---- initialise expert parameters (no priors unless enabled)
        self._init_expert_params()

        # p(t) control
        self.p_mode = p_mode.lower()
        assert self.p_mode in {"fixed", "nn"}
        # fixed p: store the provided constant (clamped to [p_min,p_max] once)
        self.register_buffer("p_min_buf", torch.tensor(float(p_min)))
        self.register_buffer("p_max_buf", torch.tensor(float(p_max)))
        if self.p_mode == "fixed":
            p0 = max(p_min, min(float(p_fixed), p_max))
            self.register_buffer("p_fixed_buf", torch.tensor(p0))
        else:
            self.phi_net = _PhiNet(num_regimes, hidden=phi_hidden)
            self.detach_g_for_phi = detach_g_for_phi

        self.register_buffer("eps_min", torch.tensor(float(eps_min)))
        self.register_buffer("temperature", torch.tensor(1.0))  # legacy

    # ----- NEW: RBF/RQ init helper -----
    def _init_expert_params(self):
        R = self.num_regimes
        ls_lo, ls_hi = self.rbf_ls_range
        os_lo, os_hi = self.rbf_os_range  # amplitude (std), NOT variance

        # values per regime
        if self.rbf_init == "fixed":
            ls_vals = torch.full((R,), float(ls_lo))
            os_vals = torch.full((R,), float(os_lo))
        elif self.rbf_init == "logspace":
            # geometric coverage over regimes
            ls_vals = torch.logspace(math.log10(ls_lo), math.log10(ls_hi), steps=R)
            os_vals = torch.logspace(math.log10(os_lo), math.log10(os_hi), steps=R)
        elif self.rbf_init == "rand_loguniform":
            u = torch.rand(R)
            ls_vals = ls_lo * (ls_hi / ls_lo) ** u
            v = torch.rand(R)
            os_vals = os_lo * (os_hi / os_lo) ** v
        elif self.rbf_init == "empirical":
            # Initial value; overwritten from the warmup GP feature pool after
            # the model has produced initial expert features. Outputscales are
            # still seeded from the configured amplitude range.
            ls_vals = torch.full((R,), float(math.sqrt(ls_lo * ls_hi)))
            v = torch.rand(R)
            os_vals = os_lo * (os_hi / os_lo) ** v
        else:
            raise ValueError(f"Unknown rbf_init: {self.rbf_init}")

        if self.rbf_randomize_order:
            perm = torch.randperm(R)
            ls_vals, os_vals = ls_vals[perm], os_vals[perm]

        # Apply to each expert
        for r, kernel_obj in enumerate(self.base_kernels):

            # --- Logic Split: Mixed vs Single ---
            if self.use_mixed_linear:
                # kernel_obj is an AdditiveKernel (Stationary + Linear)
                # k_stat is index 0, k_lin is index 1
                k_stat = kernel_obj.kernels[0]
                k_lin = kernel_obj.kernels[1]
            else:
                # kernel_obj is just the ScaleKernel(Stationary)
                k_stat = kernel_obj
                k_lin = None

            # 1. Initialize Stationary (RBF/RQ)
            base_k = k_stat.base_kernel

            # Lengthscale
            if self.rbf_ls_isotropic:
                ls_vec = torch.full((1, self.expert_gp_input_dim), float(ls_vals[r]))
            else:
                # small per-dim jitter in log-space
                jitter = 0.2 * torch.randn(self.expert_gp_input_dim)
                ls_dim = ls_vals[r] * (10.0**jitter)
                ls_vec = ls_dim.view(1, -1)

            base_k.initialize(lengthscale=ls_vec)

            # Outputscale (Amplitude)
            k_stat.initialize(outputscale=float(os_vals[r] ** 2))

            # RQ Alpha
            if self.expert_kernel_type == "rq":
                # Initialize alpha to 1.0 (sensible default for RQ)
                # You could also randomize this if desired, but 1.0 is standard
                base_k.initialize(alpha=1.0)

            # 2. Initialize Linear (If enabled)
            if k_lin is not None:
                # Initialize with tiny variance so it doesn't dominate early training
                # This makes the model "Stationary by default", learning trend only if needed.
                k_lin.initialize(outputscale=1e-4)

            # 3. Optional Priors (Only applied to Stationary part for now)
            if self.rbf_use_priors:
                ls_prior = gpytorch.priors.LogNormalPrior(
                    loc=float(math.log(ls_vals[r])),
                    scale=float(self.rbf_ls_prior_logstd),
                )
                base_k.register_prior(
                    name="lengthscale_prior",
                    prior=ls_prior,
                    param_or_closure="lengthscale",
                )

                os_prior = gpytorch.priors.LogNormalPrior(
                    loc=float(math.log((os_vals[r] ** 2) + 1e-12)),
                    scale=float(self.rbf_os_prior_logstd),
                )
                k_stat.register_prior(
                    name="outputscale_prior",
                    prior=os_prior,
                    param_or_closure="outputscale",
                )

    def initialize_empirical_lengthscales_from_features(
        self,
        feature_pool: torch.Tensor,
        *,
        max_points: int = 2048,
        min_lengthscale: float = 1e-4,
    ):
        """Initialize expert RBF/RQ lengthscales from the GP feature pool.

        `feature_pool` has full GP inputs: [gates, expert_0_features, ...].
        We estimate each regime's lengthscale from its own expert feature
        slice, so the kernel starts calibrated to the learned latent feature
        scale rather than to the raw input scale.
        """
        if str(self.rbf_init).lower() != "empirical":
            return

        if feature_pool.numel() == 0:
            print("Empirical RBF init skipped: empty feature pool.")
            return

        pool = feature_pool.detach().float().cpu()
        if pool.dim() != 2:
            pool = pool.reshape(-1, pool.size(-1))
        if pool.size(-1) < self.num_regimes + self.num_regimes * self.expert_gp_input_dim:
            raise ValueError(
                "Feature pool has too few columns for empirical kernel init: "
                f"got {pool.size(-1)}"
            )

        finite = torch.isfinite(pool).all(dim=-1)
        pool = pool[finite]
        if pool.size(0) < 2:
            print("Empirical RBF init skipped: fewer than two finite feature rows.")
            return

        if pool.size(0) > max_points:
            idx = torch.randperm(pool.size(0))[:max_points]
            pool = pool[idx]

        jitter_lo, jitter_hi = self.rbf_empirical_jitter
        jitter_lo = max(float(jitter_lo), 1e-8)
        jitter_hi = max(float(jitter_hi), jitter_lo)
        dim = int(self.expert_gp_input_dim)
        ls_summary = []

        with torch.no_grad():
            for r, kernel_obj in enumerate(self.base_kernels):
                if self.use_mixed_linear:
                    k_stat = kernel_obj.kernels[0]
                else:
                    k_stat = kernel_obj
                base_k = k_stat.base_kernel

                s = self.num_regimes + r * dim
                e = s + dim
                x = pool[:, s:e]
                x = x[torch.isfinite(x).all(dim=-1)]
                if x.size(0) < 2:
                    ls_vec = torch.ones(dim) * max(min_lengthscale, 1.0)
                elif self.rbf_ls_isotropic:
                    dist = torch.pdist(x, p=2)
                    dist = dist[torch.isfinite(dist) & (dist > 1e-12)]
                    if dist.numel() == 0:
                        std = x.std(dim=0, unbiased=False)
                        base_ls = torch.sqrt(torch.mean(std.pow(2))).item()
                    else:
                        # Convert a median Euclidean distance into a per-coordinate
                        # ARD lengthscale. This avoids overly broad kernels as
                        # gp_input_dim grows.
                        base_ls = dist.median().item() / math.sqrt(max(dim, 1))
                    base_ls = max(float(base_ls), min_lengthscale)
                    u = torch.rand(())
                    jitter = jitter_lo * (jitter_hi / jitter_lo) ** u
                    ls_vec = torch.full((dim,), base_ls * float(jitter))
                else:
                    q75 = torch.quantile(x, 0.75, dim=0)
                    q25 = torch.quantile(x, 0.25, dim=0)
                    robust_std = (q75 - q25) / 1.349
                    std = x.std(dim=0, unbiased=False)
                    ls_vec = torch.where(robust_std > min_lengthscale, robust_std, std)
                    if jitter_hi > 0:
                        u = torch.rand(dim)
                        jitter = jitter_lo * (jitter_hi / jitter_lo) ** u
                        ls_vec = ls_vec * jitter
                    ls_vec = ls_vec.clamp_min(min_lengthscale)

                device = base_k.raw_lengthscale.device
                dtype = base_k.raw_lengthscale.dtype
                ls_vec = ls_vec.to(device=device, dtype=dtype).view(1, dim)
                base_k.initialize(lengthscale=ls_vec)
                ls_summary.append(float(ls_vec.detach().mean().cpu()))

        if ls_summary:
            print(
                "Empirical RBF lengthscale init: "
                f"mean={np.mean(ls_summary):.4g}, "
                f"min={np.min(ls_summary):.4g}, max={np.max(ls_summary):.4g}, "
                f"isotropic={self.rbf_ls_isotropic}, jitter=({jitter_lo:g}, {jitter_hi:g})."
            )

    def p_of_t(self, g: torch.Tensor) -> torch.Tensor:
        """
        g: [N,R] gating weights
        returns p(t): [N] in [p_min, p_max] (nn mode) or constant vector (fixed mode).
        """
        if self.p_mode == "fixed":
            return torch.full(
                (g.shape[-2],), self.p_fixed_buf.item(), device=g.device, dtype=g.dtype
            )
        else:
            g_in = g.detach() if self.detach_g_for_phi else g
            phi = self.phi_net(g_in)  # [N] in [0,1]
            return self.p_min_buf + (self.p_max_buf - self.p_min_buf) * phi  # [N]

    # ----- kernel -----
    def _regime_contribution(self, r: int, x1: torch.Tensor, x2: torch.Tensor,
                             g1: torch.Tensor, g2: torch.Tensor, **params) -> torch.Tensor:
        """Compute one regime's contribution W_r ∘ K_r to the kernel sum.

        Pulled out of the forward loop so it can be wrapped in
        torch.utils.checkpoint when self.regime_checkpointing is True.
        """
        w1 = g1[..., r].unsqueeze(-1)
        w2 = g2[..., r].unsqueeze(-1)
        W = w1 @ w2.transpose(-2, -1)
        s = self.num_regimes + r * self.expert_gp_input_dim
        e = s + self.expert_gp_input_dim
        K_r = self.base_kernels[r](x1[..., s:e], x2[..., s:e], **params).to_dense()
        return W * K_r

    def _shared_feature_block(self, x: torch.Tensor) -> torch.Tensor:
        """Average the regime feature blocks into one shared-kernel input."""
        s = self.num_regimes
        e = s + self.num_regimes * self.expert_gp_input_dim
        blocks = x[..., s:e]
        if blocks.size(-1) != self.num_regimes * self.expert_gp_input_dim:
            raise ValueError(
                "RegimeMixingKernel expected concatenated inputs "
                "[gates, z_1, ..., z_R] but got incompatible last dimension "
                f"{x.size(-1)} for R={self.num_regimes}, "
                f"expert_gp_input_dim={self.expert_gp_input_dim}."
            )
        blocks = blocks.reshape(
            *blocks.shape[:-1], self.num_regimes, self.expert_gp_input_dim
        )
        return blocks.mean(dim=-2)

    def forward(self, x1, x2, diag: bool = False, **params):
        g1, g2 = x1[..., : self.num_regimes], x2[..., : self.num_regimes]
        if self.kernel_gate_mode == "shared":
            z1 = self._shared_feature_block(x1)
            z2 = self._shared_feature_block(x2)
            K = self.base_kernels[0](z1, z2, **params).to_dense()
            if diag:
                return torch.diagonal(K, dim1=-2, dim2=-1)
            return K

        # Signal: Σ_r (σ_r(t) σ_r(t')) ∘ K_r
        use_checkpoint = (
            self.regime_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and x1.requires_grad
        )
        K = 0.0
        for r in range(self.num_regimes):
            if use_checkpoint:
                contrib = torch.utils.checkpoint.checkpoint(
                    self._regime_contribution,
                    r, x1, x2, g1, g2,
                    use_reentrant=False,
                    **params,
                )
            else:
                contrib = self._regime_contribution(r, x1, x2, g1, g2, **params)
            K = K + contrib

        if diag:
            # The heteroskedastic noise is now HANDLED BY THE LIKELIHOOD
            return torch.diagonal(K, dim1=-2, dim2=-1)

        # The heteroskedastic noise is now HANDLED BY THE LIKELIHOOD
        return K


def _mean_over_time_batch(pi_db, mask=None, eps=1e-8):
    """
    pi_db: [T], [B,T], or [N] — dustbin probabilities
    mask:  same leading dims, 1/0 for valid positions (optional)
    returns scalar mean over all valid positions
    """
    if mask is None:
        return pi_db.float().mean()
    mask = mask.to(pi_db.dtype)
    num = (pi_db * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


class HeteroskedasticNoiseLikelihood(GaussianLikelihood):
    """
    A Gaussian likelihood that owns the heteroskedastic noise parameters
    and computes the noise level based on inputs.

    This version implements an identifiable parameterization:
    v_d(t) = c_d^2 * (sum_r sigma_r(t)^p * tau_r^2)

    Where:
    - c_d (per-task) is learned via `raw_c_task`
    - tau_r (per-regime) is learned via `raw_log_tau_main`
    - The constraint sum(log(tau_r)) = 0 is enforced by re-parameterization.
    """

    def __init__(
        self,
        kernel: RegimeMixingKernel,
        num_regimes: int,
        regime_noise_init: float,  # Used to initialize c_d
        *,
        num_tasks: int,
        likelihood_noise_floor: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            noise_constraint=gpytorch.constraints.GreaterThan(likelihood_noise_floor),
            **kwargs,
        )

        self.num_regimes = num_regimes
        self.num_tasks = int(num_tasks)
        self.kernel = kernel  # reference to the kernel to use its helper method

        if num_regimes <= 1:
            # If R=1, no constraint is needed. tau=1 is assumed.
            self.register_parameter("raw_log_tau_main", None)
        else:
            # --- Identifiable tau_r parameters ---
            # Learn R-1 unconstrained params in log-space.
            # Initialize at 0, so all tau_r start at exp(0) = 1.0
            self.register_parameter(
                "raw_log_tau_main", nn.Parameter(torch.zeros(num_regimes - 1))
            )

        # ---- per-task multipliers c_d ----
        # Initialize c_d using the provided regime_noise_init value
        init_val = float(regime_noise_init)
        if init_val <= 0:
            raise ValueError(f"regime_noise_init must be > 0, got {init_val}")

        c_init = torch.full((self.num_tasks,), init_val)

        # Use the softplus inverse to find the raw value that produces init_val
        self.register_parameter(
            "raw_c_task", nn.Parameter(torch.log(torch.expm1(c_init)))
        )

        # ---- global floor/jitter (fixed) ----
        self.noise = likelihood_noise_floor
        self.raw_noise.requires_grad = False

    @property
    def tau(self):
        """
        Constrained (positive) tau values of shape (R,).
        Enforces sum(log(tau)) = 0 (i.e., prod(tau) = 1).
        """
        if self.raw_log_tau_main is None:
            # This handles the R=1 case, where tau is just 1.
            return torch.tensor(
                [1.0], device=self.c_task.device, dtype=self.c_task.dtype
            )

        # 1. Get the R-1 unconstrained log-taus
        log_tau_main = self.raw_log_tau_main

        # 2. Compute the R-th log-tau to enforce sum(log_tau_r) = 0
        log_tau_last = -torch.sum(log_tau_main)

        # 3. Concatenate all R log-taus
        all_log_tau = torch.cat([log_tau_main, log_tau_last.unsqueeze(0)])

        # 4. Exponentiate to get the final tau values.
        #    These are guaranteed positive and satisfy prod(tau_r) = 1.
        return torch.exp(all_log_tau)

    @property
    def c_task(self):
        """Constrained (nonnegative) per-task multipliers c of shape (D,)."""
        return F.softplus(self.raw_c_task)

    def _diag_regime_variance(self, x: torch.Tensor, p_of_t_func) -> torch.Tensor:
        """
        Computes heteroskedastic variance v(t) with per-task scaling.
        x: (..., F) features; first R dims are g(t).
        """
        g = x[..., : self.num_regimes]  # (..., R)
        p = p_of_t_func(g)  # (...,)

        # This calculation is now identifiable.
        # self.tau (shape [R]) is the constrained property
        base = (g.clamp_min(1e-8).pow(p.unsqueeze(-1)) * (self.tau**2)).sum(
            -1
        )  # (...,)

        # Figure out which task/channel each row belongs to
        if x.dim() == 3:
            N, T, _ = x.shape
            d_idx = (torch.arange(N, device=x.device) % self.num_tasks).unsqueeze(-1)
            c = self.c_task[d_idx].expand(N, T)
        elif x.dim() == 2:
            N, _ = x.shape
            d_idx = torch.arange(N, device=x.device) % self.num_tasks
            c = self.c_task[d_idx]
        else:
            raise ValueError(f"Expected x to be 2D or 3D, got shape={tuple(x.shape)}")

        # apply per-task multiplier squared
        return base * (c**2)

    # TRAIN-TIME marginalization
    @staticmethod
    def _add_residual_variance(
        variance: torch.Tensor, residual_obs_var: torch.Tensor | None
    ) -> torch.Tensor:
        if residual_obs_var is None:
            return variance
        residual = residual_obs_var.to(device=variance.device, dtype=variance.dtype)
        return variance + residual.clamp_min(0.0)

    def marginal(self, function_dist, *args, **kwargs):
        x = kwargs["feats_all"]
        kernel = kwargs["kernel"]
        residual_obs_var = kwargs.get("residual_obs_var", None)
        backbone_obs_scale = kwargs.get("backbone_obs_scale", None)

        if backbone_obs_scale is not None:
            scale = backbone_obs_scale.to(
                device=function_dist.mean.device, dtype=function_dist.mean.dtype
            )
            total_noise = scale.clamp_min(1e-9).pow(2)
        else:
            noise_variances = self._diag_regime_variance(x, kernel.p_of_t)  # (...,)
            total_noise = noise_variances + self.noise  # add small constant floor

        total_noise = self._add_residual_variance(total_noise, residual_obs_var)
        marginal_covar = function_dist.covariance_matrix + torch.diag_embed(total_noise)
        return MultivariateNormal(function_dist.mean, marginal_covar)

    def forward(self, function_samples, *params, **kwargs):
        return super().forward(function_samples, *params, **kwargs)

    def expected_log_prob(self, target, function_dist, *args, **kwargs):
        feats_all = kwargs["feats_all"]
        kernel = kwargs["kernel"]
        residual_obs_var = kwargs.get("residual_obs_var", None)
        backbone_obs_scale = kwargs.get("backbone_obs_scale", None)
        qy = self.marginal(
            function_dist,
            feats_all=feats_all,
            kernel=kernel,
            residual_obs_var=residual_obs_var,
            backbone_obs_scale=backbone_obs_scale,
        )
        return qy.log_prob(target)


class SliceableStudentT(torch.distributions.StudentT):
    """
    A wrapper around torch.distributions.StudentT that supports slicing (subscripting).
    This allows it to behave like GPyTorch's MultivariateNormal in evaluation loops.
    """

    def __getitem__(self, idx):
        # Slice the parameters
        # Note: We rely on the fact that we expanded these tensors
        # to match the batch shape in the likelihood's forward method.
        new_df = self.df[idx]
        new_loc = self.loc[idx]
        new_scale = self.scale[idx]

        # Return a new instance with sliced parameters
        return SliceableStudentT(df=new_df, loc=new_loc, scale=new_scale)


class SliceableNormal(torch.distributions.Normal):
    """A Normal distribution with slicing support for GPyTorch-style callers."""

    def __getitem__(self, idx):
        return SliceableNormal(loc=self.loc[idx], scale=self.scale[idx])


LOG_2PI = math.log(2.0 * math.pi)


class RegimeStudentTLikelihood(GaussianLikelihood):
    """
    Student-t likelihood with regime-weighted heteroskedastic scale, in the same
    framework style as HeteroskedasticNoiseLikelihood.

    Identifiable parameterization (same as your Gaussian heteroskedastic):
        v_d(t) = c_d^2 * sum_r g_r(t)^p * tau_r^2

    We treat v_d(t) as *observation variance*.
    Torch StudentT uses 'scale' parameter s where Var = df/(df-2) * s^2 (df>2).
    So we convert variance -> StudentT scale via:
        s^2 = v * (df-2)/df

    Tail thickness is regime-specific. We learn one unconstrained parameter per
    regime and convert it to a local df(t) by gate-weighting in raw space,
    mirroring the way regimes control variance while keeping task-specific
    calibration in c_d.
    """

    def __init__(
        self,
        kernel,  # RegimeMixingKernel reference
        num_regimes: int,
        regime_scale_init: float = 0.1,  # DEFAULT so you don't have to pass it
        *,
        num_tasks: int,
        mc_samples: int = 8,
        use_quadrature: bool = True,
        gh_points: int = 20,
        df_init: float = 8.0,
        learn_df: bool = True,
        df_min: float = 2.1,
        df_max: float = 100.0,
        likelihood_noise_floor: float = 1e-6,
        regime_mixture_likelihood: bool = False,
        use_student_t: bool = True,
        tau_init_jitter: float = 0.0,
        df_init_jitter: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            noise_constraint=gpytorch.constraints.GreaterThan(likelihood_noise_floor),
            **kwargs,
        )

        self.kernel = kernel
        self.num_regimes = int(num_regimes)
        self.num_tasks = int(num_tasks)
        self.mc_samples = int(mc_samples)
        self.use_quadrature = bool(use_quadrature)
        self.gh_points = int(gh_points)
        self.use_student_t = bool(use_student_t)

        # --- tau_r (identifiable, same as your Gaussian heteroskedastic) ---
        # Optional log-space jitter at init: tau_init_jitter > 0 produces
        # raw_log_tau_main ~ N(0, jitter^2), so tau_r ~ exp-normal around 1.
        # Default 0.0 reproduces the legacy zeros-init exactly (bit-compat).
        if self.num_regimes <= 1:
            self.register_parameter("raw_log_tau_main", None)
        else:
            tau_jitter = float(tau_init_jitter)
            if tau_jitter > 0.0:
                tau_init = tau_jitter * torch.randn(self.num_regimes - 1)
            else:
                tau_init = torch.zeros(self.num_regimes - 1)
            self.register_parameter(
                "raw_log_tau_main", nn.Parameter(tau_init)
            )

        # --- per-task multipliers c_d (init as stddev, same as your Gaussian heteroskedastic) ---
        init_val = float(regime_scale_init)
        if init_val <= 0:
            raise ValueError(f"regime_scale_init must be > 0, got {init_val}")

        c_init = torch.full((self.num_tasks,), init_val)
        self.register_parameter(
            "raw_c_task", nn.Parameter(torch.log(torch.expm1(c_init)))
        )

        # --- fixed global floor (re-use GaussianLikelihood's noise_covar for compatibility) ---
        self.noise = float(likelihood_noise_floor)
        self.raw_noise.requires_grad = False

        # --- regime-level df parameters ---
        self.df_min = float(df_min)
        self.df_max = float(df_max)
        self.regime_mixture_likelihood = bool(regime_mixture_likelihood)
        if not (self.df_min > 2.0 and self.df_max > self.df_min):
            raise ValueError("Need df_min > 2 and df_max > df_min for finite variance.")

        df_init = float(df_init)
        df_init = min(max(df_init, self.df_min + 1e-4), self.df_max - 1e-4)
        z0 = (df_init - self.df_min) / (self.df_max - self.df_min)  # in (0,1)
        raw0 = torch.log(torch.tensor(z0) / (1.0 - torch.tensor(z0)))

        # Optional log-space jitter on per-regime raw_df at init.
        # df_init_jitter > 0 adds N(0, jitter^2) noise in raw (pre-sigmoid) space,
        # so post-sigmoid df_r spreads multiplicatively around df_init.
        # Default 0.0 reproduces the legacy "all-regimes-equal" init exactly.
        raw_df_regime = raw0.clone().detach().repeat(self.num_regimes)
        df_jitter = float(df_init_jitter)
        if df_jitter > 0.0:
            raw_df_regime = raw_df_regime + df_jitter * torch.randn(self.num_regimes)
        self.register_parameter("raw_df_regime", nn.Parameter(raw_df_regime))
        self.raw_df_regime.requires_grad_(bool(learn_df))

        if self.regime_mixture_likelihood and self.use_student_t:
            print(
                "INFO: Regime Student-t likelihood uses proper mixture scoring "
                "over regimes."
            )
        elif self.regime_mixture_likelihood:
            print(
                "INFO: Regime Gaussian likelihood uses proper mixture scoring "
                "over regimes."
            )

    @property
    def tau(self) -> torch.Tensor:
        """Constrained tau values of shape (R,), with prod(tau)=1."""
        if self.raw_log_tau_main is None:
            return torch.tensor(
                [1.0], device=self.c_task.device, dtype=self.c_task.dtype
            )

        log_tau_main = self.raw_log_tau_main
        log_tau_last = -torch.sum(log_tau_main)
        all_log_tau = torch.cat([log_tau_main, log_tau_last.unsqueeze(0)])
        return torch.exp(all_log_tau)

    @property
    def c_task(self) -> torch.Tensor:
        """Constrained per-task c of shape (D,), nonnegative."""
        return F.softplus(self.raw_c_task)

    def _raw_to_df(self, raw_df: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(raw_df)
        return self.df_min + (self.df_max - self.df_min) * z

    @property
    def df_regime(self) -> torch.Tensor:
        """Constrained per-regime df values of shape (R,)."""
        return self._raw_to_df(self.raw_df_regime)

    @property
    def df(self) -> torch.Tensor:
        """Compatibility summary of df on the transformed raw-space average."""
        return self._raw_to_df(self.raw_df_regime.mean())

    def _regime_df_weights(self, g: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        w = g.clamp_min(1e-8).pow(p.unsqueeze(-1))
        return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _local_df(self, x: torch.Tensor, p_of_t_func) -> torch.Tensor:
        g = x[..., : self.num_regimes]
        p = p_of_t_func(g)
        w = self._regime_df_weights(g, p)
        raw_local_df = (w * self.raw_df_regime).sum(dim=-1)
        df = self._raw_to_df(raw_local_df)
        if not getattr(self, "_df_nan_reported", False) and not torch.isfinite(df).all():
            # One-shot diagnostic: which upstream tensor first went non-finite.
            # Helps pin down whether the corruption is in the gates, the per-
            # regime mixing weights, or raw_df_regime itself.
            with torch.no_grad():
                report = {
                    "g_nan": bool(torch.isnan(g).any()),
                    "g_inf": bool(torch.isinf(g).any()),
                    "g_min": float(g.min()),
                    "g_max": float(g.max()),
                    "p_nan": bool(torch.isnan(p).any()),
                    "p_inf": bool(torch.isinf(p).any()),
                    "w_nan": bool(torch.isnan(w).any()),
                    "w_inf": bool(torch.isinf(w).any()),
                    "raw_df_regime_nan": bool(torch.isnan(self.raw_df_regime).any()),
                    "raw_df_regime_min": float(self.raw_df_regime.min()),
                    "raw_df_regime_max": float(self.raw_df_regime.max()),
                    "raw_local_df_nan": bool(torch.isnan(raw_local_df).any()),
                    "raw_local_df_inf": bool(torch.isinf(raw_local_df).any()),
                    "df_nan": bool(torch.isnan(df).any()),
                }
            print(f"[VSB-NaN] non-finite df detected; upstream: {report}", flush=True)
            self._df_nan_reported = True
        return df

    def local_df(self, x: torch.Tensor, *, kernel=None) -> torch.Tensor:
        kernel = kernel or self.kernel
        return self._local_df(x, kernel.p_of_t)

    def _resolve_local_df(
        self,
        x: torch.Tensor,
        p_of_t_func,
        backbone_df: torch.Tensor | None = None,
        *,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        if backbone_df is not None:
            df = backbone_df
            if device is not None or dtype is not None:
                df = df.to(device=device or df.device, dtype=dtype or df.dtype)
            return df.clamp_min(self.df_min + 1e-6)
        df = self._local_df(x, p_of_t_func)
        if device is not None or dtype is not None:
            df = df.to(device=device or df.device, dtype=dtype or df.dtype)
        return df

    def _diag_regime_variance(self, x: torch.Tensor, p_of_t_func) -> torch.Tensor:
        """
        Observation variance v(t) with per-task scaling.
        x: (..., F) features; first R dims are g(t).
        returns: (...,)
        """
        g = x[..., : self.num_regimes]  # (..., R)
        p = p_of_t_func(g)  # (...,)

        base = (g.clamp_min(1e-8).pow(p.unsqueeze(-1)) * (self.tau**2)).sum(
            -1
        )  # (...,)

        # map rows to task/channel like your Gaussian heteroskedastic class
        if x.dim() == 3:
            N, T, _ = x.shape
            d_idx = (torch.arange(N, device=x.device) % self.num_tasks).unsqueeze(-1)
            c = self.c_task[d_idx].expand(N, T)
        elif x.dim() == 2:
            N, _ = x.shape
            d_idx = torch.arange(N, device=x.device) % self.num_tasks
            c = self.c_task[d_idx]
        else:
            raise ValueError(f"Expected x to be 2D or 3D, got shape={tuple(x.shape)}")

        return base * (c**2)  # (...,)

    def _task_scale_for_x(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-row task scale c_d shaped like x without the feature dim."""
        if x.dim() == 3:
            N, T, _ = x.shape
            d_idx = (torch.arange(N, device=x.device) % self.num_tasks).unsqueeze(-1)
            return self.c_task[d_idx].expand(N, T)
        if x.dim() == 2:
            N, _ = x.shape
            d_idx = torch.arange(N, device=x.device) % self.num_tasks
            return self.c_task[d_idx]
        raise ValueError(f"Expected x to be 2D or 3D, got shape={tuple(x.shape)}")

    def mixture_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Gate weights used by the proper regime mixture likelihood."""
        g = x[..., : self.num_regimes].clamp_min(1e-12)
        return g / g.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def regime_component_params(
        self,
        x: torch.Tensor,
        *,
        kernel=None,
        residual_obs_var: torch.Tensor | None = None,
        backbone_obs_scale: torch.Tensor | None = None,
        backbone_df: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return (weights, component scale, df) for each regime component.

        Shapes are (..., R).  The latent GP location is shared across
        components. Student-t components differ by observation scale and df;
        Gaussian components return df=None.
        """
        del kernel  # kept for a symmetric call signature with local_* helpers
        weights = self.mixture_weights(x)
        R = self.num_regimes

        if self.use_student_t and backbone_df is not None:
            df = backbone_df.to(device=x.device, dtype=x.dtype).unsqueeze(-1).expand(
                *x.shape[:-1], R
            )
            df = df.clamp_min(self.df_min + 1e-6)
        elif self.use_student_t:
            view_shape = (1,) * (x.dim() - 1) + (R,)
            df = self.df_regime.to(device=x.device, dtype=x.dtype).view(view_shape)
            df = df.expand(*x.shape[:-1], R)
        else:
            df = None

        if backbone_obs_scale is not None:
            scale2 = (
                backbone_obs_scale.to(device=x.device, dtype=x.dtype)
                .clamp_min(1e-9)
                .pow(2)
                .unsqueeze(-1)
                .expand(*x.shape[:-1], R)
            )
            if residual_obs_var is not None:
                if df is not None:
                    obs_var = scale2 * df / (df - 2.0).clamp_min(1e-6)
                else:
                    obs_var = scale2
                obs_var = obs_var + residual_obs_var.to(
                    device=x.device, dtype=x.dtype
                ).unsqueeze(-1).clamp_min(0.0)
                if df is not None:
                    scale2 = obs_var * (df - 2.0) / df
                else:
                    scale2 = obs_var
            return weights, torch.sqrt(scale2.clamp_min(1e-12)), df

        c = self._task_scale_for_x(x).to(device=x.device, dtype=x.dtype)
        tau = self.tau.to(device=x.device, dtype=x.dtype)
        view_shape = (1,) * (x.dim() - 1) + (R,)
        obs_var = (c.unsqueeze(-1) * tau.view(view_shape)).pow(2)
        obs_var = obs_var + torch.as_tensor(
            self.noise, device=x.device, dtype=x.dtype
        )
        if residual_obs_var is not None:
            obs_var = obs_var + residual_obs_var.to(
                device=x.device, dtype=x.dtype
            ).unsqueeze(-1).clamp_min(0.0)
        if df is not None:
            scale2 = obs_var * (df - 2.0) / df
        else:
            scale2 = obs_var
        return weights, torch.sqrt(scale2.clamp_min(1e-12)), df

    @staticmethod
    def _add_residual_variance(
        variance: torch.Tensor, residual_obs_var: torch.Tensor | None
    ) -> torch.Tensor:
        if residual_obs_var is None:
            return variance
        residual = residual_obs_var.to(device=variance.device, dtype=variance.dtype)
        return variance + residual.clamp_min(0.0)

    def local_observation_variance(
        self,
        x: torch.Tensor,
        *,
        kernel=None,
        residual_obs_var: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kernel = kernel or self.kernel
        v = self._diag_regime_variance(x, kernel.p_of_t)
        v_total = v + self.noise
        return self._add_residual_variance(v_total, residual_obs_var)

    def _diag_gaussian_scale2(
        self,
        x: torch.Tensor,
        p_of_t_func,
        residual_obs_var: torch.Tensor | None = None,
        backbone_obs_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Observation variance for the Gaussian version of this likelihood."""
        if backbone_obs_scale is not None:
            scale2 = backbone_obs_scale.clamp_min(1e-9).pow(2)
            return self._add_residual_variance(scale2, residual_obs_var).clamp_min(
                1e-12
            )
        v = self._diag_regime_variance(x, p_of_t_func)
        v_total = v + self.noise
        return self._add_residual_variance(v_total, residual_obs_var).clamp_min(
            1e-12
        )

    def _diag_studentt_scale2(
        self,
        x: torch.Tensor,
        p_of_t_func,
        residual_obs_var: torch.Tensor | None = None,
        backbone_obs_scale: torch.Tensor | None = None,
        backbone_df: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert observation variance v -> StudentT scale^2 using:
            Var = df/(df-2) * scale^2   =>  scale^2 = Var * (df-2)/df
        """
        if backbone_obs_scale is not None:
            local_df = self._resolve_local_df(
                x,
                p_of_t_func,
                backbone_df,
                device=backbone_obs_scale.device,
                dtype=backbone_obs_scale.dtype,
            )
            scale2 = backbone_obs_scale.clamp_min(1e-9).pow(2)
            if residual_obs_var is not None:
                obs_var = scale2 * local_df / (local_df - 2.0).clamp_min(1e-6)
                obs_var = self._add_residual_variance(obs_var, residual_obs_var)
                scale2 = obs_var * (local_df - 2.0) / local_df
            return scale2.clamp_min(1e-12)

        v = self._diag_regime_variance(x, p_of_t_func)
        v_total = v + self.noise  # add floor in variance space
        v_total = self._add_residual_variance(v_total, residual_obs_var)

        local_df = self._resolve_local_df(
            x, p_of_t_func, backbone_df, device=v_total.device, dtype=v_total.dtype
        )
        scale2 = v_total * (local_df - 2.0) / local_df
        return scale2.clamp_min(1e-12)

    def local_scale(
        self,
        x: torch.Tensor,
        *,
        kernel=None,
        residual_obs_var: torch.Tensor | None = None,
        backbone_obs_scale: torch.Tensor | None = None,
        backbone_df: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kernel = kernel or self.kernel
        return torch.sqrt(
            self._diag_studentt_scale2(
                x,
                kernel.p_of_t,
                residual_obs_var=residual_obs_var,
                backbone_obs_scale=backbone_obs_scale,
                backbone_df=backbone_df,
            )
        )

    # Used in your validation code to get mean/std for Gaussian-style metrics
    def marginal(self, function_dist, *args, **kwargs):
        x = kwargs["feats_all"]
        kernel = kwargs.get("kernel", None) or self.kernel
        residual_obs_var = kwargs.get("residual_obs_var", None)
        backbone_obs_scale = kwargs.get("backbone_obs_scale", None)
        backbone_df = kwargs.get("backbone_df", None)

        if not self.use_student_t:
            v_total = self._diag_gaussian_scale2(
                x,
                kernel.p_of_t,
                residual_obs_var=None,
                backbone_obs_scale=backbone_obs_scale,
            ).to(device=function_dist.mean.device, dtype=function_dist.mean.dtype)
        elif backbone_obs_scale is not None:
            scale = backbone_obs_scale.to(
                device=function_dist.mean.device, dtype=function_dist.mean.dtype
            ).clamp_min(1e-9)
            df = self._resolve_local_df(
                x,
                kernel.p_of_t,
                backbone_df,
                device=scale.device,
                dtype=scale.dtype,
            )
            v_total = scale.pow(2) * df / (df - 2.0).clamp_min(1e-6)
        else:
            v = self._diag_regime_variance(x, kernel.p_of_t)
            v_total = v + self.noise
        v_total = self._add_residual_variance(v_total, residual_obs_var)

        marginal_covar = function_dist.covariance_matrix + torch.diag_embed(v_total)
        return MultivariateNormal(function_dist.mean, marginal_covar)

    # Not strictly used by VariationalELBO path, but keep correct + non-abstract
    def forward(self, function_samples, *params, **kwargs):
        feats_all = kwargs.get("feats_all", None)
        kernel = kwargs.get("kernel", None) or self.kernel
        residual_obs_var = kwargs.get("residual_obs_var", None)
        backbone_obs_scale = kwargs.get("backbone_obs_scale", None)
        backbone_df = kwargs.get("backbone_df", None)

        if feats_all is None:
            if self.use_student_t:
                df = self.df.to(device=function_samples.device, dtype=function_samples.dtype)
                # fallback: constant-ish scale from floor
                scale = torch.sqrt(
                    ((df - 2.0) / df)
                    * torch.as_tensor(
                        self.noise,
                        device=function_samples.device,
                        dtype=function_samples.dtype,
                    )
                )
            else:
                scale = torch.sqrt(
                    torch.as_tensor(
                        self.noise,
                        device=function_samples.device,
                        dtype=function_samples.dtype,
                    )
                )
        else:
            if self.use_student_t:
                df = self._resolve_local_df(
                    feats_all,
                    kernel.p_of_t,
                    backbone_df,
                    device=function_samples.device,
                    dtype=function_samples.dtype,
                )
                scale2 = self._diag_studentt_scale2(
                    feats_all,
                    kernel.p_of_t,
                    residual_obs_var=residual_obs_var,
                    backbone_obs_scale=backbone_obs_scale,
                    backbone_df=backbone_df,
                ).to(device=function_samples.device, dtype=function_samples.dtype)
            else:
                df = None
                scale2 = self._diag_gaussian_scale2(
                    feats_all,
                    kernel.p_of_t,
                    residual_obs_var=residual_obs_var,
                    backbone_obs_scale=backbone_obs_scale,
                ).to(device=function_samples.device, dtype=function_samples.dtype)
            scale = torch.sqrt(scale2)

        if not self.use_student_t:
            return SliceableNormal(loc=function_samples, scale=scale)
        return SliceableStudentT(df=df, loc=function_samples, scale=scale)

    def expected_log_prob(
        self, target, function_dist: MultivariateNormal, *args, **kwargs
    ):
        """
        E_{q(f)}[ log StudentT(y | f, scale(x), df) ].

        Two modes controlled by self.use_quadrature:
          - True  → Gauss-Hermite quadrature (deterministic, zero-variance gradients)
          - False → reparameterised MC sampling (original path)

        Returns [N, T] (summed to scalar upstream).
        """
        feats_all = kwargs["feats_all"]
        kernel = kwargs.get("kernel", None) or self.kernel
        residual_obs_var = kwargs.get("residual_obs_var", None)
        backbone_obs_scale = kwargs.get("backbone_obs_scale", None)
        backbone_df = kwargs.get("backbone_df", None)

        mean = function_dist.mean  # [N, T]
        if not self.use_student_t:
            scale2 = self._diag_gaussian_scale2(
                feats_all,
                kernel.p_of_t,
                residual_obs_var=residual_obs_var,
                backbone_obs_scale=backbone_obs_scale,
            ).to(device=mean.device, dtype=mean.dtype)
            scale = torch.sqrt(scale2)

            if self.regime_mixture_likelihood:
                weights, comp_scale, _ = self.regime_component_params(
                    feats_all,
                    kernel=kernel,
                    residual_obs_var=residual_obs_var,
                    backbone_obs_scale=backbone_obs_scale,
                    backbone_df=backbone_df,
                )
                if self.use_quadrature:
                    return self._expected_log_prob_regime_gaussian_mixture_quadrature(
                        target, function_dist, weights, comp_scale
                    )
                return self._expected_log_prob_regime_gaussian_mixture_mc(
                    target, function_dist, weights, comp_scale
                )
            return self._expected_log_prob_gaussian_analytic(
                target, function_dist, scale
            )

        df = self._resolve_local_df(
            feats_all,
            kernel.p_of_t,
            backbone_df,
            device=mean.device,
            dtype=mean.dtype,
        )

        scale2 = self._diag_studentt_scale2(
            feats_all,
            kernel.p_of_t,
            residual_obs_var=residual_obs_var,
            backbone_obs_scale=backbone_obs_scale,
            backbone_df=backbone_df,
        ).to(device=mean.device, dtype=mean.dtype)
        scale = torch.sqrt(scale2)

        if self.regime_mixture_likelihood:
            weights, comp_scale, comp_df = self.regime_component_params(
                feats_all,
                kernel=kernel,
                residual_obs_var=residual_obs_var,
                backbone_obs_scale=backbone_obs_scale,
                backbone_df=backbone_df,
            )
            if self.use_quadrature:
                return self._expected_log_prob_regime_mixture_quadrature(
                    target, function_dist, weights, comp_df, comp_scale
                )
            return self._expected_log_prob_regime_mixture_mc(
                target, function_dist, weights, comp_df, comp_scale
            )

        if self.use_quadrature:
            return self._expected_log_prob_quadrature(
                target, function_dist, df, scale
            )
        else:
            return self._expected_log_prob_mc(
                target, function_dist, df, scale
            )

    def _expected_log_prob_mc(self, target, function_dist, df, scale):
        """Original MC sampling path."""
        K = max(1, int(self.mc_samples))
        f_samps = function_dist.rsample(sample_shape=torch.Size([K]))  # [K, N, T]
        dist = SliceableStudentT(df=df, loc=f_samps, scale=scale)
        lp = dist.log_prob(target)  # [K, N, T]
        return lp.mean(dim=0)  # [N, T]

    def _expected_log_prob_regime_mixture_mc(
        self, target, function_dist, weights, df, scale
    ):
        """MC estimate of E_q log sum_r pi_r StudentT(y | f, scale_r, df_r)."""
        K = max(1, int(self.mc_samples))
        f_samps = function_dist.rsample(sample_shape=torch.Size([K]))  # [K, N, T]
        log_pi = torch.log(weights.clamp_min(1e-12)).unsqueeze(0)  # [1, N, T, R]
        dist = SliceableStudentT(
            df=df.unsqueeze(0),
            loc=f_samps.unsqueeze(-1),
            scale=scale.unsqueeze(0),
        )
        log_pdf = dist.log_prob(target.unsqueeze(0).unsqueeze(-1))
        log_mix = torch.logsumexp(log_pi + log_pdf, dim=-1)
        return log_mix.mean(dim=0)

    def _expected_log_prob_regime_gaussian_mixture_mc(
        self, target, function_dist, weights, scale
    ):
        """MC estimate of E_q log sum_r pi_r Normal(y | f, scale_r)."""
        K = max(1, int(self.mc_samples))
        f_samps = function_dist.rsample(sample_shape=torch.Size([K]))  # [K, N, T]
        log_pi = torch.log(weights.clamp_min(1e-12)).unsqueeze(0)
        var = scale.unsqueeze(0).pow(2).clamp_min(1e-12)
        err2 = (target.unsqueeze(0).unsqueeze(-1) - f_samps.unsqueeze(-1)).pow(2)
        log_pdf = -0.5 * (LOG_2PI + torch.log(var) + err2 / var)
        log_mix = torch.logsumexp(log_pi + log_pdf, dim=-1)
        return log_mix.mean(dim=0)

    def _expected_log_prob_regime_mixture_quadrature(
        self, target, function_dist, weights, df, scale
    ):
        """Gauss-Hermite E_q log sum_r pi_r StudentT(y | f, scale_r, df_r)."""
        mu = function_dist.mean
        var = function_dist.variance
        std = var.clamp_min(1e-12).sqrt()

        gh_x, gh_w = self._get_gauss_hermite(mu.device, mu.dtype)
        Q = gh_x.shape[0]
        f_points = mu.unsqueeze(0) + math.sqrt(2.0) * std.unsqueeze(0) * gh_x.view(
            Q, 1, 1
        )

        log_pi = torch.log(weights.clamp_min(1e-12)).unsqueeze(0)
        dist = SliceableStudentT(
            df=df.unsqueeze(0),
            loc=f_points.unsqueeze(-1),
            scale=scale.unsqueeze(0),
        )
        log_pdf = dist.log_prob(target.unsqueeze(0).unsqueeze(-1))
        log_mix = torch.logsumexp(log_pi + log_pdf, dim=-1)

        w = gh_w.view(Q, 1, 1)
        return (w * log_mix).sum(dim=0)

    def _expected_log_prob_regime_gaussian_mixture_quadrature(
        self, target, function_dist, weights, scale
    ):
        """Gauss-Hermite E_q log sum_r pi_r Normal(y | f, scale_r)."""
        mu = function_dist.mean
        var = function_dist.variance
        std = var.clamp_min(1e-12).sqrt()

        gh_x, gh_w = self._get_gauss_hermite(mu.device, mu.dtype)
        Q = gh_x.shape[0]
        f_points = mu.unsqueeze(0) + math.sqrt(2.0) * std.unsqueeze(0) * gh_x.view(
            Q, 1, 1
        )

        log_pi = torch.log(weights.clamp_min(1e-12)).unsqueeze(0)
        comp_var = scale.unsqueeze(0).pow(2).clamp_min(1e-12)
        err2 = (target.unsqueeze(0).unsqueeze(-1) - f_points.unsqueeze(-1)).pow(2)
        log_pdf = -0.5 * (LOG_2PI + torch.log(comp_var) + err2 / comp_var)
        log_mix = torch.logsumexp(log_pi + log_pdf, dim=-1)

        w = gh_w.view(Q, 1, 1)
        return (w * log_mix).sum(dim=0)

    def _expected_log_prob_gaussian_analytic(self, target, function_dist, scale):
        """Analytic E_q log Normal(y | f, scale) for the non-mixture case."""
        mu = function_dist.mean
        var_f = function_dist.variance
        var_y = scale.pow(2).clamp_min(1e-12)
        return -0.5 * (
            LOG_2PI
            + torch.log(var_y)
            + ((target - mu).pow(2) + var_f) / var_y
        )

    def _expected_log_prob_quadrature(self, target, function_dist, df, scale):
        """
        Gauss-Hermite quadrature: deterministic integration of
        E_{N(mu,sigma²)}[ log StudentT(y | f, scale, df) ]
        using per-element marginal variance from q(f).
        """
        mu = function_dist.mean        # [N, T]
        var = function_dist.variance    # [N, T]  (diagonal of covariance)
        std = var.clamp_min(1e-12).sqrt()

        # Gauss-Hermite nodes and weights (cached on first call)
        gh_x, gh_w = self._get_gauss_hermite(mu.device, mu.dtype)
        Q = gh_x.shape[0]

        # Transform standard GH nodes to N(mu, sigma²):
        # f_q = mu + sqrt(2) * sigma * gh_x
        # (GH integrates over exp(-x²), so sqrt(2) maps to N(0,1))
        f_points = mu.unsqueeze(0) + math.sqrt(2.0) * std.unsqueeze(0) * gh_x.view(Q, 1, 1)
        # f_points: [Q, N, T]

        dist = SliceableStudentT(df=df, loc=f_points, scale=scale)
        lp = dist.log_prob(target)  # [Q, N, T]

        # Weighted sum: weights already include 1/sqrt(pi) normalisation
        w = gh_w.view(Q, 1, 1)  # [Q, 1, 1]
        return (w * lp).sum(dim=0)  # [N, T]

    def _get_gauss_hermite(self, device, dtype):
        """Return cached GH nodes and weights, normalised for E_{N(0,1)}."""
        if (
            not hasattr(self, "_gh_cache")
            or self._gh_cache[0].device != device
            or self._gh_cache[0].dtype != dtype
        ):
            nodes, weights = np.polynomial.hermite.hermgauss(self.gh_points)
            gh_x = torch.tensor(nodes, device=device, dtype=dtype)
            gh_w = torch.tensor(weights, device=device, dtype=dtype)
            gh_w = gh_w / math.sqrt(math.pi)  # normalise so weights sum to 1
            self._gh_cache = (gh_x, gh_w)
        return self._gh_cache


# put near other modules
class GatedConstMean(gpytorch.means.Mean):
    def __init__(self, R: int):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(R))  # per-regime offsets

    def forward(self, x):  # x[..., :R] are the gates
        g = x[..., : self.mu.numel()]  # (N,R)
        return (g @ self.mu).squeeze(-1)  # (N,)


class _GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, mean_module, covar_module):
        q = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2)
        )
        strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, q, learn_inducing_locations=True
        )
        super().__init__(strat)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ========================== STICK-BREAKING UTIL ==================================================


# ---------- stick-breaking (compute-time) ----------
def stick_break_from_logits(
    logits,  # shape [..., R-1]
    temp=1.0,
    sb_mode="residual",  # "residual" or "renorm"
    dustbin_idx=None,  # None => last
    topk_real=None,  # optional: keep top-k among REAL experts
    eps=1e-12,
):
    """
    Returns:
      pi: [..., R] on simplex.
      aux: dict with z, log_sig, log_one_minus_sig (useful for DP prior).
    Notes:
      - residual: last is exact leftover; we do NOT renormalize (keeps residual semantics).
      - renorm: we compute SB then renormalize (legacy behaviour).
      - If dustbin_idx is not last AND you didn't reorder experts at init,
        we swap columns to place the residual at dustbin_idx.
      - topk_real (if not None): we keep top-k among REAL experts (excluding dustbin),
        and fold the dropped mass into the dustbin.
    """
    Rm1 = logits.shape[-1]
    R = Rm1 + 1
    z = logits / max(temp, eps)

    # log(sigmoid(z)) and log(1-sigmoid(z)) using softplus identities
    log_sig = -F.softplus(-z)
    log_one_minus_sig = -F.softplus(z)

    # cumsum for product of (1 - v)
    cumsum = torch.cumsum(log_one_minus_sig, dim=-1)  # [..., R-1]
    pad0 = F.pad(cumsum, (1, 0), value=0.0)  # [..., R], exclusive prefix

    # first R-1 sticks
    log_pi_main = log_sig + pad0[..., :-1]  # [..., R-1]
    # residual (last)
    log_pi_last = pad0[..., -1:]  # [..., 1] = sum_j log(1-v_j)

    log_pi = torch.cat([log_pi_main, log_pi_last], dim=-1)  # [..., R]
    pi = torch.exp(log_pi).clamp_min(eps)

    if sb_mode == "renorm":
        s = pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pi = pi / s
    else:
        # residual mode: no renorm (preserve exact leftover)
        pass

    # If you did NOT reorder experts at init and dustbin_idx != R-1, swap columns here
    if (dustbin_idx is not None) and (dustbin_idx != R - 1):
        idx = list(range(R))
        idx[dustbin_idx], idx[-1] = idx[-1], idx[dustbin_idx]
        pi = pi[..., idx]

    # Optional top-k over real experts (exclude dustbin) and fold dropped mass into dustbin
    if (topk_real is not None) and (topk_real > 0):
        db = dustbin_idx if dustbin_idx is not None else (R - 1)
        # gather masks
        arng = torch.arange(R, device=pi.device)
        real_mask = arng != db
        pi_real = pi[..., real_mask]  # [..., R-1]
        pi_db = pi[..., db : db + 1]  # [..., 1]

        k = min(topk_real, pi_real.shape[-1])
        topv, topi = torch.topk(pi_real, k=k, dim=-1)
        keep_mask = torch.zeros_like(pi_real).scatter_(-1, topi, 1.0)
        kept = pi_real * keep_mask
        dropped = pi_real * (1.0 - keep_mask)
        pi_db = pi_db + dropped.sum(dim=-1, keepdim=True)

        # reassemble pi in original order (dustbin column stays at db)
        pi_new = pi.clone()
        pi_new[..., real_mask] = kept
        pi_new[..., db : db + 1] = pi_db
        pi = pi_new

    aux = {"z": z, "log_sig": log_sig, "log_one_minus_sig": log_one_minus_sig}
    return pi, aux


# ---------------------------------------------------------------------------
# Variational stick-breaking (Nalisnick & Smyth, 2016): Kumaraswamy posteriors
# on each stick length, with a closed-form KL against a Beta(1, α) prior.
# ---------------------------------------------------------------------------

# Euler-Mascheroni constant.
_EULER_MASCHERONI = 0.5772156649015329


def kumaraswamy_kl_beta1alpha(a, b, alpha_prior, num_terms: int = 11):
    """Closed-form KL[Kumaraswamy(a, b) || Beta(1, alpha_prior)].

    Series approximation following Nalisnick & Smyth (2016, eq. 11).
    All inputs broadcastable; returns a tensor with the broadcast shape.

    Verified KL == 0 when a == 1 and b == alpha_prior (Kuma(1, α) ≡ Beta(1, α)).

    Parameters
    ----------
    a, b : torch.Tensor
        Strictly positive Kumaraswamy parameters.
    alpha_prior : float | torch.Tensor
        Concentration of the Beta(1, α) prior (>0).
    num_terms : int
        Number of terms in the series approximation of the b·Σ term. The
        sum converges geometrically; 10–11 terms are typically sufficient.
    """
    if not torch.is_tensor(alpha_prior):
        alpha_prior = torch.as_tensor(alpha_prior, dtype=a.dtype, device=a.device)
    else:
        alpha_prior = alpha_prior.to(dtype=a.dtype, device=a.device)

    log_a = torch.log(a)
    log_b = torch.log(b)
    digamma_b = torch.digamma(b)
    inv_b = 1.0 / b

    # term1: (a-1)/a · (-γ - ψ(b) - 1/b)
    term1 = (a - 1.0) / a * (-_EULER_MASCHERONI - digamma_b - inv_b)
    # term2: log(a*b)
    term2 = log_a + log_b
    # term3: log B(1, α_prior) = -log α_prior
    term3 = -torch.log(alpha_prior)
    # term4: -(b-1)/b
    term4 = -(b - 1.0) * inv_b

    # term5: (α_prior - 1) · b · Σ_{m=1}^{M} B(m/a, b) / (m + a·b)
    base = term1 + term2 + term3 + term4

    # Short-circuit when alpha_prior == 1: the series prefactor
    # (alpha_prior - 1) is exactly 0, so the trailing sum vanishes. We
    # skip the entire series block to avoid retaining M lgamma activations
    # for backward (the dominant memory cost of this routine).
    if bool(torch.equal(alpha_prior, torch.ones_like(alpha_prior))):
        return base

    # Vectorised series across m: a single broadcast lgamma op instead of
    # M sequential adds, which is ~M-fold cheaper in retained activations.
    M = int(num_terms)
    m_idx = torch.arange(1, M + 1, device=a.device, dtype=a.dtype)
    a_e = a.unsqueeze(-1)
    b_e = b.unsqueeze(-1)
    m_over_a = m_idx / a_e
    log_beta_term = (
        torch.lgamma(m_over_a)
        + torch.lgamma(b_e)
        - torch.lgamma(m_over_a + b_e)
    )
    ab_e = (a * b).unsqueeze(-1)
    sum_term = (torch.exp(log_beta_term) / (m_idx + ab_e)).sum(dim=-1)
    term5 = (alpha_prior - 1.0) * b * sum_term

    return base + term5


def kumaraswamy_mean(a, b):
    """Closed-form mean E[v] for v ~ Kumaraswamy(a, b).

    E[v] = b · B(1 + 1/a, b) = exp(log b + lgamma(1+1/a) + lgamma(b) - lgamma(1+1/a+b)).
    """
    one_plus_inv_a = 1.0 + 1.0 / a
    log_mean = (
        torch.log(b)
        + torch.lgamma(one_plus_inv_a)
        + torch.lgamma(b)
        - torch.lgamma(one_plus_inv_a + b)
    )
    return torch.exp(log_mean)


def vsb_gates_from_ab(
    a,                 # [..., R-1] Kumaraswamy parameter a > 0
    b,                 # [..., R-1] Kumaraswamy parameter b > 0
    alpha_prior,       # scalar Beta(1, α) prior concentration
    sample: bool = True,
    sb_mode: str = "renorm",
    dustbin_idx=None,
    num_kl_terms: int = 11,
    eps: float = 1e-12,
    compute_kl: bool = True,
):
    """Variational stick-breaking gates with closed-form KL.

    Returns (pi, aux) where pi is the simplex over R = (R-1)+1 components.

    aux contains:
      - kuma_a, kuma_b, v: the Kumaraswamy parameters and the (sampled or
        deterministic-mean) stick lengths.
      - kl_elem: KL[Kuma(a,b) || Beta(1, α)] per stick (shape [..., R-1]).
      - log_one_minus_sig, z: provided for interface-compat with the
        existing residual SB regulariser code paths (z is the logit of v).
    """
    Rm1 = a.shape[-1]
    R = Rm1 + 1
    # The user-supplied default eps=1e-12 is too small for float32 upper
    # clamps: 1.0 - 1e-12 rounds back to 1.0, so log(1-v) can become -inf
    # and poison the backward pass. Use a dtype-aware floor for all open-
    # interval clamps in the reparameterised stick sample.
    finfo = torch.finfo(a.dtype)
    clamp_eps = max(float(eps), float(finfo.eps))

    if sample:
        # Kumaraswamy reparameterised sample: u ~ U(0,1), v = (1 - u^(1/b))^(1/a).
        # On CUDA pow can occasionally return tiny negative round-off when the
        # base is essentially zero; clamp the inner term before the second pow
        # so that the fractional exponent does not produce NaN.
        u = torch.rand_like(a).clamp(clamp_eps, 1.0 - clamp_eps)
        inv_a = (1.0 / a).clamp(max=50.0)
        inv_b = (1.0 / b).clamp(max=50.0)
        inner = (1.0 - u.pow(inv_b)).clamp(clamp_eps, 1.0 - clamp_eps)
        v = inner.pow(inv_a)
    else:
        v = kumaraswamy_mean(a, b)

    # Replace any non-finite samples with a safe mid-stick value before the
    # log-space construction. Belt-and-braces on top of the inner clamp above.
    v = torch.nan_to_num(v, nan=0.5, posinf=1.0 - clamp_eps, neginf=clamp_eps)
    v = v.clamp(clamp_eps, 1.0 - clamp_eps)

    log_v = torch.log(v)
    log_1mv = torch.log(1.0 - v)

    # SB construction in log-space.
    cumsum = torch.cumsum(log_1mv, dim=-1)               # [..., R-1]
    pad0 = F.pad(cumsum, (1, 0), value=0.0)              # [..., R]
    log_pi_main = log_v + pad0[..., :-1]                 # [..., R-1]
    log_pi_last = pad0[..., -1:]                         # [..., 1] residual
    log_pi = torch.cat([log_pi_main, log_pi_last], dim=-1)
    pi = torch.exp(log_pi)

    if sb_mode == "renorm":
        pi = pi.clamp_min(clamp_eps)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(clamp_eps)

    if (dustbin_idx is not None) and (dustbin_idx != R - 1):
        idx = list(range(R))
        idx[dustbin_idx], idx[-1] = idx[-1], idx[dustbin_idx]
        pi = pi[..., idx]

    # KL[Kuma(a,b) || Beta(1, alpha_prior)] per stick. Skipping it on
    # paths that discard `aux` (e.g. the past-window gates) saves both
    # forward FLOPs and the activations the series would otherwise retain
    # for backward.
    if compute_kl:
        kl_elem = kumaraswamy_kl_beta1alpha(a, b, alpha_prior, num_terms=num_kl_terms)
    else:
        kl_elem = None
    z = log_v - log_1mv  # logit of v, kept for downstream interface compat

    aux = {
        "kuma_a": a,
        "kuma_b": b,
        "v": v,
        "kl_elem": kl_elem,
        "log_one_minus_sig": log_1mv,
        "log_sig": log_v,
        "z": z,
    }
    return pi, aux
