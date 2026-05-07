import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PastFiLM(nn.Module):
    def __init__(
        self, mark_dim, hidden_dim, dropout=0.0, zero_init=False  # <--- NEW ARGUMENT
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(mark_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)
        )

        self.out_proj_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj_beta = nn.Linear(hidden_dim, hidden_dim)

        if zero_init:
            with torch.no_grad():
                self.out_proj_gamma.weight.fill_(0.0)
                self.out_proj_gamma.bias.fill_(0.0)
                self.out_proj_beta.weight.fill_(0.0)
                self.out_proj_beta.bias.fill_(0.0)

    def forward(self, marks):
        x = self.mlp(marks)
        gamma = self.out_proj_gamma(x)
        beta = self.out_proj_beta(x)
        return gamma, beta


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input tensor.
    This is the standard implementation from the "Attention Is All You Need" paper.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x.shape[1] is the sequence length (H)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class HorizonMLPEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        ctx_dim,
        hidden=64,
        depth=1,
        dropout=0.0,
        zero_init=False,  # <--- NEW ARGUMENT
    ):
        super().__init__()

        # 1. Project & Position
        self.input_projection = nn.Linear(in_dim, hidden)
        self.pos_encoder = PositionalEncoding(d_model=hidden, dropout=dropout)

        # 2. Body
        layers = []
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)])
        self.mlp_body = nn.Sequential(*layers)

        # 3. Output
        self.output_layer = nn.Linear(hidden, out_dim)

        # 4. FiLM Generators (Separate Layers)
        self.film_gamma = nn.Linear(ctx_dim, out_dim)
        self.film_beta = nn.Linear(ctx_dim, out_dim)

        # 5. Initialization Logic
        if zero_init:
            # Silent/Identity Start (Best for Mean)
            with torch.no_grad():
                self.output_layer.weight.normal_(0, 0.02)  # Keep output small
                self.output_layer.bias.fill_(0.0)

                self.film_gamma.weight.fill_(0.0)
                self.film_gamma.bias.fill_(0.0)
                self.film_beta.weight.fill_(0.0)
                self.film_beta.bias.fill_(0.0)
        else:
            # Loud/Random Start (Best for Experts)
            # PyTorch defaults (Kaiming Uniform) are perfect here.
            pass

    def forward(self, marks: torch.Tensor, ctx: torch.Tensor):
        x = self.input_projection(marks)
        x = self.pos_encoder(x)
        x = self.mlp_body(x)
        feats = self.output_layer(x)

        gamma = self.film_gamma(ctx).unsqueeze(1)
        beta = self.film_beta(ctx).unsqueeze(1)

        return feats * (1.0 + gamma) + beta


class PELinearProjection(nn.Module):
    """Linear future-decoder branch with horizon positional encoding.

    Drop-in replacement for HorizonMLPEncoder with future_decoder_type="pe_linear":
        y[h, :] = A * concat(ctx, PE(h)) + b
    where PE(h) is a fixed sinusoidal positional encoding at horizon h, A is a
    single nn.Linear shared across all horizons.

    The `marks` argument is accepted for API compatibility with HorizonMLPEncoder
    but is intentionally ignored. Under timeenc=2 (with per-window normalisation
    in deregime/data.py) the marks scalar is just (h+L)/(L+H) - a normalised
    horizon index identical for every window. Sinusoidal PE captures the same
    horizon-position information more cleanly and at higher dimensionality.

    Properties relative to HorizonMLPEncoder:
      * weights are shared across all horizons (no private per-horizon rows)
      * no MLP body, no FiLM, no per-regime nonlinearity
      * per-horizon variation is supplied entirely by PE(h)
    """

    def __init__(
        self,
        in_dim,            # accepted for API parity, unused
        out_dim,
        ctx_dim,
        hidden=64,         # interpreted as PE dimension
        depth=2,           # accepted for API parity, unused
        dropout=0.0,
        zero_init=False,
        max_horizon=512,
    ):
        super().__init__()
        self.pe_dim = int(hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(ctx_dim + self.pe_dim, out_dim)

        position = torch.arange(max_horizon, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.pe_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / self.pe_dim)
        )
        pe = torch.zeros(max_horizon, self.pe_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.pe_dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe, persistent=False)

        if zero_init:
            with torch.no_grad():
                self.linear.weight.fill_(0.0)
                self.linear.bias.fill_(0.0)

    def forward(self, marks: torch.Tensor, ctx: torch.Tensor):
        H = marks.size(1)
        pe = self.pe[:H].to(ctx.dtype).unsqueeze(0).expand(ctx.size(0), -1, -1)
        ctx_b = ctx.unsqueeze(1).expand(-1, H, -1)
        x = torch.cat([ctx_b, self.dropout(pe)], dim=-1)
        return self.linear(x)


class DirectMLPProjection(nn.Module):
    """Direct context-to-multihorizon MLP future decoder.

    Drop-in replacement for HorizonMLPEncoder with future_decoder_type="direct_mlp":

        y = MLP(ctx)                 # (B, H * out_dim)
        y = reshape(y, B, H, out_dim)

    This is the plain multi-output MLP baseline: no horizon marks, no FiLM
    modulation, and no shared horizon basis. It gives each branch a direct
    nonlinear map from the past summary to the full future feature path.
    """

    def __init__(
        self,
        in_dim,  # accepted for API parity, unused
        out_dim,
        ctx_dim,
        hidden=64,
        depth=2,
        dropout=0.0,
        zero_init=False,
        max_horizon=512,
    ):
        super().__init__()
        del in_dim

        self.out_dim = int(out_dim)
        self.max_horizon = int(max_horizon)
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")

        hidden = int(hidden)
        depth = max(1, int(depth))
        layers = []
        in_features = int(ctx_dim)
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(in_features, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden
        layers.append(nn.Linear(in_features, self.max_horizon * self.out_dim))
        self.net = nn.Sequential(*layers)

        if zero_init:
            final = self.net[-1]
            with torch.no_grad():
                final.weight.fill_(0.0)
                final.bias.fill_(0.0)

    def forward(self, marks: torch.Tensor, ctx: torch.Tensor):
        H = marks.size(1)
        if H > self.max_horizon:
            raise ValueError(
                f"Requested horizon H={H} exceeds max_horizon={self.max_horizon}"
            )
        y = self.net(ctx).view(ctx.size(0), self.max_horizon, self.out_dim)
        return y[:, :H, :]


class SequenceFlattenProjection(nn.Module):
    """PatchTST-style flattened-sequence multi-horizon projection.

    This consumes the encoder's full sequence output rather than the final
    context token:

        y = Linear(flatten(seq))   # (B, max_horizon * out_dim)

    For PatchTST encoders `seq` is the encoded patch grid, so this matches the
    supervised PatchTST head. For non-patched encoders `seq` is their native
    time-token sequence.
    """

    def __init__(
        self,
        token_count,
        hidden_dim,
        out_dim,
        max_horizon,
        dropout=0.0,
        zero_init=False,
    ):
        super().__init__()
        self.token_count = int(token_count)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.max_horizon = int(max_horizon)
        if self.token_count < 1:
            raise ValueError("token_count must be >= 1")
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(
            self.token_count * self.hidden_dim,
            self.max_horizon * self.out_dim,
        )

        if zero_init:
            with torch.no_grad():
                self.linear.weight.fill_(0.0)
                self.linear.bias.fill_(0.0)

    def forward(self, seq: torch.Tensor, horizon: int | None = None):
        # seq: (B, tokens, hidden)
        if seq.size(1) != self.token_count:
            raise ValueError(
                f"Expected {self.token_count} encoder tokens, got {seq.size(1)}"
            )
        if seq.size(2) != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {seq.size(2)}"
            )
        H = self.max_horizon if horizon is None else int(horizon)
        if H > self.max_horizon:
            raise ValueError(
                f"Requested horizon H={H} exceeds max_horizon={self.max_horizon}"
            )
        flat = self.dropout(seq.reshape(seq.size(0), -1))
        y = self.linear(flat).view(seq.size(0), self.max_horizon, self.out_dim)
        return y[:, :H, :]


class BasisLinearProjection(nn.Module):
    """Low-rank linear future decoder over a shared horizon basis.

    Drop-in replacement for HorizonMLPEncoder with
    future_decoder_type="basis_linear":

        coeff = Linear(ctx)                  # (B, K * out_dim)
        y[h] = sum_k basis[h, k] * coeff[k]  # (B, H, out_dim)

    This keeps a direct linear map from the past representation into future
    features, but ties all horizon-specific maps through K smooth basis vectors
    instead of learning an independent W_h for every horizon. The `marks`
    argument is accepted for API compatibility and is intentionally ignored.
    """

    def __init__(
        self,
        in_dim,  # accepted for API parity, unused
        out_dim,
        ctx_dim,
        hidden=64,  # accepted for API parity, unused
        depth=2,  # accepted for API parity, unused
        dropout=0.0,
        zero_init=False,
        max_horizon=512,
        basis_rank=None,
        basis_type="dct",
        learnable_basis=False,
    ):
        super().__init__()
        del in_dim, hidden, depth

        self.out_dim = int(out_dim)
        self.max_horizon = int(max_horizon)
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")

        if basis_rank is None:
            basis_rank = min(8, self.max_horizon)
        self.basis_rank = max(1, min(int(basis_rank), self.max_horizon))

        basis = self._build_basis(self.max_horizon, self.basis_rank, basis_type)
        if learnable_basis:
            self.basis = nn.Parameter(basis)
        else:
            self.register_buffer("basis", basis, persistent=False)

        self.dropout = nn.Dropout(dropout)
        self.coeff_proj = nn.Linear(ctx_dim, self.basis_rank * self.out_dim)

        if zero_init:
            with torch.no_grad():
                self.coeff_proj.weight.fill_(0.0)
                self.coeff_proj.bias.fill_(0.0)

    @staticmethod
    def _build_basis(max_horizon, basis_rank, basis_type):
        basis_type = str(basis_type).lower()
        H = int(max_horizon)
        K = int(basis_rank)
        h = torch.arange(H, dtype=torch.float)

        if basis_type == "dct":
            k = torch.arange(K, dtype=torch.float).unsqueeze(0)
            basis = torch.cos(math.pi * (h.unsqueeze(1) + 0.5) * k / float(H))
            basis[:, 0] *= math.sqrt(1.0 / H)
            if K > 1:
                basis[:, 1:] *= math.sqrt(2.0 / H)
            return basis

        if basis_type == "fourier":
            cols = [torch.ones(H)]
            freq = 1
            while len(cols) < K:
                phase = 2.0 * math.pi * freq * (h + 0.5) / float(H)
                cols.append(torch.sin(phase))
                if len(cols) < K:
                    cols.append(torch.cos(phase))
                freq += 1
            basis = torch.stack(cols[:K], dim=1)
            return F.normalize(basis, p=2, dim=0)

        raise ValueError(f"Unsupported future basis_type={basis_type!r}")

    def forward(self, marks: torch.Tensor, ctx: torch.Tensor):
        H = marks.size(1)
        if H > self.max_horizon:
            raise ValueError(
                f"Requested horizon H={H} exceeds max_horizon={self.max_horizon}"
            )
        basis = self.basis[:H].to(dtype=ctx.dtype)
        coeff = self.coeff_proj(self.dropout(ctx)).view(
            ctx.size(0), self.basis_rank, self.out_dim
        )
        return torch.einsum("hk,bko->bho", basis, coeff)


class BasisMLPProjection(nn.Module):
    """Nonlinear context-to-basis future decoder.

    Drop-in replacement for HorizonMLPEncoder with
    future_decoder_type="basis_mlp":

        coeff = MLP(ctx)                    # (B, K * out_dim)
        y[h] = sum_k basis[h, k] * coeff[k] # (B, H, out_dim)

    This keeps the smooth low-rank horizon bottleneck of BasisLinearProjection
    while allowing a nonlinear map from the past summary to basis coefficients.
    The `marks` argument is accepted for API compatibility and ignored.
    """

    def __init__(
        self,
        in_dim,  # accepted for API parity, unused
        out_dim,
        ctx_dim,
        hidden=64,
        depth=2,
        dropout=0.0,
        zero_init=False,
        max_horizon=512,
        basis_rank=None,
        basis_type="dct",
        learnable_basis=False,
    ):
        super().__init__()
        del in_dim

        self.out_dim = int(out_dim)
        self.max_horizon = int(max_horizon)
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")

        if basis_rank is None:
            basis_rank = min(8, self.max_horizon)
        self.basis_rank = max(1, min(int(basis_rank), self.max_horizon))

        basis = BasisLinearProjection._build_basis(
            self.max_horizon, self.basis_rank, basis_type
        )
        if learnable_basis:
            self.basis = nn.Parameter(basis)
        else:
            self.register_buffer("basis", basis, persistent=False)

        hidden = int(hidden)
        depth = max(1, int(depth))
        layers = []
        in_features = int(ctx_dim)
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(in_features, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden
        layers.append(nn.Linear(in_features, self.basis_rank * self.out_dim))
        self.coeff_net = nn.Sequential(*layers)

        if zero_init:
            final = self.coeff_net[-1]
            with torch.no_grad():
                final.weight.fill_(0.0)
                final.bias.fill_(0.0)

    def forward(self, marks: torch.Tensor, ctx: torch.Tensor):
        H = marks.size(1)
        if H > self.max_horizon:
            raise ValueError(
                f"Requested horizon H={H} exceeds max_horizon={self.max_horizon}"
            )
        basis = self.basis[:H].to(dtype=ctx.dtype)
        coeff = self.coeff_net(ctx).view(ctx.size(0), self.basis_rank, self.out_dim)
        return torch.einsum("hk,bko->bho", basis, coeff)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Reversible Instance Normalization.
        Standardizes the input window to mean 0, var 1.
        Restores statistics at the output.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # x shape: [Batch, Time, D]
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class IdentityRevIN(nn.Module):
    """
    A pass-through replacement for RevIN when use_revin=False.
    It mimics the API (mean, stdev, normalize, denormalize) but does nothing.
    """

    def __init__(self, num_features: int, **kwargs):
        super().__init__()
        self.affine = False  # Important: Tells train loop to skip affine logic

    def _get_statistics(self, x):
        # Create dummy stats: Mean=0, Std=1
        # x shape: [Batch, Time, D] -> Stats: [Batch, 1, D]
        B, L, D = x.shape
        self.mean = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        self.stdev = torch.ones(B, 1, D, device=x.device, dtype=x.dtype)

    def _normalize(self, x):
        return x  # Pass through

    def _denormalize(self, x):
        return x  # Pass through


class StatEncoder(nn.Module):
    """Projects RevIN statistics into a latent stationarity embedding."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # <--- Clean, fast, modern
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, mu, sigma):
        # 1. Log-Space Transform (Gaussianization)
        # mu, sigma: (B, 1, D)
        mu_log = torch.sign(mu) * torch.log1p(torch.abs(mu))
        sigma_log = torch.log(sigma + 1e-5)

        # 2. Concatenate & Encode
        stats = torch.cat([mu_log, sigma_log], dim=-1)  # (B, 1, 2D)
        return self.net(stats)  # (B, 1, Emb)


class StatFiLM(nn.Module):
    """Generates Scale (Gamma) and Shift (Beta) from stationarity embedding."""

    def __init__(self, embedding_dim, feature_dim):
        super().__init__()
        # Output 2x feature_dim (one for gamma, one for beta)
        # self.generator = nn.Linear(embedding_dim, feature_dim * 2)
        self.generator_gamma = nn.Linear(embedding_dim, feature_dim)
        self.generator_beta = nn.Linear(embedding_dim, feature_dim)

        # Init weights near zero so FiLM starts as Identity
        # with torch.no_grad():
        #     self.generator.weight.normal_(0, 0.02)
        #     self.generator.bias.fill_(0.0)

    def forward(self, stat_emb):
        # params = self.generator(stat_emb)  # (B, 1, 2F)
        # gamma, beta = params.chunk(2, dim=-1)
        gamma = self.generator_gamma(stat_emb)  # (B, 1, F)
        beta = self.generator_beta(stat_emb)  # (B, 1, F)
        return gamma, beta
