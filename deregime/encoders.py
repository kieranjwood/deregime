from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

from .config import ConfigDict
from .modules import PositionalEncoding


# Encoder definitions used in the DeRegiME paper release.
# Supported backbones: PatchTST, DLinear, and TimeMixer.


class BaseEncoder(nn.Module, ABC):
    """Abstract Base Class for all sequence encoders."""

    def __init__(self, input_dim: int, output_dim: int, config: ConfigDict, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.seq_len = int(config.seq_len)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        All encoders must implement this method.

        Args:
            x (torch.Tensor): Input sequence, shape (batch, seq_len, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - sequence_output (Tensor): Features for each time step.
                - context_vector (Tensor): A summary of the sequence.
        """
        pass


class MarkAggregator(nn.Module):
    """Average-pool time marks onto the patch grid (P = (L - K)//S + 1)."""

    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=patch_len, stride=stride)

    def forward(self, marks: torch.Tensor) -> torch.Tensor:
        # marks: (B, L, M) -> (B, M, L)
        x = marks.permute(0, 2, 1)
        y = self.pool(x)  # (B, M, P)
        return y.permute(0, 2, 1)  # (B, P, M)


class PatchTSTEncoder(BaseEncoder):
    r"""
    PatchTST-style encoder (channel-independent, series-wise Transformer) adapted to your framework.

    • Channel-independent: each variate is patched and passed through a shared Transformer (no cross-channel attention).
    • Patching: x∈ℝ^{B,L,D} → per-channel patches of length P with stride S.
      Number of patch tokens Np = floor((L-P)/S) + 1.
    • Patch embedding: Linear(P→H), shared across channels.
    • Positional encoding + Transformer over the Np patch tokens.
    • Outputs:
        sequence_output: (B, Np, H)  — per-patch features, averaged over channels.
        context_vector : (B, H)      — last patch token per channel, then mean over channels.
    • No forecasting head; projection happens outside this module.

    Args in config.encoder_args:
      patch_len (int, default 16), stride (int, default=patch_len),
      n_head (int, default 4), n_layers (int, default 3), d_ff (int, default 4*H)
      dropout_rate from config.dropout_rate
    """

    class Patching(nn.Module):
        def __init__(self, seq_len, patch_len, stride, hidden_dim):
            super().__init__()
            assert seq_len >= patch_len, "Sequence length must be >= patch length."
            self.seq_len = int(seq_len)
            self.patch_len = int(patch_len)
            self.stride = int(stride)
            self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1
            self.patch_proj = nn.Linear(self.patch_len, hidden_dim)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
            # x: (B, L, D)
            B, L, D = x.shape
            assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"
            # unfold per channel → (B, D, Np, P)
            x_unf = x.permute(0, 2, 1).unfold(
                dimension=-1, size=self.patch_len, step=self.stride
            )
            B_, D_, Np, P = x_unf.shape
            assert Np == self.num_patches and P == self.patch_len
            # collapse channels and embed → (B*D, Np, H)
            x_tokens = x_unf.reshape(B_ * D_, Np, P)
            z = self.patch_proj(x_tokens)
            return z, D, L

    def __init__(self, input_dim: int, output_dim: int, config: ConfigDict, **kwargs):
        hidden_dim = output_dim
        super().__init__(input_dim, output_dim, config, **kwargs)

        args = config.encoder_args or {}
        patch_len = int(args.get("patch_len", 16))
        stride = int(args.get("stride", patch_len))
        n_head = int(args.get("n_head", 4))
        n_layers = int(args.get("n_layers", 3))
        # d_ff = int(args.get("d_ff", 4 * hidden_dim))
        d_ff = int(args.get("d_ff_mult_hidden", 2)) * hidden_dim
        p_drop = float(config.dropout_rate)

        assert hidden_dim % n_head == 0, "hidden_dim must be divisible by n_head"

        self.patching = self.Patching(config.seq_len, patch_len, stride, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, p_drop)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=p_drop,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.hidden_dim = hidden_dim
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = self.patching.num_patches
        # mark this encoder as channel-independent (returns per-channel rows)
        self.outputs_per_channel = True  # put in __init__ (a simple attribute)

    def _run_layers(self, z: torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(z)

    def forward(self, x: torch.Tensor, state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        B, L, D = x.shape

        # 1) per-channel patch tokens → (B*D, Np, H)
        z, D_ch, _ = self.patching(x)  # D_ch == D

        # 2) PE + Transformer over patch tokens.
        z = self.pos_encoder(z)  # (B*D, Np, H)
        z = self._run_layers(z)  # (B*D, Np, H)

        sequence_output = z  # (B*D, Np, H)  per-channel
        context_vector = z[:, -1, :]  # (B*D, H)      per-channel
        return sequence_output, context_vector



class DLinearEncoder(BaseEncoder):
    r"""
    DLinear-style encoder (channel-independent).

    The paper model decomposes each channel independently into seasonal and
    trend terms, then applies linear maps over the full lookback:

      seasonal, trend = decomp(x)
      forecast = Linear_seasonal(seasonal[1:L])
               + Linear_trend(trend[1:L])

    In this framework the encoder must emit hidden tokens rather than the
    final forecast distribution. The default "forecast_tokens" mode therefore
    uses the same DLinear temporal structure but maps each decomposed channel
    into pred_len hidden tokens. A downstream sequence-flatten head then maps
    those tokens to the requested forecast parameters. With a linear downstream
    head, the overall path remains a linear function of the decomposed input.

    Set encoder_args["dlinear_output_mode"] = "decomp_tokens" to recover the
    previous pointwise decomposition-token feature extractor.

    Returns:
      sequence_output: (B*D, T, H)   # T=pred_len by default, else seq_len
      context_vector : (B*D, H)      # last emitted token
    """

    class MovingAvg(nn.Module):
        """Centered moving average with endpoint replication, as in DLinear."""

        def __init__(self, kernel_size: int):
            super().__init__()
            assert kernel_size % 2 == 1, "Kernel size must be odd for centered MA."
            self.avg = nn.AvgPool1d(
                kernel_size=kernel_size,
                stride=1,
                padding=0,
            )
            self.pad = (kernel_size - 1) // 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, L, D). The official DLinear implementation repeats the
            # first/last observations before average pooling, rather than using
            # shorter edge windows.
            front = x[:, 0:1, :].repeat(1, self.pad, 1)
            end = x[:, -1:, :].repeat(1, self.pad, 1)
            x_pad = torch.cat([front, x, end], dim=1)
            return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)

    def __init__(self, input_dim: int, output_dim: int, config: ConfigDict, **kwargs):
        super().__init__(input_dim, output_dim, config, **kwargs)
        args = config.encoder_args or {}
        k = int(args.get("dlinear_kernel_size", 25))
        self.moving_avg = self.MovingAvg(kernel_size=k)
        self.output_mode = str(
            args.get("dlinear_output_mode", "forecast_tokens")
        ).lower()
        self.individual = bool(args.get("dlinear_individual", False))

        if self.output_mode in ("forecast", "forecast_tokens", "paper"):
            self.output_mode = "forecast_tokens"
            self.num_tokens = int(config.pred_len)
            out_features = self.num_tokens * output_dim
            if self.individual:
                self.Linear_Seasonal = nn.ModuleList(
                    [nn.Linear(self.seq_len, out_features) for _ in range(input_dim)]
                )
                self.Linear_Trend = nn.ModuleList(
                    [nn.Linear(self.seq_len, out_features) for _ in range(input_dim)]
                )
            else:
                self.Linear_Seasonal = nn.Linear(self.seq_len, out_features)
                self.Linear_Trend = nn.Linear(self.seq_len, out_features)
            self.per_channel_proj = None

        elif self.output_mode in ("decomp_tokens", "pointwise", "legacy"):
            self.output_mode = "decomp_tokens"
            self.num_tokens = int(config.seq_len)
            # Per-channel 2→H projection (shared across channels & time).
            # This preserves the old framework adapter, but is less faithful
            # than the DLinear paper's full-lookback linear projection.
            self.per_channel_proj = nn.Linear(2, output_dim, bias=True)
            self.Linear_Seasonal = None
            self.Linear_Trend = None
        else:
            raise ValueError(
                "encoder_args['dlinear_output_mode'] must be one of "
                "'forecast_tokens' or 'decomp_tokens'; "
                f"got {self.output_mode!r}"
            )

        # Mark this encoder as channel-independent for downstream heads.
        self.outputs_per_channel = True

    def forward(self, x: torch.Tensor, state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        B, L, D = x.shape

        # Decompose per channel
        trend = self.moving_avg(x)  # (B, L, D)
        seasonal = x - trend  # (B, L, D)

        if self.output_mode == "forecast_tokens":
            seasonal_ch = seasonal.permute(0, 2, 1).contiguous()  # (B, D, L)
            trend_ch = trend.permute(0, 2, 1).contiguous()  # (B, D, L)
            if self.individual:
                per_channel = []
                for i in range(D):
                    s_i = self.Linear_Seasonal[i](seasonal_ch[:, i, :])
                    t_i = self.Linear_Trend[i](trend_ch[:, i, :])
                    per_channel.append(
                        (s_i + t_i).view(B, self.num_tokens, self.output_dim)
                    )
                h = torch.stack(per_channel, dim=1)  # (B, D, T, H)
            else:
                s = self.Linear_Seasonal(seasonal_ch.reshape(B * D, L))
                t = self.Linear_Trend(trend_ch.reshape(B * D, L))
                sequence_output = (s + t).view(B * D, self.num_tokens, self.output_dim)
                context_vector = sequence_output[:, -1, :]
                return sequence_output, context_vector

            sequence_output = h.reshape(B * D, self.num_tokens, self.output_dim)
            context_vector = sequence_output[:, -1, :]
            return sequence_output, context_vector

        # Stack per-channel features → (B, L, D, 2) and apply shared 2→H map.
        feats_2 = torch.stack([seasonal, trend], dim=-1)  # (B, L, D, 2)
        h = self.per_channel_proj(feats_2)  # (B, L, D, H)

        # Expose per-channel rows to the rest of your pipeline
        sequence_output = h.view(B * D, L, self.output_dim)  # (B*D, L, H)
        context_vector = sequence_output[:, -1, :]  # (B*D, H)

        return sequence_output, context_vector



class TimeMixerEncoder(BaseEncoder):
    r"""
    TimeMixer-style encoder (Wang et al., ICLR 2024) adapted to this framework.

    Paper-faithful structure:
      - Channel-independent processing: (B, L, D) -> (B*D, L, 1) reshape, shared
        weights across channels (matches PatchTST/DLinear convention here).
      - Per-time embedding: Linear(1 -> H).
      - Multi-scale pyramid: M downsamplings (avg pool or stride conv) of the
        embedded sequence, producing scales x^0 (finest), x^1, ..., x^M.
      - Stack of PastDecomposableMixing (PDM) blocks. Each PDM:
          * decompose each scale into seasonal + trend (centered moving average);
          * mix seasonal components bottom-up (fine -> coarse) via Linear cascades;
          * mix trend components top-down (coarse -> fine) via Linear cascades;
          * recombine seasonal + trend, apply per-scale FFN, residual + LayerNorm.
      - Output: finest scale x^0 projected per-time to output_dim.

    Adaptations for the framework (intentional, minimal):
      - Drops the FMM (Future Multipredictor Mixing) forecaster head; downstream
        framework heads (Gaussian, Student-t, MDN, DKL, DeRegime) take its place.
      - Returns (B*D, L, output_dim) per-time features at the original resolution
        rather than direct forecasts.

    No variate-identity / channel embedding is added (paper does not include one,
    and this matches the channel-independent release convention).
    """

    class MovingAvg(nn.Module):
        """Centered moving average with endpoint replication; matches DLinear."""

        def __init__(self, kernel_size: int):
            super().__init__()
            assert kernel_size % 2 == 1, "Kernel size must be odd for centered MA."
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
            self.pad = (kernel_size - 1) // 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, L, H)
            front = x[:, 0:1, :].repeat(1, self.pad, 1)
            end = x[:, -1:, :].repeat(1, self.pad, 1)
            x_pad = torch.cat([front, x, end], dim=1)
            return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)

    class SeriesDecomp(nn.Module):
        """Split signal into seasonal + trend via centered moving average."""

        def __init__(self, kernel_size: int):
            super().__init__()
            self.ma = TimeMixerEncoder.MovingAvg(kernel_size)

        def forward(self, x: torch.Tensor):
            trend = self.ma(x)
            seasonal = x - trend
            return seasonal, trend

    class _TimeAxisMLP(nn.Module):
        """Two-layer MLP applied along the time axis of a (B, L_in, H) tensor.

        Implemented as Linear(L_in -> L_out) on the time-transposed input,
        followed by GELU and Linear(L_out -> L_out). Matches the paper's
        cross-scale linear cascades.
        """

        def __init__(self, l_in: int, l_out: int):
            super().__init__()
            self.proj1 = nn.Linear(l_in, l_out)
            self.proj2 = nn.Linear(l_out, l_out)
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (N, L_in, H)
            y = x.transpose(1, 2)              # (N, H, L_in)
            y = self.proj1(y)                  # (N, H, L_out)
            y = self.act(y)
            y = self.proj2(y)                  # (N, H, L_out)
            return y.transpose(1, 2).contiguous()

    class MultiScaleSeasonMixing(nn.Module):
        """Bottom-up (fine -> coarse) seasonal mixing across scales."""

        def __init__(self, seq_lens):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    TimeMixerEncoder._TimeAxisMLP(seq_lens[i], seq_lens[i + 1])
                    for i in range(len(seq_lens) - 1)
                ]
            )

        def forward(self, seasons):
            # seasons: list of (N, L_m, H), fine -> coarse
            out = [seasons[0]]
            cur = seasons[0]
            for i, layer in enumerate(self.layers):
                mix = layer(cur)               # (N, L_{m+1}, H)
                cur = seasons[i + 1] + mix
                out.append(cur)
            return out

    class MultiScaleTrendMixing(nn.Module):
        """Top-down (coarse -> fine) trend mixing across scales."""

        def __init__(self, seq_lens):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    TimeMixerEncoder._TimeAxisMLP(seq_lens[i + 1], seq_lens[i])
                    for i in range(len(seq_lens) - 1)
                ]
            )

        def forward(self, trends):
            # trends: list of (N, L_m, H), fine -> coarse
            out = [trends[-1]]
            cur = trends[-1]
            for i in reversed(range(len(self.layers))):
                mix = self.layers[i](cur)      # (N, L_i, H)
                cur = trends[i] + mix
                out.insert(0, cur)
            return out

    class PastDecomposableMixing(nn.Module):
        """One PDM layer.

        Decompose each scale into seasonal + trend; mix seasonal bottom-up and
        trend top-down across scales; recombine; apply per-scale FFN with outer
        residual and LayerNorm.
        """

        def __init__(self, seq_lens, hidden_dim, ma_kernel, d_ff_mult, dropout):
            super().__init__()
            self.decomp = TimeMixerEncoder.SeriesDecomp(ma_kernel)
            self.season_mix = TimeMixerEncoder.MultiScaleSeasonMixing(seq_lens)
            self.trend_mix = TimeMixerEncoder.MultiScaleTrendMixing(seq_lens)
            d_ff = max(hidden_dim * d_ff_mult, hidden_dim)
            self.ff = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, d_ff),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_ff, hidden_dim),
                    )
                    for _ in seq_lens
                ]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in seq_lens])

        def forward(self, scales):
            seasons, trends = zip(*[self.decomp(x) for x in scales])
            seasons = self.season_mix(list(seasons))
            trends = self.trend_mix(list(trends))
            out = []
            for i, (s, t) in enumerate(zip(seasons, trends)):
                z = s + t
                z = z + self.ff[i](z)
                z = self.norms[i](z + scales[i])
                out.append(z)
            return out

    def __init__(self, input_dim: int, output_dim: int, config: ConfigDict, **kwargs):
        super().__init__(input_dim, output_dim, config, **kwargs)
        args = config.encoder_args or {}

        # Hidden width: prefer kwargs (framework-passed), fall back to encoder_args,
        # then to the paper's small default of 16.
        hidden_dim = kwargs.get("hidden_dim") or int(args.get("hidden_dim", 16))
        self.hidden_dim = int(hidden_dim)

        n_layers = int(args.get("n_layers", 2))
        n_scales = int(args.get("down_sampling_layers", 3))
        down_window = int(args.get("down_sampling_window", 2))
        down_method = str(args.get("down_sampling_method", "avg")).lower()
        ma_kernel = int(args.get("moving_avg_kernel", 25))
        d_ff_mult = int(args.get("d_ff_mult_hidden", 2))
        dropout = float(config.dropout_rate)

        # Per-scale lengths, fine -> coarse.
        L = self.seq_len
        seq_lens = [L]
        for _ in range(n_scales):
            seq_lens.append(seq_lens[-1] // down_window)
        assert seq_lens[-1] >= 4, (
            f"TimeMixer coarsest scale length {seq_lens[-1]} too small "
            f"for seq_len={L}, down_sampling_layers={n_scales}, "
            f"down_sampling_window={down_window}; reduce one of these."
        )

        # Per-time, per-channel embedding (channel-independent: 1 -> H).
        self.embed = nn.Linear(1, self.hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Downsampling stack (one per coarsening step).
        if down_method == "conv":
            self.downsamplers = nn.ModuleList(
                [
                    nn.Conv1d(
                        self.hidden_dim,
                        self.hidden_dim,
                        kernel_size=down_window * 2 - 1,
                        stride=down_window,
                        padding=down_window - 1,
                    )
                    for _ in range(n_scales)
                ]
            )
        else:
            self.downsamplers = nn.ModuleList(
                [
                    nn.AvgPool1d(kernel_size=down_window, stride=down_window)
                    for _ in range(n_scales)
                ]
            )

        # Stack of PDM layers.
        self.pdm_layers = nn.ModuleList(
            [
                TimeMixerEncoder.PastDecomposableMixing(
                    seq_lens=seq_lens,
                    hidden_dim=self.hidden_dim,
                    ma_kernel=ma_kernel,
                    d_ff_mult=d_ff_mult,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Output projection per-time: H -> output_dim.
        self.out_proj = nn.Linear(self.hidden_dim, output_dim)

        # Channel-independent: each channel gets its own per-time row (B*D, L, H).
        self.outputs_per_channel = True

    def _build_pyramid(self, h: torch.Tensor):
        # h: (N, L, H) -> list fine -> coarse.
        scales = [h]
        cur = h
        for ds in self.downsamplers:
            cur_t = cur.transpose(1, 2)        # (N, H, L_m)
            cur_t = ds(cur_t)                  # (N, H, L_{m+1})
            cur = cur_t.transpose(1, 2).contiguous()
            scales.append(cur)
        return scales

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        B, L, D = x.shape
        # Channel-independent reshape: each channel becomes its own batch entry.
        h = x.permute(0, 2, 1).reshape(B * D, L, 1)
        h = self.embed(h)                      # (B*D, L, H)
        h = self.embed_dropout(h)

        scales = self._build_pyramid(h)        # list of (B*D, L_m, H)
        for pdm in self.pdm_layers:
            scales = pdm(scales)

        finest = scales[0]                     # (B*D, L, H)
        sequence_output = self.out_proj(finest)  # (B*D, L, output_dim)
        context_vector = sequence_output[:, -1, :]  # (B*D, output_dim)
        return sequence_output, context_vector


ENCODER_REGISTRY = {
    "patchtst": PatchTSTEncoder,
    "dlinear": DLinearEncoder,
    "timemixer": TimeMixerEncoder,
}


def get_encoder(encoder_type: str):
    try:
        return ENCODER_REGISTRY[encoder_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown encoder type: {encoder_type!r}. "
            f"Available: {sorted(ENCODER_REGISTRY)}"
        ) from exc
