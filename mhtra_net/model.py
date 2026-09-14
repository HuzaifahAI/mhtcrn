"""
MHTRA-Net: an enhanced GTCRN for ultra-lightweight speech enhancement.

Base architecture (GTCRN, Rong et al. 2024):
    ERB filterbank -> SFE -> [ConvBlock x2, GTConvBlock x3] -> DPGRNN x2
    -> [GTConvBlock x3 (transposed), ConvBlock x2 (transposed)] -> inverse ERB
    -> complex ratio mask

Upgrades in this file:
    (1) MHTRA  - causal Multi-Head Temporal Recurrent Attention replacing TRA.
                 Shared GRU for temporal context, H heads with independent
                 Q/K/V projections, learnable per-head temperature, causal
                 (optionally windowed) attention, sigmoid gating.
    (2) LearnableERB - the ERB analysis/synthesis matrices are trainable. They
                 are parameterised through a softplus so the effective weights
                 stay non-negative (an auditory filterbank has no negative
                 taps), and regularised with an anchor to the ERB
                 initialisation. The earlier off-diagonal Gram ("orthogonality")
                 penalty has been removed: ERB bands are triangular and
                 deliberately overlapping, so penalising overlap directly
                 opposed the anchor and drove ~48% of the entries negative.
    (3) CausalSE - squeeze-and-excitation channel attention inside every
                 GTConvBlock. The "squeeze" is a frequency mean followed by a
                 causal running mean over time, so streaming causality is kept.

Input/output convention (same as GTCRN):
    spec: (B, F, T, 2) real/imag STFT with F = n_fft//2 + 1 = 257.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .logging_utils import get_logger

log = get_logger("mhtra_net.model")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class MHTRANetConfig:
    n_fft: int = 512
    fs: int = 16000
    erb_subband_1: int = 65      # low-frequency bins kept at full resolution
    erb_subband_2: int = 64      # number of ERB bands for the high frequencies
    channels: int = 16
    num_heads: int = 4
    attn_window: int | None = None   # None = full causal context; int = look-back frames
    use_mhtra: bool = True           # False -> original single-head TRA
    use_se: bool = True
    se_reduction: int = 2
    learnable_erb: bool = True
    zero_init_out: bool = False      # start with W_o = 0 so training begins exactly at TRA (mode-B remedy)


# --------------------------------------------------------------------------- #
# Helper: numerically safe inverse of softplus
# --------------------------------------------------------------------------- #
def inv_softplus(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x such that softplus(x) == t, for t > 0. Clamped so t=0 maps to a finite value."""
    t = t.clamp_min(eps)
    return t + torch.log(-torch.expm1(-t))


# --------------------------------------------------------------------------- #
# (2) Learnable ERB filterbank
# --------------------------------------------------------------------------- #
class LearnableERB(nn.Module):
    """
    ERB band-merging (bm) and band-splitting (bs).

    The stored parameters live in an unconstrained space; the effective
    filterbank is softplus(parameter), which is always >= 0.
    """

    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000,
                 fs=16000, learnable=True):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        self.erb_subband_1 = erb_subband_1
        self.erb_subband_2 = erb_subband_2
        self.learnable = learnable

        # analysis: (erb_subband_2, nfreqs - erb_subband_1); synthesis: transpose
        raw = inv_softplus(erb_filters.clone())
        raw_t = inv_softplus(erb_filters.t().clone().contiguous())

        self.erb_weight = nn.Parameter(raw.clone(), requires_grad=learnable)
        self.ierb_weight = nn.Parameter(raw_t.clone(), requires_grad=learnable)
        # anchors live in the same (unconstrained) space as the parameters
        self.register_buffer("erb_init", raw.clone())
        self.register_buffer("ierb_init", raw_t.clone())

        log.debug("LearnableERB: %d low bins + %d ERB bands, filterbank %s, %s "
                  "(softplus-parameterised, non-negative)",
                  erb_subband_1, erb_subband_2, tuple(erb_filters.shape),
                  "trainable" if learnable else "frozen")

    # ---- filterbank construction ---------------------------------------- #
    @staticmethod
    def hz2erb(freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    @staticmethod
    def erb2hz(erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
            / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) \
                / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) \
                / (bins[i + 2] - bins[i + 1] + 1e-12)
        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1 - erb_filters[-2, bins[-2]:bins[-1] + 1]

        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    # ---- effective (constrained) weights -------------------------------- #
    def analysis_weight(self):
        return F.softplus(self.erb_weight)

    def synthesis_weight(self):
        return F.softplus(self.ierb_weight)

    # ---- band merge / split --------------------------------------------- #
    def bm(self, x):
        """Band merge. x: (B,C,T,F) -> (B,C,T,erb_subband_1+erb_subband_2)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = F.linear(x[..., self.erb_subband_1:], self.analysis_weight())
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """Band split. x_erb: (B,C,T,F_erb) -> (B,C,T,F)"""
        x_low = x_erb[..., :self.erb_subband_1]
        x_high = F.linear(x_erb[..., self.erb_subband_1:], self.synthesis_weight())
        return torch.cat([x_low, x_high], dim=-1)

    def regularization_loss(self, anchor_weight=1.0, ortho_weight=0.0):
        """
        Anchor only by default. `ortho_weight` is accepted for backwards
        compatibility but should be left at 0: the old off-diagonal Gram penalty
        punished the band overlap that defines an ERB filterbank, fought the
        anchor term, and pushed the matrix negative.
        """
        if not self.learnable:
            return self.erb_weight.new_zeros(())
        loss = anchor_weight * (F.mse_loss(self.erb_weight, self.erb_init)
                                + F.mse_loss(self.ierb_weight, self.ierb_init))
        if ortho_weight:
            W = self.analysis_weight()
            gram = W @ W.t()
            off = gram - torch.diag(torch.diagonal(gram))
            n = gram.shape[0]
            loss = loss + ortho_weight * off.pow(2).sum() / (n * (n - 1))
        return loss

    @torch.no_grad()
    def drift_from_init(self):
        """Mean absolute drift of the effective filterbank from its ERB initialisation."""
        return (self.analysis_weight() - F.softplus(self.erb_init)).abs().mean()


# --------------------------------------------------------------------------- #
# Subband Feature Extraction
# --------------------------------------------------------------------------- #
class SFE(nn.Module):
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride),
                                padding=(0, (kernel_size - 1) // 2))

    def forward(self, x):
        """x: (B,C,T,F) -> (B,C*k,T,F)"""
        B, C, T, Fq = x.shape
        return self.unfold(x).reshape(B, C * self.kernel_size, T, Fq)


# --------------------------------------------------------------------------- #
# Attention modules
# --------------------------------------------------------------------------- #
class TRA(nn.Module):
    """Original single-head Temporal Recurrent Attention (kept for ablation)."""

    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)

    def forward(self, x):
        zt = x.pow(2).mean(dim=-1)                      # (B,C,T)
        at = self.att_gru(zt.transpose(1, 2))[0]        # (B,T,2C)
        at = torch.sigmoid(self.att_fc(at)).transpose(1, 2)  # (B,C,T)
        return x * at.unsqueeze(-1)


class MHTRA(nn.Module):
    """
    (1) Causal Multi-Head Temporal Recurrent Attention.

    zt   = mean_f x^2                     frame energy per channel      (B,T,C)
    ctx  = FC(GRU(zt))                    recurrent temporal context     (B,T,C)
    q,k,v = W_q ctx, W_k ctx, W_v ctx     split into H heads
    A_h  = softmax( (q_h k_h^T) * exp(tau_h) / sqrt(d) + causal_mask )
    out  = W_o concat_h(A_h v_h)
    gate = sigmoid(ctx + out)             (B,T,C) -> broadcast over F
    y    = x * gate

    NOTE on head_dim: with channels=16 the processed half is 8, so num_heads=4
    gives head_dim=2. Dot-product similarity in 2 dimensions carries very little
    information; --heads 2 (dim 4) or --heads 1 (dim 8) are worth ablating.
    """

    def __init__(self, channels, num_heads=4, window=None, zero_init_out=False):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.window = window
        if self.head_dim < 4:
            log.debug("MHTRA: head_dim=%d is small (channels=%d, heads=%d)",
                      self.head_dim, channels, num_heads)

        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        # learnable per-head temperature (log-parameterised, init = 1.0)
        self.log_temp = nn.Parameter(torch.zeros(num_heads))

        if zero_init_out:
            # gate = sigmoid(ctx + 0) at step 0 == the original TRA; the attention
            # contribution then grows only under gradient pressure
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def attention_macs(self, T):
        """
        Multiply-accumulates of the two bare matmuls (q k^T and A v) for T frames.
        Profilers such as thop hook nn.Linear/Conv/GRU but NOT torch.matmul, so
        without this the attention core is counted as zero. Full causal context is
        computed as a dense T x T product (then masked); a look-back window of w
        frames bounds it to T*w.
        """
        L = T if self.window is None else min(T, self.window)
        return 2 * T * L * self.channels

    def _causal_mask(self, T, device):
        mask = torch.ones(T, T, dtype=torch.bool, device=device).triu(1)   # future
        if self.window is not None:
            mask = mask | torch.ones(T, T, dtype=torch.bool, device=device).tril(-self.window)
        return mask

    def forward(self, x):
        """x: (B,C,T,F)"""
        B, C, T, Fq = x.shape
        H, D = self.num_heads, self.head_dim

        zt = x.pow(2).mean(dim=-1).transpose(1, 2)      # (B,T,C)
        ctx = self.att_fc(self.att_gru(zt)[0])          # (B,T,C)

        q = self.q_proj(ctx).view(B, T, H, D).transpose(1, 2)   # (B,H,T,D)
        k = self.k_proj(ctx).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(ctx).view(B, T, H, D).transpose(1, 2)

        scale = torch.exp(self.log_temp).view(1, H, 1, 1) / math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale   # (B,H,T,T)
        scores = scores.masked_fill(self._causal_mask(T, x.device), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                              # (B,H,T,D)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)

        gate = torch.sigmoid(ctx + out)                          # (B,T,C)
        return x * gate.transpose(1, 2).unsqueeze(-1)


# --------------------------------------------------------------------------- #
# (3) Causal squeeze-and-excitation
# --------------------------------------------------------------------------- #
class CausalSE(nn.Module):
    """Channel attention whose squeeze is a causal running mean over time."""

    def __init__(self, channels, reduction=2):
        super().__init__()
        hidden = max(channels // reduction, 2)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        """x: (B,C,T,F)"""
        B, C, T, Fq = x.shape
        s = x.mean(dim=-1)                                      # (B,C,T) squeeze over F
        denom = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, T)
        s = s.cumsum(dim=-1) / denom                            # causal running mean over T
        s = s.transpose(1, 2)                                   # (B,T,C)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))        # excitation
        return x * s.transpose(1, 2).unsqueeze(-1)


# --------------------------------------------------------------------------- #
# Convolutional blocks
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Grouped Temporal Convolution block with SE + MHTRA on the processed half."""

    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding,
                 dilation, use_deconv=False, cfg: MHTRANetConfig | None = None):
        super().__init__()
        cfg = cfg or MHTRANetConfig()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        half = in_channels // 2

        self.sfe = SFE(kernel_size=3, stride=1)

        self.point_conv1 = conv_module(half * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, half, 1)
        self.point_bn2 = nn.BatchNorm2d(half)

        self.se = CausalSE(half, cfg.se_reduction) if cfg.use_se else nn.Identity()
        if cfg.use_mhtra:
            self.tra = MHTRA(half, cfg.num_heads, cfg.attn_window, cfg.zero_init_out)
        else:
            self.tra = TRA(half)

    @staticmethod
    def shuffle(x1, x2):
        """Channel shuffle of two (B,C,T,F) halves -> (B,2C,T,F)"""
        B, C, T, Fq = x1.shape
        x = torch.stack([x1, x2], dim=2)            # (B,C,2,T,F)
        return x.reshape(B, 2 * C, T, Fq)

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = F.pad(h1, [0, 0, self.pad_size, 0])            # causal padding in time
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1 = self.se(h1)
        h1 = self.tra(h1)

        return self.shuffle(h1, x2)


# --------------------------------------------------------------------------- #
# Grouped dual-path RNN
# --------------------------------------------------------------------------- #
class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        if h is None:
            n = self.num_layers * (2 if self.bidirectional else 1)
            h = torch.zeros(n, x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        y1, h1 = self.rnn1(x1, h1.contiguous())
        y2, h2 = self.rnn2(x2, h2.contiguous())
        return torch.cat([y1, y2], dim=-1), torch.cat([h1, h2], dim=-1)


class DPGRNN(nn.Module):
    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size, hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size, hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x):
        """x: (B,C,T,F)"""
        x = x.permute(0, 2, 3, 1)                                   # (B,T,F,C)
        B, T, Fq, C = x.shape

        intra = x.reshape(B * T, Fq, C)
        intra = self.intra_fc(self.intra_rnn(intra)[0])
        intra = self.intra_ln(intra.reshape(B, T, Fq, C))
        intra_out = x + intra

        inter = intra_out.permute(0, 2, 1, 3).reshape(B * Fq, T, C)  # (B*F,T,C) causal over T
        inter = self.inter_fc(self.inter_rnn(inter)[0])
        inter = inter.reshape(B, Fq, T, C).permute(0, 2, 1, 3)
        inter = self.inter_ln(inter)
        out = intra_out + inter

        return out.permute(0, 3, 1, 2)                              # (B,C,T,F)


# --------------------------------------------------------------------------- #
# Encoder / Decoder / Mask
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, cfg: MHTRANetConfig):
        super().__init__()
        C = cfg.channels
        self.layers = nn.ModuleList([
            ConvBlock(3 * 3, C, (1, 5), stride=(1, 2), padding=(0, 2)),
            ConvBlock(C, C, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            GTConvBlock(C, C, (3, 3), (1, 1), (0, 1), (1, 1), cfg=cfg),
            GTConvBlock(C, C, (3, 3), (1, 1), (0, 1), (2, 1), cfg=cfg),
            GTConvBlock(C, C, (3, 3), (1, 1), (0, 1), (5, 1), cfg=cfg),
        ])

    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return x, outs


class Decoder(nn.Module):
    def __init__(self, cfg: MHTRANetConfig):
        super().__init__()
        C = cfg.channels
        self.layers = nn.ModuleList([
            GTConvBlock(C, C, (3, 3), (1, 1), (2 * 5, 1), (5, 1), use_deconv=True, cfg=cfg),
            GTConvBlock(C, C, (3, 3), (1, 1), (2 * 2, 1), (2, 1), use_deconv=True, cfg=cfg),
            GTConvBlock(C, C, (3, 3), (1, 1), (2 * 1, 1), (1, 1), use_deconv=True, cfg=cfg),
            ConvBlock(C, C, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True),
            ConvBlock(C, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True),
        ])

    def forward(self, x, en_outs):
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x + en_outs[n - 1 - i])
        return x


class ComplexRatioMask(nn.Module):
    def forward(self, mask, spec):
        """mask, spec: (B,2,T,F)"""
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        return torch.stack([s_real, s_imag], dim=1)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class MHTRANet(nn.Module):
    def __init__(self, cfg: MHTRANetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or MHTRANetConfig()
        nfreqs = cfg.n_fft // 2 + 1
        erb_width = cfg.erb_subband_1 + cfg.erb_subband_2          # 129
        dp_width = (erb_width + 3) // 4                              # two stride-2 layers -> 33

        self.erb = LearnableERB(cfg.erb_subband_1, cfg.erb_subband_2, cfg.n_fft,
                                cfg.fs // 2, cfg.fs, learnable=cfg.learnable_erb)
        self.sfe = SFE(3, 1)
        self.encoder = Encoder(cfg)
        self.dpgrnn1 = DPGRNN(cfg.channels, dp_width, cfg.channels)
        self.dpgrnn2 = DPGRNN(cfg.channels, dp_width, cfg.channels)
        self.decoder = Decoder(cfg)
        self.mask = ComplexRatioMask()
        self.nfreqs = nfreqs

        head_dim = (cfg.channels // 2) // cfg.num_heads if cfg.use_mhtra else 0
        log.info("MHTRANet built: %s params | %d freq bins -> %d ERB bands -> %d DPGRNN width",
                 f"{count_parameters(self):,}", nfreqs, erb_width, dp_width)
        log.info("  MHTRA=%s (heads=%d, head_dim=%d, window=%s)  SE=%s  learnable_ERB=%s  channels=%d",
                 cfg.use_mhtra, cfg.num_heads, head_dim, cfg.attn_window or "full causal",
                 cfg.use_se, cfg.learnable_erb, cfg.channels)
        for name, module in [("erb", self.erb), ("encoder", self.encoder),
                             ("dpgrnn1", self.dpgrnn1), ("dpgrnn2", self.dpgrnn2),
                             ("decoder", self.decoder)]:
            log.debug("  %-8s %10s params", name, f"{count_parameters(module, False):,}")

    def forward(self, spec):
        """spec: (B,F,T,2) -> enhanced spec (B,F,T,2)"""
        log.debug("forward: input spec %s", tuple(spec.shape))
        spec_ref = spec.permute(0, 3, 2, 1)                          # (B,2,T,F)
        real = spec_ref[:, 0]
        imag = spec_ref[:, 1]
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
        feat = torch.stack([mag, real, imag], dim=1)                # (B,3,T,F)

        feat = self.erb.bm(feat)                                     # (B,3,T,129)
        feat = self.sfe(feat)                                        # (B,9,T,129)
        log.debug("  after ERB band-merge + SFE: %s", tuple(feat.shape))

        feat, en_outs = self.encoder(feat)                           # (B,C,T,33)
        log.debug("  after encoder: %s (%d skip connections)", tuple(feat.shape), len(en_outs))
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)
        log.debug("  after 2x DPGRNN: %s", tuple(feat.shape))
        m_feat = self.decoder(feat, en_outs)                         # (B,2,T,129)

        m = self.erb.bs(m_feat)                                      # (B,2,T,257)
        spec_enh = self.mask(m, spec_ref)                            # (B,2,T,F)
        log.debug("  mask %s -> enhanced spec %s", tuple(m.shape), tuple(spec_enh.shape))
        return spec_enh.permute(0, 3, 2, 1)                          # (B,F,T,2)

    def regularization_loss(self, anchor_weight=1.0, ortho_weight=0.0):
        return self.erb.regularization_loss(anchor_weight, ortho_weight)

    def attention_macs(self, T):
        """Uncounted matmul MACs of all MHTRA modules for a T-frame input (0 for the baseline)."""
        return sum(m.attention_macs(T) for m in self.modules() if isinstance(m, MHTRA))

    def load_official_gtcrn(self, state_dict):
        """
        Load a state dict from the official repo (Xiaobin-Rong/gtcrn, checkpoints/*.tar['model'])
        into this model. Requires the baseline configuration (use_mhtra=False, use_se=False).
        Key differences: encoder.en_convs -> encoder.layers, decoder.de_convs -> decoder.layers,
        erb.erb_fc.weight -> erb.erb_weight (through inv_softplus), same for ierb.
        """
        assert not self.cfg.use_mhtra and not self.cfg.use_se, \
            "official weights only fit MHTRANetConfig(use_mhtra=False, use_se=False, learnable_erb=False)"
        new = {}
        for k, v in state_dict.items():
            k = k.replace("encoder.en_convs.", "encoder.layers.").replace("decoder.de_convs.", "decoder.layers.")
            if k == "erb.erb_fc.weight":
                new["erb.erb_weight"] = inv_softplus(v); new["erb.erb_init"] = inv_softplus(v)
            elif k == "erb.ierb_fc.weight":
                new["erb.ierb_weight"] = inv_softplus(v); new["erb.ierb_init"] = inv_softplus(v)
            else:
                new[k] = v
        missing, unexpected = self.load_state_dict(new, strict=False)
        if missing or unexpected:
            raise KeyError(f"official checkpoint mismatch: missing={missing} unexpected={unexpected}")
        log.info("loaded official GTCRN weights (%d tensors)", len(new))
        return self


def count_parameters(model: nn.Module, trainable_only=True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


if __name__ == "__main__":
    from .logging_utils import log_stage, setup_logging

    setup_logging("INFO")
    log.info("=== MHTRA-Net self test ===")

    for name, cfg in {
        "GTCRN-baseline": MHTRANetConfig(use_mhtra=False, use_se=False, learnable_erb=False),
        "MHTRA-Net": MHTRANetConfig(),
        "MHTRA-Net h2": MHTRANetConfig(num_heads=2),
    }.items():
        with log_stage(log, f"shape test: {name}"):
            model = MHTRANet(cfg).eval()
            x = torch.randn(1, 257, 63, 2)
            with torch.no_grad():
                y = model(x)
            log.info("%-16s params=%s  in=%s out=%s", name, f"{count_parameters(model):,}",
                     tuple(x.shape), tuple(y.shape))
            assert y.shape == x.shape, f"output shape {tuple(y.shape)} != input {tuple(x.shape)}"

    # the softplus parameterisation must reproduce the ERB init exactly at step 0
    with log_stage(log, "ERB round-trip check"):
        erb = LearnableERB(65, 64, 512, 8000, 16000, learnable=True)
        ref = erb.erb_filter_banks(65, 64, 512, 8000, 16000)
        err = (erb.analysis_weight() - ref).abs().max().item()
        log.info("max |softplus(param) - erb_init| = %.2e -> %s", err,
                 "OK" if err < 1e-4 else "MISMATCH")
        neg = (erb.analysis_weight() < 0).float().mean().item()
        log.info("fraction of negative entries: %.4f (must be 0)", neg)

    # causality check: perturbing frame t must not change outputs before t
    with log_stage(log, "causality check"):
        model = MHTRANet().eval()
        x = torch.randn(1, 257, 40, 2)
        x2 = x.clone()
        x2[:, :, 30:] += 5.0
        with torch.no_grad():
            d = (model(x) - model(x2)).abs()[:, :, :30].max().item()
        log.info("max |diff| on frames before perturbation: %.2e -> %s", d,
                 "CAUSAL" if d < 1e-5 else "NOT CAUSAL (expected ~0)")

    log.info("=== self test finished ===")