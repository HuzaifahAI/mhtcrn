"""
Training losses for MHTRA-Net.

total = w_mrstft * MultiResolutionSTFT
      + w_time   * (negative SI-SDR, in dB)
      + w_complex* CompressedComplexLoss (mag^0.3 domain)
      + w_erb    * ERB regularisation (anchor + orthogonality)

Matching the official GTCRN HybridLoss (gtcrn/loss.py):

    official = 30*(mse(real_c) + mse(imag_c)) + 70*mse(mag_c) - log10(SI-SNR ratio)

  * the SI-SNR term is -log10(.) i.e. SI-SNR in *bels* = dB/10. NegSISDRLoss below
    returns dB, so the official weight is  w_time = 0.1  (NOT 1.0).
  * real and imaginary MSEs are computed separately and summed. CompressedComplexLoss
    now does the same, so  w_complex=100, mag_weight=0.7, complex_weight=0.3  gives
    exactly 70*mag + 30*(real + imag).
  * the official loss has no magnitude floor; mag_floor defaults to 0 (off).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
class SpectralConvergence(nn.Module):
    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-8)


class LogSTFTMagnitude(nn.Module):
    def forward(self, x_mag, y_mag):
        return F.l1_loss(torch.log(y_mag + 1e-7), torch.log(x_mag + 1e-7))


class STFTLoss(nn.Module):
    def __init__(self, fft_size, hop_size, win_length):
        super().__init__()
        self.fft_size, self.hop_size, self.win_length = fft_size, hop_size, win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.sc = SpectralConvergence()
        self.lm = LogSTFTMagnitude()

    def _mag(self, x):
        X = torch.stft(x, self.fft_size, self.hop_size, self.win_length,
                       window=self.window, return_complex=True)
        return torch.sqrt(X.real ** 2 + X.imag ** 2 + 1e-9)

    def forward(self, x, y):
        x_mag, y_mag = self._mag(x), self._mag(y)
        return self.sc(x_mag, y_mag), self.lm(x_mag, y_mag)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(1024, 2048, 512), hop_sizes=(120, 240, 50),
                 win_lengths=(600, 1200, 240)):
        super().__init__()
        self.losses = nn.ModuleList([STFTLoss(f, h, w) for f, h, w in zip(fft_sizes, hop_sizes, win_lengths)])
        log.debug("multi-resolution STFT loss: fft=%s hop=%s win=%s", fft_sizes, hop_sizes, win_lengths)

    def forward(self, x, y):
        sc, lm = 0.0, 0.0
        for loss in self.losses:
            s, l = loss(x, y)
            sc, lm = sc + s, lm + l
        n = len(self.losses)
        return sc / n + lm / n


# --------------------------------------------------------------------------- #
def si_sdr(est, ref, eps=1e-8):
    """Scale-invariant SDR in dB. est, ref: (B, N)"""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10(target.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)


class NegSISDRLoss(nn.Module):
    """Negative SI-SDR in dB. The official GTCRN loss uses bels (dB/10): weight it by 0.1."""

    def forward(self, est, ref):
        return -si_sdr(est, ref).mean()


# --------------------------------------------------------------------------- #
class CompressedComplexLoss(nn.Module):
    """
    Power-law compressed complex + magnitude MSE (compression factor 0.3).
    spec: (B, F, T, 2)

    Real and imaginary parts get *separate* MSEs that are summed, exactly as in
    the official GTCRN HybridLoss (30*(real_loss + imag_loss) + 70*mag_loss).
    The previous version took one MSE over the stacked (...,2) tensor, which is
    (real+imag)/2 and silently halved the complex weight.

    mag_floor > 0 clamps the magnitude before the power (bounds the mag^(c-1)
    gradient in near-silent bins). The official loss has no such floor and that
    gradient pressure on silent bins is part of what makes it suppress pauses,
    so it is OFF by default; keep it as an explicit ablation only.
    """

    def __init__(self, c=0.3, mag_weight=0.7, complex_weight=0.3, mag_floor=0.0):
        super().__init__()
        self.c = c
        self.mag_weight = mag_weight
        self.complex_weight = complex_weight
        self.mag_floor = mag_floor

    def _compress(self, spec):
        mag = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-12)
        if self.mag_floor > 0:
            mag = mag.clamp_min(self.mag_floor)
        mag_c = mag.pow(self.c)
        # mag_c * (spec / mag), written without the division blowing up
        spec_c = (mag_c / mag).unsqueeze(-1) * spec
        return mag_c, spec_c

    def forward(self, est_spec, ref_spec):
        est_mag, est_c = self._compress(est_spec)
        ref_mag, ref_c = self._compress(ref_spec)
        mag_loss = F.mse_loss(est_mag, ref_mag)
        real_loss = F.mse_loss(est_c[..., 0], ref_c[..., 0])
        imag_loss = F.mse_loss(est_c[..., 1], ref_c[..., 1])
        return self.mag_weight * mag_loss + self.complex_weight * (real_loss + imag_loss)


# --------------------------------------------------------------------------- #
class MHTRANetLoss(nn.Module):
    def __init__(self, w_mrstft=0.0, w_time=0.1, w_complex=100.0, w_erb=0.0,
                 erb_anchor=1.0, erb_ortho=0.0, mag_floor=0.0):
        super().__init__()
        self.mrstft = MultiResolutionSTFTLoss()
        self.time = NegSISDRLoss()
        self.complex = CompressedComplexLoss(mag_floor=mag_floor)
        self.w = dict(mrstft=w_mrstft, time=w_time, complex=w_complex, erb=w_erb)
        self.erb_anchor, self.erb_ortho = erb_anchor, erb_ortho
        log.info("loss weights: mrstft=%.3g time=%.3g complex=%.3g erb=%.3g (erb anchor=%.3g ortho=%.3g)",
                 w_mrstft, w_time, w_complex, w_erb, erb_anchor, erb_ortho)

    def forward(self, est_wav, ref_wav, est_spec, ref_spec, model=None):
        # terms with weight 0 are skipped entirely (the multi-resolution STFT alone costs three
        # extra STFT pairs per step), so they neither show in the logs nor cost compute
        parts = {}
        if self.w["mrstft"]:
            parts["mrstft"] = self.mrstft(est_wav, ref_wav)
        if self.w["time"]:
            parts["time"] = self.time(est_wav, ref_wav)
        if self.w["complex"]:
            parts["complex"] = self.complex(est_spec, ref_spec)
        if self.w["erb"] and model is not None and hasattr(model, "regularization_loss"):
            parts["erb"] = model.regularization_loss(self.erb_anchor, self.erb_ortho)
        if not parts:
            raise ValueError("every loss weight is 0 - nothing to optimise")
        total = sum(self.w[k] * v for k, v in parts.items())
        parts = {k: v.detach() for k, v in parts.items()}
        if log.isEnabledFor(logging.DEBUG):
            log.debug("loss total=%.4f | %s", float(total.detach()),
                      "  ".join(f"{k}={float(v):.4f}" for k, v in parts.items()))
            if not torch.isfinite(total.detach()):
                log.error("non-finite loss - parts: %s", {k: float(v) for k, v in parts.items()})
        return total, parts

    @torch.enable_grad()
    def grad_norms(self, est_wav, ref_wav, est_spec, ref_spec, model):
        """
        Per-term gradient norm w.r.t. the model parameters. Called for the first
        few steps to check which term actually drives training.
        """
        out = {}
        _, parts_raw = None, {}
        if self.w["complex"]:
            parts_raw["complex"] = self.complex(est_spec, ref_spec)
        if self.w["time"]:
            parts_raw["time"] = self.time(est_wav, ref_wav)
        if self.w["mrstft"]:
            parts_raw["mrstft"] = self.mrstft(est_wav, ref_wav)
        params = [p for p in model.parameters() if p.requires_grad]
        for name, term in parts_raw.items():
            g = torch.autograd.grad(self.w[name] * term, params,
                                    retain_graph=True, allow_unused=True)
            g = [x for x in g if x is not None]
            out[name] = float(torch.norm(torch.stack([x.norm() for x in g]))) if g else 0.0
        return out
