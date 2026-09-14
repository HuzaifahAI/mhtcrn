"""STFT / iSTFT helpers matching the model's (B, F, T, 2) convention."""

import torch

from .logging_utils import get_logger

log = get_logger("mhtra_net.stft")


class STFT:
    def __init__(self, n_fft=512, hop_length=256, win_length=512, device="cpu"):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # sqrt-Hann: analysis * synthesis window = Hann, which satisfies COLA at 50% overlap
        self.window = torch.sqrt(torch.hann_window(win_length)).to(device)
        log.info("STFT: n_fft=%d hop=%d win=%d (%d freq bins, %.1f ms hop) on %s",
                 n_fft, hop_length, win_length, n_fft // 2 + 1, hop_length / 16.0, device)

    def to(self, device):
        self.window = self.window.to(device)
        log.debug("STFT window moved to %s", device)
        return self

    def stft(self, wav):
        """wav: (B, N) -> spec (B, F, T, 2)"""
        X = torch.stft(wav, self.n_fft, self.hop_length, self.win_length,
                       window=self.window, center=True, return_complex=True)
        spec = torch.stack([X.real, X.imag], dim=-1)
        log.debug("stft: %s -> %s", tuple(wav.shape), tuple(spec.shape))
        return spec

    def istft(self, spec, length=None):
        """spec: (B, F, T, 2) -> wav (B, N)"""
        X = torch.complex(spec[..., 0], spec[..., 1])
        wav = torch.istft(X, self.n_fft, self.hop_length, self.win_length,
                          window=self.window, center=True, length=length)
        log.debug("istft: %s -> %s", tuple(spec.shape), tuple(wav.shape))
        return wav

    __call__ = stft


def magnitude(spec, eps=1e-12):
    """(B,F,T,2) -> (B,F,T)"""
    return torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + eps)


if __name__ == "__main__":
    from .logging_utils import setup_logging

    setup_logging("DEBUG")
    s = STFT()
    wav = torch.randn(2, 16000)
    spec = s(wav)
    rec = s.istft(spec, length=wav.shape[-1])
    err = (wav - rec).abs().max().item()
    log.info("spec %s | max reconstruction error: %.3e (%s)",
             tuple(spec.shape), err, "OK" if err < 1e-4 else "TOO HIGH")
