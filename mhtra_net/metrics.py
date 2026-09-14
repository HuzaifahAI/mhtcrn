"""
Evaluation metrics - the ONE implementation used by evaluate.py and evaluate_any.py.

Previously the in-domain script (evaluate.py) and the out-of-domain script
(evaluate_any.py) computed "pause residual" and "speech LSD" differently
(mean of per-bin dB differences vs. energy ratio; log-domain vs. linear frame
mask; sqrt-Hann vs. Hann window). The two are not on the same scale, so the
in-domain and A/B/C rows of Table 6 were not comparable. Everything below
follows the definitions in the paper:

  Eq. 16  R_pause = 10 log10( E_{t in P} |s_hat_t|^2 / E_{t in P} |s_t|^2 )   [dB]
          P = frames whose clean power is > 40 dB below the loudest clean frame
  Eq. 17  LSD     = sqrt( E_{t not in P, f} (20 log10|S_hat| - 20 log10|S|)^2 )   [dB]

Both use the model's own analysis STFT (512 / 256, sqrt-Hann, center=True) so
that "frame" means the same thing everywhere.
"""

import numpy as np

SR = 16000
FLOOR_DB = 40.0

try:
    from pesq import pesq as _pesq
except ImportError:                                        # pesq needs a C++ compiler on Windows
    _pesq = None
try:
    from pystoi import stoi as _stoi
except ImportError:
    _stoi = None


# --------------------------------------------------------------------------- #
def _stft_mag(x, n_fft=512, hop=256):
    """(N,) float -> magnitude (F, T), sqrt-Hann, centred; matches mhtra_net.stft.STFT."""
    x = np.asarray(x, dtype=np.float64)
    win = np.sqrt(np.hanning(n_fft + 1)[:-1])              # periodic Hann, as torch.hann_window
    x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(x[idx] * win, axis=-1)).T   # (F, T)


def frame_masks(clean, floor_db=FLOOR_DB):
    """Boolean (pause, speech) frame masks from the clean reference (Eq. 16/17 threshold)."""
    P = _stft_mag(clean) ** 2
    frame_db = 10 * np.log10(P.sum(0) + 1e-12)
    pause = frame_db < frame_db.max() - floor_db
    return pause, ~pause


def pause_residual_db(est, ref, floor_db=FLOOR_DB, eps=1e-12):
    """Eq. 16: energy of the estimate over energy of the reference, in pause frames (dB)."""
    n = min(len(est), len(ref))
    E, R = _stft_mag(est[:n]) ** 2, _stft_mag(ref[:n]) ** 2
    pause, _ = frame_masks(ref[:n], floor_db)
    if pause.sum() < 3:
        return float("nan")
    return float(10 * np.log10((E[:, pause].mean() + eps) / (R[:, pause].mean() + eps)))


def speech_lsd_db(est, ref, floor_db=FLOOR_DB, eps=1e-8):
    """Eq. 17: RMS log-spectral distance over speech-active frames (dB)."""
    n = min(len(est), len(ref))
    E, R = _stft_mag(est[:n]), _stft_mag(ref[:n])
    _, speech = frame_masks(ref[:n], floor_db)
    if speech.sum() < 3:
        return float("nan")
    d = 20 * np.log10(E[:, speech] + eps) - 20 * np.log10(R[:, speech] + eps)
    return float(np.sqrt((d ** 2).mean()))


def si_sdr_db(est, ref, eps=1e-8):
    est = np.asarray(est, np.float64); ref = np.asarray(ref, np.float64)
    est = est - est.mean(); ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def true_snr_db(clean, noisy):
    noise = noisy - clean
    return float(10 * np.log10((clean ** 2).sum() / max((noise ** 2).sum(), 1e-12)))


def stoi_score(ref, deg):
    return float(_stoi(ref.astype(np.float64), deg.astype(np.float64), SR, extended=False)) if _stoi else float("nan")


def pesq_pair(pair):
    """Wide-band PESQ of one (ref, deg) pair; NaN if PESQ rejects it (logged by the caller)."""
    if _pesq is None:
        return float("nan")
    ref, deg = pair
    try:
        return float(_pesq(SR, ref.astype(np.float64), deg.astype(np.float64), "wb"))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
LOWER_IS_BETTER = {"pause_resid_db", "speech_lsd_db"}


def bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(this, ref, key):
    """
    Paired difference this - ref for one metric. Returns dict with mean diff, 95% CI,
    wins (count of files where THIS is better, respecting metric direction), n, verdict.
    """
    a = np.asarray(this, np.float64); b = np.asarray(ref, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if d.size < 2:
        return None
    lo, hi = bootstrap_ci(d)
    lower = key in LOWER_IS_BETTER
    wins = int((d < 0).sum()) if lower else int((d > 0).sum())
    if lower:
        verdict = "better" if hi < 0 else ("worse" if lo > 0 else "not significant")
    else:
        verdict = "better" if lo > 0 else ("worse" if hi < 0 else "not significant")
    return {"diff": float(d.mean()), "ci95": [lo, hi], "wins": wins, "n": int(d.size),
            "verdict": verdict, "this_mean": float(a[ok].mean()), "ref_mean": float(b[ok].mean())}
