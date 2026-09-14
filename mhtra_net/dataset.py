"""
VoiceBank+DEMAND dataset (Valentini-Botinhao et al.).

Expected layout after running resample.py (all files 16 kHz mono):

    <root>/
        clean_trainset_28spk_wav/   *.wav
        noisy_trainset_28spk_wav/   *.wav
        clean_testset_wav/          *.wav
        noisy_testset_wav/          *.wav

Files are paired by filename. A validation split is carved from the training
set by speaker (default: p226 and p287 held out, as is common practice).

Training can use random fixed-length crops (segment_seconds > 0) or full-length
utterances (segment_seconds = 0, as in the official GTCRN recipe). For the latter,
`LengthBucketSampler` groups utterances of similar duration into each batch so
that zero padding stays small, and `collate_variable_length` pads and returns
per-item lengths.
"""

import os
import random
from glob import glob

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from .logging_utils import get_logger

log = get_logger(__name__)


class VoiceBankDEMAND(Dataset):
    def __init__(self, root, split="train", segment_seconds=4.0, fs=16000,
                 val_speakers=("p226", "p287"), random_gain=True):
        assert split in ("train", "val", "test")
        self.fs = fs
        self.split = split
        self.segment = int(segment_seconds * fs) if segment_seconds and split == "train" else None
        self.random_gain = random_gain and split == "train"

        if split == "test":
            clean_dir = os.path.join(root, "clean_testset_wav")
            noisy_dir = os.path.join(root, "noisy_testset_wav")
        else:
            clean_dir = os.path.join(root, "clean_trainset_28spk_wav")
            noisy_dir = os.path.join(root, "noisy_trainset_28spk_wav")

        log.info("[%s] scanning %s", split, clean_dir)
        for d in (clean_dir, noisy_dir):
            if not os.path.isdir(d):
                log.error("[%s] directory does not exist: %s", split, d)

        all_names = sorted(os.path.basename(p) for p in glob(os.path.join(clean_dir, "*.wav")))
        names = all_names
        if split == "train":
            names = [n for n in names if n.split("_")[0] not in val_speakers]
            log.info("[train] %d of %d files kept (held out speakers %s)",
                     len(names), len(all_names), ", ".join(val_speakers))
        elif split == "val":
            names = [n for n in names if n.split("_")[0] in val_speakers]
            log.info("[val] %d of %d files kept (speakers %s)",
                     len(names), len(all_names), ", ".join(val_speakers))

        self.pairs = [(os.path.join(noisy_dir, n), os.path.join(clean_dir, n)) for n in names]
        missing = [nz for nz, _ in self.pairs if not os.path.exists(nz)]
        if not self.pairs:
            log.error("[%s] no wav files under %s - did you run resample.py and pass the right --data?",
                      split, clean_dir)
            raise FileNotFoundError(f"No wav files found under {clean_dir}")
        if missing:
            log.error("[%s] %d clean files have no noisy counterpart in %s (e.g. %s)",
                      split, len(missing), noisy_dir, os.path.basename(missing[0]))
            raise FileNotFoundError(f"{len(missing)} clean files have no noisy counterpart, e.g. {missing[0]}")

        seg = f"{segment_seconds:.1f}s random crops" if self.segment else "full-length utterances"
        log.info("[%s] ready: %d pairs | %s | random_gain=%s | %d Hz",
                 split, len(self.pairs), seg, self.random_gain, fs)
        log.debug("[%s] first pair: %s <- %s", split,
                  os.path.basename(self.pairs[0][1]), os.path.basename(self.pairs[0][0]))
        self._durations = None

    def __len__(self):
        return len(self.pairs)

    @property
    def durations(self):
        """Per-item length in samples (header scan only, cached). Used by LengthBucketSampler."""
        if self._durations is None:
            self._durations = np.array([sf.info(nz).frames for nz, _ in self.pairs], dtype=np.int64)
            d = self._durations / self.fs
            log.info("[%s] durations: min %.1fs  median %.1fs  max %.1fs  total %.1f min",
                     self.split, d.min(), np.median(d), d.max(), d.sum() / 60)
        return self._durations

    def _load(self, path):
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            log.debug("%s is %d-channel, averaging to mono", os.path.basename(path), wav.shape[1])
            wav = wav.mean(axis=1)
        if sr != self.fs:
            log.error("%s has sample rate %d, expected %d - run resample.py on this dataset first",
                      path, sr, self.fs)
            raise ValueError(f"{path}: sample rate {sr} != {self.fs}. Run resample.py first.")
        return wav

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy, clean = self._load(noisy_path), self._load(clean_path)
        n = min(len(noisy), len(clean))
        if len(noisy) != len(clean):
            log.debug("%s: length mismatch noisy=%d clean=%d, truncating to %d",
                      os.path.basename(noisy_path), len(noisy), len(clean), n)
        noisy, clean = noisy[:n], clean[:n]

        if self.segment is not None:
            if n >= self.segment:
                start = random.randint(0, n - self.segment)
                noisy, clean = noisy[start:start + self.segment], clean[start:start + self.segment]
            else:
                pad = self.segment - n
                noisy, clean = np.pad(noisy, (0, pad)), np.pad(clean, (0, pad))

        if self.random_gain:
            g = 10 ** (random.uniform(-6, 6) / 20)
            noisy, clean = noisy * g, clean * g

        log.debug("[%s] item %d: %s %d samples (%.2fs)", self.split, idx,
                  os.path.basename(noisy_path), len(noisy), len(noisy) / self.fs)
        return {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean),
            "name": os.path.basename(noisy_path),
        }


def collate_variable_length(batch):
    """Pads to the longest item in the batch and returns lengths (val/test, and full-utterance training)."""
    lengths = torch.tensor([len(b["noisy"]) for b in batch])
    T = int(lengths.max())
    noisy = torch.zeros(len(batch), T)
    clean = torch.zeros(len(batch), T)
    for i, b in enumerate(batch):
        noisy[i, :lengths[i]] = b["noisy"]
        clean[i, :lengths[i]] = b["clean"]
    return {"noisy": noisy, "clean": clean, "lengths": lengths, "name": [b["name"] for b in batch]}


class LengthBucketSampler(Sampler):
    """
    Batch sampler for full-utterance training. Every epoch: shuffle, sort within
    windows of `bucket_size` batches by duration, cut into batches, shuffle the
    batch order. Items inside a batch have similar lengths, so padding is small,
    while the epoch is still random.

    A batch holds at most `batch_size` items AND at most `max_seconds` of padded
    audio (0 = no limit). The budget keeps GPU memory flat: a batch of 15 s
    utterances gets fewer items than a batch of 2 s ones. Activation memory of
    the model scales with padded frames, so this is what actually bounds it.

    The batch list is rebuilt every epoch, so len() is the count of the most
    recent draw (DataLoader queries it once per epoch).
    """

    def __init__(self, durations, batch_size, bucket_size=50, seed=0, max_seconds=0.0, fs=16000):
        self.durations = np.asarray(durations)
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.max_samples = int(max_seconds * fs)
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.batches = self._draw()
        sizes = [len(b) for b in self.batches]
        log.info("LengthBucketSampler: %d batches | items/batch min %d median %d max %d | budget %s | "
                 "bucket window %d batches | ~%.1f%% zero padding",
                 len(self.batches), min(sizes), int(np.median(sizes)), max(sizes),
                 f"{max_seconds:.0f}s of padded audio" if self.max_samples else "none",
                 bucket_size, 100 * self._padding_fraction(self.batches))

    def _cut(self, chunk):
        """chunk: indices sorted by duration -> list of batches honouring count and audio budget."""
        batches, cur = [], []
        for idx in chunk:
            longest = max(self.durations[idx], self.durations[cur].max() if cur else 0)
            over_budget = self.max_samples and cur and longest * (len(cur) + 1) > self.max_samples
            if len(cur) == self.batch_size or over_budget:
                batches.append(cur)
                cur = []
            cur.append(int(idx))
        if cur:
            batches.append(cur)
        return batches

    def _draw(self):
        n = len(self.durations)
        order = self.rng.permutation(n)
        window = self.batch_size * self.bucket_size
        batches = []
        for start in range(0, n, window):
            chunk = order[start:start + window]
            chunk = chunk[np.argsort(self.durations[chunk], kind="stable")]
            batches.extend(self._cut(chunk))
        # drop the rare 1-item batch: BatchNorm cannot normalise a single (B,C,T,F)-> fine, but the
        # gradient is noisy; keep everything else so every epoch sees every file
        batches = [b for b in batches if len(b) > 1]
        self.rng.shuffle(batches)
        return batches

    def _padding_fraction(self, batches):
        padded = sum(self.durations[b].max() * len(b) for b in batches)
        real = sum(self.durations[b].sum() for b in batches)
        return 1.0 - real / max(padded, 1)

    def __iter__(self):
        if self.epoch > 0:
            self.batches = self._draw()
        self.epoch += 1
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
