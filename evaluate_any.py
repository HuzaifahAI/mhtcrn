#!/usr/bin/env python3
r"""
Evaluate a checkpoint on any test set laid out as <root>/clean and <root>/noisy
(matching filenames), e.g. the sets produced by build_testsets.py.

    python evaluate_any.py --root D:\data\testsets\libri_demand \
        --ckpt runs\mhtra_v5b\best.pt --csv runs\mhtra_v5b\libri_demand.csv

    # paired comparison against a baseline run on the same set
    python evaluate_any.py --root D:\data\testsets\libri_demand \
        --ckpt runs\mhtra_v5b\best.pt \
        --compare runs\base_v5b_s0\libri_demand.csv

Metrics: wide-band PESQ, STOI, SI-SDR, residual level in pauses (Eq. 16), speech-frame
log-spectral distance (Eq. 17) - all from mhtra_net.metrics, identical to evaluate.py.
Use this script on the VoiceBank+DEMAND test set too (--root with clean/ and noisy/)
if you want every column of Table 6 from literally the same code path. If the manifest has an snr_db column the results are also
broken down by SNR.
"""

import argparse
import csv
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from mhtra_net import metrics as M
from mhtra_net.model import MHTRANet, MHTRANetConfig, count_parameters
from mhtra_net.stft import STFT

SR = M.SR


# --------------------------------------------------------------------------- #
# all metrics come from mhtra_net.metrics so this script and evaluate.py agree exactly
si_sdr_np = M.si_sdr_db
pause_residual_db = M.pause_residual_db
speech_lsd_db = M.speech_lsd_db
_pesq_pair = M.pesq_pair
bootstrap_ci = M.bootstrap_ci
_pesq, _stoi = M._pesq, M._stoi


# --------------------------------------------------------------------------- #
def load_model(ckpt_path, device, official=False):
    ck = torch.load(ckpt_path, map_location=device)
    if official:
        model = MHTRANet(MHTRANetConfig(use_mhtra=False, use_se=False, learnable_erb=False)).to(device).eval()
        model.load_official_gtcrn(ck["model"])
        print(f"loaded OFFICIAL GTCRN {ckpt_path}: {count_parameters(model):,} params")
        return model
    cfg = MHTRANetConfig(**ck["cfg"]) if "cfg" in ck else MHTRANetConfig()
    model = MHTRANet(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {ckpt_path}: {count_parameters(model):,} params | epoch {ck.get('epoch', '?')}")
    return model


def read_manifest(root):
    """Returns {id: snr_db} if a manifest exists, else {}."""
    path = os.path.join(root, "manifest.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["id"]: float(r["snr_db"]) for r in csv.DictReader(f) if r.get("snr_db")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="test set with clean/ and noisy/ subfolders")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default=None, help="write per-file metrics here")
    ap.add_argument("--compare", nargs="*", default=[], help="per-file CSVs from other runs")
    ap.add_argument("--pesq_workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--save_wavs", default=None, help="also write enhanced audio here")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N files")
    ap.add_argument("--official", action="store_true", help="--ckpt is from the official GTCRN repo")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device, args.official)
    stft = STFT(device=device)

    clean_dir = os.path.join(args.root, "clean")
    noisy_dir = os.path.join(args.root, "noisy")
    for d in (clean_dir, noisy_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"missing directory: {d}")
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(clean_dir, "*.wav")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no wav files in {clean_dir}")
    snr_of = read_manifest(args.root)
    print(f"{len(files)} files from {args.root}"
          + (f" (SNRs from manifest: {sorted(set(snr_of.values()))})" if snr_of else ""))

    if args.save_wavs:
        os.makedirs(args.save_wavs, exist_ok=True)

    rows, pesq_pairs_enh, pesq_pairs_noisy = [], [], []
    t0 = time.perf_counter()
    total_seconds = 0.0

    with torch.no_grad():
        for fname in tqdm(files, desc="evaluating", unit="file"):
            clean, _ = sf.read(os.path.join(clean_dir, fname), dtype="float32")
            noisy, _ = sf.read(os.path.join(noisy_dir, fname), dtype="float32")
            n = min(len(clean), len(noisy))
            clean, noisy = clean[:n], noisy[:n]
            total_seconds += n / SR

            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            enh = stft.istft(model(stft(x)), length=n).squeeze(0).cpu().numpy()

            if args.save_wavs:
                sf.write(os.path.join(args.save_wavs, fname), enh, SR)

            row = {
                "id": os.path.splitext(fname)[0],
                "snr_db": snr_of.get(os.path.splitext(fname)[0], ""),
                "sisdr_noisy": si_sdr_np(noisy, clean),
                "sisdr_enh": si_sdr_np(enh, clean),
                "stoi_noisy": M.stoi_score(clean, noisy),
                "stoi_enh": M.stoi_score(clean, enh),
                "pause_resid_db": pause_residual_db(enh, clean),
                "pause_resid_db_noisy": pause_residual_db(noisy, clean),
                "speech_lsd_db": speech_lsd_db(enh, clean),
            }
            rows.append(row)
            pesq_pairs_enh.append((clean, enh))
            pesq_pairs_noisy.append((clean, noisy))

    # PESQ in a process pool (it is CPU-bound and releases no GIL)
    if _pesq is not None:
        t1 = time.perf_counter()
        if args.pesq_workers > 0:
            with ProcessPoolExecutor(max_workers=args.pesq_workers) as pool:
                enh_scores = list(pool.map(_pesq_pair, pesq_pairs_enh, chunksize=8))
                noisy_scores = list(pool.map(_pesq_pair, pesq_pairs_noisy, chunksize=8))
        else:
            enh_scores = [_pesq_pair(p) for p in pesq_pairs_enh]
            noisy_scores = [_pesq_pair(p) for p in pesq_pairs_noisy]
        for r, pe, pn in zip(rows, enh_scores, noisy_scores):
            r["pesq_enh"], r["pesq_noisy"] = pe, pn
        bad = sum(1 for r in rows if not np.isfinite(r["pesq_enh"]))
        print(f"PESQ scored in {time.perf_counter() - t1:.1f}s"
              + (f"  WARNING: PESQ NaN on {bad}/{len(rows)} files" if bad else ""))
    else:
        print("pesq not installed - skipping PESQ")
        for r in rows:
            r["pesq_enh"] = r["pesq_noisy"] = np.nan

    elapsed = time.perf_counter() - t0
    print(f"processed {len(rows)} files ({total_seconds / 60:.1f} min of audio) "
          f"in {elapsed:.1f}s ({total_seconds / elapsed:.1f}x real time)")

    def col(key):
        return np.asarray([r[key] for r in rows], dtype=float)

    def m(key):
        return float(np.nanmean(col(key)))

    print(f"\n{'':<12}{'PESQ':>8}{'STOI':>8}{'SI-SDR':>9}")
    print(f"{'noisy':<12}{m('pesq_noisy'):>8.3f}{m('stoi_noisy'):>8.3f}{m('sisdr_noisy'):>9.2f}")
    print(f"{'enhanced':<12}{m('pesq_enh'):>8.3f}{m('stoi_enh'):>8.3f}{m('sisdr_enh'):>9.2f}")
    print(f"{'delta':<12}{m('pesq_enh') - m('pesq_noisy'):>+8.3f}"
          f"{m('stoi_enh') - m('stoi_noisy'):>+8.3f}{m('sisdr_enh') - m('sisdr_noisy'):>+9.2f}")
    lo, hi = bootstrap_ci(col("pesq_enh"))
    print(f"enhanced PESQ 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")
    print(f"diagnostics: residual in pauses {m('pause_resid_db'):+.2f} dB above clean "
          f"(noisy input: {m('pause_resid_db_noisy'):+.2f} dB) | "
          f"speech-frame LSD {m('speech_lsd_db'):.2f} dB")

    snrs = sorted({r["snr_db"] for r in rows if r["snr_db"] != ""})
    if snrs:
        print(f"\nby input SNR:{'n':>6}{'PESQ noisy->enh':>22}{'STOI noisy->enh':>20}{'SI-SDR noisy->enh':>22}")
        for s in snrs:
            sub = [r for r in rows if r["snr_db"] == s]
            def sm(key):
                return float(np.nanmean([r[key] for r in sub]))
            print(f"{s:>9.1f} dB{len(sub):>6}"
                  f"{sm('pesq_noisy'):>12.3f} ->{sm('pesq_enh'):>7.3f}"
                  f"{sm('stoi_noisy'):>11.3f} ->{sm('stoi_enh'):>6.3f}"
                  f"{sm('sisdr_noisy'):>12.2f} ->{sm('sisdr_enh'):>7.2f}")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-file metrics -> {args.csv}")

    # ---- paired comparison against other runs on the same test set ---- #
    for ref_csv in args.compare:
        if not os.path.exists(ref_csv):
            print(f"\ncomparison file not found: {ref_csv}")
            continue
        with open(ref_csv, newline="") as f:
            ref = {r["id"]: r for r in csv.DictReader(f)}
        common = [r for r in rows if r["id"] in ref]
        if not common:
            print(f"\nno overlapping ids with {ref_csv}")
            continue
        name = os.path.basename(os.path.dirname(os.path.abspath(ref_csv)))
        print(f"\n=== paired comparison vs {name} ({len(common)} files) ===")
        print("  (wins = files where THIS run is better; lower is better for pause_resid_db / speech_lsd_db)")
        for key in ("pesq_enh", "stoi_enh", "sisdr_enh", "pause_resid_db", "speech_lsd_db"):
            this, other = [], []
            for r in common:
                try:
                    this.append(float(r[key])); other.append(float(ref[r["id"]][key]))
                except (ValueError, KeyError, TypeError):
                    this.append(np.nan); other.append(np.nan)
            st = M.paired_stats(this, other, key)
            if st is None:
                continue
            print(f"  {key:<15} this {st['this_mean']:.3f}  ref {st['ref_mean']:.3f}  diff {st['diff']:+.3f}  "
                  f"CI95 [{st['ci95'][0]:+.3f}, {st['ci95'][1]:+.3f}]  wins {st['wins']}/{st['n']}  -> {st['verdict']}")


if __name__ == "__main__":
    main()