#!/usr/bin/env python3
"""
Evaluate a checkpoint on the VoiceBank+DEMAND test set.

    python evaluate.py --data /path/VoiceBank_DEMAND_16k --ckpt runs/mhtra/best.pt
    python evaluate.py --data ... --ckpt runs/mhtra/best.pt --compare runs/gtcrn_official/test_metrics.csv

Per file (CSV next to the checkpoint, or --csv):
    snr_db                 true input SNR from clean and noisy (the test set uses 17.5/12.5/7.5/2.5 dB)
    pesq/stoi/sisdr        wide-band PESQ, STOI, SI-SDR for noisy input and enhanced output
    pause_resid_db         Eq. 16: enhanced / clean energy in pause frames (dB); under-suppression > 0
    speech_lsd_db          Eq. 17: log-spectral distance to clean in speech frames (dB)
    (both from mhtra_net.metrics - the same code evaluate_any.py uses, so in-domain and
    out-of-domain diagnostics are directly comparable)

    --official  load a checkpoint from the official GTCRN repo (checkpoints/model_trained_on_vctk.tar)

Summary (console + <ckpt dir>/test_summary.json):
    overall means with a 95% bootstrap CI on enhanced PESQ, a breakdown by the four
    SNR levels, the suppression / distortion diagnostics, and - with --compare - a
    paired comparison against another run's CSV (mean difference with CI, win counts,
    per-SNR difference), which is the number that says whether a change is real.

PESQ is scored in --pesq_workers processes. --log_level DEBUG logs every file.
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from mhtra_net.dataset import VoiceBankDEMAND
from mhtra_net.logging_utils import (add_logging_args, format_duration, get_logger, log_args,
                                     log_environment, log_stage, setup_logging)
from mhtra_net import metrics as M
from mhtra_net.model import MHTRANet, MHTRANetConfig, count_parameters
from mhtra_net.stft import STFT

log = get_logger("evaluate")
pesq = M._pesq

SNR_LEVELS = (2.5, 7.5, 12.5, 17.5)
METRICS = ["pesq", "stoi", "sisdr"]


def load_model(ckpt_path, device, weights="model", official=False):
    ck = torch.load(ckpt_path, map_location=device)
    if official:
        cfg = MHTRANetConfig(use_mhtra=False, use_se=False, learnable_erb=False)
        model = MHTRANet(cfg).to(device)
        model.load_official_gtcrn(ck["model"])
        log.info("official GTCRN checkpoint loaded from %s", ckpt_path)
        return model.eval(), cfg
    if "cfg" in ck:
        cfg = MHTRANetConfig(**ck["cfg"])
    else:
        cfg = MHTRANetConfig()
        log.warning("checkpoint has no 'cfg' entry - assuming default MHTRANetConfig")
    if weights == "ema":
        if "ema" not in ck:
            raise KeyError(f"{ckpt_path} has no EMA weights")
        sd = ck["ema"]["model"]
    elif weights == "raw":
        sd = ck.get("raw_model", ck["model"])
    else:
        sd = ck["model"]
    val = ck.get("val", {}).get(ck.get("weights", "raw"), {})
    log.info("checkpoint from epoch %s | stored weights: %s | val PESQ %s | val SI-SDR %s dB",
             ck.get("epoch", "?"), ck.get("weights", "?"),
             f"{val['pesq']:.3f}" if "pesq" in val and np.isfinite(val["pesq"]) else "?",
             f"{val['sisdr']:.2f}" if "sisdr" in val else "?")
    model = MHTRANet(cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


_pesq_pair = M.pesq_pair
true_snr_db = M.true_snr_db
bootstrap_ci = M.bootstrap_ci


def estimate_macs(model, device, seconds=1.0, fs=16000):
    """
    MACs per second of audio for a `seconds`-long input. thop counts Linear/Conv/GRU
    but not torch.matmul, so the attention core (q k^T, A v) is added analytically -
    it scales with T^2 for full causal context, so `seconds` MUST be reported with the
    number. Returns (total, thop_part, attention_part), all per second.
    """
    try:
        from thop import profile
    except ImportError:
        log.warning("thop not installed - skipping the MACs estimate (pip install thop)")
        return None
    frames = int(seconds * fs / 256) + 1
    x = torch.randn(1, 257, frames, 2, device=device)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    attn = model.attention_macs(frames)
    return (macs + attn) / seconds, macs / seconds, attn / seconds


def nearest_level(snr):
    return SNR_LEVELS[int(np.argmin([abs(snr - l) for l in SNR_LEVELS]))]


def load_csv(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["name"] == "AVERAGE":
                continue
            rows[r["name"]] = {k: float(v) if v not in ("", "nan") else float("nan")
                               for k, v in r.items() if k != "name"}
    return rows


def summarise(rows):
    """Console tables + dict for test_summary.json."""
    def mean(key, subset=None):
        vals = [r[key] for r in (subset or rows) if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    out = {"files": len(rows), "files_with_pesq": int(sum(np.isfinite(r["pesq_enh"]) for r in rows))}
    lo, hi = bootstrap_ci([r["pesq_enh"] for r in rows])
    log.info("                PESQ    STOI   SI-SDR")
    for tag in ("noisy", "enh"):
        log.info("%-10s   %6.3f  %6.3f  %7.2f", "enhanced" if tag == "enh" else tag,
                 mean(f"pesq_{tag}"), mean(f"stoi_{tag}"), mean(f"sisdr_{tag}"))
        out[tag] = {m: mean(f"{m}_{tag}") for m in METRICS}
    log.info("delta        %+6.3f  %+6.3f  %+7.2f", out["enh"]["pesq"] - out["noisy"]["pesq"],
             out["enh"]["stoi"] - out["noisy"]["stoi"], out["enh"]["sisdr"] - out["noisy"]["sisdr"])
    log.info("enhanced PESQ 95%% bootstrap CI: [%.3f, %.3f]", lo, hi)
    out["enh"]["pesq_ci95"] = [lo, hi]

    log.info("by input SNR:   n    PESQ noisy->enh    STOI noisy->enh    SI-SDR noisy->enh")
    out["by_snr"] = {}
    for level in SNR_LEVELS:
        sub = [r for r in rows if r["snr_level"] == level]
        if not sub:
            continue
        log.info("  %4.1f dB   %4d    %.3f -> %.3f      %.3f -> %.3f      %5.2f -> %5.2f", level, len(sub),
                 mean("pesq_noisy", sub), mean("pesq_enh", sub), mean("stoi_noisy", sub), mean("stoi_enh", sub),
                 mean("sisdr_noisy", sub), mean("sisdr_enh", sub))
        out["by_snr"][str(level)] = {"n": len(sub), **{f"{m}_{t}": mean(f"{m}_{t}", sub)
                                                        for m in METRICS for t in ("noisy", "enh")}}

    out["diag"] = {"pause_resid_db": mean("pause_resid_db"), "speech_lsd_db": mean("speech_lsd_db"),
                   "pause_resid_noisy_db": mean("pause_resid_noisy_db")}
    log.info("diagnostics: residual in pauses %+.2f dB above clean (noisy input: %+.2f dB) | "
             "speech-frame LSD %.2f dB", out["diag"]["pause_resid_db"], out["diag"]["pause_resid_noisy_db"],
             out["diag"]["speech_lsd_db"])
    return out


def compare(rows, ref_path):
    """Paired comparison of this run against another run's per-file CSV."""
    ref = load_csv(ref_path)
    common = [r for r in rows if r["name"] in ref]
    if len(common) < len(rows):
        log.warning("compare: only %d/%d files present in %s", len(common), len(rows), ref_path)
    if not common:
        return None
    label = os.path.basename(os.path.dirname(os.path.abspath(ref_path))) or ref_path
    log.info("=== paired comparison vs %s (%d files) ===", label, len(common))
    log.info("  (wins = files where THIS run is better; lower is better for pause_resid_db / speech_lsd_db)")
    out = {"reference": ref_path, "files": len(common)}
    for key in ("pesq_enh", "stoi_enh", "sisdr_enh", "pause_resid_db", "speech_lsd_db"):
        if key not in ref[common[0]["name"]]:
            continue
        st = M.paired_stats([r[key] for r in common], [ref[r["name"]][key] for r in common], key)
        if st is None:
            continue
        log.info("  %-14s this %.3f  ref %.3f  diff %+.3f  CI95 [%+.3f, %+.3f]  wins %d/%d  -> %s",
                 key, st["this_mean"], st["ref_mean"], st["diff"], st["ci95"][0], st["ci95"][1],
                 st["wins"], st["n"], st["verdict"])
        out[key] = st
    log.info("  PESQ difference by input SNR:")
    out["pesq_by_snr"] = {}
    for level in SNR_LEVELS:
        sub = [r for r in common if r["snr_level"] == level]
        if not sub:
            continue
        d = np.array([r["pesq_enh"] - ref[r["name"]]["pesq_enh"] for r in sub])
        d = d[np.isfinite(d)]
        log.info("    %4.1f dB  n=%3d  this %.3f  ref %.3f  diff %+.3f", level, len(sub),
                 np.nanmean([r["pesq_enh"] for r in sub]), np.nanmean([ref[r["name"]]["pesq_enh"] for r in sub]),
                 d.mean())
        out["pesq_by_snr"][str(level)] = float(d.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", choices=["model", "raw", "ema"], default="model",
                    help="which weights to load: 'model' = what the checkpoint was saved as (best.pt: the "
                         "winner of raw/EMA), or force raw / ema")
    ap.add_argument("--save_wavs", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--compare", nargs="*", default=[],
                    help="per-file CSVs of other runs to compare against (paired, with confidence intervals)")
    ap.add_argument("--pesq_workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--official", action="store_true",
                    help="--ckpt is a checkpoint from the official GTCRN repo (model_trained_on_vctk.tar)")
    ap.add_argument("--macs_seconds", type=float, default=1.0,
                    help="input length used for the MACs/s estimate (attention cost grows with length)")
    ap.add_argument("--log_every", type=int, default=200,
                    help="log a progress line every N files (0 disables)")
    add_logging_args(ap)
    args = ap.parse_args()

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    setup_logging(args.log_level, args.log_file or os.path.join(ckpt_dir, "evaluate.log"))
    log.info("=== MHTRA-Net evaluation ===")
    log_args(log, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_environment(log, device)
    if pesq is None:
        log.warning("the `pesq` package is not installed - PESQ columns will be NaN "
                    "(it needs a C++ compiler on Windows; STOI and SI-SDR are unaffected)")

    with log_stage(log, f"loading checkpoint {args.ckpt}"):
        model, cfg = load_model(args.ckpt, device, args.weights, args.official)
    stft = STFT(device=device)
    with log_stage(log, "loading test set"):
        test_set = VoiceBankDEMAND(args.data, "test")
    log.info("loaded %s: %s params | cfg=%s", args.ckpt, f"{count_parameters(model):,}", cfg)

    macs_all = estimate_macs(model, device, seconds=args.macs_seconds)
    macs = macs_all[0] if macs_all else None
    if macs_all:
        log.info("~%.2f MMACs per second of audio at %.0f s input (%.2f counted by thop + %.2f attention "
                 "matmuls) - report the input length with this number", macs_all[0] / 1e6, args.macs_seconds,
                 macs_all[1] / 1e6, macs_all[2] / 1e6)

    if args.save_wavs:
        os.makedirs(args.save_wavs, exist_ok=True)
        log.info("enhanced wavs will be written to %s", args.save_wavs)

    rows, pesq_jobs = [], []             # pesq_jobs: (row index, "noisy"/"enh", (clean, deg))
    t0 = time.perf_counter()
    audio_seconds = 0.0
    with torch.no_grad():
        for i, item in enumerate(tqdm(test_set, desc="evaluating", unit="file")):
            noisy, clean = item["noisy"], item["clean"]
            spec = stft(noisy[None].to(device))
            enh = stft.istft(model(spec), length=noisy.shape[-1])[0].cpu().numpy()
            clean_np, noisy_np = clean.numpy(), noisy.numpy()
            audio_seconds += len(clean_np) / 16000

            snr = true_snr_db(clean_np, noisy_np)
            row = dict(name=item["name"], snr_db=snr, snr_level=nearest_level(snr),
                       pesq_noisy=float("nan"), pesq_enh=float("nan"))
            for tag, deg in (("noisy", noisy_np), ("enh", enh)):
                row[f"stoi_{tag}"] = M.stoi_score(clean_np, deg)
                row[f"sisdr_{tag}"] = M.si_sdr_db(deg, clean_np)
                pesq_jobs.append((i, tag, (clean_np, deg)))
            row["pause_resid_db"] = M.pause_residual_db(enh, clean_np)
            row["speech_lsd_db"] = M.speech_lsd_db(enh, clean_np)
            row["pause_resid_noisy_db"] = M.pause_residual_db(noisy_np, clean_np)
            rows.append(row)
            if args.save_wavs:
                sf.write(os.path.join(args.save_wavs, item["name"]), enh, 16000)
            if args.log_every and (i + 1) % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                log.info("%d/%d files (%.0f%%) | %.1f files/s | running SI-SDR %.2f dB | ETA %s",
                         i + 1, len(test_set), 100 * (i + 1) / len(test_set), rate,
                         np.mean([r["sisdr_enh"] for r in rows]),
                         format_duration((len(test_set) - i - 1) / max(rate, 1e-9)))
    t_model = time.perf_counter() - t0

    if pesq is not None:
        t1 = time.perf_counter()
        pairs = [job[2] for job in pesq_jobs]
        if args.pesq_workers > 0:
            with ProcessPoolExecutor(max_workers=args.pesq_workers) as pool:
                scores = list(pool.map(_pesq_pair, pairs, chunksize=16))
        else:
            scores = [_pesq_pair(p) for p in pairs]
        for (i, tag, _), s in zip(pesq_jobs, scores):
            rows[i][f"pesq_{tag}"] = s
        failures = sum(1 for r in rows if not np.isfinite(r["pesq_enh"]))
        if failures:
            log.warning("PESQ was NaN for %d/%d files - the PESQ mean is over %d files, say so in the paper",
                        failures, len(rows), len(rows) - failures)
        log.info("PESQ scored in %s with %d workers", format_duration(time.perf_counter() - t1), args.pesq_workers)

    log.info("processed %d files (%s of audio) in %s (%.1fx real time, model+STOI %s)",
             len(rows), format_duration(audio_seconds), format_duration(time.perf_counter() - t0),
             audio_seconds / max(time.perf_counter() - t0, 1e-9), format_duration(t_model))

    for r in rows:
        log.debug("%s: SNR %.1f  PESQ %.3f->%.3f  STOI %.3f->%.3f  SI-SDR %.2f->%.2f  pause %+.1f dB  LSD %.2f dB",
                  r["name"], r["snr_db"], r["pesq_noisy"], r["pesq_enh"], r["stoi_noisy"], r["stoi_enh"],
                  r["sisdr_noisy"], r["sisdr_enh"], r["pause_resid_db"], r["speech_lsd_db"])

    summary = {"ckpt": os.path.abspath(args.ckpt), "weights": args.weights, "params": count_parameters(model),
               "mmacs_per_second": macs / 1e6 if macs else None, **summarise(rows)}
    summary["compare"] = [c for c in (compare(rows, p) for p in args.compare) if c]

    keys = [k for k in rows[0] if k != "name"]
    csv_path = args.csv or os.path.join(ckpt_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name"] + keys)
        w.writeheader()
        w.writerows(rows)
        w.writerow({"name": "AVERAGE", **{k: float(np.nanmean([r[k] for r in rows])) for k in keys}})
    json_path = os.path.splitext(csv_path)[0].replace("test_metrics", "test_summary") + ".json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("per-file metrics -> %s | summary -> %s", csv_path, json_path)
    log.info("=== evaluation finished ===")


if __name__ == "__main__":
    main()
