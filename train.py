#!/usr/bin/env python3
"""
Train MHTRA-Net (or the GTCRN baseline) on VoiceBank+DEMAND.

    python train.py --data /path/VoiceBank_DEMAND_16k --out runs/mhtra
    python train.py --data ... --out runs/gtcrn --baseline        # ablation baseline

Recipe (defaults now match the official GTCRN trainer, gtcrn/loss.py + SEtrain):
    loss      70*mse(mag^0.3) + 30*(mse(real_c) + mse(imag_c)) + 0.1 * (-SI-SDR in dB)
              (--w_complex 100 --w_time 0.1 --w_mrstft 0 --mag_floor 0).
              The official SI-SNR term is -log10(ratio), i.e. bels = dB/10, so the
              old default --w_time 1 weighted it 10x too heavily; and the old complex
              term averaged real and imaginary MSEs instead of summing them (half
              weight). Those two together are the main cause of the 0.31 PESQ
              reproduction gap (pause under-suppression, high LSD).
    schedule  linear warmup for --warmup_frac of all steps, then cosine to --min_lr
              (SEtrain: 25k warmup / 250k total, 1e-3 -> 1e-6).
    data      random 3 s crops in fixed-size batches (--segment 3, the setting that trains
              best here), or full-length utterances in length-bucketed batches (--segment 0,
              like the official trainer; with this model it converged worse and makes BatchNorm
              statistics batch-dependent, so --bn_recal recalibrates them before validation)
    optimiser AdamW (weight decay 0 by default -> plain Adam)
    schedule  --sched cosine: linear warmup then single cosine decay (see above), or
              --sched plateau: ReduceLROnPlateau on the EMA validation loss (the raw loss is too
              noisy epoch to epoch and collapses the learning rate early)
    ema       an exponential moving average of the weights (--ema DECAY, 0 disables) is
              validated next to the raw weights every epoch
    best.pt   the weights (raw or EMA) with the highest validation wide-band PESQ
              (--select_metric pesq, as the official trainer) or SI-SDR

TensorBoard event files go to <out>/tb unless --no_tensorboard is passed:
    tensorboard --logdir runs

Everything is logged to the console and mirrored to <out>/log.txt: startup
environment, dataset sizes, per-<--log_interval> training steps with speed and
ETA, validation results, and every checkpoint written. --log_level DEBUG adds
per-batch tensor shapes and loss components.
"""

import argparse
import copy
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader

from mhtra_net.dataset import LengthBucketSampler, VoiceBankDEMAND, collate_variable_length
from mhtra_net.logging_utils import (add_logging_args, format_duration, get_logger, log_args,
                                     log_cuda_memory, log_environment, log_stage, setup_logging)
from mhtra_net.losses import MHTRANetLoss, si_sdr
from mhtra_net.model import MHTRANet, MHTRANetConfig, count_parameters
from mhtra_net.stft import STFT

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:                                        # tensorboard is optional
    SummaryWriter = None

try:
    from pesq import pesq as _pesq
except ImportError:                                        # pesq needs a C++ compiler on Windows
    _pesq = None

log = get_logger("train")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/mhtra")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_batch_size", type=int, default=16,
                    help="validation batches are full-length utterances; small values starve the GPU")
    ap.add_argument("--segment", type=float, default=3.0,
                    help="training crop length in seconds (3 trains best here); 0 = full utterances "
                         "with length-bucketed batches")
    ap.add_argument("--bn_recal", type=int, default=-1,
                    help="recompute BatchNorm running statistics over this many training batches before "
                         "each validation; -1 = auto (100 in full-utterance mode, 0 with crops)")
    ap.add_argument("--bucket_size", type=int, default=50,
                    help="full-utterance mode: sort by length within windows of this many batches")
    ap.add_argument("--batch_seconds", type=float, default=48.0,
                    help="full-utterance mode: cap on padded audio per batch (seconds), so long utterances "
                         "get smaller batches and GPU memory stays flat (0 = no cap)")
    ap.add_argument("--select_metric", choices=["pesq", "sisdr"], default="pesq",
                    help="validation metric that decides best.pt (pesq = wide-band PESQ, as in the "
                         "official GTCRN recipe; falls back to sisdr if the pesq package is missing)")
    ap.add_argument("--pesq_workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="processes used to score validation PESQ (0 = in the main process)")
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA decay of the weights; the EMA model is validated too and can win best.pt "
                         "(0 disables)")
    # optimiser / schedule
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--sched", choices=["cosine", "plateau"], default="cosine")
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--plateau_patience", type=int, default=10)
    ap.add_argument("--warmup_frac", type=float, default=0.10,
                    help="cosine: fraction of all training steps spent in linear LR warmup (SEtrain: 0.1)")
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=4,
                    help="batches each worker preloads (ignored when --workers 0)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_interval", type=int, default=50,
                    help="log a training progress line every N steps")
    ap.add_argument("--grad_probe", type=int, default=20,
                    help="log the gradient norm of each loss term for the first N steps of the first "
                         "epoch, to see which term actually drives training (0 = off)")
    ap.add_argument("--no_tensorboard", action="store_true",
                    help="disable TensorBoard event files (written to <out>/tb by default)")
    # architecture flags
    ap.add_argument("--baseline", action="store_true", help="original GTCRN (no MHTRA/SE/learnable ERB)")
    ap.add_argument("--heads", type=int, default=4,
                    help="MHTRA heads; with channels=16 the processed half is 8, so heads=4 gives "
                         "head_dim=2 (very small). heads=2 -> dim 4, heads=1 -> dim 8")
    ap.add_argument("--attn_window", type=int, default=None)
    ap.add_argument("--no_se", action="store_true")
    ap.add_argument("--no_learnable_erb", action="store_true")
    ap.add_argument("--no_mhtra", action="store_true")
    ap.add_argument("--zero_init_out", action="store_true",
                    help="initialise the attention output projection W_o at zero so training starts exactly "
                         "at TRA (proposed remedy for the baseline-equivalent 'mode B' solution)")
    # loss weights (defaults = official GTCRN hybrid loss, see module docstring)
    ap.add_argument("--w_mrstft", type=float, default=0.0)
    ap.add_argument("--w_time", type=float, default=0.1,
                    help="weight of -SI-SDR(dB). 0.1 == the official -log10(ratio) term. The old default of "
                         "1.0 was 10x the official weight and is the main reproduction bug")
    ap.add_argument("--w_complex", type=float, default=100.0)
    ap.add_argument("--mag_floor", type=float, default=0.0,
                    help="clamp magnitudes at this value before the 0.3 power (old default 1e-5). "
                         "The official loss has none; 0 = off")
    ap.add_argument("--w_erb", type=float, default=0.01)
    ap.add_argument("--erb_anchor", type=float, default=1.0,
                    help="weight of the anchor pulling the learnable ERB back to its auditory init")
    ap.add_argument("--erb_ortho", type=float, default=0.0,
                    help="off-diagonal Gram penalty on the ERB filterbank. Keep at 0: ERB bands are "
                         "triangular and deliberately overlapping, so this term fights the anchor and "
                         "previously drove ~48%% of the entries negative")
    add_logging_args(ap)
    return ap.parse_args()


def build_config(args):
    if args.baseline:
        log.info("architecture: GTCRN baseline (MHTRA / SE / learnable ERB all disabled)")
        return MHTRANetConfig(use_mhtra=False, use_se=False, learnable_erb=False)
    return MHTRANetConfig(num_heads=args.heads, attn_window=args.attn_window,
                          use_mhtra=not args.no_mhtra, use_se=not args.no_se,
                          learnable_erb=not args.no_learnable_erb, zero_init_out=args.zero_init_out)


# --------------------------------------------------------------------------- #
class EMA:
    """Exponential moving average of a model's parameters and float buffers (BatchNorm stats)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # deepcopy loses cuDNN's contiguous RNN weight layout; restore it once (in-place EMA
        # updates keep the storage, so it stays valid) to avoid a per-call recompaction warning
        for m in self.model.modules():
            if isinstance(m, torch.nn.RNNBase):
                m.flatten_parameters()
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))     # warm-up: trust early weights more
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(b, alpha=1 - d)
            else:
                e.copy_(b)

    def state_dict(self):
        return {"model": self.model.state_dict(), "updates": self.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.updates = state.get("updates", 0)


@torch.no_grad()
def recalibrate_bn(model, loader, stft, device, n_batches):
    """
    Precise BatchNorm: replace the momentum-based running statistics (which follow whatever
    the last few batches looked like) with an exact average over `n_batches` training batches.
    Only affects eval-mode behaviour; training uses batch statistics anyway.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns or n_batches <= 0:
        return
    t0 = time.perf_counter()
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                     # cumulative moving average
    model.train()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        model(stft(batch["noisy"].to(device, non_blocking=True)))
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    log.info("BatchNorm statistics recalibrated on %d batches in %s", n_batches,
             format_duration(time.perf_counter() - t0))


# --------------------------------------------------------------------------- #
def _pesq_pair(pair):
    """Wide-band PESQ of one (clean, enhanced) float32 pair; NaN if PESQ rejects it (e.g. no speech)."""
    ref, deg = pair
    try:
        return float(_pesq(16000, ref.astype(np.float64), deg.astype(np.float64), "wb"))
    except Exception:
        return float("nan")


@torch.no_grad()
def validate(model, loader, stft, criterion, device, tag="", pesq_pool=None, want_pesq=True):
    """Returns (loss, SI-SDR dB, PESQ) averaged over the validation set. PESQ is NaN when disabled."""
    model.eval()
    log.info("%svalidating on %d batches ...", tag, len(loader))
    t0 = time.perf_counter()
    tot_loss, tot_sisdr, n = 0.0, 0.0, 0
    pairs = []                                  # (clean, enhanced) numpy pairs for PESQ
    for bi, batch in enumerate(loader):
        noisy, clean, lengths = batch["noisy"].to(device), batch["clean"].to(device), batch["lengths"]
        noisy_spec = stft(noisy)
        enh_spec = model(noisy_spec)
        enh = stft.istft(enh_spec, length=noisy.shape[-1])
        loss, _ = criterion(enh, clean, enh_spec, stft(clean), model)
        tot_loss += loss.item() * noisy.shape[0]
        enh_cpu, clean_cpu = enh.cpu().numpy(), clean.cpu().numpy()
        for i in range(noisy.shape[0]):
            L = int(lengths[i])
            tot_sisdr += si_sdr(enh[i:i + 1, :L], clean[i:i + 1, :L]).item()
            if want_pesq:
                pairs.append((clean_cpu[i, :L], enh_cpu[i, :L]))
        n += noisy.shape[0]
        log.debug("  val batch %d/%d: %d utterances, running SI-SDR %.2f dB",
                  bi + 1, len(loader), noisy.shape[0], tot_sisdr / max(n, 1))
    t_model = time.perf_counter() - t0

    val_pesq = float("nan")
    if want_pesq:
        t1 = time.perf_counter()
        if pesq_pool is not None:
            scores = list(pesq_pool.map(_pesq_pair, pairs, chunksize=16))
        else:
            scores = [_pesq_pair(p) for p in pairs]
        scores = np.asarray(scores, dtype=np.float64)
        bad = int(np.isnan(scores).sum())
        val_pesq = float(np.nanmean(scores)) if bad < len(scores) else float("nan")
        if bad:
            log.warning("%sPESQ was NaN for %d/%d validation utterances", tag, bad, len(scores))
        log.info("%svalidation done in %s (model %s, PESQ %s) over %d utterances",
                 tag, format_duration(time.perf_counter() - t0), format_duration(t_model),
                 format_duration(time.perf_counter() - t1), n)
    else:
        log.info("%svalidation done in %s over %d utterances", tag, format_duration(t_model), n)
    return tot_loss / n, tot_sisdr / n, val_pesq


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_logging(args.log_level, args.log_file or os.path.join(args.out, "log.txt"))

    log.info("=== MHTRA-Net training ===")
    log_args(log, args)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_environment(log, device)
    if args.amp and device.type != "cuda":
        log.warning("--amp requested but no CUDA device found; running in full precision")
    if device.type == "cuda":
        # cudnn.benchmark only pays off with fixed-size batches (random crops); with
        # bucketed full utterances every batch has a new length and re-tuning costs more than it saves.
        torch.backends.cudnn.benchmark = args.segment > 0
        torch.set_float32_matmul_precision("high")
        log.info("cudnn.benchmark=%s, float32 matmul precision=high", torch.backends.cudnn.benchmark)

    cfg = build_config(args)
    with log_stage(log, "building model"):
        model = MHTRANet(cfg).to(device)
    log.info("model params: %s | on %s", f"{count_parameters(model):,}", device)
    config_path = os.path.join(args.out, "config.json")
    with open(config_path, "w") as f:
        json.dump({**vars(args), "cfg": cfg.__dict__}, f, indent=2)
    log.info("run config written to %s", config_path)

    writer = None
    if not args.no_tensorboard:
        if SummaryWriter is None:
            log.warning("tensorboard not installed - skipping event files (pip install tensorboard)")
        else:
            tb_dir = os.path.join(args.out, "tb")
            writer = SummaryWriter(tb_dir)
            writer.add_text("config", f"```json\n{json.dumps({**vars(args), 'cfg': cfg.__dict__}, indent=2)}\n```")
            log.info("tensorboard -> %s (view with: tensorboard --logdir %s)",
                     tb_dir, os.path.dirname(os.path.abspath(args.out)))

    with log_stage(log, "loading datasets"):
        train_set = VoiceBankDEMAND(args.data, "train", args.segment)
        val_set = VoiceBankDEMAND(args.data, "val")
        # persistent workers matter on Windows: without them every epoch pays the
        # process-spawn cost again. prefetch keeps the GPU fed while wavs are decoded.
        loader_kwargs = dict(num_workers=args.workers, pin_memory=True)
        if args.workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch)
        if args.segment > 0:
            log.info("training on %.1fs random crops, fixed-size batches", args.segment)
            train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                                      **loader_kwargs)
        else:
            log.info("training on full-length utterances, length-bucketed batches (official recipe)")
            sampler = LengthBucketSampler(train_set.durations, args.batch_size, args.bucket_size, args.seed,
                                          max_seconds=args.batch_seconds)
            train_loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate_variable_length,
                                      **loader_kwargs)
        val_loader = DataLoader(val_set, args.val_batch_size, shuffle=False,
                                collate_fn=collate_variable_length, **loader_kwargs)
    log.info("train=%d utterances (%d steps/epoch, batch <= %d)  val=%d utterances (%d batches)",
             len(train_set), len(train_loader), args.batch_size, len(val_set), len(val_loader))
    bn_recal = args.bn_recal if args.bn_recal >= 0 else (100 if args.segment == 0 else 0)
    log.info("BatchNorm recalibration before validation: %s",
             f"{bn_recal} training batches" if bn_recal else "off")

    stft = STFT(device=device)
    w_erb = args.w_erb if cfg.learnable_erb else 0.0       # frozen ERB has nothing to regularise
    criterion = MHTRANetLoss(args.w_mrstft, args.w_time, args.w_complex, w_erb,
                             erb_anchor=args.erb_anchor, erb_ortho=args.erb_ortho,
                             mag_floor=args.mag_floor).to(device)
    if abs(args.w_time - 0.1) > 1e-9 or args.w_complex != 100.0 or args.mag_floor > 0:
        log.warning("loss weights differ from the official GTCRN HybridLoss "
                    "(w_time=0.1, w_complex=100, mag_floor=0) - results will not be comparable to the "
                    "released checkpoint")
    if args.w_time == 0:
        log.info("SI-SDR term disabled (--w_time 0): training on the compressed spectral loss only")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.sched == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=args.plateau_factor,
                                                           patience=args.plateau_patience, min_lr=args.min_lr)
        sched_desc = (f"ReduceLROnPlateau on {'EMA' if args.ema > 0 else 'raw'} val loss: "
                      f"factor={args.plateau_factor} patience={args.plateau_patience} min_lr={args.min_lr:.2e}")
    else:
        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = int(args.warmup_frac * total_steps)

        def lr_factor(step):            # linear warmup -> cosine decay to min_lr (SEtrain schedule)
            if step < warmup_steps:
                return max(step, 1) / max(warmup_steps, 1)
            prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return (args.min_lr + 0.5 * (1 + math.cos(math.pi * prog)) * (args.lr - args.min_lr)) / args.lr

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_factor)
        sched_desc = (f"linear warmup {warmup_steps} steps ({args.warmup_frac:.0%}) then cosine over "
                      f"{total_steps} steps to min_lr={args.min_lr:.2e}")
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    log.info("optimiser: AdamW lr=%.2e wd=%.2e | %s | grad clip %.1f | amp=%s",
             args.lr, args.weight_decay, sched_desc, args.grad_clip, scaler.is_enabled())

    ema = EMA(model, args.ema) if args.ema > 0 else None
    log.info("EMA of weights: %s", f"decay {args.ema}" if ema else "disabled")

    # best.pt is chosen on validation PESQ (official GTCRN recipe) unless --select_metric sisdr
    select_metric = args.select_metric
    if select_metric == "pesq" and _pesq is None:
        log.warning("the `pesq` package is not installed - selecting best.pt on SI-SDR instead")
        select_metric = "sisdr"
    pesq_pool = None
    if select_metric == "pesq" and args.pesq_workers > 0:
        pesq_pool = ProcessPoolExecutor(max_workers=args.pesq_workers)
    log.info("best.pt selected on validation %s%s, over %s", select_metric.upper(),
             f" ({args.pesq_workers} PESQ worker processes)" if pesq_pool is not None else "",
             "raw and EMA weights" if ema else "raw weights")

    start_epoch, best_sisdr, best_pesq, best_score = 0, -1e9, -1e9, -1e9
    if args.resume:
        with log_stage(log, f"resuming from {args.resume}"):
            ck = torch.load(args.resume, map_location=device)
            model.load_state_dict(ck.get("raw_model", ck["model"]))
            optim.load_state_dict(ck["optim"])
            sched.load_state_dict(ck["sched"])
            if ema is not None and "ema" in ck:
                ema.load_state_dict(ck["ema"])
            start_epoch = ck["epoch"] + 1
            best_sisdr = ck.get("best_sisdr", best_sisdr)
            best_pesq = ck.get("best_pesq", best_pesq)
            # only trust a stored best_score if it was tracked on the same metric
            if ck.get("select_metric") == select_metric:
                best_score = ck.get("best_score", best_score)
        log.info("resumed at epoch %d | best SI-SDR %.2f dB | best PESQ %.3f | best %s so far %s",
                 start_epoch, best_sisdr, best_pesq, select_metric.upper(),
                 f"{best_score:.3f}" if best_score > -1e8 else "(none)")

    steps_per_epoch = len(train_loader)          # full-utterance mode: may vary by a few steps per epoch
    log.info("starting training: epochs %d..%d, ~%d steps each", start_epoch, args.epochs - 1, steps_per_epoch)
    run_start = time.time()
    fmt = "%.3f" if select_metric == "pesq" else "%.2f dB"

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            t0, running = time.time(), {}
            steps_per_epoch = len(train_loader)
            log.info("--- epoch %d/%d ---", epoch, args.epochs - 1)
            for it, batch in enumerate(train_loader):
                noisy, clean = batch["noisy"].to(device, non_blocking=True), batch["clean"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    noisy_spec, clean_spec = stft(noisy), stft(clean)
                    enh_spec = model(noisy_spec)
                    enh = stft.istft(enh_spec.float(), length=noisy.shape[-1])
                    loss, parts = criterion(enh, clean, enh_spec.float(), clean_spec, model)

                # which loss term actually drives the update? (first epoch, first N steps)
                if args.grad_probe and epoch == start_epoch and it < args.grad_probe:
                    try:
                        gn = criterion.grad_norms(enh, clean, enh_spec.float(), clean_spec, model)
                        log.info("grad probe step %d: %s", it,
                                 "  ".join(f"{k}={v:.4f}" for k, v in gn.items()))
                    except Exception as e:                     # never let the probe kill a run
                        log.warning("grad probe failed at step %d: %s", it, e)

                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
                if args.sched == "cosine":
                    sched.step()
                if ema is not None:
                    ema.update(model)

                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    log.error("ep %d step %d: loss is %s - parts: %s", epoch, it + 1, loss_value,
                              {k: float(v) for k, v in parts.items()})
                for k, v in parts.items():
                    running[k] = running.get(k, 0.0) + v.item()
                running["total"] = running.get("total", 0.0) + loss_value
                if it == 0 and epoch == start_epoch:
                    log.info("first step OK: batch %s -> spec %s -> enhanced %s | loss %.4f",
                             tuple(noisy.shape), tuple(noisy_spec.shape), tuple(enh.shape), loss_value)
                    log_cuda_memory(log, "after first step: ")
                if args.log_interval and (it + 1) % args.log_interval == 0:
                    elapsed = time.time() - t0
                    steps_per_s = (it + 1) / max(elapsed, 1e-9)
                    eta = (steps_per_epoch - it - 1) / max(steps_per_s, 1e-9)
                    lr = optim.param_groups[0]["lr"]
                    s = "  ".join(f"{k}={v / (it + 1):.4f}" for k, v in running.items())
                    log.info("ep %d [%d/%d] lr=%.2e %s | grad=%.2f | %.2f steps/s (%.1f utt/s) | ETA %s",
                             epoch, it + 1, steps_per_epoch, lr, s,
                             float(grad_norm), steps_per_s, steps_per_s * args.batch_size,
                             format_duration(eta))
                    if writer is not None:
                        step = epoch * steps_per_epoch + it
                        writer.add_scalar("train/total", loss_value, step)
                        for k, v in parts.items():
                            writer.add_scalar(f"train/{k}", float(v), step)
                        writer.add_scalar("train/grad_norm", float(grad_norm), step)
                        writer.add_scalar("train/lr", lr, step)
                        writer.add_scalar("perf/steps_per_s", steps_per_s, step)

            train_loss = running["total"] / max(it + 1, 1)
            want_pesq = select_metric == "pesq"
            if bn_recal:
                recalibrate_bn(model, train_loader, stft, device, bn_recal)
            results = {"raw": validate(model, val_loader, stft, criterion, device, f"epoch {epoch} [raw] ",
                                       pesq_pool=pesq_pool, want_pesq=want_pesq)}
            if ema is not None:
                results["ema"] = validate(ema.model, val_loader, stft, criterion, device, f"epoch {epoch} [ema] ",
                                          pesq_pool=pesq_pool, want_pesq=want_pesq)
            val_loss, val_sisdr, val_pesq = results["raw"]
            if args.sched == "plateau":
                lr_before = optim.param_groups[0]["lr"]
                # the EMA loss is smooth epoch to epoch; the raw loss can swing by >1 and
                # would trigger spurious reductions
                sched.step(results["ema"][0] if ema is not None else val_loss)
                if optim.param_groups[0]["lr"] < lr_before:
                    log.info("plateau: lr %.2e -> %.2e", lr_before, optim.param_groups[0]["lr"])

            epoch_time = time.time() - t0
            done = epoch - start_epoch + 1
            remaining = (args.epochs - epoch - 1) * (time.time() - run_start) / max(done, 1)
            summary = " | ".join(
                f"{w}: loss={l:.4f} SI-SDR={s:.2f} dB" + (f" PESQ={p:.3f}" if want_pesq else "")
                for w, (l, s, p) in results.items())
            log.info("epoch %d done in %s | train=%.4f | lr=%.2e | %s | run ETA %s",
                     epoch, format_duration(epoch_time), train_loss, optim.param_groups[0]["lr"], summary,
                     format_duration(remaining))
            log_cuda_memory(log, f"epoch {epoch}: ")

            if writer is not None:
                writer.add_scalar("epoch/train_loss", train_loss, epoch)
                writer.add_scalar("epoch/lr", optim.param_groups[0]["lr"], epoch)
                for w, (l, s, p) in results.items():
                    writer.add_scalar(f"epoch/val_loss_{w}", l, epoch)
                    writer.add_scalar(f"epoch/val_si_sdr_db_{w}", s, epoch)
                    if want_pesq:
                        writer.add_scalar(f"epoch/val_pesq_{w}", p, epoch)
                writer.add_scalar("perf/epoch_seconds", epoch_time, epoch)
                if cfg.learnable_erb:
                    # the ERB filterbank is trainable, so watch the *effective* (softplus)
                    # weights drift from their auditory init
                    w = model.erb.analysis_weight().detach().float().cpu()
                    writer.add_image("erb/analysis", (w - w.min()) / (w.max() - w.min() + 1e-12),
                                     epoch, dataformats="HW")
                    writer.add_scalar("erb/drift_from_init",
                                      float(model.erb.drift_from_init()), epoch)
                    writer.add_scalar("erb/negative_fraction", float((w < 0).float().mean()), epoch)
                writer.flush()

            # pick the better of raw / EMA on the selection metric; NaN can never win
            def score_of(res):
                s = res[2] if select_metric == "pesq" else res[1]
                return s if math.isfinite(s) else -1e9
            winner = max(results, key=lambda w: score_of(results[w]))
            score = score_of(results[winner])
            if score <= -1e8:
                log.warning("epoch %d: validation %s is NaN for every weight set, not eligible for best.pt",
                            epoch, select_metric.upper())
            best_sisdr = max([best_sisdr] + [r[1] for r in results.values()])
            best_pesq = max([best_pesq] + [r[2] for r in results.values() if math.isfinite(r[2])])

            common = {"optim": optim.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                      "cfg": cfg.__dict__, "select_metric": select_metric, "sched_type": args.sched,
                      "best_sisdr": best_sisdr, "best_pesq": best_pesq, "best_score": max(best_score, score),
                      "val": {w: {"loss": l, "sisdr": s, "pesq": p} for w, (l, s, p) in results.items()}}
            if ema is not None:
                common["ema"] = ema.state_dict()
            # last.pt carries the raw weights under "model" (plus EMA for resume);
            # best.pt carries whichever weights won, so evaluate.py loads it unchanged.
            last_path = os.path.join(args.out, "last.pt")
            torch.save({**common, "model": model.state_dict(), "raw_model": model.state_dict(),
                        "weights": "raw"}, last_path)
            log.info("checkpoint saved -> %s (%.1f MB)", last_path, os.path.getsize(last_path) / 1e6)
            if score > best_score:
                delta = (" (+" + fmt % (score - best_score) + ")") if best_score > -1e8 else " (first epoch)"
                best_score = score
                best_path = os.path.join(args.out, "best.pt")
                win_model = ema.model if winner == "ema" else model
                torch.save({**common, "best_score": best_score, "model": win_model.state_dict(),
                            "raw_model": model.state_dict(), "weights": winner}, best_path)
                log.info("  -> new best %s " + fmt + "%s from %s weights saved to %s (SI-SDR %.2f dB)",
                         select_metric.upper(), best_score, delta, winner.upper(), best_path, results[winner][1])
            else:
                log.info("  no improvement (best %s is still " + fmt + ")", select_metric.upper(), best_score)
    except KeyboardInterrupt:
        log.warning("interrupted by user after %s - latest checkpoint is %s",
                    format_duration(time.time() - run_start), os.path.join(args.out, "last.pt"))
        raise
    except Exception:
        log.exception("training crashed after %s", format_duration(time.time() - run_start))
        raise
    finally:
        if writer is not None:
            writer.close()
        if pesq_pool is not None:
            pesq_pool.shutdown(wait=False, cancel_futures=True)

    log.info("=== training finished: %d epochs in %s | best val PESQ %.3f | best val SI-SDR %.2f dB "
             "| best.pt chosen on %s | checkpoints in %s ===",
             args.epochs - start_epoch, format_duration(time.time() - run_start), best_pesq, best_sisdr,
             select_metric.upper(), args.out)


if __name__ == "__main__":
    main()