#!/usr/bin/env python3
r"""
Gate statistics and learned temperatures of every MHTRA block (mode A / mode B diagnostic).

    python gate_stats.py runs\mhtra_h4_s0\best.pt path\to\utterance.wav [--csv out.csv]

The old script printed the temperatures of ONE block (encoder block 0) and called
them the model's temperatures. Each of the six MHTRA modules has its own log_temp,
so all six are printed and written to the CSV.
"""
import argparse
import csv
import math

import numpy as np
import soundfile as sf
import torch

from mhtra_net.model import MHTRA, MHTRANet, MHTRANetConfig
from mhtra_net.stft import STFT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("wav")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sat", type=float, default=0.95, help="gate > sat or < 1-sat counts as saturated")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    m = MHTRANet(MHTRANetConfig(**ck["cfg"])).eval()
    m.load_state_dict(ck["model"])

    gates, attn_entropy = [], []

    def hook(mod, inp, out):
        x = inp[0]
        B, C, T, F = x.shape
        zt = x.pow(2).mean(-1).transpose(1, 2)
        ctx = mod.att_fc(mod.att_gru(zt)[0])
        q = mod.q_proj(ctx).view(B, T, mod.num_heads, mod.head_dim).transpose(1, 2)
        k = mod.k_proj(ctx).view(B, T, mod.num_heads, mod.head_dim).transpose(1, 2)
        v = mod.v_proj(ctx).view(B, T, mod.num_heads, mod.head_dim).transpose(1, 2)
        sc = torch.exp(mod.log_temp).view(1, -1, 1, 1) / math.sqrt(mod.head_dim)
        s = (q @ k.transpose(-1, -2)) * sc
        s = s.masked_fill(mod._causal_mask(T, x.device), float("-inf"))
        a = torch.softmax(s, -1)                                        # (B,H,T,T)
        o = (a @ v).transpose(1, 2).reshape(B, T, C)
        gates.append(torch.sigmoid(ctx + mod.out_proj(o)).detach())
        # mean attention entropy per head (nats): low = peaked, high = averaging
        ent = -(a * torch.log(a + 1e-12)).sum(-1).mean(dim=(0, 2))       # (H,)
        attn_entropy.append(ent.detach())

    blocks = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, MHTRA)]
    for _, mod in blocks:
        mod.register_forward_hook(hook)

    x, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000
    with torch.no_grad():
        m(STFT(device=torch.device("cpu"))(torch.from_numpy(x).unsqueeze(0)))

    print(f"{args.ckpt}  ({len(blocks)} MHTRA blocks, {x.shape[0] / sr:.1f}s utterance)")
    rows = []
    for i, ((name, mod), g, ent) in enumerate(zip(blocks, gates, attn_entropy)):
        temps = torch.exp(mod.log_temp).tolist()
        sat = ((g > args.sat) | (g < 1 - args.sat)).float().mean().item()
        print(f"  block {i} [{name}]  gate mean {g.mean():.3f} std {g.std():.3f} sat {sat:.3f}  "
              f"temps {[f'{t:.3f}' for t in temps]} (spread {max(temps) - min(temps):.3f})  "
              f"attn entropy/head {[f'{e:.2f}' for e in ent.tolist()]}")
        rows.append({"block": i, "name": name, "gate_mean": g.mean().item(), "gate_std": g.std().item(),
                     "gate_sat": sat, **{f"temp_{h}": t for h, t in enumerate(temps)},
                     **{f"attn_entropy_{h}": e for h, e in enumerate(ent.tolist())}})
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
