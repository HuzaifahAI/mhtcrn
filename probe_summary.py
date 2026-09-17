#!/usr/bin/env python3
"""
Summarise the short "escape probe" runs (runs/probe_<config>_wt<w>_s<seed>/log.txt).

The good/bad basin is decided within the first few epochs of warmup: in all 18 full sweep
runs, validation PESQ at epoch 5 was >= 1.90 for every run that ended in the good basin and
<= 1.85 for every run that did not. This script reads each probe's validation PESQ at
--epoch and reports, per (config, weight): n, escape count, escape rate with a 95% Wilson
interval, and the mean epoch-5 val PESQ. Also prints Fisher exact p-values vs the baseline.

    python probe_summary.py --runs runs --out runs/probe_summary.csv
"""
import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict


def val_pesq_at(log_path, epoch):
    pat = re.compile(r'epoch (\d+) done.*?raw: loss=(-?[\d.]+) SI-SDR=(-?[\d.]+) dB PESQ=([\d.]+).*?ema:.*?PESQ=([\d.]+)')
    best = None
    for line in open(log_path, errors="ignore"):
        m = pat.search(line)
        if m and int(m.group(1)) == epoch:
            return max(float(m.group(4)), float(m.group(5)))
        if m and int(m.group(1)) < epoch:
            best = max(float(m.group(4)), float(m.group(5)))
    return best  # run shorter than requested epoch: use the last available


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_one_sided(k1, n1, k0, n0):
    """P(X >= k1) for hypergeometric: k1 successes in n1 draws from n1+n0 with k1+k0 successes."""
    K, N, n = k1 + k0, n1 + n0, n1
    def C(a, b):
        return math.comb(a, b) if 0 <= b <= a else 0
    return sum(C(K, x) * C(N - K, n - x) for x in range(k1, min(K, n) + 1)) / C(N, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--epoch", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=1.90)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    groups = defaultdict(list)
    for lg in glob.glob(os.path.join(args.runs, "probe_*", "log.txt")):
        name = os.path.basename(os.path.dirname(lg))
        m = re.match(r"probe_(.+)_(wt[0-9p]+)_s(\d+)$", name)
        if not m:
            continue
        v = val_pesq_at(lg, args.epoch)
        if v is not None:
            groups[(m.group(1), m.group(2))].append((int(m.group(3)), v))

    rows = []
    print(f"escape = val PESQ at epoch {args.epoch} >= {args.threshold}\n")
    print(f"{'weight':<8}{'config':<10}{'n':>4}{'escapes':>9}{'rate':>7}{'95% CI':>16}{'mean PESQ@ep':>14}{'p vs base':>11}")
    for w in sorted({w for _, w in groups}, key=lambda t: float(t[2:].replace("p", "."))):
        base = groups.get(("base", w), [])
        kb = sum(v >= args.threshold for _, v in base)
        for cfg in ["base", "h1", "h1zero", "seonly", "attnonly"]:
            g = groups.get((cfg, w), [])
            if not g:
                continue
            k = sum(v >= args.threshold for _, v in g)
            lo, hi = wilson(k, len(g))
            p = fisher_one_sided(k, len(g), kb, len(base)) if cfg != "base" and base else float("nan")
            mean = sum(v for _, v in g) / len(g)
            print(f"{w:<8}{cfg:<10}{len(g):>4}{k:>9}{k / len(g):>7.2f}   [{lo:.2f}, {hi:.2f}]{mean:>14.3f}{p:>11.3g}")
            rows.append({"weight": w, "config": cfg, "n": len(g), "escapes": k, "rate": k / len(g),
                         "ci_lo": lo, "ci_hi": hi, "mean_val_pesq": mean, "fisher_p_vs_base": p,
                         "per_seed": " ".join(f"{s}:{v:.2f}" for s, v in sorted(g))})
        print()
    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wri.writeheader(); wri.writerows(rows)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()