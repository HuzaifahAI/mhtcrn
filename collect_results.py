#!/usr/bin/env python3
"""
Collect every run's test_summary.json / test_metrics.csv / gate_stats.csv / <testset>.csv under
--runs into one table, grouped by configuration (run name without the _s<seed> suffix).

    python collect_results.py --runs runs --out runs/summary.csv

Prints, per configuration: n seeds, PESQ mean ± std (over seeds), STOI, SI-SDR, pause residual,
speech LSD, the paired PESQ gain over the same-seed baseline, and how many seeds landed in
"mode B" (final loss / LSD indistinguishable from the baseline). Mode B is flagged when a run's
speech LSD is within 0.5 dB of its same-seed baseline's LSD.
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--modeb_lsd_margin", type=float, default=0.5)
    args = ap.parse_args()

    runs = {}
    for js in glob.glob(os.path.join(args.runs, "*", "test_summary.json")):
        d = os.path.dirname(js)
        name = os.path.basename(d)
        m = re.match(r"(.+)_s(\d+)$", name)
        cfg, seed = (m.group(1), int(m.group(2))) if m else (name, -1)
        s = load_json(js)
        row = {"run": name, "config": cfg, "seed": seed, "files": s.get("files"),
               "pesq": s["enh"]["pesq"], "stoi": s["enh"]["stoi"], "sisdr": s["enh"]["sisdr"],
               "pause_db": s["diag"]["pause_resid_db"], "lsd_db": s["diag"]["speech_lsd_db"],
               "mmacs": s.get("mmacs_per_second"), "params": s.get("params")}
        for c in s.get("compare", []):
            ref = os.path.basename(os.path.dirname(c["reference"]))
            tag = "vs_base" if ref.startswith("base_") else ("vs_official" if ref == "official" else f"vs_{ref}")
            if "pesq_enh" in c:
                row[f"{tag}_dpesq"] = c["pesq_enh"]["diff"]
                row[f"{tag}_ci_lo"], row[f"{tag}_ci_hi"] = c["pesq_enh"]["ci95"]
                row[f"{tag}_wins"] = f'{c["pesq_enh"]["wins"]}/{c["pesq_enh"]["n"]}'
        gs = os.path.join(d, "gate_stats.csv")
        if os.path.exists(gs):
            with open(gs) as f:
                g = list(csv.DictReader(f))
            row["gate_sat_mean"] = float(np.mean([float(r["gate_sat"]) for r in g]))
            temps = [[float(v) for k, v in r.items() if k.startswith("temp_")] for r in g]
            row["temp_spread_max"] = float(max(max(t) - min(t) for t in temps))
        for ood in glob.glob(os.path.join(d, "*.csv")):
            b = os.path.basename(ood)
            if b in ("test_metrics.csv", "gate_stats.csv"):
                continue
            with open(ood) as f:
                rows = [r for r in csv.DictReader(f)]
            vals = [float(r["pesq_enh"]) for r in rows if r.get("pesq_enh") not in (None, "", "nan")]
            if vals:
                row[f"ood_{os.path.splitext(b)[0]}_pesq"] = float(np.mean(vals))
        runs[name] = row

    # mode A / B label from same-seed baseline LSD
    for name, r in runs.items():
        base = runs.get(f"base_s{r['seed']}")
        if base and not r["config"].startswith("base"):
            r["mode"] = "B" if abs(r["lsd_db"] - base["lsd_db"]) < args.modeb_lsd_margin else "A"
        else:
            r["mode"] = ""

    # per-run table
    keys = sorted({k for r in runs.values() for k in r}, key=lambda k: (k not in ("run", "config", "seed"), k))
    print(f"{'run':<16}{'PESQ':>7}{'STOI':>7}{'SI-SDR':>8}{'pause':>7}{'LSD':>7}{'mode':>5}{'dPESQ vs base':>15}")
    for name in sorted(runs):
        r = runs[name]
        dp = r.get("vs_base_dpesq")
        print(f"{name:<16}{r['pesq']:>7.3f}{r['stoi']:>7.3f}{r['sisdr']:>8.2f}{r['pause_db']:>7.2f}{r['lsd_db']:>7.2f}"
              f"{r['mode']:>5}{('%+.3f' % dp) if dp is not None else '':>15}")

    # per-config aggregate
    print("\nper configuration (mean ± std over seeds):")
    print(f"{'config':<12}{'n':>3}{'PESQ':>16}{'STOI':>8}{'SI-SDR':>8}{'pause':>8}{'LSD':>8}{'modeB':>7}{'E[dPESQ]':>10}{'dPESQ|A':>9}")
    agg = []
    for cfg in sorted({r["config"] for r in runs.values()}):
        rs = [r for r in runs.values() if r["config"] == cfg]
        p = np.array([r["pesq"] for r in rs])
        dps = [r["vs_base_dpesq"] for r in rs if r.get("vs_base_dpesq") is not None]
        dpa = [r["vs_base_dpesq"] for r in rs if r.get("vs_base_dpesq") is not None and r["mode"] == "A"]
        nb = sum(r["mode"] == "B" for r in rs)
        line = {"config": cfg, "n": len(rs), "pesq_mean": p.mean(), "pesq_std": p.std(ddof=1) if len(p) > 1 else 0.0,
                "stoi": np.mean([r["stoi"] for r in rs]), "sisdr": np.mean([r["sisdr"] for r in rs]),
                "pause_db": np.mean([r["pause_db"] for r in rs]), "lsd_db": np.mean([r["lsd_db"] for r in rs]),
                "mode_b": nb, "dpesq_expected": np.mean(dps) if dps else float("nan"),
                "dpesq_mode_a": np.mean(dpa) if dpa else float("nan")}
        agg.append(line)
        print(f"{cfg:<12}{line['n']:>3}{line['pesq_mean']:>9.3f} ± {line['pesq_std']:.3f}{line['stoi']:>8.3f}"
              f"{line['sisdr']:>8.2f}{line['pause_db']:>8.2f}{line['lsd_db']:>8.2f}{nb:>4}/{len(rs):<2}"
              f"{line['dpesq_expected']:>10.3f}{line['dpesq_mode_a']:>9.3f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for name in sorted(runs):
                w.writerow(runs[name])
        agg_path = os.path.splitext(args.out)[0] + "_by_config.csv"
        with open(agg_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            w.writeheader(); w.writerows(agg)
        print(f"\n-> {args.out}\n-> {agg_path}")


if __name__ == "__main__":
    main()
