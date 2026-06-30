#!/usr/bin/env python3
"""
analyze_mdd.py — Experiment 2 Analysis (Path A) for the LLM-judge evaluation pilot.

Improvement from Exp 1: instead of planting an artificial gap then running a paired t-test
across problems (which measures the wrong kind of variability), this script DIRECTLY
measures the stability of the AGGREGATE SCORE across independent runs, then derives the
Minimum Detectable Difference (MDD) per judge configuration.

MDD = smallest difference in the aggregate score that can be reliably detected
      (alpha=0.05, power=0.8) ~= 2.8 * standard_error_aggregate.

This framing matches the HULA problem statement ("detecting 1-2% improvements").

No re-judging needed: reads raw scores from artifacts/pool.json.

Usage:
    python analyze_mdd.py
    python analyze_mdd.py --boot 2000        # number of bootstrap resamples
"""
import json
import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

ART = Path("artifacts")
POOL = ART / "pool.json"

# MDD factor for two-sided test alpha=0.05 (z=1.96) & power=0.8 (z=0.84).
Z_FACTOR = 1.96 + 0.84  # = 2.8


def load_matrix(pool, mode):
    """
    Build a score matrix [problem][run] for one selected candidate per problem.
    We pick the first candidate that has complete scores for this mode.
    Only use problems with >= M_min runs.
    Returns array (n_problem, M) where M = the common minimum run length.
    """
    rows = []
    for tid, pinfo in pool["problems"].items():
        for cand in pinfo["candidates"]:
            s = cand["scores"].get(mode, [])
            if len(s) > 0:
                rows.append(np.array(s, dtype=float))
                break  # one candidate per problem
    if not rows:
        return None
    M = min(len(r) for r in rows)
    mat = np.array([r[:M] for r in rows])  # (n_problem, M)
    return mat


def aggregate_se(mat, m, rng, boot):
    """
    Estimate standard error of the AGGREGATE SCORE (mean across all problems)
    when each problem is scored with the average of 'm' runs.
    Computed via bootstrap: repeatedly, for each problem sample m runs at random
    (with replacement from the M available runs), average them -> score per problem,
    then average across all problems -> one aggregate score. SD of these aggregate
    scores = aggregate SE.
    """
    n_prob, M = mat.shape
    aggs = np.empty(boot)
    for b in range(boot):
        idx = rng.integers(0, M, size=(n_prob, m))      # (n_prob, m) indeks run
        per_problem = np.take_along_axis(mat, idx, axis=1).mean(axis=1)  # (n_prob,)
        aggs[b] = per_problem.mean()
    return aggs.std(ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000, help="number of bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not POOL.exists():
        print("[error] artifacts/pool.json does not exist yet.")
        return
    pool = json.loads(POOL.read_text())
    rng = np.random.default_rng(args.seed)

    modes = ["freeform", "structured"]

    # ---- Sub-experiment A: per-item SD (context, same as Exp 1) ----
    print("=== A. Per-item noise (SD across runs) ===")
    for mode in modes:
        sds = []
        for tid, pinfo in pool["problems"].items():
            for c in pinfo["candidates"]:
                s = c["scores"].get(mode, [])
                if len(s) >= 2:
                    sds.append(np.std(s, ddof=1))
        if sds:
            print(f"  {mode:11s}: median SD/item = {np.median(sds):.3f} | mean = {np.mean(sds):.3f} (n={len(sds)})")

    # ---- Sub-experiment B: MDD on aggregate score (core of Path A) ----
    print("\n=== B. MDD on aggregate score (alpha=0.05, power=0.8) ===")
    mats = {mode: load_matrix(pool, mode) for mode in modes}
    M_avail = {mode: (m.shape[1] if m is not None else 0) for mode, m in mats.items()}
    n_prob = {mode: (m.shape[0] if m is not None else 0) for mode, m in mats.items()}
    print(f"  Problems used: freeform={n_prob['freeform']}, structured={n_prob['structured']}")
    print(f"  Runs available (M): freeform={M_avail['freeform']}, structured={M_avail['structured']}")

    Mmax = min(v for v in M_avail.values() if v > 0)
    m_grid = sorted(set([1] + list(range(2, Mmax + 1, max(1, Mmax // 6))) + [Mmax]))

    results = {mode: {} for mode in modes}
    print(f"\n  {'m (run)':>8} | {'freeform SE':>12} {'freeform MDD':>13} | {'structured SE':>14} {'structured MDD':>15}")
    print("  " + "-" * 74)
    for m in m_grid:
        line = f"  {m:>8} |"
        for mode in modes:
            mat = mats[mode]
            if mat is None:
                line += f" {'-':>12} {'-':>13} |"
                continue
            se = aggregate_se(mat, m, rng, args.boot)
            mdd = Z_FACTOR * se
            results[mode][m] = (se, mdd)
            if mode == "freeform":
                line += f" {se:>12.4f} {mdd*100:>11.2f}% |"
            else:
                line += f" {se:>14.4f} {mdd*100:>13.2f}%"
        print(line)

    # ---- Headline summary ----
    print("\n=== Headline summary ===")
    for mode in modes:
        if 1 in results[mode] and Mmax in results[mode]:
            mdd1 = results[mode][1][1] * 100
            mddM = results[mode][Mmax][1] * 100
            print(f"  {mode:11s}: MDD single-run = {mdd1:.2f}%  ->  MDD mean-of-{Mmax} = {mddM:.2f}%")
    # check whether any configuration breaks through 2%
    best = None
    for mode in modes:
        for m, (se, mdd) in results[mode].items():
            if mdd * 100 <= 2.0 and (best is None or mdd < best[2]):
                best = (mode, m, mdd)
    if best:
        print(f"\n  FIRST configuration to break through MDD <= 2%: {best[0]}, m={best[1]} "
              f"(MDD={best[2]*100:.2f}%).")
        print("  -> Supports the thesis: stabilization (averaging) restores detection of ~2% differences.")
    else:
        print("\n  No configuration achieved MDD <= 2% on this data.")
        print("  -> Report as-is; may need larger M / more balanced data.")

    # ---- Plot: MDD vs m ----
    if HAVE_PLT:
        plt.figure(figsize=(7, 5))
        for mode in modes:
            if results[mode]:
                ms = sorted(results[mode].keys())
                mdds = [results[mode][m][1] * 100 for m in ms]
                plt.plot(ms, mdds, marker="o", label=mode)
        plt.axhline(2.0, ls="--", color="gray", label="target 2%")
        plt.xlabel("Number of runs averaged (m)")
        plt.ylabel("Minimum Detectable Difference (%)")
        plt.title("MDD decreases with averaging — does it break through 2%?")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out = ART / "mdd_vs_runs.png"
        plt.savefig(out, dpi=130)
        print(f"\n  Plot saved: {out}")

    # ---- Validity (context) ----
    print("\n=== C. Validity vs ground truth (context) ===")
    from scipy import stats
    for mode in modes:
        passed, mean_score = [], []
        for tid, pinfo in pool["problems"].items():
            for c in pinfo["candidates"]:
                s = c["scores"].get(mode, [])
                if c.get("passed") is not None and len(s) > 0:
                    passed.append(int(c["passed"]))
                    mean_score.append(np.mean(s))
        passed = np.array(passed); mean_score = np.array(mean_score)
        if len(set(passed.tolist())) < 2:
            print(f"  {mode}: ground truth has no variation, skipping.")
            continue
        rho, _ = stats.spearmanr(mean_score, passed)
        print(f"  {mode:11s}: Spearman(score, pass) = {rho:.2f} "
              f"(pass-rate data = {passed.mean():.1%})")


if __name__ == "__main__":
    main()
