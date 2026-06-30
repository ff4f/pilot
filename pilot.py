#!/usr/bin/env python3
"""
Mini-pilot: stability & cost of LLM-judge evaluation for coding agents.
See README.md for full context.

Workflow (each step saves a checkpoint to ./artifacts):
  python pilot.py generate    --dataset humaneval --n 50 --k 3 --gen-temp 0.6
  python pilot.py groundtruth --dataset humaneval
  python pilot.py judge       --m 10
  python pilot.py analyze     --target-gap 0.02

LLM via env var (OpenAI-compatible; works with local Ollama, Groq, Gemini, OpenRouter, DeepSeek):
  LLM_BASE_URL, LLM_API_KEY, GEN_MODEL, JUDGE_MODEL
"""
import os, re, json, argparse, random, subprocess, time
from pathlib import Path

import numpy as np
from scipy import stats

import prompts

ART = Path("artifacts"); ART.mkdir(exist_ok=True)
POOL = ART / "pool.json"
SAMPLES = ART / "samples.jsonl"

# --------------------------------------------------------------------------
# LLM client (OpenAI-compatible)
# --------------------------------------------------------------------------
from openai import OpenAI

_client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.environ.get("LLM_API_KEY", "ollama"),
)
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5-coder:7b")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen2.5-coder:7b")


def call_llm(messages, temperature, model, max_tokens=1024, retries=3):
    """Single chat completion call with simple retry."""
    for attempt in range(retries):
        try:
            r = _client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [warn] LLM call failed: {e}")
                return ""
            time.sleep(2 * (attempt + 1))


# --------------------------------------------------------------------------
# Util
# --------------------------------------------------------------------------
def load_pool():
    return json.loads(POOL.read_text()) if POOL.exists() else None

def save_pool(pool):
    POOL.write_text(json.dumps(pool, indent=2))

def extract_code(text: str) -> str:
    """Extract the first code block from model output; if no fence, use as-is."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    return code.strip()


# --------------------------------------------------------------------------
# Step 1: generate candidates
# --------------------------------------------------------------------------
def cmd_generate(args):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    data = get_human_eval_plus() if args.dataset == "humaneval" else get_mbpp_plus()

    task_ids = sorted(data.keys())
    rng = random.Random(args.seed)
    rng.shuffle(task_ids)
    task_ids = task_ids[: args.n]

    pool = {"dataset": args.dataset, "problems": {}}
    sample_lines = []

    for i, tid in enumerate(task_ids):
        prompt = data[tid]["prompt"]
        cands = []
        for j in range(args.k):
            msg = [
                {"role": "system", "content": "You are an expert Python programmer."},
                {"role": "user", "content":
                    f"Complete this task. Return ONE self-contained Python code block "
                    f"that defines the required function exactly as specified.\n\n{prompt}"},
            ]
            raw = call_llm(msg, temperature=args.gen_temp, model=GEN_MODEL)
            code = extract_code(raw)
            cands.append({"code": code, "passed": None, "scores": {}})
            # EvalPlus: one line per candidate; multiple lines per task_id are allowed.
            sample_lines.append({"task_id": tid, "solution": code})
        pool["problems"][tid] = {"prompt": prompt, "candidates": cands}
        print(f"[{i+1}/{len(task_ids)}] {tid}: {args.k} candidates")

    save_pool(pool)
    with open(SAMPLES, "w") as f:
        for line in sample_lines:
            f.write(json.dumps(line) + "\n")
    print(f"\nSaved {len(sample_lines)} candidates -> {SAMPLES}")
    print("Next: python pilot.py groundtruth --dataset", args.dataset)


# --------------------------------------------------------------------------
# Step 2: ground truth via EvalPlus
# --------------------------------------------------------------------------
def cmd_groundtruth(args):
    print("Running EvalPlus (unit test execution)...")
    # Recommended via Docker for safe execution of foreign code.
    # Local: uncomment the following line & comment out the docker block if Docker is unavailable.
    cmd = ["evalplus.evaluate", "--dataset", args.dataset, "--samples", str(SAMPLES)]
    subprocess.run(cmd, check=False)

    res_path = ART / "samples_eval_results.json"
    if not res_path.exists():
        # result filename follows the samples filename; check artifacts folder
        cands = list(ART.glob("*_eval_results.json"))
        res_path = cands[0] if cands else None
    if not res_path or not res_path.exists():
        print("[error] EvalPlus results file not found. Check evalplus output & adjust path.")
        return

    results = json.loads(res_path.read_text())
    pass_lists = extract_pass_lists(results)   # {task_id: [bool, bool, ...]} ordered by submission

    pool = load_pool()
    n_pass, n_total = 0, 0
    for tid, pinfo in pool["problems"].items():
        statuses = pass_lists.get(tid, [])
        for j, cand in enumerate(pinfo["candidates"]):
            cand["passed"] = bool(statuses[j]) if j < len(statuses) else False
            n_pass += int(cand["passed"]); n_total += 1
    save_pool(pool)
    print(f"\nGround truth populated. Candidate pass rate: {n_pass}/{n_total} = {n_pass/n_total:.1%}")
    if not (0.15 < n_pass / n_total < 0.85):
        print("[note] Pass/fail mix is unbalanced. Consider changing --k, --gen-temp,")
        print("       or the generator model to get more variation between correct & incorrect.")
    print("Next: python pilot.py judge --m 10")


def extract_pass_lists(results: dict) -> dict:
    """
    Defensive parser for EvalPlus results. Schema can differ between versions, so
    we handle two common forms. If it fails, INSPECT the *_eval_results.json file
    and adjust this function (look for the 'plus_status' field).
    """
    out = {}
    eval_block = results.get("eval", results)
    for tid, entry in eval_block.items():
        statuses = []
        # Form A: list of dict per sample, each dict has 'plus_status'/'base_status'
        if isinstance(entry, list):
            for r in entry:
                if isinstance(r, dict):
                    st = r.get("plus_status") or r.get("base_status") or r.get("status")
                    statuses.append(str(st).lower() == "pass")
                elif isinstance(r, (list, tuple)):
                    # sometimes [status, details]
                    statuses.append(str(r[0]).lower() == "pass")
        # Form B: dict with 'plus'/'base' keys containing status lists
        elif isinstance(entry, dict):
            arr = entry.get("plus") or entry.get("base") or []
            for st in arr:
                statuses.append(str(st).lower() == "pass")
        out[tid] = statuses
    return out


# --------------------------------------------------------------------------
# Step 3: judge (M times, 2 modes) with caching
# --------------------------------------------------------------------------
def parse_freeform(text: str):
    m = re.search(r"\b(10|[1-9])\b", text)
    return int(m.group(1)) if m else None

def parse_structured(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None
    try:
        obj = json.loads(m.group(0))
        v = obj.get("final_score")
        return int(v) if v is not None else None
    except Exception:
        return None

def cmd_judge(args):
    pool = load_pool()
    if pool is None:
        print("[error] pool.json does not exist yet. Run generate first."); return

    modes = {
        "freeform": (prompts.build_freeform_messages, parse_freeform),
        "structured": (prompts.build_structured_messages, parse_structured),
    }
    total = sum(len(p["candidates"]) for p in pool["problems"].values()) * len(modes)
    done = 0
    for tid, pinfo in pool["problems"].items():
        for cand in pinfo["candidates"]:
            for mode, (builder, parser) in modes.items():
                existing = cand["scores"].get(mode, [])
                need = args.m - len(existing)
                for _ in range(max(0, need)):
                    msg = builder(pinfo["prompt"], cand["code"])
                    raw = call_llm(msg, temperature=args.judge_temp, model=JUDGE_MODEL)
                    score = parser(raw)
                    if score is not None:
                        existing.append(score / 10.0)   # normalize to [0,1]
                cand["scores"][mode] = existing
                done += 1
            # save periodically so it can be resumed
        save_pool(pool)
        print(f"  judged task {tid}  ({done}/{total} sel)")
    print(f"\nDone. Raw scores saved to {POOL}")
    print("Next: python pilot.py analyze --target-gap 0.02")


# --------------------------------------------------------------------------
# Step 4: analysis
# --------------------------------------------------------------------------
def collect(pool, mode):
    """Return list of problems; each problem = list of candidates with (passed, scores[]).
       Only use problems with >=1 candidate with complete scores."""
    out = []
    for tid, pinfo in pool["problems"].items():
        cands = []
        for c in pinfo["candidates"]:
            s = c["scores"].get(mode, [])
            if c["passed"] is not None and len(s) > 0:
                cands.append({"passed": int(c["passed"]), "scores": np.array(s, float)})
        if cands:
            out.append(cands)
    return out

def exp1_noise(pool):
    print("\n=== Experiment 1: judge noise (SD across runs per candidate) ===")
    rows = {}
    for mode in ("freeform", "structured"):
        sds = []
        for tid, pinfo in pool["problems"].items():
            for c in pinfo["candidates"]:
                s = c["scores"].get(mode, [])
                if len(s) >= 2:
                    sds.append(np.std(s, ddof=1))
        sds = np.array(sds)
        rows[mode] = sds
        print(f"  {mode:11s}: median SD/item = {np.median(sds):.3f} | mean = {np.mean(sds):.3f} (n={len(sds)})")
    return rows

def _system_scores(problems, selection, run="mean"):
    """Judge score per problem for a system with 'selection' = selected candidate index per problem.
       run='mean' -> average of M; run='single' -> one random run."""
    vals = []
    for cands, idx in zip(problems, selection):
        s = cands[idx]["scores"]
        if run == "mean":
            vals.append(s.mean())
        else:
            vals.append(s[np.random.randint(len(s))])
    return np.array(vals)

def _true_rate(problems, selection):
    return np.mean([problems[i][selection[i]]["passed"] for i in range(len(problems))])

def exp2_power(pool, target_gap, R=400, seed=0):
    print("\n=== Experiment 2: power vs true gap (main plot) ===")
    rng = np.random.default_rng(seed)
    configs = [("freeform", "single"), ("freeform", "mean"),
               ("structured", "single"), ("structured", "mean")]
    records = {c: [] for c in configs}   # (gap, detected)

    pools_by_mode = {m: collect(pool, m) for m in ("freeform", "structured")}
    # align problem sets across modes by using the same problem indices
    nprob = min(len(pools_by_mode["freeform"]), len(pools_by_mode["structured"]))

    grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    for r in range(R):
        d = grid[rng.integers(len(grid))]
        # System A: random candidate per problem (use the same problem index mapping for all modes)
        problems_ff = pools_by_mode["freeform"][:nprob]
        problems_st = pools_by_mode["structured"][:nprob]
        selA = [rng.integers(len(p)) for p in problems_ff]
        # System B: degrade on fraction d of problems -> pick FAILED candidate if available
        selB = list(selA)
        ndeg = int(d * nprob)
        deg_idx = rng.choice(nprob, size=ndeg, replace=False) if ndeg > 0 else []
        for pi in deg_idx:
            fails = [k for k, c in enumerate(problems_ff[pi]) if c["passed"] == 0]
            if fails:
                selB[pi] = fails[rng.integers(len(fails))]

        gap = _true_rate(problems_ff, selA) - _true_rate(problems_ff, selB)

        for mode, run in configs:
            problems = problems_ff if mode == "freeform" else problems_st
            a = _system_scores(problems, selA, run)
            b = _system_scores(problems, selB, run)
            try:
                t, p = stats.ttest_rel(a, b)
                detected = (p < 0.05) and (a.mean() > b.mean())
            except Exception:
                detected = False
            records[(mode, run)].append((gap, bool(detected)))

    # Aggregate power around target_gap
    print(f"\n  Power to detect gap ~ {target_gap:.0%} (window ±0.7%):")
    headline = {}
    for c in configs:
        arr = np.array(records[c])
        mask = np.abs(arr[:, 0] - target_gap) < 0.007
        power = arr[mask, 1].mean() if mask.sum() else float("nan")
        headline[c] = power
        print(f"    {c[0]:11s} {c[1]:6s}: power = {power:.2f}  (n={int(mask.sum())})")

    _plot_power(records, target_gap)
    return headline

def exp3_validity(pool, threshold=0.6):
    print("\n=== Experiment 3: validity (correlation & F1 vs ground truth) ===")
    for mode in ("freeform", "structured"):
        passed, mean_score = [], []
        for tid, pinfo in pool["problems"].items():
            for c in pinfo["candidates"]:
                s = c["scores"].get(mode, [])
                if c["passed"] is not None and len(s) > 0:
                    passed.append(int(c["passed"])); mean_score.append(np.mean(s))
        passed = np.array(passed); mean_score = np.array(mean_score)
        if len(set(passed)) < 2:
            print(f"  {mode}: ground truth has no variation, skipping."); continue
        rho, _ = stats.spearmanr(mean_score, passed)
        rpb, _ = stats.pointbiserialr(passed, mean_score)
        pred = (mean_score >= threshold).astype(int)
        tp = int(((pred == 1) & (passed == 1)).sum())
        fp = int(((pred == 1) & (passed == 0)).sum())
        fn = int(((pred == 0) & (passed == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  {mode:11s}: Spearman={rho:.2f} | point-biserial={rpb:.2f} | "
              f"P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

def _plot_power(records, target_gap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 0.14, 8)
    centers = (bins[:-1] + bins[1:]) / 2
    plt.figure(figsize=(7, 5))
    for c in records:
        arr = np.array(records[c])
        ys = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            ys.append(arr[m, 1].mean() if m.sum() else np.nan)
        plt.plot(centers, ys, marker="o", label=f"{c[0]}/{c[1]}")
    plt.axvline(target_gap, ls="--", color="gray", label=f"target {target_gap:.0%}")
    plt.axhline(0.8, ls=":", color="black", alpha=0.5, label="power 0.8")
    plt.xlabel("True quality gap (pass-rate difference A−B)")
    plt.ylabel("Power (prob. of detecting, p<0.05)")
    plt.title("Judge's ability to detect small differences")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = ART / "power_vs_gap.png"; plt.savefig(out, dpi=130)
    print(f"\n  Plot saved: {out}")

def cmd_analyze(args):
    pool = load_pool()
    if pool is None:
        print("[error] pool.json does not exist yet."); return
    exp1_noise(pool)
    head = exp2_power(pool, args.target_gap, R=args.replicates, seed=args.seed)
    exp3_validity(pool)
    print("\n--- Headline summary ---")
    ff_single = head.get(("freeform", "single"), float("nan"))
    st_mean = head.get(("structured", "mean"), float("nan"))
    print(f"Free-form 1-run judge detecting gap {args.target_gap:.0%}: power = {ff_single:.2f}")
    print(f"Structured + M-avg judge detecting gap {args.target_gap:.0%}: power = {st_mean:.2f}")
    print("If the latter is much higher, the core hypothesis is PROVEN: "
          "stability (averaging + structured output) enables detection of small differences.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Mini-pilot LLM-judge evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate"); g.set_defaults(func=cmd_generate)
    g.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    g.add_argument("--n", type=int, default=50, help="number of problems")
    g.add_argument("--k", type=int, default=3, help="candidates per problem")
    g.add_argument("--gen-temp", type=float, default=0.6)
    g.add_argument("--seed", type=int, default=0)

    gt = sub.add_parser("groundtruth"); gt.set_defaults(func=cmd_groundtruth)
    gt.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")

    j = sub.add_parser("judge"); j.set_defaults(func=cmd_judge)
    j.add_argument("--m", type=int, default=10, help="number of judge runs per candidate per mode")
    j.add_argument("--judge-temp", type=float, default=0.7)

    a = sub.add_parser("analyze"); a.set_defaults(func=cmd_analyze)
    a.add_argument("--target-gap", type=float, default=0.02)
    a.add_argument("--replicates", type=int, default=400)
    a.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
