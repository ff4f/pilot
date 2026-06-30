#!/usr/bin/env python3
"""
merge_results.py — read EvalPlus results (run via Docker) into pool.json.

Used in the EvalPlus + Docker PATH. The sequence:
  1. python pilot.py generate --dataset humaneval --n 164 --k 2
  2. (Docker) run evalplus.evaluate  -> produces artifacts/*_eval_results.json
  3. python merge_results.py               <-- this script, fills pass/fail labels into pool.json
  4. python pilot.py judge --m 8
  5. python pilot.py analyze --target-gap 0.02

EvalPlus evaluates each candidate with two levels of strictness:
  - "base"  : original HumanEval unit tests
  - "plus"  : base + dozens of additional EvalPlus edge-case tests (stricter)
By default we use "plus" (maximum rigor). If the pass rate becomes too low,
run with `--metric base`.
"""
import json
import argparse
from pathlib import Path

ART = Path("artifacts")
POOL = ART / "pool.json"


def is_pass(x):
    """Handle status as 'pass'/'fail' OR ['pass', <detail>]."""
    if isinstance(x, (list, tuple)):
        x = x[0] if x else ""
    return str(x).strip().lower() == "pass"


def extract_pass_lists(results: dict, metric: str):
    """
    Convert EvalPlus result JSON -> {task_id: [bool, bool, ...]} ordered by submission.
    Handles multiple EvalPlus schemas (differs between versions).
    """
    out = {}
    eval_block = results.get("eval", results)
    for tid, entry in eval_block.items():
        statuses = []
        # Schema A: list of dict per sample (has plus_status/base_status)
        if isinstance(entry, list):
            key = f"{metric}_status"
            for r in entry:
                if isinstance(r, dict):
                    st = r.get(key, r.get("plus_status", r.get("base_status", r.get("status"))))
                    statuses.append(is_pass(st))
                else:
                    statuses.append(is_pass(r))
        # Schema B: dict with 'plus'/'base' list (each element is status or [status, detail])
        elif isinstance(entry, dict):
            arr = entry.get(metric) or entry.get("plus") or entry.get("base") or []
            for st in arr:
                statuses.append(is_pass(st))
        out[tid] = statuses
    return out


def find_results_file():
    candidates = sorted(ART.glob("*_eval_results.json")) + sorted(ART.glob("*_eval_results.jsonl"))
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["base", "plus"], default="plus",
                    help="EvalPlus verdict strictness level (default: plus = strictest)")
    ap.add_argument("--results", default=None, help="Path to EvalPlus results file (optional; auto-searches in artifacts/)")
    args = ap.parse_args()

    if not POOL.exists():
        print("[error] artifacts/pool.json does not exist yet. Run generate first.")
        return

    res_path = Path(args.results) if args.results else find_results_file()
    if not res_path or not res_path.exists():
        print("[error] EvalPlus results file not found in artifacts/.")
        print("        Make sure the Docker step has produced *_eval_results.json there.")
        return
    print(f"Reading EvalPlus results: {res_path}")

    results = json.loads(res_path.read_text())
    pass_lists = extract_pass_lists(results, args.metric)

    # Diagnostics if parsing appears empty (schema may differ)
    total_parsed = sum(len(v) for v in pass_lists.values())
    if total_parsed == 0:
        print("[warn] Parser found no statuses. File structure may be a different version.")
        print("       Top-level keys:", list(results.keys())[:10])
        sample_key = next(iter(results.get("eval", {})), None)
        if sample_key:
            print(f"       Sample entry '{sample_key}':")
            print("      ", json.dumps(results["eval"][sample_key], indent=2)[:600])
        print("       Send the above snippet and the parser will be adjusted.")
        return

    pool = json.loads(POOL.read_text())
    n_pass, n_total, n_missing = 0, 0, 0
    for tid, pinfo in pool["problems"].items():
        statuses = pass_lists.get(tid, [])
        for j, cand in enumerate(pinfo["candidates"]):
            if j < len(statuses):
                cand["passed"] = bool(statuses[j])
            else:
                cand["passed"] = False
                n_missing += 1
            n_pass += int(cand["passed"])
            n_total += 1
    POOL.write_text(json.dumps(pool, indent=2))

    rate = n_pass / n_total if n_total else 0.0
    print(f"\nMetric: {args.metric}")
    print(f"Candidate pass rate: {n_pass}/{n_total} = {rate:.1%}")
    if n_missing:
        print(f"[note] {n_missing} candidates have no results (marked as fail). "
              f"Check whether the number of tasks in samples == pool.")
    if not (0.15 < rate < 0.85):
        print("[note] Pass/fail mix is unbalanced for a good experiment.")
        print("       Try: regenerate with --gen-temp 0.8 / --k 3, or use --metric base.")
    print("\nNext: python pilot.py judge --m 8")


if __name__ == "__main__":
    main()
