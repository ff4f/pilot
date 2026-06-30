# Reliable, Low-Cost Evaluation of LLM Coding Agents — a pilot study

A small, reproducible pilot on how to evaluate LLM-based software-development agents **reliably and cheaply**. It reproduces, on fully open data, the evaluation-instability problem reported for an industrial human-in-the-loop coding agent (HULA, MSR 2025), and measures how well simple variance-reduction techniques fix it.

Built as a proof of capability for a Master by Research application. The aim was something reproducible, statistically careful, and runnable on a laptop.

---

## Motivation

LLM-based coding agents are evaluated in two main ways: running generated code against unit tests (functional correctness), and using a separate LLM to score it (LLM-as-judge). Industrial experience with the HULA framework (Pasuksmit et al., 2025) reports that LLM-judge scoring **fluctuates from run to run**, which makes it hard to detect the small (1-2%) improvements that matter during iterative development, and calls for a more stable, lower-cost evaluation method. This pilot takes up that gap directly.

---

## A note on HULA

HULA is not public (internal to Atlassian, productised as the commercial Rovo Dev). This pilot **does not run or compare against HULA**. It reproduces the *evaluation-method* problem - which is general to LLM-judge evaluation of coding agents - on the open HumanEval+ benchmark, and uses HULA's reported numbers only as motivation. The "can it detect 2%?" question is answered by controlled measurement on open data, where ground-truth correctness is known from unit tests.

---

## Key results

Measured on HumanEval+ (164 problems), with a local `qwen2.5-coder:7b` judge scoring each candidate 8 times in two styles. "MDD" is the **Minimum Detectable Difference**: the smallest difference in aggregate quality the judge can detect reliably (alpha = 0.05, power = 0.8, approximately 2.8 x aggregate standard error).

| Judge runs averaged (m) | MDD, free-form | MDD, structured |
|---|---|---|
| 1 (single run) | 1.82% | 2.05% |
| 4 | 0.91% | 0.97% |
| 8 | 0.64% | 0.69% |

- **A single judge run sits right at the ~2% threshold** - reproducing, independently and on open data, the instability HULA reports.
- **Averaging works.** MDD falls from 1.82% (1 run) to 0.64% (8 runs), closely tracking the expected `1/sqrt(m)` relationship. A simple averaging step restores reliable detection of 2% improvements, at a predictable compute cost.
- **Structured judging did not help.** Contrary to expectation, structured rubric-based judging (CodeJudge-style) did **not** reduce variance - it was marginally worse than free-form at every `m`.
- **Stability is not validity.** The judge's scores correlated only weakly with ground-truth correctness (Spearman approximately 0.25). A stable measurement is not necessarily an accurate one - a distinction that motivates further work.

The full chronological trail - including an initial experiment whose design was flawed, diagnosed, and corrected - is in [`LAB_NOTEBOOK_EN.md`](LAB_NOTEBOOK_EN.md).

---

## Method (corrected design)

1. **Generate** candidate solutions with a generator LLM (a mix of correct and incorrect).
2. **Ground truth**: establish pass/fail for each candidate with EvalPlus (strict HumanEval+ tests).
3. **Judge**: score each candidate repeatedly with an LLM judge, in free-form and structured modes; store all raw scores.
4. **Analyse**: measure the run-to-run stability of the *aggregate* score across independent runs, and express it as the Minimum Detectable Difference for each configuration. (An earlier design instead planted artificial gaps and ran a paired t-test across problems; this measured the wrong kind of variability and was replaced - see the lab notebook.)

---

## Repository

| File | Purpose |
|---|---|
| `pilot.py` | generate candidates, run the LLM judge (and a legacy analysis) |
| `merge_results.py` | parse EvalPlus output into `pool.json` |
| `analyze_mdd.py` | the MDD analysis - produces the main result |
| `prompts.py` | free-form and structured judge prompts |
| `LAB_NOTEBOOK_EN.md` | full chronological research log (incl. the null first attempt) |
| `artifacts/` | raw judge scores (`pool.json`), EvalPlus results, plots |
| `requirements.txt` | dependencies |

---

## How to run

Stack: Python 3.9+, Ollama (free local LLMs), EvalPlus + Docker, SciPy. The pilot is about variance, not absolute judge quality, so a free local model is sufficient.

**Setup**
```bash
pip install -r requirements.txt
pip install datasets
export LLM_BASE_URL="http://localhost:11434/v1"   # local Ollama
export LLM_API_KEY="ollama"
export GEN_MODEL="qwen2.5-coder:7b"
export JUDGE_MODEL="qwen2.5-coder:7b"
```

**1. Generate candidates** (local)
```bash
python pilot.py generate --dataset humaneval --n 164 --k 3 --gen-temp 1.0
```

**2. Ground truth via EvalPlus in Docker** (the local EvalPlus sandbox fails on macOS; Docker avoids it)
```bash
rm -f artifacts/samples_eval_results.json   # EvalPlus caches results - clear before re-evaluating
docker run --rm --platform linux/amd64 -v "$(pwd)":/app \
  ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval --samples /app/artifacts/samples.jsonl
python merge_results.py
```

**3. Judge** (local; the slow step - runs for several hours, cached and resumable)
```bash
python pilot.py judge --m 8
```

**4. Analyse** (the MDD result + plot)
```bash
python analyze_mdd.py
```

Each step checkpoints to `artifacts/`, so it can be stopped and resumed without repeating expensive calls.

---

## Limitations

- Limited to short, self-contained functions (HumanEval+), not repository-scale tasks such as SWE-bench.
- A single judge model; the *phenomenon* generalises, the exact numbers do not.
- Ground truth comes from EvalPlus unit tests, not human judgement.
- The i.i.d. assumption across judge runs is a simplification; aggregate uncertainty is estimated by bootstrap.

---

## References

- Pasuksmit, J., Takerngsaksiri, W., Thongtanunam, P., Tantithamthavorn, C., et al. (2025). *Human-in-the-loop software development agents: challenges and future directions.* MSR (Industry Track).
- Takerngsaksiri, W., Pasuksmit, J., Thongtanunam, P., Tantithamthavorn, C., et al. (2025). *Human-in-the-loop software development agents.* ICSE.
- Liu, J., Xia, C. S., Wang, Y., & Zhang, L. (2023). *Is your code generated by ChatGPT really correct? Rigorous evaluation of LLMs for code generation* (EvalPlus). NeurIPS.
- Tong, W., & Zhang, T. (2024). *CodeJudge: evaluating code generation with large language models.* EMNLP.
