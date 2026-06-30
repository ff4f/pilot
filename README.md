# Mini-Pilot: Stability & Cost of Evaluation for LLM-based Coding Agents

**Goal:** empirically prove that **single-run LLM-judge is too noisy to detect ~2% quality differences**, and that **variance reduction (M-run averaging and/or structured output) fixes it** — the exact gap raised in the *HULA: Challenges and Future Directions* paper (MSR 2025).

This is a "proof of capability" for a Master by Research application to A/Prof Patanamon Thongtanunam. The target: reproducible, statistically rigorous, cheap, and completed in ~1 week.

---

## ⚠️ Important clarification about HULA

HULA is **not** publicly available (internal Atlassian → commercial product Rovo Dev). There is no code/weights to run.

Design consequences:
- This pilot **does not run or compare against HULA**.
- The pilot demonstrates **the problem with the evaluation method**, which applies generally to LLM-judge-based evaluation of coding agents.
- HULA's numbers (37.2% resolution on SWE-bench Verified, F1 0.67 vs unit tests, correlation 0.7 vs humans) are used as **motivation**, cited from their paper — not replicated.
- The question "can it detect 2%?" is answered through **controlled construction on an open benchmark** (true gap is known from unit tests).

This is actually an advantage: reproducible, self-contained, and directly answers their future work.

---

## Claims to be proven (hypotheses)

1. **H1 (the problem):** the variance of LLM-judge scores across runs is large enough that a single-run LLM-judge **fails** to detect a true quality difference of 2% (low power).
2. **H2 (remedy #1 — averaging):** averaging M runs reduces variance ≈ `σ²/M`, increasing power until 2% is detectable.
3. **H3 (remedy #2 — structured output):** a judge with structured output + fault taxonomy (CodeJudge-style) has lower per-item variance than free-form, thus requiring smaller M for the same power.
4. **H4 (validity):** the more stable judge configuration is also **more valid** (higher correlation & F1 against ground-truth unit tests).

---

## What you need (tech stack)

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.9+ | |
| Data + ground truth | **EvalPlus** (HumanEval+ / MBPP+) | `pip install evalplus`; strict unit tests included |
| Test execution | EvalPlus (via Docker, safe) | automatic ground-truth pass/fail |
| Solution generator | Any LLM (OpenAI-compatible) | generates correct & incorrect candidates |
| Judge | Any LLM (OpenAI-compatible) | the one whose noise is being evaluated |
| Analysis | numpy, scipy, matplotlib, pandas | statistics + plots |

**FREE / cheap LLM options (important for your budget):**
- **Ollama (local, free)** — `qwen2.5-coder:7b` or `llama3.1:8b`. Base URL `http://localhost:11434/v1`. Run overnight, zero cost. **Primary recommendation for pilot.**
- **Groq (free tier)** — fast, has a daily free quota.
- **Google Gemini (free tier)** — OpenAI-compatible endpoint available.
- **OpenRouter** — some `:free` models.
- **DeepSeek** — very cheap if you want a stronger judge.

> This pilot is about **variance**, not absolute judge quality. Even a mediocre judge is sufficient to demonstrate the phenomenon. So use a free one.

---

## Source material (how to get data)

EvalPlus provides problems + strict test suites. No need to create a manual dataset.

```python
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
data = get_human_eval_plus()      # 164 problems, or get_mbpp_plus() (378 problems)
# data[task_id]["prompt"] contains the specification + function signature
```

Ground truth is obtained by running:
```bash
evalplus.evaluate --dataset humaneval --samples artifacts/samples.jsonl
# produces artifacts/samples_eval_results.json with pass/fail status per candidate
```

(For a more "impressive" but heavier version: SWE-bench Lite via HuggingFace `princeton-nlp/SWE-bench_Lite` + its Docker harness. Keep as a stretch goal — start with EvalPlus first.)

---

## Experiment design (upstream → downstream)

### Step 0 — Setup
`pip install -r requirements.txt`, set LLM env vars (see `pilot.py`), prepare Docker for EvalPlus.

### Step 1 — Generate candidates
For **N problems**, generate **k candidates** per problem from the generator LLM at temperature ~0.6. The goal is to get a **mix of correct & incorrect solutions** (that's what we need). Save to `samples.jsonl`.

### Step 2 — Ground truth (true labels)
Run EvalPlus → each candidate gets a **pass(1)/fail(0)** label from strict unit tests. This is the "truth" we use to measure the true gap.

### Step 3 — Judge (the expensive step)
For each candidate, run the LLM-judge **M times** in **2 modes**:
- **free-form**: "rate 1–10 how likely this solution is correct".
- **structured**: JSON output with dimension sub-scores + fault list (fixed taxonomy), then `final_score` (CodeJudge-style → constrains output → lower variance).

Save **all raw scores** (for reproducibility). Results are cached to avoid repeating expensive calls.

**Judge call budget** ≈ `N × k × M × 2`. Size it to fit:
- Minimum viable: N=50, k=3, M=10 → 3,000 calls.
- Standard: N=80, k=4, M=12 → 7,680 calls.
- With local Ollama: free, just wait.

### Step 4 — Analysis (3 experiments)

**Experiment 1 — Noise characterization (H1, H3).**
Per candidate, compute **SD of scores across M runs**. Report median SD, free-form vs structured. *Expected:* structured < free-form.

**Experiment 2 — Power vs true gap (H1, H2, H3) — MAIN PLOT.**
- Construct many **system pairs (A, B)** by selecting one candidate per problem for each; some pairs are deliberately made with B worse on a random fraction of problems → produces a range of gaps.
- For each pair, measure the **true gap** g = pass-rate(A) − pass-rate(B) from ground truth.
- For each judge configuration {single-run, M-avg} × {free-form, structured}: take judge scores per problem for the selected candidates, run a **paired t-test** across problems. "Detected" = p < 0.05 **and** correct direction.
- Aggregate: **detection rate (power)** as a function of g. *Read at g ≈ 0.02.*
- *Expected:* single-run free-form ≈ powerless at 2%; M-avg structured ≈ has power. **This is the core evidence.**

**Experiment 3 — Validity (H4).**
Correlate average judge score per candidate with ground truth (Spearman + point-biserial), and compute **F1** from ("judge says good" ≥ threshold) vs (passes test). Per mode and M. *Expected:* M-avg structured has the highest correlation & F1 (echoing HULA numbers: corr 0.7 / F1 0.67).

---

## Deliverables

1. **1-page summary** (PDF/Markdown): claims, methods, 3 findings, and "implication: to detect 1–2% improvements in agents like HULA, evaluation requires [X] runs or structured output".
2. **3 plots**: (a) per-item SD distribution, (b) power vs true gap, (c) correlation/F1 table.
3. **Clean repo**: code + config + raw scores + seed → anyone can reproduce.
4. (Optional) **Notebook** that reproduces all plots.

---

## Why this sets you apart

- 99% of applicants write "I'm interested in your research". You come with **an experiment that reproduces their gap + measures the solution**, with proper statistics (CI, t-test, power).
- This speaks **Pick's language** (empirical, replication-driven) and leverages **your strengths** (engineering, evaluation harnesses).
- Self-contained & reproducible → credible, not empty claims.
- Directly maps to future work she wrote herself → you "come to help with her research", not bringing your own topic.

---

## Threats to validity (mention this in the summary — makes you look mature)

- Single judge & limited dataset (HumanEval+/MBPP+ = short functions, not repos) → not a universal claim; this is a **pilot**, indicating direction.
- True gap from strict EvalPlus unit tests, not from human judgment → valid trade-off for scale.
- i.i.d. assumption across runs for variance reduction = simplification; in the report, supplement with bootstrap CI if needed.
- Results depend on judge model; report model & seed, and note that the phenomenon (not the exact numbers) is what generalizes.

---

## How to run (brief)

```bash
pip install -r requirements.txt
export LLM_BASE_URL="http://localhost:11434/v1"   # example: local Ollama
export LLM_API_KEY="ollama"
export GEN_MODEL="qwen2.5-coder:7b"
export JUDGE_MODEL="qwen2.5-coder:7b"

python pilot.py generate   --dataset humaneval --n 50 --k 3 --gen-temp 0.6
python pilot.py groundtruth --dataset humaneval
python pilot.py judge      --m 10
python pilot.py analyze    --target-gap 0.02
```

Each step saves a checkpoint to `artifacts/` so it can be resumed/repeated without repeating expensive calls.

See `pilot.py` for details and `prompts.py` for judge prompts.
