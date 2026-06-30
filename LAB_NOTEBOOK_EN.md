# Lab Notebook — Reliable Low-Cost Evaluation of LLM Coding Agents

Chronological research log. Each experiment is recorded in full: hypotheses, parameters, raw results, honest interpretation (including null/negative findings), and the decision that followed. Goal: a reproducible trail plus a justification for every change of direction.

Researcher: Faliqul Fikri Al Fauzani
Context: pilot for a Master by Research proposal on evaluating LLM coding agents (the HULA research line; Pasuksmit et al., 2025).

---

## General setup (applies to all experiments unless stated otherwise)

- **Machine**: MacBook (Apple Silicon), Python 3.9, venv.
- **Benchmark + ground truth**: HumanEval+ via EvalPlus (Liu et al., 2023), run inside Docker (`ganler/evalplus`, `--platform linux/amd64`). Ground-truth verdicts use the **plus** level (base tests + ~80x additional edge-case tests).
- **Solution generator**: local Ollama, `qwen2.5-coder:7b`.
- **LLM judge**: local Ollama, `qwen2.5-coder:7b`. The judge sees only the problem specification + candidate code (no unit tests).
- **Two judge modes**:
  - *free-form*: outputs a single integer score 1-10.
  - *structured*: outputs JSON with a fixed fault taxonomy + dimension sub-scores + a final score (CodeJudge style; Tong & Zhang, 2024).
- **Pilot code**: `pilot.py` (generate/judge/analyze), `merge_results.py` (parse EvalPlus output into pool.json), `analyze_mdd.py` (Experiment 2 analysis). Raw judge scores are stored in `artifacts/pool.json`.

---

## Experiment 1 — Characterising judge noise & testing mitigations (baseline pilot)

**Date**: [fill in]
**Status**: Complete. **NULL** result for the main hypothesis (see interpretation).

### Aim
Test several hypotheses within a single pipeline:
- H1: a single-run LLM judge is too noisy to reliably detect a quality difference of ~2%.
- H2: averaging M runs lowers the noise -> restores detection of 2%.
- H3: structured output (CodeJudge style) has lower variance than free-form.
- H4 (validity): a more stable judge configuration is also more valid against ground truth.

### Exact parameters
| Parameter | Value |
|---|---|
| Dataset | HumanEval+ (164 problems) |
| Candidates per problem (k) | 2 |
| Total candidates | 328 |
| Generator temperature | 0.6 |
| Judge runs per candidate per mode (M) | 8 |
| Judge temperature | 0.7 |
| Judge modes | free-form, structured |
| Total judge calls | 328 x 8 x 2 = 5,248 |
| Ground-truth metric | plus |
| Power replicates (internal to Exp 2) | 400 |
| A/B pair construction | random pick per problem; B degraded on a fraction d of problems by swapping to a failing candidate; grid d = {0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12} |
| Detection test | paired t-test across problems, p<0.05 + correct direction |

### Raw results

**Ground truth (candidate mix):**
- Candidate pass rate: **257/328 = 78.4%** (metric: plus)

**Sub-experiment 1 - judge noise (per-candidate SD across runs):**
| Mode | Median SD/item | Mean SD/item | n |
|---|---|---|---|
| free-form | 0.053 | 0.075 | 328 |
| structured | 0.053 | 0.084 | 328 |

**Sub-experiment 2 - power to detect a ~2% gap (+/-0.7%):**
| Configuration | Power | n (pairs near 2%) |
|---|---|---|
| free-form / single | 0.00 | 4 |
| free-form / mean | 0.00 | 4 |
| structured / single | 0.00 | 4 |
| structured / mean | 0.00 | 4 |

**Sub-experiment 3 - validity vs ground truth:**
| Mode | Spearman | Point-biserial | Precision | Recall | F1 |
|---|---|---|---|---|---|
| free-form | 0.31 | 0.38 | 0.80 | 0.99 | 0.89 |
| structured | 0.33 | 0.42 | 0.81 | 0.98 | 0.89 |

### Interpretation (honest)

**H1 - partially confirmed.** Run-to-run noise is real (per-item SD ~0.05-0.08 on a 0-1 scale). The basic premise that LLM-judge evaluation is non-deterministic holds.

**H2 - NOT supported in this experiment.** Power = 0 across all configurations, including the intended mitigation (structured + mean). Averaging showed no advantage **in this design**.

**H3 - NOT supported.** Mean structured SD (0.084) was slightly higher than free-form (0.075). Structured output did not reduce variance here.

**H4 - weak/misleading.** Moderate correlation (Spearman ~0.32). The F1 of 0.89 looks high but is an artefact of the 78% pass rate (most candidates pass -> recall inflated to 0.98-0.99). Not strong evidence of validity.

### Diagnosis: why Exp 2 was null (two causes)

1. **Data too easy (78% pass rate).** Only **n=4** pairs had a gap near 2%. A power estimate from 4 points is not meaningful. qwen2.5-coder is too strong on HumanEval -> too few failing candidates to construct varied small gaps. *Fix: regenerate to a 40-60% pass rate (higher temperature / weaker model).*

2. **Flawed experiment design (the main cause).** Exp 2 tested detection via a *paired t-test across problems within a single run*. At a 2% gap over 164 problems, only ~3 problems differ - a signal too small for any method to detect, even a perfect judge. Lowering judge noise does not help because the bottleneck is not judge noise but the number of differing problems. There is also a scale mismatch: the gap is measured in *pass-rate* space, but the judge produces *scores* - two different scales. **This design measures the wrong kind of variability** relative to what the literature means (run-to-run noise on the aggregate score that masks a 1-2% improvement).

### Conclusion & decision
The pipeline is validated end to end (generate -> ground truth -> judge -> analyze runs, raw scores stored). However, the Exp 2 design does not answer the intended research question, so the mitigation results are inconclusive. **Decision: proceed to Experiment 2 with a corrected design (Approach A)** - measure run-to-run noise on the aggregate score and express it as a Minimum Detectable Difference (MDD), instead of a paired t-test across problems. Re-judging is not required; the existing raw scores are re-analysed.

### Artefacts
- `artifacts/pool.json` (raw scores from 5,248 calls + ground-truth labels)
- `artifacts/samples_eval_results.json` (EvalPlus results)
- `artifacts/power_vs_gap.png` (Exp 2 plot - flat, kept as evidence of the null result)

---

## Experiment 2 — Minimum-Detectable-Difference redesign (Approach A)

**Date**: [fill in]
**Status**: Complete. Main hypothesis (H2) **supported**; H3 and H4 reported as negative/limited.

### Motivation
Fix the Exp 2 design flaw. Instead of planting an artificial gap and running a paired t-test across problems, this measures directly the **stability of the aggregate score across independent runs**, then derives a **Minimum Detectable Difference (MDD)** per judge configuration.

MDD = the smallest difference in the aggregate score that can be detected reliably (alpha=0.05, power=0.8) ~= 2.8 x aggregate standard error.

This framing matches the HULA problem statement directly ("detect 1-2% improvements").

### Exact parameters
| Parameter | Value |
|---|---|
| Dataset | HumanEval+ (164 problems) |
| Candidates per problem (k) | 3 |
| Total candidates | 492 |
| Generator temperature | 1.0 |
| Generator | qwen2.5-coder:7b |
| Judge runs per candidate per mode (M) | 8 |
| Judge temperature | 0.7 |
| Judge | qwen2.5-coder:7b |
| Total judge calls | 492 x 8 x 2 = 7,872 |
| Ground-truth metric | plus |
| Candidate pass rate | **395/492 = 80.3%** |
| Analysis | `analyze_mdd.py`, bootstrap = 2000 |
| MDD factor | 2.8 (two-sided alpha=0.05, power=0.8) |

Note: Exp 1's raw judge scores were NOT reused; data was regenerated (k=3, temp=1.0) in an attempt to lower the pass rate. The pass rate stayed ~80% anyway (see methodology note M-2). The MDD analysis was run on this 80% data.

### Raw results

**A. Per-item noise (SD across runs):**
| Mode | Median SD/item | Mean SD/item | n |
|---|---|---|---|
| free-form | 0.053 | 0.071 | 492 |
| structured | 0.053 | 0.084 | 492 |

**B. MDD on the aggregate score (164 problems, M=8):**
| m (runs averaged) | free-form SE | free-form MDD | structured SE | structured MDD |
|---|---|---|---|---|
| 1 | 0.0065 | **1.82%** | 0.0073 | **2.05%** |
| 2 | 0.0046 | 1.30% | 0.0050 | 1.40% |
| 3 | 0.0037 | 1.05% | 0.0041 | 1.14% |
| 4 | 0.0032 | 0.91% | 0.0035 | 0.97% |
| 5 | 0.0029 | 0.81% | 0.0032 | 0.91% |
| 6 | 0.0027 | 0.75% | 0.0029 | 0.82% |
| 7 | 0.0025 | 0.69% | 0.0027 | 0.75% |
| 8 | 0.0023 | **0.64%** | 0.0025 | **0.69%** |

**C. Validity vs ground truth:**
| Mode | Spearman(score, pass) | data pass rate |
|---|---|---|
| free-form | 0.25 | 80.3% |
| structured | 0.24 | 80.3% |

Plot artefact: `artifacts/mdd_vs_runs.png` (MDD vs m).

### Interpretation (honest)

**H1 - confirmed & quantified.** Single-run MDD is 1.82% (free-form) and 2.05% (structured). Both sit **right at the 2% threshold**, meaning single-run evaluation can barely detect a 2% improvement reliably. This reproduces the HULA problem independently on open data.

**H2 - SUPPORTED.** Averaging lowers MDD consistently and **closely follows the 1/sqrt(m) law** (free-form: m=1 -> 1.82%, m=4 -> 0.91% [~2x lower, sqrt(4)=2], m=8 -> 0.64%). The first configuration to cross MDD <= 2% is free-form at m=8 (free-form m=1 is already ~1.82%, at the threshold). Averaging 8 runs gives MDD 0.64% - well below 2%. **Stabilisation via averaging restores detection of a 2% improvement, at a cost of 8x judge calls.**

**H3 - NOT supported (opposite of expectation).** Structured output did **not** reduce variance; structured MDD is **higher** than free-form at every m (e.g. m=1: 2.05% vs 1.82%; m=8: 0.69% vs 0.64%), and per-item SD is also higher (0.084 vs 0.071). On this setup, fault-taxonomy + JSON did not help stability and slightly hurt it. Worth investigating (perhaps the structured prompt is too complex for a 7B model, or it adds decision branches that increase variance). **Reported as it stands; not claimed as a successful mitigation.**

**H4 (validity) - weak, and an important caveat.** Spearman is only 0.24-0.25: the judge is stable but **correlates weakly with actual correctness**. Consequence: we can measure a 0.64% difference reliably, but what we measure is "the judge's opinion", not true quality. **Stability is not validity.** The high F1 in Exp 1 was an artefact of a high pass rate, not strong validity. This opens a research direction: improving validity without sacrificing stability.

### Decision / next steps
Experiment 2 produces evidence usable as the proposal's Preliminary Work:
- Data-supported claims: (1) noise is real, single-run MDD ~2% (reproducing the HULA problem); (2) averaging restores 2% detection following 1/sqrt(m), at 8x runs.
- Claims that MUST be reported honestly as negative/limited: structured judging did not help (H3 failed); judge validity is weak (H4).
- Full-study follow-ups: (a) investigate why structured failed on a small model; (b) improve validity (correlation to ground truth); (c) combine with selective unit-test generation (proposal Stage 3); (d) test at higher difficulty (SWE-bench) beyond short HumanEval functions.
- Practical decision: an 80% pass rate does NOT hinder the MDD analysis (MDD measures aggregate stability, not the need for many failing candidates). No further regeneration needed for this purpose.

### Artefacts
- `artifacts/pool.json` (raw scores from 7,872 calls + ground-truth labels, k=3)
- `artifacts/samples_eval_results.json` (EvalPlus results, 492 candidates)
- `artifacts/mdd_vs_runs.png` (MDD vs m - the main Exp 2 plot)
- `analyze_mdd.py` (analysis code)

---

## Methodology notes (lessons recorded for reproducibility)

- **M-1 - EvalPlus silently caches results.** If a `*_eval_results.json` file already exists, EvalPlus loads the old results ("Load from previous results") and does NOT re-evaluate. After regenerating, you MUST `rm artifacts/samples_eval_results.json` before evaluating, or the pass rate/labels become wrong (old data mixed with new candidates marked as failing).
- **M-2 - Temperature is not an effective lever for lowering pass rate on a strong model.** Raising temperature (0.6 -> 1.0) on qwen2.5-coder:7b barely changed HumanEval pass rate (~78% -> ~80%): temperature varies code style, not correctness. To genuinely lower the pass rate, use a weaker generator model (e.g. qwen2.5-coder:1.5b). (Not done, because 80% turned out to be sufficient for the MDD analysis.)
- **M-3 - Stability and validity are different things.** A very stable aggregate score (small MDD) does not guarantee the judge is accurate (correlation to ground truth can remain weak). Always report both.

---

## Template for the next experiment entry

```
## Experiment N — [title]
Date:
Status:
### Aim / hypotheses
### Exact parameters
### Raw results
### Interpretation (honest, including null findings)
### Decision / next steps
### Artefacts
```
