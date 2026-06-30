"""
Prompt templates for the LLM-judge.

Two modes are intentionally differentiated to test H3:
- FREEFORM  : free-form output, just a single number. Tends to have high variance.
- STRUCTURED: JSON output with a fixed fault taxonomy + sub-scores (CodeJudge-style).
              Constrains the output space -> reduces randomness -> lower variance.

Note: the judge is NOT given unit tests / correct answers. It evaluates based on the
problem specification + candidate code only (similar to the LLM-judge scenario in HULA).
"""

# ----------------------------- FREE-FORM -----------------------------

FREEFORM_SYSTEM = (
    "You are a meticulous senior software engineer reviewing code. "
    "You judge whether a candidate solution correctly solves the given problem."
)

FREEFORM_USER_TEMPLATE = """Problem specification:
{problem}

Candidate solution:
```python
{code}
```

On a scale of 1 to 10, how likely is this solution to be CORRECT
(i.e., to pass all hidden tests, including edge cases)?
1 = almost certainly wrong, 10 = almost certainly correct.

Respond with ONLY a single integer from 1 to 10. No words, no explanation."""


# ----------------------------- STRUCTURED ----------------------------

STRUCTURED_SYSTEM = (
    "You are a meticulous senior software engineer reviewing code. "
    "You output ONLY valid JSON, no prose, no markdown fences."
)

# Fixed fault taxonomy = key to variance reduction (model cannot freely invent categories).
FAULT_TYPES = [
    "none",
    "logic_error",
    "edge_case_missed",
    "off_by_one",
    "wrong_output_format",
    "wrong_return_type",
    "runtime_error",
    "infinite_loop_or_timeout",
    "incomplete_implementation",
]

STRUCTURED_USER_TEMPLATE = """Problem specification:
{problem}

Candidate solution:
```python
{code}
```

Evaluate the solution. Identify concrete faults (if any) using ONLY these allowed
fault types: {fault_types}.

Then score three dimensions from 0 to 10, and give a final correctness score from 1 to 10
that reflects how likely the code passes all hidden tests including edge cases.

Respond with ONLY valid JSON in EXACTLY this schema (no markdown, no extra keys):
{{"faults": ["<one or more allowed fault types>"],
  "dimension_scores": {{"logic": <int 0-10>, "edge_cases": <int 0-10>, "robustness": <int 0-10>}},
  "final_score": <int 1-10>}}"""


def build_freeform_messages(problem: str, code: str):
    return [
        {"role": "system", "content": FREEFORM_SYSTEM},
        {"role": "user", "content": FREEFORM_USER_TEMPLATE.format(problem=problem, code=code)},
    ]


def build_structured_messages(problem: str, code: str):
    user = STRUCTURED_USER_TEMPLATE.format(
        problem=problem, code=code, fault_types=", ".join(FAULT_TYPES)
    )
    return [
        {"role": "system", "content": STRUCTURED_SYSTEM},
        {"role": "user", "content": user},
    ]
