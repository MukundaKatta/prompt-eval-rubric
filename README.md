# prompt-eval-rubric

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/prompt-eval-rubric.svg)](https://pypi.org/project/prompt-eval-rubric/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Score LLM outputs against named 0.0-1.0 rubrics with optional weights.** Zero deps. BYO scoring functions.

```python
from prompt_eval_rubric import Rubric, RubricSet
import re

length_ok = Rubric(
    name="length",
    score=lambda out, ctx=None: 1.0 if 100 <= len(out) <= 500 else 0.0,
)

has_citation = Rubric(
    name="has_citation",
    score=lambda out, ctx=None: 1.0 if re.search(r"\[[\w-]+\]", out) else 0.0,
)

rubrics = RubricSet([length_ok, has_citation])
report = rubrics.evaluate(llm_output, context={"query": "what is..."})

print(report.overall)                # average across rubrics
for s in report.scores:
    print(s.name, s.value, s.reason)
```

**Weighted aggregation:**

```python
weighted = RubricSet([
    (length_ok, 0.3),
    (has_citation, 0.7),
])
report = weighted.evaluate(llm_output)
# overall = (length.value * 0.3 + has_citation.value * 0.7) / 1.0
```

## Why

Validators answer "did it pass?" Rubrics answer "how good was it on each axis?" When you're tuning a prompt, you need the second. When you're guarding production, you need the first. Use both.

`prompt-eval-rubric` is the lightweight rubric primitive that fits an offline eval loop, an A/B prompt comparison, or a regression report. The library does not call any LLM — if you want an LLM-as-judge rubric, write a `score` function that does the call itself and remember the cost.

## What it does

- Score functions return `float`, `bool`, `int`, or `(float, reason_str)` tuples
- Returns clipped to [0.0, 1.0]; NaN treated as 0.0
- Exceptions inside a score function are caught and surfaced as `Score(value=0.0, reason="error: ...")` so one broken rubric doesn't poison the whole set
- Per-rubric weights, with normalization
- Duplicate rubric names rejected at `RubricSet` construction

## Install

```bash
pip install prompt-eval-rubric
```

## API

```python
from prompt_eval_rubric import Rubric, RubricSet, Score, Report

r = Rubric(name: str, score: Callable[[output, context=None], float | bool | (float, str)])
s = r.evaluate(output, context=None) -> Score(name, value, reason)

rs = RubricSet([r1, r2, ...])               # all equal weight
rs = RubricSet([(r1, 0.3), (r2, 0.7)])      # weighted

report = rs.evaluate(output, context=None) -> Report
report.scores                                # list[Score]
report.overall                               # weighted-average float
report.by_name()                             # {name: Score}
```

## Companion libraries

- [`llm-output-validator`](https://github.com/MukundaKatta/llm-output-validator) — binary pass/fail rules (length, regex, allowed, no_pii, json). Use validators for runtime gates and rubrics for offline tuning.
- [`agentsnap`](https://github.com/MukundaKatta/agentsnap) — snapshot agent runs; pair with rubrics for regression scoring.

## License

MIT
