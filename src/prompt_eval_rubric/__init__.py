"""prompt-eval-rubric - score LLM outputs against named 0.0-1.0 rubrics.

Validators reject; rubrics rank. When you're tuning a prompt, you want
to know how good the output is on each axis — length, citation density,
tone, factuality — not just whether it passes a binary gate.

`Rubric` wraps one scoring function. `RubricSet` aggregates many,
optionally with weights, and returns a `Report` with per-rubric
`Score` entries and an `overall` value.

    from prompt_eval_rubric import Rubric, RubricSet
    import re

    length_ok = Rubric(
        name="length",
        score=lambda out, ctx=None: 1.0 if 100 <= len(out) <= 500 else 0.0,
    )

    has_citation = Rubric(
        name="has_citation",
        score=lambda out, ctx=None: 1.0 if re.search(r"\\[[\\w-]+\\]", out) else 0.0,
    )

    rubrics = RubricSet([length_ok, has_citation])
    report = rubrics.evaluate(output, context={"query": "..."})

    print(report.overall)
    for s in report.scores:
        print(s.name, s.value, s.reason)

Optional weights:

    weighted = RubricSet(
        rubrics=[(length_ok, 0.3), (has_citation, 0.7)],
    )

The library does NOT call any LLM. If you want an LLM-as-judge rubric,
write a `score` function that does the call itself (and remember to
account for cost separately).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

__version__ = "0.1.0"
__all__ = [
    "Rubric",
    "RubricSet",
    "Score",
    "Report",
]


ScoreFn = Callable[..., Any]  # (output, context=None) -> float | (float, str)


@dataclass(frozen=True)
class Score:
    """One rubric's score for one output."""

    name: str
    value: float
    reason: str | None = None


@dataclass(frozen=True)
class Report:
    """Aggregate result from `RubricSet.evaluate`."""

    scores: list[Score]
    overall: float

    def by_name(self) -> dict[str, Score]:
        return {s.name: s for s in self.scores}


# ---- Rubric ---------------------------------------------------------------


class Rubric:
    """One scoring axis.

    Args:
        name: stable identifier for this rubric (appears in `Score.name`).
        score: callable accepting `(output, context=None)` and returning
            either a float (0.0..1.0) or a `(float, reason)` tuple.

    The returned value is clipped to [0.0, 1.0]. Exceptions inside the
    score function are caught and produce a `Score(value=0.0,
    reason=f"error: {exc}")` so a bad rubric doesn't break the whole
    set evaluation.
    """

    def __init__(self, name: str, score: ScoreFn) -> None:
        if not name:
            raise ValueError("rubric name must be non-empty")
        self.name = name
        self._score = score

    def evaluate(self, output: Any, context: Any | None = None) -> Score:
        try:
            result = self._score(output, context)
            value, reason = self._unpack(result)
        except BaseException as exc:  # noqa: BLE001 - shield aggregate eval
            return Score(name=self.name, value=0.0, reason=f"error: {exc!r}")
        clipped = _clip01(value)
        return Score(name=self.name, value=clipped, reason=reason)

    @staticmethod
    def _unpack(result: Any) -> tuple[float, str | None]:
        if isinstance(result, tuple) and len(result) == 2:
            v, r = result
            if not isinstance(r, (str, type(None))):
                r = str(r)
            return float(v), r
        if isinstance(result, bool):
            return (1.0 if result else 0.0), None
        if isinstance(result, (int, float)):
            return float(result), None
        raise TypeError(
            f"rubric score must return float, bool, or (float, reason); "
            f"got {type(result).__name__}"
        )


def _clip01(v: float) -> float:
    if v != v:  # NaN
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


# ---- RubricSet -----------------------------------------------------------


RubricEntry = Rubric | tuple[Rubric, float]


class RubricSet:
    """Aggregate multiple rubrics.

    Pass either bare `Rubric`s (each gets weight 1.0) or
    `(Rubric, weight)` tuples. Weights are normalized so the overall
    score stays in [0.0, 1.0].
    """

    def __init__(self, rubrics: Iterable[RubricEntry]) -> None:
        entries: list[tuple[Rubric, float]] = []
        for r in rubrics:
            if isinstance(r, Rubric):
                entries.append((r, 1.0))
            elif (
                isinstance(r, tuple)
                and len(r) == 2
                and isinstance(r[0], Rubric)
                and isinstance(r[1], (int, float))
            ):
                w = float(r[1])
                if w < 0:
                    raise ValueError(f"weight must be >= 0; got {w}")
                entries.append((r[0], w))
            else:
                raise TypeError(
                    "RubricSet expects Rubric or (Rubric, weight); "
                    f"got {type(r).__name__}"
                )
        if not entries:
            raise ValueError("RubricSet requires at least one rubric")
        # detect duplicate names early
        names = [r.name for r, _ in entries]
        if len(set(names)) != len(names):
            raise ValueError("rubric names must be unique within a set")
        self._entries = entries

    @property
    def rubrics(self) -> list[Rubric]:
        return [r for r, _ in self._entries]

    @property
    def weights(self) -> list[float]:
        return [w for _, w in self._entries]

    def evaluate(self, output: Any, context: Any | None = None) -> Report:
        scores = [r.evaluate(output, context) for r, _ in self._entries]
        total_weight = sum(w for _, w in self._entries)
        if total_weight == 0:
            overall = 0.0
        else:
            overall = sum(
                s.value * w for s, (_, w) in zip(scores, self._entries)
            ) / total_weight
        return Report(scores=scores, overall=overall)
