"""Tests for prompt_eval_rubric."""

from __future__ import annotations

import math

import pytest

from prompt_eval_rubric import Report, Rubric, RubricSet, Score


# ---- Rubric basics --------------------------------------------------


def test_rubric_returns_score_object():
    r = Rubric("len", score=lambda out, ctx=None: 1.0 if len(out) > 5 else 0.0)
    s = r.evaluate("hello world")
    assert isinstance(s, Score)
    assert s.name == "len"
    assert s.value == 1.0


def test_rubric_score_clipped_below_zero():
    r = Rubric("x", score=lambda out, ctx=None: -5.0)
    assert r.evaluate("anything").value == 0.0


def test_rubric_score_clipped_above_one():
    r = Rubric("x", score=lambda out, ctx=None: 100.0)
    assert r.evaluate("anything").value == 1.0


def test_rubric_nan_becomes_zero():
    r = Rubric("x", score=lambda out, ctx=None: float("nan"))
    assert r.evaluate("anything").value == 0.0


def test_rubric_bool_promotes_to_one_or_zero():
    r_true = Rubric("t", score=lambda out, ctx=None: True)
    r_false = Rubric("f", score=lambda out, ctx=None: False)
    assert r_true.evaluate("x").value == 1.0
    assert r_false.evaluate("x").value == 0.0


def test_rubric_int_works():
    r = Rubric("x", score=lambda out, ctx=None: 1)
    assert r.evaluate("x").value == 1.0


def test_rubric_tuple_returns_reason():
    r = Rubric(
        "x",
        score=lambda out, ctx=None: (0.5, "halfway there"),
    )
    s = r.evaluate("x")
    assert s.value == 0.5
    assert s.reason == "halfway there"


def test_rubric_tuple_reason_coerced_to_str():
    r = Rubric("x", score=lambda out, ctx=None: (0.5, 42))
    assert r.evaluate("x").reason == "42"


def test_rubric_exception_caught_and_reported():
    def boom(out, ctx=None):
        raise RuntimeError("kaboom")

    r = Rubric("x", score=boom)
    s = r.evaluate("anything")
    assert s.value == 0.0
    assert "error" in (s.reason or "").lower()
    assert "kaboom" in (s.reason or "")


def test_rubric_rejects_non_numeric_return():
    r = Rubric("x", score=lambda out, ctx=None: "high")
    s = r.evaluate("x")
    # Exception path: TypeError → reported as error
    assert s.value == 0.0
    assert "error" in (s.reason or "").lower()


def test_rubric_empty_name_rejected():
    with pytest.raises(ValueError):
        Rubric("", score=lambda out, ctx=None: 1.0)


def test_rubric_passes_context_to_score_fn():
    seen = []

    def fn(out, ctx=None):
        seen.append(ctx)
        return 1.0

    r = Rubric("x", score=fn)
    r.evaluate("text", context={"query": "what is..."})
    assert seen == [{"query": "what is..."}]


# ---- RubricSet ----------------------------------------------------


def test_set_evaluates_all_rubrics():
    r1 = Rubric("a", score=lambda out, ctx=None: 1.0)
    r2 = Rubric("b", score=lambda out, ctx=None: 0.5)
    rs = RubricSet([r1, r2])
    report = rs.evaluate("anything")
    assert isinstance(report, Report)
    assert len(report.scores) == 2
    assert report.scores[0].value == 1.0
    assert report.scores[1].value == 0.5


def test_set_overall_unweighted_average():
    r1 = Rubric("a", score=lambda out, ctx=None: 1.0)
    r2 = Rubric("b", score=lambda out, ctx=None: 0.0)
    rs = RubricSet([r1, r2])
    report = rs.evaluate("x")
    assert report.overall == 0.5


def test_set_overall_weighted_average():
    r1 = Rubric("a", score=lambda out, ctx=None: 1.0)
    r2 = Rubric("b", score=lambda out, ctx=None: 0.0)
    rs = RubricSet([(r1, 3.0), (r2, 1.0)])
    report = rs.evaluate("x")
    # (1.0*3 + 0.0*1) / 4 = 0.75
    assert math.isclose(report.overall, 0.75)


def test_set_weight_zero_total_overall_zero():
    r = Rubric("a", score=lambda out, ctx=None: 1.0)
    rs = RubricSet([(r, 0.0)])
    report = rs.evaluate("x")
    assert report.overall == 0.0


def test_set_negative_weight_rejected():
    r = Rubric("a", score=lambda out, ctx=None: 1.0)
    with pytest.raises(ValueError):
        RubricSet([(r, -1.0)])


def test_set_empty_rejected():
    with pytest.raises(ValueError):
        RubricSet([])


def test_set_duplicate_names_rejected():
    r1 = Rubric("a", score=lambda out, ctx=None: 1.0)
    r2 = Rubric("a", score=lambda out, ctx=None: 0.0)
    with pytest.raises(ValueError):
        RubricSet([r1, r2])


def test_set_invalid_entry_type_rejected():
    with pytest.raises(TypeError):
        RubricSet(["not a rubric"])  # type: ignore[list-item]


def test_report_by_name():
    r1 = Rubric("a", score=lambda out, ctx=None: 0.8)
    r2 = Rubric("b", score=lambda out, ctx=None: 0.4)
    report = RubricSet([r1, r2]).evaluate("x")
    by = report.by_name()
    assert by["a"].value == 0.8
    assert by["b"].value == 0.4


def test_set_continues_after_one_rubric_fails():
    def boom(out, ctx=None):
        raise RuntimeError("oops")

    r1 = Rubric("bad", score=boom)
    r2 = Rubric("good", score=lambda out, ctx=None: 1.0)
    report = RubricSet([r1, r2]).evaluate("x")
    assert len(report.scores) == 2
    assert report.scores[0].value == 0.0
    assert "error" in (report.scores[0].reason or "").lower()
    assert report.scores[1].value == 1.0


def test_set_rubrics_property():
    r1 = Rubric("a", score=lambda out, ctx=None: 1.0)
    r2 = Rubric("b", score=lambda out, ctx=None: 1.0)
    rs = RubricSet([(r1, 0.3), (r2, 0.7)])
    assert [r.name for r in rs.rubrics] == ["a", "b"]
    assert rs.weights == [0.3, 0.7]
