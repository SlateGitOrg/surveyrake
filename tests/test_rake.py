from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from surveyrake.pipeline import weight_quarter
from surveyrake.rake import (
    EmptyMarginCategoryError,
    kish_n_eff,
    margin_targets,
    rake,
)

MARGINS = ["tier", "channel", "purpose", "property"]


def weighted_margin_shares(frame: pd.DataFrame, weights: np.ndarray, var: str) -> pd.Series:
    return pd.Series(weights, index=frame.index).groupby(frame[var]).sum() / weights.sum()


def test_raked_margins_equal_targets_and_weights_sum_to_population(
    responses, population_table, rake_reports
):
    for quarter, report in rake_reports.items():
        frame, _ = weight_quarter(responses, population_table, quarter, MARGINS, trim=None)
        targets = margin_targets(population_table, MARGINS, quarter)
        total = sum(targets["tier"].values())
        assert frame["weight"].sum() == pytest.approx(total, rel=1e-12)
        for var in MARGINS:
            got = weighted_margin_shares(frame, frame["weight"].to_numpy(), var)
            want = pd.Series(targets[var]) / total
            assert (got - want).abs().max() < 1e-8
        assert report.converged
        assert report.max_margin_error <= 1e-10


def test_trim_rake_caps_weights_and_keeps_margins(responses, population_table, rake_reports):
    for quarter, report in rake_reports.items():
        frame, _ = weight_quarter(responses, population_table, quarter, MARGINS, trim=5.0)
        w = frame["weight"].to_numpy()
        assert w.max() <= 5.0 * w.mean() * (1 + 1e-9)
        targets = margin_targets(population_table, MARGINS, quarter)
        total = sum(targets["tier"].values())
        for var in MARGINS:
            got = weighted_margin_shares(frame, w, var)
            assert (got - pd.Series(targets[var]) / total).abs().max() < 1e-6
        assert report.weight_max / report.weight_mean <= 5.0 * (1 + 1e-9)


def test_trimming_binds_when_cap_is_tight_and_margins_still_hold(responses, population_table):
    _, free = weight_quarter(responses, population_table, "2025-Q1", MARGINS, trim=None)
    cap = 2.5
    assert free.weight_max / free.weight_mean > cap  # the cap really cuts something
    trimmed, report = weight_quarter(responses, population_table, "2025-Q1", MARGINS, trim=cap)
    w = trimmed["weight"].to_numpy()
    assert report.converged
    assert report.trim_rounds >= 1
    assert report.n_trimmed > 0
    assert w.max() <= cap * w.mean() * (1 + 1e-9)
    targets = margin_targets(population_table, MARGINS, "2025-Q1")
    total = sum(targets["tier"].values())
    for var in MARGINS:
        got = weighted_margin_shares(trimmed, w, var)
        assert (got - pd.Series(targets[var]) / total).abs().max() < 1e-6
    # trimming shrinks weight variation, so the design effect cannot rise
    assert report.design_effect <= free.design_effect


def test_impossible_trim_is_reported_not_hidden(responses, population_table):
    # No set of weights can satisfy these margins with every weight at most 1.2x the mean.
    _, report = weight_quarter(responses, population_table, "2025-Q1", MARGINS, trim=1.2)
    assert not report.converged
    assert report.messages


def test_iteration_cap_reports_non_convergence():
    df = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "p"]})
    targets = {"a": {"x": 50.0, "y": 50.0}, "b": {"p": 60.0, "q": 40.0}}
    result = rake(df, targets, tol=1e-14, max_iter=2, trim=None)
    assert result.report.iterations == 2
    assert not result.report.converged
    assert "margins not met" in result.report.messages[0]


def test_two_margin_fixture_matches_closed_form():
    # 2x2 with margins that the sample's own cell shares cannot satisfy exactly: IPF result is the
    # classic Deming-Stephan fit, which for independent target margins is the product of margins.
    df = pd.DataFrame(
        {"a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "q"]},
    )
    targets = {"a": {"x": 30.0, "y": 70.0}, "b": {"p": 40.0, "q": 60.0}}
    w = rake(df, targets, trim=None).weights
    assert w == pytest.approx([12.0, 18.0, 28.0, 42.0], rel=1e-8)


def test_empty_category_error_names_variable_and_category(responses, population_table):
    q1 = responses[(responses["quarter"] == "2025-Q1") & (responses["tier"] != "gold")]
    targets = margin_targets(population_table, ["tier", "channel"], "2025-Q1")
    with pytest.raises(EmptyMarginCategoryError) as info:
        rake(q1, targets)
    err = info.value
    assert (err.variable, err.category) == ("tier", "gold")
    message = str(err)
    assert "'tier'" in message and "'gold'" in message
    assert "Collapse 'gold' into 'silver'" in message
    assert err.population > 0


def test_empty_property_category_is_caught_too(responses, population_table):
    q1 = responses[(responses["quarter"] == "2025-Q1") & (responses["property"] != "P03")]
    targets = margin_targets(population_table, ["property"], "2025-Q1")
    with pytest.raises(EmptyMarginCategoryError, match="'P03'"):
        rake(q1, targets)


def test_respondent_category_missing_from_population_is_rejected():
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="missing from the population"):
        rake(df, {"a": {"x": 5.0, "y": 5.0}})


def test_margins_must_agree_on_total_population():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    with pytest.raises(ValueError, match="disagree on the population total"):
        rake(df, {"a": {"x": 5.0, "y": 5.0}, "b": {"p": 6.0, "q": 6.0}})


def test_bad_inputs_are_rejected():
    df = pd.DataFrame({"a": ["x", "y"]})
    targets = {"a": {"x": 1.0, "y": 1.0}}
    with pytest.raises(ValueError, match="positive"):
        rake(df, targets, base_weights=np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="at least one margin"):
        rake(df, {})
    with pytest.raises(ValueError, match="no column"):
        rake(df, {"nope": {"x": 1.0}})


def test_kish_fixture_and_design_effect():
    assert kish_n_eff(np.array([1.0, 1.0, 2.0, 4.0])) == pytest.approx(64 / 22)
    assert round(kish_n_eff(np.array([1.0, 1.0, 2.0, 4.0])), 3) == 2.909


def test_report_describes_the_fit(rake_reports):
    r = rake_reports["2025-Q1"]
    assert r.converged and r.iterations < r.max_iter and r.tol == 1e-10
    assert set(r.margin_errors) == set(MARGINS)
    assert r.weight_min < r.weight_mean < r.weight_max
    assert r.design_effect == pytest.approx(r.n / r.n_eff)
    assert r.max_over_mean < 5.0


@st.composite
def small_surveys(draw):
    n = draw(st.integers(min_value=12, max_value=60))
    a = [0, 1, 2, *draw(st.lists(st.integers(0, 2), min_size=n - 3, max_size=n - 3))]
    b = [0, 1, *draw(st.lists(st.integers(0, 1), min_size=n - 2, max_size=n - 2))]
    w = draw(st.lists(st.floats(0.1, 10.0), min_size=n, max_size=n))
    df = pd.DataFrame({"a": [f"a{i}" for i in a], "b": [f"b{i}" for i in b]})
    return df, np.array(w)


@settings(max_examples=60, deadline=None)
@given(small_surveys(), st.floats(0.01, 100.0))
def test_scaling_base_weights_leaves_raked_weights_unchanged(survey, scale):
    df, w = survey
    targets = {"a": {"a0": 500.0, "a1": 300.0, "a2": 200.0}, "b": {"b0": 400.0, "b1": 600.0}}
    base = rake(df, targets, w, max_iter=40, trim=None).weights
    scaled = rake(df, targets, w * scale, max_iter=40, trim=None).weights
    assert scaled == pytest.approx(base, rel=1e-9)


@settings(max_examples=60, deadline=None)
@given(small_surveys())
def test_margins_that_already_hold_are_returned_after_one_iteration(survey):
    df, w = survey
    targets = {
        var: {str(k): float(v) for k, v in pd.Series(w).groupby(df[var].to_numpy()).sum().items()}
        for var in ("a", "b")
    }
    result = rake(df, targets, w, trim=None)
    assert result.report.iterations == 1
    assert result.report.converged
    assert result.weights == pytest.approx(w, rel=1e-12)
