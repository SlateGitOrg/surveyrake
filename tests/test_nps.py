from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surveyrake.nps import (
    Z95,
    bootstrap_nps,
    classify,
    design_effect,
    estimate_by,
    estimate_nps,
    trinomial_variance,
)
from surveyrake.rake import margin_targets


def test_classify_uses_standard_nps_bands():
    scores = np.arange(0, 11)
    assert classify(scores).tolist() == [-1] * 7 + [0, 0, 1, 1]


def test_classify_rejects_out_of_scale_scores():
    with pytest.raises(ValueError, match="0-10"):
        classify([3, 11])


def test_trinomial_fixture_matches_hand_calculation():
    # p_P = 0.5, p_D = 0.2, n = 100: NPS 30, variance (0.7 - 0.09) / 100, SE 7.810 points
    var = trinomial_variance(0.5, 0.2, 100)
    assert var == pytest.approx(0.0061)
    assert round(float(np.sqrt(var)) * 100, 3) == 7.810


def test_estimate_matches_weighted_hand_calculation():
    # x = [+1, +1, 0, 0, -1], weights [3, 1, 2, 1, 1]: p_P = 4/8, p_D = 1/8
    scores = np.array([10, 9, 8, 7, 3])
    w = np.array([3.0, 1.0, 2.0, 1.0, 1.0])
    e = estimate_nps(scores, w)
    assert (e.promoters, e.passives, e.detractors) == pytest.approx((0.5, 0.375, 0.125))
    assert e.nps == pytest.approx(37.5)
    assert e.n_eff == pytest.approx(64 / 16)  # (sum w)^2 / sum w^2 = 64 / 16
    assert e.design_effect == pytest.approx(5 / 4)
    expected_se = np.sqrt((0.625 - 0.375**2) / 4.0) * 100
    assert e.se == pytest.approx(expected_se)
    assert e.ci_low == pytest.approx(37.5 - Z95 * expected_se)
    assert e.ci_high == pytest.approx(37.5 + Z95 * expected_se)


def test_unweighted_estimate_has_n_eff_equal_n_and_unit_design_effect():
    e = estimate_nps(np.array([10, 9, 9, 8, 7, 6, 2, 10]))
    assert e.n == 8 and e.n_eff == pytest.approx(8.0) and e.design_effect == pytest.approx(1.0)
    assert e.nps == pytest.approx((4 - 2) / 8 * 100)


def test_shares_sum_to_one_and_scale_with_weights_invariantly():
    rng = np.random.default_rng(3)
    scores = rng.integers(0, 11, size=200)
    w = rng.uniform(0.5, 3.0, size=200)
    a, b = estimate_nps(scores, w), estimate_nps(scores, 7.5 * w)
    assert a.promoters + a.passives + a.detractors == pytest.approx(1.0)
    assert (a.nps, a.se, a.n_eff) == pytest.approx((b.nps, b.se, b.n_eff))


def test_design_effect_grows_with_weight_variation():
    assert design_effect(np.ones(10)) == pytest.approx(1.0)
    assert design_effect(np.array([1.0, 1.0, 2.0, 4.0])) == pytest.approx(4 / 2.909, abs=1e-3)


def test_estimate_rejects_bad_weights_and_empty_input():
    with pytest.raises(ValueError, match="zero responses"):
        estimate_nps(np.array([], dtype=int))
    with pytest.raises(ValueError, match="non-negative"):
        estimate_nps(np.array([9, 3]), np.array([1.0, -1.0]))


def test_estimate_by_splits_levels(weighted):
    q1 = weighted[weighted["quarter"] == "2025-Q1"]
    by_property = estimate_by(q1, q1["weight"].to_numpy(), "property")
    assert len(by_property) == 12
    assert sum(e.n for e in by_property.values()) == len(q1)
    assert "group" in estimate_by(q1, None, None)


def _q1(responses, population_table):
    q1 = responses[responses["quarter"] == "2025-Q1"].reset_index(drop=True)
    targets = margin_targets(
        population_table, ["tier", "channel", "purpose", "property"], "2025-Q1"
    )
    return q1, targets


def test_bootstrap_is_seeded_and_re_rakes(responses, population_table):
    q1, targets = _q1(responses, population_table)
    a = bootstrap_nps(q1, targets, by=None, reps=40, seed=5)
    b = bootstrap_nps(q1, targets, by=None, reps=40, seed=5)
    c = bootstrap_nps(q1, targets, by=None, reps=40, seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert a["se"].iloc[0] != c["se"].iloc[0]
    assert a["ci_low"].iloc[0] < a["ci_high"].iloc[0]
    assert a["reps"].iloc[0] == 40


def test_bootstrap_by_property_has_one_row_per_property(responses, population_table):
    q1, targets = _q1(responses, population_table)
    out = bootstrap_nps(q1, targets, by="property", reps=30, seed=1)
    assert len(out) == 12 and (out["se"] > 0).all()
    # the smallest property is the noisiest
    assert out["se"].idxmax() in {"P03", "P11", "P01"}


def test_bootstrap_skips_replicates_that_lose_a_category():
    df = pd.DataFrame(
        {
            "a": ["x"] * 29 + ["y"],
            "score": [9] * 15 + [3] * 14 + [9],
        }
    )
    targets = {"a": {"x": 90.0, "y": 10.0}}
    out = bootstrap_nps(df, targets, by=None, reps=200, seed=2, strata=None, trim=None)
    # P(replicate drops the single "y" respondent) = (29/30)^30 ~ 0.36, so some are skipped
    assert 60 < out["reps"].iloc[0] < 180


def test_bootstrap_needs_enough_replicates(responses, population_table):
    q1, targets = _q1(responses, population_table)
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_nps(q1, targets, reps=1)
