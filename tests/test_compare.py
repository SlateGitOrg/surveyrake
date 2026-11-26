from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from surveyrake.compare import (
    Difference,
    difference,
    holm_adjust,
    holm_verdicts,
    minimum_detectable_change,
    property_vs_rest,
    verdict,
)
from surveyrake.nps import NpsEstimate, estimate_nps


def estimate(nps: float, se: float) -> NpsEstimate:
    return NpsEstimate(
        n=100, n_eff=100.0, design_effect=1.0, promoters=0.5, passives=0.3, detractors=0.2,
        nps=nps, se=se, ci_low=nps - 2 * se, ci_high=nps + 2 * se,
    )  # fmt: skip


def test_z_test_matches_hand_calculation():
    d = difference(estimate(10.0, 3.0), estimate(25.0, 4.0))
    assert d.diff == pytest.approx(15.0)
    assert d.se == pytest.approx(5.0)  # sqrt(3^2 + 4^2)
    assert d.z == pytest.approx(3.0)
    assert d.p_value == pytest.approx(2 * (1 - norm.cdf(3.0)))
    assert d.ci_low == pytest.approx(15 - 1.959964 * 5, abs=1e-5)
    assert d.mde == pytest.approx((1.959964 + 0.841621) * 5, abs=1e-5)


def test_minimum_detectable_change_formula_and_power():
    mde = minimum_detectable_change(5.0)
    assert mde == pytest.approx(14.008, abs=1e-3)
    # a true change equal to the MDE is rejected with probability ~ 80%
    power = norm.cdf(mde / 5.0 - norm.ppf(0.975)) + norm.cdf(-mde / 5.0 - norm.ppf(0.975))
    assert power == pytest.approx(0.80, abs=1e-3)
    assert minimum_detectable_change(5.0, alpha=0.10) < mde
    assert minimum_detectable_change(5.0, power=0.9) > mde


def make_difference(p_value: float, mde: float) -> Difference:
    return Difference(diff=0.0, se=1.0, z=0.0, p_value=p_value, ci_low=-1, ci_high=1, mde=mde)


@pytest.mark.parametrize(
    ("p_value", "mde", "expected"),
    [
        (0.001, 40.0, "signal"),  # significant wins even when power is poor
        (0.30, 12.0, "noise"),  # a 20-point move would have shown, none did
        (0.30, 20.0, "noise"),  # boundary: MDE equal to the practical change
        (0.30, 33.0, "cannot detect"),
        (0.05, 10.0, "noise"),  # p must be strictly below alpha
    ],
)
def test_verdict_rules(p_value, mde, expected):
    assert verdict(make_difference(p_value, mde)) == expected


def test_verdict_threshold_is_configurable():
    d = make_difference(0.4, 15.0)
    assert verdict(d, practical_change=20.0) == "noise"
    assert verdict(d, practical_change=10.0) == "cannot detect"


def test_property_vs_rest_uses_independent_samples(weighted):
    q2 = weighted[weighted["quarter"] == "2025-Q2"]
    w = q2["weight"].to_numpy()
    table = property_vs_rest(q2, w).set_index("property")
    assert len(table) == 12
    p05 = q2["property"].to_numpy() == "P05"
    own, rest = estimate_nps(q2["score"][p05], w[p05]), estimate_nps(q2["score"][~p05], w[~p05])
    assert table.loc["P05", "diff"] == pytest.approx(own.nps - rest.nps)
    assert table.loc["P05", "se"] == pytest.approx(np.hypot(own.se, rest.se))
    # the comparison group excludes the property: a property compared with a group that contained
    # it would shrink its own difference
    group_nps = estimate_nps(q2["score"], w).nps
    assert table.loc["P05", "rest_nps"] != pytest.approx(group_nps)


def test_holm_adjustment_matches_hand_calculation():
    # sorted p: 0.01 x3 = 0.03, 0.03 x2 = 0.06, 0.04 x1 -> raised to 0.06 (monotone)
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # capped at 1


def test_holm_verdicts_demote_a_marginal_signal():
    diffs = [make_difference(p, 12.0) for p in (0.001, 0.02, 0.04, 0.5)]
    assert [verdict(d) for d in diffs] == ["signal", "signal", "signal", "noise"]
    assert holm_verdicts(diffs) == ["signal", "noise", "noise", "noise"]
