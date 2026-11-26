"""End-to-end checks against the synthetic truth: accuracy, coverage, variance, verdicts."""

from __future__ import annotations

import numpy as np
import pytest

from surveyrake.compare import difference, holm_verdicts, verdict
from surveyrake.nps import bootstrap_nps, estimate_nps
from surveyrake.pipeline import weight_quarter, weight_quarters
from surveyrake.rake import margin_targets
from surveyrake.synth import draw_responses

MARGINS = ["tier", "channel", "purpose", "property"]


def test_pooled_raked_group_nps_is_within_three_points_while_raw_is_off_by_eight(
    population, weighted
):
    truth = population.true_nps(None).iloc[0]
    raked = estimate_nps(weighted["score"], weighted["weight"]).nps
    raw = estimate_nps(weighted["score"]).nps
    assert abs(raked - truth) <= 3.0
    assert raw - truth >= 8.0


def test_raking_also_fixes_each_quarter_and_most_properties(population, weighted):
    for quarter in ("2025-Q1", "2025-Q2"):
        frame = weighted[weighted["quarter"] == quarter]
        truth = population.true_nps(None, quarter).iloc[0]
        assert abs(estimate_nps(frame["score"], frame["weight"]).nps - truth) < 3.5
        true_by_property = population.true_nps("property", quarter)
        z_scores = []
        for prop, idx in frame.groupby("property").indices.items():
            est = estimate_nps(frame["score"].to_numpy()[idx], frame["weight"].to_numpy()[idx])
            z_scores.append((est.nps - true_by_property[prop]) / est.se)
        # standardised errors against the truth look like noise, not bias
        assert abs(np.mean(z_scores)) < 0.8
        assert np.mean(np.abs(z_scores) < 1.96) >= 0.75


def test_ci_coverage_over_200_simulated_surveys(population):
    truth = population.true_nps(None, "2025-Q1").iloc[0]
    table = population.margin_table()
    raked_hits = raw_hits = 0
    estimates, analytic_ses = [], []
    runs = 200
    for k in range(runs):
        responses = draw_responses(population, seed=1_000 + k)
        frame, _ = weight_quarter(responses, table, "2025-Q1", MARGINS)
        raked = estimate_nps(frame["score"], frame["weight"])
        raw = estimate_nps(frame["score"])
        estimates.append(raked.nps)
        analytic_ses.append(raked.se)
        raked_hits += raked.ci_low <= truth <= raked.ci_high
        raw_hits += raw.ci_low <= truth <= raw.ci_high
    assert 0.90 <= raked_hits / runs <= 0.995
    assert raw_hits / runs < 0.50
    # the trinomial SE is slightly conservative: the real spread of the estimate across surveys
    # is somewhat below it (finite population and the raking calibration both remove variance)
    spread_ratio = np.std(estimates, ddof=1) / np.mean(analytic_ses)
    assert 0.80 <= spread_ratio <= 1.00


def test_bootstrap_se_agrees_with_trinomial_se(weighted, population_table):
    q1 = weighted[weighted["quarter"] == "2025-Q1"]
    analytic = estimate_nps(q1["score"], q1["weight"]).se
    targets = margin_targets(population_table, MARGINS, "2025-Q1")
    boot = bootstrap_nps(q1.drop(columns="weight"), targets, by=None, reps=500, seed=11)
    assert 0.7 <= boot["se"].iloc[0] / analytic <= 1.3


def _change(weighted, prop: str):
    parts = {}
    for q in ("2025-Q1", "2025-Q2"):
        frame = weighted[(weighted["quarter"] == q) & (weighted["property"] == prop)]
        parts[q] = estimate_nps(frame["score"], frame["weight"])
    return difference(parts["2025-Q1"], parts["2025-Q2"])


def test_p07_drop_is_a_signal(weighted):
    d = _change(weighted, "P07")
    assert verdict(d) == "signal"
    assert d.p_value < 0.05
    assert d.diff == pytest.approx(-20.0, abs=8.0)  # the drop is recovered, not just flagged


def test_p03_change_is_not_flagged_and_its_mde_is_large(weighted):
    d = _change(weighted, "P03")
    assert verdict(d) != "signal"
    assert d.mde > 15.0
    assert verdict(d) == "cannot detect"


def test_p07_is_flagged_and_p03_is_not_whatever_the_response_seed(population):
    table = population.margin_table()
    for seed in (3, 4, 5, 6, 7, 8):
        frame, _ = weight_quarters(draw_responses(population, seed), table)
        assert verdict(_change(frame, "P07")) == "signal"
        assert verdict(_change(frame, "P03")) != "signal"


def test_false_flag_rate_per_unchanged_property_is_near_the_nominal_five_percent(population):
    table = population.margin_table()
    null_props = [f"P{i:02d}" for i in range(1, 13) if f"P{i:02d}" != "P07"]
    flags = total = 0
    holm_flags = 0
    seeds = 60
    for seed in range(100, 100 + seeds):
        frame, _ = weight_quarters(draw_responses(population, seed), table)
        diffs = [_change(frame, p) for p in null_props]
        flags += sum(verdict(d) == "signal" for d in diffs)
        holm_flags += sum(v == "signal" for v in holm_verdicts(diffs))
        total += len(diffs)
    rate = flags / total
    # per-property tests, no correction: about 5% (finite-population truth moves a little)
    assert 0.01 <= rate <= 0.12
    # Holm controls the family-wise rate: far fewer false flags than the uncorrected tests
    assert holm_flags < flags
    assert holm_flags / seeds <= 0.20


def test_mde_shrinks_as_effective_sample_grows(weighted):
    mdes = {p: _change(weighted, p).mde for p in ("P03", "P05", "P07")}
    assert mdes["P03"] > mdes["P05"] > mdes["P07"]
