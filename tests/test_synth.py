from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surveyrake.synth import (
    CHANNELS,
    PROPERTIES,
    QUARTERS,
    TIERS,
    draw_responses,
    expected_nps,
    generate_population,
    nps_points,
    outcome_probs,
    response_propensity,
)


def test_generation_is_deterministic_in_its_seeds():
    a = generate_population(7, stays_per_quarter=4_000)
    b = generate_population(7, stays_per_quarter=4_000)
    c = generate_population(8, stays_per_quarter=4_000)
    pd.testing.assert_frame_equal(a.stays, b.stays)
    assert not a.stays["score"].equals(c.stays["score"])
    pd.testing.assert_frame_equal(draw_responses(a, 3), draw_responses(b, 3))
    assert not draw_responses(a, 3).equals(draw_responses(a, 4))


def test_population_size_and_mix(population):
    stays = population.stays
    for quarter in QUARTERS:
        assert abs((stays["quarter"] == quarter).sum() - 48_000) < 48_000 * 0.02
    tier = stays["tier"].value_counts(normalize=True)
    assert tier["none"] == pytest.approx(0.60, abs=0.015)
    assert tier["silver"] == pytest.approx(0.25, abs=0.015)
    assert tier["gold"] == pytest.approx(0.15, abs=0.015)
    assert set(stays["channel"].unique()) == set(CHANNELS)
    assert stays["score"].between(0, 10).all()


def test_about_4800_responses_per_quarter(responses):
    counts = responses["quarter"].value_counts()
    assert all(4_500 < counts[q] < 5_100 for q in QUARTERS)


def test_property_sizes_match_the_design(responses):
    q1 = responses[responses["quarter"] == "2025-Q1"]["property"].value_counts()
    assert q1.idxmin() == "P03" and 95 < q1["P03"] < 150
    assert q1.idxmax() == "P07" and 800 < q1["P07"] < 1_000


def test_response_propensity_is_log_linear_in_main_effects():
    tier, channel = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    p = response_propensity(tier.ravel(), channel.ravel()).reshape(3, 3)
    assert p[:, 0] == pytest.approx([0.05, 0.15, 0.30])
    assert p[:, 1] == pytest.approx(0.7 * np.array([0.05, 0.15, 0.30]))
    # no tier x channel interaction on the log scale
    interaction = (
        np.log(p) - np.log(p).mean(0) - np.log(p).mean(1, keepdims=True) + np.log(p).mean()
    )
    assert np.abs(interaction[:, [0, 2]]).max() < 1e-12


def test_observed_response_rates_follow_the_mechanism(population, responses):
    stays = population.stays
    keys = ["tier", "channel"]
    n_stay = stays.groupby(keys, observed=True).size()
    n_resp = responses.groupby(keys).size()
    rate = (n_resp / n_stay).dropna()
    assert rate[("gold", "direct")] == pytest.approx(0.30, rel=0.08)
    assert rate[("silver", "corporate")] == pytest.approx(0.15, rel=0.08)
    assert rate[("none", "direct")] == pytest.approx(0.05, rel=0.10)
    assert rate[("gold", "ota")] / rate[("gold", "direct")] == pytest.approx(0.7, rel=0.12)
    assert rate[("none", "ota")] / rate[("none", "direct")] == pytest.approx(0.7, rel=0.15)


def test_segment_scores_follow_the_design(population):
    stays = population.stays
    by_tier = stays.groupby("tier", observed=True)["score"].apply(nps_points)
    assert by_tier["gold"] > by_tier["silver"] > by_tier["none"]
    ota_leisure = (stays["channel"] == "ota") & (stays["purpose"] == "leisure")
    detractors = stays["score"] <= 6
    assert detractors[ota_leisure].mean() > detractors[~ota_leisure].mean() + 0.05


def test_outcome_probabilities_are_a_distribution():
    t, c, u = np.meshgrid(np.arange(3), np.arange(3), np.arange(2), indexing="ij")
    p, m, d = outcome_probs(t, c, u, 0.3)
    assert (p + m + d == pytest.approx(1.0)) and (p > 0).all() and (d > 0).all()


def test_raw_nps_overstates_the_truth_by_about_twelve_points(population, responses):
    truth = population.true_nps(None).iloc[0]
    bias = nps_points(responses["score"]) - truth
    assert 10 < bias < 15


def test_p07_truly_drops_twenty_points_and_p03_does_not_move(population):
    q1 = population.true_nps("property", "2025-Q1")
    q2 = population.true_nps("property", "2025-Q2")
    assert q2["P07"] - q1["P07"] == pytest.approx(-20.0, abs=2.5)
    assert abs(q2["P03"] - q1["P03"]) < 6.0  # sampling noise of a 1,200-stay property
    others = [p.code for p in PROPERTIES if p.code not in ("P07", "P03")]
    assert (q2[others] - q1[others]).abs().max() < 8.0


def test_calibrated_shift_gives_the_exact_expected_drop():
    from surveyrake.synth import _p07_quality_shift

    p07 = next(p for p in PROPERTIES if p.code == "P07")
    shift = _p07_quality_shift(p07)
    assert expected_nps(p07) - expected_nps(p07, shift) == pytest.approx(0.20, abs=1e-9)


def test_margin_table_counts_every_stay_once(population):
    table = population.margin_table()
    assert table["stays"].sum() == len(population.stays)
    assert set(table.columns) == {"quarter", "property", "tier", "channel", "purpose", "stays"}
    assert set(table["tier"]) == set(TIERS)


def test_truth_table_includes_group_and_pooled_rows(population):
    t = population.truth_table()
    pooled = t[(t["quarter"] == "pooled") & (t["property"] == "group")]["true_nps"].iloc[0]
    assert pooled == pytest.approx(population.true_nps(None).iloc[0])
    assert len(t) == 3 * 13
