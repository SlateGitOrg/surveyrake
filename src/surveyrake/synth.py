"""Seeded synthetic hotel group: every stay, every guest's latent score, then who responds.

The generator is a small, readable model with the properties the analysis has to cope with:

* Stays carry loyalty tier, booking channel, trip purpose and property. Composition differs by
  property, so the property margin matters for raking.
* Each guest has a latent 0-10 score. Promoter (9-10), passive (7-8) and detractor (0-6) shares
  come from a multinomial logit in the guest's segment, so gold members have more promoters and
  OTA leisure guests have more detractors (an interaction the raking model does not know about,
  which is fine: the response mechanism is what raking has to match, not the score model).
* Whether a guest responds is Bernoulli with a propensity that is log-linear in main effects only:
  tier (gold 0.30, silver 0.15, none 0.05) times 0.7 for OTA bookings. Raking on tier and channel
  therefore recovers the exact inverse-propensity weights, up to a constant.
* Property P07 is large and its true NPS falls by 20 points from 2025-Q1 to 2025-Q2. Property P03
  is the smallest and does not change.

The true population NPS (computed from every stay, not just respondents) lives on
:class:`Population`. Only tests and the synthetic scorecard in ``surveyrake diagnostics`` read it;
the estimation modules never do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

QUARTERS: tuple[str, str] = ("2025-Q1", "2025-Q2")
TIERS: tuple[str, ...] = ("none", "silver", "gold")
CHANNELS: tuple[str, ...] = ("direct", "ota", "corporate")
PURPOSES: tuple[str, ...] = ("business", "leisure")

TIER_SHARE = np.array([0.60, 0.25, 0.15])
CHANNEL_SHARE = np.array([0.40, 0.35, 0.25])
RESPONSE_RATE_BY_TIER = {"none": 0.05, "silver": 0.15, "gold": 0.30}
OTA_RESPONSE_MULTIPLIER = 0.7

DEFAULT_STAYS_PER_QUARTER = 48_000
P07_DROP_POINTS = 20.0

# Multinomial logit for (promoter, detractor) against passive. Index orders follow TIERS,
# CHANNELS and PURPOSES.
_BASE_PROMOTER, _BASE_DETRACTOR = 0.30, -0.55
_TIER_PROMOTER = np.array([0.0, 0.20, 0.66])
_TIER_DETRACTOR = np.array([0.0, -0.14, -0.42])
_CHANNEL_PROMOTER = np.array([0.15, -0.35, 0.10])
_CHANNEL_DETRACTOR = np.array([-0.10, 0.35, -0.05])
_PURPOSE_PROMOTER = np.array([0.10, 0.0])
_PURPOSE_DETRACTOR = np.array([-0.05, 0.10])
_OTA_LEISURE_PROMOTER, _OTA_LEISURE_DETRACTOR = -0.25, 0.55

# Score inside each NPS class.
_PROMOTER_TEN = 0.55
_PASSIVE_EIGHT = 0.50
_DETRACTOR_PMF = np.array([3, 1, 1, 2, 3, 5, 7], dtype=float)
_DETRACTOR_PMF /= _DETRACTOR_PMF.sum()


@dataclass(frozen=True)
class PropertySpec:
    """One hotel: its share of stays and how its guest mix and service quality differ."""

    code: str
    stay_share: float
    loyalty_tilt: float  # >0: more silver and gold members than the group average
    ota_tilt: float  # >0: more OTA bookings, fewer direct
    business_share: float
    quality: float  # shifts promoters up and detractors down (logit scale)


PROPERTIES: tuple[PropertySpec, ...] = (
    PropertySpec("P01", 0.060, -0.10, 0.10, 0.45, -0.20),
    PropertySpec("P02", 0.080, 0.15, -0.05, 0.55, 0.30),
    PropertySpec("P03", 0.028, -0.15, 0.15, 0.30, -0.35),
    PropertySpec("P04", 0.070, 0.05, 0.00, 0.40, 0.10),
    PropertySpec("P05", 0.090, 0.20, -0.10, 0.60, 0.45),
    PropertySpec("P06", 0.075, -0.15, 0.25, 0.35, -0.45),
    PropertySpec("P07", 0.1875, 0.00, 0.00, 0.50, 0.15),
    PropertySpec("P08", 0.080, 0.10, -0.15, 0.65, 0.20),
    PropertySpec("P09", 0.085, -0.05, 0.15, 0.30, -0.10),
    PropertySpec("P10", 0.0925, 0.25, 0.00, 0.45, 0.50),
    PropertySpec("P11", 0.060, -0.20, 0.30, 0.25, -0.55),
    PropertySpec("P12", 0.092, 0.05, -0.05, 0.50, 0.05),
)

PROPERTY_CODES: tuple[str, ...] = tuple(p.code for p in PROPERTIES)


def _tier_probs(spec: PropertySpec) -> np.ndarray:
    p = TIER_SHARE * np.array([1.0, 1.0 + spec.loyalty_tilt / 2, 1.0 + spec.loyalty_tilt])
    return p / p.sum()


def _channel_probs(spec: PropertySpec) -> np.ndarray:
    p = CHANNEL_SHARE * np.array([1.0 - spec.ota_tilt / 2, 1.0 + spec.ota_tilt, 1.0])
    return p / p.sum()


def _purpose_probs(spec: PropertySpec) -> np.ndarray:
    return np.array([spec.business_share, 1.0 - spec.business_share])


def outcome_probs(
    tier: np.ndarray, channel: np.ndarray, purpose: np.ndarray, quality: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (promoter, passive, detractor) probabilities for integer-coded guest segments."""
    ota_leisure = ((channel == 1) & (purpose == 1)).astype(float)
    lp = (
        _BASE_PROMOTER
        + _TIER_PROMOTER[tier]
        + _CHANNEL_PROMOTER[channel]
        + _PURPOSE_PROMOTER[purpose]
        + _OTA_LEISURE_PROMOTER * ota_leisure
        + quality
    )
    ld = (
        _BASE_DETRACTOR
        + _TIER_DETRACTOR[tier]
        + _CHANNEL_DETRACTOR[channel]
        + _PURPOSE_DETRACTOR[purpose]
        + _OTA_LEISURE_DETRACTOR * ota_leisure
        - quality
    )
    ep, ed = np.exp(lp), np.exp(ld)
    total = 1.0 + ep + ed
    return ep / total, 1.0 / total, ed / total


def expected_nps(spec: PropertySpec, quality_shift: float = 0.0) -> float:
    """Expected population NPS (as a fraction) of a property from its composition."""
    tp, cp, up = _tier_probs(spec), _channel_probs(spec), _purpose_probs(spec)
    t, c, u = np.meshgrid(np.arange(3), np.arange(3), np.arange(2), indexing="ij")
    weight = tp[t] * cp[c] * up[u]
    prom, _, det = outcome_probs(t, c, u, spec.quality + quality_shift)
    return float(np.sum(weight * (prom - det)))


def _p07_quality_shift(spec: PropertySpec) -> float:
    """Quality shift that lowers a property's expected NPS by exactly ``P07_DROP_POINTS``."""
    target = expected_nps(spec) - P07_DROP_POINTS / 100.0
    return brentq(lambda s: expected_nps(spec, s) - target, -4.0, 0.0)


def response_propensity(tier: np.ndarray, channel: np.ndarray) -> np.ndarray:
    """Probability a stay answers the survey: log-linear in tier and OTA main effects only."""
    by_tier = np.array([RESPONSE_RATE_BY_TIER[t] for t in TIERS])
    return by_tier[tier] * np.where(channel == 1, OTA_RESPONSE_MULTIPLIER, 1.0)


def nps_points(scores: np.ndarray | pd.Series) -> float:
    """Unweighted NPS in points (promoters 9-10 minus detractors 0-6, as a percentage)."""
    s = np.asarray(scores)
    return float(((s >= 9).mean() - (s <= 6).mean()) * 100.0)


@dataclass(frozen=True)
class Population:
    """Every stay in both quarters, with its latent guest score."""

    stays: pd.DataFrame

    def margin_table(self) -> pd.DataFrame:
        """Stay counts per quarter x property x tier x channel x purpose cell (the PMS extract)."""
        keys = ["quarter", "property", "tier", "channel", "purpose"]
        out = self.stays.groupby(keys, observed=True).size().rename("stays").reset_index()
        return out.astype({k: str for k in keys})

    def true_nps(self, by: str | None = "property", quarter: str | None = None) -> pd.Series:
        """True finite-population NPS in points, from every stay's latent score."""
        df = self.stays if quarter is None else self.stays[self.stays["quarter"] == quarter]
        if by is None:
            return pd.Series({"group": nps_points(df["score"])})
        return df.groupby(by, observed=True)["score"].apply(nps_points)

    def truth_table(self) -> pd.DataFrame:
        """True NPS per property and quarter, plus the group row, for the synthetic scorecard."""
        rows = []
        for quarter in (*QUARTERS, "pooled"):
            q = None if quarter == "pooled" else quarter
            for prop, value in self.true_nps("property", q).items():
                rows.append({"quarter": quarter, "property": prop, "true_nps": value})
            rows.append(
                {
                    "quarter": quarter,
                    "property": "group",
                    "true_nps": self.true_nps(None, q).iloc[0],
                }
            )
        return pd.DataFrame(rows)


def generate_population(
    seed: int = 2025, stays_per_quarter: int = DEFAULT_STAYS_PER_QUARTER
) -> Population:
    """Generate both quarters of stays. Deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    p07 = next(p for p in PROPERTIES if p.code == "P07")
    p07_shift = _p07_quality_shift(p07)

    frames = []
    for quarter in QUARTERS:
        for spec in PROPERTIES:
            n = int(round(spec.stay_share * stays_per_quarter * rng.uniform(0.97, 1.03)))
            tier = rng.choice(3, size=n, p=_tier_probs(spec))
            channel = rng.choice(3, size=n, p=_channel_probs(spec))
            purpose = rng.choice(2, size=n, p=_purpose_probs(spec))
            shift = p07_shift if (spec is p07 and quarter == QUARTERS[1]) else 0.0
            prom, _, det = outcome_probs(tier, channel, purpose, spec.quality + shift)
            u = rng.random(n)
            promoter = u < prom
            detractor = u >= 1.0 - det
            score = np.where(
                promoter,
                np.where(rng.random(n) < _PROMOTER_TEN, 10, 9),
                np.where(rng.random(n) < _PASSIVE_EIGHT, 8, 7),
            )
            score = np.where(detractor, rng.choice(7, size=n, p=_DETRACTOR_PMF), score)
            frames.append(
                pd.DataFrame(
                    {
                        "quarter": quarter,
                        "property": spec.code,
                        "tier": np.array(TIERS)[tier],
                        "channel": np.array(CHANNELS)[channel],
                        "purpose": np.array(PURPOSES)[purpose],
                        "score": score.astype(int),
                    }
                )
            )
    stays = pd.concat(frames, ignore_index=True)
    for col in ("quarter", "property", "tier", "channel", "purpose"):
        stays[col] = stays[col].astype("category")
    return Population(stays=stays)


def draw_responses(population: Population, seed: int = 1) -> pd.DataFrame:
    """Draw who answers: independent Bernoulli per stay with the log-linear propensity."""
    rng = np.random.default_rng([seed, 1])
    stays = population.stays
    tier = pd.Categorical(stays["tier"], categories=TIERS).codes
    channel = pd.Categorical(stays["channel"], categories=CHANNELS).codes
    answered = rng.random(len(stays)) < response_propensity(tier, channel)
    cols = ["quarter", "property", "tier", "channel", "purpose", "score"]
    out = stays.loc[answered, cols].reset_index(drop=True)
    for col in cols[:-1]:
        out[col] = out[col].astype(str)
    return out
