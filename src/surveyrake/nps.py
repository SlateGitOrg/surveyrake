"""Weighted Net Promoter Score: trinomial variance, Kish design effect, re-raking bootstrap.

Each respondent contributes +1 (promoter, 9-10), 0 (passive, 7-8) or -1 (detractor, 0-6), so
NPS = p_P - p_D. With weights, the shares are weighted shares and the variance of the estimate is
the trinomial variance divided by the Kish effective sample size:

    Var(NPS) = (p_P + p_D - (p_P - p_D)^2) / n_eff,   n_eff = (sum w)^2 / sum w^2

Everything here reports NPS in points (a percentage-point difference, -100 to 100).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .rake import (
    DEFAULT_TRIM,
    Targets,
    kish_n_eff,
    prepare_margins,
    rake_prepared,
)

PROMOTER_MIN = 9
DETRACTOR_MAX = 6
Z95 = float(norm.ppf(0.975))
DEFAULT_BOOTSTRAP_REPS = 500


def classify(scores: np.ndarray | pd.Series) -> np.ndarray:
    """+1 for promoters (9-10), -1 for detractors (0-6), 0 for passives (7-8)."""
    s = np.asarray(scores)
    if s.size and (s.min() < 0 or s.max() > 10):
        raise ValueError("scores must be on the 0-10 scale")
    return np.where(s >= PROMOTER_MIN, 1, np.where(s <= DETRACTOR_MAX, -1, 0))


def trinomial_variance(p_promoter: float, p_detractor: float, n_eff: float) -> float:
    """Variance of NPS as a fraction squared: (p_P + p_D - (p_P - p_D)^2) / n_eff."""
    return (p_promoter + p_detractor - (p_promoter - p_detractor) ** 2) / n_eff


def design_effect(weights: np.ndarray | pd.Series) -> float:
    """Kish design effect from unequal weights: n / n_eff = n * sum w^2 / (sum w)^2."""
    w = np.asarray(weights, dtype=float)
    return float(len(w) / kish_n_eff(w))


@dataclass(frozen=True)
class NpsEstimate:
    """A weighted NPS with its sampling uncertainty. NPS, SE and CI are in points."""

    n: int
    n_eff: float
    design_effect: float
    promoters: float
    passives: float
    detractors: float
    nps: float
    se: float
    ci_low: float
    ci_high: float

    @property
    def variance_per_effective_response(self) -> float:
        """Trinomial variance of a single effective response, as a fraction squared."""
        return self.promoters + self.detractors - (self.promoters - self.detractors) ** 2


def estimate_nps(
    scores: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
    z: float = Z95,
) -> NpsEstimate:
    """Weighted NPS, trinomial SE with Kish n_eff, and a normal-approximation CI.

    With ``weights=None`` this is the raw (unweighted) NPS and n_eff equals n.
    """
    x = classify(scores)
    n = len(x)
    if n == 0:
        raise ValueError("cannot estimate NPS from zero responses")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, one per response, with a positive sum")
    total = w.sum()
    p_pro = float(w[x == 1].sum() / total)
    p_det = float(w[x == -1].sum() / total)
    p_pas = 1.0 - p_pro - p_det
    n_eff = kish_n_eff(w)
    se = float(np.sqrt(trinomial_variance(p_pro, p_det, n_eff))) * 100.0
    nps = (p_pro - p_det) * 100.0
    return NpsEstimate(
        n=n,
        n_eff=n_eff,
        design_effect=n / n_eff,
        promoters=p_pro,
        passives=p_pas,
        detractors=p_det,
        nps=nps,
        se=se,
        ci_low=nps - z * se,
        ci_high=nps + z * se,
    )


def estimate_by(
    data: pd.DataFrame, weights: np.ndarray | pd.Series | None, by: str | None
) -> dict[str, NpsEstimate]:
    """Estimates per level of ``by`` (or one ``group`` entry when ``by`` is None)."""
    w = None if weights is None else np.asarray(weights, dtype=float)
    if by is None:
        return {"group": estimate_nps(data["score"], w)}
    out = {}
    for level, idx in data.groupby(by, sort=True).indices.items():
        out[str(level)] = estimate_nps(data["score"].to_numpy()[idx], None if w is None else w[idx])
    return out


def bootstrap_nps(
    responses: pd.DataFrame,
    targets: Targets,
    *,
    by: str | None = "property",
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = 0,
    strata: str | None = "property",
    trim: float | None = DEFAULT_TRIM,
    level: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap SE and percentile CI of NPS in which every replicate is re-raked.

    Respondents are resampled with replacement within ``strata`` (so every stratum keeps its
    size), then raked to ``targets`` again, so the spread includes the weighting step's own
    variability. Replicates in which a margin category has no respondents are skipped and counted.
    Returns one row per level of ``by`` with columns ``se``, ``ci_low``, ``ci_high``, ``reps``.
    """
    if reps < 2:
        raise ValueError("bootstrap needs at least 2 replicates")
    margins = prepare_margins(responses, targets)
    x = classify(responses["score"])
    if by is None:
        levels, group = np.array(["group"]), np.zeros(len(x), dtype=np.int64)
    else:
        levels, group = np.unique(responses[by].astype(str).to_numpy(), return_inverse=True)
    if strata is None:
        blocks = [np.arange(len(x))]
    else:
        blocks = list(responses.groupby(strata, sort=True).indices.values())

    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    skipped = 0
    for _ in range(reps):
        idx = np.concatenate([b[rng.integers(0, len(b), size=len(b))] for b in blocks])
        resampled = margins.subset(idx)
        if not resampled.covers_all_categories():
            skipped += 1
            continue
        w = rake_prepared(resampled, trim=trim)
        g = group[idx]
        num = np.bincount(g, weights=w * x[idx], minlength=len(levels))
        den = np.bincount(g, weights=w, minlength=len(levels))
        with np.errstate(invalid="ignore", divide="ignore"):
            draws.append(100.0 * num / den)
    if len(draws) < 2:
        raise ValueError("fewer than 2 usable bootstrap replicates")
    arr = np.vstack(draws)
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return pd.DataFrame(
        {
            "se": np.nanstd(arr, axis=0, ddof=1),
            "ci_low": np.nanpercentile(arr, lo, axis=0),
            "ci_high": np.nanpercentile(arr, hi, axis=0),
            "reps": len(draws),
        },
        index=pd.Index(levels, name=by or "group"),
    )
