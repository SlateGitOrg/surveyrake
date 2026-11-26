"""Is a move in NPS signal or noise? z-tests on independent weighted estimates, and power.

Two quarters' estimates for one property come from different guests, so they are independent and
their variances add. A property against the rest of the group is also independent, because the
comparison group excludes the property itself (comparing a property with a group that contains it
would understate the standard error of the difference).

The minimum detectable change (MDE) is the smallest true move a two-sided test at ``alpha`` would
catch with probability ``power``: (z_{1-alpha/2} + z_{power}) x SE of the difference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from .nps import NpsEstimate, estimate_nps

Verdict = Literal["signal", "noise", "cannot detect"]

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
DEFAULT_PRACTICAL_CHANGE = 20.0


@dataclass(frozen=True)
class Difference:
    """``after - before`` in NPS points, with a two-sided z-test."""

    diff: float
    se: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float
    mde: float


def minimum_detectable_change(
    se_diff: float, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> float:
    """Smallest true difference detected with probability ``power`` at level ``alpha``."""
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se_diff)


def difference(
    before: NpsEstimate,
    after: NpsEstimate,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> Difference:
    """z-test of ``after - before`` for two independent weighted NPS estimates."""
    diff = after.nps - before.nps
    se = sqrt(before.se**2 + after.se**2)
    z = diff / se if se > 0 else 0.0
    zc = float(norm.ppf(1 - alpha / 2))
    return Difference(
        diff=diff,
        se=se,
        z=z,
        p_value=float(2 * norm.sf(abs(z))),
        ci_low=diff - zc * se,
        ci_high=diff + zc * se,
        mde=minimum_detectable_change(se, alpha, power),
    )


def verdict(
    d: Difference, alpha: float = DEFAULT_ALPHA, practical_change: float = DEFAULT_PRACTICAL_CHANGE
) -> Verdict:
    """Call a change ``signal``, ``noise`` or ``cannot detect``.

    ``signal``: significant at ``alpha``. Otherwise the change is within sampling noise, and the
    test either had the power to see a move of ``practical_change`` points (``noise``: a move that
    size would have shown) or did not (``cannot detect``: silence here says nothing).
    """
    return _call(d.p_value, d.mde, alpha, practical_change)


def _call(p_value: float, mde: float, alpha: float, practical_change: float) -> Verdict:
    if p_value < alpha:
        return "signal"
    return "noise" if mde <= practical_change else "cannot detect"


def holm_verdicts(
    diffs: Sequence[Difference],
    alpha: float = DEFAULT_ALPHA,
    practical_change: float = DEFAULT_PRACTICAL_CHANGE,
) -> list[Verdict]:
    """Verdicts after Holm adjustment of the p-values across ``diffs`` (one family)."""
    adjusted = holm_adjust([d.p_value for d in diffs])
    return [_call(p, d.mde, alpha, practical_change) for p, d in zip(adjusted, diffs, strict=True)]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control), in input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def property_vs_rest(data: pd.DataFrame, weights: np.ndarray, by: str = "property") -> pd.DataFrame:
    """Each level's NPS minus the NPS of every other respondent, with a z-test."""
    scores = data["score"].to_numpy()
    labels = data[by].astype(str).to_numpy()
    rows = []
    for level in sorted(set(labels)):
        inside = labels == level
        own = estimate_nps(scores[inside], weights[inside])
        rest = estimate_nps(scores[~inside], weights[~inside])
        d = difference(rest, own)
        rows.append(
            {
                by: level,
                "nps": own.nps,
                "rest_nps": rest.nps,
                "diff": d.diff,
                "se": d.se,
                "z": d.z,
                "p_value": d.p_value,
            }
        )
    return pd.DataFrame(rows)
