"""Raking (iterative proportional fitting) of survey weights to known population margins.

Respondents start from base weights and are scaled, one margin at a time, until the weighted
count in every category of every margin equals the stay population's count. Weights are then
trimmed at ``trim`` times the mean weight and raked again, in a loop, until both the margins and
the cap hold.

Margin error is measured as ``max |weighted count - target count| / population`` over every
category of every margin, so it is a share of the population and comparable across margins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Targets = Mapping[str, Mapping[str, float]]

DEFAULT_TOL = 1e-10
DEFAULT_MAX_ITER = 500
DEFAULT_TRIM = 5.0
DEFAULT_MAX_TRIM_ROUNDS = 100
_CAP_SLACK = 1e-9


class EmptyMarginCategoryError(ValueError):
    """A margin category has stays in the population but no respondents to carry its weight."""

    def __init__(self, variable: str, category: str, population: float, suggestion: str):
        self.variable = variable
        self.category = category
        self.population = population
        self.suggestion = suggestion
        super().__init__(
            f"margin {variable!r}: category {category!r} has {population:,.0f} stays in the "
            f"population but no respondents, so its weight cannot be raked. {suggestion}"
        )


@dataclass(frozen=True)
class RakeReport:
    """What the raking did and whether it met its targets."""

    converged: bool
    iterations: int
    trim_rounds: int
    max_margin_error: float
    margin_errors: dict[str, float]
    tol: float
    max_iter: int
    trim: float | None
    n: int
    population: float
    weight_min: float
    weight_max: float
    weight_mean: float
    n_trimmed: int
    n_eff: float
    design_effect: float
    messages: list[str] = field(default_factory=list)

    @property
    def max_over_mean(self) -> float:
        return self.weight_max / self.weight_mean


@dataclass(frozen=True)
class RakeResult:
    weights: np.ndarray
    report: RakeReport


def margin_targets(
    population: pd.DataFrame, variables: Sequence[str], quarter: str | None = None
) -> dict[str, dict[str, float]]:
    """Stay counts per category of each variable, from the cell-count population table."""
    pop = population if quarter is None else population[population["quarter"] == quarter]
    if pop.empty:
        raise ValueError(f"no population rows for quarter {quarter!r}")
    targets: dict[str, dict[str, float]] = {}
    for var in variables:
        if var not in pop.columns:
            raise ValueError(f"unknown margin variable {var!r}; population has {list(pop.columns)}")
        counts = pop.groupby(var, sort=True)["stays"].sum()
        targets[var] = {str(k): float(v) for k, v in counts.items() if v > 0}
    return targets


def kish_n_eff(weights: np.ndarray | pd.Series) -> float:
    """Kish effective sample size: (sum w)^2 / sum w^2."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


def _population_total(targets: Targets) -> float:
    totals = {var: float(sum(cats.values())) for var, cats in targets.items()}
    reference = next(iter(totals.values()))
    for var, total in totals.items():
        if abs(total - reference) > 1e-6 * max(reference, 1.0):
            raise ValueError(
                f"margins disagree on the population total: {var!r} sums to {total:,.0f}, "
                f"expected {reference:,.0f}"
            )
    return reference


def _encode(
    data: pd.DataFrame, targets: Targets
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    """Integer codes per margin, target vectors, and category labels; validates coverage."""
    codes, vectors, labels = [], [], []
    for var, cats in targets.items():
        if var not in data.columns:
            raise ValueError(f"respondent data has no column {var!r}")
        names = list(cats)
        index = {name: i for i, name in enumerate(names)}
        values = data[var].astype(str).to_numpy()
        unknown = sorted(set(values) - set(index))
        if unknown:
            raise ValueError(
                f"margin {var!r}: respondents have categories missing from the population "
                f"margin: {unknown}"
            )
        code = np.fromiter((index[v] for v in values), dtype=np.int64, count=len(values))
        present = np.bincount(code, minlength=len(names)) > 0
        for i, name in enumerate(names):
            if not present[i]:
                raise EmptyMarginCategoryError(
                    var,
                    name,
                    cats[name],
                    _collapse_suggestion(var, names, list(cats.values()), present, i),
                )
        codes.append(code)
        vectors.append(np.array([cats[n] for n in names], dtype=float))
        labels.append(names)
    return codes, vectors, labels


def _collapse_suggestion(
    var: str, names: list[str], counts: Sequence[float], present: np.ndarray, i: int
) -> str:
    """Suggest merging into the category with respondents whose stay count is closest."""
    options = [(abs(counts[j] - counts[i]), j) for j in range(len(names)) if present[j]]
    if not options:
        return f"No category of {var!r} has respondents; drop {var!r} from the margins."
    target = names[min(options)[1]]
    return (
        f"Collapse {names[i]!r} into {target!r} (the closest in size) in both the population and "
        f"the responses, or drop {var!r} from the raked margins."
    )


def _margin_error(
    codes: list[np.ndarray], vectors: list[np.ndarray], w: np.ndarray, total: float
) -> tuple[float, list[float]]:
    per_margin = [
        float(np.max(np.abs(np.bincount(c, weights=w, minlength=len(t)) - t)) / total)
        for c, t in zip(codes, vectors, strict=True)
    ]
    return max(per_margin), per_margin


def _ipf(
    codes: list[np.ndarray],
    vectors: list[np.ndarray],
    w: np.ndarray,
    total: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    """Cycle through the margins until the largest margin error is within ``tol``."""
    w = w.copy()
    err = np.inf
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        for code, target in zip(codes, vectors, strict=True):
            sums = np.bincount(code, weights=w, minlength=len(target))
            w *= (target / sums)[code]
        err, _ = _margin_error(codes, vectors, w, total)
        if err <= tol:
            break
    return w, iterations, err


@dataclass(frozen=True)
class PreparedMargins:
    """Respondent category codes and population target vectors, ready for fitting."""

    codes: list[np.ndarray]
    vectors: list[np.ndarray]
    total: float

    def subset(self, index: np.ndarray) -> PreparedMargins:
        """The same margins for a resample (``index`` may repeat respondents)."""
        return PreparedMargins([c[index] for c in self.codes], self.vectors, self.total)

    def covers_all_categories(self) -> bool:
        return all(
            bool(np.all(np.bincount(c, minlength=len(v)) > 0))
            for c, v in zip(self.codes, self.vectors, strict=True)
        )


def prepare_margins(data: pd.DataFrame, targets: Targets) -> PreparedMargins:
    """Validate ``targets`` against ``data`` and encode them. Raises on empty categories."""
    if not targets:
        raise ValueError("at least one margin is required")
    if len(data) == 0:
        raise ValueError("no respondents to rake")
    total = _population_total(targets)
    codes, vectors, _ = _encode(data, targets)
    return PreparedMargins(codes, vectors, total)


def _fit(
    margins: PreparedMargins,
    w: np.ndarray,
    tol: float,
    max_iter: int,
    trim: float | None,
    max_trim_rounds: int,
) -> tuple[np.ndarray, int, int, float, bool]:
    """IPF then the trim-rake loop. Returns weights, sweeps, trim rounds, error, over-cap flag."""
    codes, vectors, total = margins.codes, margins.vectors, margins.total
    w, iterations, err = _ipf(codes, vectors, w, total, tol, max_iter)
    rounds = 0
    over_cap = False
    if trim is not None:
        cap = trim * total / len(w)
        while np.any(w > cap * (1 + _CAP_SLACK)) and rounds < max_trim_rounds:
            rounds += 1
            w, used, err = _ipf(codes, vectors, np.minimum(w, cap), total, tol, max_iter)
            iterations += used
        over_cap = bool(np.any(w > cap * (1 + _CAP_SLACK)))
    return w, iterations, rounds, err, over_cap


def rake_prepared(
    margins: PreparedMargins,
    base_weights: np.ndarray | None = None,
    *,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    trim: float | None = DEFAULT_TRIM,
    max_trim_rounds: int = DEFAULT_MAX_TRIM_ROUNDS,
) -> np.ndarray:
    """Weights only, for callers (the bootstrap) that rake the same margins many times."""
    n = len(margins.codes[0])
    w0 = np.ones(n) if base_weights is None else np.asarray(base_weights, dtype=float)
    return _fit(margins, w0, tol, max_iter, trim, max_trim_rounds)[0]


def rake(
    data: pd.DataFrame,
    targets: Targets,
    base_weights: np.ndarray | None = None,
    *,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    trim: float | None = DEFAULT_TRIM,
    max_trim_rounds: int = DEFAULT_MAX_TRIM_ROUNDS,
) -> RakeResult:
    """Rake ``base_weights`` (default all ones) so weighted margins equal ``targets``.

    Only the relative size of the base weights matters: the first adjustment rescales them to the
    population total. When ``trim`` is set, any weight above ``trim`` x the mean weight is cut to
    that cap and the data are raked again, repeating until margins and cap both hold.

    Raises :class:`EmptyMarginCategoryError` when a category with stays has no respondents.
    A run that does not converge is returned with ``report.converged == False``.
    """
    margins = prepare_margins(data, targets)
    n = len(data)
    w0 = np.ones(n) if base_weights is None else np.asarray(base_weights, dtype=float)
    if w0.shape != (n,) or np.any(w0 <= 0) or not np.all(np.isfinite(w0)):
        raise ValueError("base weights must be positive, finite, one per respondent")

    w, iterations, rounds, err, over_cap = _fit(margins, w0, tol, max_iter, trim, max_trim_rounds)
    messages: list[str] = []
    if err > tol:
        messages.append(f"margins not met: error {err:.2e} after {iterations} iterations")
    if over_cap:
        messages.append(f"trim cap of {trim:g}x the mean not met after {rounds} trim rounds")

    worst, per_margin = _margin_error(margins.codes, margins.vectors, w, margins.total)
    cap = None if trim is None else trim * margins.total / n
    n_trimmed = 0 if cap is None else int(np.sum(w >= cap * (1 - _CAP_SLACK)))
    n_eff = kish_n_eff(w)
    report = RakeReport(
        converged=err <= tol and not over_cap,
        iterations=iterations,
        trim_rounds=rounds,
        max_margin_error=worst,
        margin_errors=dict(zip(targets, per_margin, strict=True)),
        tol=tol,
        max_iter=max_iter,
        trim=trim,
        n=n,
        population=margins.total,
        weight_min=float(w.min()),
        weight_max=float(w.max()),
        weight_mean=float(w.mean()),
        n_trimmed=n_trimmed,
        n_eff=n_eff,
        design_effect=n / n_eff,
        messages=messages,
    )
    return RakeResult(weights=w, report=report)
