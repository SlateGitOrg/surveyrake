"""Glue: weight each quarter's respondents to that quarter's stay population."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .rake import DEFAULT_TRIM, RakeReport, margin_targets, rake

DEFAULT_MARGINS: tuple[str, ...] = ("tier", "channel", "purpose", "property")


def parse_margins(spec: str | Sequence[str]) -> list[str]:
    """Accept ``"tier,channel"`` or a list; reject empty and duplicate entries."""
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    items = [i.strip() for i in items if i.strip()]
    if not items:
        raise ValueError("at least one margin variable is required")
    if len(set(items)) != len(items):
        raise ValueError(f"duplicate margin variables in {items}")
    return items


def weight_quarter(
    responses: pd.DataFrame,
    population: pd.DataFrame,
    quarter: str,
    margins: Sequence[str] = DEFAULT_MARGINS,
    trim: float | None = DEFAULT_TRIM,
) -> tuple[pd.DataFrame, RakeReport]:
    """Respondents of one quarter with a raked ``weight`` column, and the raking report."""
    rows = responses[responses["quarter"] == quarter].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"no responses for quarter {quarter!r}")
    targets = margin_targets(population, margins, quarter)
    result = rake(rows, targets, trim=trim)
    return rows.assign(weight=result.weights), result.report


def weight_quarters(
    responses: pd.DataFrame,
    population: pd.DataFrame,
    quarters: Sequence[str] | None = None,
    margins: Sequence[str] = DEFAULT_MARGINS,
    trim: float | None = DEFAULT_TRIM,
) -> tuple[pd.DataFrame, dict[str, RakeReport]]:
    """Rake each quarter separately and stack the results (the pooled sample)."""
    chosen = list(quarters) if quarters else sorted(responses["quarter"].unique())
    frames, reports = [], {}
    for quarter in chosen:
        frame, report = weight_quarter(responses, population, quarter, margins, trim)
        frames.append(frame)
        reports[quarter] = report
    return pd.concat(frames, ignore_index=True), reports
