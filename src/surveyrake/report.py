"""Markdown tables: one row per property (plus the group) with raw and raked NPS and a verdict."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .compare import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    DEFAULT_PRACTICAL_CHANGE,
    Difference,
    Verdict,
    difference,
    holm_verdicts,
    property_vs_rest,
    verdict,
)
from .nps import NpsEstimate, estimate_nps


@dataclass(frozen=True)
class Row:
    """One line of the report. ``change`` is None for a single-quarter table."""

    label: str
    responses: int
    raw_nps: float
    estimate: NpsEstimate
    change: Difference | None = None
    verdict: Verdict | None = None
    holm_verdict: Verdict | None = None
    boot_se: float | None = None


def _estimates(
    frame: pd.DataFrame, by: str | None
) -> tuple[dict[str, NpsEstimate], dict[str, NpsEstimate]]:
    """(raked, raw) estimates per level of ``by``, plus a ``Group`` entry."""
    scores = frame["score"].to_numpy()
    weights = frame["weight"].to_numpy()
    raked: dict[str, NpsEstimate] = {}
    raw: dict[str, NpsEstimate] = {}
    if by is not None:
        for level, idx in frame.groupby(by, sort=True).indices.items():
            raked[str(level)] = estimate_nps(scores[idx], weights[idx])
            raw[str(level)] = estimate_nps(scores[idx])
    raked["Group"] = estimate_nps(scores, weights)
    raw["Group"] = estimate_nps(scores)
    return raked, raw


def build_rows(
    weighted: pd.DataFrame,
    quarter: str | None,
    *,
    by: str | None = "property",
    compare_to: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    practical_change: float = DEFAULT_PRACTICAL_CHANGE,
    boot_se: dict[str, float] | None = None,
    holm: bool = False,
) -> list[Row]:
    """Rows for ``quarter`` (all quarters pooled when None), with changes from ``compare_to``.

    With ``holm`` each level's verdict is also given after Holm adjustment of the p-values across
    the levels (the Group row is its own test and is not part of that family).
    """
    frame = weighted if quarter is None else weighted[weighted["quarter"] == quarter]
    if frame.empty:
        raise ValueError(f"no responses for quarter {quarter!r}")
    raked, raw = _estimates(frame, by)
    baseline = None
    if compare_to is not None:
        if quarter is None:
            raise ValueError("a change needs a specific quarter to compare to")
        base_frame = weighted[weighted["quarter"] == compare_to]
        if base_frame.empty:
            raise ValueError(f"no responses for quarter {compare_to!r}")
        baseline, _ = _estimates(base_frame, by)

    rows = []
    for label, est in raked.items():
        change = verdict_ = None
        if baseline is not None:
            if label not in baseline:
                raise ValueError(f"{label!r} has no responses in {compare_to}")
            change = difference(baseline[label], est, alpha, power)
            verdict_ = verdict(change, alpha, practical_change)
        rows.append(
            Row(
                label=label,
                responses=est.n,
                raw_nps=raw[label].nps,
                estimate=est,
                change=change,
                verdict=verdict_,
                boot_se=None if boot_se is None else boot_se.get(label),
            )
        )
    if holm and baseline is not None:
        members = [i for i, r in enumerate(rows) if r.label != "Group"]
        calls = holm_verdicts([rows[i].change for i in members], alpha, practical_change)
        for i, call in zip(members, calls, strict=True):
            rows[i] = replace(rows[i], holm_verdict=call)
        group = next((i for i, r in enumerate(rows) if r.label == "Group"), None)
        if group is not None:
            rows[group] = replace(rows[group], holm_verdict=rows[group].verdict)
    return rows


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A GitHub-flavoured Markdown table; the first column left-aligned, the rest right-aligned."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: Sequence[str]) -> str:
        padded = [c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    rule = ["-" * widths[0]] + ["-" * (w - 1) + ":" for w in widths[1:]]
    return "\n".join([line(headers), "| " + " | ".join(rule) + " |", *(line(r) for r in rows)])


def signed(x: float) -> str:
    return f"{x:+.1f}"


def render_rows(
    rows: Sequence[Row], label: str = "Property", from_quarter: str | None = None
) -> str:
    """The report table. Change, MDE and verdict columns appear only when rows carry a change."""
    with_change = any(r.change is not None for r in rows)
    with_boot = any(r.boot_se is not None for r in rows)
    headers = [label, "Responses", "Raw NPS", "Raked NPS", "95% CI", "n_eff", "Design effect"]
    if with_boot:
        headers += ["SE", "Boot SE"]
    if with_change:
        headers += ["MDE", f"Change vs {from_quarter}" if from_quarter else "Change", "Verdict"]
        if any(r.holm_verdict is not None for r in rows):
            headers.append("Verdict (Holm)")
    body = []
    for r in rows:
        e = r.estimate
        cells = [
            r.label,
            f"{r.responses:,}",
            f"{r.raw_nps:.1f}",
            f"{e.nps:.1f}",
            f"[{e.ci_low:.1f}, {e.ci_high:.1f}]",
            f"{e.n_eff:,.0f}",
            f"{e.design_effect:.2f}",
        ]
        if with_boot:
            cells += [f"{e.se:.2f}", "-" if r.boot_se is None else f"{r.boot_se:.2f}"]
        if with_change:
            assert r.change is not None and r.verdict is not None
            cells += [f"{r.change.mde:.1f}", signed(r.change.diff), r.verdict]
            if "Verdict (Holm)" in headers:
                cells.append(r.holm_verdict or "-")
        body.append(cells)
    return markdown_table(headers, body)


def render_vs_rest(data: pd.DataFrame, weights: np.ndarray, by: str = "property") -> str:
    """Each property's raked NPS against the raked NPS of all other respondents."""
    table = property_vs_rest(data, weights, by)
    rows = [
        [
            str(r[by]),
            f"{r.nps:.1f}",
            f"{r.rest_nps:.1f}",
            signed(r["diff"]),
            f"{r.se:.1f}",
            f"{r.p_value:.3f}",
        ]
        for _, r in table.iterrows()
    ]
    return markdown_table(
        [by.capitalize(), "Raked NPS", "Rest of group", "Difference", "SE", "p-value"], rows
    )


def verdict_legend(practical_change: float = DEFAULT_PRACTICAL_CHANGE) -> str:
    return (
        "Verdicts: `signal` = change significant at 5%; `noise` = not significant, and a change "
        f"of {practical_change:g} points would have been detected (MDE <= {practical_change:g}); "
        f"`cannot detect` = not significant, but the MDE is above {practical_change:g} points, "
        "so this sample cannot tell. MDE is the minimum detectable change of the "
        "quarter-on-quarter difference (both quarters' variances included): the smallest true "
        "change detected with "
        "80% power at 5% significance. Each property is tested on its own at 5%; pass --holm "
        "for verdicts corrected across properties."
    )
