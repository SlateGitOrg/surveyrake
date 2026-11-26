"""Command line interface: generate, rake, nps, compare, diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from . import __version__
from .compare import DEFAULT_PRACTICAL_CHANGE
from .data import load_population, load_responses, load_truth, save_dataset
from .nps import DEFAULT_BOOTSTRAP_REPS, bootstrap_nps, estimate_nps
from .pipeline import DEFAULT_MARGINS, parse_margins, weight_quarters
from .rake import DEFAULT_TRIM, RakeReport, margin_targets
from .report import (
    build_rows,
    markdown_table,
    render_rows,
    render_vs_rest,
    signed,
    verdict_legend,
)
from .synth import DEFAULT_STAYS_PER_QUARTER, QUARTERS, draw_responses, generate_population

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Rake hotel guest survey responses to known stay margins and judge NPS moves.",
)
err = Console(stderr=True)

DEFAULT_MARGINS_ARG = ",".join(DEFAULT_MARGINS)

DataDir = Annotated[Path, typer.Option("--data", help="Directory written by `generate`.")]
MarginsOpt = Annotated[str, typer.Option("--margins", help="Comma-separated variables to rake on.")]
TrimOpt = Annotated[
    float, typer.Option("--trim", help="Trim weights at this multiple of the mean (0 disables).")
]


def _fail(message: str) -> typer.Exit:
    err.print(f"error: {message}", style="red", markup=False, highlight=False)
    return typer.Exit(code=2)


def _trim(value: float) -> float | None:
    return None if value <= 0 else value


def _version(show: bool) -> None:
    if show:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """surveyrake."""


@app.command()
def generate(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("data"),
    seed: Annotated[int, typer.Option(help="Seed for the stay population.")] = 2025,
    response_seed: Annotated[int, typer.Option(help="Seed for who responds.")] = 2,
    stays_per_quarter: Annotated[int, typer.Option(min=2_000)] = DEFAULT_STAYS_PER_QUARTER,
) -> None:
    """Write synthetic stays (as margin counts), survey responses and the true NPS."""
    population = generate_population(seed, stays_per_quarter)
    responses = draw_responses(population, response_seed)
    paths = save_dataset(population, responses, out)
    for q in QUARTERS:
        n = int((responses["quarter"] == q).sum())
        typer.echo(f"{q}: {int((population.stays['quarter'] == q).sum()):,} stays, {n:,} responses")
    for path in paths:
        typer.echo(f"wrote {path}")


def _load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return load_population(data), load_responses(data)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc


def _weight(
    responses: pd.DataFrame,
    population: pd.DataFrame,
    quarters: list[str] | None,
    margins: str,
    trim: float,
) -> tuple[pd.DataFrame, dict[str, RakeReport], list[str]]:
    try:
        variables = parse_margins(margins)
        weighted, reports = weight_quarters(responses, population, quarters, variables, _trim(trim))
    except ValueError as exc:  # includes EmptyMarginCategoryError
        raise _fail(str(exc)) from exc
    return weighted, reports, variables


def _report_table(reports: dict[str, RakeReport]) -> str:
    headers = [
        "Quarter",
        "Iterations",
        "Trim rounds",
        "Max margin error",
        "Design effect",
        "Weight range (x mean)",
        "Converged",
    ]
    rows = [
        [
            quarter,
            str(r.iterations),
            str(r.trim_rounds),
            f"{r.max_margin_error:.1e}",
            f"{r.design_effect:.2f}",
            f"{r.weight_min / r.weight_mean:.2f} to {r.weight_max / r.weight_mean:.2f}",
            "yes" if r.converged else "NO",
        ]
        for quarter, r in reports.items()
    ]
    return markdown_table(headers, rows)


@app.command("rake")
def rake_command(
    data: DataDir = Path("data"),
    margins: MarginsOpt = DEFAULT_MARGINS_ARG,
    quarter: Annotated[str | None, typer.Option(help="One quarter (default: both).")] = None,
    trim: TrimOpt = DEFAULT_TRIM,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write respondent weights here.")
    ] = None,
) -> None:
    """Rake respondents to the stay-population margins and report convergence."""
    population, responses = _load(data)
    weighted, reports, variables = _weight(
        responses, population, [quarter] if quarter else None, margins, trim
    )
    typer.echo(_report_table(reports))
    for q, r in reports.items():
        for var, e in r.margin_errors.items():
            typer.echo(f"{q} margin {var}: max error {e:.2e} of population")
        for message in r.messages:
            err.print(f"warning: {q}: {message}", style="yellow", markup=False)
    if out is not None:
        weighted.to_csv(out, index=False)
        typer.echo(f"wrote {out} (raked on {', '.join(variables)})")
    if not all(r.converged for r in reports.values()):
        raise typer.Exit(code=1)


@app.command()
def nps(
    data: DataDir = Path("data"),
    by: Annotated[
        str, typer.Option(help="property, tier, channel, purpose or group.")
    ] = "property",
    quarter: Annotated[str | None, typer.Option(help="Quarter (default: both pooled).")] = None,
    margins: MarginsOpt = DEFAULT_MARGINS_ARG,
    trim: TrimOpt = DEFAULT_TRIM,
    bootstrap: Annotated[
        int, typer.Option(help=f"Re-raking bootstrap replicates (e.g. {DEFAULT_BOOTSTRAP_REPS}).")
    ] = 0,
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")] = 0,
) -> None:
    """Raked NPS with trinomial CI, effective sample size and design effect."""
    population, responses = _load(data)
    weighted, _, _ = _weight(responses, population, [quarter] if quarter else None, margins, trim)
    dimension = None if by == "group" else by
    if dimension is not None and dimension not in weighted.columns:
        raise _fail(f"cannot group by {by!r}; choose from property, tier, channel, purpose, group")
    boot = None
    if bootstrap:
        if quarter is None:
            raise _fail("--bootstrap needs --quarter (it re-rakes one quarter's survey)")
        boot = _bootstrap_se(weighted, population, quarter, margins, dimension, bootstrap, seed)
    rows = build_rows(weighted, quarter, by=dimension, boot_se=boot)
    typer.echo(f"NPS by {by}, {quarter or 'both quarters pooled'}\n")
    typer.echo(render_rows(rows, label=by.capitalize()))


def _bootstrap_se(
    weighted: pd.DataFrame,
    population: pd.DataFrame,
    quarter: str,
    margins: str,
    by: str | None,
    reps: int,
    seed: int,
) -> dict[str, float]:
    frame = weighted[weighted["quarter"] == quarter].drop(columns="weight")
    targets = margin_targets(population, parse_margins(margins), quarter)
    result = bootstrap_nps(frame, targets, by=by, reps=reps, seed=seed)
    out = {str(k): float(v) for k, v in result["se"].items()}
    group = bootstrap_nps(frame, targets, by=None, reps=reps, seed=seed)
    out["Group"] = float(group["se"].iloc[0])
    return out


@app.command()
def compare(
    data: DataDir = Path("data"),
    from_quarter: Annotated[str, typer.Option("--from", help="Earlier quarter.")] = QUARTERS[0],
    to_quarter: Annotated[str, typer.Option("--to", help="Later quarter.")] = QUARTERS[1],
    margins: MarginsOpt = DEFAULT_MARGINS_ARG,
    trim: TrimOpt = DEFAULT_TRIM,
    practical: Annotated[
        float, typer.Option(help="Smallest change (points) worth acting on.")
    ] = DEFAULT_PRACTICAL_CHANGE,
    holm: Annotated[
        bool,
        typer.Option("--holm", help="Also give verdicts with Holm correction across properties."),
    ] = False,
    out: Annotated[Path | None, typer.Option("--out", help="Also write the report here.")] = None,
) -> None:
    """Quarter-on-quarter change per property: signal, noise or cannot detect."""
    if from_quarter == to_quarter:
        raise _fail("--from and --to must be different quarters")
    population, responses = _load(data)
    weighted, _, _ = _weight(responses, population, [from_quarter, to_quarter], margins, trim)
    rows = build_rows(
        weighted, to_quarter, compare_to=from_quarter, practical_change=practical, holm=holm
    )
    latest = weighted[weighted["quarter"] == to_quarter]
    text = "\n".join(
        [
            f"Change {from_quarter} to {to_quarter}\n",
            render_rows(rows, from_quarter=from_quarter),
            "",
            verdict_legend(practical),
            "",
            f"Property against the rest of the group, {to_quarter}\n",
            render_vs_rest(latest, latest["weight"].to_numpy()),
        ]
    )
    typer.echo(text)
    if out is not None:
        out.write_text(text + "\n")


@app.command()
def diagnostics(
    data: DataDir = Path("data"),
    margins: MarginsOpt = DEFAULT_MARGINS_ARG,
    trim: TrimOpt = DEFAULT_TRIM,
    bootstrap: Annotated[int, typer.Option(help="Bootstrap replicates.")] = DEFAULT_BOOTSTRAP_REPS,
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")] = 0,
) -> None:
    """Raking diagnostics, bootstrap check of the variance, and the score against the truth."""
    population, responses = _load(data)
    weighted, reports, _ = _weight(responses, population, None, margins, trim)

    typer.echo("Raking diagnostics\n")
    typer.echo(_report_table(reports))

    typer.echo("\nAnalytic against bootstrap standard error of group NPS (points)\n")
    rows = []
    for q in reports:
        frame = weighted[weighted["quarter"] == q]
        est = estimate_nps(frame["score"], frame["weight"])
        boot = _bootstrap_se(weighted, population, q, margins, None, bootstrap, seed)["Group"]
        rows.append([q, f"{est.se:.3f}", f"{boot:.3f}", f"{boot / est.se:.2f}"])
    typer.echo(markdown_table(["Quarter", "Trinomial SE", "Bootstrap SE", "Ratio"], rows))

    truth = load_truth(data)
    if truth is None:
        return
    typer.echo("\nGroup NPS against the synthetic truth (points)\n")
    rows = []
    for q in (*reports, "pooled"):
        frame = weighted if q == "pooled" else weighted[weighted["quarter"] == q]
        raked = estimate_nps(frame["score"], frame["weight"])
        raw = estimate_nps(frame["score"])
        true = float(
            truth[(truth["quarter"] == q) & (truth["property"] == "group")]["true_nps"].iloc[0]
        )
        rows.append(
            [
                q,
                f"{raw.nps:.1f}",
                f"{raked.nps:.1f}",
                f"{true:.1f}",
                signed(raw.nps - true),
                signed(raked.nps - true),
            ]
        )
    typer.echo(
        markdown_table(
            ["Quarter", "Raw NPS", "Raked NPS", "True NPS", "Raw error", "Raked error"], rows
        )
    )


if __name__ == "__main__":
    app()
