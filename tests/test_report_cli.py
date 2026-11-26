from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from surveyrake.cli import app
from surveyrake.pipeline import parse_margins
from surveyrake.report import build_rows, markdown_table, render_rows, render_vs_rest

runner = CliRunner()


def test_markdown_table_aligns_and_has_header_rule():
    text = markdown_table(["Name", "Value"], [["a", "1"], ["longer", "22"]])
    lines = text.splitlines()
    assert lines[0] == "| Name   | Value |"
    assert lines[1] == "| ------ | ----: |"
    assert lines[2] == "| a      |     1 |"
    assert len({len(line) for line in lines}) == 1


def test_property_table_has_every_required_column_and_a_group_row(weighted):
    rows = build_rows(weighted, "2025-Q2", compare_to="2025-Q1")
    assert [r.label for r in rows][:2] == ["P01", "P02"] and rows[-1].label == "Group"
    text = render_rows(rows, from_quarter="2025-Q1")
    header = text.splitlines()[0]
    for column in (
        "Property", "Responses", "Raw NPS", "Raked NPS", "95% CI", "n_eff", "Design effect",
        "MDE", "Change vs 2025-Q1", "Verdict",
    ):  # fmt: skip
        assert column in header
    body = {line.split("|")[1].strip(): line for line in text.splitlines()[2:]}
    assert "signal" in body["P07"]
    assert "signal" not in body["P03"]
    assert len(body) == 13


def test_single_quarter_table_has_no_change_columns(weighted):
    text = render_rows(build_rows(weighted, "2025-Q2"))
    assert "Verdict" not in text and "MDE" not in text


def test_rows_for_other_dimensions_and_pooled(weighted):
    by_tier = build_rows(weighted, None, by="tier")
    assert [r.label for r in by_tier] == ["gold", "none", "silver", "Group"]
    assert by_tier[-1].responses == len(weighted)
    raw_gap = by_tier[-1].raw_nps - by_tier[-1].estimate.nps
    assert raw_gap > 8


def test_build_rows_rejects_unknown_quarters_and_pooled_changes(weighted):
    with pytest.raises(ValueError, match="no responses"):
        build_rows(weighted, "2031-Q4")
    with pytest.raises(ValueError, match="specific quarter"):
        build_rows(weighted, None, compare_to="2025-Q1")


def test_vs_rest_table_renders_every_property(weighted):
    q2 = weighted[weighted["quarter"] == "2025-Q2"]
    text = render_vs_rest(q2, q2["weight"].to_numpy())
    assert text.count("\n") == 13 and "p-value" in text


def test_parse_margins_validation():
    assert parse_margins("tier, channel") == ["tier", "channel"]
    with pytest.raises(ValueError, match="duplicate"):
        parse_margins("tier,tier")
    with pytest.raises(ValueError, match="at least one"):
        parse_margins(" , ")


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    result = runner.invoke(app, ["generate", "--out", str(out), "--stays-per-quarter", "20000"])
    assert result.exit_code == 0, result.output
    return out


def test_cli_generate_writes_three_files(small_data):
    for name in ("population.csv", "responses.csv", "truth.csv"):
        assert (small_data / name).exists()
    pop = pd.read_csv(small_data / "population.csv")
    assert list(pop.columns) == ["quarter", "property", "tier", "channel", "purpose", "stays"]


def test_cli_rake_reports_convergence_and_writes_weights(small_data, tmp_path):
    out = tmp_path / "weights.csv"
    result = runner.invoke(
        app, ["rake", "--data", str(small_data), "--margins", "tier,channel", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "Max margin error" in result.output and "2025-Q2" in result.output
    weights = pd.read_csv(out)
    assert (weights["weight"] > 0).all() and "weight" in weights.columns


def test_cli_nps_by_property_for_one_quarter(small_data):
    result = runner.invoke(
        app, ["nps", "--data", str(small_data), "--by", "property", "--quarter", "2025-Q2"]
    )
    assert result.exit_code == 0, result.output
    assert "P07" in result.output and "Raked NPS" in result.output


def test_cli_nps_with_bootstrap_adds_se_columns(small_data):
    result = runner.invoke(
        app,
        ["nps", "--data", str(small_data), "--by", "group", "--quarter", "2025-Q1",
         "--bootstrap", "20"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Boot SE" in result.output


def test_cli_compare_prints_verdicts_and_writes_report(small_data, tmp_path):
    out = tmp_path / "report.md"
    result = runner.invoke(
        app,
        ["compare", "--data", str(small_data), "--from", "2025-Q1", "--to", "2025-Q2",
         "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Change vs 2025-Q1" in result.output and "cannot detect" in result.output
    assert out.read_text().startswith("Change 2025-Q1 to 2025-Q2")


def test_cli_diagnostics_scores_against_truth(small_data):
    result = runner.invoke(app, ["diagnostics", "--data", str(small_data), "--bootstrap", "30"])
    assert result.exit_code == 0, result.output
    for text in ("Raking diagnostics", "Bootstrap SE", "True NPS", "pooled"):
        assert text in result.output


def test_cli_errors_are_clear_and_exit_nonzero(small_data, tmp_path):
    missing = runner.invoke(app, ["nps", "--data", str(tmp_path / "nowhere")])
    assert missing.exit_code == 2 and "generate" in missing.output
    bad_margin = runner.invoke(app, ["rake", "--data", str(small_data), "--margins", "tier,colour"])
    assert bad_margin.exit_code == 2 and "colour" in bad_margin.output
    same = runner.invoke(
        app, ["compare", "--data", str(small_data), "--from", "2025-Q1", "--to", "2025-Q1"]
    )
    assert same.exit_code == 2
    bad_by = runner.invoke(app, ["nps", "--data", str(small_data), "--by", "colour"])
    assert bad_by.exit_code == 2


def test_cli_names_the_empty_margin_category(small_data, tmp_path):
    data = tmp_path / "d"
    data.mkdir()
    (data / "population.csv").write_bytes((small_data / "population.csv").read_bytes())
    responses = pd.read_csv(small_data / "responses.csv")
    responses[responses["tier"] != "gold"].to_csv(data / "responses.csv", index=False)
    result = runner.invoke(app, ["rake", "--data", str(data)])
    assert result.exit_code == 2
    assert "'tier'" in result.output and "'gold'" in result.output
    assert "Collapse" in result.output


def test_cli_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.output.strip() == "0.1.0"


def test_holm_column_appears_only_when_asked(weighted):
    plain = render_rows(build_rows(weighted, "2025-Q2", compare_to="2025-Q1"))
    assert "Holm" not in plain
    rows = build_rows(weighted, "2025-Q2", compare_to="2025-Q1", holm=True)
    text = render_rows(rows, from_quarter="2025-Q1")
    assert "Verdict (Holm)" in text
    p07 = next(r for r in rows if r.label == "P07")
    assert p07.verdict == "signal" and p07.holm_verdict == "signal"
    assert all(r.holm_verdict is not None for r in rows)


def test_cli_compare_holm_flag(small_data):
    result = runner.invoke(app, ["compare", "--data", str(small_data), "--holm"])
    assert result.exit_code == 0, result.output
    assert "Verdict (Holm)" in result.output
