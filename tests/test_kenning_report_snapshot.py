"""Tests for the wyrd-d24 report-snapshot port: parser + ingest/diff +
the ``ingest-report-snapshot`` / ``report-snapshot-diff`` CLIs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.lexicon import init_schema
from wyrd.generators.kenning.lexicon.report_snapshot import (
    diff_snapshots,
    ingest_snapshot,
    parse_report_dir,
)


def _write_report_dir(
    d: Path, *, etymon: int, oe_cov: str, native_cells: int, drift: float
) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "stats.txt").write_text(
        f"source                               66\n"
        f"etymon                          {etymon}\n"
        f"reflex_etymon                     10293\n"
    )
    (d / "enrichment_status.txt").write_text(
        "L3 enrichment coverage\n"
        "- Total etymons: 2372621\n"
        "- `cognate_id`:   649324 / 2372621 ( 27.4%)\n"
        "- `stratum`:   166427 / 2372621 (  7.0%)\n"
    )
    (d / "era_coverage.txt").write_text(
        "- Total toponyms: 85,715\n- Toponyms with ≥1 attestation: 36,739\n"
    )
    (d / "rando_port_readiness.txt").write_text(
        "**Overall: FAIL — retirement gate closed**\n\n"
        "| Language | Total | Scholar | Empirical | Rando-only | Uncited | Scholar only | "
        "Coverage | C1 | C2 | C3 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |\n"
        f"| old_english | 2449 | 1884 | 274 | 0 | 291 | 76.9% | {oe_cov} | ✓ | ✓ | ✓ |\n"
    )
    (d / "empirical_priors.json").write_text(
        json.dumps({"version": "x", "native": [{}] * native_cells, "loan": [{}, {}]})
    )
    (d / "drift_english.json").write_text(
        json.dumps({"decomposition_rate_a": drift, "decomposition_rate_b": drift})
    )
    return d


def test_parse_report_dir_extracts_structured_metrics(tmp_path: Path) -> None:
    d = _write_report_dir(
        tmp_path / "snap", etymon=2372621, oe_cov="88.1%", native_cells=169, drift=1.0
    )
    metrics = {(m.category, m.metric): m for m in parse_report_dir(d)}

    assert metrics[("stats", "etymon")].value_num == 2372621
    assert metrics[("enrichment", "total_etymons")].value_num == 2372621
    assert metrics[("enrichment", "cognate_id_pct")].value_num == 27.4
    assert metrics[("era", "total_toponyms")].value_num == 85715  # comma-stripped
    # Two percent columns disambiguated: coverage is the headline, not scholar-only.
    assert metrics[("rando", "old_english.coverage")].value_num == 88.1
    assert metrics[("rando", "old_english.scholar_only_pct")].value_num == 76.9
    assert metrics[("rando", "old_english.c1")].value_num == 1.0
    assert metrics[("priors", "native_cells")].value_num == 169
    assert metrics[("drift", "english.decomposition_rate_a")].value_num == 1.0


def test_parse_tolerates_missing_files(tmp_path: Path) -> None:
    d = tmp_path / "sparse"
    d.mkdir()
    (d / "stats.txt").write_text("etymon  100\n")
    metrics = parse_report_dir(d)
    # Only stats parsed; no crash on the absent files.
    assert {(m.category, m.metric) for m in metrics} == {("stats", "etymon")}


def test_ingest_is_idempotent_and_diff_reports_deltas(tmp_path: Path) -> None:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    before = _write_report_dir(
        tmp_path / "before", etymon=2370418, oe_cov="74.4%", native_cells=165, drift=1.0
    )
    after = _write_report_dir(
        tmp_path / "after", etymon=2372621, oe_cov="88.1%", native_cells=169, drift=1.0
    )

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        n1 = ingest_snapshot(
            conn, label="before", captured_at="2026-05-29T00:00Z", metrics=parse_report_dir(before)
        )
        # Re-ingest same label → replaces, not duplicates.
        n2 = ingest_snapshot(
            conn, label="before", captured_at="2026-05-29T00:00Z", metrics=parse_report_dir(before)
        )
        assert n1 == n2
        rows = conn.execute(
            "SELECT COUNT(*) FROM report_metric WHERE snapshot_label='before'"
        ).fetchone()[0]
        assert rows == n1  # not doubled

        ingest_snapshot(
            conn, label="after", captured_at="2026-05-30T00:00Z", metrics=parse_report_dir(after)
        )
        conn.commit()

        deltas = {(d.category, d.metric): d for d in diff_snapshots(conn, "before", "after")}
        # etymon grew by exactly 2203; coverage moved 74.4 → 88.1.
        assert deltas[("stats", "etymon")].delta_num == 2203
        assert deltas[("rando", "old_english.coverage")].a_text == "74.4%"
        assert deltas[("rando", "old_english.coverage")].b_text == "88.1%"
        # Unchanged metrics (drift 1.0 both) are NOT in the diff.
        assert ("drift", "english.decomposition_rate_a") not in deltas
    finally:
        conn.close()


def test_cli_ingest_and_diff_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    before = _write_report_dir(
        tmp_path / "b", etymon=2370418, oe_cov="74.4%", native_cells=165, drift=1.0
    )
    after = _write_report_dir(
        tmp_path / "a", etymon=2372621, oe_cov="88.1%", native_cells=169, drift=1.0
    )
    runner = CliRunner()

    # Dry-run writes nothing.
    dry = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-report-snapshot",
            str(before),
            "--label",
            "before",
            "--db",
            str(db_path),
        ],
        catch_exceptions=False,
    )
    assert dry.exit_code == 0
    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM report_snapshot").fetchone()[0] == 0
    conn.close()

    for d, label, ts in (
        (before, "before", "2026-05-29T00:00Z"),
        (after, "after", "2026-05-30T00:00Z"),
    ):
        res = runner.invoke(
            cli_root,
            [
                "lexicon",
                "ingest-report-snapshot",
                str(d),
                "--label",
                label,
                "--captured-at",
                ts,
                "--db",
                str(db_path),
                "--apply",
            ],
            catch_exceptions=False,
        )
        assert res.exit_code == 0, res.output

    diff = runner.invoke(
        cli_root,
        ["lexicon", "report-snapshot-diff", "before", "after", "--db", str(db_path)],
        catch_exceptions=False,
    )
    assert diff.exit_code == 0
    out = diff.stdout
    assert "etymon" in out and "coverage" in out
