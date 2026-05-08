"""Tests for the `wyrd kenning unaccounted` mining tool."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli


def _write_corpus(tmp_path: Path) -> Path:
    """Minimal place_names file: country → region → list of names."""
    corpus = {
        "Test Country": {
            "Test Region": [
                # 'foobar' has no morpheme matches, so 'foobar' should be the
                # dominant unaccounted fragment in the output.
                "Foobar",
                "Foobar",
                "Foobar",
                # A name that does decompose; should not contribute to fragments.
                "Hill",
            ]
        }
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus))
    return path


def test_unaccounted_surfaces_unmatched_fragments(tmp_path):
    corpus = _write_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["unaccounted", "english", str(corpus), "--top", "5", "--min-length", "3"],
    )
    assert result.exit_code == 0, result.output
    # 'foobar' should appear with count >= 1 (it might over-decompose into
    # smaller chunks, but at least one fragment from it should surface).
    assert "fragment" in result.output


def test_unaccounted_json_output(tmp_path):
    corpus = _write_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["unaccounted", "english", str(corpus), "--as-json", "--min-length", "2"],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert all("fragment" in entry and "count" in entry for entry in data)


def test_unaccounted_respects_min_length(tmp_path):
    corpus = _write_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["unaccounted", "english", str(corpus), "--as-json", "--min-length", "5"],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert all(len(entry["fragment"]) >= 5 for entry in data)


def test_explain_cli_prints_each_reading():
    runner = CliRunner()
    result = runner.invoke(cli, ["explain", "Bridgewater"])
    assert result.exit_code == 0
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    # At least one reading should mention "Bridge" and "water".
    text = "\n".join(lines)
    assert "Bridge" in text
    assert "water" in text
