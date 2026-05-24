"""Tests for the audit-semantic-coherence known-polysemy allowlist
(wyrd-kutx batch 2D).

Covers the loader + the (source_lang, source_lemma) filter shape. The
full audit pass needs Ollama and embeds thousands of glosses, so we
don't exercise it end-to-end — just the unit boundaries the allowlist
touches.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from wyrd.generators.kenning.cli.lexicon.audit_semantic_coherence import (
    _load_known_polysemy,
)


def _write_allowlist(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "known_polysemy.jsonl"
    source = {"_type": "source", "ref": "known-polysemy", "title": "test"}
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(source) + "\n")
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    return path


def test_load_returns_empty_when_path_is_none():
    assert _load_known_polysemy(None) == frozenset()


def test_load_returns_empty_when_file_missing(tmp_path: Path):
    assert _load_known_polysemy(tmp_path / "nope.jsonl") == frozenset()


def test_load_skips_source_row_collects_polysemy_rows(tmp_path: Path):
    path = _write_allowlist(
        tmp_path,
        [
            {
                "_type": "known_polysemy",
                "source_lang": "old_english",
                "source_lemma": "merece",
                "reason": "Smallage = Wild celery",
            },
            {
                "_type": "known_polysemy",
                "source_lang": "celtic_mix",
                "source_lemma": "sceach",
            },
        ],
    )
    assert _load_known_polysemy(path) == frozenset(
        {("old_english", "merece"), ("celtic_mix", "sceach")}
    )


def test_load_ignores_blank_and_comment_lines(tmp_path: Path):
    path = tmp_path / "known_polysemy.jsonl"
    path.write_text(
        '{"_type": "source", "ref": "known-polysemy", "title": "test"}\n'
        "\n"
        "# This is a comment line — should be skipped\n"
        '{"_type": "known_polysemy", "source_lang": "old_english", "source_lemma": "dim"}\n'
        "\n"
    )
    assert _load_known_polysemy(path) == frozenset({("old_english", "dim")})


def test_load_skips_unknown_type_rows(tmp_path: Path):
    """A future event-type added to the same file (e.g. known_split,
    known_homograph) shouldn't poison the polysemy loader."""
    path = _write_allowlist(
        tmp_path,
        [
            {"_type": "known_split", "source_lang": "x", "source_lemma": "y"},
            {
                "_type": "known_polysemy",
                "source_lang": "old_english",
                "source_lemma": "merece",
            },
        ],
    )
    assert _load_known_polysemy(path) == frozenset({("old_english", "merece")})


def test_load_raises_on_malformed_json(tmp_path: Path):
    path = tmp_path / "known_polysemy.jsonl"
    path.write_text(
        '{"_type": "source", "ref": "known-polysemy", "title": "test"}\n'
        '{"_type": "known_polysemy", "source_lang": "old_english"  # missing brace\n'
    )
    with pytest.raises(click.ClickException, match="malformed JSON"):
        _load_known_polysemy(path)


def test_load_raises_on_missing_source_lang(tmp_path: Path):
    path = _write_allowlist(
        tmp_path,
        [{"_type": "known_polysemy", "source_lemma": "merece"}],
    )
    with pytest.raises(click.ClickException, match="missing"):
        _load_known_polysemy(path)


def test_load_raises_on_missing_source_lemma(tmp_path: Path):
    path = _write_allowlist(
        tmp_path,
        [{"_type": "known_polysemy", "source_lang": "old_english"}],
    )
    with pytest.raises(click.ClickException, match="missing"):
        _load_known_polysemy(path)


def test_load_shipped_default_allowlist_parses_cleanly():
    """The committed data/audit/known_polysemy.jsonl is the operator-
    maintained source of truth; it must load without error and carry
    the 9 batch-2D entries from the wyrd-kutx audit walkthrough."""
    path = Path(__file__).resolve().parent.parent / "data" / "audit" / "known_polysemy.jsonl"
    entries = _load_known_polysemy(path)
    assert len(entries) == 9
    # Spot-check the headline batch-2D entries are all present.
    assert ("old_english", "merece") in entries
    assert ("celtic_mix", "draigneach") in entries
    assert ("celtic_mix", "sceach") in entries
    assert ("old_scandinavian", "stokkr") in entries
