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
    _apply_allowlist_filter,
    _apply_polysemy_filter,
    _load_audit_allowlist,
    _load_known_cluster_acceptable,
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


def test_load_returns_empty_when_path_is_empty_string():
    """`--known-polysemy-file ""` is the natural operator-bypass gesture;
    click parses it to Path(".") which exists as a directory. Without
    this guard the loader would crash with IsADirectoryError mid-audit."""
    assert _load_known_polysemy(Path("")) == frozenset()


def test_load_returns_empty_when_path_is_directory(tmp_path: Path):
    """Defense in depth — if an operator points the flag at a directory
    by mistake (e.g. forgot the filename), bail out cleanly rather than
    raising IsADirectoryError deep inside the audit pass."""
    sub = tmp_path / "subdir"
    sub.mkdir()
    assert _load_known_polysemy(sub) == frozenset()


def test_load_strips_whitespace_on_loaded_keys(tmp_path: Path):
    """Hand-edited entries can pick up trailing whitespace; the loader
    normalizes so the (source_lang, source_lemma) tuple matches even
    when the operator pastes with a stray space."""
    path = _write_allowlist(
        tmp_path,
        [
            {
                "_type": "known_polysemy",
                "source_lang": " old_english ",
                "source_lemma": " merece\t",
            }
        ],
    )
    assert _load_known_polysemy(path) == frozenset({("old_english", "merece")})


# ---------------------------------------------------------------------------
# _apply_polysemy_filter
# ---------------------------------------------------------------------------


def _row(lang: str, lemma: str, **kwargs):
    base = {"source_lang": lang, "source_lemma": lemma}
    base.update(kwargs)
    return base


def test_apply_filter_no_op_when_allowlist_empty():
    rows = [_row("old_english", "dry"), _row("celtic_mix", "draigneach")]
    kept, dropped = _apply_polysemy_filter(rows, frozenset())
    assert kept == rows
    assert dropped == 0


def test_apply_filter_drops_matching_rows_only(tmp_path: Path):
    rows = [
        _row("old_english", "merece", n_glosses=2),
        _row("old_english", "gear", n_glosses=2),
        _row("celtic_mix", "sceach", n_glosses=2),
    ]
    allowlist = frozenset({("old_english", "merece"), ("celtic_mix", "sceach")})
    kept, dropped = _apply_polysemy_filter(rows, allowlist)
    assert dropped == 2
    assert kept == [_row("old_english", "gear", n_glosses=2)]


def test_apply_filter_strips_whitespace_on_both_sides():
    """Both row-side and allowlist-side keys get stripped — defense
    against whitespace drift in either layer."""
    rows = [_row("old_english ", " merece"), _row("celtic_mix", "draigneach")]
    allowlist = frozenset({("old_english", "merece")})
    kept, dropped = _apply_polysemy_filter(rows, allowlist)
    assert dropped == 1
    assert kept == [_row("celtic_mix", "draigneach")]


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


# ---------------------------------------------------------------------------
# Cross-sibling event type: _load_known_cluster_acceptable
# ---------------------------------------------------------------------------


def test_load_cluster_acceptable_collects_only_its_event_type(tmp_path: Path):
    """Same file can carry both event types; the two loaders are
    typed-filtered and don't cross-pollinate."""
    path = _write_allowlist(
        tmp_path,
        [
            {
                "_type": "known_polysemy",
                "source_lang": "old_english",
                "source_lemma": "merece",
            },
            {
                "_type": "known_cluster_acceptable",
                "source_lang": "old_english",
                "source_lemma": "ost",
                "reason": "-ost- cluster",
            },
            {
                "_type": "known_cluster_acceptable",
                "source_lang": "old_french",
                "source_lemma": "ost",
                "reason": "-ost- cluster",
            },
        ],
    )
    assert _load_known_polysemy(path) == frozenset({("old_english", "merece")})
    assert _load_known_cluster_acceptable(path) == frozenset(
        {("old_english", "ost"), ("old_french", "ost")}
    )


def test_load_audit_allowlist_arbitrary_event_type(tmp_path: Path):
    """The generic loader is event-type-parametric — a future event
    type can ride in the same file without changing the loader."""
    path = _write_allowlist(
        tmp_path,
        [
            {
                "_type": "future_event_type",
                "source_lang": "old_english",
                "source_lemma": "x",
            }
        ],
    )
    assert _load_audit_allowlist(path, "future_event_type") == frozenset({("old_english", "x")})
    # And it does NOT leak into the typed loaders:
    assert _load_known_polysemy(path) == frozenset()
    assert _load_known_cluster_acceptable(path) == frozenset()


def test_apply_polysemy_filter_back_compat_alias():
    """PR #344 exposed _apply_polysemy_filter; renamed to
    _apply_allowlist_filter in this PR. The back-compat alias must
    still work since external code or tests may import the old name."""
    rows = [
        {"source_lang": "old_english", "source_lemma": "ost"},
        {"source_lang": "celtic_mix", "source_lemma": "stad"},
    ]
    allowlist = frozenset({("old_english", "ost")})
    kept_via_alias, dropped_via_alias = _apply_polysemy_filter(rows, allowlist)
    kept_via_new, dropped_via_new = _apply_allowlist_filter(rows, allowlist)
    assert kept_via_alias == kept_via_new
    assert dropped_via_alias == dropped_via_new
    assert dropped_via_alias == 1


def test_load_shipped_cluster_acceptable_entries_parse_cleanly():
    """The committed data/audit/known_polysemy.jsonl now carries
    known_cluster_acceptable rows seeded from the cross-sibling-suspects
    audit. Pin a count + spot-check headline entries so accidental
    deletions / data-typos surface in CI."""
    path = Path(__file__).resolve().parent.parent / "data" / "audit" / "known_polysemy.jsonl"
    entries = _load_known_cluster_acceptable(path)
    # Seeded from the cross-sibling walkthrough — bump this assertion
    # when the operator adds new entries.
    assert len(entries) >= 50
    # Spot-check headline clusters (both halves of each cluster present).
    assert ("celtic_mix", "stad") in entries
    assert ("old_english", "stōd") in entries
    assert ("old_french", "ost") in entries
    assert ("old_english", "ost") in entries
    assert ("old_scandinavian", "hel") in entries
    assert ("celtic_mix", "hel") in entries
    assert ("modern_english", "sker") in entries
    assert ("old_scandinavian", "sker") in entries
