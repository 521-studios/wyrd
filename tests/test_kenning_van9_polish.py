"""Tests for wyrd-van9: wyrd-kutx polish items.

Three operator-ergonomic improvements bundled together:

1. **Suffix charset validation** on ``curate-split-etymon --into``
   suffixes (CLI rejection + applier counter for JSONL hand-edits).
2. **--strict exit flag** on ``lexicon enrich`` so CI / deploy scripts
   fail loudly on unresolved-ref / typo signals; warnings still
   surface on stderr without --strict.
3. **Defensive load_parts** that tolerates proportions usages with no
   matching meaning_db key (the inflection-shadow-drift case that
   blocked the post-wyrd-zzli bundle re-emit).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.enrichment import (
    apply_etymon_splits,
    format_unresolved_warnings,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.runtime.proportions import MeaningGenerator


def _build_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _add_etymon_with_glosses(
    db_path: Path, language: str, canonical_form: str, glosses: list[str]
) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        (canonical_form, language),
    )
    eid = cur.lastrowid
    for gloss in glosses:
        conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (eid, gloss),
        )
    conn.commit()
    conn.close()
    return eid


# ---------------------------------------------------------------------------
# Item 1 — CLI suffix charset rejection
# ---------------------------------------------------------------------------


def test_cli_rejects_suffix_with_colon(tmp_path: Path):
    """Colons in a suffix would corrupt the ref `<lang>:<form>#<suffix>`
    parse — locked out of the CLI."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-split-etymon",
            "old-english:gear",
            "--into",
            "suffix=foo:bar|glosses=x",
            "--curation-file",
            str(tmp_path / "_curation.jsonl"),
        ],
    )
    assert result.exit_code != 0
    assert "disallowed" in result.output.lower() or "charset" in result.output.lower()


def test_cli_rejects_suffix_with_hash(tmp_path: Path):
    """Hash is the disambiguator that joins form to suffix — operators
    putting it inside the suffix value would produce double-hash refs."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-split-etymon",
            "old-english:gear",
            "--into",
            "suffix=fo#o|glosses=x",
            "--curation-file",
            str(tmp_path / "_curation.jsonl"),
        ],
    )
    assert result.exit_code != 0


def test_cli_rejects_suffix_with_whitespace(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-split-etymon",
            "old-english:gear",
            "--into",
            "suffix=foo bar|glosses=x",
            "--curation-file",
            str(tmp_path / "_curation.jsonl"),
        ],
    )
    assert result.exit_code != 0


def test_cli_accepts_valid_suffix_charset(tmp_path: Path):
    """[a-z0-9_-]+ is the allowed charset — hyphen, underscore,
    lowercase alphanumeric all OK."""
    runner = CliRunner()
    curation_file = tmp_path / "_curation.jsonl"
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-split-etymon",
            "old-english:gear",
            "--into",
            "suffix=foo_bar-2024|glosses=x|primary",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Item 1b — Applier-side invalid-suffix counter (JSONL hand-edit defense)
# ---------------------------------------------------------------------------


def test_applier_counts_invalid_suffix_children(tmp_path: Path):
    """Hand-edited JSONL with a suffix like `weir:bad` bypasses the CLI;
    the applier defends with the same charset check + a counter."""
    db_path = _build_db(tmp_path)
    _add_etymon_with_glosses(db_path, "old-english", "gear", ["a", "b"])
    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [
                {"suffix": "weir:bad", "glosses": ["a"], "primary": True},
                {"suffix": "year", "glosses": ["b"]},
            ]
        }
    }
    counts = apply_etymon_splits(db, state, apply=True)
    assert counts["children_skipped_invalid_suffix"] == 1
    assert counts["children_created"] == 1  # only 'year' survived
    db.close()


# ---------------------------------------------------------------------------
# Item 2 — format_unresolved_warnings + --strict CLI flag
# ---------------------------------------------------------------------------


def test_format_warnings_empty_when_clean():
    """No unresolved signals → empty string → operator sees nothing on
    stderr, exit 0 even under --strict."""
    result = {
        "curation": {"unresolved_etymon": 0, "unresolved_lemma_ref": 0},
        "gloss_suppressions": {"unresolved_etymon": 0, "unresolved_gloss": 0},
        "etymon_splits": {
            "unresolved_etymon": 0,
            "glosses_missing": 0,
            "children_skipped_invalid_suffix": 0,
        },
    }
    assert format_unresolved_warnings(result) == ""


def test_format_warnings_surfaces_each_section():
    """All three event types surface their own counters under named
    sections, so the operator can find the offending events."""
    result = {
        "curation": {"unresolved_etymon": 1, "unresolved_lemma_ref": 0},
        "gloss_suppressions": {"unresolved_etymon": 0, "unresolved_gloss": 2},
        "etymon_splits": {
            "unresolved_etymon": 0,
            "glosses_missing": 0,
            "tags_missing": 0,
            "children_skipped_invalid_suffix": 1,
        },
    }
    text = format_unresolved_warnings(result)
    assert "wyrd-van9" in text
    assert "curation.unresolved_etymon: 1" in text
    assert "gloss_suppressions.unresolved_gloss: 2" in text
    assert "etymon_splits.children_skipped_invalid_suffix: 1" in text


def test_format_warnings_render_order_follows_section_table():
    """Multiple simultaneously-flagged signals render in a fixed order —
    sections in `_UNRESOLVED_WARNING_SECTIONS` insertion order, counters in
    their declared tuple order — independent of the input dict's key order.
    Locks the ordering contract so a future table reorder can't silently
    change operator output (all other tests fire one section/counter)."""
    # etymon_splits intentionally lists glosses_missing before unresolved_etymon
    # in the INPUT; the render must still follow the table (unresolved_etymon first).
    result = {
        "curation": {"unresolved_etymon": 1, "unresolved_merge_ref": 2},
        "gloss_suppressions": {"unresolved_gloss": 3},
        "gloss_additions": {"unresolved_etymon": 4},
        "etymon_splits": {"glosses_missing": 5, "unresolved_etymon": 6},
    }
    text = format_unresolved_warnings(result)
    expected_order = [
        "curation.unresolved_etymon: 1",
        "curation.unresolved_merge_ref: 2",
        "gloss_suppressions.unresolved_gloss: 3",
        "gloss_additions.unresolved_etymon: 4",
        "etymon_splits.unresolved_etymon: 6",
        "etymon_splits.glosses_missing: 5",
    ]
    positions = [text.index(line) for line in expected_order]
    assert positions == sorted(positions), f"lines rendered out of order:\n{text}"


_ALL_COUNTER_NAMES = [
    ("curation", "unresolved_etymon"),
    ("curation", "unresolved_lemma_ref"),
    ("curation", "unresolved_merge_ref"),
    ("curation", "self_reference_lemma"),
    ("curation", "self_reference_merge"),
    ("gloss_suppressions", "unresolved_etymon"),
    ("gloss_suppressions", "unresolved_gloss"),
    # wyrd-wz82: etymon_gloss_add adds one warnings section with a
    # single counter (add is symmetric to suppress but has no per-gloss
    # resolution step, since duplicates surface as
    # ``additions_already_present`` rather than as an unresolved-ref
    # signal).
    ("gloss_additions", "unresolved_etymon"),
    ("etymon_splits", "unresolved_etymon"),
    ("etymon_splits", "glosses_missing"),
    ("etymon_splits", "tags_missing"),
    ("etymon_splits", "children_skipped_no_suffix"),
    ("etymon_splits", "children_skipped_invalid_suffix"),
    ("etymon_splits", "multiple_primary_collapsed"),
]


@pytest.mark.parametrize("section,counter", _ALL_COUNTER_NAMES)
def test_format_warnings_surfaces_every_counter(section, counter):
    """Round-2 follow-up (test-coverage P2): the helper iterates 13
    counters across 3 sections. Pinning each name individually means
    a typo in any one key (e.g. ``slef_reference_lemma``) is caught
    rather than silently never firing. The other 12 sections are zero-
    filled so the formatter sees only this counter as non-zero."""
    # Build a result dict with every counter zeroed, then set just
    # the target one to a non-zero value.
    by_section: dict[str, dict[str, int]] = {}
    for sec, name in _ALL_COUNTER_NAMES:
        by_section.setdefault(sec, {})[name] = 0
    by_section[section][counter] = 7
    text = format_unresolved_warnings(by_section)
    assert f"{section}.{counter}: 7" in text


def test_format_warnings_skips_absent_sections():
    """When a result dict has no curation / suppression / split sub-
    sections, the formatter doesn't crash and produces no warnings."""
    assert format_unresolved_warnings({}) == ""
    assert format_unresolved_warnings({"curation": None}) == ""


# ---------------------------------------------------------------------------
# Item 3 — load_parts defensive tolerance for missing meaning_db keys
# ---------------------------------------------------------------------------


def test_load_parts_skips_missing_keys_does_not_crash(caplog):
    """Pre-fix: load_parts would KeyError on a proportions usage with no
    Meaning in meaning_db, taking down the whole MeaningGenerator. Now
    it skips + counts + warns via the runtime logger."""
    import logging

    meaning_db = {"-known": []}  # No Meaning entries needed for the count test
    tag_db: dict[str, list[str]] = {}
    proportions = {"-known": 100, "-missing-from-bundle-": 50}
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="wyrd.generators.kenning.runtime.proportions"):
        # Constructing MeaningGenerator invokes load_parts(proportions)
        # internally; the missing key shouldn't raise.
        mg = MeaningGenerator(meaning_db, tag_db, proportions)
    assert mg is not None
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "wyrd-van9" in log_text
    assert "1 proportions" in log_text


def test_load_parts_no_warning_when_all_keys_present(caplog):
    """Clean pair: every proportions usage has a Meaning → no warning
    fires."""
    import logging

    meaning_db = {"-known": []}
    tag_db: dict[str, list[str]] = {}
    proportions = {"-known": 100}
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="wyrd.generators.kenning.runtime.proportions"):
        MeaningGenerator(meaning_db, tag_db, proportions)
    assert not any("wyrd-van9" in r.getMessage() for r in caplog.records)


def test_load_parts_partial_generators_usable_after_skip():
    """Round-2 follow-up (test-coverage P2): after a skip, the kept-key
    path should still produce a usable Generator. Pre-fix the whole
    MeaningGenerator was unusable on a single missing key; this test
    pins that we degrade gracefully and known keys keep working.

    Uses the live runtime's _load_meanings so the test isn't sensitive
    to the Meaning() constructor signature. Picks one real usage from
    the bundle as the 'known' key and a synthetic '-not-in-bundle-'
    string as the 'missing' key."""
    from wyrd.generators.kenning import _load_meanings

    meaning_db, tag_db = _load_meanings()
    # Pick the first usage in the bundle as the "known" key.
    known_usage = next(iter(meaning_db))
    # Mix a real proportions entry with a synthetic missing one.
    proportions = {known_usage: 100, "-synthetic-not-in-bundle-xyzzy": 50}
    mg = MeaningGenerator(meaning_db, tag_db, proportions)
    # The known key produced at least one generator bucket — the skip
    # didn't destroy the partial state.
    assert mg.generators, "kept-key generators should be non-empty after skip"
    # Confirm at least one bucket contains the known usage so a
    # downstream NameGenerator could draw from it.
    found = False
    for gen in mg.generators.values():
        if known_usage in gen.elements:
            found = True
            break
    assert found, f"known usage {known_usage!r} not in any post-skip bucket"


# ---------------------------------------------------------------------------
# CLI --strict integration
# ---------------------------------------------------------------------------


def test_cli_strict_exits_nonzero_on_unresolved(tmp_path: Path):
    """--strict + a curation event referencing a non-existent etymon →
    exit non-zero."""
    # Build a tiny DB; no etymons.
    db_path = _build_db(tmp_path)
    # Write a curation file with a ref that won't resolve.
    cur_file = tmp_path / "_curation.jsonl"
    rows = [
        {"_type": "source", "ref": "manual-curation", "title": "test"},
        {
            "_type": "etymon_curation",
            "ref": "old-english:nonexistent",
            "lemma_ref": "old-english:also-nonexistent",
        },
    ]
    cur_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    jsonl_dir = tmp_path / "mining"
    jsonl_dir.mkdir()
    (jsonl_dir / "_curation.jsonl").write_text(cur_file.read_text())
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "enrich",
            "--db",
            str(db_path),
            "--jsonl-dir",
            str(jsonl_dir),
            "--apply",
            "--strict",
        ],
    )
    assert result.exit_code != 0
    assert "wyrd-van9" in result.output or "unresolved" in result.output.lower()


def test_cli_no_strict_warns_but_exits_zero(tmp_path: Path):
    """Without --strict, unresolved signals warn on stderr but exit 0
    so interactive review isn't disrupted."""
    db_path = _build_db(tmp_path)
    rows = [
        {"_type": "source", "ref": "manual-curation", "title": "test"},
        {
            "_type": "etymon_curation",
            "ref": "old-english:nonexistent",
            "lemma_ref": "old-english:also-nonexistent",
        },
    ]
    jsonl_dir = tmp_path / "mining"
    jsonl_dir.mkdir()
    (jsonl_dir / "_curation.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "enrich",
            "--db",
            str(db_path),
            "--jsonl-dir",
            str(jsonl_dir),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "wyrd-van9" in result.output or "unresolved" in result.output.lower()
