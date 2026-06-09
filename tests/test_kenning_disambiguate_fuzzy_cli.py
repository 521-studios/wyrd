"""CLI-level tests for ``wyrd kenning lexicon disambiguate-fuzzy`` (D25).

The unit-level disambiguator pieces (``find_ambiguous_rows`` /
``disambiguate_one`` / ``apply_disambiguator_result``) are covered in
``test_kenning_disambiguator.py``. These pin the wyrd-8uvi-extracted command
orchestration (``_process_one_case`` / ``_DisambiguationRun``) end-to-end via
CliRunner against a seeded DB, with the LLM transport stubbed at the class
level — no network call.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli.lexicon.disambiguate_fuzzy import lexicon_disambiguate_fuzzy
from wyrd.generators.kenning.extractors.gemini import GeminiClient
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


def _seed_ambiguous(db_path: Path) -> tuple[int, int, int]:
    """A fuzzy row matched_form='heath' linked to the ON 'herath' etymon, with
    OE 'heath' also within edit distance 1 — so the row is ambiguous. Returns
    (heath_id, herath_id, row_id)."""
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.upsert_source(id="test_book", title="Test Book")
        heath_id = db.upsert_etymon("heath", "old-english")
        db.add_gloss(heath_id, "untilled land")
        herath_id = db.upsert_etymon("herath", "old-norse")
        db.add_gloss(herath_id, "district")
        db.commit()
        cur = db.conn.execute(
            "INSERT INTO etymon_text_match "
            "(etymon_id, source_id, matched_form, match_count, edit_distance, snippet, method) "
            "VALUES (?, 'test_book', 'heath', 1, 1, 'bare untilled land', 'fuzzy-search-v1')",
            (herath_id,),
        )
        row_id = cur.lastrowid
        db.commit()
    return heath_id, herath_id, row_id


@pytest.fixture
def _stub_pick_heath(monkeypatch):
    """Stub the LLM transport so every case resolves to the OE 'heath' etymon
    (id 1 in a freshly-seeded DB). Returned shape matches the disambiguator's
    responseSchema (choice / confidence / reason)."""
    monkeypatch.setattr(
        GeminiClient,
        "chat_json",
        lambda self, *a, **k: {
            "choice": "1",
            "confidence": "high",
            "reason": "phrase 'untilled land' matches OE heath",
        },
    )


def _row(db_path: Path, row_id: int) -> tuple:
    c = sqlite3.connect(db_path)
    row = c.execute(
        "SELECT etymon_id, method, disambiguator_reason FROM etymon_text_match WHERE id = ?",
        (row_id,),
    ).fetchone()
    c.close()
    return row


def test_disambiguate_fuzzy_dry_run_does_not_write(tmp_path: Path, _stub_pick_heath) -> None:
    db_path = tmp_path / "lex.db"
    _heath_id, herath_id, row_id = _seed_ambiguous(db_path)
    result = CliRunner().invoke(lexicon_disambiguate_fuzzy, ["--db", str(db_path)])
    assert result.exit_code == 0
    # Dry-run: the row is untouched — still linked to herath, still fuzzy-search.
    assert _row(db_path, row_id) == (herath_id, "fuzzy-search-v1", None)


def test_disambiguate_fuzzy_apply_reassigns(tmp_path: Path, _stub_pick_heath) -> None:
    db_path = tmp_path / "lex.db"
    heath_id, _herath_id, row_id = _seed_ambiguous(db_path)
    result = CliRunner().invoke(lexicon_disambiguate_fuzzy, ["--db", str(db_path), "--apply"])
    assert result.exit_code == 0
    # --apply: the row is reassigned to the chosen etymon, method + reason set.
    etymon_id, method, reason = _row(db_path, row_id)
    assert etymon_id == heath_id
    assert method == "llm-disambiguated-v1"
    assert "untilled land" in reason


def test_disambiguate_fuzzy_no_ambiguous_rows_is_noop(tmp_path: Path) -> None:
    db_path = tmp_path / "lex.db"
    init_schema(db_path)  # empty — no fuzzy rows
    result = CliRunner().invoke(lexicon_disambiguate_fuzzy, ["--db", str(db_path)])
    assert result.exit_code == 0


def test_disambiguate_fuzzy_agentic_requires_sources_dir(tmp_path: Path) -> None:
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    result = CliRunner().invoke(lexicon_disambiguate_fuzzy, ["--db", str(db_path), "--agentic"])
    assert result.exit_code != 0
    assert "--agentic requires --sources-dir" in result.output
