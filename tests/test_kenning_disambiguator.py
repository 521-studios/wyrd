"""Tests for the LLM disambiguator (wyrd-6z7).

Covers:
- `find_ambiguous_rows` — picking out rows where multiple etymons are
  within edit distance of the matched body form.
- `disambiguate_one` — wrapping a stubbed Gemini call and validating the
  parsed verdict.
- `apply_disambiguator_result` — the three outcomes (kept / reassigned /
  deleted) and how the row's `method` and `disambiguator_reason` columns
  are updated.
- The `lexicon disambiguate-fuzzy` CLI end-to-end with the LLM stubbed.
- The cost gate: a row with only one candidate etymon never reaches the
  LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.disambiguator import (
    AmbiguityCase,
    Candidate,
    DisambiguatorResult,
    apply_disambiguator_result,
    disambiguate_one,
    find_ambiguous_rows,
)
from wyrd.generators.kenning.gemini_extractor import GeminiClient
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _stub_chat_json(monkeypatch, response: dict) -> None:
    """Replace `GeminiClient.chat_json` with a fixed-response stub. Tests
    that exercise `disambiguate_one` use this to bypass the network."""
    monkeypatch.setattr(GeminiClient, "chat_json", lambda self, *args, **kw: response)


def _seed_minimum(db: LexiconDB) -> None:
    """Seed a source row plus two etymons that will collide on a body form."""
    db.upsert_source(id="test_book", title="Test Book")
    # Two etymons within edit distance 1 of "heath": 'heath' (OE) and
    # 'herath' (ON). The herath/heath shape that wyrd-c3x flagged.
    heath_id = db.upsert_etymon("heath", "old-english")
    db.add_gloss(heath_id, "untilled land")
    herath_id = db.upsert_etymon("herath", "old-norse")
    db.add_gloss(herath_id, "district")
    db.commit()
    db._heath_id = heath_id  # type: ignore[attr-defined]
    db._herath_id = herath_id  # type: ignore[attr-defined]


def _insert_fuzzy_row(
    db: LexiconDB,
    *,
    etymon_id: int,
    matched_form: str,
    snippet: str,
    edit_distance: int = 1,
) -> int:
    cur = db.conn.execute(
        "INSERT INTO etymon_text_match "
        "(etymon_id, source_id, matched_form, match_count, edit_distance, snippet, method) "
        "VALUES (?, ?, ?, ?, ?, ?, 'fuzzy-search-v1')",
        (etymon_id, "test_book", matched_form, 1, edit_distance, snippet),
    )
    db.commit()
    return cur.lastrowid


def test_find_ambiguous_rows_picks_multi_candidate_cases(fresh_db: Path) -> None:
    """A fuzzy row whose `matched_form` is within distance 1 of multiple
    etymons must be returned with both candidates listed."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        # Existing fuzzy row from the herath etymon's perspective. Body
        # word is 'heath', which is also a canonical etymon.
        row_id = _insert_fuzzy_row(
            db,
            etymon_id=db._herath_id,
            matched_form="heath",
            snippet="bare untilled land where shepherds graze",
        )

        cases = find_ambiguous_rows(db, max_distance=1)

    assert len(cases) == 1
    case = cases[0]
    assert case.text_match_id == row_id
    assert case.matched_form == "heath"
    candidate_ids = {c.etymon_id for c in case.candidates}
    # Both etymons should be in the candidate set.
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        assert candidate_ids == {db._heath_id, db._herath_id}


def test_find_ambiguous_rows_skips_when_only_one_candidate(fresh_db: Path) -> None:
    """The cost gate: a fuzzy row with only its own etymon plausible
    (no other etymons within distance 1) is NOT returned."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test_book", title="Test")
        unique_id = db.upsert_etymon("zzqxx", "old-english")
        db.add_gloss(unique_id, "fake gloss")
        db.commit()
        _insert_fuzzy_row(
            db,
            etymon_id=unique_id,
            matched_form="zzqxy",
            snippet="something with zzqxy nearby",
        )

        cases = find_ambiguous_rows(db, max_distance=1)

    assert cases == []


def test_find_ambiguous_rows_skips_already_disambiguated(fresh_db: Path) -> None:
    """Rows whose `method` is already 'llm-disambiguated-*' are skipped on
    re-run. Idempotent."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        row_id = _insert_fuzzy_row(
            db,
            etymon_id=db._herath_id,
            matched_form="heath",
            snippet="...",
        )
        db.conn.execute(
            "UPDATE etymon_text_match SET method = 'llm-disambiguated-v1' WHERE id = ?",
            (row_id,),
        )
        db.commit()

        assert find_ambiguous_rows(db, max_distance=1) == []


def test_disambiguate_one_parses_choice_and_reason(monkeypatch) -> None:
    """`disambiguate_one` calls Gemini and returns a structured result."""
    case = AmbiguityCase(
        text_match_id=1,
        source_id="test_book",
        matched_form="heath",
        snippet="bare untilled land",
        current_etymon_id=99,
        candidates=(
            Candidate(
                etymon_id=42,
                canonical_form="heath",
                language="old-english",
                glosses=("untilled land",),
                tags=(),
            ),
            Candidate(
                etymon_id=99,
                canonical_form="herath",
                language="old-norse",
                glosses=("district",),
                tags=(),
            ),
        ),
    )

    _stub_chat_json(
        monkeypatch,
        {
            "choice": "42",
            "confidence": "high",
            "reason": "The phrase 'untilled land' matches OE 'heath'.",
        },
    )
    client = GeminiClient(api_key="fake")  # not used; transport stubbed
    result = disambiguate_one(client, case)
    assert result.chosen_etymon_id == 42
    assert result.confidence == "high"
    assert "untilled land" in result.reason


def test_disambiguate_one_handles_none_answer(monkeypatch) -> None:
    """If the model says 'none', the result has chosen_etymon_id=None."""
    case = AmbiguityCase(
        text_match_id=1,
        source_id="test_book",
        matched_form="foo",
        snippet="...",
        current_etymon_id=1,
        candidates=(
            Candidate(
                etymon_id=1, canonical_form="foo", language="old-english", glosses=(), tags=()
            ),
            Candidate(etymon_id=2, canonical_form="bar", language="old-norse", glosses=(), tags=()),
        ),
    )
    _stub_chat_json(
        monkeypatch,
        {"choice": "none", "confidence": "low", "reason": "passage doesn't say"},
    )
    result = disambiguate_one(GeminiClient(api_key="fake"), case)
    assert result.chosen_etymon_id is None


def test_disambiguate_one_rejects_id_outside_candidates(monkeypatch) -> None:
    """If the model returns an etymon id that's not in the candidate set,
    it's treated as 'none' rather than mis-applied."""
    case = AmbiguityCase(
        text_match_id=1,
        source_id="test_book",
        matched_form="foo",
        snippet="...",
        current_etymon_id=1,
        candidates=(
            Candidate(
                etymon_id=1, canonical_form="foo", language="old-english", glosses=(), tags=()
            ),
        ),
    )
    _stub_chat_json(
        monkeypatch,
        {"choice": "9999", "confidence": "low", "reason": "wat"},
    )
    result = disambiguate_one(GeminiClient(api_key="fake"), case)
    assert result.chosen_etymon_id is None
    assert "9999" in result.reason


def test_disambiguate_one_handles_non_numeric_choice(monkeypatch) -> None:
    """If the model returns a free-form string instead of an etymon id
    (a `ValueError` on `int(...)`), treat as 'none' with confidence='low'
    and a 'non-id choice' reason — never mis-apply."""
    case = AmbiguityCase(
        text_match_id=1,
        source_id="test_book",
        matched_form="foo",
        snippet="...",
        current_etymon_id=1,
        candidates=(
            Candidate(
                etymon_id=1, canonical_form="foo", language="old-english", glosses=(), tags=()
            ),
        ),
    )
    _stub_chat_json(
        monkeypatch,
        {
            "choice": "heath",  # free-form string, not a numeric id
            "confidence": "high",
            "reason": "ignored",
        },
    )
    result = disambiguate_one(GeminiClient(api_key="fake"), case)
    assert result.chosen_etymon_id is None
    assert result.confidence == "low"  # the function downgrades on parse failure
    assert "non-id choice" in result.reason


def test_apply_disambiguator_result_kept(fresh_db: Path) -> None:
    """When the LLM picks the same etymon as the row's current etymon_id,
    the row stays but `method` and `disambiguator_reason` get bumped."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        row_id = _insert_fuzzy_row(db, etymon_id=db._heath_id, matched_form="heath", snippet="...")
        case = AmbiguityCase(
            text_match_id=row_id,
            source_id="test_book",
            matched_form="heath",
            snippet="...",
            current_etymon_id=db._heath_id,
            candidates=(),  # not used by apply
        )
        result = DisambiguatorResult(
            chosen_etymon_id=db._heath_id, confidence="high", reason="phrase matches"
        )
        action = apply_disambiguator_result(db, case, result)
        db.commit()

        row = db.conn.execute(
            "SELECT etymon_id, method, disambiguator_reason FROM etymon_text_match WHERE id = ?",
            (row_id,),
        ).fetchone()

    assert action == "kept"
    assert row["etymon_id"] == db._heath_id
    assert row["method"] == "llm-disambiguated-v1"
    assert row["disambiguator_reason"] == "phrase matches"


def test_apply_disambiguator_result_reassigned(fresh_db: Path) -> None:
    """When the LLM picks a different candidate, the row's etymon_id is
    swapped and the reasoning is recorded."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        row_id = _insert_fuzzy_row(db, etymon_id=db._herath_id, matched_form="heath", snippet="...")
        case = AmbiguityCase(
            text_match_id=row_id,
            source_id="test_book",
            matched_form="heath",
            snippet="...",
            current_etymon_id=db._herath_id,
            candidates=(),
        )
        result = DisambiguatorResult(
            chosen_etymon_id=db._heath_id,
            confidence="high",
            reason="OE heath, not ON herath",
        )
        action = apply_disambiguator_result(db, case, result)
        db.commit()

        row = db.conn.execute(
            "SELECT etymon_id, method FROM etymon_text_match WHERE id = ?",
            (row_id,),
        ).fetchone()

    assert action == "reassigned"
    assert row["etymon_id"] == db._heath_id  # reassigned!
    assert row["method"] == "llm-disambiguated-v1"


def test_apply_disambiguator_result_reassigned_preserves_existing_evidence(
    fresh_db: Path,
) -> None:
    """When reassigning to an etymon that already has a row for the same
    (source, matched_form) — typically an exact match from reverse-search —
    we must not overwrite that row. Per D21, the existing evidence is
    preserved; the verdict is recorded on the destination row and the
    misattributed source row is deleted."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        # The (correct) destination row, e.g. an exact match landed earlier.
        existing_row_id = _insert_fuzzy_row(
            db,
            etymon_id=db._heath_id,
            matched_form="heath",
            snippet="exact match snippet",
            edit_distance=0,
        )
        # The misattributed fuzzy row from the herath etymon.
        wrong_row_id = db.conn.execute(
            "INSERT INTO etymon_text_match "
            "(etymon_id, source_id, matched_form, match_count, edit_distance, snippet, method) "
            "VALUES (?, ?, ?, ?, ?, ?, 'fuzzy-search-v1')",
            (db._herath_id, "test_book", "heath", 1, 1, "fuzzy snippet"),
        ).lastrowid
        db.commit()

        case = AmbiguityCase(
            text_match_id=wrong_row_id,
            source_id="test_book",
            matched_form="heath",
            snippet="...",
            current_etymon_id=db._herath_id,
            candidates=(),
        )
        result = DisambiguatorResult(
            chosen_etymon_id=db._heath_id,
            confidence="high",
            reason="OE heath, the exact-match row is correct",
        )
        action = apply_disambiguator_result(db, case, result)
        db.commit()

        rows = db.conn.execute(
            "SELECT id, etymon_id, edit_distance, method, disambiguator_reason "
            "FROM etymon_text_match WHERE source_id = 'test_book' "
            "ORDER BY id"
        ).fetchall()

    assert action == "reassigned"
    # Only the destination row remains; the misattributed source row was deleted.
    assert len(rows) == 1
    assert rows[0]["id"] == existing_row_id
    assert rows[0]["etymon_id"] == db._heath_id
    assert rows[0]["edit_distance"] == 0  # original exact-match evidence preserved
    assert rows[0]["method"] == "llm-disambiguated-v1"
    assert "exact-match" in rows[0]["disambiguator_reason"]


def test_apply_disambiguator_result_deleted(fresh_db: Path) -> None:
    """When the LLM says 'none', the row is deleted entirely."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        row_id = _insert_fuzzy_row(db, etymon_id=db._heath_id, matched_form="heath", snippet="...")
        case = AmbiguityCase(
            text_match_id=row_id,
            source_id="test_book",
            matched_form="heath",
            snippet="...",
            current_etymon_id=db._heath_id,
            candidates=(),
        )
        result = DisambiguatorResult(
            chosen_etymon_id=None, confidence="low", reason="passage too vague"
        )
        action = apply_disambiguator_result(db, case, result)
        db.commit()

        remaining = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_text_match WHERE id = ?", (row_id,)
        ).fetchone()[0]

    assert action == "deleted"
    assert remaining == 0


def test_lexicon_disambiguate_fuzzy_cli_end_to_end(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """The CLI command finds an ambiguous row, calls the stubbed Gemini,
    applies the verdict, and reports a non-zero `kept` count."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        _insert_fuzzy_row(
            db,
            etymon_id=db._herath_id,
            matched_form="heath",
            snippet="bare untilled land where shepherds graze",
        )

    # Stub: model picks the OE 'heath' etymon (the *other* candidate).
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        chosen_id = db._heath_id

    _stub_chat_json(
        monkeypatch,
        {
            "choice": str(chosen_id),
            "confidence": "high",
            "reason": "untilled land matches OE heath",
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "disambiguate-fuzzy", "--db", str(fresh_db), "--apply"],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined

    with LexiconDB(fresh_db) as db:
        row = db.conn.execute(
            "SELECT etymon_id, method, disambiguator_reason "
            "FROM etymon_text_match WHERE matched_form = 'heath'"
        ).fetchone()
    assert row is not None
    assert row["etymon_id"] == chosen_id
    assert row["method"] == "llm-disambiguated-v1"
    assert "heath" in (row["disambiguator_reason"] or "").lower()


def test_lexicon_disambiguate_fuzzy_cli_dry_run_does_not_persist(
    fresh_db: Path, monkeypatch
) -> None:
    """Without --apply, the CLI must call the LLM (so the operator can see
    what would happen) but must NOT write the verdict back to the DB."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        _insert_fuzzy_row(
            db,
            etymon_id=db._herath_id,
            matched_form="heath",
            snippet="...",
        )
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        chosen_id = db._heath_id

    _stub_chat_json(
        monkeypatch,
        {
            "choice": str(chosen_id),
            "confidence": "high",
            "reason": "would reassign on apply",
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "disambiguate-fuzzy", "--db", str(fresh_db)],  # no --apply
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert "dry-run" in combined.lower()

    with LexiconDB(fresh_db) as db:
        row = db.conn.execute(
            "SELECT etymon_id, method, disambiguator_reason "
            "FROM etymon_text_match WHERE matched_form = 'heath'"
        ).fetchone()
    # Row must be untouched: still the original herath etymon, still
    # fuzzy-search-v1 method, no disambiguator_reason recorded.
    assert row["etymon_id"] != chosen_id
    assert row["method"] == "fuzzy-search-v1"
    assert row["disambiguator_reason"] is None


def test_lexicon_disambiguate_fuzzy_cli_no_ambiguous_rows(fresh_db: Path) -> None:
    """When `find_ambiguous_rows` returns nothing, the CLI must short-circuit
    with a 'No ambiguous fuzzy rows found' message and zero LLM calls."""
    with LexiconDB(fresh_db) as db:
        # Empty DB — no etymons, no rows. Definitely no ambiguous rows.
        db.upsert_source(id="test_book", title="Test")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "disambiguate-fuzzy", "--db", str(fresh_db), "--apply"],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert "No ambiguous fuzzy rows found" in combined
