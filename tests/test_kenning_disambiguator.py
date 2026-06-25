"""Tests for the LLM disambiguator (wyrd-6z7, wyrd-uct).

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
- wyrd-uct: agentic expand-context loop (`disambiguate_one_agentic` +
  `SnippetExpander`).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.extractors.gemini import GeminiClient
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.disambiguator import (
    AmbiguityCase,
    Candidate,
    DisambiguatorResult,
    ExpansionStats,
    SnippetExpander,
    apply_disambiguator_result,
    disambiguate_one,
    disambiguate_one_agentic,
    find_ambiguous_rows,
)


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
    assert case.text_match_ids == (row_id,)
    assert case.matched_form == "heath"
    candidate_ids = {c.etymon_id for c in case.candidates}
    # Both etymons should be in the candidate set.
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        assert candidate_ids == {db._heath_id, db._herath_id}
        # Candidate ORDER is load-bearing (it's the order the disambiguator
        # lists them to the LLM) and tracks by_norm first-seen = etymon
        # insertion order: heath was upserted before herath.
        assert [c.etymon_id for c in case.candidates] == [db._heath_id, db._herath_id]


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
        source_id="test_book",
        matched_form="heath",
        snippet="bare untilled land",
        text_match_ids=(1,),
        current_etymon_ids=(99,),
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
        source_id="test_book",
        matched_form="foo",
        snippet="...",
        text_match_ids=(1,),
        current_etymon_ids=(1,),
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
        source_id="test_book",
        matched_form="foo",
        snippet="...",
        text_match_ids=(1,),
        current_etymon_ids=(1,),
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
        source_id="test_book",
        matched_form="foo",
        snippet="...",
        text_match_ids=(1,),
        current_etymon_ids=(1,),
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
            source_id="test_book",
            matched_form="heath",
            snippet="...",
            text_match_ids=(row_id,),
            current_etymon_ids=(db._heath_id,),
            candidates=(),  # not used by apply
        )
        result = DisambiguatorResult(
            chosen_etymon_id=db._heath_id, confidence="high", reason="phrase matches"
        )
        counts = apply_disambiguator_result(db, case, result)
        db.commit()

        row = db.conn.execute(
            "SELECT etymon_id, method, disambiguator_reason FROM etymon_text_match WHERE id = ?",
            (row_id,),
        ).fetchone()

    assert counts == {"kept": 1, "reassigned": 0, "deleted": 0}
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
            source_id="test_book",
            matched_form="heath",
            snippet="...",
            text_match_ids=(row_id,),
            current_etymon_ids=(db._herath_id,),
            candidates=(),
        )
        result = DisambiguatorResult(
            chosen_etymon_id=db._heath_id,
            confidence="high",
            reason="OE heath, not ON herath",
        )
        counts = apply_disambiguator_result(db, case, result)
        db.commit()

        row = db.conn.execute(
            "SELECT etymon_id, method FROM etymon_text_match WHERE id = ?",
            (row_id,),
        ).fetchone()

    assert counts == {"kept": 0, "reassigned": 1, "deleted": 0}
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
            source_id="test_book",
            matched_form="heath",
            snippet="...",
            text_match_ids=(wrong_row_id,),
            current_etymon_ids=(db._herath_id,),
            candidates=(),
        )
        result = DisambiguatorResult(
            chosen_etymon_id=db._heath_id,
            confidence="high",
            reason="OE heath, the exact-match row is correct",
        )
        counts = apply_disambiguator_result(db, case, result)
        db.commit()

        rows = db.conn.execute(
            "SELECT id, etymon_id, edit_distance, method, disambiguator_reason "
            "FROM etymon_text_match WHERE source_id = 'test_book' "
            "ORDER BY id"
        ).fetchall()

    assert counts == {"kept": 0, "reassigned": 1, "deleted": 0}
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
            source_id="test_book",
            matched_form="heath",
            snippet="...",
            text_match_ids=(row_id,),
            current_etymon_ids=(db._heath_id,),
            candidates=(),
        )
        result = DisambiguatorResult(
            chosen_etymon_id=None, confidence="low", reason="passage too vague"
        )
        counts = apply_disambiguator_result(db, case, result)
        db.commit()

        remaining = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_text_match WHERE id = ?", (row_id,)
        ).fetchone()[0]

    assert counts == {"kept": 0, "reassigned": 0, "deleted": 1}
    assert remaining == 0


def test_find_ambiguous_rows_groups_rows_sharing_source_and_form(fresh_db: Path) -> None:
    """wyrd-9ae regression: two fuzzy rows for the same body word in the
    same source — but attached to different etymons by fuzzy-search —
    must yield ONE AmbiguityCase, not two. Without this dedupe the
    disambiguator pays for two LLM calls answering the identical
    question."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        # Both rows describe 'heath' showing up in test_book — one
        # tagged against ON 'herath' (close fuzzy), one against OE
        # 'heath' but at edit_distance=1 to simulate two competing
        # fuzzy attributions. Use a third etymon for the second
        # attribution so both rows have distinct current_etymon_ids
        # but identical (source_id, matched_form).
        third_id = db.upsert_etymon("haethy", "old-english")
        db.add_gloss(third_id, "alt spelling")
        db.commit()
        row_a = _insert_fuzzy_row(
            db,
            etymon_id=db._herath_id,
            matched_form="heath",
            snippet="bare untilled land",
        )
        row_b = _insert_fuzzy_row(
            db,
            etymon_id=third_id,
            matched_form="heath",
            snippet="another snippet",
        )

        cases = find_ambiguous_rows(db, max_distance=1)

    assert len(cases) == 1, (
        "two fuzzy rows for the same (source, matched_form) must collapse "
        "to one AmbiguityCase — one LLM call serves both"
    )
    case = cases[0]
    # Deterministic order (rows queried ORDER BY m.id; row_a inserted first).
    assert case.text_match_ids == (row_a, row_b)
    assert set(case.current_etymon_ids) == {db._herath_id, third_id}


def test_apply_disambiguator_result_applies_verdict_to_every_row_in_group(
    fresh_db: Path,
) -> None:
    """wyrd-9ae regression: a group of rows under one ambiguity case must
    all receive the verdict consistently. If the LLM picks one row's
    current etymon, that row gets 'kept' and the other gets 'reassigned'."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        third_id = db.upsert_etymon("haethy", "old-english")
        db.add_gloss(third_id, "alt spelling")
        db.commit()
        row_a = _insert_fuzzy_row(
            db,
            etymon_id=db._herath_id,
            matched_form="heath",
            snippet="bare untilled land",
        )
        row_b = _insert_fuzzy_row(
            db,
            etymon_id=third_id,
            matched_form="heath",
            snippet="another snippet",
        )
        case = AmbiguityCase(
            source_id="test_book",
            matched_form="heath",
            snippet="bare untilled land",
            text_match_ids=(row_a, row_b),
            current_etymon_ids=(db._herath_id, third_id),
            candidates=(),
        )
        # LLM picks the OE 'heath' etymon — neither row currently points
        # there, so both are reassigned.
        result = DisambiguatorResult(
            chosen_etymon_id=db._heath_id, confidence="high", reason="OE heath"
        )
        counts = apply_disambiguator_result(db, case, result)
        db.commit()

        rows = db.conn.execute(
            "SELECT etymon_id, method FROM etymon_text_match "
            "WHERE source_id = 'test_book' AND matched_form = 'heath'"
        ).fetchall()

    # Both rows landed on the chosen etymon. Per D21 evidence-preservation
    # the second reassign collapsed onto the first (UNIQUE constraint),
    # so we end up with one row in the DB even though the case had two.
    assert counts["kept"] + counts["reassigned"] + counts["deleted"] == 2
    assert counts["reassigned"] == 2
    assert all(r["etymon_id"] == db._heath_id for r in rows)
    assert all(r["method"] == "llm-disambiguated-v1" for r in rows)


def test_apply_disambiguator_result_handles_mixed_kept_and_reassigned(
    fresh_db: Path,
) -> None:
    """wyrd-9ae corner case: in a 2-row group where the LLM picks one
    row's CURRENT etymon, that row is 'kept' and the other is
    'reassigned' onto the kept row — UNIQUE (etymon_id, source_id,
    matched_form) means the second row's reassign hits the kept row's
    slot, so the misattributed source row is deleted and the kept row
    absorbs the verdict per D21. The hardest path in the new code; pin
    the per-row counts so a regression that double-counted 'kept' or
    lost the source row would surface."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        third_id = db.upsert_etymon("haethy", "old-english")
        db.add_gloss(third_id, "alt spelling")
        db.commit()
        kept_row = _insert_fuzzy_row(
            db,
            etymon_id=db._heath_id,
            matched_form="heath",
            snippet="bare untilled land",
        )
        reassigned_row = _insert_fuzzy_row(
            db,
            etymon_id=third_id,
            matched_form="heath",
            snippet="another snippet",
        )
        case = AmbiguityCase(
            source_id="test_book",
            matched_form="heath",
            snippet="bare untilled land",
            text_match_ids=(kept_row, reassigned_row),
            current_etymon_ids=(db._heath_id, third_id),
            candidates=(),
        )
        # LLM picks the OE 'heath' etymon — first row is kept, second
        # row reassigns into the first's slot via the UNIQUE constraint.
        result = DisambiguatorResult(
            chosen_etymon_id=db._heath_id, confidence="high", reason="OE heath"
        )
        counts = apply_disambiguator_result(db, case, result)
        db.commit()

        rows = db.conn.execute(
            "SELECT id, etymon_id, method FROM etymon_text_match "
            "WHERE source_id = 'test_book' AND matched_form = 'heath' "
            "ORDER BY id"
        ).fetchall()

    assert counts == {"kept": 1, "reassigned": 1, "deleted": 0}
    # Only the originally-kept row remains (the second's reassign
    # collapsed onto it via UNIQUE constraint, source row deleted).
    assert len(rows) == 1
    assert rows[0]["id"] == kept_row
    assert rows[0]["etymon_id"] == db._heath_id
    assert rows[0]["method"] == "llm-disambiguated-v1"


def test_apply_disambiguator_result_deletes_every_row_in_group_on_none(
    fresh_db: Path,
) -> None:
    """wyrd-9ae: a 'none' verdict on a multi-row group must delete every
    row in the group. Pins the early-return delete loop so a future
    regression that only deleted the first row would fail loudly."""
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        third_id = db.upsert_etymon("haethy", "old-english")
        db.add_gloss(third_id, "alt spelling")
        db.commit()
        row_a = _insert_fuzzy_row(db, etymon_id=db._herath_id, matched_form="heath", snippet="...")
        row_b = _insert_fuzzy_row(db, etymon_id=third_id, matched_form="heath", snippet="...")
        case = AmbiguityCase(
            source_id="test_book",
            matched_form="heath",
            snippet="...",
            text_match_ids=(row_a, row_b),
            current_etymon_ids=(db._herath_id, third_id),
            candidates=(),
        )
        result = DisambiguatorResult(
            chosen_etymon_id=None, confidence="low", reason="passage too vague"
        )
        counts = apply_disambiguator_result(db, case, result)
        db.commit()

        remaining = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_text_match "
            "WHERE matched_form = 'heath' AND source_id = 'test_book'"
        ).fetchone()[0]

    assert counts == {"kept": 0, "reassigned": 0, "deleted": 2}
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


# --- wyrd-uct: agentic expand-context loop --------------------------------


def _stub_chat_json_sequence(monkeypatch, responses: list[dict]) -> list[dict]:
    """Replace `GeminiClient.chat_json` with a deterministic queue: each
    call pops the next response. Returns the same list so the test can
    assert how many calls were consumed (``len(initial) - len(returned)``).

    Tests use this to script the agentic loop: e.g. [expand, expand,
    answer] feeds three turns of the conversation.
    """
    queue = list(responses)

    def _stub(self, *args, **kw):  # noqa: ARG001 — drop-in for chat_json
        if not queue:
            raise AssertionError("agentic loop made more LLM calls than scripted")
        return queue.pop(0)

    monkeypatch.setattr(GeminiClient, "chat_json", _stub)
    return queue


# Toy body fixture for the expander tests. The 200-char window around
# 'heath' (positions ~210-215) initially shows only "the heath here".
# The next ~80 chars contain "an upland district" — the cue that
# disambiguates herath (ON, district) vs heath (OE, untilled land).
_LEAD = "x" * 200
_PASSAGE = (
    f"{_LEAD}the heath here, where shepherds graze. "
    "Geographically this is an upland district, well above the river meadows below."
)


def _heath_case(
    heath_id: int = 42, herath_id: int = 99, *, snippet: str | None = None
) -> AmbiguityCase:
    """Build a synthetic herath/heath ambiguity case. The default snippet
    is the 'first 200 chars' window — narrow enough that the model
    should request expansion."""
    if snippet is None:
        # 100 chars before, 100 chars after — same shape as the
        # production _TEXT_MATCH_SNIPPET_RADIUS would produce.
        idx = _PASSAGE.find("heath")
        start = max(0, idx - 100)
        end = min(len(_PASSAGE), idx + len("heath") + 100)
        snippet = _PASSAGE[start:end].strip().replace("heath", "«heath»", 1)
    return AmbiguityCase(
        source_id="test_book",
        matched_form="heath",
        snippet=snippet,
        text_match_ids=(1,),
        current_etymon_ids=(herath_id,),
        candidates=(
            Candidate(
                etymon_id=heath_id,
                canonical_form="heath",
                language="old-english",
                glosses=("untilled land",),
                tags=(),
            ),
            Candidate(
                etymon_id=herath_id,
                canonical_form="herath",
                language="old-norse",
                glosses=("district",),
                tags=(),
            ),
        ),
    )


def test_snippet_expander_returns_marked_snippet() -> None:
    """`make_snippet` finds the first occurrence of the matched form
    and returns a body slice with the form bracketed for highlight."""
    expander = SnippetExpander.from_in_memory({"test_book": _PASSAGE})
    snippet = expander.make_snippet("test_book", "heath", radius_before=20, radius_after=20)
    assert snippet is not None
    assert "«heath»" in snippet
    # The radius-bounded slice should be tight — the bracketed token
    # plus ~40 chars of context (20 before + 20 after).
    assert len(snippet) <= len("«heath»") + 50  # +5 fudge for trim/replace edges


def test_snippet_expander_marks_boundary_occurrence_not_substring() -> None:
    """Regression: the highlight marker must wrap the SAME ``\\b``-anchored
    occurrence the window is built around — not the first plain-substring match.
    When the form is also a substring of an earlier longer word (``heath`` inside
    ``heathen``), a ``snippet.replace(norm, ...)`` marked the wrong word, feeding a
    misleading anchor to the LLM disambiguator (whose job is etymon attribution).
    Now marked by position."""
    body = "the heathen folk lived near the heath today and more"
    expander = SnippetExpander.from_in_memory({"b": body})
    snippet = expander.make_snippet("b", "heath", radius_before=40, radius_after=20)
    assert snippet is not None
    assert "the «heath» today" in snippet  # the standalone occurrence
    assert "«heath»en" not in snippet  # NOT the substring inside "heathen"
    assert "heathen" in snippet  # the longer word survives unmarked
    # A form that appears ONLY as a substring of a longer word has no \b-anchored
    # occurrence → None (the marker is never misapplied).
    only_sub = SnippetExpander.from_in_memory({"b": "only heathen here, no standalone"})
    assert only_sub.make_snippet("b", "heath", radius_before=40, radius_after=20) is None


def test_snippet_expander_returns_none_for_missing_source() -> None:
    """Missing source_id → None, so the orchestrator can short-circuit
    to a forced commit instead of looping forever."""
    expander = SnippetExpander.from_in_memory({"other_book": "..."})
    assert expander.make_snippet("test_book", "heath", radius_before=20, radius_after=20) is None


def test_snippet_expander_returns_none_when_form_not_found() -> None:
    """Form not present in body → None. Guards against fuzzy matches
    where the OCR-normalized body diverges from the matched form."""
    expander = SnippetExpander.from_in_memory({"test_book": "no such word here"})
    assert expander.make_snippet("test_book", "zzqxx", radius_before=20, radius_after=20) is None


def test_snippet_expander_uses_word_boundary_anchor() -> None:
    """Anchor parity with fuzzy/reverse-search: the expander must
    locate the standalone form, not a substring inside a longer word.
    Without ``\\b`` the expander would anchor on 'heath' inside
    'heathen' (idx 0) and widen around the wrong position; with the
    boundary it skips to the standalone 'heath' (idx ~14)."""
    body = "the heathens lived where heath rolled to the sea"
    expander = SnippetExpander.from_in_memory({"test_book": body})
    snippet = expander.make_snippet("test_book", "heath", radius_before=10, radius_after=10)
    assert snippet is not None
    # The bracketed form sits in the standalone-'heath' window, well
    # past the leading 'heathens'.
    assert "«heath»" in snippet
    assert "heathens" not in snippet


def test_snippet_expander_widens_around_same_position() -> None:
    """A second `make_snippet` with a larger radius produces a strict
    superset (after stripping markers) — the orchestrator relies on
    this to grow the context monotonically."""
    expander = SnippetExpander.from_in_memory({"test_book": _PASSAGE})
    narrow = expander.make_snippet("test_book", "heath", radius_before=20, radius_after=20)
    wide = expander.make_snippet("test_book", "heath", radius_before=80, radius_after=80)
    assert narrow is not None and wide is not None
    # Strip the marker wrappers so the substring relation holds.
    narrow_plain = narrow.replace("«", "").replace("»", "")
    wide_plain = wide.replace("«", "").replace("»", "")
    assert narrow_plain in wide_plain
    assert len(wide) > len(narrow)


def test_disambiguate_one_agentic_commits_immediately(monkeypatch) -> None:
    """When the model answers on the first turn, the loop returns
    that verdict with zero expansions and no forced_commit flag — so
    the cost path matches the original `disambiguate_one`."""
    case = _heath_case(heath_id=42)
    expander = SnippetExpander.from_in_memory({"test_book": _PASSAGE})

    queue = _stub_chat_json_sequence(
        monkeypatch,
        [
            {
                "action": "answer",
                "choice": "42",
                "confidence": "high",
                "reason": "phrase clearly describes untilled land",
            }
        ],
    )
    result, stats = disambiguate_one_agentic(GeminiClient(api_key="fake"), case, expander)
    assert result.chosen_etymon_id == 42
    assert result.confidence == "high"
    assert stats.expansions == 0
    assert stats.forced_commit is False
    assert queue == []  # exactly one LLM call


def test_disambiguate_one_agentic_resolves_via_expansion(monkeypatch) -> None:
    """Acceptance criterion (wyrd-uct):

    The first 200 chars are genuinely ambiguous between herath ('district')
    and heath ('untilled land'). The next 300+ chars contain 'upland
    district' — the cue that resolves it. The model requests expansion,
    sees the resolving phrase, and commits to herath (id 99).

    Verifies: the loop honors the expansion, widens the snippet, and
    arrives at the right answer.
    """
    case = _heath_case(heath_id=42, herath_id=99)
    expander = SnippetExpander.from_in_memory({"test_book": _PASSAGE})

    queue = _stub_chat_json_sequence(
        monkeypatch,
        [
            # Turn 1: model thinks the snippet is too narrow.
            {
                "action": "expand",
                "direction": "after",
                "chars": 300,
                "reason": "need more context to choose between district and untilled land",
            },
            # Turn 2: with the wider snippet visible, model commits to ON herath.
            {
                "action": "answer",
                "choice": "99",
                "confidence": "high",
                "reason": "the phrase 'upland district' anchors ON herath, not OE heath",
            },
        ],
    )
    result, stats = disambiguate_one_agentic(GeminiClient(api_key="fake"), case, expander)
    assert result.chosen_etymon_id == 99
    assert result.confidence == "high"
    assert "district" in result.reason.lower()
    assert stats.expansions == 1
    assert stats.forced_commit is False
    assert queue == []  # both scripted calls consumed


def test_disambiguate_one_agentic_forces_commit_at_max_expansions(monkeypatch) -> None:
    """If the model keeps requesting expansions past `max_expansions`,
    the orchestrator must force a commit on the final round. The model
    in this test is stubbed to ALWAYS request expansion — the harness
    sees the FINAL ROUND directive in the prompt and is expected to
    answer; here we instead simulate a model that keeps asking, which
    means the loop must label the case forced_commit=True and bail out
    with chosen_etymon_id=None rather than spinning forever."""
    case = _heath_case()
    expander = SnippetExpander.from_in_memory({"test_book": _PASSAGE})

    # 4 expansion responses scripted (3 honored expansions + 1 final
    # round on which the model still says 'expand'). Loop should stop
    # after consuming the 4th call and return a 'none' verdict.
    queue = _stub_chat_json_sequence(
        monkeypatch,
        [
            {"action": "expand", "direction": "both", "chars": 100, "reason": "more"}
            for _ in range(4)
        ],
    )
    result, stats = disambiguate_one_agentic(
        GeminiClient(api_key="fake"), case, expander, max_expansions=3
    )
    assert result.chosen_etymon_id is None
    assert result.confidence == "low"
    assert stats.forced_commit is True
    assert queue == []  # exactly 4 LLM calls consumed (3 honored + final force)


def test_disambiguate_one_agentic_short_circuits_when_source_missing(monkeypatch) -> None:
    """If the expander can't load the source body, the loop must skip
    ahead to the final round on the next iteration so the model is
    asked to commit on the (still-original) snippet rather than
    re-prompting forever."""
    case = _heath_case()
    # Empty in-memory expander — the matched_form won't be findable.
    empty_expander = SnippetExpander.from_in_memory({})

    queue = _stub_chat_json_sequence(
        monkeypatch,
        [
            # Turn 1: model asks to expand — expander fails, orchestrator
            # advances expansions to max so the next turn forces commit.
            {"action": "expand", "direction": "after", "chars": 200},
            # Turn 2 (final round): model still asks to expand → 'none'.
            {"action": "expand", "direction": "after", "chars": 200},
        ],
    )
    result, stats = disambiguate_one_agentic(
        GeminiClient(api_key="fake"), case, empty_expander, max_expansions=3
    )
    assert result.chosen_etymon_id is None
    assert stats.forced_commit is True
    # 2 calls: original "expand" + 1 forced final round (no honored
    # widening happened because the expander returned None on turn 1).
    assert queue == []


def test_disambiguate_one_agentic_caps_total_chars(monkeypatch) -> None:
    """When an expansion would push the snippet past `total_char_cap`,
    the orchestrator truncates and skips ahead to the final round
    rather than feeding the model an oversized prompt."""
    case = _heath_case()
    # Generate a body large enough that radius_before/after expansions
    # exceed the cap quickly.
    big_body = "a " * 5000 + "the heath here. " + "z " * 5000
    expander = SnippetExpander.from_in_memory({"test_book": big_body})

    queue = _stub_chat_json_sequence(
        monkeypatch,
        [
            # Turn 1: model asks for a huge expansion — orchestrator
            # truncates to cap and forces the next iteration to commit.
            {"action": "expand", "direction": "both", "chars": 500},
            # Turn 2 (forced commit): model finally answers.
            {
                "action": "answer",
                "choice": "42",
                "confidence": "medium",
                "reason": "best read on the wider context",
            },
        ],
    )
    result, stats = disambiguate_one_agentic(
        GeminiClient(api_key="fake"),
        case,
        expander,
        max_expansions=5,
        total_char_cap=500,
    )
    assert result.chosen_etymon_id == 42
    assert stats.final_chars <= 500
    # Cap forced an early-commit branch — only the two scripted calls
    # ran, even though max_expansions=5 in principle allowed more.
    assert queue == []
    # The wider snippet was discarded (would have garbled the «...»
    # marker on byte-truncate), so the model committed on the original
    # snippet under the FINAL ROUND directive — that's a forced commit.
    assert stats.forced_commit is True
    # Zero honored expansions — every widening was rejected by the cap.
    assert stats.expansions == 0


def test_disambiguate_one_agentic_treats_unknown_action_as_none(monkeypatch) -> None:
    """If the model returns an action outside {answer, expand}, the
    orchestrator must NOT mis-apply — it returns 'none' with low
    confidence, mirroring `disambiguate_one`'s defensive parsing."""
    case = _heath_case()
    expander = SnippetExpander.from_in_memory({"test_book": _PASSAGE})

    _stub_chat_json_sequence(
        monkeypatch,
        [
            {"action": "wat", "choice": "42", "confidence": "high", "reason": "?"},
        ],
    )
    result, stats = disambiguate_one_agentic(GeminiClient(api_key="fake"), case, expander)
    assert result.chosen_etymon_id is None
    assert result.confidence == "low"
    assert stats.forced_commit is True


def test_disambiguate_one_agentic_validates_chosen_id(monkeypatch) -> None:
    """An answer with an etymon id outside the candidate set is treated
    as 'none' (same rule as `disambiguate_one`). Pins the validation
    branch in `_parse_answer`."""
    case = _heath_case(heath_id=42, herath_id=99)
    expander = SnippetExpander.from_in_memory({"test_book": _PASSAGE})

    _stub_chat_json_sequence(
        monkeypatch,
        [
            {
                "action": "answer",
                "choice": "9999",
                "confidence": "high",
                "reason": "wat",
            },
        ],
    )
    result, _stats = disambiguate_one_agentic(GeminiClient(api_key="fake"), case, expander)
    assert result.chosen_etymon_id is None
    assert result.confidence == "low"
    assert "9999" in result.reason


def test_disambiguate_one_agentic_handles_missing_expand_fields(monkeypatch) -> None:
    """A model that responds action='expand' without direction/chars
    must still be honored — the orchestrator falls back to default
    direction='both' + sensible chars rather than crashing."""
    case = _heath_case(heath_id=42)
    expander = SnippetExpander.from_in_memory({"test_book": _PASSAGE})

    queue = _stub_chat_json_sequence(
        monkeypatch,
        [
            # Turn 1: expand with no direction/chars — defaults apply.
            {"action": "expand"},
            # Turn 2: commit.
            {
                "action": "answer",
                "choice": "42",
                "confidence": "medium",
                "reason": "ok",
            },
        ],
    )
    result, stats = disambiguate_one_agentic(GeminiClient(api_key="fake"), case, expander)
    assert result.chosen_etymon_id == 42
    assert stats.expansions == 1
    assert queue == []


def test_expansion_stats_is_immutable() -> None:
    """`ExpansionStats` is frozen so it can be hashed / used in sets
    when downstream telemetry aggregates per-case stats."""
    s = ExpansionStats(expansions=2, final_chars=800, forced_commit=False)
    with pytest.raises(FrozenInstanceError):
        s.expansions = 5  # type: ignore[misc]


def test_lexicon_disambiguate_fuzzy_cli_agentic_requires_sources_dir(
    fresh_db: Path,
) -> None:
    """The CLI must reject ``--agentic`` without ``--sources-dir`` —
    expansion needs a body of source text to draw from. Without the
    guard, the agentic loop would short-circuit on every case and add
    cost without value."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "disambiguate-fuzzy", "--db", str(fresh_db), "--agentic"],
    )
    combined = result.output + (result.stderr or "")
    # click.UsageError exits with code 2.
    assert result.exit_code != 0
    assert "sources-dir" in combined.lower() or "agentic" in combined.lower()


def test_lexicon_disambiguate_fuzzy_cli_agentic_end_to_end(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """End-to-end: ``--agentic`` with ``--sources-dir`` runs the
    agentic loop. The model first asks to expand, then commits to the
    correct etymon. The CLI applies the verdict and reports agentic
    stats showing one expansion."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    # Body must contain 'heath' so the expander can locate and widen.
    (sources_dir / "test_book.txt").write_text(_PASSAGE)

    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        _insert_fuzzy_row(
            db,
            etymon_id=db._herath_id,
            matched_form="heath",
            snippet="bare untilled land",
        )
    with LexiconDB(fresh_db) as db:
        _seed_minimum(db)
        chosen_id = db._heath_id

    queue = _stub_chat_json_sequence(
        monkeypatch,
        [
            {
                "action": "expand",
                "direction": "after",
                "chars": 200,
                "reason": "need more context",
            },
            {
                "action": "answer",
                "choice": str(chosen_id),
                "confidence": "high",
                "reason": "untilled land matches OE heath after expansion",
            },
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "disambiguate-fuzzy",
            "--db",
            str(fresh_db),
            "--agentic",
            "--sources-dir",
            str(sources_dir),
            "--apply",
        ],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    # Agentic stats line surfaces in the summary.
    assert "Agentic stats" in combined
    assert "expansions=1" in combined
    # Both LLM calls consumed.
    assert queue == []

    # Verdict was applied to the row.
    with LexiconDB(fresh_db) as db:
        row = db.conn.execute(
            "SELECT etymon_id, method FROM etymon_text_match WHERE matched_form = 'heath'"
        ).fetchone()
    assert row is not None
    assert row["etymon_id"] == chosen_id
    assert row["method"] == "llm-disambiguated-v1"
