"""Tests for _select_review_candidates (wyrd-eca).

Covers the candidate-query filter that decides which already-mined
toponym_etymology rows are eligible for Tier-2 LLM review. The default
filter targets Ollama/Qwen-tagged rows; the wyrd-eca extension adds an
opt-in `include_haiku=True` mode for non-Celtic books that were mined
Tier-1 with Haiku (Bannister 1916, Harrison 1898, etc.) and would
otherwise be invisible to the Gemini Tier-2 pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.cli import _select_review_candidates
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


@pytest.fixture
def review_db(tmp_path: Path) -> Path:
    """Seed a fresh DB with toponym_etymology rows covering every
    provider tag we care about: ollama / qwen, anthropic-haiku,
    anthropic-sonnet (a Tier-2-style row), and gemini (also Tier-2).

    Each row writes its source_id and a unique modern_name so the
    candidate query can be inspected by toponym name."""
    db_path = tmp_path / "review.db"
    init_schema(db_path)
    rows = [
        # (modern_name, source_id, notes_provider_tag)
        ("Stockport", "mawer_1920_northumberland_durham", "extracted_by:ollama:qwen3"),
        ("Wigan", "mawer_1920_northumberland_durham", "extracted_by:llm:qwen"),
        ("Bristol", "ekwall_1922_lancashire", "extracted_by:llama"),
        # Haiku-tagged rows (Bannister + Harrison content shape)
        (
            "Hereford",
            "bannister_1916_herefordshire",
            "extracted_by:anthropic:claude-haiku-4-5-20251001",
        ),
        (
            "Liverpool",
            "harrison_1898_liverpool",
            "extracted_by:anthropic:claude-haiku-4-5-20251001",
        ),
        # Sonnet-tagged Tier-3 row — should never come back as a Tier-2
        # candidate even when --include-haiku is set, because its notes
        # don't mention 'haiku'.
        (
            "Eboracum",
            "smith_1928_north_riding_yorkshire",
            "extracted_by:anthropic:claude-sonnet-4-6",
        ),
        # Gemini-tagged row — already Tier-2'd, should be excluded by
        # the NOT EXISTS guard once a Gemini provider_tag is requested.
        ("Glasgow", "johnston_1892_place_names_of_scotland", "extracted_by:gemini:flash-2-5"),
    ]
    with LexiconDB(db_path) as db:
        for source_id in {r[1] for r in rows}:
            db.conn.execute("INSERT INTO source (id, title) VALUES (?, ?)", (source_id, source_id))
        for modern_name, source_id, notes in rows:
            cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", (modern_name,))
            db.conn.execute(
                "INSERT INTO toponym_etymology "
                "(toponym_id, source_id, historical_form, confidence, notes) "
                "VALUES (?, ?, ?, 'low', ?)",
                (cur.lastrowid, source_id, modern_name + "-tun", notes),
            )
        db.commit()
    return db_path


def _picked_names(rows) -> set[str]:
    return {r["modern_name"] for r in rows}


def test_default_picks_ollama_qwen_rows(review_db: Path):
    """Default behavior (no --include-haiku): Ollama/Qwen/llama-tagged
    rows are eligible Tier-2 candidates. The Haiku-tagged rows are NOT
    picked up — that's the gap wyrd-eca exists to address."""
    rows = _select_review_candidates(
        db_path=review_db,
        provider_tag="extracted_by:gemini:",
        confidence=("low",),
        book=None,
        limit=None,
    )
    picked = _picked_names(rows)
    # Ollama / Qwen / llama all match the original OR clauses.
    assert "Stockport" in picked
    assert "Wigan" in picked
    assert "Bristol" in picked
    # Haiku rows excluded by default.
    assert "Hereford" not in picked
    assert "Liverpool" not in picked
    # Sonnet excluded (no haiku token in notes).
    assert "Eboracum" not in picked
    # Gemini excluded by NOT EXISTS guard against the chosen
    # provider_tag (gemini).
    assert "Glasgow" not in picked


def test_include_haiku_widens_to_haiku_rows(review_db: Path):
    """With include_haiku=True, the candidate set widens to also include
    rows tagged extracted_by:anthropic:...haiku.... The original
    Ollama/Qwen rows still come through; new Haiku-tagged rows
    additionally surface."""
    rows = _select_review_candidates(
        db_path=review_db,
        provider_tag="extracted_by:gemini:",
        confidence=("low",),
        book=None,
        limit=None,
        include_haiku=True,
    )
    picked = _picked_names(rows)
    # Original Ollama/Qwen rows still match.
    assert "Stockport" in picked
    assert "Wigan" in picked
    assert "Bristol" in picked
    # Haiku rows now included — the wyrd-eca extension.
    assert "Hereford" in picked
    assert "Liverpool" in picked
    # Sonnet still excluded (notes don't mention haiku).
    assert "Eboracum" not in picked
    # Gemini still excluded by NOT EXISTS.
    assert "Glasgow" not in picked


def test_include_haiku_still_excludes_already_tier2_reviewed_rows(review_db: Path, tmp_path: Path):
    """Idempotency: a Haiku row that ALREADY has a Gemini Tier-2 review
    must NOT be picked up even when --include-haiku is set. Pin the
    NOT EXISTS guard against the chosen provider_tag.

    Mutates the fixture: adds a Gemini-tagged sibling row for the
    Hereford toponym so the NOT EXISTS clause fires."""
    with LexiconDB(review_db) as db:
        topo_id = db.conn.execute(
            "SELECT id FROM toponym WHERE modern_name = ?", ("Hereford",)
        ).fetchone()[0]
        db.conn.execute(
            "INSERT INTO toponym_etymology "
            "(toponym_id, source_id, historical_form, confidence, notes) "
            "VALUES (?, 'bannister_1916_herefordshire', 'Hereford-tun', 'high', "
            "'extracted_by:gemini:flash-2-5')",
            (topo_id,),
        )
        db.commit()

    rows = _select_review_candidates(
        db_path=review_db,
        provider_tag="extracted_by:gemini:",
        confidence=("low",),
        book=None,
        limit=None,
        include_haiku=True,
    )
    picked = _picked_names(rows)
    # Hereford has been Tier-2'd already — should NOT come back as a
    # candidate for re-review.
    assert "Hereford" not in picked
    # Liverpool still uncovered — should still come through.
    assert "Liverpool" in picked


def test_include_haiku_book_filter_scopes_to_one_source(review_db: Path):
    """The --book scope works with --include-haiku: pass a Haiku-mined
    source_id and the result includes only that book's Haiku rows.
    Pinning this matches the ticket's recommended invocation pattern:
    `... review --book bannister_1916_herefordshire --include-haiku`."""
    rows = _select_review_candidates(
        db_path=review_db,
        provider_tag="extracted_by:gemini:",
        confidence=("low",),
        book="bannister_1916_herefordshire",
        limit=None,
        include_haiku=True,
    )
    picked = _picked_names(rows)
    assert picked == {"Hereford"}


# --- wyrd-0rsk: --include-sonnet companion to --include-haiku --------------


def test_default_excludes_sonnet_rows(review_db: Path):
    """Default behavior (no --include-sonnet): Sonnet-Tier-1-tagged
    rows are NOT picked up. The Eboracum row in the fixture is Sonnet-
    tagged; it must NOT surface as a Tier-2 candidate when neither
    include_haiku nor include_sonnet is set."""
    rows = _select_review_candidates(
        db_path=review_db,
        provider_tag="extracted_by:gemini:",
        confidence=("low",),
        book=None,
        limit=None,
    )
    assert "Eboracum" not in _picked_names(rows)


def test_include_sonnet_widens_to_sonnet_rows(review_db: Path):
    """With include_sonnet=True the candidate set widens to include
    Sonnet-Tier-1-tagged rows. Mirror of
    test_include_haiku_widens_to_haiku_rows."""
    rows = _select_review_candidates(
        db_path=review_db,
        provider_tag="extracted_by:gemini:",
        confidence=("low",),
        book=None,
        limit=None,
        include_sonnet=True,
    )
    picked = _picked_names(rows)
    # Original Ollama/Qwen rows still match.
    assert "Stockport" in picked
    # Sonnet row now included — the wyrd-0rsk extension.
    assert "Eboracum" in picked
    # Haiku rows still excluded (different flag).
    assert "Hereford" not in picked


def test_include_haiku_and_sonnet_compose(review_db: Path):
    """include_haiku and include_sonnet are orthogonal — setting both
    catches Anthropic-tagged rows from either model. Useful when a
    pipeline doesn't care which Anthropic model wrote the row."""
    rows = _select_review_candidates(
        db_path=review_db,
        provider_tag="extracted_by:gemini:",
        confidence=("low",),
        book=None,
        limit=None,
        include_haiku=True,
        include_sonnet=True,
    )
    picked = _picked_names(rows)
    assert "Stockport" in picked  # Qwen
    assert "Hereford" in picked  # Haiku
    assert "Eboracum" in picked  # Sonnet
    assert "Glasgow" not in picked  # Gemini, NOT EXISTS guard


def test_include_sonnet_book_filter_scopes_to_one_source(review_db: Path):
    """The --book scope works with --include-sonnet, matching the
    documented invocation pattern for re-reviewing a Sonnet-mined book."""
    rows = _select_review_candidates(
        db_path=review_db,
        provider_tag="extracted_by:gemini:",
        confidence=("low",),
        book="smith_1928_north_riding_yorkshire",
        limit=None,
        include_sonnet=True,
    )
    assert _picked_names(rows) == {"Eboracum"}


def test_include_sonnet_still_excludes_already_tier2_reviewed_rows(review_db: Path):
    """Idempotency: a Sonnet row that ALREADY has a Gemini Tier-2
    review must NOT be picked up even when --include-sonnet is set."""
    with LexiconDB(review_db) as db:
        topo_id = db.conn.execute(
            "SELECT id FROM toponym WHERE modern_name = ?", ("Eboracum",)
        ).fetchone()[0]
        db.conn.execute(
            "INSERT INTO toponym_etymology "
            "(toponym_id, source_id, historical_form, confidence, notes) "
            "VALUES (?, 'smith_1928_north_riding_yorkshire', 'Eboracum-tun', 'high', "
            "'extracted_by:gemini:flash-2-5')",
            (topo_id,),
        )
        db.commit()

    rows = _select_review_candidates(
        db_path=review_db,
        provider_tag="extracted_by:gemini:",
        confidence=("low",),
        book=None,
        limit=None,
        include_sonnet=True,
    )
    assert "Eboracum" not in _picked_names(rows)


def test_default_off_means_haiku_only_book_yields_zero_candidates(review_db: Path):
    """A book that was mined ONLY with Haiku (no Ollama/Qwen rows)
    yields zero Tier-2 candidates under the default filter — the
    "silent slip" symptom wyrd-eca was filed against. With
    include_haiku=False (default), Bannister has no candidates."""
    rows = _select_review_candidates(
        db_path=review_db,
        provider_tag="extracted_by:gemini:",
        confidence=("low",),
        book="bannister_1916_herefordshire",
        limit=None,
    )
    assert _picked_names(rows) == set()
