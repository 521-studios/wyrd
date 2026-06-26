"""Tests for the wyrd-ami fantasy-name etymology pipeline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from wyrd.generators.kenning import fantasy_pipeline as fp
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    """Lexicon DB with the schema created and migrated."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        migrate_schema(db)
        db.commit()
    return db_path


def _seed_etymon(
    db: LexiconDB,
    *,
    canonical_form: str,
    language: str,
    gloss: str | None = None,
) -> int:
    cur = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        (canonical_form, language),
    )
    eid = cur.lastrowid
    if gloss:
        db.conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (eid, gloss),
        )
    return eid


def _seed_descent(
    db: LexiconDB,
    *,
    parent_id: int,
    child_id: int,
    edge_type: str = "inheritance",
    source_id: str = "test",
) -> None:
    db.conn.execute(
        "INSERT OR IGNORE INTO source (id, title, author) VALUES (?, 'Test', 'Test')",
        (source_id,),
    )
    db.conn.execute(
        """INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence)
           VALUES (?, ?, ?, ?, 'high')""",
        (parent_id, child_id, edge_type, source_id),
    )


# ---------------------------------------------------------------------
# descent_walking_lookup
# ---------------------------------------------------------------------


def test_descent_walking_lookup_finds_greek_ancestor_for_modern_form(fresh_db: Path) -> None:
    """The harpy case: input matches modern-english harpy; lookup walks
    one descent edge back to ancient-greek ἅρπυια and returns it."""
    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(
            db, canonical_form="ἅρπυια", language="ancient-greek", gloss="snatcher"
        )
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "harpy")
    assert len(matches) == 1
    assert matches[0].canonical_form == "ἅρπυια"
    assert matches[0].language == "ancient-greek"
    assert matches[0].edge_type == "inheritance"


def test_descent_walking_lookup_returns_direct_hit_with_priority(fresh_db: Path) -> None:
    """When the input form itself is an approved-family etymon (e.g.
    'wyrm' as OE), it shows up as a direct hit BEFORE any ancestor
    surfaced via descent."""
    with LexiconDB(fresh_db) as db:
        oe_id = _seed_etymon(db, canonical_form="wyrm", language="old-english", gloss="serpent")
        modern_id = _seed_etymon(db, canonical_form="wyrm", language="modern-english")
        _seed_descent(db, parent_id=oe_id, child_id=modern_id)
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "wyrm")
    # Direct OE hit comes first.
    assert matches[0].language == "old-english"
    assert matches[0].edge_type == "(direct)"


def test_descent_walking_lookup_no_match_returns_empty(fresh_db: Path) -> None:
    """No etymon matches the input form → empty list, no exception."""
    with LexiconDB(fresh_db) as db:
        # Seed something unrelated so the table isn't empty.
        _seed_etymon(db, canonical_form="bera", language="old-english")
        db.commit()

    assert fp.descent_walking_lookup(fresh_db, "gnoll") == []


def _seed_variant(
    db: LexiconDB,
    *,
    etymon_id: int,
    form: str,
    variant_class: str = "alternative",
    source_id: str = "test",
) -> None:
    """Insert one etymon_variant row (helper for the wyrd-fqil tests)."""
    db.conn.execute(
        "INSERT OR IGNORE INTO source (id, title, author) VALUES (?, 'Test', 'Test')",
        (source_id,),
    )
    db.conn.execute(
        "INSERT INTO etymon_variant (etymon_id, form, variant_class, source_id) "
        "VALUES (?, ?, ?, ?)",
        (etymon_id, form, variant_class, source_id),
    )


def test_descent_walking_lookup_resolves_via_alt_form_variant(fresh_db: Path) -> None:
    """The Chaucer-`harpie` case: input matches an etymon_variant row;
    its parent etymon (modern-english `harpy`) seeds the BFS, and the
    walk reaches ancient-greek ἅρπυια via the descent edge.

    modern-english itself isn't in APPROVED_LANGUAGES so it doesn't
    surface as a direct-via-variant hit; what matters is that the
    variant seeds the descent walk into the approved-family ancestor."""
    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(db, canonical_form="ἅρπυια", language="ancient-greek")
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        # Wiktextract-style alt-form: 'harpie' is recorded as a variant
        # of modern-english `harpy`, not as its own etymon row.
        _seed_variant(db, etymon_id=modern_id, form="harpie", variant_class="alternative")
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "harpie")
    # Greek is reachable via variant → modern-english → descent edge → Greek.
    by_lang = {m.language: m for m in matches}
    assert "ancient-greek" in by_lang
    assert by_lang["ancient-greek"].canonical_form == "ἅρπυια"


def test_descent_walking_lookup_variant_to_approved_lang_marked_via_variant(
    fresh_db: Path,
) -> None:
    """When the variant's parent etymon IS in an approved language, the
    seed match is surfaced with edge_type='(direct-via-variant)'."""
    with LexiconDB(fresh_db) as db:
        # OE 'cot' is approved; 'cotan' is its genitive inflection
        # recorded as an etymon_variant row, not a separate lemma.
        oe_id = _seed_etymon(db, canonical_form="cot", language="old-english")
        _seed_variant(db, etymon_id=oe_id, form="cotan", variant_class="inflection")
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "cotan")
    assert len(matches) == 1
    assert matches[0].language == "old-english"
    assert matches[0].canonical_form == "cot"
    assert matches[0].edge_type == "(direct-via-variant)"


def test_descent_walking_lookup_variant_match_is_case_insensitive(fresh_db: Path) -> None:
    """COLLATE NOCASE on etymon_variant.form: 'COTAN' matches the same
    row as 'cotan' regardless of casing."""
    with LexiconDB(fresh_db) as db:
        oe_id = _seed_etymon(db, canonical_form="cot", language="old-english")
        _seed_variant(db, etymon_id=oe_id, form="cotan", variant_class="inflection")
        db.commit()

    for query in ("COTAN", "Cotan", "cotan", "cOtAn"):
        matches = fp.descent_walking_lookup(fresh_db, query)
        assert any(m.canonical_form == "cot" for m in matches), f"failed for {query!r}"


def test_descent_walking_lookup_lemma_match_takes_precedence_over_variant(fresh_db: Path) -> None:
    """If the input form matches BOTH a lemma row AND a variant row
    (different etymons), the lemma match comes first as `(direct)` and
    the variant match comes second as `(direct-via-variant)`."""
    with LexiconDB(fresh_db) as db:
        # The input form is itself a lemma in OE.
        _seed_etymon(db, canonical_form="word", language="old-english")
        # AND it's a variant of a different etymon (e.g. an alternative
        # spelling of some ME word). Synthetic but exercises the
        # ordering invariant.
        me_id = _seed_etymon(db, canonical_form="weord", language="middle-english")
        _seed_variant(db, etymon_id=me_id, form="word", variant_class="alternative")
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "word")
    assert len(matches) >= 2
    # Direct lemma match first
    assert matches[0].language == "old-english"
    assert matches[0].edge_type == "(direct)"
    # Then variant-mediated match
    assert matches[1].language == "middle-english"
    assert matches[1].edge_type == "(direct-via-variant)"


def test_descent_walking_lookup_variant_only_match_works(fresh_db: Path) -> None:
    """When the input form has NO direct lemma match but IS in
    etymon_variant, the variant alone seeds the BFS."""
    with LexiconDB(fresh_db) as db:
        oe_id = _seed_etymon(db, canonical_form="cot", language="old-english")
        # 'cotan' is a genitive inflection; not a separate lemma.
        _seed_variant(db, etymon_id=oe_id, form="cotan", variant_class="inflection")
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "cotan")
    assert len(matches) == 1
    assert matches[0].canonical_form == "cot"
    assert matches[0].language == "old-english"
    assert matches[0].edge_type == "(direct-via-variant)"


def test_descent_walking_lookup_skips_unapproved_languages(fresh_db: Path) -> None:
    """Ancestors whose language isn't in the approved set are walked
    THROUGH but not returned — Proto-Finnic ancestors should not
    appear, but should still let us reach an OE ancestor beyond them."""
    with LexiconDB(fresh_db) as db:
        oe_id = _seed_etymon(db, canonical_form="root-form", language="old-english")
        finnic_id = _seed_etymon(db, canonical_form="*intermediate", language="urj-fin-pro")
        modern_id = _seed_etymon(db, canonical_form="testword", language="modern-english")
        _seed_descent(db, parent_id=finnic_id, child_id=modern_id)
        _seed_descent(db, parent_id=oe_id, child_id=finnic_id)
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "testword")
    # Should reach OE through the Finnic intermediate.
    langs = {m.language for m in matches}
    assert "old-english" in langs
    assert "urj-fin-pro" not in langs


# ---------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------


def test_resolve_via_pre_filter_with_semantic_match(fresh_db: Path) -> None:
    """When pre-filter resolves AND semantic check returns 'match',
    accept without calling full LLM research."""
    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(db, canonical_form="ἅρπυια", language="ancient-greek")
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        db.commit()

    def _llm_full_must_not_be_called(name, description):
        raise AssertionError("Full LLM research should not be called when semantic check passes")

    def _stub_semantic(name, description, ancestor, glosses):
        return {"verdict": "match", "reasoning": "Greek harpyiai match the creature"}

    res = fp.resolve(
        fresh_db,
        "harpy",
        "winged Greek monster",
        llm_caller=_llm_full_must_not_be_called,
        semantic_check_caller=_stub_semantic,
    )
    assert res.usable is True
    assert res.etymon_id == greek_id
    assert res.resolution_method == "descent_lookup"
    assert res.confidence == "high"
    assert res.bar_reason is None


def test_resolve_homograph_collision_bars_with_homograph_reason(fresh_db: Path) -> None:
    """The Cloaker case: pre-filter resolves to a real ancestor, but
    semantic check rejects it as a homograph collision. Full research
    then bars it as modern_coinage; we override that to the more
    specific homograph_collision since pre-filter DID find a form match."""
    with LexiconDB(fresh_db) as db:
        ofr_id = _seed_etymon(db, canonical_form="cloque", language="old-french", gloss="bell-cape")
        modern_id = _seed_etymon(db, canonical_form="cloaker", language="modern-english")
        _seed_descent(db, parent_id=ofr_id, child_id=modern_id)
        db.commit()

    def _stub_semantic(name, description, ancestor, glosses):
        return {"verdict": "mismatch", "reasoning": "1981 Gygax coinage"}

    def _stub_full(name, description):
        return {
            "attested_in": "outside_family",
            "historical_form": None,
            "gloss": None,
            "citation": None,
            "confidence": "high",
            "bar_reason": "modern_coinage",
            "reasoning": "D&D 1981",
        }

    res = fp.resolve(
        fresh_db,
        "Cloaker",
        "A D&D monster that disguises itself as a cloak",
        llm_caller=_stub_full,
        semantic_check_caller=_stub_semantic,
    )
    assert res.usable is False
    # Override of full-research's modern_coinage → homograph_collision, since
    # pre-filter DID find a form-matched ancestor.
    assert res.bar_reason == fp.BAR_REASON_HOMOGRAPH


def test_resolve_skip_llm_accepts_pre_filter_blindly(fresh_db: Path) -> None:
    """skip_llm=True bypasses semantic verification; pre-filter wins
    even if the result is a homograph. This is the legacy behavior used
    by tests and offline coverage estimation."""
    with LexiconDB(fresh_db) as db:
        ofr_id = _seed_etymon(db, canonical_form="cloque", language="old-french")
        modern_id = _seed_etymon(db, canonical_form="cloaker", language="modern-english")
        _seed_descent(db, parent_id=ofr_id, child_id=modern_id)
        db.commit()

    res = fp.resolve(fresh_db, "Cloaker", "D&D monster", skip_llm=True)
    # Pre-filter trusted; no semantic verification.
    assert res.usable is True
    assert res.etymon_id == ofr_id
    assert "not semantically verified" in (res.reasoning or "")


def test_resolve_skip_llm_bars_pre_filter_misses(fresh_db: Path) -> None:
    """With skip_llm=True, pre-filter misses are barred without an LLM call."""
    res = fp.resolve(fresh_db, "gnoll", "hyena humanoid", skip_llm=True)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_NO_ETYMOLOGY
    assert res.resolution_method == "descent_lookup"


def test_resolve_via_llm_normalizes_attested_in_case_and_separators(fresh_db: Path) -> None:
    """The LLM frequently returns 'Latin' / 'Old French' / 'old_french'
    instead of the dashed-lowercase 'latin' / 'old-french' that
    APPROVED_LANGUAGES uses. The pipeline must normalize at the boundary
    so capitalization doesn't cause a usable etymology to mis-classify
    as outside_language_family. Pinned regression for the 2026-05-06
    pfsrd2 mining run that lost ~80 morphemes to this bug."""
    with LexiconDB(fresh_db) as db:
        target_id = _seed_etymon(db, canonical_form="harpyia", language="latin")
        db.commit()

    for raw_lang in ("Latin", "LATIN", " latin ", "latin"):

        def _stub_llm(name, description, _raw_lang=raw_lang):
            return {
                "attested_in": _raw_lang,
                "historical_form": "harpyia",
                "gloss": None,
                "citation": "Etymonline",
                "confidence": "high",
                "bar_reason": None,
                "reasoning": "stub",
            }

        res = fp.resolve(fresh_db, "Harpy", "Greek monster", llm_caller=_stub_llm)
        assert res.usable is True, f"failed for {raw_lang!r}"
        assert res.etymon_id == target_id, f"failed for {raw_lang!r}"


def test_resolve_via_llm_normalizes_space_separated_language_names(fresh_db: Path) -> None:
    """Multi-word language names from the LLM ('Old French', 'Old
    Norse') normalize to dashed form to match APPROVED_LANGUAGES."""
    with LexiconDB(fresh_db) as db:
        target_id = _seed_etymon(db, canonical_form="harpie", language="old-french")
        db.commit()

    def _stub_llm(name, description):
        return {
            "attested_in": "Old French",  # space-separated, capitalized
            "historical_form": "harpie",
            "gloss": None,
            "citation": "Etymonline",
            "confidence": "high",
            "bar_reason": None,
            "reasoning": "stub",
        }

    res = fp.resolve(fresh_db, "Harpie", "Greek monster", llm_caller=_stub_llm)
    assert res.usable is True
    assert res.etymon_id == target_id


def test_resolve_via_llm_when_pre_filter_misses(fresh_db: Path) -> None:
    """Pre-filter misses → LLM is called. When the LLM-named (form, lang)
    pair exists in the etymon table, the resolution is usable and
    anchored to that etymon's id."""
    with LexiconDB(fresh_db) as db:
        target_id = _seed_etymon(
            db, canonical_form="ἅρπυια", language="ancient-greek", gloss="snatcher"
        )
        db.commit()

    def _stub_llm(name, description):
        assert name == "Harpy"
        return {
            "attested_in": "ancient-greek",
            "historical_form": "ἅρπυια",
            "gloss": "snatcher",
            "citation": "LSJ",
            "confidence": "high",
            "bar_reason": None,
            "reasoning": "stub reasoning",
        }

    res = fp.resolve(fresh_db, "Harpy", "Greek monster", llm_caller=_stub_llm)
    assert res.usable is True
    assert res.etymon_id == target_id
    assert res.resolution_method == "llm_full_research"
    assert res.citation == "LSJ"


def test_resolve_via_llm_attested_but_not_in_corpus_bars(fresh_db: Path) -> None:
    """When the LLM identifies a real attested form in an approved language
    but no matching etymon row exists, bar with attested_but_not_in_corpus
    rather than returning usable=True with etymon_id=None — that would
    let unmineable morphemes leak through to generation."""

    # Note: NO etymon seeded for the LLM-named (form, language).
    def _stub_llm(name, description):
        return {
            "attested_in": "old-english",
            "historical_form": "rare-real-word",
            "gloss": "real meaning",
            "citation": "Bosworth-Toller",
            "confidence": "high",
            "bar_reason": None,
            "reasoning": "OE attested but not in our corpus",
        }

    res = fp.resolve(fresh_db, "RareCreature", "test", llm_caller=_stub_llm)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_NOT_IN_CORPUS
    # Records the LLM finding so a future backfill can ingest it.
    assert res.unapproved_language == "old-english"
    assert res.unapproved_form == "rare-real-word"


def test_resolve_via_llm_low_confidence_is_barred(fresh_db: Path) -> None:
    """Per the user's conservative-default directive: low confidence
    LLM answers get barred as uncertain_attestation, even when the LLM
    names a real language."""

    def _stub_llm(name, description):
        return {
            "attested_in": "old-english",
            "historical_form": "*maybe-something",
            "gloss": None,
            "citation": None,
            "confidence": "low",
            "bar_reason": None,
            "reasoning": "I'm not sure",
        }

    res = fp.resolve(fresh_db, "Mystery", "unknown creature", llm_caller=_stub_llm)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_UNCERTAIN


def test_resolve_via_llm_unapproved_language_records_lang_and_form(fresh_db: Path) -> None:
    """When the LLM finds a real etymology in a language NOT in
    APPROVED_LANGUAGES, the row is barred but records what language and
    historical form so we can build a 'languages to consider approving'
    report. Per user 2026-05-05."""

    def _stub_llm(name, description):
        return {
            # Japanese is intentionally NOT in APPROVED_LANGUAGES per
            # user 2026-05-06 ('no credible English-speaker intuition
            # for Japanese-derived fantasy creature names yet'). When
            # the LLM returns it, the pipeline records the language
            # for future-prioritization signal.
            "attested_in": "japanese",
            "historical_form": "kappa",
            "gloss": "river demon",
            "citation": "Kojien",
            "confidence": "high",
            "bar_reason": None,
            "reasoning": "Japanese folkloric creature",
        }

    res = fp.resolve(fresh_db, "Kappa", "Japanese water imp", llm_caller=_stub_llm)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_OUTSIDE_FAMILY
    assert res.unapproved_language == "japanese"
    assert res.unapproved_form == "kappa"


def test_write_resolution_persists_unapproved_language_and_form(fresh_db: Path) -> None:
    """The unapproved_language + unapproved_form fields round-trip through
    the DB so a later query can aggregate languages-to-approve."""
    res = fp.Resolution(
        usable=False,
        etymon_id=None,
        bar_reason=fp.BAR_REASON_OUTSIDE_FAMILY,
        resolution_method="llm_full_research",
        confidence="high",
        citation="Monier-Williams",
        reasoning="Sanskrit demon",
        unapproved_language="sanskrit",
        unapproved_form="rakṣasa",
    )
    with LexiconDB(fresh_db) as db:
        fp.write_resolution(
            db.conn, input_name="Rakshasa", input_description="demon", resolution=res
        )
        db.commit()
    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM fantasy_morpheme WHERE input_name='Rakshasa'").fetchone()
    assert row["unapproved_language"] == "sanskrit"
    assert row["unapproved_form"] == "rakṣasa"
    assert row["bar_reason"] == fp.BAR_REASON_OUTSIDE_FAMILY


def test_resolve_via_llm_modern_coinage_is_barred(fresh_db: Path) -> None:
    """Dunsany's Gnoll: LLM correctly identifies as outside the family."""

    def _stub_llm(name, description):
        return {
            "attested_in": "outside_family",
            "historical_form": None,
            "gloss": None,
            "citation": None,
            "confidence": "high",
            "bar_reason": "modern_coinage",
            "reasoning": "Dunsany 1912",
        }

    res = fp.resolve(fresh_db, "Gnoll", "hyena humanoid", llm_caller=_stub_llm)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_MODERN_COINAGE


def test_resolve_semantic_check_failure_falls_through_to_full_research(fresh_db: Path) -> None:
    """If the semantic-check LLM call raises (network/JSON failure),
    that candidate is skipped and we fall through to full research
    rather than crashing."""
    with LexiconDB(fresh_db) as db:
        ofr_id = _seed_etymon(db, canonical_form="cloque", language="old-french")
        modern_id = _seed_etymon(db, canonical_form="cloaker", language="modern-english")
        _seed_descent(db, parent_id=ofr_id, child_id=modern_id)
        db.commit()

    def _broken_semantic(name, description, ancestor, glosses):
        raise RuntimeError("simulated semantic-check network failure")

    def _stub_full(name, description):
        return {
            "attested_in": "modern_coinage",
            "historical_form": None,
            "gloss": None,
            "citation": None,
            "confidence": "high",
            "bar_reason": "modern_coinage",
            "reasoning": "D&D 1981",
        }

    res = fp.resolve(
        fresh_db,
        "Cloaker",
        "D&D monster",
        llm_caller=_stub_full,
        semantic_check_caller=_broken_semantic,
    )
    # Semantic check failed → skipped → full research bars as
    # modern_coinage → upgraded to homograph_collision since
    # pre-filter DID find a form match.
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_HOMOGRAPH


def test_resolve_via_llm_modern_coinage_no_bar_reason_inferred(fresh_db: Path) -> None:
    """When the LLM returns attested_in='modern_coinage' but omits
    bar_reason, _resolve_via_llm infers BAR_REASON_MODERN_COINAGE
    rather than collapsing to no_etymology_found."""

    def _stub_llm(name, description):
        return {
            "attested_in": "modern_coinage",
            "historical_form": None,
            "gloss": None,
            "citation": None,
            "confidence": "high",
            # bar_reason intentionally omitted
            "reasoning": "Tolkien coinage",
        }

    res = fp.resolve(fresh_db, "Hobbit", "Tolkien creature", llm_caller=_stub_llm)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_MODERN_COINAGE


def test_resolve_via_llm_anchors_to_existing_etymon_when_attested(fresh_db: Path) -> None:
    """Pre-filter misses but LLM identifies a real attested form whose
    (form, language) pair is already in the etymon table → resolution
    is anchored to that etymon's id."""
    with LexiconDB(fresh_db) as db:
        target_id = _seed_etymon(db, canonical_form="trǫll", language="old-norse", gloss="giant")
        db.commit()

    def _stub_llm(name, description):
        return {
            "attested_in": "old-norse",
            "historical_form": "trǫll",
            "gloss": "giant",
            "citation": "Cleasby-Vigfusson",
            "confidence": "high",
            "bar_reason": None,
            "reasoning": "ON troll",
        }

    res = fp.resolve(fresh_db, "Some-troll-variant", "ON-derived monster", llm_caller=_stub_llm)
    assert res.usable is True
    assert res.etymon_id == target_id  # anchored to existing row


def test_resolve_via_llm_failure_bars_as_uncertain(fresh_db: Path) -> None:
    """A network/parse failure during LLM call doesn't crash; rows get
    barred as uncertain so the input can be re-processed later."""

    def _stub_llm(name, description):
        raise RuntimeError("simulated network failure")

    res = fp.resolve(fresh_db, "Mystery", "unknown", llm_caller=_stub_llm)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_UNCERTAIN


# ---------------------------------------------------------------------
# write_resolution
# ---------------------------------------------------------------------


def test_write_resolution_inserts_one_row(fresh_db: Path) -> None:
    res = fp.Resolution(
        usable=True,
        etymon_id=42,
        bar_reason=None,
        resolution_method="descent_lookup",
        confidence="high",
        citation="wiktionary-descent-chain",
        reasoning="test",
    )
    with LexiconDB(fresh_db) as db:
        # need an etymon row to satisfy FK
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (42, 'x', 'old-english')"
        )
        rid = fp.write_resolution(db.conn, input_name="Test", input_description="t", resolution=res)
        db.commit()
    assert rid > 0

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fantasy_morpheme").fetchall()
    assert len(rows) == 1
    r = rows[0]
    assert r["input_name"] == "Test"
    assert r["usable"] == 1
    assert r["etymon_id"] == 42
    assert r["approach_version"] == fp.APPROACH_VERSION


def test_write_resolution_upserts_on_same_name_and_version(fresh_db: Path) -> None:
    """Re-running with the same approach_version updates in place; a
    bumped version creates a NEW row so we keep the audit trail."""
    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'x', 'old-english')"
        )
        # First write: barred
        fp.write_resolution(
            db.conn,
            input_name="Test",
            input_description="d1",
            resolution=fp.Resolution(
                usable=False,
                etymon_id=None,
                bar_reason=fp.BAR_REASON_NO_ETYMOLOGY,
                resolution_method="descent_lookup",
                confidence="low",
                citation=None,
                reasoning="r1",
            ),
        )
        # Second write same name + version: upserts to usable
        fp.write_resolution(
            db.conn,
            input_name="Test",
            input_description="d2",
            resolution=fp.Resolution(
                usable=True,
                etymon_id=1,
                bar_reason=None,
                resolution_method="llm_full_research",
                confidence="high",
                citation="LSJ",
                reasoning="r2",
            ),
        )
        # Third write with a different version: separate row
        fp.write_resolution(
            db.conn,
            input_name="Test",
            input_description="d3",
            resolution=fp.Resolution(
                usable=True,
                etymon_id=1,
                bar_reason=None,
                resolution_method="manual",
                confidence="high",
                citation=None,
                reasoning="r3",
            ),
            approach_version="fantasy-v2",
        )
        db.commit()

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fantasy_morpheme ORDER BY id").fetchall()
    assert len(rows) == 2
    # First row got upserted to the second write's content
    assert rows[0]["usable"] == 1
    assert rows[0]["resolution_method"] == "llm_full_research"
    assert rows[0]["citation"] == "LSJ"
    assert rows[0]["approach_version"] == fp.APPROACH_VERSION
    # Second row is the v2 audit
    assert rows[1]["approach_version"] == "fantasy-v2"
    assert rows[1]["resolution_method"] == "manual"


def test_tag_etymon_as_fantasy_applies_both_tags_and_is_idempotent(fresh_db: Path) -> None:
    """Default tags include both 'fantasy' (register) and 'monster'
    (creature inventory marker, matches the existing seed entries)."""
    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'x', 'old-english')"
        )
        fp.tag_etymon_as_fantasy(db.conn, 1)
        fp.tag_etymon_as_fantasy(db.conn, 1)  # second call must be a no-op
        db.commit()

    conn = sqlite3.connect(fresh_db)
    tags = {r[0] for r in conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id = 1").fetchall()}
    assert tags == {"fantasy", "monster"}


def test_backfill_fantasy_tag_from_monster_tag(fresh_db: Path) -> None:
    """Etymons already carrying `monster` get `fantasy` added; etymons
    without `monster` are left alone; etymons that already have both
    are no-ops."""
    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES "
            "(1, 'wyrm',  'old-english'),"
            "(2, 'elf',   'old-english'),"
            "(3, 'beorg', 'old-english'),"  # death-tagged but not monster
            "(4, 'troll', 'old-norse')"
        )
        db.conn.executemany(
            "INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, ?)",
            [
                (1, "monster"),
                (1, "magic"),
                (2, "monster"),
                (2, "magic"),
                (3, "death"),
                (4, "monster"),
                (4, "fantasy"),  # already both — no-op
            ],
        )
        db.commit()

        n_etymons, n_tags = fp.backfill_fantasy_tag_from_monster_tag(db.conn)
        db.commit()

    assert n_etymons == 2  # wyrm + elf, NOT troll (already tagged) NOR beorg (no monster)

    conn = sqlite3.connect(fresh_db)

    def tags_of(eid: int) -> set[str]:
        return {
            r[0]
            for r in conn.execute(
                "SELECT tag FROM etymon_tag WHERE etymon_id = ?", (eid,)
            ).fetchall()
        }

    assert tags_of(1) == {"monster", "magic", "fantasy"}
    assert tags_of(2) == {"monster", "magic", "fantasy"}
    assert tags_of(3) == {"death"}
    assert tags_of(4) == {"monster", "fantasy"}


def test_backfill_fantasy_tag_is_idempotent(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'x', 'old-english')"
        )
        db.conn.execute("INSERT INTO etymon_tag (etymon_id, tag) VALUES (1, 'monster')")
        fp.backfill_fantasy_tag_from_monster_tag(db.conn)
        fp.backfill_fantasy_tag_from_monster_tag(db.conn)  # 2nd is a no-op
        db.commit()
    conn = sqlite3.connect(fresh_db)
    tags = {r[0] for r in conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id = 1").fetchall()}
    assert tags == {"monster", "fantasy"}


def test_cli_backfill_fantasy_tags_dry_run_writes_nothing(fresh_db: Path) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'x', 'old-english')"
        )
        db.conn.execute("INSERT INTO etymon_tag (etymon_id, tag) VALUES (1, 'monster')")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "backfill-fantasy-tags", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0
    assert "Would backfill" in (result.output + (result.stderr or ""))
    conn = sqlite3.connect(fresh_db)
    tags = {r[0] for r in conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id = 1").fetchall()}
    assert tags == {"monster"}  # nothing added


def test_cli_backfill_fantasy_tags_apply_writes(fresh_db: Path) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'x', 'old-english')"
        )
        db.conn.execute("INSERT INTO etymon_tag (etymon_id, tag) VALUES (1, 'monster')")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "backfill-fantasy-tags", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0
    conn = sqlite3.connect(fresh_db)
    tags = {r[0] for r in conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id = 1").fetchall()}
    assert tags == {"monster", "fantasy"}


# ---------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------


def test_cli_mine_fantasy_name_single_with_apply(fresh_db: Path) -> None:
    """End-to-end CLI: pre-filter resolves, --apply writes the row + tags.
    Uses --skip-llm to keep the test offline (semantic check requires the
    network). The semantic-check-via-LLM path is covered by the unit
    tests on resolve()."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(db, canonical_form="ἅρπυια", language="ancient-greek")
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--name",
            "Harpy",
            "--description",
            "winged creature from Greek mythology",
            "--skip-llm",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fantasy_morpheme").fetchall()
    assert len(rows) == 1
    assert rows[0]["usable"] == 1
    assert rows[0]["etymon_id"] == greek_id
    # 'fantasy' tag got applied to the resolved etymon
    tags = {
        r[0]
        for r in conn.execute(
            "SELECT tag FROM etymon_tag WHERE etymon_id = ?", (greek_id,)
        ).fetchall()
    }
    assert "fantasy" in tags


def test_cli_mine_fantasy_name_dry_run_writes_nothing(fresh_db: Path) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(db, canonical_form="ἅρπυια", language="ancient-greek")
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--name",
            "Harpy",
            "--description",
            "winged creature",
            "--skip-llm",
        ],
    )
    assert result.exit_code == 0
    conn = sqlite3.connect(fresh_db)
    rows = conn.execute("SELECT COUNT(*) FROM fantasy_morpheme").fetchone()
    assert rows[0] == 0


def test_cli_mine_fantasy_name_batch_mode(fresh_db: Path, tmp_path: Path) -> None:
    """JSONL batch input: two names, one resolves, one is barred via skip-llm."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(db, canonical_form="ἅρπυια", language="ancient-greek")
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        db.commit()

    batch = tmp_path / "names.jsonl"
    batch.write_text(
        json.dumps({"name": "Harpy", "description": "Greek winged creature"})
        + "\n"
        + json.dumps({"name": "Gnoll", "description": "Dunsany hyena humanoid"})
        + "\n"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--batch",
            str(batch),
            "--skip-llm",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fantasy_morpheme ORDER BY input_name").fetchall()
    assert len(rows) == 2
    by_name = {r["input_name"]: r for r in rows}
    assert by_name["Harpy"]["usable"] == 1
    assert by_name["Gnoll"]["usable"] == 0
    assert by_name["Gnoll"]["bar_reason"] == fp.BAR_REASON_NO_ETYMOLOGY


def test_cli_mine_fantasy_name_rejects_conflicting_input_modes(
    fresh_db: Path, tmp_path: Path
) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    batch = tmp_path / "names.jsonl"
    batch.write_text("")  # exists but empty so Click's exists=True passes
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--name",
            "Harpy",
            "--description",
            "x",
            "--batch",
            str(batch),
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in (result.output + (result.stderr or ""))


def test_write_resolution_input_name_is_case_insensitive(fresh_db: Path) -> None:
    """input_name has COLLATE NOCASE so write_resolution(`Harpy`) and
    write_resolution(`harpy`) at the same approach_version target the
    same row instead of producing duplicates."""
    res = fp.Resolution(
        usable=False,
        etymon_id=None,
        bar_reason=fp.BAR_REASON_NO_ETYMOLOGY,
        resolution_method="descent_lookup",
        confidence="low",
        citation=None,
        reasoning="r",
    )
    with LexiconDB(fresh_db) as db:
        fp.write_resolution(db.conn, input_name="Harpy", input_description="d1", resolution=res)
        fp.write_resolution(db.conn, input_name="harpy", input_description="d2", resolution=res)
        fp.write_resolution(db.conn, input_name="HARPY", input_description="d3", resolution=res)
        db.commit()

    conn = sqlite3.connect(fresh_db)
    n = conn.execute("SELECT COUNT(*) FROM fantasy_morpheme").fetchone()[0]
    assert n == 1  # all three writes upserted to the same row


def test_migration_adds_unapproved_columns_to_existing_table(tmp_path: Path) -> None:
    """fantasy_morpheme tables created before the unapproved_language /
    unapproved_form columns existed must get them via ALTER on a
    subsequent migrate_schema run."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        # Drop the alembic-created fantasy_morpheme and rebuild it
        # WITHOUT the unapproved_language / unapproved_form columns.
        # This simulates a legacy DB created before those columns
        # shipped — the scenario migrate_schema's ADD COLUMN branch
        # is supposed to handle for in-place upgrades. No IF EXISTS
        # so a future regression where alembic stops creating the
        # table fails loudly here instead of silently no-op'ing into
        # a freshly-created-legacy-shape state.
        #
        # COLLATE NOCASE on input_name matches the alembic shape so
        # _create_fantasy_morpheme_table's recreate-for-collation
        # branch doesn't fire and mask the ADD COLUMN behavior we're
        # actually asserting on.
        db.conn.executescript("DROP TABLE fantasy_morpheme;")
        db.conn.executescript(
            """
            CREATE TABLE fantasy_morpheme (
              id                INTEGER PRIMARY KEY AUTOINCREMENT,
              input_name        TEXT NOT NULL COLLATE NOCASE,
              input_description TEXT,
              usable            INTEGER NOT NULL CHECK (usable IN (0, 1)),
              etymon_id         INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
              bar_reason        TEXT,
              resolution_method TEXT NOT NULL,
              approach_version  TEXT NOT NULL,
              confidence        TEXT,
              citation          TEXT,
              reasoning         TEXT,
              processed_at      TEXT NOT NULL,
              UNIQUE (input_name, approach_version)
            );
            """
        )
        db.commit()

        # Now run the migration — it should add the two new columns.
        applied = migrate_schema(db)
        db.commit()

    assert applied.get("fantasy_morpheme.unapproved_language") is True
    assert applied.get("fantasy_morpheme.unapproved_form") is True

    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(fantasy_morpheme)").fetchall()}
    assert "unapproved_language" in cols
    assert "unapproved_form" in cols


def test_cli_mine_fantasy_name_model_flag_propagates(fresh_db: Path, monkeypatch) -> None:
    """`--model X` must be plumbed through to both the semantic-check
    caller and the full-research caller via functools.partial."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod
    from wyrd.generators.kenning import fantasy_pipeline as fp_mod

    captured_models: list[tuple[str, str]] = []

    def _spy_full(name, description, *, model="default", **kw):
        captured_models.append(("full", model))
        return {
            "attested_in": "none",
            "historical_form": None,
            "gloss": None,
            "citation": None,
            "confidence": "low",
            "bar_reason": "no_etymology_found",
            "reasoning": "stub",
        }

    monkeypatch.setattr(fp_mod, "_llm_full_research", _spy_full)
    # Pre-filter must miss so we hit the LLM path.
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--name",
            "NeverInCorpus",
            "--description",
            "test creature",
            "--model",
            "gemini-2.5-pro",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert ("full", "gemini-2.5-pro") in captured_models


def test_cli_mine_fantasy_name_requires_input(fresh_db: Path) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "mine-fantasy-name", "--db", str(fresh_db)],
    )
    assert result.exit_code != 0


def test_cli_mine_fantasy_name_emits_progress_line(fresh_db: Path, tmp_path: Path) -> None:
    """Mining batches must emit a `[completed/total]` progress line so
    operators can see how far the run has gotten. Convention is set by
    `lexicon mine-llm` and documented in CLAUDE.md."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    batch = tmp_path / "names.jsonl"
    # 12 inputs so we cross the every-10 progress threshold + emit the
    # final line. All names will fail to resolve in --skip-llm mode
    # (no etymon corpus seeded), but the progress line still fires.
    lines = [json.dumps({"name": f"Creature{i}", "description": "test"}) for i in range(12)]
    batch.write_text("\n".join(lines) + "\n")
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--batch",
            str(batch),
            "--skip-llm",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    # The 10/12 line should appear (every-10 threshold) and the 12/12
    # final line as well.
    combined = result.output + (result.stderr or "")
    assert "[10/12]" in combined
    assert "[12/12]" in combined


def test_cli_ingest_etymonline_emits_progress_line(fresh_db: Path, tmp_path: Path) -> None:
    """Same convention for the etymonline file walker."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    src_dir = tmp_path / "etym"
    src_dir.mkdir()
    for i in range(3):
        (src_dir / f"{i}.txt").write_text(f"word{i}(n.)\n\nlate 14c., from Old French foo{i}.\n\n")
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "ingest-etymonline", str(src_dir), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0
    combined = result.output + (result.stderr or "")
    # Each file gets a [N/3] prefix.
    assert "[1/3]" in combined
    assert "[3/3]" in combined


def test_resolve_via_llm_alias_map_sanskrit_to_iso(fresh_db: Path) -> None:
    """LLM returns 'Sanskrit' (descriptive); pipeline maps to 'sa' (ISO
    code in etymon table) and resolves usable. Pinned regression for
    the wave-2 language expansion (Hebrew/Arabic/Persian/Sanskrit/
    Akkadian/Egyptian) where APPROVED_LANGUAGES is the ISO code and
    the LLM's natural output is the descriptive name."""
    with LexiconDB(fresh_db) as db:
        target_id = _seed_etymon(db, canonical_form="rakṣasa", language="sa")
        db.commit()

    def _stub_llm(name, description):
        return {
            "attested_in": "Sanskrit",  # descriptive, capitalized
            "historical_form": "rakṣasa",
            "gloss": "demon",
            "citation": "Monier-Williams",
            "confidence": "high",
            "bar_reason": None,
            "reasoning": "stub",
        }

    res = fp.resolve(fresh_db, "Rakshasa", "Hindu demon", llm_caller=_stub_llm)
    assert res.usable is True
    assert res.etymon_id == target_id


def test_resolve_via_llm_alias_map_arabic_with_etymonline_genie_case(fresh_db: Path) -> None:
    """The Genie misclassification case — LLM returns 'arabic' for the
    djinn etymology and pipeline resolves to 'ar' as the canonical code."""
    with LexiconDB(fresh_db) as db:
        target_id = _seed_etymon(db, canonical_form="jinn", language="ar")
        db.commit()

    def _stub_llm(name, description):
        return {
            "attested_in": "arabic",  # already lowercase
            "historical_form": "jinn",
            "gloss": "spirit",
            "citation": "Lane's Arabic-English Lexicon",
            "confidence": "high",
            "bar_reason": None,
            "reasoning": "stub",
        }

    res = fp.resolve(fresh_db, "Genie", "Arabian spirit", llm_caller=_stub_llm)
    assert res.usable is True
    assert res.etymon_id == target_id


def test_resolve_via_llm_alias_map_egyptian_variants(fresh_db: Path) -> None:
    """Both 'Egyptian' and 'ancient-egyptian' should map to 'egy'."""
    with LexiconDB(fresh_db) as db:
        target_id = _seed_etymon(db, canonical_form="anpw", language="egy")
        db.commit()

    for raw_lang in ("Egyptian", "ancient-egyptian", "Ancient Egyptian"):

        def _stub_llm(name, description, _raw=raw_lang):
            return {
                "attested_in": _raw,
                "historical_form": "anpw",
                "gloss": "Anubis",
                "citation": "Faulkner",
                "confidence": "high",
                "bar_reason": None,
                "reasoning": "stub",
            }

        res = fp.resolve(fresh_db, "Anubis", "Egyptian deity", llm_caller=_stub_llm)
        assert res.usable is True, f"failed for {raw_lang!r}"
        assert res.etymon_id == target_id, f"failed for {raw_lang!r}"


@pytest.mark.parametrize(
    "descriptive_name,iso_code,canonical_form",
    [
        # Iranian precursors / postcursors (wyrd-c97z)
        ("old-persian", "peo", "ahura"),
        ("Old Persian", "peo", "ahura"),
        ("parthian", "xpr", "frawardin"),
        # Indo-Aryan precursors / postcursors
        ("pali", "pi", "rakkhasa"),
        ("Pali", "pi", "rakkhasa"),
        ("prakrit", "pra", "rakkhasa"),
        # Egyptian descendant
        ("coptic", "cop", "anuph"),
        ("Coptic", "cop", "anuph"),
        # Aramaic descendant
        ("classical-syriac", "syc", "shedda"),
        ("Syriac", "syc", "shedda"),
        # Proto-language reconstructions
        ("proto-semitic", "sem-pro", "*ʔil-"),
        ("Proto-Indo-Iranian", "iir-pro", "*sandh-"),
        ("Proto-Indo-Aryan", "inc-pro", "*rakṣas-"),
        ("Proto-Iranian", "ira-pro", "*ahura-"),
        ("Proto-Afroasiatic", "afa-pro", "*napas-"),
        ("Proto-West-Semitic", "sem-wes-pro", "*malk-"),
        # Substrate / Armenian
        ("sumerian", "sux", "kur"),
        ("Old Armenian", "axm", "vahagn"),
        ("Classical Armenian", "axm", "vahagn"),
    ],
)
def test_resolve_via_llm_alias_map_wave2_precursors_postcursors(
    fresh_db: Path, descriptive_name: str, iso_code: str, canonical_form: str
) -> None:
    """wyrd-c97z: the 16 precursor/postcursor codes added to
    APPROVED_LANGUAGES come with paired _LANGUAGE_ALIAS_MAP entries
    so the LLM's natural descriptive output (e.g., "Old Persian",
    "Coptic", "Pali", "Proto-Semitic") routes to the correct ISO
    code. A typo in any of the 14 new alias values would silently
    re-classify those families as outside_language_family — this
    test would fail. Covers normalization (case + space-to-dash)
    end-to-end since the production pipeline always lowercases +
    space-replaces before alias lookup."""
    with LexiconDB(fresh_db) as db:
        target_id = _seed_etymon(db, canonical_form=canonical_form, language=iso_code)
        db.commit()

    def _stub_llm(name, description, _raw=descriptive_name):
        return {
            "attested_in": _raw,
            "historical_form": canonical_form,
            "gloss": "stub",
            "citation": "stub",
            "confidence": "high",
            "bar_reason": None,
            "reasoning": "stub",
        }

    res = fp.resolve(fresh_db, f"FantasyName_{iso_code}", "stub description", llm_caller=_stub_llm)
    assert res.usable is True, f"alias {descriptive_name!r} failed to resolve"
    assert res.etymon_id == target_id


def test_descent_walking_lookup_prefers_attested_over_reconstructed(fresh_db: Path) -> None:
    """When BFS surfaces both an attested ancestor (e.g. OE `tūn`) and a
    reconstructed proto-form (e.g. proto-germanic `*tūnaz`) for the same
    input, the attested form must win the first-pick slot. Pre-fix, SQL
    row-insertion order picked arbitrarily — `*tūnaz` could surface
    before `tūn`, feeding the semantic-check pass a reconstruction when
    a concrete attestation was available. Pinning the contract: starred
    canonical forms (linguistic convention for reconstructions) sort
    AFTER non-starred forms in the descent_walking_lookup return list."""
    with LexiconDB(fresh_db) as db:
        # Mod-en `town` is the BFS seed (not approved itself, but seeds
        # the walk via the lemma lookup). Both ancestors are at BFS
        # layer 1; only the dedup-time sort separates them.
        modern_id = _seed_etymon(db, canonical_form="town", language="modern-english")
        oe_id = _seed_etymon(db, canonical_form="tūn", language="old-english")
        pg_id = _seed_etymon(db, canonical_form="*tūnaz", language="proto-germanic")
        # Insert the reconstructed edge FIRST so without the sort it
        # would naturally surface before the attested edge — this is the
        # adversarial case the sort is supposed to fix.
        _seed_descent(db, parent_id=pg_id, child_id=modern_id)
        _seed_descent(db, parent_id=oe_id, child_id=modern_id)
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "town")
    forms = [m.canonical_form for m in matches]
    assert "tūn" in forms, "attested OE ancestor must surface"
    assert "*tūnaz" in forms, "reconstructed proto ancestor should still surface (just later)"
    assert forms.index("tūn") < forms.index("*tūnaz"), (
        "attested form must precede reconstructed form regardless of insertion order"
    )


def test_canonical_languages_excludes_precursor_postcursor_stack(fresh_db: Path) -> None:
    """The LLM prompt uses CANONICAL_LANGUAGES (not APPROVED_LANGUAGES)
    so obscure precursor/postcursor codes don't pollute the model's
    register list. APPROVED_LANGUAGES = CANONICAL ∪ stack, but the
    stack codes are pipeline-internal (descent-walk only). This test
    pins the partition so a future "merge them all into one" refactor
    can't silently put `iir-pro` and `afa-pro` strings back in the
    Gemini prompt."""
    # Sanity: APPROVED is a strict superset of CANONICAL.
    assert fp.CANONICAL_LANGUAGES < fp.APPROVED_LANGUAGES
    # The wave-2 stack lives in APPROVED but NOT in CANONICAL.
    for code in (
        "hbo",
        "sem-pro",
        "afa-pro",
        "iir-pro",
        "ira-pro",
        "inc-pro",
        "peo",
        "xpr",
        "cop",
        "pra",
        "pi",
        "axm",
        "syc",
        "sux",
        "fa-cls",
        "sem-wes-pro",
    ):
        assert code in fp.APPROVED_LANGUAGES, f"{code} should be in APPROVED_LANGUAGES"
        assert code not in fp.CANONICAL_LANGUAGES, (
            f"{code} should NOT be in CANONICAL_LANGUAGES (it's pipeline-internal)"
        )
    # The canonical register codes ARE in CANONICAL.
    for code in ("he", "ar", "fa", "sa", "akk", "egy", "arc", "pal", "old-english"):
        assert code in fp.CANONICAL_LANGUAGES, f"{code} should be in CANONICAL_LANGUAGES"


def test_descent_walking_lookup_resolves_through_wave2_precursor(fresh_db: Path) -> None:
    """wyrd-c97z: a Modern Hebrew lemma (he) with a descent edge to
    its Biblical Hebrew precursor (hbo) should surface BOTH the he
    seed AND the hbo ancestor as approved-family hits, because the
    wave-2 stratification put hbo into APPROVED_LANGUAGES alongside
    he. Before this PR, hbo was unapproved and the BFS would walk
    PAST it without surfacing the precursor as a usable resolution
    target; the chain would dead-end at the language gate even when
    the data was genuine. Pinning the contract so a future bare-`he`
    revert silently truncates the walk → test fails."""
    with LexiconDB(fresh_db) as db:
        biblical_id = _seed_etymon(db, canonical_form="gōlem", language="hbo")
        modern_id = _seed_etymon(db, canonical_form="golem", language="he")
        _seed_descent(db, parent_id=biblical_id, child_id=modern_id)
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "golem")
    by_lang = {m.language: m for m in matches}
    assert "he" in by_lang, "Modern Hebrew seed should surface as direct hit"
    assert "hbo" in by_lang, "Biblical Hebrew precursor must surface — confirms wave-2 hbo approval"
    assert by_lang["hbo"].canonical_form == "gōlem"


# ---------------------------------------------------------------------
# fetch_resolved_input_names + --skip-resolved
# ---------------------------------------------------------------------


def test_fetch_resolved_input_names_returns_empty_for_fresh_db(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        assert fp.fetch_resolved_input_names(db.conn) == set()


def test_fetch_resolved_input_names_returns_resolved_names_only(fresh_db: Path) -> None:
    """Only rows matching the requested approach_version are returned."""
    with LexiconDB(fresh_db) as db:
        etymon_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        db.commit()
        for in_name, approach in [
            ("Harpy", fp.APPROACH_VERSION),
            ("Troll", fp.APPROACH_VERSION),
            ("OldVersionName", "fantasy-v0-old"),
        ]:
            res = fp.Resolution(
                usable=True,
                etymon_id=etymon_id,
                resolution_method="descent_lookup",
                bar_reason=None,
                confidence="high",
                citation="test",
                reasoning="seed",
            )
            fp.write_resolution(
                db.conn,
                input_name=in_name,
                input_description="seed",
                resolution=res,
                approach_version=approach,
            )
        db.commit()

        names = fp.fetch_resolved_input_names(db.conn)
        assert names == {"Harpy", "Troll"}

        old_names = fp.fetch_resolved_input_names(db.conn, "fantasy-v0-old")
        assert old_names == {"OldVersionName"}


def test_cli_mine_fantasy_name_skip_resolved_filters_inputs(fresh_db: Path, tmp_path: Path) -> None:
    """--skip-resolved skips inputs whose name already has a row for
    the current approach_version, paying for only the new ones."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    # Seed: Harpy already resolved, Troll not yet.
    with LexiconDB(fresh_db) as db:
        etymon_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        db.commit()
        res = fp.Resolution(
            usable=True,
            etymon_id=etymon_id,
            resolution_method="descent_lookup",
            bar_reason=None,
            confidence="high",
            citation="seed",
            reasoning="prior run",
        )
        fp.write_resolution(db.conn, input_name="Harpy", input_description="prior", resolution=res)
        db.commit()

    # Need both etymons present so the offline pre-filter resolves Troll
    # without an LLM call (--skip-llm). Harpy etymon already inserted.
    with LexiconDB(fresh_db) as db:
        troll_greek_id = _seed_etymon(db, canonical_form="trǫll", language="old-norse")
        troll_modern_id = _seed_etymon(db, canonical_form="troll", language="modern-english")
        _seed_descent(db, parent_id=troll_greek_id, child_id=troll_modern_id)
        db.commit()

    batch_path = tmp_path / "inputs.jsonl"
    batch_path.write_text(
        '{"name": "Harpy", "description": "winged"}\n'
        '{"name": "Troll", "description": "Norse cave-dweller"}\n'
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--batch",
            str(batch_path),
            "--skip-llm",
            "--skip-resolved",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    combined = result.output + (result.stderr or "")
    # Routing summary reports the skip count.
    assert "Routing 1 fantasy-name input(s)" in combined
    assert "skipped 1 already-resolved" in combined
    # Harpy was NOT re-processed (no log line).
    assert "USABLE  Harpy" not in combined
    # Troll WAS processed.
    assert "USABLE  Troll" in combined or "BARRED  Troll" in combined


def test_cli_mine_fantasy_name_skip_resolved_no_op_on_fresh_db(
    fresh_db: Path, tmp_path: Path
) -> None:
    """--skip-resolved against a fresh DB skips nothing."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    with LexiconDB(fresh_db) as db:
        gid = _seed_etymon(db, canonical_form="ἅρπυια", language="ancient-greek")
        mid = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=gid, child_id=mid)
        db.commit()

    batch_path = tmp_path / "inputs.jsonl"
    batch_path.write_text('{"name": "Harpy", "description": "winged"}\n')

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--batch",
            str(batch_path),
            "--skip-llm",
            "--skip-resolved",
            "--apply",
        ],
    )
    assert result.exit_code == 0
    combined = result.output + (result.stderr or "")
    assert "Routing 1 fantasy-name input(s) (skipped 0 already-resolved)" in combined


def test_cli_mine_fantasy_name_skip_resolved_is_case_insensitive(
    fresh_db: Path, tmp_path: Path
) -> None:
    """fantasy_morpheme.input_name uses COLLATE NOCASE — so an input
    'harpy' should be skipped when a row already exists under 'Harpy'.
    Pinned regression for the case-sensitivity bug Gemini caught
    2026-05-06: the Python `in` check on the resolved-names set was
    case-sensitive while the DB constraint was not, leading to
    redundant LLM calls for casing variants."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    with LexiconDB(fresh_db) as db:
        etymon_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        db.commit()
        res = fp.Resolution(
            usable=True,
            etymon_id=etymon_id,
            resolution_method="descent_lookup",
            bar_reason=None,
            confidence="high",
            citation="seed",
            reasoning="prior run",
        )
        # Seed with capital-H 'Harpy'.
        fp.write_resolution(db.conn, input_name="Harpy", input_description="prior", resolution=res)
        db.commit()

    batch_path = tmp_path / "inputs.jsonl"
    # Input uses lowercase 'harpy' — must still be recognized as
    # already-resolved.
    batch_path.write_text('{"name": "harpy", "description": "winged"}\n')

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--batch",
            str(batch_path),
            "--skip-llm",
            "--skip-resolved",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    combined = result.output + (result.stderr or "")
    assert "Routing 0 fantasy-name input(s) (skipped 1 already-resolved)" in combined
    # No processing line printed.
    assert "USABLE" not in combined
    assert "BARRED" not in combined


def test_descent_walking_lookup_chunks_frontier_over_sqlite_var_limit(
    fresh_db: Path, monkeypatch
) -> None:
    """A BFS frontier wider than SQLite's variable limit must not raise
    'too many SQL variables' — the descent-edge IN-list is chunked. Forces a low
    limit + tiny chunk so a modest 8-wide second-level frontier overflows an
    UNchunked query (8 > 6) but a chunked one stays under (4 <= 6)."""
    with LexiconDB(fresh_db) as db:
        child = _seed_etymon(db, canonical_form="frontierbeast", language="modern-english")
        for i in range(8):
            pid = _seed_etymon(db, canonical_form=f"anc{i}", language="old-english", gloss=f"g{i}")
            _seed_descent(db, parent_id=pid, child_id=child)
        db.commit()

    monkeypatch.setattr(fp, "_BFS_PARAM_CHUNK", 4)
    orig_connect = fp._connect_ro

    def _ro_low_limit(path):
        conn = orig_connect(path)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 6)
        return conn

    monkeypatch.setattr(fp, "_connect_ro", _ro_low_limit)

    # Unchunked, the second-level query binds all 8 frontier ids at once and
    # raises sqlite3.OperationalError "too many SQL variables"; chunked, it works.
    matches = fp.descent_walking_lookup(fresh_db, "frontierbeast")
    assert {m.canonical_form for m in matches} == {f"anc{i}" for i in range(8)}
