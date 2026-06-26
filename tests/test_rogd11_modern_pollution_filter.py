"""wyrd-rogd.11 — the era-grid modern stage must carry the morpheme's own
common-noun reflex, not the cluster tier's derived proper nouns (toponyms,
anthroponyms) or mistagged foreign cognates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.bundle._family import (
    _fetch_family_era_reflexes,
    _is_derived_name_pollution,
)


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def test_pollution_predicate_flags_proper_nouns_and_multiword() -> None:
    for bad in (
        "West Ham",
        "Notting Hill",
        "Wesley",
        "Derek",
        "Hank",
        "Coldingham",
        "Düne",
        "Ouest",
    ):
        assert _is_derived_name_pollution(bad), bad
    for good in ("ton", "hill", "-bury", "ceaster", "wīc", "thorpe"):
        assert not _is_derived_name_pollution(good), good


def _mate(db, root_id: int, form: str, lang: str) -> int:
    eid = db.upsert_etymon(form, lang)
    db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root_id, eid))
    return eid


def test_modern_stage_drops_derived_names_keeps_reflex(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("tūn", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        _mate(db, root, "ton", "modern-english")  # genuine reflex
        _mate(db, root, "West Ham", "modern-english")  # multi-word toponym
        _mate(db, root, "Wesley", "modern-english")  # anthroponym
        _mate(db, root, "Düne", "modern-english")  # capitalized foreign cognate
        db.commit()
        out = _fetch_family_era_reflexes(db, [root], "old-english")
        assert [e["form"] for e in out.get("modern-english", [])] == ["ton"]


def test_historical_stage_is_not_filtered(fresh_db: Path) -> None:
    # The filter is modern-only; a capitalized OE cluster mate (an attested
    # period spelling) must survive in the old-english stage.
    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("tūn", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        _mate(db, root, "Tuun", "old-english")  # capitalized, but historical → kept
        db.commit()
        out = _fetch_family_era_reflexes(db, [root], "old-english")
        oe = {e["form"] for e in out.get("old-english", [])}
        assert "Tuun" in oe and "tūn" in oe


def test_all_modern_pollution_empties_the_stage(fresh_db: Path) -> None:
    # When nothing but derived names survive, the modern stage is omitted
    # entirely (no empty bucket manufactured).
    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("hyll", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        _mate(db, root, "Notting Hill", "modern-english")
        _mate(db, root, "Hillary", "modern-english")
        db.commit()
        out = _fetch_family_era_reflexes(db, [root], "old-english")
        assert "modern-english" not in out


def test_pollution_predicate_edge_cases() -> None:
    # empty / whitespace-only → pollution (err toward empty)
    assert _is_derived_name_pollution("")
    assert _is_derived_name_pollution("   ")
    # non-space whitespace (tab, non-breaking space) is still multi-word
    assert _is_derived_name_pollution("West\tHam")
    assert _is_derived_name_pollution("West Ham")
    # leading punctuation before an uppercase letter is still a proper noun
    assert _is_derived_name_pollution(".Wesley")
    assert _is_derived_name_pollution("(Hank)")
    # a leading-dash affix whose first alpha char is lowercase is kept
    assert not _is_derived_name_pollution("-ton")


def test_pollution_predicate_drops_single_char_and_function_words() -> None:
    # wyrd-pzg5: single-character segmentation fragments → pollution
    for frag in ("s", "k", "e", "a", "-s", "x-"):
        assert _is_derived_name_pollution(frag), frag
    # forms with no alphabetic character (numeric / punctuation artifacts) → pollution
    for nonalpha in ("123", "??", "—", "12-3", "."):
        assert _is_derived_name_pollution(nonalpha), nonalpha
    # closed-class function words / pronouns ride in via deep cluster over-merge
    for fw in ("the", "they", "them", "thy", "thee", "their", "and", "or", "what", "um", "oh"):
        assert _is_derived_name_pollution(fw), fw  # lowercase → stopword branch
        assert _is_derived_name_pollution(fw.upper()), fw  # caps form caught too


def test_pollution_predicate_keeps_legit_short_elements() -> None:
    # wyrd-pzg5 guard: genuine 2-char place-name elements must NOT be dropped —
    # the stopword set deliberately excludes 'by' (-by), and short elements are
    # never single-char. A regression here would silently empty real grids.
    for good in ("ea", "by", "ey", "ash", "low", "ley", "ham", "ing", "ait", "nab"):
        assert not _is_derived_name_pollution(good), good


def test_stopwords_disjoint_from_known_place_name_elements() -> None:
    # wyrd-pzg5 invariant (enforced, not just commented): the stopword set must
    # never contain a real place-name element, or it would silently empty that
    # morpheme's modern grid + drop its sampleable surface. Guard against a
    # future stopword addition colliding with the curated EPNS element inventory.
    from wyrd.generators.kenning.lexicon.bundle._family import _MODERN_REFLEX_STOPWORDS
    from wyrd.generators.kenning.lexicon.toponym_reflex_audit import PROTECTED_ELEMENTS

    collision = _MODERN_REFLEX_STOPWORDS & PROTECTED_ELEMENTS
    assert not collision, f"stopwords collide with place-name elements: {sorted(collision)}"


def test_filtered_forms_do_not_reach_forms_by_lang(fresh_db: Path) -> None:
    # The seam guarantee: a dropped modern proper noun must not survive into
    # forms_by_lang (what the generator samples), not just the era-grid view.
    from wyrd.generators.kenning.lexicon.bundle._family import (
        _merge_era_reflexes_into_forms_by_lang,
    )

    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("tūn", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        _mate(db, root, "ton", "modern-english")
        _mate(db, root, "West Ham", "modern-english")
        db.commit()
        era_reflexes = _fetch_family_era_reflexes(db, [root], "old-english")
        forms_by_lang: dict[str, list[str]] = {}
        _merge_era_reflexes_into_forms_by_lang(forms_by_lang, era_reflexes)
        assert forms_by_lang.get("modern-english") == ["ton"]


def test_reflex_linked_word_includes_cluster_era_reflexes_in_pool() -> None:
    """wyrd-nxhh consumption seam: the reflex-LINKED word path
    (``_word_for_reflex``, the common case — most morphemes have a modern
    reflex) must fold the cluster-carried era-reflex surfaces into the
    generator's draw pool, not just the member-descendant walk.

    Those surfaces come from the wider ``cognate_id`` cluster, so they live in
    ``fam["forms_by_lang"]`` (merged by ``_merge_era_reflexes_into_forms_by_lang``)
    but NOT in ``member_form_by_id`` (the member-only descendant walk). Pre-fix
    ``_word_for_reflex`` read only the walk, so ``modern_english`` came back
    ``None`` and the surfaces wyrd-nxhh added were silently un-sampleable — even
    though ``_synthesize_word_for_family`` (the reflex-LESS path) reads
    ``forms_by_lang`` directly and got them right."""
    from wyrd.generators.kenning.lexicon.bundle._subject import _word_for_reflex

    # OE 'ceaster' member, with modern cluster mates 'caster'/'chester' that are
    # in forms_by_lang (post-merge) but NOT member_form_by_id.
    fam = {
        "member_descendants": {1: [1]},
        "member_form_by_id": {1: ("old-english", "ceaster")},
        "forms_by_lang": {"old-english": ["ceaster"], "modern-english": ["caster", "chester"]},
        "reflexes": [],
        "era_reflexes": {},
        "glosses": [],
        "tags": [],
        "modifier_type": None,
    }
    word = _word_for_reflex({"surface_form": "chester"}, [(fam, [1])])
    # The cluster reflexes reach the draw pool (pre-fix: None).
    assert word.get("modern_english") == ["caster", "chester"]
    # The member's own language is unaffected.
    assert word.get("old_english") == ["ceaster"]


def test_reflex_linked_word_pool_dedups_walk_and_forms_by_lang() -> None:
    """A form present in BOTH the member walk and forms_by_lang must appear once
    (the union is dedup-append, no draw-probability double-count)."""
    from wyrd.generators.kenning.lexicon.bundle._subject import _word_for_reflex

    fam = {
        "member_descendants": {1: [1]},
        "member_form_by_id": {1: ("modern-english", "ton")},
        "forms_by_lang": {"modern-english": ["ton", "chester"]},
        "reflexes": [],
        "era_reflexes": {},
        "glosses": [],
        "tags": [],
        "modifier_type": None,
    }
    word = _word_for_reflex({"surface_form": "ton"}, [(fam, [1])])
    assert word.get("modern_english") == ["ton", "chester"]  # 'ton' not doubled


def test_fantasy_export_also_filters_modern_pollution(fresh_db: Path) -> None:
    from wyrd.generators.kenning.lexicon.fantasy_export import collect_fantasy_morphemes

    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("tūn", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        _mate(db, root, "ton", "modern-english")
        _mate(db, root, "Pemberton", "modern-english")  # derived toponym
        db.conn.execute(
            "INSERT INTO fantasy_morpheme "
            "(input_name, usable, etymon_id, resolution_method, approach_version, processed_at) "
            "VALUES ('Tunhold', 1, ?, 'test', 'v1', '2026-01-01')",
            (root,),
        )
        db.commit()
        out = collect_fantasy_morphemes(db)
        modern = [e["form"] for e in out["Tunhold"]["era_reflexes"].get("modern-english", [])]
        assert modern == ["ton"]


def _descent_parent(
    db, child_id: int, parent_form: str, parent_lang: str, edge: str = "inheritance"
) -> int:
    """Give ``child_id`` a descent parent in ``parent_lang`` via an ``edge``-type edge."""
    pid = db.upsert_etymon(parent_form, parent_lang)
    db.conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) VALUES (?, ?, ?, 'wikt')",
        (pid, child_id, edge),
    )
    return pid


def test_foreign_reentry_dropped_from_modern_stage(fresh_db: Path) -> None:
    # wyrd-6hbv Phase 2: a modern reflex descending from a NON-feeder language
    # (a cross-language Descendants re-entry) is dropped; a feeder-descended one
    # (and an orphan with no descent) is kept.
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wikt", title="Wiktionary")
        root = db.upsert_etymon("tūn", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        town = _mate(db, root, "town", "modern-english")
        _descent_parent(db, town, "toun", "middle-english")  # feeder → kept
        kaona = _mate(db, root, "kaona", "modern-english")
        _descent_parent(db, kaona, "kaona", "haw")  # Hawaiian → dropped
        db.commit()
        out = _fetch_family_era_reflexes(db, [root], "old-english")
        assert [e["form"] for e in out.get("modern-english", [])] == ["town"]


def test_foreign_reentry_ids_flags_any_nonfeeder_parent(fresh_db: Path) -> None:
    from wyrd.generators.kenning.lexicon.bundle._family import _foreign_reentry_ids

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wikt", title="Wiktionary")
        # 'bungoo' has a junk feeder parent (bank) AND a distant one (Dyirbal) →
        # flagged on the distant parent (ANY non-feeder parent flags it).
        bungoo = db.upsert_etymon("bungoo", "modern-english")
        _descent_parent(db, bungoo, "bank", "modern-english")
        _descent_parent(db, bungoo, "banggu", "bdy")
        town = db.upsert_etymon("town", "modern-english")  # only a feeder parent
        _descent_parent(db, town, "toun", "middle-english")
        orphan = db.upsert_etymon("orphan", "modern-english")  # no parents → kept
        db.commit()
        assert _foreign_reentry_ids(db, {bungoo, town, orphan}) == {bungoo}
        assert _foreign_reentry_ids(db, set()) == set()  # empty-input guard


def test_foreign_reentry_ids_edge_type_gate(fresh_db: Path) -> None:
    # The descent gate is inheritance/borrowing/derivation — a non-feeder parent
    # via any of those flags the reflex; a peer 'cognate' edge does NOT.
    from wyrd.generators.kenning.lexicon.bundle._family import _foreign_reentry_ids

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wikt", title="Wiktionary")
        borrowed = db.upsert_etymon("haikal", "modern-english")
        _descent_parent(db, borrowed, "haykal-ar", "ar", edge="borrowing")  # flagged
        derived = db.upsert_etymon("argaman", "modern-english")
        _descent_parent(db, derived, "argamannu", "akk", edge="derivation")  # flagged
        peer = db.upsert_etymon("peerword", "modern-english")
        _descent_parent(db, peer, "rus-cognate", "ru", edge="cognate")  # cognate → NOT flagged
        db.commit()
        assert _foreign_reentry_ids(db, {borrowed, derived, peer}) == {borrowed, derived}


def test_family_era_reflexes_union_across_members(fresh_db: Path) -> None:
    # wyrd-rogd.16: the era-grid must UNION every family member's cluster, not
    # just the root's. A rich-cluster member (modern 'green', 3 OE-stage forms)
    # folded into a poor root (OE 'green', 1 form) must keep all of them.
    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("grene", "old-english")  # ancestor / family root
        rich = db.upsert_etymon("green", "modern-english")  # folded-in reflex
        # two separate cognate clusters
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (rich, rich))
        # OE-stage cluster mates: root has 1, the reflex's cluster has 3 more
        _mate(db, root, "groeni", "old-english")  # root's own OE mate
        for f in ("graeni", "grene2", "groene"):
            mate_id = db.upsert_etymon(f, "old-english")
            db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (rich, mate_id))
        db.commit()
        # root-only would see just groeni; the family union sees all 4
        root_only = _fetch_family_era_reflexes(db, [root], "old-english")
        union = _fetch_family_era_reflexes(db, [root, rich], "old-english")
        oe_root_only = {e["form"] for e in root_only.get("old-english", [])}
        oe_union = {e["form"] for e in union.get("old-english", [])}
        assert "groeni" in oe_root_only and "graeni" not in oe_root_only  # root cluster only
        assert {"groeni", "graeni", "grene2", "groene"} <= oe_union  # unioned
        assert len(oe_union) > len(oe_root_only)
