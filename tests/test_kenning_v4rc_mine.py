"""wyrd-v4rc: Mine- morpheme disambiguation + missing extractive sense.

Two fixes, two mechanisms:

* **Part B (curation)** — the OE ``mine`` etymon (gloss 'a slope';
  Roberts 1914 Sussex) was wrongly linked as an inflected child of the
  possessive pronoun ``mīn``/``min`` ('my') because the dative/plural
  pronoun form is also spelled ``mine``. The bundle rollup then dragged
  'my' + 'inflection of mīn:' onto the slope subject. The committed
  curation events in ``data/mining/_curation.jsonl`` (1) clear the
  wrong ``lemma_id`` link and (2) suppress the derivative-pointer
  gloss. These tests apply the REAL committed events to a conflation
  fixture so a future edit to the curation file that breaks the fix
  fails here.

* **Part A (sidecar)** — extractive ``mine`` (an excavation for
  ore/minerals, Old French ``mine`` ← Gaulish) has no glossed, citable
  home in the lexicon DB: the whole Romance/Celtic cognate cluster is
  gloss-less and the only glossed modern-english ``mine`` etymon is the
  unrelated possessive pronoun. So the sense is injected via the
  ``curated_senses.json`` runtime sidecar, slotted under ``old_french``
  (its true origin) — NOT ``modern_english``, because
  ``_rank_siblings`` drops modern-english-only homographs when a
  historical etymon (OE ``mine`` 'a slope') shares the surface. These
  tests pin that the sidecar surfaces extractive ``mine`` as a
  distinct, user-facing sense that survives sibling-ranking.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning import (
    _curated_sense_subjects,
    _load_curated_senses,
    _load_meanings,
    _rank_siblings,
    available_tags,
)
from wyrd.generators.kenning.enrichment import (
    apply_curation_overrides,
    apply_gloss_suppressions,
)
from wyrd.generators.kenning.jsonl.build import (
    collect_curation_overrides,
    collect_gloss_suppressions,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.runtime.meaning import Meaning

# The committed operator-curation file holding the wyrd-v4rc events.
_CURATION_FILE = Path(__file__).resolve().parent.parent / "data" / "mining" / "_curation.jsonl"


def _build_conflation_fixture(tmp_path: Path) -> tuple[Path, int, int]:
    """Build a minimal DB reproducing the pre-fix OE mine/min
    conflation: ``mine`` (gloss 'a slope' + 'inflection of mīn:') linked
    as an inflected child of ``min`` ('my'). Returns (db_path, mine_id,
    min_id)."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    min_id = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('min', 'old-english')"
    ).lastrowid
    conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, 'my')", (min_id,))
    mine_id = conn.execute(
        "INSERT INTO etymon (canonical_form, language, lemma_id, inflection, lemma_method) "
        "VALUES ('mine', 'old-english', ?, 'dative_or_pl', 'link-lemmas-v1')",
        (min_id,),
    ).lastrowid
    for gloss in ("a slope", "inflection of mīn:"):
        conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (mine_id, gloss))
    conn.commit()
    conn.close()
    return db_path, mine_id, min_id


# ---------------------------------------------------------------------------
# Part B — committed curation events disambiguate the OE conflation
# ---------------------------------------------------------------------------


def test_committed_curation_clears_wrong_lemma_link_on_oe_mine():
    """The committed ``old-english:mine`` curation event clears the
    lemma link (slope is NOT an inflection of the 'my' pronoun)."""
    assert _CURATION_FILE.exists(), f"curation file missing: {_CURATION_FILE}"
    state = collect_curation_overrides([_CURATION_FILE])
    assert "old-english:mine" in state, (
        "wyrd-v4rc curation event for old-english:mine missing from the committed file"
    )
    assert state["old-english:mine"]["lemma_ref"] is None, (
        f"expected lemma_ref cleared (None); got {state['old-english:mine']!r}"
    )


def test_committed_curation_suppresses_derivative_gloss_on_oe_mine():
    """The committed gloss-suppression event drops 'inflection of mīn:'
    from OE mine — the derivative pointer is noise once the lemma link
    is broken."""
    state = collect_gloss_suppressions([_CURATION_FILE])
    assert "old-english:mine" in state, (
        "wyrd-v4rc gloss-suppression event for old-english:mine missing from committed file"
    )
    dropped = {s["gloss"] for s in state["old-english:mine"]["suppressions"]}
    assert "inflection of mīn:" in dropped, (
        f"expected 'inflection of mīn:' suppressed; got {dropped!r}"
    )


def test_committed_curation_applied_to_conflation_fixture_disambiguates(tmp_path: Path):
    """End-to-end: apply the REAL committed wyrd-v4rc events to a
    conflation fixture. Post-apply, OE mine (a) has no lemma link to
    the 'my' pronoun and (b) carries only 'a slope' — the derivative
    pointer is gone. The 'my' gloss never rolls onto mine because the
    inflection link is severed."""
    db_path, mine_id, _min_id = _build_conflation_fixture(tmp_path)

    # Isolate the wyrd-v4rc events: collect over the committed file,
    # filter to old-english:mine so the other refs (which don't exist
    # in this minimal fixture) don't add noise to the apply counts.
    curation = {
        ref: payload
        for ref, payload in collect_curation_overrides([_CURATION_FILE]).items()
        if ref == "old-english:mine"
    }
    suppressions = {
        ref: payload
        for ref, payload in collect_gloss_suppressions([_CURATION_FILE]).items()
        if ref == "old-english:mine"
    }

    with LexiconDB(db_path) as db:
        cur_counts = apply_curation_overrides(db, curation, apply=True)
        sup_counts = apply_gloss_suppressions(db, suppressions, apply=True)

    assert cur_counts["lemma_id_cleared"] == 1
    assert sup_counts["glosses_dropped"] == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT lemma_id FROM etymon WHERE id = ?", (mine_id,)).fetchone()
    glosses = {
        r["gloss"]
        for r in conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (mine_id,))
    }
    conn.close()

    assert row["lemma_id"] is None, "lemma link to the 'my' pronoun must be cleared"
    assert glosses == {"a slope"}, (
        f"OE mine should carry only the toponym sense after disambiguation; got {glosses!r}"
    )


# ---------------------------------------------------------------------------
# Part A — sidecar surfaces extractive 'mine' as a distinct sense
# ---------------------------------------------------------------------------


def test_curated_senses_sidecar_loads_extractive_mine():
    """The curated-senses sidecar JSON loads and includes the
    extractive 'mine' entry. Slotted under ``old_french`` (its true
    origin) so it survives ``_rank_siblings`` next to the OE slope
    homograph — see test_extractive_mine_survives_rank_siblings."""
    _load_curated_senses.cache_clear()
    senses = _load_curated_senses()
    by_name = {e["name"]: e for e in senses}
    assert "mine" in by_name, "extractive 'mine' missing from curated_senses.json"
    mine = by_name["mine"]
    assert mine["language_field"] == "old_french", (
        "extractive mine must be slotted under old_french (historical stratum) so "
        "_rank_siblings doesn't drop it as a modern-english homograph"
    )
    assert "extracting" in mine["gloss"].lower()
    assert mine["modifier_type"] == "Topographical"


def test_curated_sense_subjects_emit_pre_and_post_variants():
    """Each curated sense produces pre- (``mine-``) and post-position
    (``mine``) subjects so it matches at either end of a compound."""
    _load_curated_senses.cache_clear()
    subjects = _curated_sense_subjects()
    mine_usages = {
        s["words"][0]["modern_usage"]
        for s in subjects
        if s["words"][0].get("old_french") == ["mine"]
    }
    assert mine_usages == {"mine-", "mine"}


def test_extractive_mine_surfaces_in_runtime_bundle():
    """End-to-end: the runtime meaning_db carries the extractive 'mine'
    sense as a distinct subject (old_french source, 'extracting' gloss)
    — separate from the OE slope sense. This is the user-visible fix:
    an English speaker querying 'mine' now gets the mineral-extraction
    reading, not only the OE 'slope'."""
    _load_meanings.cache_clear()
    mdb, _tag_db = _load_meanings()
    # Find the extractive sense at either surface key.
    extractive = []
    for key in ("mine", "mine-", "-mine-"):
        for m in mdb.get(key, []):
            if "old_french" in m.sources and any("extracting" in g.lower() for g in m.meanings):
                extractive.append(m)
    assert extractive, "extractive 'mine' sense (old_french) not found in runtime bundle"
    assert any("geology" in m.tags for m in extractive)


def test_extractive_mine_survives_rank_siblings_next_to_oe_slope():
    """The crux of Part A: ``_rank_siblings`` drops modern-english-only
    homographs when a historical etymon shares the surface (wyrd-ywm9
    'Green- → step/rung' pollution fix). Extractive mine is slotted
    under old_french precisely so it reads as a historical-stratum
    sibling and is NOT dropped when OE 'mine' (slope) is present.

    Fixture: OE slope (EN) + OF extractive (FR) at the same surface.
    Both must survive ranking; OE sorts first by stratum, OF rides
    along so the explainer can surface both senses."""
    oe_slope = Meaning("mine-", [], ["a slope"], {"old_english": ["mine"]})
    of_extractive = Meaning(
        "mine-", ["geology"], ["a mine; an excavation for ore"], {"old_french": ["mine"]}
    )
    ranked = _rank_siblings([oe_slope, of_extractive])
    ranked_sources = [list(m.sources)[0] for m in ranked]
    assert "old_french" in ranked_sources, (
        "extractive (old_french) sibling must survive _rank_siblings; a "
        "modern_english slot would have been dropped here"
    )
    assert "old_english" in ranked_sources
    # OE stratum sorts before OF.
    assert ranked_sources.index("old_english") < ranked_sources.index("old_french")


def test_modern_english_homograph_still_dropped_by_rank_siblings():
    """Regression guard: the old_french slotting fix must NOT defeat the
    wyrd-ywm9 homograph filter. A genuinely modern-english-only sibling
    at a surface with a historical etymon is still dropped — confirms
    we routed extractive mine through the historical-stratum path
    rather than weakening the filter itself."""
    oe_slope = Meaning("mine-", [], ["a slope"], {"old_english": ["mine"]})
    modern_noise = Meaning("mine-", [], ["a modern homograph"], {"modern_english": ["mine"]})
    ranked = _rank_siblings([oe_slope, modern_noise])
    ranked_sources = [list(m.sources)[0] for m in ranked]
    assert ranked_sources == ["old_english"], (
        f"modern-english-only homograph should be filtered when OE etymon present; "
        f"got {ranked_sources!r}"
    )


def test_extractive_mine_tag_is_user_facing():
    """The curated-sense tag ('geology') is a normal theme tag, NOT an
    internal-cohort tag like 'saint' — so the extractive sense shows up
    as a pickable morpheme rather than being hidden from the SPA tag
    dropdown."""
    _load_meanings.cache_clear()
    tags = available_tags()
    assert "geology" in tags, (
        "'geology' should be a user-facing theme tag; the extractive-mine sense "
        "must not be hidden behind an internal-cohort tag"
    )
