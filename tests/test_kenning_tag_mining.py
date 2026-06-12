"""Tests for wyrd-xz3g: LLM tag-coverage backfill.

Covers the pure classification logic (vocab filter, alias normalization,
parent expansion, the mandatory 'none' outcome), the L2 ``_tags.jsonl``
round-trip (``existing_refs`` / ``collect_tags``), the ``select_targets``
scope query, and the ``apply_tag_additions`` enrichment pass against a real
tmp lexicon (D10: no mocking the data layer).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from wyrd.generators.kenning.enrichment import apply_tag_additions, run_full_enrichment
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.tag_mining import (
    TAG_VOCAB,
    build_messages,
    collect_tags,
    existing_refs,
    parse_response,
    record,
    select_targets,
)

# ---------------------------------------------------------------------------
# parse_response — vocab filter, aliases, parents, 'none', unparseable
# ---------------------------------------------------------------------------


def test_parse_filters_to_vocab_and_keeps_known():
    tags, conf = parse_response('{"tags": ["military", "not_a_tag"], "confidence": "high"}')
    assert tags == frozenset({"military"})
    assert conf == "high"


def test_parse_normalizes_aliases():
    # body→anatomy, topology→topography (operator fold + typo canonicalization)
    tags, _ = parse_response('{"tags": ["body", "topology"]}')
    assert tags == frozenset({"anatomy", "topography"})


def test_parse_expands_parents():
    # tree⊂plant, bird⊂animal
    tags, _ = parse_response('{"tags": ["tree", "bird"]}')
    assert tags == frozenset({"tree", "plant", "bird", "animal"})


def test_parse_kinship_is_in_vocab():
    tags, _ = parse_response('{"tags": ["kinship"]}')
    assert tags == frozenset({"kinship"})
    assert "kinship" in TAG_VOCAB


def test_parse_none_outcome_is_empty_not_failure():
    tags, conf = parse_response('{"tags": [], "confidence": "high"}')
    assert tags == frozenset()
    assert conf == "high"


def test_parse_confidence_clamped_and_defaulted():
    _, conf = parse_response('{"tags": ["water"], "confidence": "bogus"}')
    assert conf == "low"
    _, conf2 = parse_response('{"tags": ["water"]}')
    assert conf2 == "low"


def test_parse_unparseable_returns_none():
    assert parse_response("not json") is None
    assert parse_response('{"tags": "military"}') is None  # tags not a list
    assert parse_response(None) is None


def test_parse_accepts_dict_input():
    tags, _ = parse_response({"tags": ["water"], "confidence": "medium"})
    assert tags == frozenset({"water"})


# ---------------------------------------------------------------------------
# build_messages / record
# ---------------------------------------------------------------------------


def test_build_messages_carries_vocab_and_gloss():
    system, user = build_messages("a hill")
    assert "kinship" in system and "topography" in system
    assert "PRECISION" in system  # the precision rule survives edits
    assert "a hill" in user


def test_record_shapes():
    tagged = record("old-english", "gar", (frozenset({"military"}), "high"), "gemma4:26b")
    assert tagged["ref"] == "old-english:gar"
    assert tagged["tags"] == ["military"]
    assert tagged["confidence"] == "high"
    assert tagged["method"] == "llm-tags-v1"

    none = record("old-english", "bon", (frozenset(), "high"), "gemma4:26b")
    assert none["tags"] == [] and "error" not in none

    bad = record("old-english", "x", None, "gemma4:26b")
    assert bad["error"] == "unparseable" and "tags" not in bad


# ---------------------------------------------------------------------------
# L2 round-trip: existing_refs / collect_tags
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_existing_refs_reads_all_refs(tmp_path: Path):
    p = tmp_path / "_tags.jsonl"
    _write_jsonl(p, [
        {"ref": "old-english:gar", "tags": ["military"]},
        {"ref": "old-english:bon", "tags": []},  # 'none' still counts as resolved
    ])
    assert existing_refs(str(p)) == {"old-english:gar", "old-english:bon"}
    assert existing_refs(str(tmp_path / "absent.jsonl")) == set()


def test_collect_tags_skips_none_and_error_keeps_tagged(tmp_path: Path):
    p = tmp_path / "_tags.jsonl"
    _write_jsonl(p, [
        {"ref": "old-english:gar", "tags": ["military", "personal-name"]},
        {"ref": "old-english:bon", "tags": []},          # none → skipped
        {"ref": "old-english:x", "error": "unparseable"},  # error → skipped
    ])
    state = collect_tags(str(p))
    assert state == {"old-english:gar": {"tags": ["military", "personal-name"]}}


def test_collect_tags_last_write_wins(tmp_path: Path):
    p = tmp_path / "_tags.jsonl"
    _write_jsonl(p, [
        {"ref": "old-english:gar", "tags": ["military"]},
        {"ref": "old-english:gar", "tags": ["military", "personal-name"]},
    ])
    assert collect_tags(str(p))["old-english:gar"]["tags"] == ["military", "personal-name"]


# ---------------------------------------------------------------------------
# select_targets — gloss-but-no-tag, generation-reachable (reflex_etymon)
# ---------------------------------------------------------------------------


def _etymon(conn, language, form, *, gloss=None, tag=None, reflex=True) -> int:
    eid = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid
    if gloss:
        conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, gloss))
    if tag:
        conn.execute("INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, ?)", (eid, tag))
    if reflex:
        rid = conn.execute(
            "INSERT INTO reflex (surface_form, position) VALUES (?, 'post')", (form,)
        ).lastrowid
        conn.execute(
            "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, eid)
        )
    conn.commit()
    return eid


def test_select_targets_only_untagged_glossed_reachable(tmp_path: Path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    _etymon(conn, "old-english", "gar", gloss="spear")               # ✓ target
    _etymon(conn, "old-english", "tun", gloss="enclosure", tag="architecture")  # ✗ tagged
    _etymon(conn, "old-english", "nogloss", gloss=None)              # ✗ no gloss
    _etymon(conn, "old-english", "orphan", gloss="x", reflex=False)  # ✗ not reachable
    conn.close()

    targets = select_targets(str(db_path))
    assert [(lang, form) for lang, form, _ in targets] == [("old-english", "gar")]
    assert targets[0][2] == "spear"


# ---------------------------------------------------------------------------
# apply_tag_additions — real tmp lexicon (D10)
# ---------------------------------------------------------------------------


def test_apply_adds_tags(tmp_path: Path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    _etymon(conn, "old-english", "gar", gloss="spear")
    conn.close()

    db = LexiconDB(db_path)
    counts = apply_tag_additions(
        db, {"old-english:gar": {"tags": ["military", "personal-name"]}}, apply=True
    )
    assert counts["tags_added"] == 2
    assert counts["unresolved_etymon"] == 0
    rows = {r[0] for r in db.conn.execute("SELECT tag FROM etymon_tag").fetchall()}
    assert rows == {"military", "personal-name"}


def test_apply_idempotent(tmp_path: Path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    _etymon(conn, "old-english", "gar", gloss="spear")
    conn.close()

    db = LexiconDB(db_path)
    state = {"old-english:gar": {"tags": ["military"]}}
    apply_tag_additions(db, state, apply=True)
    counts = apply_tag_additions(db, state, apply=True)
    assert counts["tags_added"] == 0
    assert counts["tags_already_present"] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM etymon_tag").fetchone()[0] == 1


def test_apply_dry_run_no_writes(tmp_path: Path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    _etymon(conn, "old-english", "gar", gloss="spear")
    conn.close()

    db = LexiconDB(db_path)
    counts = apply_tag_additions(db, {"old-english:gar": {"tags": ["military"]}}, apply=False)
    assert counts["tags_added"] == 1  # validated/counted
    assert db.conn.execute("SELECT COUNT(*) FROM etymon_tag").fetchone()[0] == 0


def test_apply_unresolved_ref_counted(tmp_path: Path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    counts = apply_tag_additions(db, {"old-english:ghost": {"tags": ["water"]}}, apply=True)
    assert counts["unresolved_etymon"] == 1
    assert counts["tags_added"] == 0


def test_run_full_enrichment_wires_tag_slot(tmp_path: Path):
    """End-to-end: tag_state passed to run_full_enrichment lands in etymon_tag
    via the apply-tag-additions curation slot (pin the slot registration)."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    _etymon(conn, "old-english", "gar", gloss="spear")
    conn.close()

    db = LexiconDB(db_path)
    result = run_full_enrichment(
        db,
        apply=True,
        tag_state={"old-english:gar": {"tags": ["military"]}},
        skip_l3_derivations=True,  # no bundle needed; slot runs before L3
    )
    assert result["tag_additions"]["tags_added"] == 1
    rows = {r[0] for r in db.conn.execute("SELECT tag FROM etymon_tag").fetchall()}
    assert rows == {"military"}
