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

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.enrichment import apply_tag_additions, run_full_enrichment
from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.tag_mining import (
    SOURCE_REF,
    TAG_VOCAB,
    build_messages,
    collect_tags,
    existing_refs,
    parse_response,
    record,
    select_targets,
    source_row,
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
    # json.loads yields a non-dict (list / primitive) — must not AttributeError.
    assert parse_response('["military", "water"]') is None
    assert parse_response("42") is None


def test_parse_parents_stay_in_vocab():
    # Parent expansion is intersected with the vocab, so a hypothetical
    # out-of-vocab parent could never be injected. tree→plant, both in vocab.
    tags, _ = parse_response('{"tags": ["tree"]}')
    assert tags <= frozenset(TAG_VOCAB)


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


def test_existing_refs_reads_tag_rows_not_source(tmp_path: Path):
    p = tmp_path / "_tags.jsonl"
    _write_jsonl(
        p,
        [
            source_row(),  # the canonical source row must NOT enter the skip-set
            {"_type": "tags", "ref": "old-english:gar", "tags": ["military"]},
            {"_type": "tags", "ref": "old-english:bon", "tags": []},  # 'none' = resolved
        ],
    )
    assert existing_refs(str(p)) == {"old-english:gar", "old-english:bon"}
    assert SOURCE_REF not in existing_refs(str(p))
    assert existing_refs(str(tmp_path / "absent.jsonl")) == set()


def test_collect_tags_skips_none_and_error_keeps_tagged(tmp_path: Path):
    p = tmp_path / "_tags.jsonl"
    _write_jsonl(
        p,
        [
            {"ref": "old-english:gar", "tags": ["military", "personal-name"]},
            {"ref": "old-english:bon", "tags": []},  # none → skipped
            {"ref": "old-english:x", "error": "unparseable"},  # error → skipped
        ],
    )
    state = collect_tags(str(p))
    assert state == {"old-english:gar": {"tags": ["military", "personal-name"]}}


def test_collect_tags_last_write_wins(tmp_path: Path):
    p = tmp_path / "_tags.jsonl"
    _write_jsonl(
        p,
        [
            {"ref": "old-english:gar", "tags": ["military"]},
            {"ref": "old-english:gar", "tags": ["military", "personal-name"]},
        ],
    )
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
        conn.execute("INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, eid))
    conn.commit()
    return eid


def test_select_targets_only_untagged_glossed_reachable(tmp_path: Path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    _etymon(conn, "old-english", "gar", gloss="spear")  # ✓ target
    _etymon(conn, "old-english", "tun", gloss="enclosure", tag="architecture")  # ✗ tagged
    _etymon(conn, "old-english", "nogloss", gloss=None)  # ✗ no gloss
    _etymon(conn, "old-english", "orphan", gloss="x", reflex=False)  # ✗ not reachable
    conn.close()

    targets = select_targets(str(db_path))
    assert [(lang, form) for lang, form, _ in targets] == [("old-english", "gar")]
    assert targets[0][2] == "spear"


def _breakdown(conn, etymon_id: int, n_toponyms: int) -> None:
    """Add ``etymon_id`` as an element in ``n_toponyms`` distinct scholarly
    toponym_etymology breakdowns (wyrd-oth3 admit signal)."""
    conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('skeat', 'Skeat')")
    for i in range(n_toponyms):
        tid = conn.execute(
            "INSERT INTO toponym (modern_name) VALUES (?)", (f"Place{etymon_id}_{i}",)
        ).lastrowid
        teid = conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id) VALUES (?, 'skeat')", (tid,)
        ).lastrowid
        conn.execute(
            "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
            "VALUES (?, 1, ?)",
            (teid, etymon_id),
        )
    conn.commit()


def test_select_targets_includes_reflexless_breakdown_admit_cohort(tmp_path: Path):
    """wyrd-jrwc: a reflex-less morpheme used in >=2 scholarly toponym breakdowns
    is an admitted (oth3) generation surface and MUST be a tag target, even
    though the reflex-only gate would miss it. A morpheme in only 1 breakdown is
    below the realistic-admit threshold and stays excluded (myv4's tail)."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    admit = _etymon(
        conn, "old-english", "worth", gloss="enclosure", reflex=False
    )  # ✓ via breakdown
    _breakdown(conn, admit, n_toponyms=2)
    tail = _etymon(
        conn, "old-english", "rarewic", gloss="dwelling", reflex=False
    )  # ✗ only 1 breakdown
    _breakdown(conn, tail, n_toponyms=1)
    conn.close()

    targets = {(lang, form) for lang, form, _ in select_targets(str(db_path))}
    assert ("old-english", "worth") in targets  # reflex-less, but >=2 breakdowns → admitted
    assert ("old-english", "rarewic") not in targets  # only 1 breakdown → below threshold


def test_select_targets_excludes_already_tagged_breakdown_admit(tmp_path: Path):
    """The untagged filter gates BOTH reachability channels, not just the reflex
    one: an etymon used in >=2 toponym breakdowns but already carrying a tag must
    NOT be re-selected (the AND NOT EXISTS etymon_tag is outside the OR group)."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    tagged = _etymon(
        conn, "old-english", "burh", gloss="fortified place", reflex=False, tag="architecture"
    )
    _breakdown(conn, tagged, n_toponyms=2)
    conn.close()

    targets = {(lang, form) for lang, form, _ in select_targets(str(db_path))}
    assert ("old-english", "burh") not in targets  # >=2 breakdowns but already tagged → excluded


def _breakdowns_of_one_toponym(conn, etymon_id: int, n_breakdowns: int) -> None:
    """Add ``etymon_id`` as an element in ``n_breakdowns`` distinct scholarly
    breakdowns of a SINGLE shared toponym — so count(DISTINCT toponym_id) is 1
    however many breakdowns there are. Distinguishes the admit threshold's
    count(DISTINCT toponym_id) from a plain count(*) over breakdown rows."""
    conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('skeat', 'Skeat')")
    tid = conn.execute(
        "INSERT INTO toponym (modern_name) VALUES (?)", (f"Shared{etymon_id}",)
    ).lastrowid
    for _ in range(n_breakdowns):
        teid = conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id) VALUES (?, 'skeat')", (tid,)
        ).lastrowid
        conn.execute(
            "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
            "VALUES (?, 1, ?)",
            (teid, etymon_id),
        )
    conn.commit()


def test_select_targets_breakdown_admit_counts_distinct_toponyms_not_breakdowns(tmp_path: Path):
    """The admit threshold counts DISTINCT toponyms, not raw breakdown rows: an
    etymon in 2 breakdowns of the SAME toponym is 1 distinct toponym — below the
    >=2 threshold, so it stays excluded. Guards against a regression from
    count(DISTINCT te.toponym_id) to count(*)."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    eid = _etymon(conn, "old-english", "ham", gloss="homestead", reflex=False)
    _breakdowns_of_one_toponym(conn, eid, n_breakdowns=2)
    conn.close()

    targets = {(lang, form) for lang, form, _ in select_targets(str(db_path))}
    # 2 breakdown rows but 1 distinct toponym → below threshold → excluded
    assert ("old-english", "ham") not in targets


# ---------------------------------------------------------------------------
# apply_tag_additions — real tmp lexicon (D10)
# ---------------------------------------------------------------------------


def test_apply_adds_tags(tmp_path: Path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    _etymon(conn, "old-english", "gar", gloss="spear")
    conn.close()

    with LexiconDB(db_path) as db:
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

    with LexiconDB(db_path) as db:
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

    with LexiconDB(db_path) as db:
        counts = apply_tag_additions(db, {"old-english:gar": {"tags": ["military"]}}, apply=False)
        assert counts["tags_added"] == 1  # validated/counted
        assert db.conn.execute("SELECT COUNT(*) FROM etymon_tag").fetchone()[0] == 0


def test_apply_unresolved_ref_counted(tmp_path: Path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
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

    with LexiconDB(db_path) as db:
        result = run_full_enrichment(
            db,
            apply=True,
            tag_state={"old-english:gar": {"tags": ["military"]}},
            skip_l3_derivations=True,  # no bundle needed; slot runs before L3
        )
        assert result["tag_additions"]["tags_added"] == 1
        rows = {r[0] for r in db.conn.execute("SELECT tag FROM etymon_tag").fetchall()}
        assert rows == {"military"}


def test_apply_mixed_resolved_and_unresolved_batch(tmp_path: Path):
    """One call with both a resolvable and an unresolvable ref: accumulate the
    resolvable, count the other — don't abort the batch."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    _etymon(conn, "old-english", "gar", gloss="spear")
    conn.close()
    with LexiconDB(db_path) as db:
        counts = apply_tag_additions(
            db,
            {"old-english:gar": {"tags": ["military"]}, "old-english:ghost": {"tags": ["water"]}},
            apply=True,
        )
        assert counts["tags_added"] == 1 and counts["unresolved_etymon"] == 1
        assert {r[0] for r in db.conn.execute("SELECT tag FROM etymon_tag")} == {"military"}


# ---------------------------------------------------------------------------
# _tags.jsonl L2 round-trip — the rebuild must NOT crash (wyrd-xz3g P1)
# ---------------------------------------------------------------------------


def test_source_row_is_canonical_source(tmp_path: Path):
    sr = source_row()
    assert sr["_type"] == "source" and sr["ref"] == SOURCE_REF


def test_collect_tags_skips_source_row(tmp_path: Path):
    p = tmp_path / "_tags.jsonl"
    _write_jsonl(p, [source_row(), {"ref": "old-english:gar", "tags": ["military"]}])
    # the source row has no `tags` → skipped; only the real tag row survives.
    assert collect_tags(str(p)) == {"old-english:gar": {"tags": ["military"]}}


def test_build_from_jsonl_accepts_tags_file(tmp_path: Path):
    """rebuild-from-jsonl globs EVERY *.jsonl and replays it; a _tags.jsonl with
    an unregistered _type (or no source row) crashes the whole rebuild. This
    pins that a well-formed _tags.jsonl replays inert without raising."""
    d = tmp_path / "mining"
    d.mkdir()
    # a minimal valid per-source L2 file (source + one etymon)
    _write_jsonl(
        d / "src.jsonl",
        [
            {"_type": "source", "ref": "testsrc", "title": "T"},
            {
                "_type": "etymon",
                "ref": "old-english:gar",
                "canonical_form": "gar",
                "language": "old-english",
            },
        ],
    )
    # a _tags.jsonl exactly as mine-tags-llm writes it: source row + tag rows
    _write_jsonl(
        d / "_tags.jsonl",
        [
            source_row(),
            {
                "_type": "tags",
                "ref": "old-english:gar",
                "language": "old-english",
                "form": "gar",
                "model": "gemma4:26b",
                "method": "llm-tags-v1",
                "tags": ["military"],
                "confidence": "high",
            },
        ],
    )
    db_path = tmp_path / "l.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        # Must not raise (pre-fix: ReplayError 'unknown _type tags').
        build_from_jsonl(db.conn, jsonl_paths_in(d))


def test_collect_tags_tolerates_corrupt_line(tmp_path: Path):
    p = tmp_path / "_tags.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"_type": "tags", "ref": "old-english:gar", "tags": ["military"]}) + "\n"
        )
        fh.write("{ truncated junk\n")  # crash-interrupted flush
    # skips the bad line rather than aborting the enrichment replay.
    assert collect_tags(str(p)) == {"old-english:gar": {"tags": ["military"]}}
    assert existing_refs(str(p)) == {"old-english:gar"}


def test_collect_tags_later_none_does_not_retract(tmp_path: Path):
    """Sticky-by-design: a later 'none' row for a ref does NOT clear an earlier
    tagged decision (none rows are skipped, so the last TAGGED row wins)."""
    p = tmp_path / "_tags.jsonl"
    _write_jsonl(
        p,
        [
            {"ref": "old-english:gar", "tags": ["military"]},
            {"ref": "old-english:gar", "tags": []},  # later 'none'
        ],
    )
    assert collect_tags(str(p)) == {"old-english:gar": {"tags": ["military"]}}


def test_select_targets_concatenates_multiple_glosses(tmp_path: Path):
    """GROUP_CONCAT folds a multi-gloss etymon into one ' || '-joined string —
    exactly the text the LLM classifies."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    eid = _etymon(conn, "old-english", "gar", gloss="spear")
    conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, "javelin"))
    conn.commit()
    conn.close()
    ((_lang, _form, glosses),) = select_targets(str(db_path))
    # Glosses sorted in the subquery → deterministic concat ('javelin' < 'spear')
    # regardless of insertion order, so the prompt (and _tags.jsonl) is stable.
    assert glosses == "javelin || spear"


def test_select_targets_no_gloss_duplication_with_multiple_reflexes(tmp_path: Path):
    """An etymon with MULTIPLE reflexes must not Cartesian-product the glosses:
    reachability is an EXISTS check, not a JOIN, so each gloss appears once."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    eid = _etymon(conn, "old-english", "gar", gloss="spear")  # 1st reflex via helper
    rid = conn.execute(
        "INSERT INTO reflex (surface_form, position) VALUES ('gar-', 'pre')"
    ).lastrowid
    conn.execute("INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, eid))
    conn.commit()
    conn.close()
    targets = select_targets(str(db_path))
    assert len(targets) == 1  # one row, not one-per-reflex
    assert targets[0][2] == "spear"  # gloss NOT duplicated to 'spear || spear'


# ---------------------------------------------------------------------------
# mine-tags-llm CLI — thin test with a stubbed Ollama client (no network)
# ---------------------------------------------------------------------------


def test_mine_tags_llm_cli(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    _etymon(conn, "old-english", "gar", gloss="spear")  # → tagged
    _etymon(conn, "old-english", "del", gloss="a part of")  # → none
    conn.close()

    # Stub the Ollama client so the CLI runs offline + deterministically.
    from wyrd.generators.kenning.cli.lexicon import mine_tags_llm as mod

    class _StubClient:
        def __init__(self, *a, **k):
            pass

        def chat_json(self, system, user, schema):
            return {"tags": ["military"], "confidence": "high"} if "spear" in user else {"tags": []}

    monkeypatch.setattr(mod, "OllamaClient", _StubClient)

    out = tmp_path / "_tags.jsonl"
    result = CliRunner().invoke(
        cli_root,
        ["lexicon", "mine-tags-llm", "--db", str(db_path), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    # first line is the canonical source row; then one tagged + one none row.
    assert lines[0]["_type"] == "source" and lines[0]["ref"] == SOURCE_REF
    assert collect_tags(str(out)) == {"old-english:gar": {"tags": ["military"]}}

    # --skip-resolved: a second run has nothing to do (both refs recorded).
    result2 = CliRunner().invoke(
        cli_root,
        ["lexicon", "mine-tags-llm", "--db", str(db_path), "--output", str(out)],
    )
    assert result2.exit_code == 0
    assert "Nothing to do" in result2.output


def test_mine_tags_llm_api_failure_not_recorded(tmp_path: Path, monkeypatch):
    """An API/network failure must NOT write a record (an error row would be
    skip-resolved next run, permanently dropping the etymon). The ref stays
    retryable, and the run aborts after _MAX_CONSECUTIVE_FAILS."""
    from wyrd.generators.kenning.cli.lexicon import mine_tags_llm as mod

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    for n in range(15):
        _etymon(conn, "old-english", f"e{n}", gloss="spear")
    conn.close()

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def chat_json(self, *a, **k):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(mod, "OllamaClient", _Boom)
    monkeypatch.setattr(mod, "_MAX_CONSECUTIVE_FAILS", 3)

    out = tmp_path / "_tags.jsonl"
    result = CliRunner().invoke(
        cli_root, ["lexicon", "mine-tags-llm", "--db", str(db_path), "--output", str(out)]
    )
    assert result.exit_code == 0
    assert "Aborting after 3 consecutive API failures" in result.output
    # Only the source row was written — NO etymon records, so every ref is
    # still retryable on the next run.
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert [row["_type"] for row in lines] == ["source"]
    assert existing_refs(str(out)) == set()


def test_select_todo_filters_resolved_limits_and_counts(monkeypatch, tmp_path):
    """``_select_todo`` (extracted from the mine-tags-llm command body) drops
    already-resolved refs when ``skip_resolved``, caps the list at ``limit``, and
    returns the raw target / resolved counts for the run's plan line."""
    from wyrd.generators.kenning.cli.lexicon import mine_tags_llm as mt

    targets = [("oe", "tun", "settlement"), ("welsh", "llan", "church"), ("oe", "ham", "home")]
    monkeypatch.setattr(mt.tag_mining, "select_targets", lambda _p: list(targets))
    monkeypatch.setattr(mt.tag_mining, "existing_refs", lambda _p: {"oe:tun"})
    db, out = tmp_path / "l.db", tmp_path / "_tags.jsonl"

    # skip_resolved drops the oe:tun ref; counts are the RAW totals.
    todo, n_targets, n_resolved = mt._select_todo(db, out, skip_resolved=True, limit=None)
    assert [(lang, form) for lang, form, _ in todo] == [("welsh", "llan"), ("oe", "ham")]
    assert (n_targets, n_resolved) == (3, 1)

    # skip_resolved=False ignores the resolved set entirely (count 0, nothing dropped).
    todo_all, _, n_res_off = mt._select_todo(db, out, skip_resolved=False, limit=None)
    assert len(todo_all) == 3 and n_res_off == 0

    # limit caps the POST-filter list.
    todo_capped, _, _ = mt._select_todo(db, out, skip_resolved=True, limit=1)
    assert todo_capped == [("welsh", "llan", "church")]
