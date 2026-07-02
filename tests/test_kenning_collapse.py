"""apply_collapses + collect_collapses — wyrd-y651 (consolidation epic P2).

The shared collapse applier: fold a form-of / variant etymon into its
lemma (tombstone via merged_into_id, register the variant, migrate
reflexes + citations stamping attested_form).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from wyrd.generators.kenning.enrichment import apply_collapses
from wyrd.generators.kenning.jsonl.build import (
    _resolve_collapse_cycles,
    collect_collapses,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


def _seed(db_path: Path) -> dict[str, int]:
    """burh (form-of, pointer gloss) + burg (real lemma). burh carries a
    scholarly citation + a reflex; burg carries the real gloss."""
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO source (id, title) VALUES ('ekwall', 'Ekwall 1960')")
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'burh', 'old-english')"
    )
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (2, 'burg', 'old-english')"
    )
    conn.execute(
        "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (1, 'alternative form of burg')"
    )
    conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (2, 'fort')")
    # scholarly citation on the form-of etymon → must migrate to the lemma
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (1, 'ekwall', 'p12')"
    )
    # reflex linked to the form-of etymon → must migrate to the lemma
    cur = conn.execute(
        "INSERT INTO reflex (surface_form, position, productivity) VALUES ('-burh', 'post', 0)"
    )
    rid = cur.lastrowid
    conn.execute("INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, 1)", (rid,))
    conn.commit()
    conn.close()
    return {"burh": 1, "burg": 2, "reflex": rid}


_COLLAPSE = {
    "old-english:burh": {
        "into": "old-english:burg",
        "variant_class": "alternative",
        "method": "deterministic-gloss-overlap",
    }
}


def test_apply_collapses_folds_form_into_lemma(tmp_path: Path) -> None:
    ids = _seed(tmp_path / "lex.db")
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, _COLLAPSE, apply=True)
        conn = db.conn
        # 1. tombstoned into the lemma
        merged = conn.execute("SELECT merged_into_id FROM etymon WHERE id = 1").fetchone()[0]
        assert merged == ids["burg"]
        # 2. form registered as a variant of the lemma
        var = conn.execute(
            "SELECT variant_class FROM etymon_variant WHERE etymon_id = 2 AND form = 'burh'"
        ).fetchone()
        assert var is not None and var[0] == "alternative"
        # 3. reflex migrated to the lemma
        assert (
            conn.execute(
                "SELECT etymon_id FROM reflex_etymon WHERE reflex_id = ?", (ids["reflex"],)
            ).fetchone()[0]
            == ids["burg"]
        )
        # 4. citation migrated to the lemma AND stamped with the attested form
        cit = conn.execute("SELECT etymon_id, attested_form FROM etymon_citation").fetchone()
        assert cit[0] == ids["burg"] and cit[1] == "burh"
    assert counts["collapses_processed"] == 1
    assert counts["variants_created"] == 1
    assert counts["reflexes_moved"] == 1
    assert counts["citations_moved"] == 1


def test_apply_collapses_is_idempotent(tmp_path: Path) -> None:
    _seed(tmp_path / "lex.db")
    with LexiconDB(tmp_path / "lex.db") as db:
        apply_collapses(db, _COLLAPSE, apply=True)
        second = apply_collapses(db, _COLLAPSE, apply=True)
    # `from` is now tombstoned → _resolve (which excludes merged rows)
    # no longer finds it, so the second run is a clean unresolved-skip.
    assert second["collapses_processed"] == 0
    assert second["unresolved_from"] == 1


def test_collapse_fold_then_rescreen_revert_unfolds_on_replay(tmp_path: Path) -> None:
    """wyrd-21p8: appending an ``into: ""`` revert row AFTER a fold row for the same
    ref cancels the fold on replay — ``collect_collapses`` is last-write-wins per
    ref, so the ref's merged state is ``into: ""``, ``apply_collapses`` no-ops, and
    the etymon is NEVER tombstoned (``merged_into_id`` stays NULL). This is the
    un-tombstone mechanism the variant-fold rescreen relies on: the L2 ledger is the
    fix; the live DB heals on the next full rebuild/replay."""
    ids = _seed(tmp_path / "lex.db")
    ledger = tmp_path / "_collapses.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "_type": "collapse",
                "ref": "old-english:burh",
                "into": "old-english:burg",
                "variant_class": "alternative",
                "method": "llm-variant-fold-v1",
            }
        )
        + "\n"
        + json.dumps(
            {
                "_type": "collapse",
                "ref": "old-english:burh",
                "into": "",
                "variant_class": "alternative",
                "method": "llm-variant-fold-rescreen-v1",
                "reason": "distinct morphemes wrongly folded on framing/surface",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = collect_collapses([ledger])
    assert state["old-english:burh"]["into"] == ""  # revert wins (last-write)
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, state, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (ids["burh"],)
        ).fetchone()[0]
    assert merged is None  # NOT tombstoned — the fold was reverted
    assert counts["collapses_processed"] == 0
    assert counts["empty_into_skipped"] == 1


def test_collapse_rescreen_revert_replay_is_idempotent(tmp_path: Path) -> None:
    """The reverted (no-op) ledger state re-applies with zero effect — a second
    replay of the corrected ledger changes nothing and keeps ``merged_into_id``
    NULL (idempotency of the un-fold path)."""
    ids = _seed(tmp_path / "lex.db")
    ledger = tmp_path / "_collapses.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "_type": "collapse",
                "ref": "old-english:burh",
                "into": "old-english:burg",
                "method": "llm-variant-fold-v1",
            }
        )
        + "\n"
        + json.dumps(
            {
                "_type": "collapse",
                "ref": "old-english:burh",
                "into": "",
                "method": "llm-variant-fold-rescreen-v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = collect_collapses([ledger])
    with LexiconDB(tmp_path / "lex.db") as db:
        apply_collapses(db, state, apply=True)
        second = apply_collapses(db, state, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (ids["burh"],)
        ).fetchone()[0]
    assert merged is None
    assert second["collapses_processed"] == 0
    assert second["empty_into_skipped"] == 1


def test_apply_collapses_dry_run_writes_nothing(tmp_path: Path) -> None:
    _seed(tmp_path / "lex.db")
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, _COLLAPSE, apply=False)
        merged = db.conn.execute("SELECT merged_into_id FROM etymon WHERE id = 1").fetchone()[0]
    assert counts["collapses_processed"] == 1  # validated
    assert counts["applied"] is False
    assert merged is None  # but not written


def test_apply_collapses_skips_self_and_unresolved(tmp_path: Path) -> None:
    _seed(tmp_path / "lex.db")
    state = {
        "old-english:burg": {"into": "old-english:burg"},  # self
        "old-english:nope": {"into": "old-english:burg"},  # unresolved from
        "old-english:burh": {"into": "old-english:ghost"},  # unresolved into
    }
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, state, apply=True)
    assert counts["self_collapse_skipped"] == 1
    assert counts["unresolved_from"] == 1
    assert counts["unresolved_into"] == 1
    assert counts["collapses_processed"] == 0


def test_reflex_conflict_left_on_tombstone(tmp_path: Path) -> None:
    """A reflex `into` ALREADY carries can't move (PK conflict) — it stays
    on the tombstone, counted, not lost."""
    ids = _seed(tmp_path / "lex.db")
    conn = sqlite3.connect(tmp_path / "lex.db")
    # a shared reflex linked to BOTH burh and burg
    cur = conn.execute(
        "INSERT INTO reflex (surface_form, position, productivity) VALUES ('-shared', 'post', 0)"
    )
    shared = cur.lastrowid
    conn.execute("INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, 1)", (shared,))
    conn.execute("INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, 2)", (shared,))
    conn.commit()
    conn.close()
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, _COLLAPSE, apply=True)
        # the shared reflex still has its burh link (conflict on move)
        rows = db.conn.execute(
            "SELECT etymon_id FROM reflex_etymon WHERE reflex_id = ? ORDER BY etymon_id",
            (shared,),
        ).fetchall()
    assert counts["reflexes_skipped_conflict"] == 1
    assert [r[0] for r in rows] == [1, 2]  # burh link survives on the tombstone
    _ = ids


def test_citation_conflict_left_on_tombstone(tmp_path: Path) -> None:
    """Citation twin of the reflex-conflict case: a cite `into` already
    carries on the same UNIQUE (etymon_id, source_id, COALESCE(page,'')) key
    can't move — it stays on the tombstone, counted, not lost. Pins the
    citations_skipped_conflict accounting + the attested_form IS NULL guard in
    _migrate_collapse_evidence (wyrd-8uvi)."""
    ids = _seed(tmp_path / "lex.db")
    conn = sqlite3.connect(tmp_path / "lex.db")
    # burg (the lemma) already carries the SAME (ekwall, p12) cite burh has.
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (2, 'ekwall', 'p12')"
    )
    conn.commit()
    conn.close()
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, _COLLAPSE, apply=True)
        # burh's cite couldn't move (conflict) → still on the tombstone, unstamped.
        burh_cit = db.conn.execute(
            "SELECT source_id, page, attested_form FROM etymon_citation WHERE etymon_id = 1"
        ).fetchone()
    assert counts["citations_skipped_conflict"] == 1
    assert counts["citations_moved"] == 0
    assert burh_cit is not None and burh_cit[0] == "ekwall" and burh_cit[1] == "p12"
    assert burh_cit[2] is None  # never stamped — the move was skipped
    _ = ids


def test_bad_variant_class_coerced_to_other(tmp_path: Path) -> None:
    """An out-of-vocabulary variant_class (hand-edited / LLM row) is
    coerced to 'other' rather than tripping the CHECK constraint."""
    _seed(tmp_path / "lex.db")
    state = {"old-english:burh": {"into": "old-english:burg", "variant_class": "bogus-class"}}
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, state, apply=True)
        vc = db.conn.execute(
            "SELECT variant_class FROM etymon_variant WHERE etymon_id = 2 AND form = 'burh'"
        ).fetchone()[0]
    assert counts["collapses_processed"] == 1
    assert vc == "other"


def _seed_homograph(db_path: Path) -> dict[str, int]:
    """A homograph-conflated row (wyrd-qp9c): OE ``don`` is BOTH the toponym
    "hill" (a reduced form of the lemma ``dūn``) AND carries the verb "to do"
    lineage — a parent edge from PGmc ``*dōn`` and a child edge to ModE ``do``.
    The detach op must delete those two verb edges before the fold so
    cluster_cognates never redirects the verb lineage onto ``dūn``."""
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO source (id, title) VALUES ('wikt', 'Wiktionary')")
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'don', 'old-english')"
    )
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (2, 'dūn', 'old-english')"
    )
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (3, '*dōn', 'proto-germanic')"
    )
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (4, 'do', 'modern-english')"
    )
    # the two contaminating verb edges: *dōn -> don, don -> do
    conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
        "VALUES (3, 1, 'inheritance', 'wikt')"
    )
    conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
        "VALUES (1, 4, 'inheritance', 'wikt')"
    )
    conn.commit()
    conn.close()
    return {"don": 1, "dūn": 2, "verb_parent": 3, "verb_child": 4}


_DETACH_FOLD = {
    "old-english:don": {
        "into": "old-english:dūn",
        "variant_class": "alternative",
        "method": "manual-homograph-detach",
        "detach_parents": ["proto-germanic:*dōn"],
        "detach_children": ["modern-english:do"],
    }
}


def test_apply_collapses_detach_removes_edges_before_fold(tmp_path: Path) -> None:
    ids = _seed_homograph(tmp_path / "lex.db")
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, _DETACH_FOLD, apply=True)
        conn = db.conn
        # both verb edges deleted
        assert conn.execute("SELECT COUNT(*) FROM etymon_descent").fetchone()[0] == 0
        # no edge references the folded toponym → nothing for cluster_cognates
        # to redirect onto the lemma via merged_into_id
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM etymon_descent WHERE parent_id = ? OR child_id = ?",
                (ids["don"], ids["don"]),
            ).fetchone()[0]
            == 0
        )
        # and the toponym is folded into the clean lemma
        merged = conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (ids["don"],)
        ).fetchone()[0]
        assert merged == ids["dūn"]
    assert counts["detach_edges_removed"] == 2
    assert counts["collapses_processed"] == 1
    assert counts["detach_unresolved_endpoint"] == 0


def test_apply_collapses_detach_only_row_no_fold(tmp_path: Path) -> None:
    """A row with detach but no ``into`` removes the edges without tombstoning
    the from etymon (and is NOT counted as an empty-into skip)."""
    ids = _seed_homograph(tmp_path / "lex.db")
    state = {
        "old-english:don": {
            "detach_parents": ["proto-germanic:*dōn"],
            "detach_children": ["modern-english:do"],
        }
    }
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, state, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (ids["don"],)
        ).fetchone()[0]
        edges = db.conn.execute("SELECT COUNT(*) FROM etymon_descent").fetchone()[0]
    assert counts["detach_edges_removed"] == 2
    assert counts["collapses_processed"] == 0
    assert counts["empty_into_skipped"] == 0  # detach-only is not an empty skip
    assert merged is None  # no fold
    assert edges == 0


def test_apply_collapses_detach_dry_run_counts_without_deleting(tmp_path: Path) -> None:
    _seed_homograph(tmp_path / "lex.db")
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, _DETACH_FOLD, apply=False)
        edges = db.conn.execute("SELECT COUNT(*) FROM etymon_descent").fetchone()[0]
    assert counts["detach_edges_removed"] == 2  # counted
    assert edges == 2  # but nothing deleted


def test_apply_collapses_detach_unresolved_endpoint_counted(tmp_path: Path) -> None:
    """A detach endpoint that doesn't resolve is counted, not fatal; the
    resolvable edge is still removed."""
    _seed_homograph(tmp_path / "lex.db")
    state = {
        "old-english:don": {
            "into": "old-english:dūn",
            "detach_children": ["modern-english:do", "modern-english:ghost"],
        }
    }
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, state, apply=True)
    assert counts["detach_edges_removed"] == 1
    assert counts["detach_unresolved_endpoint"] == 1
    assert counts["collapses_processed"] == 1


def test_apply_collapses_detach_resolves_tombstoned_endpoint(tmp_path: Path) -> None:
    """A detach endpoint that is itself tombstoned (folded by an earlier row)
    still resolves — etymon_descent edges reference ids regardless of
    merged_into_id, so the wrong-sense edge must still be deleted, not silently
    counted unresolved (the _resolve `merged_into_id IS NULL` filter would miss
    it)."""
    ids = _seed_homograph(tmp_path / "lex.db")
    conn = sqlite3.connect(tmp_path / "lex.db")
    # tombstone the verb child `do` (4) into `dūn` (2) — simulating a prior fold
    conn.execute(
        "UPDATE etymon SET merged_into_id = ? WHERE id = ?", (ids["dūn"], ids["verb_child"])
    )
    conn.commit()
    conn.close()
    state = {"old-english:don": {"detach_children": ["modern-english:do"]}}
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, state, apply=True)
        # the don -> do edge is gone despite `do` being tombstoned
        remaining = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_descent WHERE parent_id = ? AND child_id = ?",
            (ids["don"], ids["verb_child"]),
        ).fetchone()[0]
    assert remaining == 0
    assert counts["detach_edges_removed"] == 1
    assert counts["detach_unresolved_endpoint"] == 0


def test_apply_collapses_detach_resolves_tombstoned_from(tmp_path: Path) -> None:
    """A tombstoned `from` (already folded) still has its wrong-sense edges
    detached, and is NOT miscounted as detach_unresolved_from — that counter is
    reserved for genuine typos (ref matches no etymon)."""
    ids = _seed_homograph(tmp_path / "lex.db")
    conn = sqlite3.connect(tmp_path / "lex.db")
    conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (ids["dūn"], ids["don"]))
    conn.commit()
    conn.close()
    state = {
        "old-english:don": {
            "detach_parents": ["proto-germanic:*dōn"],
            "detach_children": ["modern-english:do"],
        }
    }
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, state, apply=True)
        edges = db.conn.execute("SELECT COUNT(*) FROM etymon_descent").fetchone()[0]
    assert edges == 0  # both edges removed even though `don` is a tombstone
    assert counts["detach_edges_removed"] == 2
    assert counts["detach_unresolved_from"] == 0


def test_apply_collapses_detach_unresolved_from_is_genuine_typo(tmp_path: Path) -> None:
    """detach_unresolved_from fires only when the ref matches no etymon at all."""
    _seed_homograph(tmp_path / "lex.db")
    state = {"old-english:nope": {"detach_children": ["modern-english:do"]}}
    with LexiconDB(tmp_path / "lex.db") as db:
        counts = apply_collapses(db, state, apply=True)
        edges = db.conn.execute("SELECT COUNT(*) FROM etymon_descent").fetchone()[0]
    assert counts["detach_unresolved_from"] == 1
    assert counts["detach_edges_removed"] == 0
    assert edges == 2  # nothing touched


def test_collect_collapses_preserves_detach_fields(tmp_path: Path) -> None:
    """detach_parents / detach_children survive replay + cycle resolution."""
    p = tmp_path / "_collapses.jsonl"
    rows = [
        {
            "_type": "collapse",
            "ref": "old-english:don",
            "into": "old-english:dūn",
            "detach_parents": ["gmw-pro:*dōn"],
            "detach_children": ["modern-english:do", "middle-english:don"],
            "method": "manual-homograph-detach",
        },
        # the wrong-direction prior fold, reverted to a no-op
        {"_type": "collapse", "ref": "old-english:dūn", "into": ""},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    state = collect_collapses([p])
    don = state["old-english:don"]
    assert don["into"] == "old-english:dūn"  # survives (dūn -> '' is a no-op, no cycle)
    assert don["detach_parents"] == ["gmw-pro:*dōn"]
    assert don["detach_children"] == ["modern-english:do", "middle-english:don"]
    assert state["old-english:dūn"]["into"] == ""


def test_curate_collapse_etymon_cli_emits_row(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.curate_collapse import (
        lexicon_curate_collapse_etymon,
    )

    ledger = tmp_path / "_collapses.jsonl"
    runner = CliRunner()
    res = runner.invoke(
        lexicon_curate_collapse_etymon,
        [
            "old-english:don",
            "--into",
            "old-english:dūn",
            "--detach-parent",
            "gmw-pro:*dōn",
            "--detach-child",
            "modern-english:do",
            "--collapse-file",
            str(ledger),
        ],
    )
    assert res.exit_code == 0, res.output
    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    # first line is the synthetic source row; second is our event
    assert lines[0]["_type"] == "source" and lines[0]["ref"] == "collapse"
    row = lines[1]
    assert row["_type"] == "collapse"
    assert row["ref"] == "old-english:don"
    assert row["into"] == "old-english:dūn"
    assert row["detach_parents"] == ["gmw-pro:*dōn"]
    assert row["detach_children"] == ["modern-english:do"]
    # round-trips through the collector the applier consumes
    assert collect_collapses([ledger])["old-english:don"]["detach_parents"] == ["gmw-pro:*dōn"]


def test_curate_collapse_etymon_cli_detach_only_omits_into(tmp_path: Path) -> None:
    """A detach-only event (no --into) OMITS the `into` key — it must not write
    the `into: ''` revert sentinel and masquerade as a fold-revert."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.curate_collapse import (
        lexicon_curate_collapse_etymon,
    )

    ledger = tmp_path / "_collapses.jsonl"
    res = CliRunner().invoke(
        lexicon_curate_collapse_etymon,
        ["old-english:don", "--detach-child", "modern-english:do", "--collapse-file", str(ledger)],
    )
    assert res.exit_code == 0, res.output
    row = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()][1]
    assert "into" not in row
    assert row["detach_children"] == ["modern-english:do"]


def test_curate_collapse_etymon_cli_requires_an_action(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.curate_collapse import (
        lexicon_curate_collapse_etymon,
    )

    res = CliRunner().invoke(
        lexicon_curate_collapse_etymon,
        ["old-english:don", "--collapse-file", str(tmp_path / "_collapses.jsonl")],
    )
    assert res.exit_code != 0
    assert "at least one of" in res.output.lower()


def test_curate_collapse_etymon_cli_allows_empty_into_revert(tmp_path: Path) -> None:
    """`--into ''` is a valid revert event (must not trip the usage guard)."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.curate_collapse import (
        lexicon_curate_collapse_etymon,
    )

    ledger = tmp_path / "_collapses.jsonl"
    res = CliRunner().invoke(
        lexicon_curate_collapse_etymon,
        ["old-english:dūn", "--into", "", "--collapse-file", str(ledger)],
    )
    assert res.exit_code == 0, res.output
    row = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()][1]
    assert row["ref"] == "old-english:dūn" and row["into"] == ""


def test_collect_collapses_round_trip(tmp_path: Path) -> None:
    """A _collapses.jsonl row replays into {from_ref: payload}; last-write
    wins, and into:'' reverts."""
    p = tmp_path / "_collapses.jsonl"
    rows = [
        {
            "_type": "collapse",
            "ref": "old-english:burh",
            "into": "old-english:burg",
            "variant_class": "alternative",
            "method": "deterministic-gloss-overlap",
        },
        {
            "_type": "collapse",
            "ref": "old-english:fen",
            "into": "old-english:fenn",
            "variant_class": "alternative",
        },
        # revert the fen collapse
        {"_type": "collapse", "ref": "old-english:fen", "into": ""},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    state = collect_collapses([p])
    assert state["old-english:burh"]["into"] == "old-english:burg"
    assert state["old-english:fen"]["into"] == ""  # reverted (last-write-wins)


def test_collect_collapses_resolves_cycles(tmp_path: Path) -> None:
    """collect_collapses breaks cycles at load (the LLM merge can judge a
    pair same-meaning in both directions). One direction survives, the
    cycle-closer is demoted to a no-op; chains flatten to the survivor."""
    p = tmp_path / "_collapses.jsonl"
    rows = [
        # mutual cycle: fau<->fou
        {"_type": "collapse", "ref": "old-french:fou", "into": "old-french:fau"},
        {"_type": "collapse", "ref": "old-french:fau", "into": "old-french:fou"},
        # chain: a -> b -> c
        {"_type": "collapse", "ref": "x:a", "into": "x:b"},
        {"_type": "collapse", "ref": "x:b", "into": "x:c"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    state = collect_collapses([p])
    intos = {r: p_["into"] for r, p_ in state.items()}
    # exactly one direction of the cycle survives (sorted-ref: 'fau' first)
    assert intos["old-french:fau"] == "old-french:fou"
    assert intos["old-french:fou"] == ""  # cycle-closer demoted
    # chain flattened to the terminal survivor c
    assert intos["x:a"] == "x:c"
    assert intos["x:b"] == "x:c"


def test_resolve_collapse_cycles_passthrough_and_multicluster() -> None:
    """Direct unit pin of _resolve_collapse_cycles (wyrd-8uvi C901
    extraction): no-op rows (empty/absent ``into``) pass through with
    their payload untouched, and independent clusters resolve without
    cross-contamination. The cycle+chain branches are covered by
    test_collect_collapses_resolves_cycles via the public collector."""
    state = {
        # No-op rows: the payload object must pass through unchanged.
        "x:noop_empty": {"into": "", "method": "revert"},
        "x:noop_absent": {"method": "rejected"},
        # Cluster 1: a -> b (kept, b is its own survivor).
        "x:a": {"into": "x:b"},
        "x:b": {"into": "x:c"},
        # Cluster 2: independent chain p -> q (must NOT fold into cluster 1).
        "x:p": {"into": "x:q"},
    }
    out = _resolve_collapse_cycles(state)
    # No-op passthrough: same payload, untouched (identity preserved).
    assert out["x:noop_empty"] is state["x:noop_empty"]
    assert out["x:noop_absent"] is state["x:noop_absent"]
    # Cluster 1 flattens to terminal survivor c; cluster 2 stays separate.
    assert out["x:a"]["into"] == "x:c"
    assert out["x:b"]["into"] == "x:c"
    assert out["x:p"]["into"] == "x:q"


def test_resolve_collapse_cycles_path_compression_multi_hop() -> None:
    """A late edge attaching above an already-built chain forces the
    union-find path-compression inner loop in _uf_find to fire (the one
    non-trivial branch the helper owns). With a -> b -> c already unioned,
    `d -> a` makes find(a) walk a->b->c and rewire the intermediates to
    the root; all four refs must resolve to the terminal survivor c.
    Sorted-ref processing order (a, b, d) builds the chain before d
    arrives, so find(d->a) traverses multiple hops (wyrd-8uvi)."""
    state = {
        "x:a": {"into": "x:b"},
        "x:b": {"into": "x:c"},
        "x:d": {"into": "x:a"},
    }
    out = _resolve_collapse_cycles(state)
    assert out["x:a"]["into"] == "x:c"
    assert out["x:b"]["into"] == "x:c"
    assert out["x:d"]["into"] == "x:c"


def test_folded_form_exports_with_lemma_gloss_not_pointer(tmp_path: Path) -> None:
    """wyrd-6r09 (P4): after a collapse folds a form-of etymon into its
    lemma, the exported family shows the LEMMA's real meaning — the folded
    member's pointer gloss is dropped, while its form is kept in the
    family. This is what makes the fold visible in generation."""
    from wyrd.generators.kenning.lexicon.bundle._export import export_meanings

    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO source (id, title) VALUES ('rando-port', 'seed')")
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'Hea', 'old-english')"
    )
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (2, 'ea', 'old-english')"
    )
    conn.execute(
        "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (1, 'h-prothesized form of ea')"
    )
    conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (2, 'river')")
    conn.execute("INSERT INTO etymon_citation (etymon_id, source_id) VALUES (2, 'rando-port')")
    conn.execute("INSERT INTO etymon_citation (etymon_id, source_id) VALUES (1, 'rando-port')")
    conn.commit()
    conn.close()

    with LexiconDB(db_path) as db:
        apply_collapses(
            db,
            {"old-english:Hea": {"into": "old-english:ea", "variant_class": "alternative"}},
            apply=True,
        )
        subjects = export_meanings(db, include_rando=True, min_witnesses=1)

    assert len(subjects) == 1
    s = subjects[0]
    assert s["meaning"] == ["river"]  # the pointer gloss is gone
    assert set(s["words"][0]["old_english"]) == {"ea", "Hea"}  # form still in the family
