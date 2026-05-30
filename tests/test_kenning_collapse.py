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
from wyrd.generators.kenning.jsonl.build import collect_collapses
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
