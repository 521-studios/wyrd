"""wyrd-rogd.10 Phase 1 — morpheme_id emission (dormant).

Covers: the owning-morpheme id rides on each word, is excluded from the
Meaning.sources language axis, and the new `morpheme` table regroups the
per-word entries by morpheme_id (the unfragmenting) instead of the connective
surface string. The runtime does NOT yet follow this table (Phase 2).
"""

from __future__ import annotations

import sqlite3

from wyrd.generators.kenning.lexicon.bundle._subject import _word_morpheme_id
from wyrd.generators.kenning.lexicon.runtime_db_export import _write_morphemes
from wyrd.generators.kenning.runtime.meaning import load_meanings


def test_morpheme_id_excluded_from_sources():
    # The int morpheme_id must NOT be treated as a language form-array (it
    # crashed primary_language()'s form loop before the guard).
    data = {
        "subjects": [
            {
                "meaning": ["enclosure"],
                "modifier_tags": [],
                "modifier_type": "descriptor",
                "words": [
                    {
                        "modern_usage": "-ton",
                        "morpheme_id": "old-english:tūn",
                        "old-english": ["tūn"],
                    }
                ],
            }
        ]
    }
    meaning_db, _ = load_meanings(data)
    m = meaning_db["-ton"][0]
    assert "morpheme_id" not in m.sources
    assert sorted(m.sources) == ["old-english"]
    assert m.primary_language() == "old-english"  # the loop that used to crash


def _morpheme_table_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE morpheme (morpheme_id TEXT PRIMARY KEY, primary_language TEXT, "
        "stratum TEXT, data BLOB NOT NULL)"
    )
    return conn


def test_write_morphemes_regroups_surface_variants_under_one_id():
    # Three connective-form words of ONE morpheme (root 42) plus a different
    # morpheme (root 99). _write_morphemes folds the three into a single row.
    subjects = [
        {
            "meaning": ["m"],
            "modifier_tags": [],
            "modifier_type": "descriptor",
            "words": [
                {"modern_usage": "-ing-", "morpheme_id": "old-english:ing", "old-english": ["ing"]},
                {"modern_usage": "-ing", "morpheme_id": "old-english:ing", "old-english": ["ing"]},
                {"modern_usage": "Ing-", "morpheme_id": "old-english:ing", "old-english": ["ing"]},
                {"modern_usage": "-ton", "morpheme_id": "old-english:tūn", "old-english": ["tūn"]},
            ],
        }
    ]
    conn = _morpheme_table_conn()
    n = _write_morphemes(conn, subjects)
    assert n == 2  # two distinct morpheme_ids, NOT four connective rows
    ids = {r[0] for r in conn.execute("SELECT morpheme_id FROM morpheme")}
    assert ids == {"old-english:ing", "old-english:tūn"}
    import json

    blob = conn.execute("SELECT data FROM morpheme WHERE morpheme_id='old-english:ing'").fetchone()[
        0
    ]
    entries = json.loads(blob)["entries"]
    assert len(entries) == 3  # all three surface variants merged under id 42


def test_write_morphemes_skips_words_without_id():
    subjects = [
        {
            "meaning": ["m"],
            "modifier_tags": [],
            "modifier_type": "descriptor",
            "words": [{"modern_usage": "-orphan"}],  # no morpheme_id
        }
    ]
    conn = _morpheme_table_conn()
    assert _write_morphemes(conn, subjects) == 0


def test_word_morpheme_id_is_deterministic_content_key():
    fams = [
        {"root_canonical_form": "tūn", "root_language": "old-english"},
        {"root_canonical_form": "ham", "root_language": "old-english"},
    ]
    # min by (canonical_form, language) → "ham" < "tūn" → the content key, order-independent.
    assert _word_morpheme_id(fams) == "old-english:ham"
    assert _word_morpheme_id(list(reversed(fams))) == "old-english:ham"
    assert _word_morpheme_id([]) is None
