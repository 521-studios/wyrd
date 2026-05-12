"""Tests for the read-only lexicon browser — wyrd-7oma (Phase 1a).

In-memory schema fixture covers the tables the browser reads
(``source``, ``etymon``, ``etymon_gloss``, ``etymon_tag``,
``etymon_citation``, ``etymon_descent``, ``toponym``,
``toponym_attestation``, ``toponym_etymology``,
``toponym_etymology_element``, ``toponym_decomposition``,
``mining_run``).

The tests are split between ``fetch_*`` (DB queries return plain
data) and ``format_*`` (markdown render). Two layers keep the
test surface small: data shape is one contract, markdown is
another, neither couples to the other.
"""

from __future__ import annotations

import sqlite3

import pytest

from wyrd.generators.kenning.browse import (
    fetch_decompositions,
    fetch_etymon,
    fetch_source,
    fetch_toponym_detail,
    fetch_toponyms_matching,
    format_decompositions,
    format_etymon,
    format_source,
    format_toponym,
    format_toponym_list,
    parse_etymon_ref,
    parse_toponym_ref,
)

# ---------------------------------------------------------------------------
# Fixture: minimal schema with all browser-touching columns
# ---------------------------------------------------------------------------


def _build_fixture_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE source (
            id TEXT PRIMARY KEY,
            author TEXT, title TEXT NOT NULL, year INTEGER,
            region TEXT, language_focus TEXT, notes TEXT
        );
        CREATE TABLE etymon (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL,
            language TEXT NOT NULL,
            modifier_type TEXT, position_pref TEXT, notes TEXT,
            lemma_id INTEGER, inflection TEXT, lemma_method TEXT,
            merged_into_id INTEGER, cognate_id INTEGER, cognate_method TEXT,
            pronunciation_ipa TEXT, pronunciation_dialect TEXT,
            original_script TEXT, transliteration TEXT,
            english_shaped TEXT, stratum TEXT,
            UNIQUE (canonical_form, language)
        );
        CREATE TABLE etymon_gloss (
            etymon_id INTEGER NOT NULL, gloss TEXT NOT NULL,
            PRIMARY KEY (etymon_id, gloss)
        );
        CREATE TABLE etymon_tag (
            etymon_id INTEGER NOT NULL, tag TEXT NOT NULL,
            PRIMARY KEY (etymon_id, tag)
        );
        CREATE TABLE etymon_citation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL, source_id TEXT NOT NULL,
            page TEXT, short_quote TEXT, context_snippet TEXT
        );
        CREATE TABLE etymon_descent (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER NOT NULL, child_id INTEGER NOT NULL,
            edge_type TEXT NOT NULL, source_id TEXT NOT NULL,
            confidence TEXT, notes TEXT
        );
        CREATE TABLE mining_run (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL, provider TEXT NOT NULL,
            model TEXT NOT NULL, mode TEXT NOT NULL,
            started_at TEXT, completed_at TEXT,
            parsed_count INTEGER NOT NULL DEFAULT 0,
            accepted INTEGER NOT NULL DEFAULT 0,
            declined INTEGER NOT NULL DEFAULT 0,
            rejected INTEGER NOT NULL DEFAULT 0,
            by_failure TEXT, notes TEXT
        );
        CREATE TABLE toponym (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modern_name TEXT NOT NULL, country TEXT, region TEXT
        );
        CREATE TABLE toponym_attestation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id INTEGER NOT NULL, form TEXT NOT NULL,
            date_year INTEGER, source_doc TEXT
        );
        CREATE TABLE toponym_etymology (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id INTEGER NOT NULL, source_id TEXT NOT NULL,
            page TEXT, historical_form TEXT, confidence TEXT,
            notes TEXT, attested_year INTEGER
        );
        CREATE TABLE toponym_etymology_element (
            toponym_etymology_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
            etymon_id INTEGER NOT NULL, inflection TEXT,
            surface_in_modern TEXT,
            PRIMARY KEY (toponym_etymology_id, ordinal)
        );
        CREATE TABLE toponym_decomposition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id INTEGER NOT NULL,
            decomposition_signature TEXT NOT NULL,
            morpheme_ids TEXT NOT NULL,
            unaccounted_fragments TEXT NOT NULL,
            unaccounted_count INTEGER NOT NULL DEFAULT 0,
            morpheme_count INTEGER NOT NULL DEFAULT 0,
            is_canonical INTEGER NOT NULL DEFAULT 0,
            canonical_source TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    return conn


def _add_etymon(conn: sqlite3.Connection, language: str, canonical_form: str, **kwargs) -> int:
    cols = ["language", "canonical_form", *kwargs.keys()]
    vals = [language, canonical_form, *kwargs.values()]
    placeholders = ", ".join("?" * len(cols))
    cur = conn.execute(
        f"INSERT INTO etymon ({', '.join(cols)}) VALUES ({placeholders})", tuple(vals)
    )
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Ref parsing
# ---------------------------------------------------------------------------


def test_parse_etymon_ref_splits_on_first_colon():
    assert parse_etymon_ref("old-english:cot") == ("old-english", "cot")


def test_parse_etymon_ref_preserves_colons_in_form():
    """canonical_form may itself contain colons (rare for non-Latin
    script forms); split only on the first colon."""
    assert parse_etymon_ref("hebrew:foo:bar") == ("hebrew", "foo:bar")


def test_parse_etymon_ref_raises_without_colon():
    with pytest.raises(ValueError, match="must be 'lang:form'"):
        parse_etymon_ref("cot")


def test_parse_etymon_ref_raises_empty_language():
    with pytest.raises(ValueError, match="empty language"):
        parse_etymon_ref(":cot")


def test_parse_toponym_ref_with_region():
    assert parse_toponym_ref("Acton@Cheshire") == ("Acton", "Cheshire")


def test_parse_toponym_ref_without_region():
    assert parse_toponym_ref("Acton") == ("Acton", None)


def test_parse_toponym_ref_dash_region_means_null():
    """The '-' placeholder is our JSONL dumper's null-region marker;
    when looking up, '-' should be treated as None (the actual DB
    column value)."""
    assert parse_toponym_ref("Acton@-") == ("Acton", None)


# ---------------------------------------------------------------------------
# fetch_etymon
# ---------------------------------------------------------------------------


def test_fetch_etymon_returns_none_for_missing():
    conn = _build_fixture_db()
    assert fetch_etymon(conn, "old-english:ghost") is None


def test_fetch_etymon_minimal():
    conn = _build_fixture_db()
    _add_etymon(conn, "old-english", "cot")
    data = fetch_etymon(conn, "old-english:cot")
    assert data is not None
    assert data["ref"] == "old-english:cot"
    assert data["canonical_form"] == "cot"
    assert data["language"] == "old-english"
    assert data["glosses"] == []
    assert data["tags"] == []
    assert data["citations"] == []
    assert data["descent_in"] == []
    assert data["descent_out"] == []


def test_fetch_etymon_with_glosses_tags_citations():
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('skeat', 'Cambs')")
    eid = _add_etymon(
        conn,
        "old-english",
        "cot",
        modifier_type="noun",
        stratum="common",
        pronunciation_ipa="/kɒt/",
    )
    conn.execute("INSERT INTO etymon_gloss VALUES (?, 'cottage')", (eid,))
    conn.execute("INSERT INTO etymon_gloss VALUES (?, 'hut')", (eid,))
    conn.execute("INSERT INTO etymon_tag VALUES (?, 'architecture')", (eid,))
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page, short_quote) "
        "VALUES (?, 'skeat', '15', 'OE cot, dwelling')",
        (eid,),
    )
    data = fetch_etymon(conn, "old-english:cot")
    assert data["glosses"] == ["cottage", "hut"]
    assert data["tags"] == ["architecture"]
    assert data["modifier_type"] == "noun"
    assert data["stratum"] == "common"
    assert data["pronunciation_ipa"] == "/kɒt/"
    assert data["citations"] == [
        {"source_id": "skeat", "page": "15", "short_quote": "OE cot, dwelling"}
    ]


def test_fetch_etymon_with_descent_in_and_out():
    """Etymon with both an ancestor (descent_in) and a descendant
    (descent_out)."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('wiki', 'Wiktionary')")
    proto = _add_etymon(conn, "proto-germanic", "*tunaz")
    oe = _add_etymon(conn, "old-english", "tun")
    me = _add_etymon(conn, "middle-english", "toun")
    conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence) "
        "VALUES (?, ?, 'inheritance', 'wiki', 'high')",
        (proto, oe),
    )
    conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence) "
        "VALUES (?, ?, 'inheritance', 'wiki', 'high')",
        (oe, me),
    )
    data = fetch_etymon(conn, "old-english:tun")
    assert len(data["descent_in"]) == 1
    assert data["descent_in"][0]["parent_ref"] == "proto-germanic:*tunaz"
    assert len(data["descent_out"]) == 1
    assert data["descent_out"][0]["child_ref"] == "middle-english:toun"


def test_fetch_etymon_with_lemma_family():
    """A lemma row reports its inflected children; an inflected child
    reports its lemma parent."""
    conn = _build_fixture_db()
    lemma = _add_etymon(conn, "old-english", "cot")
    _add_etymon(
        conn,
        "old-english",
        "cotan",
        lemma_id=lemma,
        inflection="genitive",
        lemma_method="link-lemmas-v1",
    )
    _add_etymon(
        conn,
        "old-english",
        "cotes",
        lemma_id=lemma,
        inflection="genitive",
    )
    lemma_data = fetch_etymon(conn, "old-english:cot")
    assert lemma_data["lemma_ref"] is None  # it IS the lemma
    assert len(lemma_data["inflections"]) == 2
    refs = {i["ref"] for i in lemma_data["inflections"]}
    assert refs == {"old-english:cotan", "old-english:cotes"}

    child_data = fetch_etymon(conn, "old-english:cotan")
    assert child_data["lemma_ref"] == "old-english:cot"
    assert child_data["inflection"] == "genitive"
    assert child_data["lemma_method"] == "link-lemmas-v1"


def test_fetch_etymon_with_ocr_cluster():
    """A merged-into row reports its canonical target; the canonical
    reports its merged-in losers."""
    conn = _build_fixture_db()
    canon = _add_etymon(conn, "old-english", "tun")
    _add_etymon(conn, "old-english", "tuu", merged_into_id=canon)
    _add_etymon(conn, "old-english", "tnn", merged_into_id=canon)

    canon_data = fetch_etymon(conn, "old-english:tun")
    assert canon_data["merged_into_ref"] is None
    assert set(canon_data["merged_in"]) == {"old-english:tuu", "old-english:tnn"}

    loser_data = fetch_etymon(conn, "old-english:tuu")
    assert loser_data["merged_into_ref"] == "old-english:tun"


# ---------------------------------------------------------------------------
# fetch_toponym_detail + fetch_toponyms_matching
# ---------------------------------------------------------------------------


def test_fetch_toponyms_matching_returns_all_regions():
    conn = _build_fixture_db()
    conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES ('Acton', 'England', 'Cheshire')"
    )
    conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES ('Acton', 'England', 'Suffolk')"
    )
    matches = fetch_toponyms_matching(conn, "Acton", None)
    assert len(matches) == 2
    assert {m["region"] for m in matches} == {"Cheshire", "Suffolk"}


def test_fetch_toponyms_matching_filters_by_region():
    conn = _build_fixture_db()
    conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES ('Acton', 'England', 'Cheshire')"
    )
    conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES ('Acton', 'England', 'Suffolk')"
    )
    matches = fetch_toponyms_matching(conn, "Acton", "Suffolk")
    assert len(matches) == 1
    assert matches[0]["region"] == "Suffolk"


def test_fetch_toponym_detail_full():
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('mawer', 'N&D')")
    cot = _add_etymon(conn, "old-english", "cot")
    tun = _add_etymon(conn, "old-english", "tun")
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES ('Cotton', 'England', 'Norfolk')"
    )
    tid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (?, 'Cotuna', 1086, 'Domesday Book')",
        (tid,),
    )
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year) VALUES (?, 'Cottune', 1227)",
        (tid,),
    )
    cur = conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, historical_form, confidence, attested_year) "
        "VALUES (?, 'mawer', 'Cotuna', 'high', 1086)",
        (tid,),
    )
    eid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) VALUES (?, 1, ?)",
        (eid, cot),
    )
    conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id, inflection) "
        "VALUES (?, 2, ?, 'oblique')",
        (eid, tun),
    )
    data = fetch_toponym_detail(conn, tid)
    assert data["modern_name"] == "Cotton"
    assert data["region"] == "Norfolk"
    assert len(data["attestations"]) == 2
    assert data["attestations"][0] == {
        "form": "Cotuna",
        "date_year": 1086,
        "source_doc": "Domesday Book",
    }
    assert len(data["etymologies"]) == 1
    et = data["etymologies"][0]
    assert et["historical_form"] == "Cotuna"
    assert et["elements"] == [
        {
            "ordinal": 1,
            "etymon_ref": "old-english:cot",
            "inflection": None,
            "surface_in_modern": None,
        },
        {
            "ordinal": 2,
            "etymon_ref": "old-english:tun",
            "inflection": "oblique",
            "surface_in_modern": None,
        },
    ]


def test_fetch_toponym_detail_sorts_attestations_by_year():
    conn = _build_fixture_db()
    cur = conn.execute("INSERT INTO toponym (modern_name) VALUES ('Foo')")
    tid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year) VALUES (?, 'F1611', 1611)",
        (tid,),
    )
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year) VALUES (?, 'F1086', 1086)",
        (tid,),
    )
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form) VALUES (?, 'F-undated')",
        (tid,),
    )
    data = fetch_toponym_detail(conn, tid)
    # Earliest year first, undated last (COALESCE'd to 9999).
    assert [a["form"] for a in data["attestations"]] == ["F1086", "F1611", "F-undated"]


# ---------------------------------------------------------------------------
# fetch_decompositions
# ---------------------------------------------------------------------------


def test_fetch_decompositions_orders_canonical_first():
    conn = _build_fixture_db()
    cur = conn.execute("INSERT INTO toponym (modern_name) VALUES ('Cotton')")
    tid = cur.lastrowid
    # Non-canonical with unaccounted fragments
    conn.execute(
        """INSERT INTO toponym_decomposition
           (toponym_id, decomposition_signature, morpheme_ids, unaccounted_fragments,
            unaccounted_count, morpheme_count, is_canonical)
           VALUES (?, 'sig-a', '["Cot", {"unaccounted": "t"}, "on"]', '["t"]', 1, 2, 0)""",
        (tid,),
    )
    # Canonical
    conn.execute(
        """INSERT INTO toponym_decomposition
           (toponym_id, decomposition_signature, morpheme_ids, unaccounted_fragments,
            unaccounted_count, morpheme_count, is_canonical, canonical_source)
           VALUES (?, 'sig-b', '["Cot", "ton"]', '[]', 0, 2, 1, 'scholar')""",
        (tid,),
    )
    decompositions = fetch_decompositions(conn, tid)
    assert len(decompositions) == 2
    assert decompositions[0]["is_canonical"] is True
    assert decompositions[0]["canonical_source"] == "scholar"
    assert decompositions[0]["morphemes"] == ["Cot", "ton"]
    assert decompositions[1]["is_canonical"] is False
    assert decompositions[1]["unaccounted"] == ["t"]


def test_fetch_decompositions_handles_malformed_json():
    """Defensive: a corrupted morpheme_ids field shouldn't crash the
    browser (lex-data hygiene shouldn't block read-only navigation)."""
    conn = _build_fixture_db()
    cur = conn.execute("INSERT INTO toponym (modern_name) VALUES ('Foo')")
    tid = cur.lastrowid
    conn.execute(
        """INSERT INTO toponym_decomposition
           (toponym_id, decomposition_signature, morpheme_ids, unaccounted_fragments)
           VALUES (?, 'sig', 'NOT-JSON', 'ALSO-NOT')""",
        (tid,),
    )
    decompositions = fetch_decompositions(conn, tid)
    assert decompositions[0]["morphemes"] == []
    assert decompositions[0]["unaccounted"] == []


# ---------------------------------------------------------------------------
# fetch_source
# ---------------------------------------------------------------------------


def test_fetch_source_returns_none_for_missing():
    conn = _build_fixture_db()
    assert fetch_source(conn, "ghost") is None


def test_fetch_source_counts_contributions():
    conn = _build_fixture_db()
    conn.execute(
        "INSERT INTO source (id, author, title, year, region) "
        "VALUES ('skeat', 'WW Skeat', 'Place-Names of Cambs', 1901, 'Cambridgeshire')"
    )
    cot = _add_etymon(conn, "old-english", "cot")
    tun = _add_etymon(conn, "old-english", "tun")
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, 'skeat', '15')",
        (cot,),
    )
    conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
        "VALUES (?, ?, 'inheritance', 'skeat')",
        (cot, tun),
    )
    cur = conn.execute("INSERT INTO toponym (modern_name, region) VALUES ('Cotton', 'Cambs')")
    tid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id) VALUES (?, 'skeat')",
        (tid,),
    )
    conn.execute(
        "INSERT INTO mining_run (source_id, provider, model, mode) "
        "VALUES ('skeat', 'anthropic', 'claude', 'mine')"
    )
    data = fetch_source(conn, "skeat")
    assert data["citation_count"] == 1
    assert data["descent_count"] == 1
    assert data["etymology_count"] == 1
    assert data["mining_run_count"] == 1
    assert len(data["toponyms"]) == 1
    assert data["toponyms"][0] == {"modern_name": "Cotton", "region": "Cambs"}


# ---------------------------------------------------------------------------
# Markdown formatters — assert on key content, not exact whitespace
# ---------------------------------------------------------------------------


def _build_etymon_data_with_everything(conn):
    """Helper that creates a fully-loaded etymon for formatter tests."""
    conn.execute("INSERT INTO source (id, title) VALUES ('skeat', 'X')")
    eid = _add_etymon(
        conn,
        "old-english",
        "cot",
        modifier_type="noun",
        stratum="common",
        pronunciation_ipa="/kɒt/",
        notes="dwelling-place",
    )
    conn.execute("INSERT INTO etymon_gloss VALUES (?, 'cottage')", (eid,))
    conn.execute("INSERT INTO etymon_tag VALUES (?, 'architecture')", (eid,))
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page, short_quote) "
        "VALUES (?, 'skeat', '15', 'OE cot')",
        (eid,),
    )
    return eid


def test_format_etymon_includes_header_and_fields():
    conn = _build_fixture_db()
    _build_etymon_data_with_everything(conn)
    md = format_etymon(fetch_etymon(conn, "old-english:cot"))
    assert "## etymon: old-english:cot" in md
    assert "Canonical form" in md and "`cot`" in md
    assert "Stratum: common" in md
    assert "IPA: /kɒt/" in md
    assert "Glosses (1)" in md and "cottage" in md
    assert "Tags (1)" in md and "architecture" in md
    assert "Citations (1)" in md and "`skeat`" in md and "p.15" in md and '"OE cot"' in md


def test_format_etymon_omits_empty_sections():
    conn = _build_fixture_db()
    _add_etymon(conn, "old-english", "cot")
    md = format_etymon(fetch_etymon(conn, "old-english:cot"))
    # No glosses → no Glosses section header.
    assert "Glosses" not in md
    assert "Tags" not in md
    assert "Citations" not in md


def test_format_etymon_shows_lemma_family():
    conn = _build_fixture_db()
    lemma = _add_etymon(conn, "old-english", "cot")
    _add_etymon(
        conn,
        "old-english",
        "cotan",
        lemma_id=lemma,
        inflection="genitive",
    )
    md = format_etymon(fetch_etymon(conn, "old-english:cot"))
    assert "Inflected variants (1)" in md
    assert "`old-english:cotan`" in md
    assert "(genitive)" in md


def test_format_etymon_shows_descent_arrows():
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('wiki', 'W')")
    proto = _add_etymon(conn, "proto-germanic", "*tunaz")
    oe = _add_etymon(conn, "old-english", "tun")
    conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence) "
        "VALUES (?, ?, 'inheritance', 'wiki', 'high')",
        (proto, oe),
    )
    md = format_etymon(fetch_etymon(conn, "old-english:tun"))
    assert "Descent: ancestors (1)" in md
    assert "`proto-germanic:*tunaz` → this" in md
    assert "inheritance" in md and "high" in md


def test_format_toponym_includes_attestations_and_etymologies():
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('mawer', 'N&D')")
    cot = _add_etymon(conn, "old-english", "cot")
    cur = conn.execute("INSERT INTO toponym (modern_name, region) VALUES ('Cotton', 'Norfolk')")
    tid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year) VALUES (?, 'Cotuna', 1086)",
        (tid,),
    )
    cur = conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, historical_form, confidence) "
        "VALUES (?, 'mawer', 'Cotuna', 'high')",
        (tid,),
    )
    eid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) VALUES (?, 1, ?)",
        (eid, cot),
    )
    md = format_toponym(fetch_toponym_detail(conn, tid))
    assert "## toponym: Cotton@Norfolk" in md
    assert "Attestations (1)" in md
    assert "1086: Cotuna" in md
    assert "Scholar etymologies (1)" in md
    assert "`mawer`" in md and "high" in md
    assert "`old-english:cot`" in md


def test_format_toponym_truncates_long_notes():
    conn = _build_fixture_db()
    long_notes = "x" * 500
    cur = conn.execute("INSERT INTO toponym (modern_name) VALUES ('Foo')")
    tid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) VALUES (?, 'src', ?)",
        (tid, long_notes),
    )
    md = format_toponym(fetch_toponym_detail(conn, tid))
    # 280-char threshold + "..." appended.
    assert "..." in md
    assert "x" * 500 not in md


def test_format_toponym_list():
    md = format_toponym_list(
        [
            {"modern_name": "Acton", "country": "England", "region": "Cheshire"},
            {"modern_name": "Acton", "country": "England", "region": "Suffolk"},
        ]
    )
    assert "2 matching toponyms" in md
    assert "`Acton@Cheshire`" in md
    assert "`Acton@Suffolk`" in md


def test_format_decompositions_with_canonical_and_alternatives():
    """Real-data shape: list of [kind, value] tagged tuples."""
    decompositions = [
        {
            "is_canonical": True,
            "canonical_source": "scholar",
            "morphemes": [["morpheme", "Cot-"], ["morpheme", "-ton"]],
            "unaccounted": [],
            "unaccounted_count": 0,
            "morpheme_count": 2,
            "signature": "x",
            "decomposition_id": 1,
        },
        {
            "is_canonical": False,
            "canonical_source": None,
            "morphemes": [["morpheme", "Cot-"], ["unaccounted", "t"], ["morpheme", "on"]],
            "unaccounted": ["t"],
            "unaccounted_count": 1,
            "morpheme_count": 2,
            "signature": "y",
            "decomposition_id": 2,
        },
    ]
    md = format_decompositions("Cotton", "Norfolk", decompositions)
    assert "## Decompositions for Cotton@Norfolk" in md
    assert "Canonical pick (1)" in md
    assert "Alternatives (1)" in md
    assert "★ Cot- | -ton" in md
    assert "[scholar]" in md
    assert "[t]" in md  # unaccounted fragment rendered with brackets


def test_format_decompositions_backward_compat_with_dict_shape():
    """Legacy shape: bare strings + ``{"unaccounted": ...}`` dicts.
    Doc'd in lexicon.sql but the actual mining writer emits the tagged-
    tuple shape. Both should render."""
    decompositions = [
        {
            "is_canonical": True,
            "canonical_source": "scholar",
            "morphemes": ["Cot", {"unaccounted": "t"}, "on"],
            "unaccounted": ["t"],
            "unaccounted_count": 1,
            "morpheme_count": 2,
            "signature": "z",
            "decomposition_id": 3,
        },
    ]
    md = format_decompositions("Foo", None, decompositions)
    assert "Cot | [t] | on" in md


def test_format_decompositions_empty_state():
    md = format_decompositions("Foo", None, [])
    assert "(no decompositions" in md


def test_format_source_header_and_counts():
    conn = _build_fixture_db()
    conn.execute(
        "INSERT INTO source (id, author, title, year) "
        "VALUES ('skeat', 'WW Skeat', 'Place-Names of Cambs', 1901)"
    )
    md = format_source(fetch_source(conn, "skeat"))
    assert "## Source: skeat" in md
    assert "Title: Place-Names of Cambs" in md
    assert "Author: WW Skeat" in md
    assert "Year: 1901" in md
    assert "Contribution counts" in md
    assert "Etymon citations: 0" in md


def test_format_source_with_list_toponyms_flag():
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('skeat', 'X')")
    cur = conn.execute("INSERT INTO toponym (modern_name, region) VALUES ('Cotton', 'Cambs')")
    tid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id) VALUES (?, 'skeat')",
        (tid,),
    )
    data = fetch_source(conn, "skeat")
    assert "Toponyms covered" not in format_source(data, list_toponyms=False)
    md = format_source(data, list_toponyms=True)
    assert "Toponyms covered (1)" in md
    assert "`Cotton@Cambs`" in md
