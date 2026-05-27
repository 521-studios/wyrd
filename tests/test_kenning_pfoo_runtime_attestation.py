"""End-to-end coverage for the wyrd-pfoo per-Meaning attestation
pipeline at the runtime side.

The matcher / splitter tests live in
``test_kenning_trie_matcher.py`` + ``test_kenning_pfoo_culture_tiebreaker.py``.
This file covers the downstream half: the L4 write/read round-trip
(``_insert_attested_languages`` / ``_read_proportions_attested_languages``),
the dev-subset narrowing of attested_languages, and the per-Meaning
filter branches in ``vector_name_select.build_non_position_eligible``.
"""

from __future__ import annotations

import sqlite3

import pytest

from wyrd.generators.kenning.lexicon.runtime_db_export import (
    _insert_attested_languages,
    _write_proportions,
    select_dev_subset,
)
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.runtime_db_adapter import (
    _read_proportions_attested_languages,
    proportions_dict_for_culture,
)
from wyrd.generators.kenning.runtime.vector_name_select import (
    build_non_position_eligible,
)
from wyrd.generators.kenning.vectors.schemas import EligibilityGate


# ---- L4 write/read round-trip ---------------------------------------------


@pytest.fixture
def attestation_schema_conn():
    """In-memory SQLite with just the proportions schema needed for
    these tests. Avoids the full runtime.sql to keep test setup focused
    on the wyrd-pfoo surface."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE proportions_usage (
            culture TEXT NOT NULL, usage_key TEXT NOT NULL,
            weight INTEGER NOT NULL, cumulative INTEGER NOT NULL,
            PRIMARY KEY (culture, cumulative)
        );
        CREATE TABLE proportions_single_usage (
            culture TEXT NOT NULL, usage_key TEXT NOT NULL,
            weight INTEGER NOT NULL, cumulative INTEGER NOT NULL,
            PRIMARY KEY (culture, cumulative)
        );
        CREATE TABLE proportions_structure (
            culture TEXT NOT NULL, template BLOB NOT NULL,
            weight INTEGER NOT NULL, cumulative INTEGER NOT NULL,
            PRIMARY KEY (culture, cumulative)
        );
        CREATE TABLE proportions_tag_marginal (
            culture TEXT NOT NULL, tag TEXT NOT NULL,
            weight INTEGER NOT NULL,
            PRIMARY KEY (culture, tag)
        );
        CREATE TABLE proportions_tag_cooccurrence (
            culture TEXT NOT NULL, tag1 TEXT NOT NULL, tag2 TEXT NOT NULL,
            weight INTEGER NOT NULL,
            PRIMARY KEY (culture, tag1, tag2)
        );
        CREATE TABLE proportions_attested_language (
            culture TEXT NOT NULL,
            usage_key TEXT NOT NULL,
            primary_language TEXT NOT NULL,
            PRIMARY KEY (culture, usage_key, primary_language)
        );
        """
    )
    yield conn
    conn.close()


def test_insert_attested_languages_writes_each_pair(attestation_schema_conn):
    """wyrd-pfoo: ``_insert_attested_languages`` writes one row per
    (culture, usage_key, primary_language) triple. Pin the actual
    table contents — without a direct test the row shape can drift
    silently."""
    conn = attestation_schema_conn
    n = _insert_attested_languages(
        conn,
        "welsh",
        {"pen-": ["celtic_mix"], "-bryn": ["celtic_mix", "old_english"]},
    )
    assert n == 3
    rows = conn.execute(
        "SELECT culture, usage_key, primary_language "
        "FROM proportions_attested_language ORDER BY usage_key, primary_language"
    ).fetchall()
    assert rows == [
        ("welsh", "-bryn", "celtic_mix"),
        ("welsh", "-bryn", "old_english"),
        ("welsh", "pen-", "celtic_mix"),
    ]


def test_insert_attested_languages_empty_input_writes_nothing(attestation_schema_conn):
    """Empty attestation map → zero rows inserted. The early-return
    branch keeps re-emit cheap when a culture has no decompositions."""
    conn = attestation_schema_conn
    n = _insert_attested_languages(conn, "english", {})
    assert n == 0
    rows = conn.execute("SELECT COUNT(*) FROM proportions_attested_language").fetchone()
    assert rows == (0,)


def test_read_proportions_attested_languages_round_trips(attestation_schema_conn):
    """Round-trip: write attestation rows for two cultures, read each
    back via the adapter, confirm the per-culture grouping is correct
    (no cross-culture leakage). Values are sorted alphabetically by
    the adapter's ORDER BY clause for deterministic iteration."""
    conn = attestation_schema_conn
    _insert_attested_languages(conn, "welsh", {"pen-": ["celtic_mix"]})
    _insert_attested_languages(conn, "english", {"-ham": ["old_english"]})

    welsh = _read_proportions_attested_languages(conn, "welsh")
    english = _read_proportions_attested_languages(conn, "english")
    assert welsh == {"pen-": ["celtic_mix"]}
    assert english == {"-ham": ["old_english"]}


def test_read_proportions_attested_languages_legacy_bundle_falls_back():
    """wyrd-pfoo back-compat: a runtime DB built before the
    ``proportions_attested_language`` table existed must still load.
    The adapter catches ``sqlite3.OperationalError`` and returns an
    empty dict so ``load_proportions`` falls back to per-usage
    filtering only (the pre-wyrd-pfoo behaviour)."""
    conn = sqlite3.connect(":memory:")
    try:
        # Schema with no attestation table at all → SELECT must error.
        result = _read_proportions_attested_languages(conn, "welsh")
        assert result == {}
    finally:
        conn.close()


def test_proportions_dict_for_culture_includes_attested_languages_when_populated(
    attestation_schema_conn,
):
    """The adapter's full ``proportions_dict_for_culture`` output
    surfaces the new key even when the other tables are sparse.
    Pinned because the runtime ``load_proportions`` reads this exact
    dict shape."""
    conn = attestation_schema_conn
    _insert_attested_languages(conn, "welsh", {"pen-": ["celtic_mix"]})
    out = proportions_dict_for_culture(conn, "welsh")
    assert out["attested_languages"] == {"pen-": ["celtic_mix"]}


# ---- select_dev_subset narrowing ------------------------------------------


def _props(usages=None, single=None, attested=None):
    return {
        "usages": usages or {},
        "single_usages": single or {},
        "structures": [],
        "tag_marginal": {},
        "tag_cooccurrence": {},
        "attested_languages": attested or {},
    }


def test_select_dev_subset_narrows_attested_languages_to_kept_usages():
    """wyrd-pfoo: ``select_dev_subset`` drops attestation rows whose
    usage_key fell out of the top-N proportions. Without this the dev
    seed would carry orphan rows that the runtime filter can never
    reach (the per-usage filter rejects them first)."""
    proportions = {
        "welsh": _props(
            usages={"-keep": 100, "-also-keep": 50, "-drop": 1},
            attested={
                "-keep": ["celtic_mix"],
                "-also-keep": ["celtic_mix", "old_english"],
                "-drop": ["celtic_mix"],
            },
        ),
    }
    _, _, _, trimmed = select_dev_subset(
        subjects=[],
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_by_culture=proportions,
        top_n_per_culture=2,
    )
    welsh = trimmed["welsh"]
    assert welsh["attested_languages"] == {
        "-also-keep": ["celtic_mix", "old_english"],
        "-keep": ["celtic_mix"],
    }


def test_select_dev_subset_uses_per_culture_keep_set_no_cross_culture_leak():
    """The narrowing must use THIS culture's kept usages, not the
    cumulative cross-culture set. A usage attested only in culture A
    must NOT survive in culture B's attestation just because A kept
    it. Pin the per-culture isolation."""
    proportions = {
        "english": _props(
            usages={"-ton": 100},
            attested={"-ton": ["old_english"]},
        ),
        "welsh": _props(
            usages={"pen-": 100},
            attested={
                "pen-": ["celtic_mix"],
                # ``-ton`` decomposed against Welsh place names too,
                # but isn't in Welsh's top-N this run. Without
                # per-culture keep_set, the trim would leak it.
                "-ton": ["old_english"],
            },
        ),
    }
    _, _, _, trimmed = select_dev_subset(
        subjects=[],
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_by_culture=proportions,
        top_n_per_culture=1,
    )
    assert trimmed["english"]["attested_languages"] == {"-ton": ["old_english"]}
    assert trimmed["welsh"]["attested_languages"] == {"pen-": ["celtic_mix"]}


# ---- vector_name_select.build_non_position_eligible per-Meaning filter ----


def _bare_gate(culture: str = "welsh") -> EligibilityGate:
    """An EligibilityGate with no era / stratum filters — isolates the
    per-Meaning filter from the other gate predicates so each branch
    can be exercised independently."""
    return EligibilityGate(culture=culture)


def _meaning(usage: str, language: str, tags=None) -> Meaning:
    return Meaning(usage, tags or [], [], {language: [usage.replace("-", "")]})


def test_build_eligible_none_per_meaning_means_no_filter():
    """``culture_attested_meanings=None`` (legacy bundle path) means
    no per-Meaning narrowing — every Meaning whose usage passes the
    per-usage filter is admitted, regardless of primary language."""
    oe = _meaning("-ton", "old_english", ["topography"])
    celtic = _meaning("-ton", "celtic_mix", ["topography"])
    meaning_db = {"-ton": [oe, celtic]}
    pool = build_non_position_eligible(
        meaning_db,
        gate=_bare_gate(),
        exclude_tags=frozenset(),
        pack_meaning_dbs=None,
        packs=(),
        culture_attested_usages=frozenset({"-ton"}),
        culture_attested_meanings=None,
    )
    assert set(pool) == {oe, celtic}


def test_build_eligible_per_meaning_narrows_to_admitted_languages():
    """The new filter: when ``culture_attested_meanings[usage]`` is set,
    only Meanings whose ``primary_language()`` is in that set are
    admitted. Wrong-sense Celtic Meanings get dropped where the
    per-usage filter would have admitted them."""
    oe = _meaning("-ton", "old_english", ["topography"])
    celtic = _meaning("-ton", "celtic_mix", ["topography"])
    meaning_db = {"-ton": [oe, celtic]}
    pool = build_non_position_eligible(
        meaning_db,
        gate=_bare_gate(),
        exclude_tags=frozenset(),
        pack_meaning_dbs=None,
        packs=(),
        culture_attested_usages=frozenset({"-ton"}),
        # Only OE Meanings were the attested sense for this usage.
        culture_attested_meanings={"-ton": frozenset({"old_english"})},
    )
    assert pool == [oe]


def test_build_eligible_missing_usage_key_in_per_meaning_defensive_admit():
    """When ``culture_attested_meanings`` is set but the specific
    usage_key isn't in it (data drift between per-usage and per-Meaning
    emit), the per-Meaning filter is skipped defensively — all Meanings
    under that usage are admitted. Pin so a refactor that empties the
    pool on data drift gets caught."""
    oe = _meaning("-ton", "old_english", ["topography"])
    celtic = _meaning("-ton", "celtic_mix", ["topography"])
    meaning_db = {"-ton": [oe, celtic]}
    pool = build_non_position_eligible(
        meaning_db,
        gate=_bare_gate(),
        exclude_tags=frozenset(),
        pack_meaning_dbs=None,
        packs=(),
        culture_attested_usages=frozenset({"-ton"}),
        # Some OTHER usage attested; -ton missing → defensive admit.
        culture_attested_meanings={"pen-": frozenset({"celtic_mix"})},
    )
    assert set(pool) == {oe, celtic}


def test_build_eligible_empty_admitted_langs_drops_all_meanings():
    """The semantic split between None (skip filter) and empty
    frozenset (filter active, no languages admitted). Pin so a future
    refactor that flips the check from ``is not None`` to truthiness
    gets caught — that would silently make this case admit all
    Meanings instead of dropping them."""
    oe = _meaning("-ton", "old_english", ["topography"])
    celtic = _meaning("-ton", "celtic_mix", ["topography"])
    meaning_db = {"-ton": [oe, celtic]}
    pool = build_non_position_eligible(
        meaning_db,
        gate=_bare_gate(),
        exclude_tags=frozenset(),
        pack_meaning_dbs=None,
        packs=(),
        culture_attested_usages=frozenset({"-ton"}),
        # Empty frozenset → filter active, nothing admits.
        culture_attested_meanings={"-ton": frozenset()},
    )
    assert pool == []


def test_build_eligible_per_meaning_skips_meanings_with_no_primary_language():
    """A Meaning whose ``sources`` dict is empty has no primary
    language, so it can't satisfy any per-Meaning admission set.
    Filter drops it cleanly."""
    bare = Meaning("-orphan", ["topography"], [], {})
    meaning_db = {"-orphan": [bare]}
    pool = build_non_position_eligible(
        meaning_db,
        gate=_bare_gate(),
        exclude_tags=frozenset(),
        pack_meaning_dbs=None,
        packs=(),
        culture_attested_usages=frozenset({"-orphan"}),
        culture_attested_meanings={"-orphan": frozenset({"celtic_mix"})},
    )
    assert pool == []


# ---- _write_proportions integration ---------------------------------------


def test_write_proportions_inserts_attested_language_rows(attestation_schema_conn):
    """End-to-end: ``_write_proportions`` calls
    ``_insert_attested_languages`` and the rows land in the table.
    Pin so a regression that drops the call (or skips the field) is
    caught by the counts dict."""
    conn = attestation_schema_conn
    counts = _write_proportions(
        conn,
        {
            "welsh": _props(
                usages={"pen-": 5},
                attested={"pen-": ["celtic_mix"]},
            ),
        },
    )
    assert counts["proportions_attested_language"] == 1
    rows = conn.execute(
        "SELECT culture, usage_key, primary_language FROM proportions_attested_language"
    ).fetchall()
    assert rows == [("welsh", "pen-", "celtic_mix")]
