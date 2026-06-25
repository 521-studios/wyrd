"""Tests for wyrd-08qv: multi-extractor consensus canonicalization for
toponym etymologies.

Builds small in-memory lexicon DBs and exercises the clustering +
promotion logic against synthetic fixtures. Uses ``init_schema`` to
get a fresh DB with the canonical columns/index/view already in
place, then populates minimal etymon + toponym + toponym_etymology +
toponym_etymology_element rows by direct INSERT (bypassing the
mining pipeline).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema
from wyrd.generators.kenning.toponym_etymology_canonical import (
    CanonicalDecision,
    _build_cluster_key,
    _extractor_from_notes,
    _normalize_form,
    apply_canonical_decisions,
    compute_canonical_decisions,
)

# ---------- fixtures -----------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Fresh lexicon DB with schema applied. Yields the LexiconDB."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    # Seed a minimal source so toponym_etymology rows can satisfy FK.
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test Source')")
    db.commit()
    yield db


def _add_toponym(db: LexiconDB, toponym_id: int, modern_name: str) -> None:
    db.conn.execute(
        "INSERT INTO toponym (id, modern_name) VALUES (?, ?)",
        (toponym_id, modern_name),
    )


def _add_etymon(db: LexiconDB, etymon_id: int, canonical_form: str, language: str) -> None:
    db.conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (?, ?, ?)",
        (etymon_id, canonical_form, language),
    )


def _add_etymology(
    db: LexiconDB,
    *,
    toponym_id: int,
    extractor: str,
    elements: list[tuple[int, str]],  # [(etymon_id, language), …]
    confidence: str = "high",
    source: str = "test_src",
) -> int:
    """Insert a toponym_etymology row + its element rows. Returns the
    row id. The ``extractor`` is wrapped into the standard notes-prefix
    shape so the cluster-builder can detect it."""
    notes = f"extracted_by:{extractor}; test fixture"
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence, notes) "
        "VALUES (?, ?, ?, ?)",
        (toponym_id, source, confidence, notes),
    )
    ety_id = cur.lastrowid
    for ordinal, (etymon_id, _language) in enumerate(elements):
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (ety_id, ordinal, etymon_id),
        )
    db.commit()
    return ety_id


# ---------- _normalize_form ----------------------------------------------


def test_normalize_form_lowercases():
    assert _normalize_form("Tun") == "tun"
    assert _normalize_form("TŪN") == "tun"


def test_normalize_form_strips_macrons_and_diacritics():
    """Two extractors emit 'tūn' and 'tun' — the macron is a
    transliteration convention, not semantic difference. They MUST
    normalize equal for clustering to count them as agreement."""
    assert _normalize_form("tūn") == "tun"
    assert _normalize_form("dēor") == _normalize_form("deor")
    assert _normalize_form("hām") == "ham"


def test_normalize_form_strips_mawer_parenthetical():
    """Mawer's '(a)' notation marks an optional inflectional vowel —
    Bedeling(a) means 'Bedeling' or 'Bedelinga'. Both should normalize
    to the same cluster key since the element root is identical."""
    assert _normalize_form("Bedeling(a)") == "bedeling"
    assert _normalize_form("bedeling(a)") == "bedeling"
    assert _normalize_form("dingt[on]") == "dingt"


def test_normalize_form_strips_whitespace_hyphens_asterisks():
    assert _normalize_form("  tun  ") == "tun"
    assert _normalize_form("bedeling-tun") == "bedelingtun"
    # IE-reconstruction asterisks shouldn't affect equality.
    assert _normalize_form("*déiwos") == _normalize_form("deiwos")


def test_normalize_form_strips_unicode_dashes_too():
    """ANY dash is non-semantic for the cluster key, not just ASCII hyphen-minus.
    A Unicode dash variant (en-dash, U+2010 hyphen, em-dash) — which scholarly
    source text uses — must drop too, or the same morpheme forks into separate
    clusters: ``al–Quadim`` (en-dash) MUST normalize equal to ``al-Quadim``."""
    assert _normalize_form("al–Quadim") == _normalize_form("al-Quadim") == "alquadim"
    assert _normalize_form("bedeling‐tun") == "bedelingtun"  # U+2010 HYPHEN
    assert _normalize_form("tūn—stem") == "tunstem"  # em-dash
    assert _normalize_form("tūn–") == "tun"  # trailing en-dash
    # Underscore is still KEPT (real morpheme boundary, not punctuation).
    assert _normalize_form("tun_stem") == "tun_stem"


def test_normalize_form_real_disagreement_stays_distinct():
    """Two cases the normalizer must NOT collapse:

    * 'tun' vs 'thun' — Mawer's scholarly notes call out the t-/th-
      alternation as phonologically meaningful (Anglo-Norman /θ/ → /t/
      assimilation). Two extractors emitting these are disagreeing,
      not agreeing.
    * 'hr6sa' vs 'rosa' — raw OCR noise (digits inside what should be
      letters) is a real production signal. The normalizer should NOT
      strip digits, because doing so would silently merge OCR-broken
      forms with their clean counterparts and falsely produce
      consensus. A future maintainer "fixing" the normalizer to strip
      digits would silently break this pin.
    """
    assert _normalize_form("tun") != _normalize_form("thun")
    assert _normalize_form("hr6sa") != _normalize_form("rosa")


def test_normalize_form_handles_empty():
    assert _normalize_form("") == ""
    assert _normalize_form("   ") == ""


# ---------- _extractor_from_notes ----------------------------------------


def test_extractor_from_notes_parses_llm_provider():
    """Original mining notes-prefix shape (qwen via OllamaClient)."""
    notes = "extracted_by:llm:qwen3.5:9b; Scholar identifies Bedeling…"
    assert _extractor_from_notes(notes) == "llm:qwen3.5:9b"


def test_extractor_from_notes_parses_anthropic():
    """Anthropic notes carry the same `extracted_by:provider:model;`
    shape; inner colons (model id) don't confuse the regex."""
    notes = "extracted_by:anthropic:claude-haiku-4-5-20251001; Scholar identifies…"
    assert _extractor_from_notes(notes) == "anthropic:claude-haiku-4-5-20251001"


def test_extractor_from_notes_parses_gemini_pipe_delimited():
    """Gemini path uses ' | ' as the delimiter rather than '; '. Both
    forms must parse — they're operationally equivalent provenance
    tags from different ingest passes."""
    notes = "extracted_by:gemini:gemini-2.5-flash | Bedlington. c. 1050 H.S.C…"
    assert _extractor_from_notes(notes) == "gemini:gemini-2.5-flash"


def test_extractor_from_notes_returns_none_for_legacy_or_curated():
    """Rows without the extracted_by prefix are valid (hand-curated
    or legacy ingest); they should NOT crash the parser."""
    assert _extractor_from_notes(None) is None
    assert _extractor_from_notes("") is None
    assert _extractor_from_notes("scholar attestation without prefix") is None


def test_extractor_from_notes_returns_none_for_empty_extractor_value():
    """Malformed notes like ``"extracted_by:   ;..."`` strip to an
    empty extractor id. Treat as anonymous (return None) rather than
    returning ``""`` — otherwise multiple malformed rows would
    collapse to a single ``""`` witness in the cluster's witness set
    and falsely produce consensus. R3 silent-failure-hunter LOW."""
    assert _extractor_from_notes("extracted_by:   ; something") is None
    assert _extractor_from_notes("extracted_by: | something") is None


# ---------- compute_canonical_decisions (clustering + promotion) --------


def test_two_witnesses_agree_promote_canonical(db):
    """The Bedlington pattern: qwen + gemini agree on element-tuple,
    haiku disagrees with a richer hypothesis. The agreeing 2-witness
    cluster wins; haiku's single-witness cluster does not."""
    _add_toponym(db, 1, "Bedlington")
    _add_etymon(db, 1, "bedeling", "old-english")
    _add_etymon(db, 2, "tun", "old-english")
    _add_etymon(db, 3, "tūn", "old-english")  # macron variant of #2
    _add_etymon(db, 4, "bedel", "old-english")
    qwen_ety = _add_etymology(
        db,
        toponym_id=1,
        extractor="llm:qwen3.5:9b",
        elements=[(1, "old-english"), (3, "old-english")],
    )
    haiku_ety = _add_etymology(
        db,
        toponym_id=1,
        extractor="anthropic:claude-haiku-4-5",
        elements=[(1, "old-english"), (2, "old-english"), (4, "old-english")],
    )
    gemini_ety = _add_etymology(
        db,
        toponym_id=1,
        extractor="gemini:gemini-2.5-flash",
        elements=[(1, "old-english"), (2, "old-english")],
    )
    decisions = compute_canonical_decisions(db)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.toponym_id == 1
    # Lowest-id row in the winning cluster gets promoted — that's qwen's row.
    assert d.promoted_etymology_id == qwen_ety
    assert d.consensus_size == 2
    assert d.runner_up_witness_count == 1
    assert d.total_clusters == 2
    # Cluster key reflects the normalized winning tuple — JSON-encoded
    # for unambiguous round-trip (R1 type-design + silent-failure).
    assert d.cluster_key == '[["bedeling","old-english"],["tun","old-english"]]'
    # haiku's hypothesis is not the canonical one (mention it just
    # to make the test's intent visible).
    assert d.promoted_etymology_id != haiku_ety
    assert d.promoted_etymology_id != gemini_ety  # gemini's id is higher than qwen's


def test_single_witness_does_not_promote(db):
    """A toponym with only ONE extractor's hypothesis — no consensus
    possible, regardless of confidence. canonical stays unset."""
    _add_toponym(db, 1, "Lonely")
    _add_etymon(db, 1, "lonely", "old-english")
    _add_etymology(db, toponym_id=1, extractor="llm:qwen", elements=[(1, "old-english")])
    decisions = compute_canonical_decisions(db)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.promoted_etymology_id is None
    assert d.consensus_size == 0
    assert d.runner_up_witness_count == 1
    assert d.total_clusters == 1


def test_same_extractor_run_twice_is_one_witness(db):
    """If qwen happens to mine the same source twice (e.g., re-ran
    after a transient failure), the two rows are NOT two witnesses.
    Consensus requires distinct extractors."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    _add_etymology(db, toponym_id=1, extractor="llm:qwen3.5:9b", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="llm:qwen3.5:9b", elements=[(1, "old-english")])
    decisions = compute_canonical_decisions(db)
    d = decisions[0]
    assert d.promoted_etymology_id is None  # one extractor, one witness, no consensus
    assert d.consensus_size == 0


def test_three_witnesses_all_agree(db):
    """Strongest signal: all three extractors converge. Pick the lowest
    id deterministically; consensus_size reflects all three."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    qwen = _add_etymology(db, toponym_id=1, extractor="llm:qwen", elements=[(1, "old-english")])
    _add_etymology(
        db, toponym_id=1, extractor="anthropic:claude-haiku", elements=[(1, "old-english")]
    )
    _add_etymology(db, toponym_id=1, extractor="gemini:flash", elements=[(1, "old-english")])
    decisions = compute_canonical_decisions(db)
    d = decisions[0]
    assert d.promoted_etymology_id == qwen
    assert d.consensus_size == 3
    assert d.total_clusters == 1


def test_two_clusters_tied_at_two_witnesses_picks_lowest_id(db):
    """Edge case: 4 extractors split 2-vs-2 on different element-tuples.
    Tie broken deterministically by min-row-id in the winning cluster.
    Without this, re-runs over the same data could pick different
    canonicals on different days — defeating the 'settled answer'
    semantics. ALSO verify the LOSING cluster's two rows received
    cluster_key + consensus_size stamps (R1 pr-test-analyzer
    MEDIUM)."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "a", "old-english")
    _add_etymon(db, 2, "b", "old-english")
    a_qwen = _add_etymology(db, toponym_id=1, extractor="llm:qwen", elements=[(1, "old-english")])
    a_claude = _add_etymology(
        db, toponym_id=1, extractor="anthropic:claude", elements=[(1, "old-english")]
    )
    # The 'b' cluster has higher row ids — should NOT win.
    b_gemini = _add_etymology(
        db, toponym_id=1, extractor="gemini:flash", elements=[(2, "old-english")]
    )
    b_openai = _add_etymology(
        db, toponym_id=1, extractor="llm:llama3", elements=[(2, "old-english")]
    )
    decisions = compute_canonical_decisions(db)
    d = decisions[0]
    assert d.consensus_size == 2
    assert d.runner_up_witness_count == 2  # genuine tie
    assert d.promoted_etymology_id == a_qwen
    # Apply and check the losing cluster rows got stamped too.
    apply_canonical_decisions(db, decisions)
    loser_rows = list(
        db.conn.execute(
            "SELECT id, is_canonical, consensus_size, cluster_key "
            "FROM toponym_etymology WHERE id IN (?, ?) ORDER BY id",
            (b_gemini, b_openai),
        )
    )
    for r in loser_rows:
        assert r["is_canonical"] == 0
        assert r["consensus_size"] == 2  # the losing cluster also has 2 witnesses
        assert r["cluster_key"] == '[["b","old-english"]]'
    # And the winning cluster's non-canonical row (a_claude) — same
    # cluster_key as a_qwen, is_canonical=0.
    claude_row = db.conn.execute(
        "SELECT is_canonical, consensus_size, cluster_key FROM toponym_etymology WHERE id = ?",
        (a_claude,),
    ).fetchone()
    assert claude_row["is_canonical"] == 0
    assert claude_row["consensus_size"] == 2
    assert claude_row["cluster_key"] == '[["a","old-english"]]'


def test_three_clusters_tied_at_two_witnesses_each(db):
    """6 extractors split 2-vs-2-vs-2 on three different element
    tuples. The lowest-id-row cluster wins (deterministic). R1
    test-coverage MEDIUM: 2-way tie was covered, 3-way wasn't."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "a", "old-english")
    _add_etymon(db, 2, "b", "old-english")
    _add_etymon(db, 3, "c", "old-english")
    # First cluster ('a') — lowest ids 1, 2 → MIN is 1.
    a_qwen = _add_etymology(db, toponym_id=1, extractor="llm:qwen", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="anthropic:claude", elements=[(1, "old-english")])
    # Second cluster ('b') — ids 3, 4.
    _add_etymology(db, toponym_id=1, extractor="gemini:flash", elements=[(2, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="llm:llama3", elements=[(2, "old-english")])
    # Third cluster ('c') — ids 5, 6.
    _add_etymology(db, toponym_id=1, extractor="anthropic:sonnet", elements=[(3, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="gemini:pro", elements=[(3, "old-english")])
    decisions = compute_canonical_decisions(db)
    d = decisions[0]
    assert d.consensus_size == 2
    assert d.runner_up_witness_count == 2  # 3-way tie at 2
    assert d.total_clusters == 3
    assert d.promoted_etymology_id == a_qwen  # lowest id across all eligible clusters


def test_no_elements_anywhere_is_noop(db):
    """Some etymology rows might exist without any element rows (legacy
    or partial ingest). The canonicalizer must handle the empty-elements
    case without crashing AND without promoting anything (you can't
    cluster empty tuples meaningfully)."""
    _add_toponym(db, 1, "X")
    db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) "
        "VALUES (1, 'test_src', 'extracted_by:llm:qwen; no elements')"
    )
    db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) "
        "VALUES (1, 'test_src', 'extracted_by:anthropic:claude; no elements')"
    )
    db.commit()
    decisions = compute_canonical_decisions(db)
    # Both rows have empty element tuple — they DO cluster together
    # (key = empty string), and they have 2 distinct witnesses, so the
    # cluster IS eligible. But the apply-side detects all-empty
    # elements and resets is_canonical to 0 (you don't promote a
    # decomposition with no elements). Pin both halves of that contract.
    summary = apply_canonical_decisions(db, decisions)
    assert summary.toponyms_no_elements == 1
    assert summary.toponyms_promoted == 0
    canonical_count = db.conn.execute(
        "SELECT COUNT(*) FROM toponym_etymology WHERE toponym_id = 1 AND is_canonical = 1"
    ).fetchone()[0]
    assert canonical_count == 0


def test_normalization_collapses_macron_variants(db):
    """Two extractors emit 'tūn' and 'tun'. They MUST count as
    agreement (per _normalize_form's documented contract). Pinning
    this end-to-end through compute_canonical_decisions AND verifying
    both rows get the SAME cluster_key after apply — without this,
    a bug that bypasses _normalize_form in cluster_key construction
    could still appear to 'cluster together' via some other path."""
    _add_toponym(db, 1, "Bedlington")
    _add_etymon(db, 1, "bedeling", "old-english")
    _add_etymon(db, 2, "tūn", "old-english")  # macron
    _add_etymon(db, 3, "tun", "old-english")  # no macron
    qwen = _add_etymology(
        db, toponym_id=1, extractor="llm:qwen", elements=[(1, "old-english"), (2, "old-english")]
    )
    gemini = _add_etymology(
        db,
        toponym_id=1,
        extractor="gemini:flash",
        elements=[(1, "old-english"), (3, "old-english")],
    )
    decisions = compute_canonical_decisions(db)
    d = decisions[0]
    assert d.consensus_size == 2
    assert d.promoted_etymology_id == qwen
    # Apply and verify BOTH rows received the same cluster_key (the
    # macron and no-macron rows must literally share a key in the DB,
    # not just happen to cluster).
    apply_canonical_decisions(db, decisions)
    keys = list(
        db.conn.execute(
            "SELECT cluster_key FROM toponym_etymology WHERE id IN (?, ?) ORDER BY id",
            (qwen, gemini),
        )
    )
    assert keys[0]["cluster_key"] == keys[1]["cluster_key"], (
        "macron + no-macron rows must share cluster_key"
    )


def test_th_alternation_does_NOT_collapse(db):
    """'tun' and 'thun' are real phonological variants per Mawer's
    notes — the t-/th- alternation marks Anglo-Norman assimilation.
    Two extractors emitting these MUST count as disagreement, not
    consensus. Without this, the canonicalizer would merge genuine
    spelling-variant hypotheses into false consensus."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "tun", "old-english")
    _add_etymon(db, 2, "thun", "old-english")
    _add_etymology(db, toponym_id=1, extractor="llm:qwen", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="gemini:flash", elements=[(2, "old-english")])
    decisions = compute_canonical_decisions(db)
    d = decisions[0]
    assert d.promoted_etymology_id is None
    assert d.total_clusters == 2
    assert d.runner_up_witness_count == 1


def test_source_filter_restricts_to_one_source(db):
    """The --source flag scopes canonicalization to toponyms with at
    least one row from the named source. A toponym with NO rows from
    that source is skipped entirely."""
    # Seed a second source.
    db.conn.execute("INSERT INTO source (id, title) VALUES ('other_src', 'Other')")
    _add_toponym(db, 1, "FromTest")
    _add_toponym(db, 2, "FromOther")
    _add_etymon(db, 1, "x", "old-english")
    _add_etymology(
        db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")], source="test_src"
    )
    _add_etymology(
        db,
        toponym_id=1,
        extractor="haiku",
        elements=[(1, "old-english")],
        source="test_src",
    )
    _add_etymology(
        db, toponym_id=2, extractor="qwen", elements=[(1, "old-english")], source="other_src"
    )
    decisions = compute_canonical_decisions(db, source_id="test_src")
    assert {d.toponym_id for d in decisions} == {1}


def test_compute_is_pure_read(db):
    """compute_canonical_decisions must not mutate the DB. Pin this
    so a future refactor doesn't accidentally turn the read into a
    write."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    _add_etymology(db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="haiku", elements=[(1, "old-english")])
    before = db.conn.execute(
        "SELECT id, is_canonical, consensus_size, cluster_key FROM toponym_etymology"
    ).fetchall()
    compute_canonical_decisions(db)
    after = db.conn.execute(
        "SELECT id, is_canonical, consensus_size, cluster_key FROM toponym_etymology"
    ).fetchall()
    assert [tuple(r) for r in before] == [tuple(r) for r in after]


# ---------- apply_canonical_decisions ------------------------------------


def test_apply_marks_canonical_row_only(db):
    """The chosen row gets is_canonical=1; other rows in the SAME
    cluster stay is_canonical=0 but carry the cluster_size + key
    for audit visibility."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    qwen = _add_etymology(db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")])
    haiku = _add_etymology(db, toponym_id=1, extractor="haiku", elements=[(1, "old-english")])
    decisions = compute_canonical_decisions(db)
    apply_canonical_decisions(db, decisions)
    rows = list(
        db.conn.execute(
            "SELECT id, is_canonical, consensus_size, cluster_key "
            "FROM toponym_etymology ORDER BY id"
        )
    )
    assert rows[0]["id"] == qwen
    assert rows[0]["is_canonical"] == 1
    assert rows[0]["consensus_size"] == 2
    assert rows[0]["cluster_key"] == '[["x","old-english"]]'
    assert rows[1]["id"] == haiku
    assert rows[1]["is_canonical"] == 0  # other row in same cluster
    assert rows[1]["consensus_size"] == 2  # but participates in the agreement
    assert rows[1]["cluster_key"] == '[["x","old-english"]]'


def test_apply_stamps_non_canonical_clusters_too(db):
    """Even rows in losing clusters get cluster_size + cluster_key
    stamped — so the audit trail shows 'this hypothesis had N
    witnesses and lost to a cluster with M'."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "a", "old-english")
    _add_etymon(db, 2, "b", "old-english")
    _add_etymology(db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="haiku", elements=[(1, "old-english")])
    loser = _add_etymology(db, toponym_id=1, extractor="gemini", elements=[(2, "old-english")])
    decisions = compute_canonical_decisions(db)
    apply_canonical_decisions(db, decisions)
    row = db.conn.execute(
        "SELECT is_canonical, consensus_size, cluster_key FROM toponym_etymology WHERE id = ?",
        (loser,),
    ).fetchone()
    assert row["is_canonical"] == 0
    assert row["consensus_size"] == 1  # the loser's cluster had 1 witness
    assert row["cluster_key"] == '[["b","old-english"]]'


def test_apply_is_idempotent(db):
    """Re-running canonicalization over the same data must not change
    anything. Pin this so a future regression that always-writes
    can be caught."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    _add_etymology(db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="haiku", elements=[(1, "old-english")])
    d1 = compute_canonical_decisions(db)
    s1 = apply_canonical_decisions(db, d1)
    # 2 rows updated on first pass: BOTH rows in the winning cluster
    # get cluster_key/consensus_size stamped (one as canonical, one
    # as cluster sibling).
    assert s1.rows_updated == 2
    # Second pass over identical state: zero updates.
    d2 = compute_canonical_decisions(db)
    s2 = apply_canonical_decisions(db, d2)
    assert s2.rows_updated == 0
    assert s2.toponyms_promoted == 1


def test_apply_demotes_when_consensus_lost(db):
    """When a previously-canonical row's cluster loses consensus
    (e.g., the agreeing extractor gets removed via cleanup), the
    is_canonical flag must be cleared. Without this, stale canonicals
    would persist and downstream consumers would read outdated data."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    qwen = _add_etymology(db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")])
    haiku = _add_etymology(db, toponym_id=1, extractor="haiku", elements=[(1, "old-english")])
    # First pass: promote.
    apply_canonical_decisions(db, compute_canonical_decisions(db))
    assert (
        db.conn.execute(
            "SELECT is_canonical FROM toponym_etymology WHERE id = ?", (qwen,)
        ).fetchone()["is_canonical"]
        == 1
    )
    # Remove the haiku row — now only one witness.
    db.conn.execute("DELETE FROM toponym_etymology WHERE id = ?", (haiku,))
    db.conn.execute(
        "DELETE FROM toponym_etymology_element WHERE toponym_etymology_id = ?", (haiku,)
    )
    db.commit()
    # Second pass: demote. Also verify cluster_key + consensus_size
    # reflect the NEW single-witness state — without these checks a
    # regression that demoted is_canonical but left stale cluster
    # metadata could slip through. R1 pr-test-analyzer MEDIUM.
    apply_canonical_decisions(db, compute_canonical_decisions(db))
    row = db.conn.execute(
        "SELECT is_canonical, consensus_size, cluster_key FROM toponym_etymology WHERE id = ?",
        (qwen,),
    ).fetchone()
    assert row["is_canonical"] == 0
    assert row["consensus_size"] == 1
    # cluster_key is still stamped — single-witness rows aren't
    # canonical but they ARE clustered (cluster of size 1).
    assert row["cluster_key"] == '[["x","old-english"]]'


def test_canonical_view_filters_to_is_canonical_one(db):
    """The toponym_etymology_canonical VIEW must return only
    is_canonical=1 rows — that's the downstream consumer contract."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    _add_etymology(db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="haiku", elements=[(1, "old-english")])
    apply_canonical_decisions(db, compute_canonical_decisions(db))
    canonical_count = db.conn.execute(
        "SELECT COUNT(*) FROM toponym_etymology_canonical"
    ).fetchone()[0]
    all_count = db.conn.execute("SELECT COUNT(*) FROM toponym_etymology").fetchone()[0]
    assert all_count == 2
    assert canonical_count == 1


def test_on_decision_callback_fires_per_toponym(db):
    """The audit callback fires once per decision, even for toponyms
    with no consensus — operators need visibility into BOTH outcomes
    to triage."""
    _add_toponym(db, 1, "Yes")
    _add_toponym(db, 2, "No")
    _add_etymon(db, 1, "x", "old-english")
    _add_etymology(db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="haiku", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=2, extractor="qwen", elements=[(1, "old-english")])
    decisions = compute_canonical_decisions(db)
    captured: list[CanonicalDecision] = []
    apply_canonical_decisions(db, decisions, on_decision=captured.append)
    assert len(captured) == 2
    assert {c.toponym_id for c in captured} == {1, 2}
    by_id = {c.toponym_id: c for c in captured}
    assert by_id[1].promoted_etymology_id is not None
    assert by_id[2].promoted_etymology_id is None


# ---------- CLI integration ---------------------------------------------


def _seed_minimal_db(db_path: Path) -> None:
    """Build a small lexicon DB with one toponym + two agreeing
    extractors + one disagreeing for the CLI tests."""
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('s', 'S')")
    db.conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'X')")
    db.conn.execute("INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'a', 'oe')")
    db.conn.execute("INSERT INTO etymon (id, canonical_form, language) VALUES (2, 'b', 'oe')")
    for extractor, etymon_id in [
        ("llm:qwen", 1),
        ("anthropic:claude-haiku", 1),
        ("gemini:flash", 2),
    ]:
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, notes) VALUES (1, 's', ?)",
            (f"extracted_by:{extractor}; fixture",),
        )
        ety_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, 0, ?)",
            (ety_id, etymon_id),
        )
    db.commit()


def test_cli_dry_run_does_not_write(tmp_path):
    """Without --apply, the command reports counts but does NOT
    modify the DB. Critical for previewing a corpus-scale run."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "canonicalize-toponym-etymology", "--db", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    assert "promote=1" in result.output
    assert "(dry-run" in result.output
    # Verify nothing changed
    db = LexiconDB(db_path)
    canon = db.conn.execute(
        "SELECT COUNT(*) FROM toponym_etymology WHERE is_canonical = 1"
    ).fetchone()[0]
    assert canon == 0


def test_cli_apply_writes_canonical(tmp_path):
    """--apply commits the canonicalization. Verify is_canonical
    column gets set + the canonical view returns one row + the SPECIFIC
    row chosen is the lowest-id deterministic pick (qwen's row, not
    haiku's). R1 pr-test-analyzer MEDIUM: 'which row' wasn't pinned."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "canonicalize-toponym-etymology", "--db", str(db_path), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "promoted=1" in result.output
    assert "(APPLIED)" in result.output
    db = LexiconDB(db_path)
    rows = list(
        db.conn.execute("SELECT id, is_canonical, notes FROM toponym_etymology ORDER BY id")
    )
    # _seed_minimal_db inserts qwen, claude-haiku, gemini in that order
    # — qwen gets id=1, gets is_canonical=1 (lowest id in winning cluster).
    assert rows[0]["is_canonical"] == 1
    assert "llm:qwen" in rows[0]["notes"]
    # The other rows: haiku is in the SAME cluster (is_canonical=0,
    # not the chosen one); gemini is in a DIFFERENT cluster.
    assert rows[1]["is_canonical"] == 0
    assert rows[2]["is_canonical"] == 0


def test_cli_audit_out_writes_jsonl(tmp_path):
    """--audit-out emits one JSON record per decision so operators
    can diff cross-run promotions or feed downstream consumers a list
    of newly-canonical toponyms."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)
    audit_path = tmp_path / "audit.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "canonicalize-toponym-etymology",
            "--db",
            str(db_path),
            "--apply",
            "--audit-out",
            str(audit_path),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(ln) for ln in audit_path.read_text().splitlines() if ln.strip()]
    assert len(rows) == 1
    assert rows[0]["toponym_id"] == 1
    assert rows[0]["consensus_size"] == 2
    assert rows[0]["promoted_etymology_id"] is not None
    assert rows[0]["cluster_key"] == '[["a","oe"]]'


def test_cli_audit_out_refuses_to_overwrite_without_force(tmp_path):
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text("existing\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "canonicalize-toponym-etymology",
            "--db",
            str(db_path),
            "--apply",
            "--audit-out",
            str(audit_path),
        ],
    )
    assert result.exit_code != 0
    assert "exists" in result.output


def test_cli_source_filter(tmp_path):
    """--source restricts to toponyms with at least one row from the
    named source. End-to-end pin of the filter contract."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('s1', 'S1')")
    db.conn.execute("INSERT INTO source (id, title) VALUES ('s2', 'S2')")
    db.conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'A')")
    db.conn.execute("INSERT INTO toponym (id, modern_name) VALUES (2, 'B')")
    db.conn.execute("INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'x', 'oe')")
    for source_id, tid in [("s1", 1), ("s1", 1), ("s2", 2)]:
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, notes) VALUES (?, ?, ?)",
            (tid, source_id, f"extracted_by:llm:m{tid}; fx"),
        )
        ety = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, 0, 1)",
            (ety,),
        )
    db.commit()
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "canonicalize-toponym-etymology",
            "--db",
            str(db_path),
            "--source",
            "s1",
        ],
    )
    assert result.exit_code == 0, result.output
    # Only toponym 1 has rows in s1, so we examine 1 toponym (not 2).
    assert "Examining 1 toponym" in result.output
    assert "filtered to source_id='s1'" in result.output
    # Both s1 rows are the same extractor (llm:m1) → one witness → no
    # consensus. Pins the _echo_plan_preview no_consensus counter.
    assert "no_consensus=1" in result.output


def test_cli_preview_counts_no_elements(tmp_path):
    """The _echo_plan_preview no_elements counter is value-pinned end-
    to-end: a toponym whose only rows carry no element tuples renders
    `no_elements=1 promote=0 no_consensus=0` in the preview line.
    Guards against a counter swap in the C901 extraction (wyrd-8uvi)."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('s', 'S')")
    db.conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'X')")
    # Two distinct extractors, but NEITHER has element rows → all-empty.
    for extractor in ("llm:qwen", "anthropic:claude-haiku"):
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, notes) VALUES (1, 's', ?)",
            (f"extracted_by:{extractor}; no elements",),
        )
    db.commit()
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "canonicalize-toponym-etymology", "--db", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    assert "no_elements=1" in result.output
    assert "promote=0" in result.output
    assert "no_consensus=0" in result.output


def test_cli_handles_empty_db(tmp_path):
    """Fresh DB with no toponym_etymology rows: clean no-op exit."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "canonicalize-toponym-etymology", "--db", str(db_path), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "No toponym_etymology rows" in result.output


# ---------- migration tests ---------------------------------------------


def test_migration_idempotent_on_already_migrated_db(tmp_path):
    """Running migrate_schema twice on the same DB doesn't error and
    leaves the columns/index/view in a consistent state."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    # Second pass: should be a no-op for the new columns.
    result = migrate_schema(db)
    # Fresh installs have the columns from data/seed/lexicon.sql, so migration
    # sees them and skips the ADD COLUMN.
    assert result["toponym_etymology.is_canonical"] is False
    assert result["toponym_etymology.consensus_size"] is False
    assert result["toponym_etymology.cluster_key"] is False


def test_migration_adds_columns_to_legacy_db(tmp_path):
    """Simulate a pre-wyrd-08qv DB by manually creating the table
    without the new columns, then run migrate. Verify columns +
    index + view get added."""
    db_path = tmp_path / "lexicon.db"
    # Hand-build a minimal legacy DB shape (just enough for the
    # canonical migration to find toponym_etymology).
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, title TEXT);
        CREATE TABLE toponym (id INTEGER PRIMARY KEY, modern_name TEXT);
        CREATE TABLE etymon (id INTEGER PRIMARY KEY, canonical_form TEXT, language TEXT);
        CREATE TABLE toponym_etymology (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id INTEGER NOT NULL REFERENCES toponym(id),
            source_id TEXT NOT NULL REFERENCES source(id),
            page TEXT,
            historical_form TEXT,
            confidence TEXT,
            notes TEXT,
            attested_year INTEGER
        );
        CREATE TABLE toponym_etymology_element (
            toponym_etymology_id INTEGER NOT NULL REFERENCES toponym_etymology(id),
            ordinal INTEGER NOT NULL,
            etymon_id INTEGER NOT NULL REFERENCES etymon(id),
            inflection TEXT,
            surface_in_modern TEXT,
            confidence TEXT  -- wyrd-2n1
        );
        """
    )
    conn.commit()
    conn.close()
    # Run only the toponym-etymology canonical migration helper directly
    # so we don't have to satisfy the full etymon-canonical schema for
    # this legacy fixture.
    from wyrd.generators.kenning.lexicon import _migrate_toponym_etymology_canonical

    db = LexiconDB(db_path)
    applied: dict[str, bool] = {}
    _migrate_toponym_etymology_canonical(db, applied)
    db.commit()
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(toponym_etymology)")}
    assert {"is_canonical", "consensus_size", "cluster_key"}.issubset(cols)
    views = {
        r[0]
        for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='toponym_etymology_canonical'"
        )
    }
    assert views == {"toponym_etymology_canonical"}
    assert applied["toponym_etymology.is_canonical"] is True
    assert applied["toponym_etymology.consensus_size"] is True
    assert applied["toponym_etymology.cluster_key"] is True


# ---------- _build_cluster_key ------------------------------------------


def test_cluster_key_is_deterministic_for_same_elements():
    a = (("bedeling", "old-english"), ("tun", "old-english"))
    b = (("bedeling", "old-english"), ("tun", "old-english"))
    assert _build_cluster_key(a) == _build_cluster_key(b)


def test_cluster_key_respects_ordinal_order():
    """'tun + bedeling' is NOT the same decomposition as
    'bedeling + tun'. Different element order means different
    morphological reading."""
    a = (("bedeling", "old-english"), ("tun", "old-english"))
    b = (("tun", "old-english"), ("bedeling", "old-english"))
    assert _build_cluster_key(a) != _build_cluster_key(b)


def test_cluster_key_handles_pipe_and_colon_in_form():
    """A normalized form containing '|' or ':' (rare but possible
    for reconstructed forms or unusual transliterations) must NOT
    collide with a different element-tuple's key. R1 silent-failure-
    hunter MEDIUM + type-design MEDIUM: the previous pipe-delimited
    'form:lang|form:lang' format was ambiguous on these chars."""
    # Two semantically-different element tuples that the old
    # pipe-delimited format would have collapsed.
    tuple_a = (("a:b", "lang"),)
    tuple_b = (("a", "b:lang"),)
    assert _build_cluster_key(tuple_a) != _build_cluster_key(tuple_b)
    # And one with a pipe in the form:
    tuple_c = (("a|b", "lang"),)
    tuple_d = (("a", "lang|b"),)
    assert _build_cluster_key(tuple_c) != _build_cluster_key(tuple_d)


# ---------- new behaviors added in R1 -----------------------------------


def test_anonymous_rows_do_not_witness_consensus(db):
    """A row without ``extracted_by:`` prefix contributes ZERO
    witnesses for consensus purposes. Two anonymous rows with the
    same element-tuple do NOT promote — consensus requires at least
    2 DISTINCT extractor tags. R1 silent-failure-hunter HIGH: the
    previous ``_anon:{row.id}`` scheme allowed false consensus from
    multiple legacy/curated rows."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    # Two rows with notes lacking the extracted_by prefix.
    db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) "
        "VALUES (1, 'test_src', 'hand-curated by scholar A')"
    )
    db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) "
        "VALUES (1, 'test_src', 'hand-curated by scholar B')"
    )
    db.commit()
    for ety_id in (1, 2):
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, 0, 1)",
            (ety_id,),
        )
    db.commit()
    decisions = compute_canonical_decisions(db)
    d = decisions[0]
    # Two anonymous rows form a single cluster of size 2, but witness
    # count is 0 (no extractor tags), so no canonical promotion.
    assert d.promoted_etymology_id is None
    assert d.consensus_size == 0


def test_anonymous_row_alongside_llm_does_not_promote_alone(db):
    """1 LLM row + 1 anonymous row in the same cluster has only 1
    witness (the LLM). Cannot promote to canonical (needs >=2)."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    # Anonymous row first (id=1), LLM row second (id=2).
    db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) "
        "VALUES (1, 'test_src', 'no prefix here')"
    )
    db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) "
        "VALUES (1, 'test_src', 'extracted_by:llm:qwen; fixture')"
    )
    db.commit()
    for ety_id in (1, 2):
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, 0, 1)",
            (ety_id,),
        )
    db.commit()
    decisions = compute_canonical_decisions(db)
    d = decisions[0]
    # Cluster has 2 rows, 1 witness → not promoted.
    assert d.promoted_etymology_id is None
    assert d.consensus_size == 0
    assert d.runner_up_witness_count == 1  # the LLM row contributes 1


def test_two_llms_agree_with_anonymous_rider(db):
    """2 LLM extractors agreeing + 1 anonymous row in the same
    cluster: 2 witnesses → canonical promotion. The anonymous row
    rides along as cluster member (gets cluster_key + consensus_size
    stamped) but doesn't itself count as a witness."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    qwen = _add_etymology(db, toponym_id=1, extractor="llm:qwen", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="anthropic:claude", elements=[(1, "old-english")])
    # Third row: anonymous (notes WITHOUT extracted_by prefix).
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) "
        "VALUES (1, 'test_src', 'no extractor tag here')"
    )
    anon_id = cur.lastrowid
    db.conn.execute(
        "INSERT INTO toponym_etymology_element "
        "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, 0, 1)",
        (anon_id,),
    )
    db.commit()
    decisions = compute_canonical_decisions(db)
    d = decisions[0]
    assert d.consensus_size == 2  # qwen + claude only, anon doesn't count
    assert d.promoted_etymology_id == qwen  # lowest id in winning cluster
    apply_canonical_decisions(db, decisions)
    # Anonymous rider got cluster metadata too.
    row = db.conn.execute(
        "SELECT is_canonical, consensus_size, cluster_key FROM toponym_etymology WHERE id = ?",
        (anon_id,),
    ).fetchone()
    assert row["is_canonical"] == 0  # not the winning row
    assert row["consensus_size"] == 2  # cluster size visible
    assert row["cluster_key"] == '[["x","old-english"]]'


def test_e2e_bedlington_compute_apply_view(db):
    """End-to-end Bedlington shape: 3 extractors, 2 agree, 1 has
    extra element. Apply → query canonical VIEW → expect exactly one
    row. The full happy path the production pipeline relies on.
    R1 test-coverage-reviewer MEDIUM: no single test stitched compute
    → apply → VIEW → assert in one Bedlington-shape scenario."""
    _add_toponym(db, 1, "Bedlington")
    _add_etymon(db, 1, "bedeling", "old-english")
    _add_etymon(db, 2, "tun", "old-english")
    _add_etymon(db, 3, "bedel", "old-english")  # haiku's extra element
    qwen = _add_etymology(
        db, toponym_id=1, extractor="llm:qwen", elements=[(1, "old-english"), (2, "old-english")]
    )
    _add_etymology(
        db,
        toponym_id=1,
        extractor="anthropic:claude-haiku",
        elements=[(1, "old-english"), (2, "old-english"), (3, "old-english")],
    )
    _add_etymology(
        db,
        toponym_id=1,
        extractor="gemini:flash",
        elements=[(1, "old-english"), (2, "old-english")],
    )
    apply_canonical_decisions(db, compute_canonical_decisions(db))
    # The VIEW returns exactly the one canonical row.
    rows = list(db.conn.execute("SELECT * FROM toponym_etymology_canonical"))
    assert len(rows) == 1
    assert rows[0]["id"] == qwen
    assert rows[0]["consensus_size"] == 2
    assert rows[0]["cluster_key"] == '[["bedeling","old-english"],["tun","old-english"]]'


def test_multi_source_toponym(db):
    """A toponym with toponym_etymology rows from MULTIPLE sources
    treats them all as candidates for consensus. R1 test-coverage-
    reviewer MEDIUM: source-agnostic clustering wasn't pinned."""
    # Add a second source.
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src2', 'Test Source 2')")
    db.commit()
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    qwen_a = _add_etymology(
        db, toponym_id=1, extractor="llm:qwen", elements=[(1, "old-english")], source="test_src"
    )
    _add_etymology(
        db,
        toponym_id=1,
        extractor="gemini:flash",
        elements=[(1, "old-english")],
        source="test_src2",
    )
    decisions = compute_canonical_decisions(db)
    d = decisions[0]
    # Two distinct extractors agree, regardless of source — consensus.
    assert d.consensus_size == 2
    assert d.promoted_etymology_id == qwen_a


def test_compute_is_pure_read_returns_stable_decisions(db):
    """compute_canonical_decisions called twice on the same DB must
    return field-identical CanonicalDecision lists. R1 pr-test
    LOW: determinism across repeat calls wasn't pinned."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    _add_etymology(db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")])
    _add_etymology(db, toponym_id=1, extractor="haiku", elements=[(1, "old-english")])
    d1 = compute_canonical_decisions(db)
    d2 = compute_canonical_decisions(db)
    # CanonicalDecision is now frozen=True, so equality is field-by-field.
    assert d1 == d2


def test_canonical_decision_is_frozen():
    """CanonicalDecision is documented as frozen for audit-log safety.
    Mutation attempts must raise dataclasses.FrozenInstanceError.
    R1 type-design MEDIUM + R2 pr-test-analyzer MEDIUM: previous test
    used pytest.raises((AttributeError, Exception)) which would
    swallow any error and make the test near-tautological."""
    from dataclasses import FrozenInstanceError

    d = CanonicalDecision(
        toponym_id=1,
        promoted_etymology_id=2,
        consensus_size=2,
        cluster_key="x",
        runner_up_witness_count=1,
        total_clusters=2,
    )
    with pytest.raises(FrozenInstanceError):
        d.toponym_id = 99  # type: ignore[misc]


def test_cli_verbose_outputs_per_decision_lines(tmp_path):
    """--verbose surfaces per-toponym decision lines. R1 test-coverage
    HIGH: this flag was unpinned and a refactor could silently break
    the format operators rely on for triage."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "canonicalize-toponym-etymology",
            "--db",
            str(db_path),
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    # Promote line for the consensual toponym
    assert "promote ety#" in result.output
    assert "consensus=2" in result.output
    assert "clusters=2" in result.output
    assert "runner_up=1" in result.output


def test_cli_verbose_outputs_no_consensus_line(tmp_path):
    """--verbose surfaces the 'no consensus' decision line for
    single-witness toponyms. R2 test-coverage HIGH: the promote
    branch was tested but the no-consensus branch is a separate
    format string that could silently break."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('s', 'S')")
    db.conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'X')")
    db.conn.execute("INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'x', 'oe')")
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) "
        "VALUES (1, 's', 'extracted_by:llm:qwen; fixture')"
    )
    db.conn.execute(
        "INSERT INTO toponym_etymology_element "
        "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, 0, 1)",
        (cur.lastrowid,),
    )
    db.commit()
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "canonicalize-toponym-etymology",
            "--db",
            str(db_path),
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    # No-consensus format pinned
    assert "no consensus" in result.output
    assert "max_witnesses=" in result.output


def test_compute_canonical_plans_direct(db):
    """The lower-level builder ``compute_canonical_plans`` is the
    preferred API for the CLI + future callers needing the apply-
    side row updates. Pin its shape directly so a refactor that
    breaks _RowUpdate field ordering or _ToponymPlan structure
    fails this test rather than slipping through the high-level
    compute_canonical_decisions wrapper. R2 test-coverage MEDIUM."""
    from wyrd.generators.kenning.toponym_etymology_canonical import compute_canonical_plans

    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    qwen = _add_etymology(db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")])
    haiku = _add_etymology(db, toponym_id=1, extractor="haiku", elements=[(1, "old-english")])
    plans = compute_canonical_plans(db)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.decision.toponym_id == 1
    assert plan.decision.promoted_etymology_id == qwen
    assert plan.all_empty_elements is False
    # row_updates is a tuple of _RowUpdate NamedTuples — field access
    # via attribute (not positional indexing).
    assert len(plan.row_updates) == 2
    qwen_update = next(u for u in plan.row_updates if u.row_id == qwen)
    haiku_update = next(u for u in plan.row_updates if u.row_id == haiku)
    assert qwen_update.is_canonical == 1
    assert haiku_update.is_canonical == 0
    assert qwen_update.consensus_size == 2
    assert haiku_update.consensus_size == 2
    assert qwen_update.cluster_key == '[["x","old-english"]]'
    assert haiku_update.cluster_key == '[["x","old-english"]]'


def test_anonymous_and_empty_elements_does_not_promote(db):
    """Two anonymous rows with EMPTY element tuples: cluster_key
    is the same ('[]'), 0 witnesses, all-empty-elements branch
    fires → no promotion. Pins the interaction between the new
    anonymous-witness semantics and the all-empty short-circuit.
    R2 test-coverage MEDIUM."""
    _add_toponym(db, 1, "X")
    db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) "
        "VALUES (1, 'test_src', 'no extractor tag here')"
    )
    db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, notes) "
        "VALUES (1, 'test_src', 'also no extractor tag')"
    )
    db.commit()
    summary = apply_canonical_decisions(db, compute_canonical_decisions(db))
    assert summary.toponyms_no_elements == 1
    assert summary.toponyms_promoted == 0
    rows = list(
        db.conn.execute(
            "SELECT is_canonical, consensus_size, cluster_key FROM toponym_etymology ORDER BY id"
        )
    )
    for row in rows:
        assert row["is_canonical"] == 0
        # All-empty branch stamps consensus_size=0 (matches the
        # actual witness count) and cluster_key=None. R2 silent-
        # failure MEDIUM.
        assert row["consensus_size"] == 0
        assert row["cluster_key"] is None


def test_stale_decision_does_not_write_wrong_canonical(db):
    """Back-compat path: caller passes a stale CanonicalDecision
    referring to a row that no longer exists. The apply path
    must NOT inherit the stale promoted_etymology_id — it must
    re-derive from current DB state. R2 silent-failure-hunter
    HIGH: previously the stale id would have been written as
    canonical (matching nothing in WHERE id=?, silent zero-update;
    audit JSONL still claims promotion)."""
    _add_toponym(db, 1, "X")
    _add_etymon(db, 1, "x", "old-english")
    qwen = _add_etymology(db, toponym_id=1, extractor="qwen", elements=[(1, "old-english")])
    haiku = _add_etymology(db, toponym_id=1, extractor="haiku", elements=[(1, "old-english")])
    # Capture stale decision before any DB modification.
    stale_decisions = compute_canonical_decisions(db)
    # Now delete the would-be winner row (the qwen row).
    db.conn.execute("DELETE FROM toponym_etymology WHERE id = ?", (qwen,))
    db.conn.execute("DELETE FROM toponym_etymology_element WHERE toponym_etymology_id = ?", (qwen,))
    db.commit()
    # Apply with the STALE decision. Without the fix, this would
    # silently fail to write canonical (qwen row gone, haiku row
    # not marked). With the fix, the back-compat path re-derives
    # plans from current state — only haiku remains, single witness,
    # no consensus → demoted to is_canonical=0.
    apply_canonical_decisions(db, stale_decisions)
    surviving = db.conn.execute(
        "SELECT is_canonical, consensus_size FROM toponym_etymology WHERE id = ?",
        (haiku,),
    ).fetchone()
    assert surviving["is_canonical"] == 0  # single witness — no consensus
    assert surviving["consensus_size"] == 1  # cluster has 1 witness now
