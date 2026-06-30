"""wyrd-an8u: surrounding whitespace is never part of a morpheme's identity.

A dirty usage like ``'Oak- '`` (trailing space) used to fold to a distinct
bare surface (``'Oak '``) from the clean ``'Oak-'`` → ``'Oak'``, duplicating
the meaning row and splitting proportion weight across the space/no-space
variants. The three surface-key creation sites now strip surrounding
whitespace alongside the dash-fold:

  * ``word._bare_surface`` — the proportions usages / single_usages /
    bare_word_position key path (and the render key).
  * ``runtime_db_export._bare_modern_usage`` — the meaning / dormant-morpheme
    blob ``modern_usage`` key path.
  * ``proportions_builder._accumulate_attested_languages`` — the
    proportions_attested_language key path.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from wyrd.generators.kenning.lexicon.proportions_builder import (
    _accumulate_attested_languages,
)
from wyrd.generators.kenning.lexicon.runtime_db_export import (
    _bare_modern_usage,
    _insert_attested_languages,
    _iter_single_usage_rows,
    _iter_usage_rows,
    _write_meanings,
)
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.word import Word, _bare_surface


def _meaning(usage: str) -> Meaning:
    return Meaning(
        usage=usage,
        tags=["x"],
        meanings=["gloss"],
        sources={"old_english": [usage.lower().replace("-", "").strip()]},
    )


# ---- word._bare_surface (proportions usages + render key) ------------------


def test_bare_surface_strips_surrounding_whitespace():
    assert _bare_surface(_meaning("Oak- ")) == "Oak"  # trailing
    assert _bare_surface(_meaning(" Oak-")) == "Oak"  # leading
    assert _bare_surface(_meaning("Har ")) == "Har"


def test_bare_surface_preserves_interior_whitespace():
    # ``.strip()`` trims only the ends — a genuine multi-token surface keeps
    # its interior space (guards against a regression to ``.replace(" ", "")``).
    assert _bare_surface(_meaning("Mac Leod")) == "Mac Leod"


def test_bare_surface_dirty_and_clean_fold_to_same_identity():
    # The merge guarantee: a whitespace-dirty surface keys identically to the
    # clean one, so their proportion counts sum instead of splitting.
    assert _bare_surface(_meaning("Oak- ")) == _bare_surface(_meaning("Oak-"))


def test_word_lone_sample_uses_whitespace_clean_surface():
    assert Word([_meaning("Oak- ")]).get_lone_samples() == {"Oak"}


def test_word_compound_samples_use_whitespace_clean_surface():
    samples = Word([_meaning("Oak- "), _meaning("ton")]).get_samples()
    assert ("Oak", "pre") in samples
    assert all(" " not in surface for surface, _position in samples)


# ---- runtime_db_export._bare_modern_usage (meaning blob key) ---------------


def test_bare_modern_usage_strips_dash_and_whitespace():
    # D53: case is identity-neutral too (render owns it), so the bare surface is
    # also lower-cased — dashes + surrounding whitespace + case all folded.
    assert _bare_modern_usage("Oak- ") == "oak"
    assert _bare_modern_usage(" Oak-") == "oak"  # leading
    assert _bare_modern_usage("Har ") == "har"
    assert _bare_modern_usage("Mac Leod") == "mac leod"  # interior space kept, case folded


def test_bare_modern_usage_dirty_and_clean_collide():
    # Same key → _write_meanings groups them into ONE meaning row (unioned
    # entries) rather than two whitespace-variant rows.
    assert _bare_modern_usage("Oak- ") == _bare_modern_usage("Oak-")


def test_write_meanings_merges_whitespace_variant_into_one_row():
    """The downstream consequence the ticket targets: after the export de-dash
    + whitespace-strip + case-fold (D53), a dirty ``'Oak- '`` word and a clean
    ``'Oak-'`` word collapse into ONE ``'oak'`` meaning row with both entries
    unioned — not two rows splitting the weight."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE meaning (usage_key TEXT, primary_language TEXT, "
        "stratum TEXT, data BLOB NOT NULL)"
    )
    subjects = [
        {
            "meaning": ["oak tree"],
            "modifier_tags": ["flora"],
            "modifier_type": "noun",
            "words": [{"modern_usage": "Oak- ", "old_english": ["ac"]}],
        },
        {
            "meaning": ["oak wood"],
            "modifier_tags": ["flora"],
            "modifier_type": "noun",
            "words": [{"modern_usage": "Oak-", "old_english": ["ac"]}],
        },
    ]
    # Mirror write_runtime_db's pre-_write_meanings normalization.
    for subject in subjects:
        for word in subject["words"]:
            word["modern_usage"] = _bare_modern_usage(word["modern_usage"])

    n = _write_meanings(conn, subjects)
    assert n == 1  # one merged 'oak' row, not two whitespace variants
    rows = conn.execute("SELECT usage_key, data FROM meaning").fetchall()
    assert [r[0] for r in rows] == ["oak"]
    entries = json.loads(rows[0][1])["entries"]
    assert len(entries) == 2  # both subjects' entries unioned onto the winner
    conn.close()


def test_write_meanings_merges_case_variant_into_one_row():
    """D53 (wyrd-x5y4.2): case is never morpheme identity. A word-initial
    ``'Abbey'`` (capitalized because it led a source toponym) and an internal
    ``'abbey'`` are the SAME morpheme — they collapse into ONE ``'abbey'`` row
    with both entries unioned, not two case-variant rows splitting the weight.
    This is the ~566-duplicate fix; the mechanism is the existing de-dash union
    (regroup-then-repick), now keyed on the case-folded bare surface.

    The degraded-metadata case is covered too: the capitalized variant carries
    NO ``primary_language`` while the lowercase one is ``modern_english`` — the
    re-pick must land on the populated value, not the empty one."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE meaning (usage_key TEXT, primary_language TEXT, "
        "stratum TEXT, data BLOB NOT NULL)"
    )
    subjects = [
        {
            "meaning": ["monastery"],
            "modifier_tags": ["religious"],
            "modifier_type": "noun",
            # Word-initial occurrence: capitalized in the source, no language axis
            # (the degraded duplicate — mirrors the real 'Acre' with no language).
            "words": [{"modern_usage": "Abbey"}],
        },
        {
            "meaning": ["abbey ruins"],
            "modifier_tags": ["religious"],
            "modifier_type": "noun",
            # Internal occurrence: lowercase, with a real language-form axis.
            "words": [{"modern_usage": "abbey", "modern_english": ["abbey"]}],
        },
    ]
    # Mirror write_runtime_db's pre-_write_meanings normalization (de-dash +
    # whitespace-strip + case-fold).
    for subject in subjects:
        for word in subject["words"]:
            word["modern_usage"] = _bare_modern_usage(word["modern_usage"])

    n = _write_meanings(conn, subjects)
    assert n == 1  # 'Abbey' + 'abbey' → one row, not two case variants
    rows = conn.execute("SELECT usage_key, primary_language, data FROM meaning").fetchall()
    assert [r[0] for r in rows] == ["abbey"]  # canonical lower-case key
    # Re-pick over the union lands on the POPULATED language, not the degraded
    # variant's absent one.
    assert rows[0][1] == "modern_english"
    entries = json.loads(rows[0][2])["entries"]
    assert len(entries) == 2  # both subjects' entries unioned onto the winner
    conn.close()


# ---- proportions_builder attested-language key -----------------------------


def test_attested_language_surface_strips_whitespace():
    name = SimpleNamespace(words={"w": [Word([_meaning("Oak- ")])]})
    attested: dict[str, set[str]] = {}
    _accumulate_attested_languages(name, attested)
    # Keyed by the clean bare surface, not 'oak ' with a trailing space.
    assert "oak" in attested
    assert all(" " not in surface for surface in attested)


# ---- export write-time defensive folds (operator-JSON / legacy keys) -------


def test_iter_usage_rows_strips_whitespace_dirty_key():
    # The nested part-pool writer folds the key to a bare, whitespace-clean
    # surface.
    rows = list(_iter_usage_rows({"Oak- ": {"pre": 5}}))
    assert rows == [("Oak", "pre", 5)]


def test_iter_single_usage_rows_strips_whitespace_dirty_key():
    # The single_usage (lone-word) writer is a SEPARATE path from
    # _iter_usage_rows; pin its own defensive whitespace fold so a revert to
    # replace-only is caught.
    assert list(_iter_single_usage_rows({"Oak- ": 5})) == [("Oak", "bare", 5)]


def test_insert_attested_languages_strips_whitespace_dirty_key():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE proportions_attested_language "
        "(culture TEXT, usage_key TEXT, primary_language TEXT)"
    )
    n = _insert_attested_languages(conn, "english", {"Oak- ": ["old_english"]})
    assert n == 1
    usage_key = conn.execute("SELECT usage_key FROM proportions_attested_language").fetchone()[0]
    assert usage_key == "oak"  # bare + lowercased + whitespace-stripped
    conn.close()
