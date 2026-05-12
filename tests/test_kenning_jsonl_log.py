"""Tests for the JSONL event-log replay kernel — wyrd-f295.

The kernel is the foundation of the JSONL-as-source-of-truth migration:
everything downstream (``lexicon dump-jsonl``, ``lexicon build``,
``lexicon compact``) relies on :func:`replay` having predictable
semantics. Tests focus on:

- ``_op`` vocabulary semantics (add / set / patch / remove)
- patch field semantics (scalar last-write-wins, ``_add`` / ``_remove`` /
  ``_set`` suffixes for array fields)
- round-trip invariants (replay/compact idempotency)
- error reporting (line numbers, bad ops, duplicate adds)
- list-type rows (append-only, no event semantics)
- file I/O (read_jsonl skips blanks + comments, write_jsonl preserves
  non-ASCII glosses)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyrd.generators.kenning.jsonl_log import (
    KEYED_TYPES,
    LIST_TYPES,
    ReplayError,
    compact,
    compact_file,
    read_jsonl,
    replay,
    replay_file,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Canonical-state rows (no _op): seed the initial state
# ---------------------------------------------------------------------------


def test_canonical_state_seed_etymon():
    rows = [
        {
            "_type": "etymon",
            "ref": "old-english:cot",
            "language": "old-english",
            "canonical_form": "cot",
            "glosses": ["cottage", "hut"],
            "tags": ["architecture"],
        }
    ]
    state = replay(rows)
    assert state.get("etymon", "old-english:cot") == {
        "language": "old-english",
        "canonical_form": "cot",
        "glosses": ["cottage", "hut"],
        "tags": ["architecture"],
    }


def test_canonical_state_multiple_keyed_rows_sort_by_ref():
    rows = [
        {"_type": "etymon", "ref": "old-english:tun", "canonical_form": "tun"},
        {"_type": "etymon", "ref": "old-english:cot", "canonical_form": "cot"},
    ]
    state = replay(rows)
    canonical = state.to_canonical_rows()
    refs = [r["ref"] for r in canonical if r["_type"] == "etymon"]
    assert refs == ["old-english:cot", "old-english:tun"]


# ---------------------------------------------------------------------------
# _op = add  (strict insert)
# ---------------------------------------------------------------------------


def test_op_add_new_ref():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "old-english:cot", "canonical_form": "cot"},
    ]
    state = replay(rows)
    assert state.get("etymon", "old-english:cot") == {"canonical_form": "cot"}


def test_op_add_duplicate_raises():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "old-english:cot", "canonical_form": "cot"},
        {"_op": "add", "_type": "etymon", "ref": "old-english:cot", "canonical_form": "cot"},
    ]
    with pytest.raises(ReplayError, match="duplicate add"):
        replay(rows)


def test_op_add_on_existing_canonical_raises():
    rows = [
        {"_type": "etymon", "ref": "old-english:cot", "canonical_form": "cot"},
        {"_op": "add", "_type": "etymon", "ref": "old-english:cot", "canonical_form": "x"},
    ]
    with pytest.raises(ReplayError, match="duplicate add"):
        replay(rows)


# ---------------------------------------------------------------------------
# _op = set  (idempotent overwrite)
# ---------------------------------------------------------------------------


def test_op_set_overwrites():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "old-english:cot", "glosses": ["a"]},
        {"_op": "set", "_type": "etymon", "ref": "old-english:cot", "glosses": ["b"]},
    ]
    state = replay(rows)
    assert state.get("etymon", "old-english:cot") == {"glosses": ["b"]}


def test_op_set_idempotent_on_missing_ref():
    rows = [
        {"_op": "set", "_type": "etymon", "ref": "old-english:cot", "glosses": ["a"]},
    ]
    state = replay(rows)
    assert state.get("etymon", "old-english:cot") == {"glosses": ["a"]}


# ---------------------------------------------------------------------------
# _op = patch  (partial update)
# ---------------------------------------------------------------------------


def test_op_patch_scalar_last_write_wins():
    rows = [
        {
            "_op": "add",
            "_type": "etymon",
            "ref": "oe:cot",
            "canonical_form": "cot",
            "stratum": "common",
        },
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "stratum": "rare"},
    ]
    state = replay(rows)
    assert state.get("etymon", "oe:cot") == {"canonical_form": "cot", "stratum": "rare"}


def test_op_patch_array_add_unions():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "glosses": ["a", "b"]},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "glosses_add": ["b", "c"]},
    ]
    state = replay(rows)
    # "b" already present — not duplicated; "c" appended; order preserved.
    assert state.get("etymon", "oe:cot") == {"glosses": ["a", "b", "c"]}


def test_op_patch_array_remove_diffs():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "glosses": ["a", "b", "c"]},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "glosses_remove": ["b"]},
    ]
    state = replay(rows)
    assert state.get("etymon", "oe:cot") == {"glosses": ["a", "c"]}


def test_op_patch_array_set_replaces():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "glosses": ["a", "b", "c"]},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "glosses_set": ["x"]},
    ]
    state = replay(rows)
    assert state.get("etymon", "oe:cot") == {"glosses": ["x"]}


def test_op_patch_add_on_missing_field_initializes():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "canonical_form": "cot"},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "glosses_add": ["new"]},
    ]
    state = replay(rows)
    assert state.get("etymon", "oe:cot") == {"canonical_form": "cot", "glosses": ["new"]}


def test_op_patch_missing_ref_raises():
    rows = [
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "stratum": "rare"},
    ]
    with pytest.raises(ReplayError, match="patch on missing"):
        replay(rows)


def test_op_patch_mixed_scalar_and_array_in_single_row():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "stratum": "common", "tags": ["arch"]},
        {
            "_op": "patch",
            "_type": "etymon",
            "ref": "oe:cot",
            "stratum": "rare",
            "tags_add": ["domestic"],
            "tags_remove": ["arch"],
        },
    ]
    state = replay(rows)
    assert state.get("etymon", "oe:cot") == {"stratum": "rare", "tags": ["domestic"]}


# ---------------------------------------------------------------------------
# _op = remove  (delete by ref)
# ---------------------------------------------------------------------------


def test_op_remove_existing():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "canonical_form": "cot"},
        {"_op": "remove", "_type": "etymon", "ref": "oe:cot"},
    ]
    state = replay(rows)
    assert state.get("etymon", "oe:cot") is None


def test_op_remove_missing_is_noop():
    rows = [
        {"_op": "remove", "_type": "etymon", "ref": "oe:cot"},
    ]
    state = replay(rows)
    assert state.get("etymon", "oe:cot") is None


def test_op_add_after_remove_succeeds():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "canonical_form": "cot"},
        {"_op": "remove", "_type": "etymon", "ref": "oe:cot"},
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "canonical_form": "cott"},
    ]
    state = replay(rows)
    assert state.get("etymon", "oe:cot") == {"canonical_form": "cott"}


# ---------------------------------------------------------------------------
# List-type rows (citations, attestations, ...): append-only
# ---------------------------------------------------------------------------


def test_list_type_canonical_appends():
    rows = [
        {"_type": "citation", "etymon_ref": "oe:cot", "page": 15, "attested_year": 1086},
        {"_type": "citation", "etymon_ref": "oe:cot", "page": 16, "attested_year": 1086},
    ]
    state = replay(rows)
    assert state.lists["citation"] == [
        {"etymon_ref": "oe:cot", "page": 15, "attested_year": 1086},
        {"etymon_ref": "oe:cot", "page": 16, "attested_year": 1086},
    ]


def test_list_type_with_op_add_also_appends():
    rows = [
        {"_op": "add", "_type": "citation", "etymon_ref": "oe:cot", "page": 1},
    ]
    state = replay(rows)
    assert state.lists["citation"] == [{"etymon_ref": "oe:cot", "page": 1}]


def test_list_type_rejects_patch():
    rows = [
        {"_op": "patch", "_type": "citation", "etymon_ref": "oe:cot", "page": 1},
    ]
    with pytest.raises(ReplayError, match="does not support _op"):
        replay(rows)


def test_list_type_rejects_remove():
    rows = [
        {"_op": "remove", "_type": "citation", "etymon_ref": "oe:cot"},
    ]
    with pytest.raises(ReplayError, match="does not support _op"):
        replay(rows)


# ---------------------------------------------------------------------------
# Validation: bad inputs
# ---------------------------------------------------------------------------


def test_unknown_type_raises():
    with pytest.raises(ReplayError, match="unknown _type"):
        replay([{"_type": "made-up", "ref": "x"}])


def test_missing_type_raises():
    with pytest.raises(ReplayError, match="missing '_type'"):
        replay([{"ref": "x"}])


def test_unknown_op_raises():
    with pytest.raises(ReplayError, match="unknown _op"):
        replay([{"_op": "MERGE", "_type": "etymon", "ref": "x"}])


def test_keyed_type_missing_ref_raises():
    with pytest.raises(ReplayError, match="missing non-empty 'ref'"):
        replay([{"_type": "etymon", "canonical_form": "cot"}])


def test_keyed_type_empty_ref_raises():
    with pytest.raises(ReplayError, match="missing non-empty 'ref'"):
        replay([{"_type": "etymon", "ref": ""}])


def test_non_dict_row_raises():
    with pytest.raises(ReplayError, match="not a JSON object"):
        replay(["not a dict"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Round-trip invariants
# ---------------------------------------------------------------------------


def _state_dict(state):
    """Compare-friendly dict snapshot of a ReplayState."""
    return {
        "keyed": {t: dict(d) for t, d in state.keyed.items()},
        "lists": {t: list(v) for t, v in state.lists.items()},
    }


def test_compact_preserves_state():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "glosses": ["a"]},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "glosses_add": ["b"]},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "stratum": "rare"},
        {"_type": "citation", "etymon_ref": "oe:cot", "page": 5},
    ]
    state_a = replay(rows)
    compacted = compact(rows)
    state_b = replay(compacted)
    assert _state_dict(state_a) == _state_dict(state_b)


def test_compact_is_idempotent():
    rows = [
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "glosses": ["a"]},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "glosses_add": ["b"]},
    ]
    once = compact(rows)
    twice = compact(once)
    assert once == twice


def test_replay_event_then_compact_equivalent_to_single_set():
    event_form = [
        {"_op": "add", "_type": "etymon", "ref": "oe:cot", "glosses": ["a"]},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "glosses_add": ["b"]},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "stratum": "rare"},
    ]
    canonical_form = [
        {"_type": "etymon", "ref": "oe:cot", "glosses": ["a", "b"], "stratum": "rare"},
    ]
    assert _state_dict(replay(event_form)) == _state_dict(replay(canonical_form))


def test_mixed_canonical_and_events_replay_correctly():
    """Canonical-state row at top of file + later events on top of it."""
    rows = [
        {"_type": "etymon", "ref": "oe:cot", "glosses": ["a"], "stratum": "common"},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "glosses_add": ["b"]},
        {"_op": "patch", "_type": "etymon", "ref": "oe:cot", "stratum": "rare"},
    ]
    state = replay(rows)
    assert state.get("etymon", "oe:cot") == {"glosses": ["a", "b"], "stratum": "rare"}


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def test_read_jsonl_skips_blanks_and_comments(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        "\n"
        "# a comment\n"
        '{"_type": "etymon", "ref": "oe:cot"}\n'
        "\n"
        '{"_type": "etymon", "ref": "oe:tun"}\n'
    )
    rows = list(read_jsonl(p))
    # Line numbers are 1-indexed against the raw file.
    assert [(ln, r["ref"]) for ln, r in rows] == [(3, "oe:cot"), (5, "oe:tun")]


def test_read_jsonl_bad_json_reports_line(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"_type": "etymon", "ref": "ok"}\nnot-json\n')
    with pytest.raises(ReplayError) as exc_info:
        list(read_jsonl(p))
    assert exc_info.value.line_no == 2


def test_replay_file_annotates_line_numbers(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"_type": "etymon", "ref": "oe:cot"}\n'
        '{"_op": "patch", "_type": "etymon", "ref": "oe:nope", "stratum": "rare"}\n'
    )
    with pytest.raises(ReplayError) as exc_info:
        replay_file(p)
    assert exc_info.value.line_no == 2


def test_write_jsonl_round_trip(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    rows = [
        {"_type": "etymon", "ref": "oe:þorn", "glosses": ["þorn-tree"]},
        {"_type": "etymon", "ref": "oe:æcer", "glosses": ["field"]},
    ]
    n = write_jsonl(p, rows)
    assert n == 2
    # Non-ASCII chars stay literal, not \uXXXX escaped.
    text = p.read_text(encoding="utf-8")
    assert "þorn" in text
    assert "æcer" in text
    # Round-trip parses back to the same rows.
    parsed = [json.loads(line) for line in text.splitlines()]
    assert parsed == rows


def test_compact_file_in_place(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"_op": "add", "_type": "etymon", "ref": "oe:cot", "glosses": ["a"]}\n'
        '{"_op": "patch", "_type": "etymon", "ref": "oe:cot", "glosses_add": ["b"]}\n'
    )
    n = compact_file(p)
    assert n == 1
    after = p.read_text().splitlines()
    parsed = json.loads(after[0])
    assert parsed == {"_type": "etymon", "ref": "oe:cot", "glosses": ["a", "b"]}


def test_write_jsonl_creates_parent_dir(tmp_path: Path):
    p = tmp_path / "nested" / "subdir" / "x.jsonl"
    write_jsonl(p, [{"_type": "etymon", "ref": "oe:cot"}])
    assert p.exists()


# ---------------------------------------------------------------------------
# Type set sanity (catches accidental table drift)
# ---------------------------------------------------------------------------


def test_keyed_and_list_types_disjoint():
    assert KEYED_TYPES.isdisjoint(LIST_TYPES)


def test_to_canonical_rows_order_is_stable():
    rows = [
        {"_type": "etymon", "ref": "oe:b", "x": 1},
        {"_type": "etymon", "ref": "oe:a", "x": 1},
        {"_type": "citation", "etymon_ref": "oe:a", "page": 2},
        {"_type": "citation", "etymon_ref": "oe:b", "page": 1},
    ]
    canonical = replay(rows).to_canonical_rows()
    # Keyed types come first (alphabetical by ref within type);
    # list types preserve insertion order.
    assert [(r["_type"], r.get("ref") or r.get("page")) for r in canonical] == [
        ("etymon", "oe:a"),
        ("etymon", "oe:b"),
        ("citation", 2),
        ("citation", 1),
    ]
