"""Unit tests for the collapse-graph helpers extracted from
``detect_deterministic_collapses`` — ``_break_collapse_cycles`` (union-find
cycle breaking) and ``_flatten_collapse_chains`` (terminal-survivor flattening).

These pin the subtle invariants directly: deterministic cycle breaking (the
first ``(ref, into)``-sorted direction survives), bidirectional-pair safety,
and chain flattening to the terminal survivor.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.collapse_detect import (
    _break_collapse_cycles,
    _flatten_collapse_chains,
)


def _edge(ref, into):
    # carry an extra field so we can assert the row is preserved, not rebuilt
    return {"ref": ref, "into": into, "variant_class": f"{ref}->{into}"}


def _pairs(rows):
    return [(r["ref"], r["into"]) for r in rows]


def test_break_keeps_all_edges_of_an_acyclic_forest():
    rows = [_edge("a", "b"), _edge("b", "c"), _edge("d", "c")]
    assert _pairs(_break_collapse_cycles(rows)) == [("a", "b"), ("b", "c"), ("d", "c")]


def test_break_drops_the_cycle_closer_of_a_bidirectional_pair():
    # wode<->wood registered both ways: only the (ref,into)-sorted-first survives.
    rows = [_edge("wood", "wode"), _edge("wode", "wood")]
    kept = _break_collapse_cycles(rows)
    assert _pairs(kept) == [("wode", "wood")]  # "wode" < "wood", processed first


def test_break_drops_one_edge_of_a_three_cycle():
    rows = [_edge("a", "b"), _edge("b", "c"), _edge("c", "a")]
    kept = _break_collapse_cycles(rows)
    # exactly one edge dropped (the one that would close the cycle)
    assert len(kept) == 2


def test_break_is_order_independent_in_input_but_deterministic_in_output():
    rows = [_edge("c", "a"), _edge("b", "c"), _edge("a", "b")]
    # same set of edges, shuffled input order -> same kept set as sorted input
    assert _pairs(_break_collapse_cycles(rows)) == _pairs(
        _break_collapse_cycles(list(reversed(rows)))
    )


def test_flatten_repoints_a_chain_at_its_terminal_survivor():
    # wella -> well -> wiell: both non-terminal nodes re-point at wiell.
    kept = [_edge("wella", "well"), _edge("well", "wiell")]
    flat = _flatten_collapse_chains(kept)
    assert {(r["ref"], r["into"]) for r in flat} == {("wella", "wiell"), ("well", "wiell")}


def test_flatten_preserves_extra_row_fields():
    kept = [_edge("a", "b"), _edge("b", "c")]
    flat = _flatten_collapse_chains(kept)
    by_ref = {r["ref"]: r for r in flat}
    assert by_ref["a"]["variant_class"] == "a->b"  # original field carried through
    assert by_ref["a"]["into"] == "c"  # but into re-pointed to the survivor


def test_flatten_leaves_a_single_edge_untouched():
    assert _flatten_collapse_chains([_edge("a", "b")]) == [
        {"ref": "a", "into": "b", "variant_class": "a->b"}
    ]
