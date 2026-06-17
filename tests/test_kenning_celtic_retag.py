"""Tests for the celtic→branch L2 re-tag (wyrd-xdam.2).

Fixture-only: in-memory row dicts (the L2 JSONL shape), no live DB.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.celtic_retag import (
    RetagReport,
    build_branch_map,
    retag_rows,
)


def _rows():
    return [
        # toponyms with countries spanning the branches + an unknown
        {"_type": "toponym", "ref": "Galway@Galway", "country": "Ireland"},  # goidelic
        {"_type": "toponym", "ref": "Bangor@Gwynedd", "country": "Wales"},  # brittonic
        {
            "_type": "toponym",
            "ref": "Penrith@Cumberland",
            "country": "England",
        },  # brittonic substrate
        {"_type": "toponym", "ref": "Gaul@-", "country": None},  # unknown (Gaulish)
        # etymons
        {"_type": "etymon", "ref": "celtic:gee", "language": "celtic"},  # goidelic-only
        {"_type": "etymon", "ref": "celtic:bee", "language": "celtic"},  # brittonic-only (England)
        {"_type": "etymon", "ref": "celtic:wee", "language": "celtic"},  # brittonic-only (Wales)
        {"_type": "etymon", "ref": "celtic:mix", "language": "celtic"},  # both branches → stay
        {"_type": "etymon", "ref": "celtic:unk", "language": "celtic"},  # unknown-tainted → stay
        {"_type": "etymon", "ref": "celtic:orphan", "language": "celtic"},  # unattached → stay
        # a tags record keyed by a clean celtic ref (must cascade)
        {"_type": "tags", "ref": "celtic:gee", "language": "celtic", "tags": ["water"]},
        # usage links
        {
            "_type": "etymology_element",
            "toponym_ref": "Galway@Galway",
            "elements": [{"ordinal": 0, "etymon_ref": "celtic:gee"}],
        },
        {
            "_type": "etymology_element",
            "toponym_ref": "Bangor@Gwynedd",
            "elements": [{"ordinal": 0, "etymon_ref": "celtic:wee"}],
        },
        {
            "_type": "etymology_element",
            "toponym_ref": "Penrith@Cumberland",
            "elements": [{"ordinal": 0, "etymon_ref": "celtic:bee"}],
        },
        # celtic:mix used in BOTH a goidelic and a brittonic toponym
        {
            "_type": "etymology_element",
            "toponym_ref": "Galway@Galway",
            "elements": [{"ordinal": 1, "etymon_ref": "celtic:mix"}],
        },
        {
            "_type": "etymology_element",
            "toponym_ref": "Bangor@Gwynedd",
            "elements": [{"ordinal": 1, "etymon_ref": "celtic:mix"}],
        },
        # celtic:unk used in a goidelic toponym AND an unknown one → tainted, stays celtic
        {
            "_type": "etymology_element",
            "toponym_ref": "Galway@Galway",
            "elements": [{"ordinal": 2, "etymon_ref": "celtic:unk"}],
        },
        {
            "_type": "etymology_element",
            "toponym_ref": "Gaul@-",
            "elements": [{"ordinal": 0, "etymon_ref": "celtic:unk"}],
        },
    ]


def test_build_branch_map_clean_single_branch_only():
    m = build_branch_map(_rows())
    assert m == {"celtic:gee": "goidelic", "celtic:bee": "brittonic", "celtic:wee": "brittonic"}
    # mix (both), unk (unknown-tainted), orphan (unattached) are NOT mapped → stay celtic
    for stay in ("celtic:mix", "celtic:unk", "celtic:orphan"):
        assert stay not in m


def test_england_substrate_is_brittonic():
    m = build_branch_map(_rows())
    assert m["celtic:bee"] == "brittonic"  # Penrith@Cumberland (England)


def test_unknown_country_taints_even_with_one_known_branch():
    # celtic:unk is used by a Goidelic toponym AND an unknown one — must NOT become goidelic
    m = build_branch_map(_rows())
    assert "celtic:unk" not in m


def test_retag_rows_rewrites_etymon_language_ref_and_cascades():
    rows = _rows()
    m = build_branch_map(rows)
    report = RetagReport()
    changed = retag_rows(rows, m, report)
    assert changed
    by = {(r["_type"], r.get("ref")): r for r in rows if "ref" in r}
    # the clean goidelic etymon record: language + ref flipped
    gee = by[("etymon", "goidelic:gee")]
    assert gee["language"] == "goidelic"
    assert ("etymon", "celtic:gee") not in by  # old ref gone
    # the tags record keyed by celtic:gee cascaded too
    tag = by[("tags", "goidelic:gee")]
    assert tag["language"] == "goidelic"
    # element refs cascaded
    elem_refs = {
        e["etymon_ref"] for r in rows if r["_type"] == "etymology_element" for e in r["elements"]
    }
    assert "celtic:gee" not in elem_refs and "goidelic:gee" in elem_refs
    assert "celtic:wee" not in elem_refs and "brittonic:wee" in elem_refs
    # mixed/unknown stay celtic (untouched)
    assert by[("etymon", "celtic:mix")]["language"] == "celtic"
    assert "celtic:mix" in elem_refs


def test_retag_is_idempotent():
    rows = _rows()
    m = build_branch_map(rows)
    retag_rows(rows, m, RetagReport())
    # second pass: no celtic clean refs remain, so build_branch_map is empty + no change
    m2 = build_branch_map(rows)
    assert m2 == {}
    report2 = RetagReport()
    assert retag_rows(rows, m2, report2) is False
    assert report2.records_rewritten == 0
