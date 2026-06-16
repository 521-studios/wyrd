"""Integrity tests for the Class-A region classification model (wyrd-3q6m.1, D49).

These lock the *shape* of ``regions_england.yaml`` — the node vocabulary,
containment skeleton, and stratum/axis typing — plus the three granularity
decisions the user signed off on (Yorkshire Ridings, single-Sussex, the
quarantine set). They read only the committed YAML; they never touch the live
authoring DB (CI has none).
"""
from importlib import resources

import yaml

_DATA_PACKAGE = "wyrd.generators.kenning.data"
_FILENAME = "regions_england.yaml"

VALID_LEVELS = {"country", "county", "subdivision"}
VALID_STRATA = {"historic", "modern-administrative"}


def _load():
    text = resources.files(_DATA_PACKAGE).joinpath(_FILENAME).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_parses_with_required_top_level_keys():
    m = _load()
    for key in ("country", "dedup_stratum", "strata", "nodes", "deferred_zones", "quarantine", "aliases"):
        assert key in m, f"missing top-level key {key!r}"
    assert m["country"] == "England"
    assert m["dedup_stratum"] in m["strata"], "dedup_stratum must be a declared stratum"


def test_every_node_well_formed():
    m = _load()
    for n in m["nodes"]:
        assert set(n) == {"canonical", "level", "stratum", "parent"}, f"unexpected node keys: {n}"
        assert n["level"] in VALID_LEVELS, f"bad level on {n['canonical']!r}: {n['level']!r}"
        assert n["stratum"] in VALID_STRATA, f"bad stratum on {n['canonical']!r}: {n['stratum']!r}"


def test_canonical_names_unique():
    m = _load()
    names = [n["canonical"] for n in m["nodes"]]
    dupes = {x for x in names if names.count(x) > 1}
    assert not dupes, f"duplicate canonical node names: {dupes}"


def test_parent_edges_resolve_and_single_root():
    m = _load()
    names = {n["canonical"] for n in m["nodes"]}
    roots = [n for n in m["nodes"] if n["parent"] is None]
    assert [r["canonical"] for r in roots] == ["England"], "exactly one root: England"
    assert roots[0]["level"] == "country"
    for n in m["nodes"]:
        if n["parent"] is not None:
            assert n["parent"] in names, f"{n['canonical']!r} has dangling parent {n['parent']!r}"


def test_no_value_appears_in_two_buckets():
    """A region string is a node OR a deferred zone OR quarantined — never two."""
    m = _load()
    nodes = {n["canonical"] for n in m["nodes"]}
    zones = {z["value"] for z in m["deferred_zones"]}
    quar = {q["value"] for q in m["quarantine"]}
    assert not (nodes & zones), nodes & zones
    assert not (nodes & quar), nodes & quar
    assert not (zones & quar), zones & quar


def test_yorkshire_riding_granularity_decision():
    """Decision 1: Yorkshire is a county parent of exactly the three historic
    Ridings (the dedup leaves). Locks the Riding-level granularity call."""
    m = _load()
    by_name = {n["canonical"]: n for n in m["nodes"]}
    assert by_name["Yorkshire"]["level"] == "county"
    ridings = {n["canonical"] for n in m["nodes"]
               if n["parent"] == "Yorkshire" and n["level"] == "subdivision"}
    assert ridings == {"North Riding", "East Riding", "West Riding"}, ridings


def test_sussex_is_a_single_historic_node():
    """Decision 2: historic 'Sussex' is the single dedup node; East/West Sussex
    are NOT nodes (they normalize up via the .2 alias map)."""
    m = _load()
    names = {n["canonical"] for n in m["nodes"]}
    assert "Sussex" in names
    assert "East Sussex" not in names
    assert "West Sussex" not in names


def test_quarantine_holds_the_merged_volume_string():
    """Decision 3: the merged-volume 'Northumberland and Durham' string is
    quarantined (fail-closed), not aliased to either county."""
    m = _load()
    quar = {q["value"] for q in m["quarantine"]}
    assert "Northumberland and Durham" in quar
    # ...and is therefore not a node masquerading as a real region.
    assert "Northumberland and Durham" not in {n["canonical"] for n in m["nodes"]}


def test_danelaw_is_a_deferred_zone_not_an_admin_node():
    m = _load()
    zones = {z["value"] for z in m["deferred_zones"]}
    assert "England (Danelaw)" in zones
    assert "England (Danelaw)" not in {n["canonical"] for n in m["nodes"]}
