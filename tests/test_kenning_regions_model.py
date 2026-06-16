"""Integrity tests for the Class-A region classification model (wyrd-3q6m.1, D49).

These lock the *shape* of ``regions_england.yaml`` — the node vocabulary,
containment skeleton, and stratum/axis typing — plus the four decisions the
user signed off on: the two granularity calls (Yorkshire Ridings, single-Sussex),
the quarantine of merged-volume strings (a fail-closed data-quality call), and
the Danelaw zone-vs-admin-node split. They read only the committed YAML; they
never touch the live authoring DB (CI has none).
"""

from collections import Counter
from importlib import resources

import yaml

_DATA_PACKAGE = "wyrd.generators.kenning.data"
_FILENAME = "regions_england.yaml"

VALID_LEVELS = {"country", "county", "subdivision"}
VALID_STRATA = {"historic", "modern-administrative"}
VALID_ALIAS_KINDS = {"code", "naming", "casing", "long-form", "normalize-up"}
# Per-kind target contract (the alias header documents these): (level, stratum)
# the `to` node must have. All kinds resolve to the historic dedup stratum;
# long-form lands on a subdivision (the Ridings), the rest on a county. No kind
# may target the country root.
_ALIAS_KIND_TARGET = {
    "code": ("county", "historic"),
    "naming": ("county", "historic"),
    "casing": ("county", "historic"),
    "long-form": ("subdivision", "historic"),
    "normalize-up": ("county", "historic"),
}
# Containment rank: a parent must sit exactly one level above its child.
_LEVEL_RANK = {"country": 0, "county": 1, "subdivision": 2}


def _load():
    text = resources.files(_DATA_PACKAGE).joinpath(_FILENAME).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict), f"{_FILENAME} did not parse as a mapping"
    return data


def test_parses_with_required_top_level_keys():
    m = _load()
    for key in (
        "country",
        "dedup_stratum",
        "strata",
        "nodes",
        "deferred_zones",
        "quarantine",
        "aliases",
    ):
        assert key in m, f"missing top-level key {key!r}"
    assert m["country"] == "England"
    assert m["dedup_stratum"] in m["strata"], "dedup_stratum must be a declared stratum"
    # type shape: the loaders downstream assume these container types.
    assert isinstance(m["strata"], dict)
    for key in ("nodes", "deferred_zones", "quarantine", "aliases"):
        assert isinstance(m[key], list), f"{key} must be a list"


def test_buckets_are_non_empty():
    """Guards against a vacuous-pass: an empty list in any of the three iterated
    buckets would make the per-entry assertions over it pass by iterating zero
    times."""
    m = _load()
    assert m["nodes"], "nodes must not be empty"
    assert m["deferred_zones"], "deferred_zones must not be empty"
    assert m["quarantine"], "quarantine must not be empty"


def test_every_node_well_formed():
    m = _load()
    for n in m["nodes"]:
        assert isinstance(n, dict), f"node must be a mapping: {n}"
        assert set(n) == {"canonical", "level", "stratum", "parent"}, f"unexpected node keys: {n}"
        assert isinstance(n["canonical"], str) and n["canonical"].strip(), (
            f"empty/non-str canonical: {n}"
        )
        assert n["parent"] is None or (isinstance(n["parent"], str) and n["parent"].strip()), (
            f"bad parent: {n}"
        )
        assert n["level"] in VALID_LEVELS, f"bad level on {n['canonical']!r}: {n['level']!r}"
        assert n["stratum"] in VALID_STRATA, f"bad stratum on {n['canonical']!r}: {n['stratum']!r}"


def test_deferred_zones_and_quarantine_shapes():
    """The two non-node buckets get the same closed-key lock as nodes, so a
    malformed entry (missing/extra field) is caught, not discovered at use-site."""
    m = _load()
    for z in m["deferred_zones"]:
        assert isinstance(z, dict), f"deferred_zone must be a mapping: {z}"
        assert set(z) == {"value", "zone", "migrate_to"}, f"unexpected deferred_zone keys: {z}"
        assert all(isinstance(z[k], str) and z[k].strip() for k in z), f"blank/non-str field: {z}"
    for q in m["quarantine"]:
        assert isinstance(q, dict), f"quarantine must be a mapping: {q}"
        assert set(q) == {"value", "reason"}, f"unexpected quarantine keys: {q}"
    # intra-bucket value uniqueness — set-based membership tests elsewhere would
    # silently dedup a value listed twice within a bucket (nodes get this via
    # test_canonical_names_unique).
    for bucket in ("deferred_zones", "quarantine"):
        values = [e["value"] for e in m[bucket]]
        dupes = {v for v in values if values.count(v) > 1}
        assert not dupes, f"duplicate {bucket} values: {dupes}"
        assert all(isinstance(q[k], str) and q[k].strip() for k in q), f"blank/non-str field: {q}"


def test_canonical_names_unique():
    m = _load()
    counts = Counter(n["canonical"] for n in m["nodes"])
    dupes = {name for name, c in counts.items() if c > 1}
    assert not dupes, f"duplicate canonical node names: {dupes}"


def test_strata_closed_against_declaration():
    """The stratum vocabulary is closed three ways: the constant matches the
    model's own ``strata:`` block, and every stratum used by a node is declared
    (no typo'd or undeclared stratum, no declared-but-unused stratum)."""
    m = _load()
    declared = set(m["strata"])
    assert declared == VALID_STRATA, f"strata: block {declared} != VALID_STRATA {VALID_STRATA}"
    used = {n["stratum"] for n in m["nodes"]}
    assert used == declared, f"node strata {used} != declared {declared}"


def test_parent_edges_resolve_and_single_root():
    m = _load()
    names = {n["canonical"] for n in m["nodes"]}
    roots = [n for n in m["nodes"] if n["parent"] is None]
    assert [r["canonical"] for r in roots] == ["England"], "exactly one root: England"
    assert roots[0]["level"] == "country"
    for n in m["nodes"]:
        if n["parent"] is not None:
            assert n["parent"] in names, f"{n['canonical']!r} has dangling parent {n['parent']!r}"


def test_containment_levels_are_sane():
    """Every non-root node's parent sits exactly one level above it: counties
    under the country root, subdivisions under counties. Catches an inverted
    edge (county under a subdivision) or a skipped level (subdivision straight
    under the country root) — either would corrupt the dedup blocking key."""
    m = _load()
    by_name = {n["canonical"]: n for n in m["nodes"]}
    for n in m["nodes"]:
        if n["parent"] is None:
            continue
        child_rank = _LEVEL_RANK[n["level"]]
        parent_rank = _LEVEL_RANK[by_name[n["parent"]]["level"]]
        assert parent_rank == child_rank - 1, (
            f"{n['canonical']!r} ({n['level']}) parents to "
            f"{n['parent']!r} ({by_name[n['parent']]['level']}) — not exactly one level up"
        )


def test_non_root_edges_stay_within_a_stratum():
    """The load-bearing containment invariant (nodes comment): county/subdivision
    edges never cross strata; only edges hanging off the country root England may.
    Catches a modern node reparented under a historic county (or vice-versa) —
    which would mix strata in one subtree and corrupt the dedup blocking key."""
    m = _load()
    by_name = {n["canonical"]: n for n in m["nodes"]}
    for n in m["nodes"]:
        if n["parent"] is None or n["parent"] == "England":
            continue
        parent_stratum = by_name[n["parent"]]["stratum"]
        assert parent_stratum == n["stratum"], (
            f"{n['canonical']!r} ({n['stratum']}) parents to {n['parent']!r} "
            f"({parent_stratum}) — non-root edge crosses strata"
        )


def test_every_node_reaches_root_without_cycles():
    """Walking parents from any node terminates at the single root England — no
    cycles (A→B→A with no root), no components disconnected from the root, and
    England is the only country-level node. Parent-resolution alone (above)
    doesn't prove a tree; this does."""
    m = _load()
    by_name = {n["canonical"]: n for n in m["nodes"]}
    country_nodes = [n["canonical"] for n in m["nodes"] if n["level"] == "country"]
    assert country_nodes == ["England"], f"exactly one country-level node: {country_nodes}"
    for n in m["nodes"]:
        seen, cur = set(), n["canonical"]
        while by_name[cur]["parent"] is not None:
            assert cur not in seen, f"cycle reaching {n['canonical']!r}"
            seen.add(cur)
            cur = by_name[cur]["parent"]
        assert cur == "England", f"{n['canonical']!r} does not reach the root"


def test_modern_administrative_stratum_is_known_but_not_a_dedup_target():
    """Modern-administrative nodes are KNOWN vocabulary but never the dedup key:
    the dedup_stratum is historic, and the lossy successor Cumbria is carried as
    a node (it can't be split into Cumberland + Westmorland without evidence)."""
    m = _load()
    by_name = {n["canonical"]: n for n in m["nodes"]}
    assert m["dedup_stratum"] == "historic"
    modern = {n["canonical"] for n in m["nodes"] if n["stratum"] == "modern-administrative"}
    assert modern, "modern-administrative vocabulary must exist"
    assert "Cumbria" in modern
    assert by_name["Cumbria"]["stratum"] == "modern-administrative"


def test_no_value_appears_in_two_buckets():
    """A region string is a node OR a deferred zone OR quarantined — never two."""
    m = _load()
    nodes = {n["canonical"] for n in m["nodes"]}
    zones = {z["value"] for z in m["deferred_zones"]}
    quar = {q["value"] for q in m["quarantine"]}
    assert not (nodes & zones), nodes & zones
    assert not (nodes & quar), nodes & quar
    assert not (zones & quar), zones & quar


def test_aliases_are_well_formed_and_valid():
    """wyrd-3q6m.2: each alias is a {from, to, kind} record; `from` is a variant
    (never itself a node / zone / quarantine value), `to` resolves to a real
    node, `kind` is known, and `from` values are unique (no value folding two
    ways)."""
    m = _load()
    nodes = {n["canonical"] for n in m["nodes"]}
    zones = {z["value"] for z in m["deferred_zones"]}
    quar = {q["value"] for q in m["quarantine"]}
    aliases = m["aliases"]
    assert isinstance(aliases, list) and aliases, "aliases must be a non-empty list"
    for a in aliases:
        assert isinstance(a, dict), f"alias must be a mapping: {a}"
        assert set(a) == {"from", "to", "kind"}, f"unexpected alias keys: {a}"
        assert isinstance(a["from"], str) and a["from"].strip(), f"bad from: {a}"
        assert a["kind"] in VALID_ALIAS_KINDS, f"bad alias kind: {a}"
        assert a["to"] in nodes, f"alias {a['from']!r} -> non-node {a['to']!r}"
        assert a["from"] not in nodes, f"alias from {a['from']!r} collides with a node"
        assert a["from"] not in zones, f"alias from {a['from']!r} collides with a zone"
        assert a["from"] not in quar, f"alias from {a['from']!r} collides with quarantine"
    # dedup check after per-alias validation, so a malformed entry fails with a
    # descriptive assertion above rather than a bare KeyError here.
    froms = [a["from"] for a in aliases]
    assert len(froms) == len(set(froms)), f"duplicate alias 'from' keys: {froms}"


def test_alias_from_set_is_complete():
    """Lock the alias vocabulary: exactly the live/known coding variants fold,
    nothing more (a stray addition or a dropped fold is caught)."""
    m = _load()
    froms = {a["from"] for a in m["aliases"]}
    expected = {
        "SUF",
        "LEC",
        "BUK",
        "SUR",
        "SUS",
        "County Durham",
        "Isle Of Wight",
        "West Riding of Yorkshire",
        "East Riding of Yorkshire",
        "East Sussex",
        "West Sussex",
    }
    assert froms == expected, froms


def test_normalize_up_aliases_target_a_historic_county():
    """The sole sanctioned cross-stratum fold: a normalize-up alias must land on
    a historic county (the dedup stratum). Lossy modern areas (Cumbria, metros)
    are NOT aliased — they stay Class-B succession."""
    m = _load()
    by_name = {n["canonical"]: n for n in m["nodes"]}
    normalize_up = [a for a in m["aliases"] if a["kind"] == "normalize-up"]
    assert {a["from"] for a in normalize_up} == {"East Sussex", "West Sussex"}
    for a in normalize_up:
        tgt = by_name[a["to"]]
        assert tgt["level"] == "county" and tgt["stratum"] == "historic", a


def test_alias_targets_match_their_kind_contract():
    """Every alias resolves to a node of the (level, stratum) its kind promises:
    all kinds land on the historic dedup stratum; long-form on a subdivision, the
    rest on a county; none on the country root. Catches e.g. a `naming` alias
    pointed at a modern-administrative node (wrong-stratum dedup key) or a
    `long-form` alias pointed at a county."""
    m = _load()
    by_name = {n["canonical"]: n for n in m["nodes"]}
    for a in m["aliases"]:
        tgt = by_name[a["to"]]
        assert (tgt["level"], tgt["stratum"]) == _ALIAS_KIND_TARGET[a["kind"]], (
            f"alias {a['from']!r} (kind={a['kind']}) -> {a['to']!r} "
            f"({tgt['level']}, {tgt['stratum']}); expected {_ALIAS_KIND_TARGET[a['kind']]}"
        )


def test_yorkshire_riding_granularity_decision():
    """Decision 1: Yorkshire is a county parent of exactly the three historic
    Ridings (the dedup leaves). Locks the Riding-level granularity call."""
    m = _load()
    by_name = {n["canonical"]: n for n in m["nodes"]}
    assert by_name["Yorkshire"]["level"] == "county"
    ridings = {
        n["canonical"]
        for n in m["nodes"]
        if n["parent"] == "Yorkshire" and n["level"] == "subdivision"
    }
    assert ridings == {"North Riding", "East Riding", "West Riding"}, ridings


def test_london_boroughs_is_the_sole_subdivision_of_greater_london():
    """'London Boroughs' is the collective of boroughs WITHIN Greater London,
    modeled as its sole subdivision child — not a county-level peer of it."""
    m = _load()
    by_name = {n["canonical"]: n for n in m["nodes"]}
    assert by_name["London Boroughs"]["level"] == "subdivision"
    assert by_name["London Boroughs"]["parent"] == "Greater London"
    gl_subs = {
        n["canonical"]
        for n in m["nodes"]
        if n["parent"] == "Greater London" and n["level"] == "subdivision"
    }
    assert gl_subs == {"London Boroughs"}, gl_subs


def test_sussex_is_a_single_historic_node():
    """Decision 2: historic 'Sussex' is the single dedup node; East/West Sussex
    are NOT nodes (they normalize up via the .2 alias map)."""
    m = _load()
    names = {n["canonical"] for n in m["nodes"]}
    assert "Sussex" in names
    assert "East Sussex" not in names
    assert "West Sussex" not in names


def test_quarantine_holds_the_merged_volume_strings():
    """Decision 3: the merged-volume / merged-county strings are quarantined
    (fail-closed), not aliased to either constituent county. Every quarantine
    value is asserted present and absent from the node vocabulary, so dropping
    any one of them from the file is caught."""
    m = _load()
    quar = {q["value"] for q in m["quarantine"]}
    nodes = {n["canonical"] for n in m["nodes"]}
    expected = {
        "Northumberland and Durham",
        "Cumberland and Westmorland",
        "South-West Yorkshire",
        "The Channel Islands",
    }
    assert quar == expected, quar
    assert not (quar & nodes), f"quarantined values must not be canonical nodes: {quar & nodes}"


def test_danelaw_is_a_deferred_zone_not_an_admin_node():
    """Decision 4: cultural/linguistic zones are a separate (deferred) axis, not
    administrative nodes. 'England (Danelaw)' is a zone (→ wyrd-hytz), never a
    dedup node."""
    m = _load()
    zones = {z["value"] for z in m["deferred_zones"]}
    assert "England (Danelaw)" in zones
    assert "England (Danelaw)" not in {n["canonical"] for n in m["nodes"]}
