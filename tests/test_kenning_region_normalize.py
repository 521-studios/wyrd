"""Tests for the one-time L2 vocab normalization logic (wyrd-3q6m.4).

Pure / fixture-based: build_riding_map reads a tmp CSV; normalize_rows operates
on in-memory row lists. No live DB / bundle.
"""

from __future__ import annotations

from pathlib import Path

from wyrd.generators.kenning.lexicon.region_normalize import (
    NormReport,
    build_riding_map,
    normalize_rows,
)


def _topo(name, region, country="England"):
    return {
        "_type": "toponym",
        "ref": f"{name}@{region}",
        "modern_name": name,
        "country": country,
        "region": region,
    }


def _etym(name, region):
    return {"_type": "etymology_element", "toponym_ref": f"{name}@{region}", "elements": []}


# --- build_riding_map --------------------------------------------------------


def test_build_riding_map_parses_single_riding(tmp_path: Path):
    csv = tmp_path / "yorkshire.csv"
    csv.write_text(
        "PlaceName,Etymology,Derivation,ReferenceTitle\n"
        'Aberford,x,y,"Smith; Place-names of the West Riding of Yorkshire; Ekwall"\n'
        'Pickering,x,y,"Smith; Place-names of the North Riding of Yorkshire"\n'
        'Nowhere,x,y,"Ekwall; Dictionary of English Place-Names"\n',  # no Riding
        encoding="utf-8",
    )
    m = build_riding_map(csv)
    assert m == {"Aberford": "West Riding", "Pickering": "North Riding"}  # Nowhere omitted


def test_build_riding_map_omits_multi_riding(tmp_path: Path):
    csv = tmp_path / "yorkshire.csv"
    csv.write_text(
        "PlaceName,Etymology,Derivation,ReferenceTitle\n"
        'Ambiguous,x,y,"West Riding of Yorkshire; North Riding of Yorkshire"\n'
        # same name across TWO rows with different Ridings → still omitted
        # (accumulate-across-rows, not last-write-wins).
        'Aldbrough,x,y,"East Riding of Yorkshire"\n'
        'Aldbrough,x,y,"North Riding of Yorkshire"\n',
        encoding="utf-8",
    )
    assert build_riding_map(csv) == {}


# --- normalize_rows: region --------------------------------------------------


def test_region_alias_fold_updates_field_and_ref():
    rows = [_topo("Ipswich", "SUF")]
    normalize_rows(rows, report=(rep := NormReport()))
    assert rows[0]["region"] == "Suffolk"
    assert rows[0]["ref"] == "Ipswich@Suffolk"
    assert rep.region_folded == 1


def test_yorkshire_recode_updates_toponym_and_etymology_ref():
    rows = [_topo("Aberford", "Yorkshire"), _etym("Aberford", "Yorkshire")]
    normalize_rows(rows, riding_map={"Aberford": "West Riding"}, report=(rep := NormReport()))
    assert rows[0]["region"] == "West Riding"
    assert rows[0]["ref"] == "Aberford@West Riding"
    assert rows[1]["toponym_ref"] == "Aberford@West Riding"  # cross-ref follows the rename
    assert rep.yorkshire_recoded == 1
    assert rep.refs_rewritten == 2


def test_yorkshire_without_riding_mapping_stays_yorkshire():
    rows = [_topo("Mystery", "Yorkshire")]
    normalize_rows(rows, riding_map={}, report=(rep := NormReport()))
    assert rows[0]["region"] == "Yorkshire"  # valid county node, left as-is
    assert rep.yorkshire_left_county == 1  # counted for observability


def test_region_fold_rewrites_etymology_ref():
    """The common case: a plain region fold (not Yorkshire) must also follow the
    rename on the etymology cross-ref."""
    rows = [_topo("Ipswich", "SUF"), _etym("Ipswich", "SUF")]
    normalize_rows(rows, report=NormReport())
    assert rows[1]["toponym_ref"] == "Ipswich@Suffolk"


def test_non_england_null_country_derived_from_region():
    """A null country on a non-England region is filled from the region→country
    map (Argyll→Scotland) — the region itself stays as-is (deferred to .5)."""
    rows = [_topo("Oban", "Argyll", country=None)]
    normalize_rows(rows, report=NormReport())
    assert rows[0]["region"] == "Argyll"
    assert rows[0]["country"] == "Scotland"


def test_unfoldable_region_left_unchanged_and_reported():
    rows = [_topo("Somewhere", "Northumberland and Durham")]
    normalize_rows(rows, report=(rep := NormReport()))
    assert rows[0]["region"] == "Northumberland and Durham"
    assert rep.unfoldable == {"Northumberland and Durham": 1}


def test_non_england_region_passes_through():
    rows = [_topo("Oban", "Argyll", country="Scotland")]
    normalize_rows(rows, report=NormReport())
    assert rows[0]["region"] == "Argyll"  # deferred to wyrd-3q6m.5


# --- normalize_rows: country -------------------------------------------------


def test_country_aliases():
    rows = [
        _topo("Dublin", "Leinster", country="Republic of Ireland"),
        _topo("Douglas", "Man", country="The Isle of Man"),
        _topo("Brest", "Finistere", country="Brittany"),
    ]
    normalize_rows(rows, report=(rep := NormReport()))
    assert [r["country"] for r in rows] == ["Ireland", "Isle of Man", "France"]
    assert rep.country_aliased == 3


def test_null_country_derived_from_canonical_region():
    rows = [_topo("Ipswich", "SUF", country=None)]  # SUF→Suffolk→England
    normalize_rows(rows, report=(rep := NormReport()))
    assert rows[0]["country"] == "England"
    assert rep.country_derived == 1


# --- robustness + idempotency ------------------------------------------------


def test_event_row_without_modern_name_does_not_crash():
    rows = [
        _topo("Aberford", "Yorkshire"),
        {"_type": "toponym", "ref": "Aberford@Yorkshire", "op": "remove"},  # event row
    ]
    normalize_rows(rows, riding_map={"Aberford": "West Riding"}, report=NormReport())
    assert rows[1]["ref"] == "Aberford@West Riding"  # remove-event ref follows too


def test_idempotent_second_run_is_a_noop():
    rows = [_topo("Ipswich", "SUF"), _topo("Aberford", "Yorkshire"), _etym("Aberford", "Yorkshire")]
    normalize_rows(rows, riding_map={"Aberford": "West Riding"}, report=NormReport())
    snapshot = [dict(r) for r in rows]
    normalize_rows(rows, riding_map={"Aberford": "West Riding"}, report=(rep := NormReport()))
    assert rows == snapshot
    assert rep.region_folded == 0 and rep.yorkshire_recoded == 0 and rep.refs_rewritten == 0
