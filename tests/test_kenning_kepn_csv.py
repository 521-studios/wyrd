"""Tests for the KEPN county-CSV ingester (wyrd-bc5z)."""

from __future__ import annotations

from pathlib import Path

from wyrd.generators.kenning.jsonl.build import build_from_jsonl
from wyrd.generators.kenning.kepn_csv_ingester import (
    EXTERNAL_SOURCE_ID,
    SOURCE_ID,
    county_from_filename,
    emit_kepn_jsonl,
    extract_personal_name,
    fold_name,
    load_name_index,
    match_name,
    parse_derivation,
)
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.schema import init_schema

_CSV_HEADER = "PlaceName,Etymology,Derivation,ReferenceTitle\n"


def _write_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    import csv

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PlaceName", "Etymology", "Derivation", "ReferenceTitle"])
        w.writerows(rows)
    return path


def _briggs_fixture(path: Path) -> Path:
    """Minimal Briggs-shaped name index JSONL."""
    import json

    rows = [
        {"_type": "source", "ref": "briggs_2024_personal_names_index", "title": "Briggs"},
        {"_type": "etymon", "ref": "old-english:Earda", "canonical_form": "Earda",
         "language": "old-english", "tags": ["male name"], "glosses": ["a personal name"]},
        {"_type": "citation", "etymon_ref": "old-english:Earda"},
        {"_type": "etymon", "ref": "old-english:Æþelstan", "canonical_form": "Æþelstan",
         "language": "old-english", "tags": ["male name"], "glosses": ["a personal name"]},
        {"_type": "citation", "etymon_ref": "old-english:Æþelstan"},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


def _build(tmp_path: Path, *jsonl_paths: Path):
    dbp = tmp_path / "lex.db"
    init_schema(dbp)
    db = LexiconDB(dbp)
    counts = build_from_jsonl(db.conn, list(jsonl_paths))
    db.commit()
    return db, counts


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parse_rejoins_internal_semicolon_gloss():
    deriv = "stōw Old English - A place; a place of assembly; a holy place.; tūn Old English - A farmstead."
    els = parse_derivation(deriv)
    assert [(e.form, e.lang) for e in els] == [("stōw", "old-english"), ("tūn", "old-english")]
    # the internal "; " inside the first gloss is re-joined, not split into elements
    assert els[0].gloss == "A place; a place of assembly; a holy place."


def test_parse_celtic_abbrev_and_modern_english():
    els = parse_derivation("cadeir C - A chair-shaped hill; spa Modern English - A spa.")
    assert [(e.form, e.lang) for e in els] == [("cadeir", "celtic"), ("spa", "modern-english")]


def test_parse_personal_name_slot_flagged():
    els = parse_derivation("lēah Old English - A clearing; Personal name (Old English) Old English - Personal name")
    assert els[0].is_personal_name is False
    assert els[1].is_personal_name is True


# ---------------------------------------------------------------------------
# fold + match
# ---------------------------------------------------------------------------

def test_fold_equates_ascii_and_oe_spellings():
    assert fold_name("Aethelstan") == fold_name("Æþelstan") == "aethelstan"
    assert fold_name("Earda's") == "earda"


def test_match_tiers(tmp_path):
    idx = load_name_index(_briggs_fixture(tmp_path / "briggs.jsonl"))
    assert match_name("Aethelstan", idx)[1] == "t1"   # fold-exact
    assert match_name("Eorda", idx)[1] == "t2"          # edit<=1 of earda
    assert match_name("Zzzqq", idx)[1] == "miss"


def test_extract_personal_name():
    assert extract_personal_name("'Earda's wood/clearing'.") == "Earda"
    assert extract_personal_name("A spring with no owner.") is None


def test_county_from_filename():
    assert county_from_filename(Path("rutland.csv")) == "Rutland"
    assert county_from_filename(Path("north-riding-yorkshire.csv")) == "North Riding of Yorkshire"


# ---------------------------------------------------------------------------
# emit + build
# ---------------------------------------------------------------------------

def test_external_citation_only_when_referenced(tmp_path):
    csv = _write_csv(tmp_path / "shire.csv", [
        ("Reffed", "'Ash spring'.", "æsc Old English - Ash; wella Old English - Spring.", "E. Ekwall; DEPN"),
        ("Unreffed", "'Oak ford'.", "āc Old English - Oak; ford Old English - Ford.", ""),
    ])
    out = tmp_path / "kepn.jsonl"
    emit_kepn_jsonl(csv, out, county="Shire", name_index={})
    db, counts = _build(tmp_path, out)
    assert counts.get("citation_orphans", 0) == 0
    c = db.conn

    def srcs(form):
        r = c.execute(
            "SELECT GROUP_CONCAT(DISTINCT ci.source_id) FROM etymon e "
            "JOIN etymon_citation ci ON ci.etymon_id=e.id WHERE e.canonical_form=?", (form,)
        ).fetchone()
        return set((r[0] or "").split(",")) if r and r[0] else set()

    assert srcs("æsc") == {SOURCE_ID, EXTERNAL_SOURCE_ID}   # referenced place → both
    assert srcs("āc") == {SOURCE_ID}                        # unreferenced → kepn only
    db.close()


def test_personal_name_match_corroborates_briggs(tmp_path):
    briggs = _briggs_fixture(tmp_path / "briggs.jsonl")
    idx = load_name_index(briggs)
    csv = _write_csv(tmp_path / "shire.csv", [
        ("Ardeley", "'Earda's wood/clearing'.", "lēah Old English - A clearing; Personal name (Old English) Old English - Personal name", "DEPN"),
    ])
    out = tmp_path / "kepn.jsonl"
    stats = emit_kepn_jsonl(csv, out, county="Shire", name_index=idx)
    assert stats.pn_match_t1 == 1 and stats.pn_miss == 0
    db, counts = _build(tmp_path, briggs, out)
    assert counts.get("citation_orphans", 0) == 0
    r = db.conn.execute(
        "SELECT GROUP_CONCAT(DISTINCT ci.source_id) FROM etymon e "
        "JOIN etymon_citation ci ON ci.etymon_id=e.id WHERE e.canonical_form='Earda'"
    ).fetchone()
    assert "briggs_2024_personal_names_index" in r[0] and SOURCE_ID in r[0]
    # the breakdown links the name element to the Earda etymon (2 elements)
    n = db.conn.execute("SELECT COUNT(*) FROM toponym_etymology_element").fetchone()[0]
    assert n == 2
    db.close()


def test_personal_name_miss_yields_partial_breakdown_note(tmp_path):
    idx = load_name_index(_briggs_fixture(tmp_path / "briggs.jsonl"))
    csv = _write_csv(tmp_path / "shire.csv", [
        ("Grimston", "'Grimr's farmstead'.", "tūn Old English - A farmstead; Personal name (Old Norse) Old Norse - Personal name", "DEPN"),
    ])
    out = tmp_path / "kepn.jsonl"
    stats = emit_kepn_jsonl(csv, out, county="Shire", name_index=idx)
    assert stats.pn_miss == 1
    db, counts = _build(tmp_path, out)
    note = db.conn.execute("SELECT notes FROM toponym_etymology").fetchone()[0]
    assert "unresolved personal name" in note and "Grimr" in note
    # only the lexical element survives in the breakdown
    assert db.conn.execute("SELECT COUNT(*) FROM toponym_etymology_element").fetchone()[0] == 1
    db.close()


def test_whitelist_quarantines_unknown_form(tmp_path):
    csv = _write_csv(tmp_path / "shire.csv", [
        ("Junkton", "'Junk'.", "tūn Old English - A farmstead; xyzzy Old English - Nonsense.", ""),
    ])
    out = tmp_path / "kepn.jsonl"
    stats = emit_kepn_jsonl(csv, out, county="Shire", name_index={}, element_whitelist=frozenset({"tūn"}))
    assert stats.whitelist_quarantined == 1
    assert stats.lexical_elements == 1  # only tūn admitted
    db, counts = _build(tmp_path, out)
    assert db.conn.execute("SELECT COUNT(*) FROM etymon WHERE canonical_form='xyzzy'").fetchone()[0] == 0
    db.close()
