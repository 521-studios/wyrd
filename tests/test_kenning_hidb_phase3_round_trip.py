"""wyrd-hidb Phase 3 round-trip invariant.

Pins the end-to-end contract: starting from any DB, the pipeline

    enrich → export-meanings → JSON_pre
    dump-jsonl → wipe → rebuild → enrich → export-meanings → JSON_rebuilt

must produce ``JSON_pre == JSON_rebuilt`` byte-for-byte. This is what the
acceptance criteria for ``wyrd kenning lexicon rebuild-from-jsonl
--with-enrichment`` requires — the JSONL set is the source of truth, and
the bundle export must be a deterministic function of that L2 state plus
the (also-deterministic) L3 derivation passes.

Any pass that introduces non-determinism (iteration order leaks,
unstamped datetime, RNG without a fixed seed) breaks this test. The
test is the CI signal for that whole class of regression.

Test uses a small synthetic fixture rather than the live ``data/mining/``
JSONLs because the live data depends on bulk wiktextract slices that
aren't on CI runners. The fixture is sized to keep the test under a few
seconds while genuinely exercising each L3 pass — the apply-curation
pass via an empty ``curation_state={}`` (the wrapper still runs its
validation), derive-english-shaped via a Hebrew etymon, and
project-period-forms via a ``toponym_attestation`` row with a date_year.
Coverage assertions guard against any pass vacuously passing on no input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.enrichment import run_full_enrichment
from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in
from wyrd.generators.kenning.jsonl.dump import dump_all_sources
from wyrd.generators.kenning.lexicon import (
    LexiconDB,
    collect_canonical_decompositions,
    collect_fantasy_morphemes,
    export_meanings,
    init_schema,
)


def _populate_round_trip_fixture(db_path: Path) -> None:
    """Small but L3-exercising L2 fact set. Three independent witnesses on
    each etymon so the default 3-witness gate would pass (we lower to 2
    in the export call below as belt-and-braces, but the fixture is
    designed to be promotable on its own)."""
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        # Three sources so etymon_consensus rolls up to ≥2 witnesses per family.
        db.upsert_source(id="skeat", title="Skeat — Place-Names of Cambridgeshire", year=1901)
        db.upsert_source(id="mawer", title="Mawer — Northumberland & Durham", year=1920)
        db.upsert_source(id="ekwall", title="Ekwall — Lancashire", year=1922)

        # Etymons spanning four languages. Each pulls in a different L3 pass:
        # - cot/tun (old-english): classify-stratum native-english path,
        #   link-lemmas via the inflected 'tun' element below.
        # - *tunaz (proto-germanic): descent edge feeds classify-stratum +
        #   cluster-cognates (proto-* root walking).
        # - brae (welsh): classify-stratum welsh path.
        # - bayit (he): derive-english-shaped non-Latin-script transliteration.
        #   PHASE2A_NON_LATIN_LANGS in english_shaping.py includes "he".
        cot_id = db.upsert_etymon("cot", "old-english")
        tun_id = db.upsert_etymon("tun", "old-english")
        proto_id = db.upsert_etymon("*tunaz", "proto-germanic")
        brae_id = db.upsert_etymon("brae", "welsh")
        bayit_id = db.upsert_etymon("bayit", "he")

        # Glosses + tags for export's modifier-type/meaning signature
        # (subjects are grouped by (modifier_type, glosses, tags)).
        for gloss in ("cottage", "hut"):
            db.conn.execute(
                "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
                (cot_id, gloss),
            )
        db.conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, 'enclosure')",
            (tun_id,),
        )
        db.conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, 'hill')",
            (brae_id,),
        )
        db.conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, 'house')",
            (bayit_id,),
        )
        for tag in ("architecture", "settlement"):
            db.conn.execute(
                "INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, ?)",
                (cot_id, tag),
            )
        db.conn.execute(
            "INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, 'settlement')",
            (tun_id,),
        )
        db.conn.execute(
            "INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, 'topography')",
            (brae_id,),
        )

        # ≥2 distinct-witness citations per etymon so they survive the
        # promotion gate without relying on rando-port carve-outs. The
        # Hebrew etymon (bayit) deliberately gets two witnesses too so it
        # passes the gate and reaches the bundle output — otherwise the
        # round-trip wouldn't observe whether english_shaped survived the
        # round trip.
        for etymon_id, page_map in (
            (cot_id, {"skeat": "15", "mawer": "7", "ekwall": "201"}),
            (tun_id, {"skeat": "20", "mawer": "8", "ekwall": "202"}),
            (brae_id, {"skeat": "33", "mawer": "9"}),
            (bayit_id, {"skeat": "44", "mawer": "10"}),
        ):
            for source_id, page in page_map.items():
                db.conn.execute(
                    "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, ?, ?)",
                    (etymon_id, source_id, page),
                )

        # Descent edge so classify-stratum can assign a non-default
        # stratum to tun (proto-germanic ancestor → native-english bucket).
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence) "
            "VALUES (?, ?, 'inheritance', 'skeat', 'high')",
            (proto_id, tun_id),
        )

        # A toponym with an attributed etymology — gives the decompose
        # pass a row to scan and pick a canonical breakdown for.
        cur = db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) "
            "VALUES ('Cotton', 'England', 'Norfolk')"
        )
        topo_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, historical_form, confidence) "
            "VALUES (?, 'mawer', 'Cotuna', 'high')",
            (topo_id,),
        )
        ety_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, 1, ?)",
            (ety_id, cot_id),
        )
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id, inflection) VALUES (?, 2, ?, 'oblique')",
            (ety_id, tun_id),
        )
        # Attested historical form with a date_year so project_period_forms
        # has a row to scan (wyrd-unuo Phase 3.3 — projects period-keyed
        # surface forms from toponym_attestation rows).
        db.conn.execute(
            "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
            "VALUES (?, 'Cotuna', 1086, 'Domesday Book')",
            (topo_id,),
        )
        db.commit()


def _enrich_and_export_bundle(db_path: Path) -> tuple[str, dict[str, Any]]:
    """Run the canonical L3 chain + emit the bundle JSON the runtime would
    consume. Returns ``(bundle_json, enrichment_result)`` — the caller uses
    the result dict to assert non-vacuous coverage (otherwise passes that
    do no work would silently pass the byte-diff assertion).

    ``curation_state={}`` is passed (instead of the default None) so the
    apply-curation pass actually runs — with zero events it's a no-op, but
    the wrapper executes its validation + counting path. Without this,
    apply-curation is skipped entirely and the round-trip never observes
    whether the pass round-trips cleanly."""
    with LexiconDB(db_path) as db:
        result = run_full_enrichment(db, apply=True, curation_state={})
        # min_witnesses=2 + empty lang_thresholds bypasses the per-language
        # preset for the fixture's small witness pool. Production code uses
        # the preset (RECOMMENDED_LANG_THRESHOLDS); we're testing the
        # round-trip contract, not the witness gate calibration.
        subjects = export_meanings(
            db,
            min_witnesses=2,
            lang_thresholds={},
        )
        canonical_decompositions = collect_canonical_decompositions(db)
        fantasy_morphemes = collect_fantasy_morphemes(db)

    # Mirrors the dict-building block in cli.py's lexicon_export_meanings
    # (around cli.py:1354). Optional fields are only present when non-empty
    # — keeps empty-dict noise out. The CLI also conditionally adds a
    # ``joiners`` key sourced from --joiners-from; the test doesn't use a
    # sidecar so that branch is intentionally absent here.
    bundle: dict[str, Any] = {"subjects": subjects}
    if canonical_decompositions:
        bundle["canonical_decompositions"] = canonical_decompositions
    if fantasy_morphemes:
        bundle["fantasy_morphemes"] = fantasy_morphemes
    return json.dumps(bundle, ensure_ascii=False, indent=2), result


def _format_bundle_drift(pre: str, rebuilt: str, max_window: int = 200) -> str:
    """Build a human-readable drift report when bundles diverge."""
    if pre == rebuilt:
        return "(no drift)"
    if len(pre) != len(rebuilt):
        size_msg = f"size: pre={len(pre)}, rebuilt={len(rebuilt)}"
    else:
        size_msg = f"size match ({len(pre)} bytes) — content differs"
    # Find first divergence. strict=False because we already handled the
    # length-mismatch case above; this loop only runs to the shorter length.
    for i, (a, b) in enumerate(zip(pre, rebuilt, strict=False)):
        if a != b:
            start = max(0, i - max_window // 2)
            end = i + max_window // 2
            return (
                f"Round-trip bundle drift detected.\n"
                f"  {size_msg}\n"
                f"  first divergence at byte {i}\n"
                f"  pre[{start}:{end}]:     ...{pre[start:end]!r}...\n"
                f"  rebuilt[{start}:{end}]: ...{rebuilt[start:end]!r}..."
            )
    # One is a prefix of the other
    longer = pre if len(pre) > len(rebuilt) else rebuilt
    label = "pre" if len(pre) > len(rebuilt) else "rebuilt"
    trailing = longer[min(len(pre), len(rebuilt)) :][:max_window]
    return (
        f"Round-trip bundle drift detected (length mismatch).\n"
        f"  {size_msg}\n"
        f"  {label} has trailing bytes: ...{trailing!r}..."
    )


def test_round_trip_rebuild_enrich_export_byte_identical(tmp_path: Path) -> None:
    """The wyrd-hidb Phase 3 invariant: a DB → its dump → its rebuild → its
    export produces the same bundle bytes as the original DB → its export.
    Pins the entire L2-as-source-of-truth contract end-to-end."""
    pre_db_path = tmp_path / "pre.db"
    _populate_round_trip_fixture(pre_db_path)

    bundle_pre, pre_result = _enrich_and_export_bundle(pre_db_path)
    # Sanity: the fixture is non-degenerate; bundle has actual content.
    parsed = json.loads(bundle_pre)
    assert parsed["subjects"], (
        "fixture promoted zero subjects — witness gate misconfigured for the test"
    )

    # Coverage assertions: each L3 pass must report non-trivial work,
    # otherwise the byte-diff would still pass on two empty results and
    # the round-trip test would be lying about what it pins. The numeric
    # thresholds are loose (>= 1) — exact counts shift as the fixture or
    # passes evolve.
    assert pre_result["order"] == [
        "normalize-ocr",
        "link-lemmas",
        "apply-curation",
        "decompose",
        "cluster-cognates",
        "classify-stratum",
        "derive-english-shaped",
        "tag-phonological-vectors",
        "project-period-forms",
    ], "orchestrator skipped a pass — fixture doesn't exercise the full chain"
    assert pre_result["curation"] is not None, "apply-curation pass didn't run"
    # decompose: at least one canonical pick (otherwise the matcher couldn't
    # account for any toponym fragment and we're not exercising the pass).
    assert pre_result["decompose"]["toponyms_scanned"] >= 1
    # classify-stratum: at least one language got non-empty by_stratum.
    stratum_langs = pre_result["stratum"]["languages"]
    assert any(lang["by_stratum"] for lang in stratum_langs.values()), (
        "classify-stratum produced no by_stratum entries — fixture has no classifiable etymons"
    )
    # derive-english-shaped: candidates>=1 means a Hebrew-or-similar row
    # was scanned. The fixture seeds bayit/he so this must be >0.
    assert pre_result["english_shaped"]["candidates"] >= 1, (
        "derive-english-shaped scanned zero rows — fixture must include "
        "at least one PHASE2A_NON_LATIN_LANGS etymon"
    )
    # project-period-forms: must have observed the toponym_attestation row.
    assert pre_result["period_forms"]["rows_scanned"] >= 1, (
        "project-period-forms scanned zero rows — fixture must seed a "
        "toponym_attestation row with date_year"
    )

    # Dump JSONL from the enriched-pre DB. dump_all_sources walks the L2
    # tables (per L2_L3_BOUNDARY); L3 columns are NOT in the dump.
    dump_dir = tmp_path / "dumped"
    dump_dir.mkdir()
    with LexiconDB(pre_db_path) as db:
        dump_all_sources(db.conn, dump_dir, exclude=())

    # Wipe + rebuild from the dumped JSONL set.
    rebuilt_db_path = tmp_path / "rebuilt.db"
    init_schema(rebuilt_db_path)
    with LexiconDB(rebuilt_db_path) as db:
        build_from_jsonl(db.conn, jsonl_paths_in(dump_dir))

    bundle_rebuilt, _ = _enrich_and_export_bundle(rebuilt_db_path)

    assert bundle_pre == bundle_rebuilt, _format_bundle_drift(bundle_pre, bundle_rebuilt)


def test_round_trip_dump_directory_is_jsonl_only(tmp_path: Path) -> None:
    """Belt-and-braces: dump_all_sources produces only JSONL files
    matching the source IDs we seeded. If a future change accidentally
    emits a non-JSONL artifact (e.g. a sidecar), the rebuild step would
    fail or behave unexpectedly — pinning the shape here catches it
    before the round-trip assertion does."""
    pre_db_path = tmp_path / "pre.db"
    _populate_round_trip_fixture(pre_db_path)

    dump_dir = tmp_path / "dumped"
    dump_dir.mkdir()
    with LexiconDB(pre_db_path) as db:
        dump_all_sources(db.conn, dump_dir, exclude=())

    files = sorted(p.name for p in dump_dir.iterdir())
    # One JSONL per source in the source table. The fixture seeds no
    # manual-curation source so _curation.jsonl isn't expected here;
    # the shape check below stays language-agnostic.
    assert "skeat.jsonl" in files
    assert "mawer.jsonl" in files
    assert "ekwall.jsonl" in files
    for name in files:
        assert name.endswith(".jsonl"), f"unexpected non-JSONL file in dump: {name}"
