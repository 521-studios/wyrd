"""wyrd-alzd: dashboard attestation numbers must survive the round trip.

Companion to the Phase 3 byte-identity test in
``tests/test_kenning_hidb_phase3_round_trip.py``. That test pins that
the bundle JSON is byte-stable across dump → rebuild → re-export. THIS
test pins a strictly stronger contract: the **dashboard's** structured
view of the bundle + lexicon DB must also be stable, including
metrics that read from BOTH surfaces (corpus depth, citation depth,
inflection / variant / era-reflex / stratum / IPA coverage, and the
load-bearing bundle attestation breakdown — scholar_attested /
empirical_only / rando_only / uncited bin counts).

A round trip that preserves bundle bytes but loses lexicon-side
provenance (e.g. drops citations or descent edges) would silently
drift the dashboard while the Phase 3 byte test still passes. wyrd-hidb's
acceptance criterion names that exact case ("dashboard's
bundle-attestation numbers survive the round trip") and this test is
the CI signal for it.

The dashboard's ``generated_at`` field is a non-determinism source
(``datetime.now(UTC)`` at compute time); it's stripped before
comparison. Everything else — every per-language scorecard's numeric
counts, rates, and per-tag breakdowns — is expected to match
bit-for-bit.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.enrichment import run_full_enrichment
from wyrd.generators.kenning.jsonl.build import (
    build_from_jsonl,
    collect_curation_overrides,
    jsonl_paths_in,
)
from wyrd.generators.kenning.jsonl.dump import dump_all_sources
from wyrd.generators.kenning.language_quality import (
    LanguageQualityReport,
    compute_report,
)
from wyrd.generators.kenning.lexicon import (
    LexiconDB,
    collect_canonical_decompositions,
    collect_fantasy_morphemes,
    export_meanings,
    init_schema,
)

# Languages in the fixture that DEFAULT_LANGUAGES recognizes — restrict
# compute_report to these to avoid bloating the report with every
# language we never seed.
_FIXTURE_LANGUAGES = ["old-english", "welsh"]


def _populate_fixture(db_path: Path) -> None:
    """Same shape as the Phase 3 round-trip fixture — minimum content
    needed to produce a non-empty dashboard for old-english + welsh
    AND exercise each L3 pass in the canonical orchestrator order
    (normalize-ocr → link-lemmas → apply-curation → decompose →
    cluster-cognates → classify-stratum → derive-english-shaped →
    project-period-forms).

    Also seeds a rando-port-only etymon (``ham``) so the bundle's
    attestation breakdown has a non-zero ``uncited`` bin: the export
    filters ``rando-port`` from the per-word citation list (per
    ``_NON_SCHOLAR_SOURCES`` in lexicon.py), so a rando-only family
    ships with an empty citation list and the classifier bins it as
    ``uncited`` (the ``rando_only`` bin only fires on legacy bundles
    that didn't apply that filter). Without this etymon the four-bin
    equality assertion would pass vacuously on three of the four bins
    — only ``scholar_attested`` would have a non-zero value to
    compare."""
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.upsert_source(id="skeat", title="Skeat — Cambridgeshire", year=1901)
        db.upsert_source(id="mawer", title="Mawer — N&D", year=1920)
        db.upsert_source(id="ekwall", title="Ekwall — Lancashire", year=1922)
        # rando-port is the legacy hand-curated seed source. Cited
        # etymons admitted via this source land in the bundle as
        # ``rando_only`` (no scholar witnesses) — needed below so the
        # attestation breakdown test has a non-zero count to compare.
        db.upsert_source(id="rando-port", title="Rando-port legacy seed", year=2020)

        cot_id = db.upsert_etymon("cot", "old-english")
        tun_id = db.upsert_etymon("tun", "old-english")
        proto_id = db.upsert_etymon("*tunaz", "proto-germanic")
        brae_id = db.upsert_etymon("brae", "welsh")
        # rando-only etymon: cited solely by 'rando-port', no scholar
        # witnesses → admitted via include_rando=True, then the bundle
        # export drops 'rando-port' from the per-word citation list
        # (lexicon.py:_NON_SCHOLAR_SOURCES). The dashboard's
        # _bundle_attestation_breakdown then bins this as ``uncited``
        # (empty citation list, language sibling present) — not
        # ``rando_only`` (which only fires on legacy bundles without
        # the rando-port filter).
        ham_id = db.upsert_etymon("ham", "old-english")

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
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, 'homestead')",
            (ham_id,),
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
        for etymon_id, page_map in (
            (cot_id, {"skeat": "15", "mawer": "7", "ekwall": "201"}),
            (tun_id, {"skeat": "20", "mawer": "8", "ekwall": "202"}),
            (brae_id, {"skeat": "33", "mawer": "9"}),
            # ham_id intentionally gets ONLY a rando-port citation so the
            # bundle attestation breakdown's ``uncited`` bin is non-zero
            # in both pre and rebuilt reports. The bundle export filters
            # rando-port out of citation lists (lexicon.py:
            # _NON_SCHOLAR_SOURCES), so ham ships with no citations and
            # bins as ``uncited`` — without this etymon the four-bin
            # equality would pass vacuously on three bins.
            (ham_id, {"rando-port": "1"}),
        ):
            for source_id, page in page_map.items():
                db.conn.execute(
                    "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, ?, ?)",
                    (etymon_id, source_id, page),
                )
        # proto-germanic ancestor of tun → classify-stratum and
        # cluster-cognates have edges to walk.
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence) "
            "VALUES (?, ?, 'inheritance', 'skeat', 'high')",
            (proto_id, tun_id),
        )
        # Toponym + attribution + attestation: decompose has a row to
        # scan, project-period-forms has an attestation to project.
        cur = db.conn.execute(
            "INSERT INTO toponym (modern_name, country) VALUES ('Cotton', 'England')"
        )
        topo_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
            "VALUES (?, 'mawer', 'high')",
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
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, 2, ?)",
            (ety_id, tun_id),
        )
        db.conn.execute(
            "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
            "VALUES (?, 'Cotuna', 1086, 'Domesday Book')",
            (topo_id,),
        )
        db.commit()


def _enrich_and_export_bundle(db_path: Path, jsonl_dir: Path) -> dict[str, Any]:
    """Run the canonical L3 chain on ``db_path`` and return the bundle
    as a parsed dict (the format ``language_quality.compute_report``
    consumes — dict-shape per wyrd-c1vq). ``jsonl_dir`` feeds
    ``collect_curation_overrides`` so curation_state matches what the
    CLI rebuild path uses."""
    paths = jsonl_paths_in(jsonl_dir)
    # ``or None`` mirrors the CLI's ``lexicon rebuild-from-jsonl`` path
    # (cli.py:lexicon_rebuild_from_jsonl). Note the silent semantic:
    # an empty curation dict coerces to None, which makes
    # run_full_enrichment skip apply_curation_overrides entirely (vs
    # running it as a no-op). For the round-trip dashboard test this
    # doesn't matter — both pre and rebuilt go through the same
    # coercion, so the dashboard's scorecards stay symmetric. The
    # alternative (always pass {} so apply-curation runs as no-op)
    # would be a stronger test of pass-exercise but would diverge
    # from what the production CLI actually does.
    curation_state = collect_curation_overrides(paths) or None
    with LexiconDB(db_path) as db:
        run_full_enrichment(db, apply=True, curation_state=curation_state)
        subjects = export_meanings(db, min_witnesses=2, lang_thresholds={})
        canonical_decompositions = collect_canonical_decompositions(db)
        fantasy_morphemes = collect_fantasy_morphemes(db)
    bundle: dict[str, Any] = {"subjects": subjects}
    if canonical_decompositions:
        bundle["canonical_decompositions"] = canonical_decompositions
    if fantasy_morphemes:
        bundle["fantasy_morphemes"] = fantasy_morphemes
    return bundle


def _compute_dashboard(db_path: Path, bundle: dict[str, Any]) -> LanguageQualityReport:
    """Open ``db_path`` and run ``compute_report`` against its
    connection. Restricts to the fixture's languages so we don't waste
    cycles on every DEFAULT_LANGUAGE entry the report would otherwise
    touch."""
    with LexiconDB(db_path) as db:
        return compute_report(
            db.conn,
            bundle,
            languages=_FIXTURE_LANGUAGES,
            compute_tier4=True,  # synthetic fixture is small; full path is fast
        )


def _report_to_comparable(report: LanguageQualityReport) -> dict[str, Any]:
    """Convert the report to a dict suitable for byte-stable
    comparison: strip the ``generated_at`` field (datetime.now is
    non-deterministic by construction). Everything else — scorecards
    + reference tags + bundle_total_words + slice-filter labels — is
    expected to be a deterministic function of (DB, bundle)."""
    blob = asdict(report)
    blob.pop("generated_at", None)
    return blob


def _format_dashboard_drift(
    pre: dict[str, Any], rebuilt: dict[str, Any], max_diffs: int = 10
) -> str:
    """Build a human-readable summary of WHERE two reports diverge.
    pytest's default assertion message is OK for flat dicts; the
    LanguageQualityReport has nested scorecards keyed positionally by
    language order, so per-language drill-down beats the raw diff."""
    if pre == rebuilt:
        return "(no drift)"

    diffs: list[str] = []

    # Compare the top-level scalar fields first. sorted() on the key
    # union keeps the failure report deterministic across runs — set
    # iteration order otherwise depends on hash randomization, which
    # would shuffle diagnostic output between runs.
    for key in sorted(pre.keys() | rebuilt.keys()):
        if key == "languages":
            continue
        if pre.get(key) != rebuilt.get(key):
            diffs.append(f"  top-level {key}: pre={pre.get(key)!r}, rebuilt={rebuilt.get(key)!r}")

    # Pair languages by their ``language`` field. Currently a LanguageQualityReport
    # has exactly one scorecard per language (per compute_report's loop in
    # language_quality). The .get-from-list-comprehension below silently
    # collapses duplicates if that invariant ever breaks — guard against
    # the silent data loss with an explicit check so a future schema
    # change fails loudly here instead of producing a misleading drift
    # report.
    def _index_by_language(blob: dict[str, Any], side: str) -> dict[str, dict[str, Any]]:
        cards = blob.get("languages", [])
        indexed: dict[str, dict[str, Any]] = {}
        for card in cards:
            lang = card["language"]
            if lang in indexed:
                raise ValueError(
                    f"{side} report has duplicate scorecard for language {lang!r}; "
                    f"LanguageQualityReport schema assumes one card per language"
                )
            indexed[lang] = card
        return indexed

    pre_langs = _index_by_language(pre, "pre")
    rebuilt_langs = _index_by_language(rebuilt, "rebuilt")
    for lang in sorted(pre_langs.keys() | rebuilt_langs.keys()):
        pre_card = pre_langs.get(lang)
        rebuilt_card = rebuilt_langs.get(lang)
        # pre_card is None → the language only appears in the REBUILT
        # report → it was added during the round-trip.
        if pre_card is None:
            diffs.append(f"  language {lang!r}: added in rebuilt report")
            continue
        # rebuilt_card is None → the language only appears in the PRE
        # report → it was removed during the round-trip.
        if rebuilt_card is None:
            diffs.append(f"  language {lang!r}: removed from rebuilt report")
            continue
        for field_name in sorted(pre_card.keys() | rebuilt_card.keys()):
            pre_val = pre_card.get(field_name)
            rebuilt_val = rebuilt_card.get(field_name)
            if pre_val == rebuilt_val:
                continue
            # For dict-typed scorecard fields (lexicon_tag_coverage,
            # bundle_tag_coverage, etc. — `dict[str, int]` per the
            # LanguageScorecard dataclass) repr-ing the entire dict
            # on a single drifting key is noisy. Surface the
            # symmetric difference at the key level instead so the
            # failure message points at the actual diverging tags /
            # categories, not the whole map.
            if isinstance(pre_val, dict) and isinstance(rebuilt_val, dict):
                for sub_key in sorted(pre_val.keys() | rebuilt_val.keys()):
                    if pre_val.get(sub_key) != rebuilt_val.get(sub_key):
                        diffs.append(
                            f"  language {lang!r}: field {field_name!r}[{sub_key!r}] "
                            f"pre={pre_val.get(sub_key)!r}, "
                            f"rebuilt={rebuilt_val.get(sub_key)!r}"
                        )
            else:
                diffs.append(
                    f"  language {lang!r}: field {field_name!r} "
                    f"pre={pre_val!r}, rebuilt={rebuilt_val!r}"
                )

    head = ["Dashboard drift detected — round-trip changed scorecard fields."]
    body = diffs[:max_diffs]
    if len(diffs) > max_diffs:
        body.append(f"  ... and {len(diffs) - max_diffs} more")
    return "\n".join(head + body)


def _setup_pre_state(tmp_path: Path) -> tuple[Path, dict[str, Any], Path]:
    """Build the fixture DB, dump JSONL, and emit the pre-state bundle.
    Returns ``(pre_db_path, pre_bundle, jsonl_dir)`` — the inputs the
    round-trip test compares against."""
    pre_db = tmp_path / "pre.db"
    _populate_fixture(pre_db)

    jsonl_dir = tmp_path / "dumped"
    jsonl_dir.mkdir()
    with LexiconDB(pre_db) as db:
        dump_all_sources(db.conn, jsonl_dir, exclude=())

    pre_bundle = _enrich_and_export_bundle(pre_db, jsonl_dir)
    return pre_db, pre_bundle, jsonl_dir


def _rebuild_from_jsonl(target_db: Path, jsonl_dir: Path) -> None:
    """Init a fresh schema at ``target_db`` and replay the JSONL set.
    L2 only — callers must follow with ``_enrich_and_export_bundle``
    to run the L3 chain. (Split this way so a future negative test
    can mutate the rebuilt L2 state between rebuild and enrich.)"""
    init_schema(target_db)
    with LexiconDB(target_db) as db:
        build_from_jsonl(db.conn, jsonl_paths_in(jsonl_dir))


def test_dashboard_scorecards_byte_match_across_round_trip(tmp_path: Path) -> None:
    """The wyrd-alzd invariant: compute_report against the pre-DB +
    pre-bundle produces the same scorecards (modulo ``generated_at``)
    as compute_report against the rebuilt-DB + rebuilt-bundle.

    A failure means some metric is sensitive to a state the round-trip
    doesn't preserve — typically a citation, descent edge, or stratum
    that was lost in the dump/rebuild cycle."""
    pre_db, pre_bundle, jsonl_dir = _setup_pre_state(tmp_path)
    pre_report = _compute_dashboard(pre_db, pre_bundle)

    # Sanity: the fixture promoted something so the scorecards are
    # non-degenerate. If this fails the test below isn't actually
    # checking anything meaningful.
    assert pre_report.languages, "fixture produced an empty dashboard — check witness gate"
    assert any(card.bundle_attestation_total > 0 for card in pre_report.languages), (
        "fixture's bundle_attestation_total is zero everywhere — the named "
        "wyrd-hidb acceptance criterion can't be exercised on this fixture"
    )

    rebuilt_db = tmp_path / "rebuilt.db"
    _rebuild_from_jsonl(rebuilt_db, jsonl_dir)
    rebuilt_bundle = _enrich_and_export_bundle(rebuilt_db, jsonl_dir)
    rebuilt_report = _compute_dashboard(rebuilt_db, rebuilt_bundle)

    pre_dict = _report_to_comparable(pre_report)
    rebuilt_dict = _report_to_comparable(rebuilt_report)
    assert pre_dict == rebuilt_dict, _format_dashboard_drift(pre_dict, rebuilt_dict)


def test_dashboard_attestation_breakdown_specifically_byte_matches(tmp_path: Path) -> None:
    """Narrower version of the round-trip test focused on the four bin
    counts wyrd-hidb's acceptance criterion names by name —
    ``scholar_attested`` / ``empirical_only`` / ``rando_only`` /
    ``uncited`` — plus the ``bundle_attestation_total`` denominator
    they're rates of. Pins these even if a future change loosens the
    broader scorecard equality above (e.g. adds a new
    non-deterministic field)."""
    pre_db, pre_bundle, jsonl_dir = _setup_pre_state(tmp_path)
    pre_report = _compute_dashboard(pre_db, pre_bundle)

    # Sanity: the fixture's bundle attestation is non-degenerate AND
    # at least two of the four named bins (scholar_attested, uncited)
    # are non-zero — otherwise the bin-equality assertion would pass
    # vacuously on the all-zero bins. _populate_fixture seeds a
    # rando-port-only etymon specifically so the uncited bin fires.
    pre_totals = sum(card.bundle_attestation_total for card in pre_report.languages)
    pre_scholar = sum(card.bundle_scholar_attested for card in pre_report.languages)
    pre_uncited = sum(card.bundle_uncited for card in pre_report.languages)
    assert pre_totals > 0, "fixture's attestation total is zero — bin equality is vacuous"
    assert pre_scholar > 0, "fixture has no scholar_attested subjects"
    assert pre_uncited > 0, (
        "fixture has no uncited subjects — the rando-port-only ham etymon should "
        "have produced one; check _NON_SCHOLAR_SOURCES filter still applies"
    )

    rebuilt_db = tmp_path / "rebuilt.db"
    _rebuild_from_jsonl(rebuilt_db, jsonl_dir)
    rebuilt_bundle = _enrich_and_export_bundle(rebuilt_db, jsonl_dir)
    rebuilt_report = _compute_dashboard(rebuilt_db, rebuilt_bundle)

    pre_bins = {
        card.language: (
            card.bundle_attestation_total,
            card.bundle_scholar_attested,
            card.bundle_empirical_only,
            card.bundle_rando_only,
            card.bundle_uncited,
        )
        for card in pre_report.languages
    }
    rebuilt_bins = {
        card.language: (
            card.bundle_attestation_total,
            card.bundle_scholar_attested,
            card.bundle_empirical_only,
            card.bundle_rando_only,
            card.bundle_uncited,
        )
        for card in rebuilt_report.languages
    }
    assert pre_bins == rebuilt_bins, (
        f"bundle_attestation_breakdown drift detected.\n"
        f"  pre:     {pre_bins}\n"
        f"  rebuilt: {rebuilt_bins}"
    )


def test_dashboard_drift_detector_catches_dropped_citations(tmp_path: Path) -> None:
    """Negative test for the round-trip invariant: if the rebuilt DB is
    deliberately mutated to drop citations, the round-trip assertion
    must fail loudly. Otherwise the test above might be falsely passing
    because compute_report's metrics aren't actually sensitive to the
    state being preserved.

    The drift-detector helper formats the diff so the failure message
    names which language + which field changed — pin that the formatter
    actually surfaces the citation-related fields (not just any field)."""
    pre_db, pre_bundle, jsonl_dir = _setup_pre_state(tmp_path)
    pre_report = _compute_dashboard(pre_db, pre_bundle)

    rebuilt_db = tmp_path / "rebuilt.db"
    _rebuild_from_jsonl(rebuilt_db, jsonl_dir)
    # Deliberately drop every citation post-rebuild so the citation-depth
    # + bundle attestation breakdown will differ from pre-DB.
    with LexiconDB(rebuilt_db) as db:
        db.conn.execute("DELETE FROM etymon_citation")
        db.commit()
    rebuilt_bundle = _enrich_and_export_bundle(rebuilt_db, jsonl_dir)
    rebuilt_report = _compute_dashboard(rebuilt_db, rebuilt_bundle)

    pre_dict = _report_to_comparable(pre_report)
    rebuilt_dict = _report_to_comparable(rebuilt_report)
    assert pre_dict != rebuilt_dict, "dropping citations should produce dashboard drift"

    # The diff formatter must surface the change. The exact field that
    # diverges depends on which metrics read citation_count;
    # ``avg_citations`` sorts to the top of the alphabetical field
    # list ('a' prefix) so the default ``max_diffs`` cap can't push
    # it out of the rendered summary — pin that field specifically.
    # Using max_diffs=None (effectively uncapped) when calling the
    # formatter would also work, but pinning a fixed-position field
    # exercises the default rendering operators actually see.
    formatted = _format_dashboard_drift(pre_dict, rebuilt_dict)
    assert "Dashboard drift detected" in formatted
    assert "avg_citations" in formatted, (
        "drift formatter should name avg_citations on a dropped-citations scenario"
    )


def test_report_to_comparable_strips_generated_at(tmp_path: Path) -> None:
    """Belt-and-braces: ``generated_at`` IS the only non-deterministic
    field. If a future change adds another datetime-stamped field, the
    round-trip test above starts failing flakily. This unit-level test
    pins the helper's behavior so the failure mode points at the
    helper rather than at the round-trip itself."""
    pre_db, pre_bundle, jsonl_dir = _setup_pre_state(tmp_path)
    report = _compute_dashboard(pre_db, pre_bundle)
    blob = _report_to_comparable(report)
    assert "generated_at" not in blob
    # And the inverse: every field listed in the dataclass except
    # generated_at survives.
    full_blob = asdict(report)
    expected_keys = set(full_blob.keys()) - {"generated_at"}
    assert set(blob.keys()) == expected_keys


def test_format_dashboard_drift_no_drift_returns_marker() -> None:
    """Early-return path: two identical dicts produce '(no drift)'.
    Load-bearing for the test-pass case — if the formatter ever
    started reporting something else for matching reports, the
    integration tests' assertion messages would degrade."""
    same = {"schema_version": "1.0", "languages": []}
    assert _format_dashboard_drift(same, same) == "(no drift)"


def test_format_dashboard_drift_reports_top_level_scalar_change() -> None:
    """Top-level scalar field changes (not 'languages') are surfaced
    in the rendered summary. Pinned because the integration-level
    negative test only exercises per-language field drift; this
    branch would otherwise go untested."""
    pre = {"schema_version": "1.0", "bundle_total_words": 100, "languages": []}
    rebuilt = {"schema_version": "1.0", "bundle_total_words": 150, "languages": []}
    rendered = _format_dashboard_drift(pre, rebuilt)
    assert "top-level bundle_total_words" in rendered
    assert "pre=100" in rendered
    assert "rebuilt=150" in rendered


def test_format_dashboard_drift_truncates_at_max_diffs() -> None:
    """When the diff list exceeds ``max_diffs``, the renderer caps
    output with '... and N more' so the failure message doesn't dump
    every per-field drift on a wholesale rewrite. Pin the truncation
    so a future change to the cap doesn't silently bloat the report."""
    # 20 languages each contributing a single field-level diff →
    # max_diffs=5 should keep the first 5 and append the cap line.
    pre = {
        "schema_version": "1.0",
        "languages": [
            {"language": f"lang_{i:02d}", "total_etymons": i, "bundle_attestation_total": 0}
            for i in range(20)
        ],
    }
    rebuilt = {
        "schema_version": "1.0",
        "languages": [
            {"language": f"lang_{i:02d}", "total_etymons": i + 1, "bundle_attestation_total": 0}
            for i in range(20)
        ],
    }
    rendered = _format_dashboard_drift(pre, rebuilt, max_diffs=5)
    assert "Dashboard drift detected" in rendered
    assert "... and 15 more" in rendered


def test_format_dashboard_drift_rejects_duplicate_languages() -> None:
    """LanguageQualityReport.languages is one-card-per-language by
    schema. If a future change introduces duplicates, the drift
    helper's dict-by-language indexing would silently collapse them.
    Pin the explicit ValueError so the failure mode is loud instead."""
    import pytest

    duplicate_pre = {
        "schema_version": "1.0",
        "languages": [
            {"language": "old-english", "total_etymons": 3},
            {"language": "old-english", "total_etymons": 5},  # duplicate
        ],
    }
    rebuilt = {"schema_version": "1.0", "languages": []}
    with pytest.raises(ValueError, match="duplicate scorecard for language 'old-english'"):
        _format_dashboard_drift(duplicate_pre, rebuilt)


def test_format_dashboard_drift_dict_field_surfaces_per_key_diff() -> None:
    """When a scorecard field is dict-typed (e.g. lexicon_tag_coverage,
    bundle_tag_coverage), the formatter should surface per-key
    differences rather than re-printing the whole dict. Pin this so a
    future change doesn't regress the rendering — operators debugging
    a tag-coverage drift would otherwise see hundreds of unchanged
    entries crowding out the actually-diverging keys."""
    pre = {
        "schema_version": "1.0",
        "languages": [
            {
                "language": "old-english",
                "lexicon_tag_coverage": {"architecture": 3, "settlement": 2, "water": 5},
            }
        ],
    }
    rebuilt = {
        "schema_version": "1.0",
        "languages": [
            {
                "language": "old-english",
                # Only 'water' differs; 'architecture' + 'settlement' match.
                "lexicon_tag_coverage": {"architecture": 3, "settlement": 2, "water": 6},
            }
        ],
    }
    rendered = _format_dashboard_drift(pre, rebuilt)
    # The drifting key is named in the output.
    assert "lexicon_tag_coverage" in rendered
    assert "'water'" in rendered
    assert "pre=5" in rendered
    assert "rebuilt=6" in rendered
    # The non-drifting keys are NOT mentioned — that's the win over
    # the old whole-dict-repr behavior.
    assert "architecture" not in rendered
    assert "settlement" not in rendered


def test_format_dashboard_drift_classifies_added_and_removed_languages() -> None:
    """Pin the add/removed direction in ``_format_dashboard_drift`` —
    pre_card is None means the language appears in REBUILT only (added),
    rebuilt_card is None means it appears in PRE only (removed). The
    set-union iteration is also sorted so the failure report is stable
    across runs."""
    pre = {
        "schema_version": "1.0",
        "bundle_total_words": 100,
        "languages": [
            {"language": "old-english", "total_etymons": 3, "bundle_attestation_total": 1},
        ],
    }
    rebuilt = {
        "schema_version": "1.0",
        "bundle_total_words": 100,
        "languages": [
            {"language": "welsh", "total_etymons": 1, "bundle_attestation_total": 0},
        ],
    }
    rendered = _format_dashboard_drift(pre, rebuilt)
    # old-english is in pre only → REMOVED.
    assert "language 'old-english': removed from rebuilt report" in rendered
    # welsh is in rebuilt only → ADDED.
    assert "language 'welsh': added in rebuilt report" in rendered
    # And deterministic order — old-english sorts before welsh, so
    # 'old-english: removed' must appear before 'welsh: added'.
    assert rendered.index("'old-english'") < rendered.index("'welsh'")


def test_dashboard_report_is_json_serializable(tmp_path: Path) -> None:
    """The dashboard's role downstream of this test is to be persisted
    for trend tracking via ``report_to_json``. Pin that the report
    produced from the round-trip fixture is still JSON-serializable so
    a future field that breaks asdict-to-json round-tripping fails
    here, not in a downstream trend job."""
    pre_db, pre_bundle, jsonl_dir = _setup_pre_state(tmp_path)
    report = _compute_dashboard(pre_db, pre_bundle)
    serialized = json.dumps(asdict(report), indent=2, sort_keys=False)
    # Round-trip parse to confirm valid JSON.
    parsed = json.loads(serialized)
    assert parsed["schema_version"] == "1.0"
    assert isinstance(parsed["languages"], list)
