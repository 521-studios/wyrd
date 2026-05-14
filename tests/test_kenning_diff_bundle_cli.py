"""Integration tests for ``lexicon diff-bundle`` (wyrd-w3x0).

End-to-end: build a small DB fixture, run the L3 chain + export to get a
"committed" bundle, dump the JSONL, then invoke ``diff-bundle`` against
the dumped JSONL set and the committed bundle. Asserts:

* Round-trip clean → exit 0 + 'No drift' in stdout.
* Committed bundle mutated → exit 1 + structured drift summary.

Bypasses L1 bulk ingest via --skip-bulk because CI runners don't have
~/.wyrd/sources/. The CI-friendly synthetic shape mirrors how operators
would invoke diff-bundle during release prep, just with a tiny dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.enrichment import run_full_enrichment
from wyrd.generators.kenning.jsonl_build import collect_curation_overrides, jsonl_paths_in
from wyrd.generators.kenning.jsonl_dump import dump_all_sources
from wyrd.generators.kenning.lexicon import (
    LexiconDB,
    collect_canonical_decompositions,
    collect_fantasy_morphemes,
    export_meanings,
    init_schema,
)


def _populate_fixture(db_path: Path) -> None:
    """Trimmed version of the Phase 3 round-trip fixture. Two languages,
    three citations per etymon (≥2-witness gate), one toponym with
    attribution — enough to produce a non-trivial promoted bundle."""
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.upsert_source(id="skeat", title="Skeat — Cambridgeshire", year=1901)
        db.upsert_source(id="mawer", title="Mawer — N&D", year=1920)
        db.upsert_source(id="ekwall", title="Ekwall — Lancashire", year=1922)

        cot_id = db.upsert_etymon("cot", "old-english")
        tun_id = db.upsert_etymon("tun", "old-english")

        for gloss in ("cottage", "hut"):
            db.conn.execute(
                "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
                (cot_id, gloss),
            )
        db.conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, 'enclosure')",
            (tun_id,),
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
        for etymon_id, page_map in (
            (cot_id, {"skeat": "15", "mawer": "7", "ekwall": "201"}),
            (tun_id, {"skeat": "20", "mawer": "8", "ekwall": "202"}),
        ):
            for source_id, page in page_map.items():
                db.conn.execute(
                    "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, ?, ?)",
                    (etymon_id, source_id, page),
                )
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
        db.commit()


def _build_committed_bundle(db_path: Path, jsonl_dir: Path) -> str:
    """Emit a bundle JSON string in exactly the format
    ``lexicon export-meanings --output`` would write — payload + trailing
    newline.

    ``jsonl_dir`` is required because ``run_full_enrichment``'s
    ``curation_state`` arg must mirror what ``diff-bundle`` does (it
    calls ``collect_curation_overrides(paths) or None``). With the
    current fixture this is a no-op (no curation events), but pinning
    the symmetry here prevents a silent false-no-drift if a future
    fixture grows curation rows."""
    paths = jsonl_paths_in(jsonl_dir)
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
    return json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"


def _setup_committed_and_jsonl(tmp_path: Path) -> tuple[Path, Path]:
    """Build a fixture DB, dump JSONL into a directory, then build the
    committed bundle using the dumped paths (so curation-state resolution
    matches what diff-bundle does internally). Returns
    ``(committed_bundle_path, jsonl_dir)`` — both arguments diff-bundle
    would consume on a real operator run.

    The dump-before-build order matters: ``_build_committed_bundle``
    resolves ``curation_state`` via ``collect_curation_overrides`` over
    the JSONL paths, so the JSONL files must exist first."""
    pre_db = tmp_path / "pre.db"
    _populate_fixture(pre_db)

    jsonl_dir = tmp_path / "dumped"
    jsonl_dir.mkdir()
    with LexiconDB(pre_db) as db:
        dump_all_sources(db.conn, jsonl_dir, exclude=())

    committed_bundle = tmp_path / "committed.json"
    committed_bundle.write_text(_build_committed_bundle(pre_db, jsonl_dir))

    return committed_bundle, jsonl_dir


def test_diff_bundle_reports_no_drift_when_rebuild_matches_committed(tmp_path: Path) -> None:
    """End-to-end happy path: rebuild from the JSONL that produced the
    committed bundle, exit 0, 'No drift' line in stdout."""
    committed_bundle, jsonl_dir = _setup_committed_and_jsonl(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "diff-bundle",
            "--jsonl-dir",
            str(jsonl_dir),
            "--bundle",
            str(committed_bundle),
            "--skip-bulk",
            "--min-witnesses",
            "2",
            "--no-preset",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert "No drift" in result.output


def test_diff_bundle_reports_drift_when_committed_diverges(tmp_path: Path) -> None:
    """Mutate the committed bundle so it doesn't match what a fresh
    rebuild would produce. Expect exit 1 + a structured drift summary
    that names the divergence."""
    committed_bundle, jsonl_dir = _setup_committed_and_jsonl(tmp_path)

    # Deliberately mutate the committed bundle: remove one subject and
    # change a tag in another. Both should surface in the structural diff.
    bundle = json.loads(committed_bundle.read_text())
    assert len(bundle["subjects"]) >= 1, "fixture didn't promote any subjects"
    bundle["subjects"].pop(0)  # drop one subject → 'subjects_removed' should fire
    committed_bundle.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "diff-bundle",
            "--jsonl-dir",
            str(jsonl_dir),
            "--bundle",
            str(committed_bundle),
            "--skip-bulk",
            "--min-witnesses",
            "2",
            "--no-preset",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 1, result.output + (result.stderr or "")
    assert "Bundle drift detected" in result.output
    # The rebuild promotes the subject the committed bundle dropped, so
    # it appears in the 'Added (sample):' bucket of the structural diff.
    assert "Subjects:" in result.output
    assert "+1 added" in result.output or "added" in result.output


def test_diff_bundle_reports_drift_summary_uses_structural_classification(
    tmp_path: Path,
) -> None:
    """Pin the first-byte-divergence window AND the subjects section
    both appear when drift exists. Operators rely on these two views
    being side-by-side to triangulate the change."""
    committed_bundle, jsonl_dir = _setup_committed_and_jsonl(tmp_path)

    # Mutate to flip a meaning — affects subject identity tuple, so this
    # registers as both 'subject removed' (with old meaning) AND
    # 'subject added' (with new meaning).
    text = committed_bundle.read_text()
    mutated = text.replace('"cottage"', '"cabin"', 1)
    assert mutated != text, "fixture didn't contain 'cottage' to mutate"
    committed_bundle.write_text(mutated)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "diff-bundle",
            "--jsonl-dir",
            str(jsonl_dir),
            "--bundle",
            str(committed_bundle),
            "--skip-bulk",
            "--min-witnesses",
            "2",
            "--no-preset",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    # Byte-divergence window — load-bearing on long-bundle drift cases
    # where the structural diff is partial.
    assert "First byte-level divergence" in result.output
    # Structural lines for the meaning-change case.
    assert "Subjects:" in result.output


# --- --lang-threshold parser error branches -----------------------------
#
# Three click.BadParameter paths cover the malformed --lang-threshold
# spec cases. These don't need the rebuild fixture — Click raises on
# argument-parse before we get to the rebuild — so we point at a
# minimal valid --jsonl-dir + --bundle just to satisfy the existing
# (already-validated) options.


def _common_args_for_parser_error(tmp_path: Path) -> list[str]:
    """Minimal args the parser-error tests share. Both files exist
    (Click validates exists=True) but the rebuild never runs because
    --lang-threshold parsing raises first."""
    jsonl_dir = tmp_path / "jsonl"
    jsonl_dir.mkdir()
    bundle = tmp_path / "bundle.json"
    bundle.write_text('{"subjects": []}\n', encoding="utf-8")
    return [
        "lexicon",
        "diff-bundle",
        "--jsonl-dir",
        str(jsonl_dir),
        "--bundle",
        str(bundle),
        "--skip-bulk",
    ]


def test_diff_bundle_rejects_lang_threshold_without_equals(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [*_common_args_for_parser_error(tmp_path), "--lang-threshold", "old-english"],
    )
    assert result.exit_code != 0
    assert "LANG=N" in result.output + (result.stderr or "")


def test_diff_bundle_rejects_lang_threshold_empty_side(tmp_path: Path) -> None:
    """``=3`` (empty LANG) and ``old-english=`` (empty N) both fall under
    the second BadParameter branch."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [*_common_args_for_parser_error(tmp_path), "--lang-threshold", "=3"],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "non-empty" in combined


def test_diff_bundle_rejects_lang_threshold_with_non_integer(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [*_common_args_for_parser_error(tmp_path), "--lang-threshold", "old-english=many"],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "integer" in combined
