"""CLI integration tests for the curation commands — wyrd-gfg7.

Covers three commands the wyrd-2jhs / wyrd-lene / wyrd-f4nl PRs added
that previously only had helper-function coverage:

- ``lexicon curate-etymon`` (wyrd-2jhs): appends etymon_curation
  events; validates mutual-exclusivity of --lemma-ref / --merged-into-ref.
- ``lexicon prune-toponym`` (wyrd-lene): appends toponym remove
  events; validates the ref-existence guard.
- ``lexicon diff-rebuild`` (wyrd-f4nl): row-count diff between live
  and rebuilt DBs; exits 1 on any delta.

The helper-level logic these commands wrap already has unit-test
coverage; these tests focus on the CLI wiring (option parsing,
error mapping, file IO, exit codes).
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.jsonl_log import write_jsonl


def _source_file_with(tmp_path: Path, source_id: str, rows: list[dict]) -> Path:
    """Seed a per-source JSONL file with a source row + the given
    rows. Used to give prune-toponym a target file."""
    path = tmp_path / f"{source_id}.jsonl"
    write_jsonl(path, [{"_type": "source", "ref": source_id, "title": "X"}, *rows])
    return path


# ---------------------------------------------------------------------------
# lexicon curate-etymon
# ---------------------------------------------------------------------------


def test_curate_etymon_appends_lemma_event(tmp_path: Path):
    """Happy path: --lemma-ref appends an etymon_curation event with
    the lemma_ref payload + the operator's reason."""
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-etymon",
            "old-english:caelf",
            "--lemma-ref",
            "old-english:cealf",
            "--inflection",
            "spelling_variant",
            "--reason",
            "scribal æ/ae",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert curation_file.exists()
    lines = curation_file.read_text().splitlines()
    # Source row + event row.
    assert len(lines) == 2
    event = json.loads(lines[1])
    assert event["_type"] == "etymon_curation"
    assert event["ref"] == "old-english:caelf"
    assert event["lemma_ref"] == "old-english:cealf"
    assert event["inflection"] == "spelling_variant"
    assert event["reason"] == "scribal æ/ae"


def test_curate_etymon_creates_source_row_on_first_invocation(tmp_path: Path):
    """When the curation file doesn't exist, the command creates it
    with the synthetic manual-curation source row at the top so the
    build contract holds."""
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-etymon",
            "old-english:caelf",
            "--lemma-ref",
            "old-english:cealf",
            "--curation-file",
            str(curation_file),
        ],
    )
    source_row = json.loads(curation_file.read_text().splitlines()[0])
    assert source_row["_type"] == "source"
    assert source_row["ref"] == "manual-curation"


def test_curate_etymon_appends_to_existing_file_without_dupe_source(tmp_path: Path):
    """A second invocation appends an event without re-emitting the
    synthetic source row. Catches the regression where the file-exists
    branch fires the if-creates-file path too."""
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-etymon",
            "old-english:caelf",
            "--lemma-ref",
            "old-english:cealf",
            "--curation-file",
            str(curation_file),
        ],
    )
    # Second invocation against the same file.
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-etymon",
            "old-english:tun",
            "--lemma-ref",
            "old-english:toun",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = curation_file.read_text().splitlines()
    # Exactly one source row + two events.
    sources = [
        line for line in lines if json.loads(line).get("_type") == "source"
    ]
    assert len(sources) == 1
    events = [
        line for line in lines if json.loads(line).get("_type") == "etymon_curation"
    ]
    assert len(events) == 2


def test_curate_etymon_merged_into_ref_only(tmp_path: Path):
    """Symmetric to the --lemma-ref happy path: --merged-into-ref
    alone records a tombstone curation with no lemma_ref key."""
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-etymon",
            "old-norse:vath",
            "--merged-into-ref",
            "old-norse:vað",
            "--reason",
            "OCR cluster",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output
    event = json.loads(curation_file.read_text().splitlines()[1])
    assert event["merged_into_ref"] == "old-norse:vað"
    # No --lemma-ref → key absent (NOT present-with-None).
    assert "lemma_ref" not in event
    assert event["reason"] == "OCR cluster"


def test_curate_etymon_empty_lemma_ref_appends_null_clear(tmp_path: Path):
    """``''`` for --lemma-ref records explicit null (clear semantics)
    rather than the empty string."""
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-etymon",
            "old-english:caelf",
            "--lemma-ref",
            "",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output
    event = json.loads(curation_file.read_text().splitlines()[1])
    assert event["lemma_ref"] is None


def test_curate_etymon_rejects_both_lemma_and_merge_set(tmp_path: Path):
    """Mutual-exclusivity check (wyrd-2jhs round-3 review): setting
    both --lemma-ref and --merged-into-ref to truthy values produces
    a contradiction the applier can't resolve cleanly."""
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-etymon",
            "old-english:caelf",
            "--lemma-ref",
            "old-english:cealf",
            "--merged-into-ref",
            "old-english:ghost",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()
    # File NOT created when validation rejects.
    assert not curation_file.exists()


def test_curate_etymon_clearing_both_succeeds(tmp_path: Path):
    """The mutex relaxation per wyrd-2jhs round 3: clearing both
    columns with ``''`` is a legit revert-to-standalone."""
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-etymon",
            "old-english:caelf",
            "--lemma-ref",
            "",
            "--merged-into-ref",
            "",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output
    event = json.loads(curation_file.read_text().splitlines()[1])
    assert event["lemma_ref"] is None
    assert event["merged_into_ref"] is None


def test_curate_etymon_requires_at_least_one_ref(tmp_path: Path):
    """No --lemma-ref or --merged-into-ref → no curation to record."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-etymon",
            "old-english:caelf",
            "--curation-file",
            str(tmp_path / "_curation.jsonl"),
        ],
    )
    assert result.exit_code != 0
    assert "need at least one" in result.output.lower()


# ---------------------------------------------------------------------------
# lexicon prune-toponym
# ---------------------------------------------------------------------------


def test_prune_toponym_appends_remove_event(tmp_path: Path):
    """Happy path: prune-toponym appends a remove event with the
    operator's reason for git-blame audit."""
    _source_file_with(
        tmp_path,
        "skeat",
        [
            {
                "_type": "toponym",
                "ref": "Ghost@Cambs",
                "modern_name": "Ghost",
                "region": "Cambs",
            },
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "prune-toponym",
            "Ghost@Cambs",
            "skeat",
            "--reason",
            "test cleanup",
            "--jsonl-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = (tmp_path / "skeat.jsonl").read_text().splitlines()
    # Source + toponym + remove event = 3 lines.
    assert len(lines) == 3
    event = json.loads(lines[-1])
    assert event == {
        "_op": "remove",
        "_type": "toponym",
        "ref": "Ghost@Cambs",
        "_reason": "test cleanup",
    }


def test_prune_toponym_without_reason_omits_reason_field(tmp_path: Path):
    """When --reason is absent, the appended event has no _reason key
    (rather than _reason=None or _reason=''). Verifies the
    ``if reason is not None`` branch in the CLI."""
    _source_file_with(
        tmp_path,
        "skeat",
        [
            {
                "_type": "toponym",
                "ref": "Ghost@Cambs",
                "modern_name": "Ghost",
                "region": "Cambs",
            },
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "prune-toponym",
            "Ghost@Cambs",
            "skeat",
            "--jsonl-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    event = json.loads((tmp_path / "skeat.jsonl").read_text().splitlines()[-1])
    assert event["_op"] == "remove"
    assert "_reason" not in event


def test_prune_toponym_rejects_unknown_ref(tmp_path: Path):
    """The ref-validation guard (Claude-fallback PR #180 finding):
    typo'd ref → UsageError, no file mutation. Without the guard, a
    typo would silently append a no-op remove event and look like
    success."""
    _source_file_with(
        tmp_path,
        "skeat",
        [
            {
                "_type": "toponym",
                "ref": "Ghost@Cambs",
                "modern_name": "Ghost",
                "region": "Cambs",
            },
        ],
    )
    pre_lines = len((tmp_path / "skeat.jsonl").read_text().splitlines())
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "prune-toponym",
            "Typo@Cambs",  # not in the file
            "skeat",
            "--jsonl-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
    # File unchanged — no event appended on rejection.
    post_lines = len((tmp_path / "skeat.jsonl").read_text().splitlines())
    assert pre_lines == post_lines


def test_prune_toponym_rejects_missing_source_file(tmp_path: Path):
    """When the named source file doesn't exist, prune-toponym fails
    fast with a clear error rather than silently creating one."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "prune-toponym",
            "Ghost@Cambs",
            "nonexistent",
            "--jsonl-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "no such" in result.output.lower()


def test_prune_toponym_jsonl_dir_default_data_mining(tmp_path: Path, monkeypatch):
    """Without --jsonl-dir, prune-toponym looks in data/mining/. Test
    by chdir'ing the tmp_path into a fake data/mining hierarchy and
    relying on Click's default-path resolution."""
    mining_dir = tmp_path / "data" / "mining"
    mining_dir.mkdir(parents=True)
    write_jsonl(
        mining_dir / "skeat.jsonl",
        [
            {"_type": "source", "ref": "skeat", "title": "X"},
            {
                "_type": "toponym",
                "ref": "Ghost@Cambs",
                "modern_name": "Ghost",
                "region": "Cambs",
            },
        ],
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "prune-toponym", "Ghost@Cambs", "skeat"],
    )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# lexicon diff-rebuild
# ---------------------------------------------------------------------------


def test_diff_rebuild_zero_deltas_on_self_rebuild(tmp_path: Path):
    """Rebuild from a JSONL dir into a fresh DB, then diff-rebuild
    against that DB using the same source dir. All deltas should be
    zero, exit code 0."""
    from wyrd.generators.kenning.lexicon import init_schema

    # Minimal JSONL: a single source with a single etymon.
    write_jsonl(
        tmp_path / "skeat.jsonl",
        [
            {"_type": "source", "ref": "skeat", "title": "X"},
            {
                "_type": "etymon",
                "ref": "old-english:cot",
                "language": "old-english",
                "canonical_form": "cot",
            },
        ],
    )
    # Pre-build the DB so diff-rebuild has something to diff against.
    db_path = tmp_path / "current.db"
    init_schema(db_path)
    runner = CliRunner()
    runner.invoke(
        cli_root,
        ["lexicon", "rebuild-from-jsonl", "--db", str(db_path), "--jsonl-dir", str(tmp_path)],
    )
    # Diff against the SAME JSONL — should be zero deltas.
    result = runner.invoke(
        cli_root,
        ["lexicon", "diff-rebuild", "--db", str(db_path), "--jsonl-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "No row-count deltas" in (result.stderr or result.output)


def test_diff_rebuild_with_enrichment_runs_orchestrator(tmp_path: Path):
    """--with-enrichment triggers run_ocr_lemma_enrichment on the
    temp rebuild DB. Verify the flag wires through by seeding two
    OCR-mergeable etymons; without enrichment merged_into_id stays
    NULL on the rebuild and the diff would flag a delta from the
    pre-built DB (which has the merge). With enrichment, the rebuild
    runs the merge and the delta closes."""
    from wyrd.generators.kenning.lexicon import LexiconDB, init_schema

    write_jsonl(
        tmp_path / "skeat.jsonl",
        [
            {"_type": "source", "ref": "skeat", "title": "X"},
            {
                "_type": "etymon",
                "ref": "old-english:cæt",
                "language": "old-english",
                "canonical_form": "cæt",
            },
            {
                "_type": "etymon",
                "ref": "old-english:caet",
                "language": "old-english",
                "canonical_form": "caet",
            },
        ],
    )
    db_path = tmp_path / "current.db"
    init_schema(db_path)
    runner = CliRunner()
    # Build + run enrichment once on the live DB so the merged_into_id
    # column has a non-trivial value to round-trip against.
    runner.invoke(
        cli_root,
        ["lexicon", "rebuild-from-jsonl", "--db", str(db_path), "--jsonl-dir", str(tmp_path)],
    )
    from wyrd.generators.kenning.enrichment import run_ocr_lemma_enrichment

    with LexiconDB(db_path) as db:
        run_ocr_lemma_enrichment(db, apply=True)
    # diff-rebuild WITH enrichment should produce zero deltas because
    # the temp rebuild also runs the orchestrator and ends up at the
    # same state.
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "diff-rebuild",
            "--db",
            str(db_path),
            "--jsonl-dir",
            str(tmp_path),
            "--with-enrichment",
        ],
    )
    assert result.exit_code == 0, result.output


def test_diff_rebuild_exit_1_on_mismatch(tmp_path: Path):
    """Build a DB from one JSONL, mutate the JSONL, then diff-rebuild.
    Exit code should be 1, output should call out the delta."""
    from wyrd.generators.kenning.lexicon import init_schema

    jsonl_path = tmp_path / "skeat.jsonl"
    write_jsonl(
        jsonl_path,
        [
            {"_type": "source", "ref": "skeat", "title": "X"},
            {
                "_type": "etymon",
                "ref": "old-english:cot",
                "language": "old-english",
                "canonical_form": "cot",
            },
        ],
    )
    db_path = tmp_path / "current.db"
    init_schema(db_path)
    runner = CliRunner()
    runner.invoke(
        cli_root,
        ["lexicon", "rebuild-from-jsonl", "--db", str(db_path), "--jsonl-dir", str(tmp_path)],
    )
    # Add a new etymon to the JSONL — rebuild will now produce more rows.
    with jsonl_path.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "_type": "etymon",
                    "ref": "old-english:tun",
                    "language": "old-english",
                    "canonical_form": "tun",
                }
            )
            + "\n"
        )
    result = runner.invoke(
        cli_root,
        ["lexicon", "diff-rebuild", "--db", str(db_path), "--jsonl-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    # Output includes the etymon delta row.
    assert "etymon" in result.output
    # ``+1`` because the new JSONL has 1 more etymon than the original
    # DB. Sign + magnitude are contiguous per the format-string fix in
    # PR #181 (post-Claude-fallback round).
    assert "+1" in result.output
