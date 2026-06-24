"""CLI rendering tests for ``clear-enrichment`` count lines.

``lexicon_clear_enrichment`` renders one ``  {verb} {count} {noun}`` line per
clearable enrichment artifact, driven by the ``_CLEAR_COUNT_LINES`` (key, noun)
table. The table's failure mode is a row pairing a result key with the wrong
noun (or a typo'd key that silently prints nothing). These tests pin the
key->noun mapping independently of the production table so a mispairing is
caught (pr-test-analyzer #754).
"""

from __future__ import annotations

from click.testing import CliRunner

from wyrd.generators.kenning.cli.lexicon import clear_enrichment as ce_mod

# Expected (result key -> rendered noun), declared here independently of the
# production _CLEAR_COUNT_LINES table. If a table row mispairs a key with a
# noun, the CLI output won't match this map and the test fails.
_EXPECTED_NOUNS = {
    "ocr_merges_to_clear": "merged_into_id rows",
    "lemma_links_to_clear": "lemma links",
    "text_match_rows_to_clear": "text-match rows",
    "cognate_assignments_to_clear": "cognate_id assignments",
    "attested_years_to_clear": "attested_year values",
    "attestation_rows_to_clear": "toponym_attestation rows",
    "period_form_rows_to_clear": "etymon_period_form rows",
    "canonical_nodes_to_clear": "canonical nodes (+ binds/labels)",
}


class _FakeDB:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _invoke(monkeypatch, tmp_path, result, *, apply):
    db = tmp_path / "lexicon.db"
    db.write_text("")  # must exist for click.Path(exists=True)
    monkeypatch.setattr(ce_mod, "LexiconDB", lambda *a, **k: _FakeDB())
    monkeypatch.setattr(ce_mod, "clear_enrichment", lambda db, *, stage, apply: result)
    args = ["--db", str(db), "--stage", result["stage"]]
    if apply:
        args.append("--apply")
    return CliRunner().invoke(ce_mod.lexicon_clear_enrichment, args)


def test_every_table_row_has_a_pinned_expectation():
    """Guards completeness: a new _CLEAR_COUNT_LINES row must add an expectation."""
    assert {key for key, _ in ce_mod._CLEAR_COUNT_LINES} == set(_EXPECTED_NOUNS)


def test_each_count_line_pairs_key_with_correct_noun(tmp_path, monkeypatch):
    # Distinct non-zero count per key so a mispaired line is visible in output.
    result = {"stage": "all-derived"}
    for i, key in enumerate(_EXPECTED_NOUNS, start=1):
        result[key] = i
    res = _invoke(monkeypatch, tmp_path, result, apply=True)
    assert res.exit_code == 0, res.output
    for i, (key, noun) in enumerate(_EXPECTED_NOUNS.items(), start=1):
        assert f"  cleared {i} {noun}" in res.output, f"missing/mispaired line for {key}"


def test_dry_run_verb_and_zero_counts_suppressed(tmp_path, monkeypatch):
    # Zero counts print nothing; a non-zero count uses the dry-run verb.
    result = {"stage": "ocr", **dict.fromkeys(_EXPECTED_NOUNS, 0)}
    result["lemma_links_to_clear"] = 4
    res = _invoke(monkeypatch, tmp_path, result, apply=False)
    assert res.exit_code == 0, res.output
    assert "  would clear 4 lemma links" in res.output
    assert "merged_into_id rows" not in res.output  # 0 -> suppressed
    assert "(dry-run; pass --apply to commit)" in res.output
