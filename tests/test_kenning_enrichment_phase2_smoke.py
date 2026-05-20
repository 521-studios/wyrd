"""Phase 2 smoke: run_full_enrichment without skip_l3_derivations
on a synthetic DB exercises all passes without crashing."""

from pathlib import Path

from wyrd.generators.kenning.enrichment import run_full_enrichment
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


def test_run_full_enrichment_phase2_chain_smoke(tmp_path: Path):
    """Empty (schema-only) DB → run the full chain dry-run. Every
    pass should report zero candidates without crashing."""
    db_path = tmp_path / "test.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        result = run_full_enrichment(db, apply=False)
    assert result["order"] == [
        "normalize-ocr",
        "link-lemmas",
        "decompose",
        "cluster-cognates",
        "classify-stratum",
        "derive-english-shaped",
        "tag-phonological-vectors",
        "project-period-forms",
    ]
    # Every L3 result is non-None (since we didn't skip)
    for key in (
        "decompose",
        "cognates",
        "stratum",
        "english_shaped",
        "phonological_vectors",
        "period_forms",
    ):
        assert result[key] is not None, f"missing {key} section"
    # Curation absent without curation_state
    assert result["curation"] is None
