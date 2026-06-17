"""The ``wyrd kenning lexicon`` click sub-group.

This package holds the ``lexicon``-scoped CLI subcommands as part of
the wyrd-g143 split. Slice 2 added the read-only navigation /
reporting family. Slice 3 added the bulk-sources storage trio, the
JSONL dump/rebuild/diff machinery (the L2 → L3 pipeline backbone),
the operator-CSV ingest family, the export-meanings bundle exporter,
and compact-jsonl. Subsequent slices (4-5) move the remaining
``@lexicon.command`` bodies out of the cli back-compat shim into
per-subcommand modules under this subpackage.

Each per-command module exposes an ``add_to(parent: click.Group)``
hook. This package's ``__init__`` calls each module's ``add_to`` to
register its subcommand on the shared ``lexicon`` click group.

Note: this package's name collides at the directory level with the
sibling ``wyrd.generators.kenning.lexicon`` DB-layer package — they
are distinct absolute import paths (``wyrd.generators.kenning.cli.lexicon``
vs ``wyrd.generators.kenning.lexicon``) and Python's absolute-import
resolution keeps them apart. This package holds CLI subcommands; the
DB-layer package holds the SQLite + enrichment surface.
"""

from __future__ import annotations

import click

from wyrd.generators.kenning.cli.lexicon import (
    adjudicate_element_glosses as _adjudicate_element_glosses_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    audit_cluster_reflexes as _audit_cluster_reflexes_module,
)
from wyrd.generators.kenning.cli.lexicon import audit_descent as _audit_descent_module
from wyrd.generators.kenning.cli.lexicon import (
    audit_etymology_alignment as _audit_etymology_alignment_module,
)
from wyrd.generators.kenning.cli.lexicon import audit_merges as _audit_merges_module
from wyrd.generators.kenning.cli.lexicon import audit_reflexes as _audit_reflexes_module
from wyrd.generators.kenning.cli.lexicon import (
    audit_semantic_coherence as _audit_semantic_coherence_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    audit_short_quotes as _audit_short_quotes_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    audit_toponym_reflexes as _audit_toponym_reflexes_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    backfill_fantasy_tags as _backfill_fantasy_tags_module,
)
from wyrd.generators.kenning.cli.lexicon import backfill_pages as _backfill_pages_module
from wyrd.generators.kenning.cli.lexicon import (
    backfill_toponym_country as _backfill_toponym_country_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    bridge_celtic_forms as _bridge_celtic_forms_module,
)
from wyrd.generators.kenning.cli.lexicon import bridge_language as _bridge_language_module
from wyrd.generators.kenning.cli.lexicon import (
    bridge_phonological_oe as _bridge_phonological_oe_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    bridge_phonological_on as _bridge_phonological_on_module,
)
from wyrd.generators.kenning.cli.lexicon import browse as _browse_module
from wyrd.generators.kenning.cli.lexicon import build as _build_module
from wyrd.generators.kenning.cli.lexicon import (
    canonicalize_toponym_etymology as _canonicalize_toponym_etymology_module,
)
from wyrd.generators.kenning.cli.lexicon import classify_stratum as _classify_stratum_module
from wyrd.generators.kenning.cli.lexicon import (
    cleanup_wiktionary_empirical as _cleanup_wiktionary_empirical_module,
)
from wyrd.generators.kenning.cli.lexicon import clear_enrichment as _clear_enrichment_module
from wyrd.generators.kenning.cli.lexicon import cluster_cognates as _cluster_cognates_module
from wyrd.generators.kenning.cli.lexicon import (
    commit_toponym_candidates as _commit_toponym_candidates_module,
)
from wyrd.generators.kenning.cli.lexicon import compact_jsonl as _compact_jsonl_module
from wyrd.generators.kenning.cli.lexicon import convert_place_names as _convert_place_names_module
from wyrd.generators.kenning.cli.lexicon import curate_collapse as _curate_collapse_module
from wyrd.generators.kenning.cli.lexicon import curate_etymon as _curate_etymon_module
from wyrd.generators.kenning.cli.lexicon import curate_gloss as _curate_gloss_module
from wyrd.generators.kenning.cli.lexicon import decompose as _decompose_module
from wyrd.generators.kenning.cli.lexicon import (
    derive_english_shaped as _derive_english_shaped_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    derive_pronunciation_ipa as _derive_pronunciation_ipa_module,
)
from wyrd.generators.kenning.cli.lexicon import detect_collapses as _detect_collapses_module
from wyrd.generators.kenning.cli.lexicon import diff_bundle as _diff_bundle_module
from wyrd.generators.kenning.cli.lexicon import diff_rebuild as _diff_rebuild_module
from wyrd.generators.kenning.cli.lexicon import disambiguate_fuzzy as _disambiguate_fuzzy_module
from wyrd.generators.kenning.cli.lexicon import (
    dump_empirical_priors as _dump_empirical_priors_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    dump_genitive_priors as _dump_genitive_priors_module,
)
from wyrd.generators.kenning.cli.lexicon import dump_jsonl as _dump_jsonl_module
from wyrd.generators.kenning.cli.lexicon import enrich as _enrich_module
from wyrd.generators.kenning.cli.lexicon import enrichment_status as _enrichment_status_module
from wyrd.generators.kenning.cli.lexicon import era_cell as _era_cell_module
from wyrd.generators.kenning.cli.lexicon import era_coverage as _era_coverage_module
from wyrd.generators.kenning.cli.lexicon import era_reflex as _era_reflex_module
from wyrd.generators.kenning.cli.lexicon import era_timeline as _era_timeline_module
from wyrd.generators.kenning.cli.lexicon import export_meanings as _export_meanings_module
from wyrd.generators.kenning.cli.lexicon import (
    export_runtime_db as _export_runtime_db_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    extract_pfsrd2_monsters as _extract_pfsrd2_monsters_module,
)
from wyrd.generators.kenning.cli.lexicon import fetch_bulk_sources as _fetch_bulk_sources_module
from wyrd.generators.kenning.cli.lexicon import fold_variants as _fold_variants_module
from wyrd.generators.kenning.cli.lexicon import fuzzy_search as _fuzzy_search_module
from wyrd.generators.kenning.cli.lexicon import grade_decomposition as _grade_decomposition_module
from wyrd.generators.kenning.cli.lexicon import import_mining_log as _import_mining_log_module
from wyrd.generators.kenning.cli.lexicon import (
    import_modern_reflexes as _import_modern_reflexes_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    ingest_briggs_personal_names as _ingest_briggs_personal_names_module,
)
from wyrd.generators.kenning.cli.lexicon import ingest_domesday as _ingest_domesday_module
from wyrd.generators.kenning.cli.lexicon import ingest_etymonline as _ingest_etymonline_module
from wyrd.generators.kenning.cli.lexicon import ingest_hearth_tax as _ingest_hearth_tax_module
from wyrd.generators.kenning.cli.lexicon import (
    ingest_hundred_rolls as _ingest_hundred_rolls_module,
)
from wyrd.generators.kenning.cli.lexicon import ingest_kepn as _ingest_kepn_module
from wyrd.generators.kenning.cli.lexicon import (
    ingest_os_open_names as _ingest_os_open_names_module,
)
from wyrd.generators.kenning.cli.lexicon import ingest_speed_1611 as _ingest_speed_1611_module
from wyrd.generators.kenning.cli.lexicon import (
    ingest_toponym_mentions as _ingest_toponym_mentions_module,
)
from wyrd.generators.kenning.cli.lexicon import ingest_wiktionary as _ingest_wiktionary_module
from wyrd.generators.kenning.cli.lexicon import language_report as _language_report_module
from wyrd.generators.kenning.cli.lexicon import link_lemmas as _link_lemmas_module
from wyrd.generators.kenning.cli.lexicon import link_reflexes as _link_reflexes_module
from wyrd.generators.kenning.cli.lexicon import (
    lookup_attested_years as _lookup_attested_years_module,
)
from wyrd.generators.kenning.cli.lexicon import merge_collapses as _merge_collapses_module
from wyrd.generators.kenning.cli.lexicon import merge_lemmas as _merge_lemmas_module
from wyrd.generators.kenning.cli.lexicon import migrate as _migrate_module
from wyrd.generators.kenning.cli.lexicon import mine_attestations as _mine_attestations_module
from wyrd.generators.kenning.cli.lexicon import (
    mine_cognate_descents as _mine_cognate_descents_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    mine_element_glosses as _mine_element_glosses_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    mine_empirical_baselines as _mine_empirical_baselines_module,
)
from wyrd.generators.kenning.cli.lexicon import mine_fantasy_name as _mine_fantasy_name_module
from wyrd.generators.kenning.cli.lexicon import (
    mine_genitive_priors as _mine_genitive_priors_module,
)
from wyrd.generators.kenning.cli.lexicon import mine_llm as _mine_llm_module
from wyrd.generators.kenning.cli.lexicon import (
    mine_pronunciation_llm as _mine_pronunciation_llm_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    mine_same_morpheme_binds as _mine_same_morpheme_binds_module,
)
from wyrd.generators.kenning.cli.lexicon import mine_skeat as _mine_skeat_module
from wyrd.generators.kenning.cli.lexicon import mine_tags_llm as _mine_tags_llm_module
from wyrd.generators.kenning.cli.lexicon import (
    mine_toponym_mentions as _mine_toponym_mentions_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    mine_toponym_mentions_staged as _mine_toponym_mentions_staged_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    mine_toponym_mentions_tiered as _mine_toponym_mentions_tiered_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    mine_wiktextract_corpus as _mine_wiktextract_corpus_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    mine_wiktextract_forms as _mine_wiktextract_forms_module,
)
from wyrd.generators.kenning.cli.lexicon import normalize_l2_vocab as _normalize_l2_vocab_module
from wyrd.generators.kenning.cli.lexicon import normalize_ocr as _normalize_ocr_module
from wyrd.generators.kenning.cli.lexicon import parse_pages as _parse_pages_module
from wyrd.generators.kenning.cli.lexicon import path as _path_module
from wyrd.generators.kenning.cli.lexicon import (
    prepare_toponym_candidates as _prepare_toponym_candidates_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    project_canonical as _project_canonical_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    project_period_forms as _project_period_forms_module,
)
from wyrd.generators.kenning.cli.lexicon import prune_etymon as _prune_etymon_module
from wyrd.generators.kenning.cli.lexicon import prune_toponym as _prune_toponym_module
from wyrd.generators.kenning.cli.lexicon import push_bulk_sources as _push_bulk_sources_module
from wyrd.generators.kenning.cli.lexicon import rando_port_readiness as _rando_port_readiness_module
from wyrd.generators.kenning.cli.lexicon import (
    rebuild_from_jsonl as _rebuild_from_jsonl_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    recompute_reflex_productivity as _recompute_reflex_productivity_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    recover_inprogress_chunks as _recover_inprogress_chunks_module,
)
from wyrd.generators.kenning.cli.lexicon import refill_short_quotes as _refill_short_quotes_module
from wyrd.generators.kenning.cli.lexicon import (
    report as _report_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    report_snapshot as _report_snapshot_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    report_wikipedia_backfill as _report_wikipedia_backfill_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    retag_celtic_branch as _retag_celtic_branch_module,
)
from wyrd.generators.kenning.cli.lexicon import reverse_search as _reverse_search_module
from wyrd.generators.kenning.cli.lexicon import (
    reverse_search_toponyms as _reverse_search_toponyms_module,
)
from wyrd.generators.kenning.cli.lexicon import review as _review_module
from wyrd.generators.kenning.cli.lexicon import set_stratum as _set_stratum_module
from wyrd.generators.kenning.cli.lexicon import stats as _stats_module
from wyrd.generators.kenning.cli.lexicon import synsets as _synsets_module
from wyrd.generators.kenning.cli.lexicon import (
    tag_phonological_vectors as _tag_phonological_vectors_module,
)
from wyrd.generators.kenning.cli.lexicon import verify_bulk_sources as _verify_bulk_sources_module
from wyrd.generators.kenning.cli.lexicon import (
    vocab_health as _vocab_health_module,
)


@click.group("lexicon")
def lexicon() -> None:
    """Manage the authoring lexicon DB (etymology data store)."""


# Register each extracted subcommand module on the @lexicon group.
# Adding a new lexicon subcommand means creating cli/lexicon/<name>.py
# (exposing add_to(parent)) and adding a one-line module import + a
# `<name>_module.add_to(lexicon)` call below. The browse sub-group
# follows the same pattern at the package level (cli/lexicon/browse/).
_audit_cluster_reflexes_module.add_to(lexicon)
_audit_descent_module.add_to(lexicon)
_audit_etymology_alignment_module.add_to(lexicon)
_audit_merges_module.add_to(lexicon)
_audit_reflexes_module.add_to(lexicon)
_audit_semantic_coherence_module.add_to(lexicon)
_audit_short_quotes_module.add_to(lexicon)
_audit_toponym_reflexes_module.add_to(lexicon)
_cleanup_wiktionary_empirical_module.add_to(lexicon)
_backfill_fantasy_tags_module.add_to(lexicon)
_backfill_pages_module.add_to(lexicon)
_backfill_toponym_country_module.add_to(lexicon)
_bridge_celtic_forms_module.add_to(lexicon)
_bridge_language_module.add_to(lexicon)
_bridge_phonological_oe_module.add_to(lexicon)
_bridge_phonological_on_module.add_to(lexicon)
_browse_module.add_to(lexicon)
_build_module.add_to(lexicon)
_canonicalize_toponym_etymology_module.add_to(lexicon)
_classify_stratum_module.add_to(lexicon)
_clear_enrichment_module.add_to(lexicon)
_cluster_cognates_module.add_to(lexicon)
_commit_toponym_candidates_module.add_to(lexicon)
_compact_jsonl_module.add_to(lexicon)
_curate_collapse_module.add_to(lexicon)
_curate_etymon_module.add_to(lexicon)
_curate_gloss_module.add_to(lexicon)
_decompose_module.add_to(lexicon)
_detect_collapses_module.add_to(lexicon)
_dump_empirical_priors_module.add_to(lexicon)
_dump_genitive_priors_module.add_to(lexicon)
_extract_pfsrd2_monsters_module.add_to(lexicon)
_derive_english_shaped_module.add_to(lexicon)
_derive_pronunciation_ipa_module.add_to(lexicon)
_mine_pronunciation_llm_module.add_to(lexicon)
_diff_bundle_module.add_to(lexicon)
_diff_rebuild_module.add_to(lexicon)
_disambiguate_fuzzy_module.add_to(lexicon)
_dump_jsonl_module.add_to(lexicon)
_enrich_module.add_to(lexicon)
_enrichment_status_module.add_to(lexicon)
_era_cell_module.add_to(lexicon)
_era_coverage_module.add_to(lexicon)
_era_reflex_module.add_to(lexicon)
_era_timeline_module.add_to(lexicon)
_export_meanings_module.add_to(lexicon)
_export_runtime_db_module.add_to(lexicon)
_fetch_bulk_sources_module.add_to(lexicon)
_fuzzy_search_module.add_to(lexicon)
_grade_decomposition_module.add_to(lexicon)
_ingest_briggs_personal_names_module.add_to(lexicon)
_ingest_hearth_tax_module.add_to(lexicon)
_ingest_kepn_module.add_to(lexicon)
_convert_place_names_module.add_to(lexicon)
_ingest_hundred_rolls_module.add_to(lexicon)
_ingest_os_open_names_module.add_to(lexicon)
_ingest_domesday_module.add_to(lexicon)
_ingest_etymonline_module.add_to(lexicon)
_ingest_speed_1611_module.add_to(lexicon)
_ingest_toponym_mentions_module.add_to(lexicon)
_ingest_wiktionary_module.add_to(lexicon)
_import_mining_log_module.add_to(lexicon)
_import_modern_reflexes_module.add_to(lexicon)
_language_report_module.add_to(lexicon)
_retag_celtic_branch_module.add_to(lexicon)
_link_lemmas_module.add_to(lexicon)
_lookup_attested_years_module.add_to(lexicon)
_fold_variants_module.add_to(lexicon)
_merge_collapses_module.add_to(lexicon)
_merge_lemmas_module.add_to(lexicon)
_link_reflexes_module.add_to(lexicon)
_migrate_module.add_to(lexicon)
_mine_attestations_module.add_to(lexicon)
_mine_empirical_baselines_module.add_to(lexicon)
_mine_genitive_priors_module.add_to(lexicon)
_mine_same_morpheme_binds_module.add_to(lexicon)
_mine_cognate_descents_module.add_to(lexicon)
_mine_fantasy_name_module.add_to(lexicon)
_mine_llm_module.add_to(lexicon)
_mine_element_glosses_module.add_to(lexicon)
_adjudicate_element_glosses_module.add_to(lexicon)
_mine_skeat_module.add_to(lexicon)
_mine_tags_llm_module.add_to(lexicon)
_mine_toponym_mentions_module.add_to(lexicon)
_mine_toponym_mentions_staged_module.add_to(lexicon)
_mine_toponym_mentions_tiered_module.add_to(lexicon)
_mine_wiktextract_corpus_module.add_to(lexicon)
_mine_wiktextract_forms_module.add_to(lexicon)
_normalize_l2_vocab_module.add_to(lexicon)
_normalize_ocr_module.add_to(lexicon)
_parse_pages_module.add_to(lexicon)
_path_module.add_to(lexicon)
_prepare_toponym_candidates_module.add_to(lexicon)
_project_canonical_module.add_to(lexicon)
_project_period_forms_module.add_to(lexicon)
_prune_etymon_module.add_to(lexicon)
_prune_toponym_module.add_to(lexicon)
_push_bulk_sources_module.add_to(lexicon)
_rando_port_readiness_module.add_to(lexicon)
_rebuild_from_jsonl_module.add_to(lexicon)
_recompute_reflex_productivity_module.add_to(lexicon)
_recover_inprogress_chunks_module.add_to(lexicon)
_refill_short_quotes_module.add_to(lexicon)
_reverse_search_toponyms_module.add_to(lexicon)
_review_module.add_to(lexicon)
_synsets_module.add_to(lexicon)
_report_module.add_to(lexicon)
_vocab_health_module.add_to(lexicon)
_report_snapshot_module.add_to(lexicon)
_report_wikipedia_backfill_module.add_to(lexicon)
_reverse_search_module.add_to(lexicon)
_set_stratum_module.add_to(lexicon)
_stats_module.add_to(lexicon)
_tag_phonological_vectors_module.add_to(lexicon)
_verify_bulk_sources_module.add_to(lexicon)


def add_to(parent: click.Group) -> None:
    """Register the ``lexicon`` sub-group on a parent click group.

    Called once by ``wyrd.generators.kenning.cli.__init__`` while the
    top-level ``@cli`` group is being constructed.
    """
    parent.add_command(lexicon)
