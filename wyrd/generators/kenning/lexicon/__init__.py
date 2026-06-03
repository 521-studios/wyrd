"""Lexicon data store: SQLite-backed authoring DB for place-name etymology.

Authoritative for authoring (mining sources, tracking citations, recording
disagreement). The runtime keeps reading meanings.json, which is exported from
this DB by `wyrd kenning lexicon export-meanings`.
"""

from __future__ import annotations

# wyrd-67fv: every authoring concern lives in a dedicated submodule
# (``lexicon.constants``, ``lexicon.db``, ``lexicon.schema``,
# ``lexicon.citations``, ``lexicon.seed``, ``lexicon.empirical_priors``,
# ``lexicon.sql``). The blocks below re-export the PUBLIC surface — plus
# the small set of underscore-prefixed names tests actually import — so
# external callers (cli/, era/rewind.py, disambiguator.py, the test suite)
# keep their existing ``from wyrd.generators.kenning.lexicon import …``
# imports unchanged. Underscore helpers without an external caller are
# NOT re-exported; in-package callers reach them via the submodule path
# directly (``from wyrd.generators.kenning.lexicon.schema import …``).
from wyrd.generators.kenning.lexicon.attestation_mining import (  # noqa: E402, F401
    _extract_attestation_pairs,  # used by toponym_reverse_search + test_kenning_lexicon
    mine_toponym_attestations,
)
from wyrd.generators.kenning.lexicon.attestation_years import (  # noqa: E402, F401
    _earliest_year_in_notes,  # tested directly by test_kenning_lexicon
    lookup_attested_years,
)
from wyrd.generators.kenning.lexicon.bridges import (  # noqa: E402, F401
    bridge_celtic_forms,
    bridge_generic_language,
    bridge_phonological_oe,
    bridge_phonological_on,
)
from wyrd.generators.kenning.lexicon.bundle import (  # noqa: E402, F401
    _LANG_CODE_TO_JSON_FIELD,  # used by runtime/decomposition.py
    RECOMMENDED_LANG_THRESHOLDS,  # used by language_quality/audits.py
    export_meanings,
)
from wyrd.generators.kenning.lexicon.bundle._emit import (  # noqa: E402, F401
    _emit_inflection_list,  # tested directly by test_kenning_lexicon
    _emit_variant_list,  # tested directly by test_kenning_lexicon
)
from wyrd.generators.kenning.lexicon.bundle._export import (  # noqa: E402, F401
    _build_witness_filter,  # tested directly by test_kenning_lexicon
)
from wyrd.generators.kenning.lexicon.bundle._family import (  # noqa: E402, F401
    _NON_SCHOLAR_SOURCES,  # used by wikipedia_backfill_report
    _fetch_cluster_mate_tags,  # tested directly by test_kenning_lexicon
    _fetch_member_variants,  # tested directly by test_kenning_wiktextract_forms
    _fetch_reflex_glosses,  # tested directly by test_kenning_lexicon
    _fetch_root_era_reflexes,  # tested directly by test_kenning_lexicon
    _filter_concatenation_glosses,  # tested directly by test_kenning_lexicon
    _gather_family,  # tested directly by test_kenning_lexicon
)
from wyrd.generators.kenning.lexicon.bundle._subject import (  # noqa: E402, F401
    _emit_era_reflexes,  # tested directly by test_kenning_lexicon
)
from wyrd.generators.kenning.lexicon.citations import (  # noqa: E402, F401
    _normalize_for_quote_match,  # tested directly by test_kenning_lexicon
    _quote_body_excerpt,  # tested directly by test_kenning_lexicon
    backfill_citation_pages,
    detect_running_headers,
    page_for_offset,
    parse_running_header_pages,
    parse_skeat_section_header_pages,
)
from wyrd.generators.kenning.lexicon.cognate_cluster import (  # noqa: E402, F401
    cluster_cognates,
)
from wyrd.generators.kenning.lexicon.constants import (  # noqa: E402, F401
    LANGUAGE_FIELDS,
    NON_LANGUAGE_FIELDS,  # re-exported for tests; not used inside this module
    normalize_ocr_form,
    position_from_usage,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB  # noqa: E402, F401
from wyrd.generators.kenning.lexicon.decomposition_export import (  # noqa: E402, F401
    collect_canonical_decompositions,
)
from wyrd.generators.kenning.lexicon.empirical_priors import (  # noqa: E402, F401
    collect_empirical_priors,
)
from wyrd.generators.kenning.lexicon.enrichment import (  # noqa: E402, F401
    annotate_fragments_with_corpus_evidence,
    clear_enrichment,
    cluster_ocr_variants,
    derive_lemma_candidate,
    derive_lemma_candidates,
    derive_mutation_lemma_candidate,
    fuzzy_search_attestations,
    levenshtein,
    link_lemmas,
    reverse_search_attestations,
)
from wyrd.generators.kenning.lexicon.era_reflex import (  # noqa: E402, F401
    EraReflex,
    etymon_era_reflexes,
)
from wyrd.generators.kenning.lexicon.fantasy_export import (  # noqa: E402, F401
    collect_fantasy_morphemes,
)
from wyrd.generators.kenning.lexicon.ingest import (  # noqa: E402, F401
    ingest_parsed_entries,
)
from wyrd.generators.kenning.lexicon.period_form import (  # noqa: E402, F401
    _find_longest_suffix_match,  # tested directly by test_kenning_lexicon
    _strip_diacritics,  # tested directly by test_kenning_lexicon
    project_period_forms,
)
from wyrd.generators.kenning.lexicon.schema import (  # noqa: E402, F401
    _migrate_toponym_etymology_canonical,  # used by test_kenning_toponym_etymology_canonical
    init_schema,
    migrate_schema,
    record_mining_run,
)
from wyrd.generators.kenning.lexicon.seed import seed_from_meanings  # noqa: E402, F401
from wyrd.generators.kenning.lexicon.synsets import (  # noqa: E402, F401
    assign_etymon_to_meaning_synset,
    get_meaning_preserving_candidates,
    get_meaning_synsets_for_etymon,
    list_meaning_synsets,
    seed_meaning_synsets,
)

# Underscore-prefixed alias for the existing internal callers; the
# canonical name is the un-prefixed export from ``lexicon.constants``.
# Kept until the rest of the package's call sites migrate.
_position_from_usage = position_from_usage
