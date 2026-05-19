"""Enrichment passes run against the lexicon DB.

Each module under this subpackage is one logical enrichment family,
operating on rows produced by the mining pipeline. They're idempotent
by design — re-running on already-enriched state is either a no-op
or replaces the prior output. ``clear_enrichment`` lets operators
revert any single stage and re-run with a new heuristic.

* ``lemma_linking`` — link inflected etymons to their canonical
  lemma via INFLECTION_RULES (suffix-stripping) + MUTATION_RULES
  (Celtic initial mutations). D8.
* ``reverse_search`` — body-text scan for canonical-form occurrences;
  populates ``etymon_text_match``. D12.
* ``fuzzy_search`` — Levenshtein-1 matching for spelling variants
  beyond exact canonical forms. Builds on reverse_search. D15.
* ``ocr_cluster`` — non-destructive OCR variant clustering via
  ``merged_into_id``. D22.
* ``clear`` — selective revert of any enrichment stage. D22.

This file is a thin re-export surface for backward compatibility
with the historic ``from wyrd.generators.kenning.lexicon import
link_lemmas`` shape; the canonical import path going forward is
``from wyrd.generators.kenning.lexicon.enrichment.<pass> import …``.
"""

from wyrd.generators.kenning.lexicon.enrichment.clear import clear_enrichment
from wyrd.generators.kenning.lexicon.enrichment.fuzzy_search import (
    fuzzy_search_attestations,
    levenshtein,
)
from wyrd.generators.kenning.lexicon.enrichment.lemma_linking import (
    INFLECTION_RULES,
    MUTATION_RULES,
    derive_lemma_candidate,
    derive_lemma_candidates,
    derive_mutation_lemma_candidate,
    link_lemmas,
)
from wyrd.generators.kenning.lexicon.enrichment.ocr_cluster import cluster_ocr_variants
from wyrd.generators.kenning.lexicon.enrichment.reverse_search import (
    annotate_fragments_with_corpus_evidence,
    reverse_search_attestations,
)

__all__ = [
    "INFLECTION_RULES",
    "MUTATION_RULES",
    "annotate_fragments_with_corpus_evidence",
    "clear_enrichment",
    "cluster_ocr_variants",
    "derive_lemma_candidate",
    "derive_lemma_candidates",
    "derive_mutation_lemma_candidate",
    "fuzzy_search_attestations",
    "levenshtein",
    "link_lemmas",
    "reverse_search_attestations",
]
