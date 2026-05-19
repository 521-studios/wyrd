"""The kenning ``language_quality/`` subpackage — dashboard scorecards
+ reporting (wyrd-wzwa Phase 1, parent epic wyrd-eni4).

Split out of the original 1,997-LOC ``language_quality.py`` flat
module into a per-area subpackage:

* ``models.py`` — dataclasses + module constants (``LanguageScorecard``,
  ``LanguageQualityReport``, ``ERA_CHAINS``, ``DEFAULT_LANGUAGES``,
  ``REFERENCE_TAG_COUNT``, ``FALLBACK_REFERENCE_TAGS``,
  ``_BUNDLE_LANG_KEY``, ``_SHARED_BUNDLE_SIBLINGS``,
  ``_GRANDFATHER_CITATION_SOURCES``).
* ``audits.py`` — per-metric DB + bundle audit helpers, the
  ``populate_eligible_etymon_table`` setup pass, and the
  ``compute_scorecard`` / ``compute_report`` /
  ``compute_rando_port_grandfather_audit`` orchestrators.
* ``reporting.py`` — ``report_to_json`` + ``report_to_markdown`` and
  the small private format helpers.

Back-compat: this package re-exports every name that was previously
importable from ``wyrd.generators.kenning.language_quality`` — both
the public API (``compute_report``, ``compute_scorecard``,
``LanguageScorecard``, ``LanguageQualityReport``, ``DEFAULT_LANGUAGES``,
``load_reference_tags``, ``report_to_json``, ``report_to_markdown``,
``populate_eligible_etymon_table``,
``compute_rando_port_grandfather_audit``) and the underscore-prefixed
internals that tests pin (``_BUNDLE_LANG_KEY``,
``_bundle_attestation_breakdown``, ``_bundle_era_reflex_coverage``,
``_bundle_ipa_coverage``, ``_bundle_language_word_count``,
``_bundle_tag_coverage_for_sibling``, ``_bundle_total_words``,
``_lexicon_ipa_coverage``, ``_tier4_phonology_coverage``).

External callers (``bundle/rando_port_readiness.py`` uses
``_bundle_attestation_breakdown``; ``cli/lexicon/language_report.py``
uses the public orchestrators) keep working unchanged.
"""

from .audits import (  # noqa: F401
    _bundle_attestation_breakdown,
    _bundle_era_reflex_coverage,
    _bundle_ipa_coverage,
    _bundle_language_word_count,
    _bundle_subjects,
    _bundle_tag_coverage_for_sibling,
    _bundle_total_words,
    _citation_depth,
    _corpus_depth,
    _ensure_eligible_etymon_table,
    _era_reflex_coverage,
    _inflection_coverage,
    _lexicon_ipa_coverage,
    _lexicon_tag_coverage,
    _stratum_coverage,
    _tier4_phonology_coverage,
    _variant_coverage,
    compute_rando_port_grandfather_audit,
    compute_report,
    compute_scorecard,
    load_reference_tags,
    populate_eligible_etymon_table,
)
from .models import (  # noqa: F401
    _BUNDLE_LANG_KEY,
    _GRANDFATHER_CITATION_SOURCES,
    _SHARED_BUNDLE_SIBLINGS,
    DEFAULT_LANGUAGES,
    ERA_CHAINS,
    FALLBACK_REFERENCE_TAGS,
    REFERENCE_TAG_COUNT,
    LanguageQualityReport,
    LanguageScorecard,
)
from .reporting import (  # noqa: F401
    report_to_json,
    report_to_markdown,
)
