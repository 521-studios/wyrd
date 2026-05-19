"""Data classes + module constants for the language-quality dashboard.

Companion to ``audits.py`` (computation) and ``reporting.py`` (output).
This module owns the pure-data definitions that don't touch the
lexicon DB or the bundle:

* ``ERA_CHAINS`` — per-language historical chain (parent / child
  cells) for tier 2 forward/back era-reflex metrics. Terminus-back
  (no parent) + terminus-forward (no child) markers prevent unfair
  zero-penalties on chain endpoints.
* ``DEFAULT_LANGUAGES`` — languages the dashboard reports on. Each
  is verified at query time; languages with zero etymons drop out.
* ``REFERENCE_TAG_COUNT`` + ``FALLBACK_REFERENCE_TAGS`` — top-N
  English semantic-tag distribution used as the reference profile
  for the C-metric (semantic-tag coverage).
* ``_GRANDFATHER_CITATION_SOURCES`` — source-IDs that admit a family
  root via paths OTHER than scholarly witness consensus (rando-port
  legacy seed). Used by ``_bundle_attestation_breakdown`` to bin
  shipped subjects.
* ``_BUNDLE_LANG_KEY`` — per-language map from dashboard-language
  name to the bundle sibling key (mirrors ``_LANG_CODE_TO_JSON_FIELD``
  in lexicon).
* ``_SHARED_BUNDLE_SIBLINGS`` — bundle-sibling -> [list of dashboard
  languages] map so the markdown can render a "[shared with X, Y, Z]"
  annotation rather than mis-implying per-language exclusivity.

Two dataclasses:

* ``LanguageScorecard`` — per-language metric snapshot.
* ``LanguageQualityReport`` — wrapper around the per-language
  scorecards + the reference profile snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- era chain config -----------------------------------------------------
#
# Per-language historical chain. ``older=[]`` means terminus-back (no
# parent cell in the corpus by construction — proto-languages); ``younger=[]``
# means terminus-forward (modern-language leaves). Used by tier 2
# forward/back metrics so a language with no parent cell doesn't get
# penalized for 0% backward linkage.
#
# Languages absent from this map fall back to "no chain known" — both
# directions reported as N/A in the scorecard.

ERA_CHAINS: dict[str, dict[str, list[str]]] = {
    # English chain
    "old-english": {"older": [], "younger": ["middle-english"]},
    "middle-english": {
        "older": ["old-english"],
        "younger": ["early-modern-english", "modern-english"],
    },
    "early-modern-english": {
        "older": ["middle-english"],
        "younger": ["modern-english"],
    },
    "modern-english": {
        "older": ["early-modern-english", "middle-english"],
        "younger": [],
    },
    # Welsh chain
    "old-welsh": {
        "older": [],
        "younger": ["middle-welsh", "modern-welsh", "welsh"],
    },
    "middle-welsh": {"older": ["old-welsh"], "younger": ["modern-welsh", "welsh"]},
    "modern-welsh": {"older": ["middle-welsh", "old-welsh"], "younger": []},
    "welsh": {"older": ["middle-welsh", "old-welsh"], "younger": []},
    # Goidelic chain
    "old-irish": {
        "older": [],
        "younger": ["middle-irish", "irish", "scottish-gaelic"],
    },
    "middle-irish": {
        "older": ["old-irish"],
        "younger": ["irish", "scottish-gaelic"],
    },
    "irish": {"older": ["middle-irish", "old-irish"], "younger": []},
    "scottish-gaelic": {"older": ["middle-irish", "old-irish"], "younger": []},
    # Brythonic Breton
    "breton": {"older": ["middle-breton", "old-breton"], "younger": []},
    # Norse / Germanic
    "old-norse": {
        "older": ["proto-germanic", "proto-norse"],
        "younger": ["icelandic", "faroese", "norwegian", "swedish", "danish"],
    },
    "icelandic": {"older": ["old-norse"], "younger": []},
    # Romance
    "latin": {"older": [], "younger": ["old-french", "norman-french", "french"]},
    "old-french": {
        "older": ["latin"],
        "younger": ["norman-french", "french", "modern-french"],
    },
    "norman-french": {"older": ["old-french", "latin"], "younger": []},
    # rando-port meta-buckets (carried in the legacy seed)
    "celtic": {
        "older": [],
        "younger": ["welsh", "irish", "scottish-gaelic", "breton"],
    },
}


# Default languages the dashboard reports on. Each is verified at query
# time — languages with zero etymons in the DB are dropped from the
# output rather than reported as zeros across every axis.
DEFAULT_LANGUAGES: tuple[str, ...] = (
    "modern-english",
    "middle-english",
    "old-english",
    "modern-welsh",
    "welsh",
    "old-welsh",
    "irish",
    "old-irish",
    "scottish-gaelic",
    "breton",
    "old-norse",
    "old-french",
    "norman-french",
    "latin",
    "celtic",
)


# Number of top English tags to use as the reference semantic profile.
# Read from english_proportions.json's tag_marginal at runtime; this is
# a fallback if the proportions file isn't reachable.
REFERENCE_TAG_COUNT: int = 15

FALLBACK_REFERENCE_TAGS: tuple[str, ...] = (
    "architecture",
    "topography",
    "social",
    "descriptive",
    "plant",
    "agriculture",
    "water",
    "animal",
    "measurement",
    "geology",
    "religious",
    "tree",
    "color",
    "size",
    "domestic",
)


# wyrd-lc94: source-IDs that admit a family root via paths OTHER than
# scholarly witness consensus. The bundle attestation breakdown bins
# each shipped subject by which path it relied on so the dashboard can
# surface the actionable runtime-quality signal alongside the existing
# DB-level metrics. See ``_bundle_attestation_breakdown`` for the
# classification rules.
_GRANDFATHER_CITATION_SOURCES: frozenset[str] = frozenset({"rando-port"})
_EMPIRICAL_CITATION_SOURCES: frozenset[str] = frozenset({"wiktionary-empirical"})


# --- dataclasses ----------------------------------------------------------


@dataclass
class LanguageScorecard:
    """Metrics for one language. Values are absolute counts plus rate
    fields where a rate is meaningful. Era-chain fields surface the
    structural endpoint markers so consumers can render N/A for
    terminal-at-this-end axes.

    Schema field-order matches the rendered markdown sections (A → I)
    so JSON snapshots stay self-explanatory."""

    language: str
    # A. Corpus depth (raw + generator-eligible — wyrd-bfv1)
    #
    # ``total_etymons`` / ``total_lemmas`` are the raw DB-side counts
    # (dominated by the wyrd-dxu2 Kaikki wiktextract ingest on
    # modern-english — 1.4M rows that are mostly common-vocabulary
    # mining material the kenning generator never uses).
    # ``eligible_etymons`` / ``eligible_lemmas`` restrict to the
    # generator-eligible set (cited + 1-hop descent + lemma rollup +
    # fantasy-tagged) and drive every coverage-rate calculation. See
    # ``populate_eligible_etymon_table`` for the eligibility definition.
    total_etymons: int
    total_lemmas: int
    eligible_etymons: int = 0
    eligible_lemmas: int = 0
    promotion_threshold: int = 0
    promotion_eligible: int = 0
    avg_witnesses: float = 0.0
    source_count: int = 0
    # B. Bundle representation
    bundle_sibling: str | None = None
    bundle_word_count: int = 0
    bundle_shared_with: list[str] = field(default_factory=list)
    # C. Semantic-profile coverage (vs English reference)
    reference_tags: list[str] = field(default_factory=list)
    # Lexicon-side: counts of etymons in this language tagged with each
    # reference tag. Reads etymon_tag directly so middle-english /
    # modern-english (no bundle sibling) still get a meaningful reading.
    lexicon_tagged_etymons: int = 0
    lexicon_tag_coverage: dict[str, int] = field(default_factory=dict)
    lexicon_tag_hit_rate: float = 0.0
    # Bundle-side: counts of bundle subjects whose modifier_tags include
    # the reference tag AND whose words contain a sibling for this
    # language. Captures the runtime-visibility view; bundle-less
    # languages (middle-english, norman-french) read 0 across the board.
    bundle_tag_coverage: dict[str, int] = field(default_factory=dict)
    bundle_tag_hit_rate: float = 0.0
    # D. Inflection coverage
    lemmas_with_inflections: int = 0
    inflection_density: float = 0.0
    distinct_inflection_labels: int = 0
    # E. Spelling-variant coverage
    etymons_with_variants: int = 0
    variant_density: float = 0.0
    # F. Era-reflex coverage
    chain_older: list[str] = field(default_factory=list)
    chain_younger: list[str] = field(default_factory=list)
    chain_known: bool = False
    tier1_cognate_count: int = 0
    tier2_forward_count: int = 0
    tier2_back_count: int = 0
    tier3_period_form_count: int = 0
    # F (Tier 4). Phonology-rule fallback (wyrd-98cs, audit wyrd-pmne).
    # Etymons whose canonical_form would produce a transformed reflex
    # for at least one other era in the same phonology chain when
    # routed through ``phonology_rules.apply_rules`` via
    # ``phonology_rules.rule_form``. Not persisted in DB tables — synthesized
    # at bundle-export / runtime — so this count is computed by walking
    # the rule chain over every etymon in a chain-eligible language at
    # report time. Languages without a registered phonology chain
    # (Goidelic, Norse, Romance, etc.) carry ``-1`` to distinguish 'no
    # rules registered' from '0 etymons fire'.
    tier4_phonology_count: int = -1
    any_reflex_count: int = 0
    any_reflex_rate: float = 0.0
    # F₂. Bundle-side era-reflex coverage (wyrd-h5it). The lex-side F
    # above counts DB-persisted reflex evidence (T1 cognate / T2 descent
    # / T3 period-form) over a denominator of every etymon in the
    # language — for languages with bulk wiktextract ingest like
    # modern-english (1.4M etymons, mostly common-noun headwords not in
    # the toponymy descent graph), that denominator drowns the
    # toponym-relevant signal. F₂ counts bundle subjects-with-sibling
    # whose era_reflexes map carries an entry for THIS language as a
    # target — including the wyrd-98cs phonology-rule:v1 reflexes that
    # don't appear in DB tables. Operationally the more useful 'can the
    # SPA render this language' signal. Compare with lex-side any-reflex
    # rate to see deploy-ready vs DB-persistence reading.
    bundle_subjects_with_reflex: int = 0
    bundle_reflex_rate: float = 0.0
    # G. Stratum coverage
    stratum_classified: int = 0
    stratum_density: float = 0.0
    # H. Pronunciation / IPA coverage (wyrd-dxu2)
    # Two parallel views, mirroring the DB-vs-bundle split used for tag
    # coverage (C / C₂) and attestation (B₂):
    #   * Lexicon side: how many etymons in this language have a non-null
    #     ``pronunciation_ipa`` column. Surfaces the DB-curation signal
    #     ('how much of the lexicon has IPA at all').
    #   * Bundle side: how many shipped subjects in this language have a
    #     non-empty ``*_pronunciation`` slot. Surfaces the runtime-quality
    #     signal ('what % of what we ship can be rendered with IPA in
    #     the SPA's etymological-provenance panel').
    # The two ARE NOT THE SAME — the bundle-side number can include IPA
    # inherited from cluster-mate languages (e.g. modern_english forms
    # that share a surface with a middle-english member with IPA), so
    # the bundle figure overstates true per-language coverage. Both are
    # useful: the DB number is the cleanest 'do we have IPA for this
    # language?' signal, the bundle number is what users actually see.
    lexicon_etymons_with_ipa: int = 0
    lexicon_ipa_density: float = 0.0
    bundle_subjects_with_ipa: int = 0
    bundle_ipa_rate: float = 0.0
    # I. Citation depth
    etymons_with_citations: int = 0
    avg_citations: float = 0.0
    # J. Bundle attestation breakdown (wyrd-lc94)
    # Per-subject classification of how each shipped subject containing
    # this language reached the bundle. Surfaces the runtime-quality
    # signal that DB-level corpus-depth metrics don't. See
    # ``_bundle_attestation_breakdown`` for classification rules.
    bundle_attestation_total: int = 0
    bundle_scholar_attested: int = 0
    bundle_empirical_only: int = 0
    bundle_rando_only: int = 0
    bundle_uncited: int = 0
    bundle_scholar_attestation_rate: float = 0.0
    # K. Rando-port grandfather audit (wyrd-vn09)
    # Family-level (etymon-rollup) classification of which morpheme
    # families ride solely on the rando-port legacy seed for admission
    # vs which are corroborated through the cluster. Distinct from J's
    # bundle-attestation reading because that one operated on the
    # bundle's per-word citation list, which propagates scholar
    # attestation across cluster siblings. THIS metric operates on the
    # raw etymon_citation table at the family level — the cleanest
    # answer to 'how many morphemes would lose their bundle slot if we
    # turned off include_rando=True'.
    #
    # ``rando_pure_grandfather_families``: families where every cited
    # etymon's only citation is 'rando-port'. Curation candidates.
    # ``rando_mixed_grandfather_families``: families with rando-port +
    # scholar/empirical on some sibling. Already corroborated through
    # the cluster — safe to keep, useful as a trend indicator (count
    # should drift down as mining propagates).
    # ``rando_total_cited_families``: denominator (families whose root
    # is in this language, with at least one citation anywhere).
    rando_pure_grandfather_families: int = 0
    rando_mixed_grandfather_families: int = 0
    rando_total_cited_families: int = 0
    rando_pure_grandfather_rate: float = 0.0


@dataclass
class LanguageQualityReport:
    """Top-level report — one scorecard per language plus the reference
    profile used to compute metric C, plus a generation timestamp.
    JSON-serializable via ``asdict``."""

    schema_version: str
    generated_at: str
    reference_tags: list[str]
    bundle_total_words: int
    languages: list[LanguageScorecard] = field(default_factory=list)
    # wyrd-5ecx: active slice filter (source_id or tag) propagated
    # into the rendered markdown header so the operator can tell
    # which subset of the corpus is being measured. None = full
    # generator-eligible set (cited any source + fantasy).
    source_filter: str | None = None
    tag_filter: str | None = None


_BUNDLE_LANG_KEY: dict[str, str | None] = {
    "old-english": "old_english",
    # ME / EModE / Scots route to the modern_english sibling at export
    # per _LANG_CODE_TO_JSON_FIELD in lexicon.py; the dashboard map
    # mirrors that so bundle-side tag visibility reads correctly.
    # Originally mapped to None in PR #144, which mis-reported
    # middle-english at 0/15 bundle tag hits despite 13/15 lex tag
    # hits — fixed alongside wyrd-i1s1's cluster-mate tag rollup.
    "middle-english": "modern_english",
    "early-modern-english": "modern_english",
    "modern-english": "modern_english",
    "scots": "modern_english",
    "old-welsh": "celtic_mix",
    "welsh": "celtic_mix",
    "modern-welsh": "celtic_mix",
    "middle-welsh": "celtic_mix",
    "irish": "celtic_mix",
    "old-irish": "celtic_mix",
    "middle-irish": "celtic_mix",
    "scottish-gaelic": "celtic_mix",
    "breton": "celtic_mix",
    "old-norse": "old_scandinavian",
    "icelandic": "old_scandinavian",
    # OF / Anglo-Norman / NF / Middle French route to old_french per
    # export-side map.
    "old-french": "old_french",
    "norman-french": "old_french",
    "anglo-norman": "old_french",
    "middle-french": "old_french",
    "latin": "latin",
    "biblical": "biblical",
    "celtic": "celtic_mix",
    "germanic": "germanic",
    "greek": "greek",
}


# Languages that share a bundle sibling with siblings of the same family.
# Used to render a "[shared with X, Y, Z]" annotation in the markdown
# so a per-language bundle count of 1200 isn't misread as exclusive
# (Welsh AND Irish AND Breton all see the same celtic_mix count).
_SHARED_BUNDLE_SIBLINGS: dict[str, list[str]] = {
    "celtic_mix": [
        "welsh",
        "old-welsh",
        "modern-welsh",
        "middle-welsh",
        "irish",
        "old-irish",
        "middle-irish",
        "scottish-gaelic",
        "breton",
        "celtic",
    ],
    "old_scandinavian": ["old-norse", "icelandic"],
    # ME / EModE / ModE / Scots share the modern_english sibling; the
    # dashboard surfaces the sharing so per-language bundle counts
    # aren't read as exclusive.
    "modern_english": [
        "middle-english",
        "early-modern-english",
        "modern-english",
        "scots",
    ],
    # OF / NF / Anglo-Norman / Middle French share old_french.
    "old_french": ["old-french", "norman-french", "anglo-norman", "middle-french"],
}
