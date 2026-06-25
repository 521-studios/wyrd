"""Morpheme model: a Meaning is a piece of a word like 'Bre-' or '-combe'.

Ported from Rando (rando/meaning.py) — Python 2 → 3.

Variants (D18) and inflections (D8) ride along on the Meaning so
generation can optionally emit an attested archaic spelling or an
inflected morphological form instead of the modern reflex.
"""

from __future__ import annotations

import json
import random

from wyrd.generators.kenning.vectors.schemas import _DIMENSION_NAMES, PhonologicalVector

# wyrd-eyjk/D40: tags that mark a morpheme as a proper noun. The QUALIFYING
# tags — {male/female/family name} (via is_name) and {saint} (via is_saint) —
# gate is_pure_proper_noun; {personal-name, religious} are COMPANION tags that
# only keep a row "pure" alongside a qualifying one (so a synthesized saint
# subject tagged {saint, personal-name, religious} reads as pure). Single
# source of truth — __init__._is_pure_proper_noun delegates to the method below.
_PROPER_NOUN_TAGS = frozenset(
    {"male name", "female name", "family name", "saint", "personal-name", "religious"}
)

# wyrd-gwj3: connector SURFACES that require a complement and are never a valid
# standalone English place-name word — the Latin/French toponym connectors
# (`X cum Y`, `Chester-le-Street`, `Stoke sur Y`, `Stoke juxta Y`). They leaked
# into the morpheme pool as untagged etymons (decomposed from real `X cum Y`
# toponyms), so they can't be filtered by tag — matched by bare surface instead.
# Excluded from the base generation pool so they don't dangle as a lone word
# (`Newton Cum`). Their legitimate home is joiner-insertion
# (`_apply_joiner_insertion`), which draws from a separate joiner pool — not the
# base pool — and these connector surfaces are not yet wired into that pool.
# Conservative set: only unambiguous connectors. NOT included: `magna` /
# `parva` (real standalone qualifiers — `Wigston Magna`), and `saint` / `st`
# (the dedication particle has a legitimate saint-SLOT mechanism — `Saint
# <name>` — so excluding it would break valid dedications; the rarer trailing-
# `St` dangle + `Saint <common-noun>` oddities are a separate saint-slot-quality
# concern).
_CONNECTOR_PARTICLE_SURFACES = frozenset({"cum", "le", "la", "sur", "sous", "juxta"})

# Suffix used in meanings.json to mark per-language variant pools, e.g.
# "old_english" canonical forms have their variants in "old_english_variants".
# Matches the export-meanings emit shape in lexicon.py:_emit_variant_list.
_VARIANT_SUFFIX = "_variants"

# Suffix used for per-language inflection metadata (D8). Each entry is
# (form, inflection_label) where inflection_label is a grammatical case
# string like "dative_or_pl", "genitive_strong". Lemmas don't appear here —
# they're the unmarked headword.
_INFLECTION_SUFFIX = "_inflections"

# Suffix used for per-language scholarly attribution (wyrd-9kh.1). Each
# entry is a list of source_id strings (e.g., 'mawer_1920_northumberland_durham')
# sorted alphabetically. Surfaces in the runtime explainer so a GM can see
# which scholars attest each morpheme.
_CITATIONS_SUFFIX = "_citations"

# Suffix used for per-language attested-year metadata (D5-1 / wyrd-bag).
# Each entry is (form, year) where year is the earliest plausibly-
# attested year for the form on the production corpus, sorted by year
# ascending. Runtime consumer: D46's record gate (``record_start`` —
# the earliest date can vouch a morpheme OLDER than its language stage
# suggests). The D5-2 attested-inside-window filter that originally
# consumed it was retired by D44.
_ATTESTED_YEARS_SUFFIX = "_attested_years"

# Suffix used for per-language english_shaped renderings (wyrd-ha9q
# Phase 2c). Each entry is (form, english_shaped) where english_shaped
# is the diacritic-stripped / known-form-overridden Latin rendering of
# the canonical form (e.g. Hebrew גולם → 'golem', Arabic عفريت → 'ifrit',
# Sanskrit रक्षस → 'rakshasa'). Sparse — only non-Latin source-language
# rows whose wyrd-ha9q derive_english_shaped produced a non-None value
# land here. Latin-script source langs (old-english / latin / welsh /
# proto-germanic / etc.) don't carry this field; canonical_form is
# already English-readable for those.
_ENGLISH_SHAPED_SUFFIX = "_english_shaped"

# Suffix used for per-language native-script renderings (wyrd-ha9q
# Phase 2a → exposed to the runtime in wyrd-qhs0 Phase 2d). Each entry
# is (form, original_script) where original_script is the vocalized
# native-script form from wiktextract head_templates' `wv` / `head`
# arg (Hebrew niqqud, Arabic harakat, Egyptian `<hiero>`-tagged
# markup). Distinct from canonical_form: canonical_form for non-Latin
# rows is whatever wiktextract's `entry.word` was (often unvocalized
# native script); original_script carries the explicitly-vocalized
# form when wiktextract supplies one. Empty / absent for Latin-script
# source langs.
_ORIGINAL_SCRIPT_SUFFIX = "_original_script"

# Suffix used for per-language academic transliterations (wyrd-ha9q
# Phase 2a). Each entry is (form, transliteration) where the value is
# the academic Latin-script form with diacritics from wiktextract
# head_templates' `tr` arg (e.g. 'ʿifrīt', 'rakṣasa', 'kɛ́lɛḇ'). The
# english_shaped column strips diacritics off this value (or applies
# a known-form override) for English-friendly rendering;
# transliteration is the full academic-register version.
_TRANSLITERATION_SUFFIX = "_transliteration"

# Suffix used for per-language IPA + dialect pairs (wyrd-ha9q Phase
# 2a). Each entry is (form, ipa, dialect) where ipa is the most
# authoritative IPA string from wiktextract's `sounds[*]` array and
# dialect is the first tag on the chosen sound entry (Biblical-Hebrew,
# Modern Standard Arabic, Yemenite-Hebrew, ...). The pair updates
# atomically — see lexicon.py.upsert_etymon's CASE expression for the
# integrity rule.
_PRONUNCIATION_SUFFIX = "_pronunciation"

# Suffix used for per-language within-language stratum tags (wyrd-lr4
# Phase 2). Each entry is (form, stratum) where stratum is a
# language-specific register bucket — for Welsh: 'latin-loan' /
# 'english-loan' / 'brittonic-substrate' / 'medieval-welsh' /
# 'native-welsh'. Sparse — only languages with a Phase 1 classifier
# populate this (Welsh-family today; French / OE / ON follow). Empty
# / absent for older bundles and unclassified languages. Phase 3
# wires the generator filter; Phase 2 just plumbs the data through so
# the bundle carries it.
_STRATUM_SUFFIX = "_stratum"
# wyrd-ecjp.10a: per-language phonological_vector sibling per the D26
# pattern. Each entry is {"form": canonical_form,
# "phonological_vector": {<dim>: <float>, ...}}. Sparse — only forms
# whose etymons have been kq7w.1-enriched populate this.
_PHONOLOGICAL_VECTOR_SUFFIX = "_phonological_vector"

# D46 (wyrd-6ah2): the year each source-language LAYER enters the British
# place-name record — the stage half of the era accretion gate's
# "in the record by then" rule. None = founding stratum (Celtic substrate,
# OE, Latin, the ancient pack/fantasy languages): predates every era we
# model, passes everything. Contact layers carry their historical entry
# point into BRITISH naming (not the language's own birth — Old Norse and
# Norman French existed earlier elsewhere; what's gated is when they could
# plausibly appear in a name on this island): Old Norse with the Danelaw
# (~865, rounded to the oe-late cell boundary), French layers with the
# Conquest. English stages use their D5 era-cell starts. Languages absent
# from this map default to None (pass) — thin data errs toward inclusion;
# the gate's hard exclusions are positively-evidenced late arrivals.
_LANG_RECORD_ENTRY: dict[str, int | None] = {
    "old_english": None,
    "celtic_mix": None,
    "latin": None,
    "germanic": None,
    "akkadian": None,
    "biblical": None,
    "egyptian": None,
    "hebrew": None,
    "arabic": None,
    "persian": None,
    "sanskrit": None,
    "aramaic": None,
    "armenian": None,
    "old_scandinavian": 800,
    "old_french": 1066,
    "norman_french": 1066,
    "middle_english": 1100,
    "modern_english": 1500,
}


class Joiner:
    """A consumed phonological joiner (wyrd-q0g6 Phase 1).

    A Joiner is NEITHER ``str`` NOR ``Meaning`` — that's the load-
    bearing discriminator. Code walking decompositions on the
    str-vs-Meaning axis (``count_unaccounted``, ``has_name``,
    ``get_structure``, ``_decomposition_score``) treats Joiners as
    'matched-but-no-semantic': not counted as unaccounted (non-str),
    not counted as semantic structure (non-Meaning). ``__str__``
    returns the surface so ``Word.__str__`` keeps the joiner chars
    in the rendered output.

    The morpheme_count line of ``_decomposition_score`` does count
    Joiners (they're non-str), giving a tiny Occam preference for
    no-joiner parses on otherwise-tied scores — the desired bias.
    """

    __slots__ = ("surface", "lang_field")

    def __init__(self, surface: str, lang_field: str | None = None):
        self.surface = surface
        self.lang_field = lang_field

    def __str__(self) -> str:
        return self.surface

    def __repr__(self) -> str:
        return f"Joiner({self.surface!r}, lang_field={self.lang_field!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Joiner)
            and self.surface == other.surface
            and self.lang_field == other.lang_field
        )

    def __hash__(self) -> int:
        return hash(("joiner", self.surface, self.lang_field))


class Meaning:
    def __init__(
        self,
        usage,
        tags,
        meanings,
        sources,
        variants=None,
        inflections=None,
        citations=None,
        attested_years=None,
        english_shaped=None,
        original_script=None,
        transliteration=None,
        pronunciation=None,
        stratum=None,
        era_reflexes=None,
        era_reflex_glosses=None,
        phonological_vector=None,
        phonological_vectors=None,
        morpheme_id=None,
    ):
        self.usage = usage
        # wyrd-rogd.10 Phase 2: the owning morpheme's content id (or None for
        # bundles predating Phase 1 / un-attributable words). Lets the runtime
        # resolve the era-grid against the unified morpheme record by id rather
        # than the per-connective-surface Meaning.
        self.morpheme_id = morpheme_id
        self.tags = tags
        self.meanings = meanings
        self.sources = sources
        # variants is a dict[lang_field, list[(form, weight)]] keyed by the
        # JSON field name (e.g. "old_english", "celtic_mix"). Optional —
        # legacy meanings.json data has no variants; the field is empty.
        self.variants = variants or {}
        # inflections is a dict[lang_field, list[(form, inflection_label)]],
        # carrying grammatical case data for inflected forms in the family.
        # The lemma form does NOT appear here.
        self.inflections = inflections or {}
        # citations is a dict[lang_field, list[source_id]] — distinct sorted
        # source_ids per language attesting the morpheme. The runtime
        # explainer surfaces these so a GM can see which scholars cite each
        # morpheme. Empty for entries from rando-port-only sources or older
        # bundles that pre-date the wyrd-9kh.1 citation field.
        self.citations = citations or {}
        # attested_years is a dict[lang_field, list[(form, year)]] sorted
        # by ascending year — D5-1 / wyrd-bag. Feeds D46's record gate
        # via record_start (dates vouch age); the original D5-2
        # attested-inside-window filter was retired by D44.
        self.attested_years = attested_years or {}
        # english_shaped is a dict[lang_field, dict[canonical_form,
        # english_shaped_form]] — wyrd-ha9q Phase 2c. Maps the
        # native-script / diacritic-bearing canonical form to its
        # English-friendly rendering for non-Latin source-language
        # families (Hebrew, Arabic, Persian, Sanskrit, Akkadian, Egyptian,
        # Aramaic + their precursors). Empty for Latin-script langs and
        # for older bundles that pre-date the wyrd-ha9q export.
        self.english_shaped = english_shaped or {}
        # wyrd-qhs0 Phase 2d: the OTHER three wyrd-ha9q rendering
        # columns surfaced for the SPA panel.
        # original_script: dict[lang_field, dict[canonical_form, str]]
        #   — vocalized native-script form (Hebrew with niqqud, Arabic
        #   with harakat, Egyptian `<hiero>`-tagged markup).
        # transliteration: dict[lang_field, dict[canonical_form, str]]
        #   — academic Latin-script form with diacritics (ʿifrīt,
        #   rakṣasa, kɛ́lɛḇ).
        # pronunciation: dict[lang_field, dict[canonical_form, dict]]
        #   where the inner dict has 'ipa' and 'dialect' keys. The
        #   {ipa, dialect} pair (rather than two parallel attrs) is
        #   load-bearing: wiktextract associates each IPA string with
        #   the dialect tag that was on the same `sounds[*]` entry,
        #   and wyrd-ha9q Phase 2a's upsert_etymon CASE expression
        #   updates the two columns atomically so the dialect tag
        #   never describes an IPA we don't store.
        # All three are empty for Latin-script source langs and older
        # bundles.
        self.original_script = original_script or {}
        self.transliteration = transliteration or {}
        self.pronunciation = pronunciation or {}
        # wyrd-lr4 Phase 2: stratum is a dict[lang_field, dict[
        # canonical_form, stratum_tag]] — the within-language register
        # bucket each form belongs to (Welsh: 'native-welsh',
        # 'latin-loan', etc.). Phase 2 plumbs the data; the Phase 3
        # generator filter consumes it. Empty for Latin-script langs
        # without a Phase 1 classifier, for unclassified rows in
        # classified languages, and for legacy bundles.
        self.stratum = stratum or {}
        # wyrd-obpw Phase 3.3 + wyrd-jbcu source tag: per-target-
        # language era reflexes for the family root, computed at
        # bundle-build time. Internal storage is the rich shape
        # ``{target_language: [(form, source), ...]}``; ``source`` is
        # 'cluster' / 'descent' / 'period-form' / 'phonology-rule:v1'.
        # Legacy bundles (list-of-strings entries) load as (form,
        # 'cluster') tuples — the safe back-compat default since
        # everything in the legacy bundle was attestation-backed.
        # The SPA-side rewinder consumes via
        # ``era_reflex_for(target_language)`` (forms only, back-compat)
        # or ``era_reflex_sources_for(target_language)`` (form → source
        # dict). Empty for legacy proto-language roots and untracked
        # classical families.
        self.era_reflexes: dict[str, list[tuple[str, str]]] = era_reflexes or {}
        # wyrd-rogd.1: per-target-language ``{form: gloss}`` — the reflex
        # etymon's OWN representative meaning, carried alongside era_reflexes
        # (separately, so era_reflex_for / era_reflex_sources_for stay
        # back-compatible) so the SPA era-grid can surface semantic DRIFT when
        # a swapped cognate no longer means what the morpheme does. Sparse:
        # only forms whose reflex etymon had a non-pointer gloss appear.
        self.era_reflex_glosses: dict[str, dict[str, str]] = era_reflex_glosses or {}
        # wyrd-ecjp.5 / wyrd-kq7w.1: per-meaning phonological vector
        # consumed by the vector-scoring path's phon_score sub-scorer.
        # None when the bundle doesn't carry the field (legacy bundles
        # pre-kq7w.1, or bundles emitted before wyrd-ecjp.8 wires the
        # column through bundle export). The vector-scoring path treats
        # None as a default-empty PhonologicalVector — phon_score
        # returns 0 in that case and the score falls back to the
        # sem/pos/baseline axes.
        self.phonological_vector = phonological_vector
        # wyrd-ecjp.10a Phase 7: per-(lang, canonical_form) phonological
        # vectors from the <lang>_phonological_vector bundle siblings.
        # Shape: dict[lang_field, dict[canonical_form, PhonologicalVector]].
        # The singular `phonological_vector` above is the "primary"
        # vector (first non-empty in deterministic lang × form walk) for
        # back-compat with callers that read one-vector-per-meaning.
        # Empty dict for legacy bundles that pre-date wyrd-ecjp.10a.
        self.phonological_vectors = phonological_vectors or {}
        self._set_location()

    def _set_location(self):
        # wyrd-aicu: identity is now BARE (the producer de-dashes every stored
        # usage — D45), so a Meaning's usage never carries a position-dash and
        # ``.location`` is always "bare". The field survives as a render/scoring
        # hint (read by realism_reference, proportions, and the explain API), NOT
        # a match-time gate — matching is string-only and position is derived
        # from the span at sample time (wyrd-eyjk/D40). The retired dash-shape →
        # position decode (-x- inner / x- pre / -x post) lived here; position now
        # rides as an explicit axis on the proportions, never in the string.
        self.location = "bare"

    def __str__(self):
        return self.usage.lower().replace("-", "")

    def __repr__(self):
        return (
            f"{{usage: {self.usage}, tags: {self.tags}, meanings: {self.meanings}, "
            f"sources: {self.sources}, location: {self.location}}}"
        )

    def is_name(self):
        return any(tag in ("female name", "male name", "family name") for tag in self.tags)

    def primary_language(self) -> str | None:
        """wyrd-pfoo: the alphabetically-first source-language bundle
        field with at least one non-empty form. Used as the per-Meaning
        identifier across the culture-attestation pipeline — the
        matcher's culture-aligned tiebreaker, the splitter's
        per-Meaning attestation emit, and the vector path's
        per-Meaning eligibility filter all key on this value. Walking
        sources alphabetically keeps the choice deterministic across
        re-runs (sorted, not dict-insertion order).

        The "non-empty form" check matches
        ``vector_name_select._lemma_ref_for``'s semantics — a language
        whose forms array contains only empty strings / None / dict
        entries with no form/canonical_form fields doesn't count as
        attested. Without this alignment, a Meaning with
        ``{"old_english": [""]}`` would surface ``"old_english"`` as
        its primary language through ``primary_language()`` but get
        skipped by ``_lemma_ref_for``'s lemma-ref derivation —
        producing a per-Meaning attestation key that doesn't
        correspond to any usable scoring input downstream.

        Cached on first call via the ``_primary_language_cache``
        attribute. ``sources`` is set once at __init__ and never
        mutated afterwards, so the cached value stays correct. Hot
        path: the splitter calls this O(decompositions ×
        morphemes-per-name) times across the per-culture place-name
        corpus, and the vector path calls it once per pool-eligibility
        check. ``None`` is a valid cached value (meaning lacks any
        usable source-language data), so we use ``hasattr`` rather
        than a None sentinel."""
        if not hasattr(self, "_primary_language_cache"):
            sources = self.sources or {}
            self._primary_language_cache: str | None = None
            for key in sorted(sources):
                if Meaning._first_nonempty_form(sources[key]) is not None:
                    self._primary_language_cache = key
                    break
        return self._primary_language_cache

    def is_saint(self):
        return "saint" in self.tags

    def is_given_name(self):
        """True when tagged as a personal GIVEN name (male/female). Given names
        (`John`, `Mary`, `Edmund`) dangle awkwardly as a bare place-name element
        — a real toponym either composes them (`Edmund`+`-ton`) or marks a
        dedication (`St John`). FAMILY names (`Smith`, Norman manorial families)
        are NOT given names: they form legitimate manorial/toponymic places, so
        they're excluded here. Used (alongside is_saint) by the base-pool
        exclusion (wyrd-g1hj) to keep pure given-name etymons out of plain
        generation."""
        return any(tag in ("male name", "female name") for tag in self.tags)

    def is_connector_particle(self):
        """wyrd-gwj3: True when this morpheme's bare surface is a connector that
        requires a complement (`cum`, `le`, `la`, `sur`, `sous`, `juxta`). These
        are never a valid standalone English place-name word; excluded from the
        base generation pool so they don't dangle as a lone word (`Newton Cum`).
        Matched by SURFACE because they leaked in as untagged etymons (no
        connector tag). The saint particle is deliberately NOT here — see
        ``_CONNECTOR_PARTICLE_SURFACES``."""
        return self.usage.replace("-", "").lower() in _CONNECTOR_PARTICLE_SURFACES

    def is_pure_proper_noun(self):
        """True when this is name/saint-tagged and carries NO common-noun tag
        — a personal name / saint / synthesized saint-subject with no
        place-element sense (``Bourne``, ``Andrew``, the ``Mary``/``John``
        dedication subjects), as opposed to a place element merely co-tagged
        with a name (``stān`` 'stone' + 'Stan'). Such morphemes belong only in
        name/saint slots, never the generic base-generation pool (wyrd-eyjk/D40
        — keeps synthesized saint subjects from leaking into plain names)."""
        if not (self.is_name() or self.is_saint()):
            return False
        return all(tag in _PROPER_NOUN_TAGS for tag in self.tags)

    def is_manorial_subject(self):
        """wyrd-57d8: True for a synthesized Norman manorial-family subject
        (``Malet``, ``Beauchamp``, ``Mandeville`` …), tagged ``manorial`` at
        load by ``__init__._norman_manorial_subjects``. These exist only so
        ``KenningExplain`` can decompose the manorial-affix names Kenning emits
        with ``manorial_affix > 0``; they are NEVER a base-generation morpheme.
        Their sole legitimate entry into output is the dedicated
        ``manorial_affix`` knob, so — like synthesized saint subjects (D40) —
        they're excluded from the base pool (see ``is_base_pool_excluded``)."""
        return "manorial" in self.tags

    def is_base_pool_excluded(self):
        """wyrd-57d8: the request-INDEPENDENT class exclusions from base
        generation, in one place so every base pool applies them identically —
        the scored native pool (``vector_name_select._native_meaning_eligible``),
        the proportions frequency-weight pool
        (``proportions.MeaningGenerator.load_parts``), AND the
        repeat-diversification re-pick
        (``proportions.NewName._collect_repick_pools``). The re-pick previously
        read ``meaning_db`` raw and so re-introduced an excluded morpheme to
        break a ``corner corner`` repeat (the manorial ``Malet`` leak).

        Excluded classes (each belongs only in its dedicated slot, never the
        generic base pool):

        - pure-proper-noun saints / personal given names — reach output only via
          the param-gated St-dedication synthesis (wyrd-eyjk/D40 + wyrd-g1hj);
        - synthesized Norman manorial subjects — reach output only via the
          ``manorial_affix`` knob (wyrd-57d8);
        - connector particles (``cum`` / ``le`` / ``juxta`` …) — require a
          complement; their home is joiner insertion (wyrd-gwj3).

        Family-name etymons that carry a real place sense are NOT excluded here
        (they form legitimate manorial places); nor is a place element merely
        co-tagged with a name (`stān`+`Stan` — not a pure proper noun). Only the
        three classes enumerated above are excluded."""
        if self.is_pure_proper_noun() and (self.is_saint() or self.is_given_name()):
            return True
        if self.is_manorial_subject():
            return True
        return self.is_connector_particle()

    def era_reflex_for(self, target_language: str) -> list[str]:
        """wyrd-obpw Phase 3.3: return the cluster-mate forms attested
        for ``target_language`` (e.g. 'middle-english') as a sorted
        list, or [] when the family root has no reflexes at that
        target.

        Drives the SPA-side rewinder: for each anchored morpheme the
        consumer asks ``meaning.era_reflex_for('middle-english')`` to
        get the ME forms without needing the lexicon DB. Empty list
        is the standard 'no era data' signal — caller falls back to
        ``meaning.usage`` per the rewind convention.

        The data is computed at bundle-build time via
        ``etymon_era_reflexes`` against the family root's cognate-
        cluster + descent + period-form + phonology-rule fallback
        tiers (wyrd-98cs added the phonology tier; wyrd-jbcu wired it
        through to the bundle). Source-aware consumers can call
        ``era_reflex_sources_for(target_language)`` to get the
        ``{form: source}`` mapping instead.
        """
        return [form for form, _source in self.era_reflexes.get(target_language, ())]

    def era_reflex_sources_for(self, target_language: str) -> dict[str, str]:
        """wyrd-jbcu: return ``{form: source}`` for ``target_language``,
        where source is 'cluster' (Tier 1, cognate cluster mate),
        'descent' (Tier 2, descent edge), 'period-form' (Tier 3,
        projected period form), or 'phonology-rule:v1' (Tier 4,
        derived via phonology rules). Empty dict when the family root
        has no reflexes for that target.

        Source-aware consumers (KenningRewind / KenningEraMap) can
        render phonology-rule-derived forms differently — e.g. italic
        with a tooltip — so users know the form is inferred rather
        than attested. Source-agnostic consumers can keep using
        ``era_reflex_for`` which returns a forms-only list."""
        return dict(self.era_reflexes.get(target_language, ()))

    def era_reflex_gloss_for(self, target_language: str) -> dict[str, str]:
        """wyrd-rogd.1: return ``{form: gloss}`` for ``target_language`` — the
        reflex etymon's own representative meaning per form. Sparse (only forms
        whose reflex had a non-pointer gloss) and empty for targets with no
        reflexes / no gloss data. The SPA era-grid compares a cell's gloss to
        the morpheme's own to flag semantic drift on a cognate swap."""
        return dict(self.era_reflex_glosses.get(target_language, {}))

    def respelling_for(self, form: str, lang_field: str) -> str | None:
        """wyrd-17t: SAMPA-lite respelling of ``form`` in
        ``lang_field`` for non-modern-English readers.

        Computed via per-language grapheme→phonetic-respelling
        rules in ``wyrd.generators.kenning.runtime.respelling``. Returns
        None when:

        * the language has no respeller (modern English, untracked
          languages — see the ``respelling`` dispatch list),
        * ``form`` is empty.

        Caller pattern: render the respelling next to the canonical
        form so a GM can sound it out at the table::

            Pont-Dwfr   /pont DOO-vr/  (Welsh)

        Pure-rule output, no DB / bundle lookup. Future refinement:
        consult scholar-attested IPA mined from Mawer / Skeat
        bracket-notation (wyrd-17t source #1, deferred) before
        falling through to rules.
        """
        from .respelling import respell

        return respell(form, lang_field)

    def record_start(self) -> int | None:
        """D46 (wyrd-6ah2): the morpheme's earliest evidence of existence in
        the British place-name record — the MOST GENEROUS of two signals:

        * the earliest record-entry year among its source-language layers
          (``_LANG_RECORD_ENTRY``; a founding-stratum layer → None). Only
          layers with at least one non-empty form count — an empty bucket
          like ``{"old_english": []}`` must not vouch a modern coinage into
          every era (same non-empty-form semantics as ``primary_language``).
        * its earliest attested year across ``attested_years`` (a date can
          vouch a morpheme is OLDER than its stage suggests — a
          modern_english bucket with a 1086 Domesday date passes oe-late;
          and a sourceless-but-dated morpheme is gated by its dates).

        Returns None when any populated layer is founding-stratum (or an
        unmapped language — thin data errs toward inclusion), or when there
        is no evidence on either axis (coverage rule: missing data is not
        evidence of lateness). Cached on first call via the hasattr pattern
        (the ``primary_language`` idiom) — sources / attested_years are set
        once at __init__ and never mutated.
        """
        if not hasattr(self, "_record_start_cache"):
            self._record_start_cache: int | None = self._compute_record_start()
        return self._record_start_cache

    def _compute_record_start(self) -> int | None:
        years: list[int] = []
        if isinstance(self.sources, dict):
            for lang, forms in self.sources.items():
                if not self._forms_nonempty(forms):
                    continue
                entry = _LANG_RECORD_ENTRY.get(lang)
                if entry is None:
                    return None  # founding stratum / unmapped: vouches every era
                years.append(entry)
        for attested in self.attested_years.values():
            years.extend(year for _form, year in attested)
        return min(years) if years else None

    @staticmethod
    def _first_nonempty_form(forms: list | None) -> str | None:
        """The first usable form string in ``forms``, or ``None`` if there is
        none. Single source for the "non-empty form" scan shared by
        :meth:`_forms_nonempty`, :meth:`primary_language`, and
        ``vector_name_select._lemma_ref_for``: an empty string / ``None`` / a
        form-less dict entry doesn't count, and a dict entry yields its
        ``form`` / ``canonical_form``. Centralized so the three can't drift —
        a forms[0]-only copy diverging from this all-entries scan was the
        ``_lemma_ref_for`` bug (#788)."""
        if not forms:
            return None
        # A source value is normally a list of form entries, but ``load_meanings``
        # copies the word field verbatim, so loosely-shaped data can hand us a bare
        # string (``{"old_english": "burna"}`` rather than ``["burna"]``). Treat it
        # as the single form — iterating the str would scan its CHARACTERS and yield
        # the first letter (the old forms[0]-style copies returned "b" for "burna").
        if isinstance(forms, str):
            return forms  # non-empty (guarded above)
        for entry in forms:
            if isinstance(entry, dict):
                form_str = entry.get("form") or entry.get("canonical_form") or ""
            else:
                form_str = entry
            if form_str:
                return form_str
        return None

    @staticmethod
    def _forms_nonempty(forms: list | None) -> bool:
        """At least one usable form — see :meth:`_first_nonempty_form`."""
        return Meaning._first_nonempty_form(forms) is not None

    def in_stratum(self, stratum: str | None) -> bool:
        """wyrd-lr4 Phase 3 stratum filter: True if any of this
        morpheme's per-form stratum tags match ``stratum``, or if the
        morpheme has no stratum data at all (treated as 'always
        include' — same convention as the era filter for 'no data →
        pass').

        ``stratum`` of ``None`` means 'no filter' → always True.

        The 'no stratum data → pass' rule is deliberate. Only Welsh-
        family etymons are classified today; routing every culture
        through a strict stratum gate would gut bundles for languages
        without a Phase 1 classifier (Latin, Old English, French, ...).
        As Phase 4 lands the rule can tighten per language; for now,
        missing data is not evidence of absence.
        """
        if stratum is None:
            return True
        if not self.stratum:
            return True
        for forms in self.stratum.values():
            for tag in forms.values():
                if tag == stratum:
                    return True
        return False

    def key(self):
        key = [self.location]
        if self.is_name():
            key.append("name")
        elif self.usage.replace("-", "").lower() == "saint":
            key.append("saint")
        return tuple(key)

    def pick_variant(self, rng: random.Random, spelling_variety: float) -> str | None:
        """Optionally pick a variant spelling for this meaning's surface form.

        With probability ``spelling_variety``, draw from this meaning's variant
        pool (across all languages) weighted by the per-variant weights. Returns
        the variant form (raw, lowercase, no dashes). Returns None if the variant
        pool is empty or the random draw didn't trigger.

        Case and dash-marker handling stays at the caller — this method only
        decides which raw form to substitute. The caller mimics the original
        usage's case + position markers when rendering.
        """
        if spelling_variety <= 0 or not self.variants:
            return None
        # Pool is the union of every language's variant list. Weights stay as
        # the per-attestation match_count totals from the lexicon, so a
        # variant attested 37 times across the corpus dominates over one
        # attested only once.
        pool: list[tuple[str, int]] = []
        for forms in self.variants.values():
            for form, weight in forms:
                if weight > 0:
                    pool.append((form, weight))
        if not pool:
            return None
        # Two-stage draw: first decide whether to substitute at all, then
        # which variant to pick. Keeps spelling_variety as a clean
        # "probability of substitution" knob rather than getting tangled up
        # with the within-pool weighting.
        if rng.random() >= spelling_variety:
            return None
        total = sum(w for _, w in pool)
        threshold = rng.uniform(0, total)
        running = 0
        for form, weight in pool:
            running += weight
            if threshold < running:
                return form
        return pool[-1][0]

    def pick_inflection(
        self, rng: random.Random, inflection_density: float
    ) -> tuple[str, str] | None:
        """Optionally pick an inflected form (D8) for this meaning's surface.

        With probability ``inflection_density``, draw a (form, label) pair
        uniformly from this meaning's inflection pool across all languages.
        Returns ``None`` if the pool is empty, the draw didn't trigger, or
        density is 0.

        Returned label is the grammatical case (e.g. "dative_or_pl") so the
        caller can surface it in the explainer breakdown via "<lemma>@<label>".

        Inflections are picked uniformly because mining doesn't track per-
        inflection frequency — we have the labels, not the counts. A future
        refinement could weight by attestation if the data lands.
        """
        if inflection_density <= 0 or not self.inflections:
            return None
        # Fast-path the substitution gate before flattening the pool: if the
        # roll didn't trigger we can return immediately and skip the
        # cross-language flatten. Saves work in the common case where
        # inflection_density is set but most morphemes don't fire.
        if rng.random() >= inflection_density:
            return None
        pool = self._flat_inflections()
        if not pool:
            return None
        return rng.choice(pool)

    def _flat_inflections(self) -> list[tuple[str, str]]:
        """Flatten the per-language inflections dict into a single list,
        cached after first call. The cache is populated lazily because most
        Meanings never reach generation paths that need it (default
        inflection_density=0 short-circuits before this is called)."""
        cached = self.__dict__.get("_flat_inflections_cache")
        if cached is None:
            cached = [entry for entries in self.inflections.values() for entry in entries]
            self._flat_inflections_cache = cached
        return cached


def _mimic_case(template: str, variant: str) -> str:
    """Render ``variant`` using the casing convention of ``template``.

    Place names carry capital letters at meaningful positions: the canonical
    'Bridg-' is title-case at word-start, '-water' is lowercase mid-name.
    Variants are stored raw (mostly lowercase) in the lexicon, so the caller
    projects the original usage's casing onto the variant.

    Conservative on internal casing: a template that starts with a capital
    capitalizes only the variant's first character and leaves the rest
    untouched, preserving any internal capitals (Mc-, O', multi-word).
    A template that's all-lowercase forces the variant lowercase to match.
    """
    bare = template.replace("-", "")
    if not bare:
        return variant
    if bare[:1].isupper():
        return variant[:1].upper() + variant[1:]
    return variant.lower()


def _parse_phon_vector(raw):
    """wyrd-ecjp.5: parse a per-meaning phonological vector from the
    bundle's ``phonological_vector`` field. Accepts:

    * ``None`` / missing — return None (legacy bundles, pre-kq7w.1).
    * a dict — already-parsed JSON shape (bundle-export emits this
      directly under wyrd-ecjp.8); construct a PhonologicalVector
      from it, dropping unknown keys silently.
    * a str — JSON blob (the raw etymon.phonological_vector column
      shape, in case a bundle embeds the column verbatim).

    Returns a PhonologicalVector or None. Never raises — a malformed
    vector blob (non-numeric value, nested-dict / list / None at a
    dimension key, JSON decode failure, non-dict extras) falls back
    to None and the vector-scoring path treats it as a default-empty
    vector (phon_score returns 0). The contract is "best-effort
    parse; degrade silently rather than crash NameGenerator startup
    on one bad bundle row."
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None
    # Build the dataclass kwargs from the known dimension names; drop
    # unknowns to extras for forward-compat with future dimension
    # additions. Wrap the float() coercions in try/except: a malformed
    # value at any dimension key (None, nested dict, non-numeric str)
    # would otherwise raise TypeError / ValueError and propagate out
    # of load_meanings — the "never raises" contract requires the
    # whole-row fallback to None.
    try:
        kwargs = {k: float(v) for k, v in raw.items() if k in _PHON_KNOWN_DIM_NAMES}
        extras_in = raw.get("extras")
        if isinstance(extras_in, dict):
            extras_in = {k: float(v) for k, v in extras_in.items()}
        else:
            # Non-dict extras (e.g. operator hand-edit gone wrong) →
            # treat as absent rather than crashing.
            extras_in = {}
        # Capture any unknown top-level keys as extras alongside the
        # explicit extras dict (forward-compat: bundle may emit new
        # dims at top level before the schema's named-fields list
        # catches up).
        extra_kwargs = {
            k: float(v) for k, v in raw.items() if k not in _PHON_KNOWN_DIM_NAMES and k != "extras"
        }
    except (TypeError, ValueError):
        return None
    extras = {**extras_in, **extra_kwargs}
    if extras:
        return PhonologicalVector(extras=extras, **kwargs)
    return PhonologicalVector(**kwargs)


# Module-level constant: the canonical PhonologicalVector dimensions.
# Used by ``_parse_phon_vector`` to route bundle-emitted keys to
# dataclass kwargs vs the ``extras`` forward-compat slot.
#
# Derived directly from the schema's ``_DIMENSION_NAMES`` frozenset
# (itself built from the ``PhonologicalFeatureName`` Literal via
# ``get_args``) so adding a dim to the schema automatically flows
# through the runtime loader on the next import — no second hand-
# maintained name list to drift. wyrd-3uzp caught the pre-derivation
# drift: wyrd-119p + wyrd-mkry's new dimensions (liquid_l_m_n,
# rhotic_r, vowel_tenseness) were silently routing into ``extras``
# instead of the named fields because this set carried a 14-name
# copy that hadn't been updated.
_PHON_KNOWN_DIM_NAMES = _DIMENSION_NAMES


def _normalize_era_reflexes(
    raw: dict,
) -> dict[str, list[tuple[str, str]]]:
    """wyrd-jbcu: normalize era_reflexes to the internal
    ``{lang: [(form, source), ...]}`` shape. Accepts both the legacy
    ``{lang: [form, ...]}`` shape (entries default to source='cluster'
    since legacy bundles only carried attestation-backed reflexes)
    and the new ``{lang: [{form, source}, ...]}`` shape. Skips
    malformed entries silently rather than crashing — a stray
    integer or null shouldn't break the entire bundle load."""
    out: dict[str, list[tuple[str, str]]] = {}
    if not isinstance(raw, dict):
        return out
    for lang, entries in raw.items():
        if not isinstance(entries, list):
            continue
        normalized: list[tuple[str, str]] = []
        for entry in entries:
            if isinstance(entry, str):
                normalized.append((entry, "cluster"))
            elif isinstance(entry, dict) and "form" in entry:
                normalized.append((entry["form"], entry.get("source") or "cluster"))
        if normalized:
            out[lang] = normalized
    return out


def _normalize_era_reflex_glosses(raw: dict) -> dict[str, dict[str, str]]:
    """wyrd-rogd.1: extract the per-form gloss map ``{lang: {form: gloss}}``
    from the bundle's era_reflexes field — only the {form, source, gloss} dict
    entries that carry a non-empty ``gloss``. This runs OVER THE SAME raw field
    as ``_normalize_era_reflexes`` but extracts a DIFFERENT projection (gloss,
    not source), kept separate so the (form, source) shape the rewinder
    consumes stays untouched — it is not a structural parallel of that
    function.

    Like its sibling it is fail-soft: malformed / glossless / bare-string
    entries are skipped silently (an export bug shouldn't take down the load),
    here yielding no gloss for that form. Legacy bundles (bare-string or
    {form, source}-only entries) therefore produce an empty map.

    Assumes one entry per form per lang (upheld upstream by the ``best:
    dict[str, EraReflex]`` form-dedup in ``_fetch_family_era_reflexes``); on a
    hypothetical duplicate form the dict-comprehension would last-write-wins."""
    out: dict[str, dict[str, str]] = {}
    if not isinstance(raw, dict):
        return out
    for lang, entries in raw.items():
        if not isinstance(entries, list):
            continue
        glosses = {
            entry["form"]: entry["gloss"]
            for entry in entries
            # form must be a non-empty STRING (it's a dict key — a non-string
            # form from a malformed export would be unhashable and crash the
            # load; fail-soft skips it instead). gloss must be a non-empty str.
            if isinstance(entry, dict)
            and isinstance(entry.get("form"), str)
            and entry["form"]
            and isinstance(entry.get("gloss"), str)
            and entry["gloss"]
        }
        if glosses:
            out[lang] = glosses
    return out


def _bundle_subjects(data) -> list:
    """Extract the subjects list from a meanings.json bundle that may
    be either the legacy list-of-subjects shape or the wyrd-q0g6 dict-
    shape ``{"subjects": [...], "joiners": {...}}``. Other bundle-level
    fields (joiners) are read by ``load_joiners`` separately."""
    if isinstance(data, dict):
        return data.get("subjects") or []
    return data


def load_canonical_decompositions(data) -> dict[str, dict[str, str]]:
    """Extract the per-toponym canonical decomposition map from a
    bundle (wyrd-h8k1 Phase 3b).

    Returns ``{modern_name: {"signature": sha1_hex, "source": label}}``.
    Empty for legacy list-shape bundles AND for dict-shape bundles
    that don't carry a ``canonical_decompositions`` field. Callers
    (the explainer) can rely on ``.get(name)`` returning ``None`` when
    no canonical was projected.

    The signature is the SHA-1 hex from
    ``decomposition._signature_for_payload`` over a flattened slot
    list — KenningExplain re-derives this signature for each candidate
    decomposition the matcher emits and matches it against the bundle
    map to mark + front-load the canonical reading.
    """
    if not isinstance(data, dict):
        return {}
    raw = data.get("canonical_decompositions") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        signature = entry.get("signature")
        if not isinstance(signature, str) or not signature:
            continue
        out[name] = {
            "signature": signature,
            "source": entry.get("source") or "",
        }
    return out


def load_fantasy_morphemes(data) -> dict[str, dict]:
    """wyrd-vz7f: extract the per-creature fantasy-morpheme map from a
    bundle for the runtime ``KenningCreature`` Generator.

    Returns ``{lowercase_input_name: {input_name, etymon_id, language,
    canonical_form, english_shaped, glosses, citation, era_reflexes}}``.
    Keyed lowercase to match the lexicon's ``fantasy_morpheme.input_name
    COLLATE NOCASE`` semantics so 'Harpy' and 'harpy' resolve to the
    same row at runtime.

    Empty for legacy list-shape bundles AND for dict-shape bundles
    that don't carry a ``fantasy_morphemes`` field. ``era_reflexes``
    is normalized to the wyrd-jbcu source-aware shape via
    ``_normalize_era_reflexes`` so KenningCreature can read source
    tags consistently with the toponym path.
    """
    if not isinstance(data, dict):
        return {}
    raw = data.get("fantasy_morphemes") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for input_name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        normalized_era = _normalize_era_reflexes(entry.get("era_reflexes") or {})
        out[input_name.lower()] = {
            "input_name": entry.get("input_name") or input_name,
            "etymon_id": entry.get("etymon_id"),
            "language": entry.get("language") or "",
            "canonical_form": entry.get("canonical_form") or "",
            "english_shaped": entry.get("english_shaped") or "",
            "glosses": list(entry.get("glosses") or []),
            "citation": entry.get("citation") or "",
            "era_reflexes": normalized_era,
        }
    return out


def load_joiners(data) -> dict[str, list[tuple[str, int]]]:
    """Extract the per-language-field joiner pool from a bundle (wyrd-q0g6
    Phase 1).

    Returns a dict keyed by lang_field (``"old_english"``,
    ``"celtic_mix"``, etc.) carrying ``[(form, weight), ...]`` tuples.
    Empty for legacy list-shape bundles AND for dict-shape bundles
    that don't carry a ``joiners`` field. Callers (the matcher hook)
    can rely on iteration over an empty dict being a no-op.

    A joiner entry's ``form`` is the lowercase surface string that
    matches between morphemes; ``weight`` is the attestation count
    Phase 2 will populate from the corpus audit. Phase 1 ships the
    schema; the bundle ships with no joiners until a Phase 2 audit
    populates them.
    """
    if isinstance(data, dict):
        raw = data.get("joiners") or {}
    else:
        raw = {}
    return {
        lang_field: [(entry["form"], entry["weight"]) for entry in entries]
        for lang_field, entries in raw.items()
        if entries
    }


def _extract_phonological_vectors(
    word: dict,
) -> tuple[PhonologicalVector | None, dict[str, dict[str, PhonologicalVector]]]:
    """Parse a word's phonological-vector data (wyrd-ecjp.5 / kq7w.1 / 10a).

    Reads the per-language ``<lang>_phonological_vector`` siblings (D26)
    into the plural ``phonological_vectors`` dict keyed by
    ``(lang, canonical_form)``, skipping malformed entries (best-effort
    parse, degrade silently). For the back-compat singular
    ``phonological_vector`` it reads a legacy top-level
    ``phonological_vector`` key first, falling back to the first
    non-empty vector across the deterministic lang × form walk.

    Returns ``(phonological_vector, phonological_vectors)``.
    """
    phonological_vectors: dict[str, dict[str, PhonologicalVector]] = {}
    for k, v in word.items():
        if not k.endswith(_PHONOLOGICAL_VECTOR_SUFFIX):
            continue
        lang_key = k[: -len(_PHONOLOGICAL_VECTOR_SUFFIX)]
        per_form: dict[str, PhonologicalVector] = {}
        for entry in v:
            # Skip malformed entries: a non-dict entry or a dict missing
            # the "form" key would crash the parse loop. Bundles can in
            # principle ship arbitrary JSON, and the "best-effort parse;
            # degrade silently" contract that _parse_phon_vector
            # documents extends to the surrounding entry shape.
            if not isinstance(entry, dict) or "form" not in entry:
                continue
            parsed = _parse_phon_vector(entry.get("phonological_vector"))
            if parsed is not None:
                per_form[entry["form"]] = parsed
        if per_form:
            phonological_vectors[lang_key] = per_form
    legacy_top_level = word.get("phonological_vector")
    if legacy_top_level is not None:
        phonological_vector = _parse_phon_vector(legacy_top_level)
    else:
        # Deterministic lang × form walk: pick the first non-empty
        # vector. next(gen, None) is the Pythonic "first match or None"
        # — avoids the nested-break pattern.
        phonological_vector = next(
            (
                phonological_vectors[lang_key][form]
                for lang_key in sorted(phonological_vectors)
                for form in sorted(phonological_vectors[lang_key])
            ),
            None,
        )
    return phonological_vector, phonological_vectors


def load_meanings(data):
    meaning_db: dict[str, list[Meaning]] = {}
    tags_db: dict[str, list[str]] = {}
    for subject in _bundle_subjects(data):
        tags = subject["modifier_tags"]
        meanings = subject["meaning"]
        for word in subject["words"]:
            usage = word["modern_usage"]
            # Split the word's fields into (a) language form arrays the runtime
            # already used, (b) variant pools (D18) under "<lang>_variants",
            # (c) inflection metadata (D8) under "<lang>_inflections".
            sources = {
                k: v
                for k, v in word.items()
                if k != "modern_usage"
                and k != "era_reflexes"  # wyrd-obpw: top-level, not a source
                and k != "morpheme_id"  # wyrd-rogd.10: the owning-morpheme id, not a language
                and not k.endswith(_VARIANT_SUFFIX)
                and not k.endswith(_INFLECTION_SUFFIX)
                and not k.endswith(_CITATIONS_SUFFIX)
                and not k.endswith(_ATTESTED_YEARS_SUFFIX)
                and not k.endswith(_ENGLISH_SHAPED_SUFFIX)
                and not k.endswith(_ORIGINAL_SCRIPT_SUFFIX)
                and not k.endswith(_TRANSLITERATION_SUFFIX)
                and not k.endswith(_PRONUNCIATION_SUFFIX)
                and not k.endswith(_STRATUM_SUFFIX)
                and not k.endswith(_PHONOLOGICAL_VECTOR_SUFFIX)
            }
            variants = {
                k[: -len(_VARIANT_SUFFIX)]: [(entry["form"], entry["weight"]) for entry in v]
                for k, v in word.items()
                if k.endswith(_VARIANT_SUFFIX)
            }
            inflections = {
                k[: -len(_INFLECTION_SUFFIX)]: [(entry["form"], entry["inflection"]) for entry in v]
                for k, v in word.items()
                if k.endswith(_INFLECTION_SUFFIX)
            }
            citations = {
                k[: -len(_CITATIONS_SUFFIX)]: list(v)
                for k, v in word.items()
                if k.endswith(_CITATIONS_SUFFIX)
            }
            attested_years = {
                k[: -len(_ATTESTED_YEARS_SUFFIX)]: [(entry["form"], entry["year"]) for entry in v]
                for k, v in word.items()
                if k.endswith(_ATTESTED_YEARS_SUFFIX)
            }
            # wyrd-ha9q Phase 2c: english_shaped is keyed by canonical
            # form so the runtime can look up the rendering of a specific
            # form within a language (each language can carry multiple
            # forms — e.g. Sanskrit family with both rakṣas and rakṣasa).
            english_shaped = {
                k[: -len(_ENGLISH_SHAPED_SUFFIX)]: {
                    entry["form"]: entry["english_shaped"] for entry in v
                }
                for k, v in word.items()
                if k.endswith(_ENGLISH_SHAPED_SUFFIX)
            }
            # wyrd-qhs0 Phase 2d: same shape as english_shaped — each
            # is keyed by canonical form so the runtime can look up the
            # rendering of a specific form within a language family.
            original_script = {
                k[: -len(_ORIGINAL_SCRIPT_SUFFIX)]: {
                    entry["form"]: entry["original_script"] for entry in v
                }
                for k, v in word.items()
                if k.endswith(_ORIGINAL_SCRIPT_SUFFIX)
            }
            transliteration = {
                k[: -len(_TRANSLITERATION_SUFFIX)]: {
                    entry["form"]: entry["transliteration"] for entry in v
                }
                for k, v in word.items()
                if k.endswith(_TRANSLITERATION_SUFFIX)
            }
            pronunciation = {
                k[: -len(_PRONUNCIATION_SUFFIX)]: {
                    entry["form"]: {
                        "ipa": entry["ipa"],
                        "dialect": entry.get("dialect"),
                    }
                    for entry in v
                }
                for k, v in word.items()
                if k.endswith(_PRONUNCIATION_SUFFIX)
            }
            # wyrd-lr4 Phase 2: stratum is keyed by canonical form so the
            # Phase 3 generator filter can ask 'is THIS form in THIS
            # stratum?' rather than gating an entire language.
            stratum = {
                k[: -len(_STRATUM_SUFFIX)]: {entry["form"]: entry["stratum"] for entry in v}
                for k, v in word.items()
                if k.endswith(_STRATUM_SUFFIX)
            }
            # wyrd-obpw Phase 3.3: era_reflexes is a TOP-LEVEL field
            # on each word entry (not a per-language sibling) since
            # it represents the family root's cluster reflexes —
            # one set per family, not per source language. Empty for
            # legacy bundles that pre-date the wyrd-obpw export.
            #
            # wyrd-jbcu schema upgrade: each entry can be EITHER a
            # bare form string (legacy bundles, source defaults to
            # 'cluster' since everything in the legacy bundle was
            # attestation-backed) OR a {"form": str, "source": str}
            # dict (new schema). Internal storage is always the rich
            # (form, source) tuple shape.
            era_reflexes_field = word.get("era_reflexes") or {}
            era_reflexes = _normalize_era_reflexes(era_reflexes_field)
            era_reflex_glosses = _normalize_era_reflex_glosses(era_reflexes_field)
            # wyrd-ecjp.5 / wyrd-kq7w.1: per-meaning phonological vector.
            # wyrd-ecjp.10a Phase 7: bundle now emits per-language
            # siblings (<lang>_phonological_vector) per the D26 pattern.
            # Each sibling is a list of {"form": canonical_form,
            # "phonological_vector": {<dim>: <float>, ...}} entries.
            # The per-language dict lands on Meaning.phonological_vectors
            # (plural) keyed by (lang, canonical_form); for back-compat
            # with callers that read the singular Meaning.phonological_vector
            # we also pick a primary vector (first non-empty across the
            # deterministic lang × form walk). Legacy bundles that emit
            # a top-level "phonological_vector" key (pre-10a) still
            # work — we read it first and only fall back to the
            # per-language walk when absent.
            phonological_vector, phonological_vectors = _extract_phonological_vectors(word)
            # Singular and plural Meanings share every constructor arg
            # except `usage`. Bundle them so a future kwarg addition can't
            # silently drop on one branch and not the other.
            common_kwargs = {
                "tags": tags,
                "meanings": meanings,
                "sources": sources,
                "variants": variants,
                "inflections": inflections,
                "citations": citations,
                "attested_years": attested_years,
                "english_shaped": english_shaped,
                "original_script": original_script,
                "transliteration": transliteration,
                "pronunciation": pronunciation,
                "stratum": stratum,
                "era_reflexes": era_reflexes,
                "era_reflex_glosses": era_reflex_glosses,
                "phonological_vector": phonological_vector,
                "phonological_vectors": phonological_vectors,
                "morpheme_id": word.get("morpheme_id"),  # wyrd-rogd.10 Phase 2
            }
            meaning = Meaning(usage, **common_kwargs)
            for tag in tags:
                tags_db.setdefault(tag, []).append(usage)
            meaning_db.setdefault(usage, []).append(meaning)
            if meaning.is_name() and not usage.endswith("s"):
                plural = f"{usage}s"
                plural_meaning = Meaning(plural, **common_kwargs)
                for tag in tags:
                    tags_db.setdefault(tag, []).append(plural)
                meaning_db.setdefault(plural, []).append(plural_meaning)
            # wyrd-lx8: register one shadow Meaning per inflected form so
            # the matcher recognizes inflected surface forms (cotum,
            # brycgan, denu) and resolves them to their lemma without
            # leaving the inflection suffix as 'unaccounted'. Inflections
            # are already on the canonical Meaning for D8 generation
            # picking; this loop ALSO surfaces them as match candidates.
            #
            # Each shadow Meaning preserves the canonical's tags / sources
            # / inflections / variants etc. so post-match consumers see
            # the same metadata as if the lemma matched directly. The
            # shadow's ``usage`` carries the canonical's dash markers so
            # its position constraint (pre/post/inner) matches the lemma.
            #
            # No shadow if the inflected form's surface equals the
            # lemma's — would create a duplicate Meaning at the same key.
            _register_inflection_shadows(
                usage, inflections, common_kwargs, meaning_db, tags_db, tags
            )
    return meaning_db, tags_db


def _register_inflection_shadows(
    canonical_usage: str,
    inflections: dict,
    common_kwargs: dict,
    meaning_db: dict,
    tags_db: dict,
    tags: list,
) -> None:
    """Add one shadow Meaning per inflected form to ``meaning_db``.

    The shadow shares everything but ``usage`` with the canonical
    Meaning. Its ``usage`` wraps the inflected form in the same dash
    markers as the canonical so position constraints round-trip
    (a post-suffix lemma stays a post-suffix in inflected form).

    Skips:
    * Inflected forms whose dashed-usage equals the canonical (would
      register a duplicate).
    * Empty / whitespace-only forms.
    """
    leading_dash = "-" if canonical_usage.startswith("-") else ""
    trailing_dash = "-" if canonical_usage.endswith("-") else ""
    canonical_lower = canonical_usage.lower()
    for _lang, infl_pairs in inflections.items():
        for form, _label in infl_pairs:
            form = (form or "").strip()
            if not form:
                continue
            shadow_usage = f"{leading_dash}{form}{trailing_dash}"
            if shadow_usage.lower() == canonical_lower:
                continue
            shadow = Meaning(shadow_usage, **common_kwargs)
            for tag in tags:
                tags_db.setdefault(tag, []).append(shadow_usage)
            meaning_db.setdefault(shadow_usage, []).append(shadow)
