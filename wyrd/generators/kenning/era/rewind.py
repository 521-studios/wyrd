"""wyrd-rni Phase 3.1: time-rewind explainer.

Given a modern (or invented) town name, decompose it into morphemes
via the existing kenning explainer pipeline, anchor each morpheme to
its source-language etymon (typically Old English for British place
names), and render the compound at multiple historical era stops.

Sample output for the input ``"Whitchurch"``::

    Whitchurch — decomposed: hwit + cirice
      oe-late  (800-1100):  hwīt + cirice
      me       (1100-1500): whit + chirche
      modern   (1700+):     white + church

The CLI mode is ``wyrd kenning rewind <name>``; the underlying
``rewind_name`` function is also importable for downstream demos
(notably wyrd-381's stratified era-map). Both consume the era-reflex
primitive shipped in wyrd-skm Phase 3.0b — the time-aware infra is
factored out of this module so the rewinder is a thin orchestration
layer over morpheme decomposition + era-reflex lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wyrd.generators.kenning.era.cells import (
    canonical_language_for_cell,
    era_cells_for_family,
)
from wyrd.generators.kenning.lexicon import (
    LANGUAGE_FIELDS,
    EraReflex,
    LexiconDB,
    etymon_era_reflexes,
)
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.name import Name

# Default era ladder for English-family rewind: three cells across
# 800-present. wyrd-rni's "this name through time" output reads as a
# 3-row table rather than the full 5-cell ladder so the rewind output
# stays compact for typical CLI / SPA usage. wyrd-381 (stratified
# era-map) will consume the same primitive but with a denser ladder.
DEFAULT_ENGLISH_ERAS: tuple[tuple[str, str], ...] = (
    ("english", "oe-late"),
    ("english", "me"),
    ("english", "modern"),
)

# Source-language preference order for picking the anchor etymon.
# Most British place-name morphemes are OE; Old Norse second (Danelaw
# / Five Boroughs); Celtic third. Modern-english is the last-resort
# anchor — typically an inflected modern form whose history we can
# still walk via cognate cluster.
#
# Some entries look like typos but are real bundle keys preserved
# from the legacy meanings.json data:
# - ``"old _english"`` (with stray space) was a misspelled key in
#   the original Rando-port data; the bundle preserves it so older
#   meanings entries don't silently lose their source attribution.
# - ``"old_scandanavian"`` (note the misspelling) is the legacy
#   spelling of "old_scandinavian" used in some early entries.
# Both are also normalised by ``LANGUAGE_FIELDS`` in lexicon.py so
# the anchor lookup against the etymon table works either way.
_ANCHOR_LANG_PREFERENCE: tuple[str, ...] = (
    "old_english",
    "old _english",
    "old_scandinavian",
    "old_scandanavian",
    "old_french",
    "celtic_mix",
    "latin",
    "modern_english",
)

# wyrd-085k: free particles that scribes historically wrote as
# separate tokens in place names — Stratford **on** Avon, Henley
# **upon** Thames, Bury Saint Edmunds (the **of** would be a
# separator in 'Bury of Saint Edmunds'-style historical citations).
#
# Intentionally EXCLUDED (despite being prepositions in modern
# English):
#   * ``by`` — common Old Norse settlement suffix (Whitby = OE
#     'white' + ON 'býr'; Derby = OE 'deer' + ON 'býr'). Treating
#     'by' as a particle would mis-render every Norse-suffix name
#     in the corpus.
#   * ``in`` — rare standalone preposition in English place names;
#     more often a Germanic locative reduction ('in' + 'wood' →
#     Inwood = solid compound).
#   * ``at`` — same as 'in', plus the OE form 'æt' was a fully
#     productive prefix that did concatenate.
# The detection rule pairs lowercased-form membership with the
# absence of hyphen markers in the morpheme's original modern_usage:
# a morpheme written ``"-by"`` (post-marker) is a settlement suffix
# regardless of whether 'by' is in this set, so the hyphen-check
# is a second guard against the by/by-suffix collision.
_FREE_PARTICLES: frozenset[str] = frozenset({"on", "upon", "under", "of"})


def _is_free_particle(meaning: Meaning) -> bool:
    """wyrd-085k: classify a morpheme as a free particle for the
    smart-join renderer.

    A morpheme is a free particle when BOTH conditions hold:
      1. Its modern_usage has no hyphen markers — a hyphen-marked
         usage is a positional suffix / prefix / infix (``-by``,
         ``Whit-``, ``-on-``) that scribes concatenated into the
         compound regardless of whether the bare word matches a
         preposition. Checked FIRST so the comparison is short-
         circuited before lowercase / membership.
      2. The lowercased ``modern_usage`` is in ``_FREE_PARTICLES``.
    """
    usage = meaning.usage
    if "-" in usage:
        return False
    return usage.lower() in _FREE_PARTICLES


def render_form_particle_pairs(pairs: list[tuple[str, bool]], *, smart_join: bool = True) -> str:
    """wyrd-085k / wyrd-2pio: shared name renderer used by both
    rewinders (CLI + SPA-facing KenningRewind via era-map). Takes
    a list of ``(form, is_particle)`` tuples — agnostic to the
    upstream's morpheme dataclass shape.

    Free particles (``on``, ``upon``, ``under``, ``of``) get
    surrounded by spaces so they read as separate tokens, matching
    historical scribal practice (Stratford **on** Avon, Henley
    **upon** Thames). Adjacent non-particle morphemes concatenate
    directly. Each space-separated token gets its first letter
    title-cased; particles stay fully lowercase to match modern
    typesetting convention for prepositions in proper names.

    ``smart_join=False`` delegates to ``_concat_form_pairs_simple``
    — particle handling is bypassed. Used at the modern-english
    era stop (wyrd-t2bh).

    Empty input returns the empty string. A single non-particle
    morpheme returns its form title-cased. A leading or trailing
    particle is preserved as its own token (rendering
    ``[('on', True), ('foo', False)]`` as ``"on Foo"``, not as a
    malformed compound) — only with smart_join=True.
    """
    if not smart_join:
        return _concat_form_pairs_simple(pairs)
    if not pairs:
        return ""
    return _titlecase_join_tokens(_group_pairs_into_tokens(pairs))


def _group_pairs_into_tokens(pairs: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    """Smart-join pass 1: collapse each run of adjacent non-particle forms into
    one ``(token, is_particle=False)`` token, and emit each particle as its own
    ``(form.lower(), is_particle=True)`` token. The particle flag rides along so
    pass 2 can decide casing by origin, not by the rendered surface string."""
    tokens: list[tuple[str, bool]] = []  # (token, is_particle), space-separated
    pending: list[str] = []  # buffer of adjacent non-particle forms
    for form, is_particle in pairs:
        if is_particle:
            if pending:
                tokens.append(("".join(pending), False))
                pending = []
            tokens.append((form.lower(), True))
        else:
            pending.append(form)
    if pending:
        tokens.append(("".join(pending), False))
    return tokens


def _titlecase_join_tokens(tokens: list[tuple[str, bool]]) -> str:
    """Smart-join pass 2: keep particle tokens lowercase, title-case non-particle
    tokens, drop empties, and space-join. Casing is keyed on the ``is_particle``
    flag — NOT on ``_FREE_PARTICLES`` membership — so a non-particle run whose
    forms happen to concatenate to a free-particle string (``un`` + ``der`` →
    ``Under``) is still title-cased (wyrd-mfmr).

    Title-case is **position-driven**: the word-initial letter is upper-cased
    and the rest of the token is LOWER-cased. The tail lowercase is load-bearing
    — morpheme surfaces inherit the source toponym's case (a word-initial part
    is stored title-cased, e.g. ``Tun``), so a capitalized INNER morpheme
    concatenated into a token (``stan`` + ``Tun`` → ``stanTun``) would otherwise
    render as CamelCase (``StanTun``). Lower-casing the tail makes case a pure
    function of position, never of the stored surface."""
    rendered: list[str] = []
    for token, is_particle in tokens:
        if not token:
            continue  # drop empties uniformly (incl. empty particles) — no double spaces
        if is_particle:
            rendered.append(token)
        else:
            rendered.append(token[0].upper() + token[1:].lower())
    return " ".join(rendered)


def _concat_form_pairs_simple(pairs: list[tuple[str, bool]]) -> str:
    """wyrd-t2bh / wyrd-2pio: simple concatenation of form-particle
    pairs, position-driven title-case (word-initial upper, rest lower).
    Used at the modern-english era stop where canonical short-circuit
    (wyrd-8qbi) already produces the operator's input shape.

    The tail lower-case mirrors ``_titlecase_join_tokens`` — a capitalized
    inner morpheme surface (``stan`` + ``Tun``) must not leak through as
    CamelCase (``StanTun``); case is a function of position, not surface."""
    if not pairs:
        return ""
    joined = "".join(form for form, _ in pairs)
    if joined:
        joined = joined[0].upper() + joined[1:].lower()
    return joined


def _render_morphemes_as_name(morphemes: list[MorphemeRewind], *, smart_join: bool = True) -> str:
    """wyrd-085k: join MorphemeRewind forms into a medieval-style
    name. Thin adapter over ``render_form_particle_pairs`` that
    pulls each morpheme's picked form + is_free_particle flag.
    See the shared renderer's docstring for the join contract."""
    pairs = [(_pick_form(m), m.is_free_particle) for m in morphemes]
    return render_form_particle_pairs(pairs, smart_join=smart_join)


@dataclass
class MorphemeRewind:
    """Per-morpheme rewind data at one era stop."""

    canonical: str
    """The morpheme's modern usage (from Meaning.usage)."""

    source_form: str
    """The anchor etymon's canonical_form in its source language."""

    source_language: str
    """Anchor's etymon.language tag (e.g. ``'old-english'``)."""

    source_etymon_id: int | None
    """Anchor etymon row id, or None when no anchor was found in the
    lexicon DB. Without an anchor the rewind falls back to the
    morpheme's modern usage at every era."""

    era_form: str | None
    """Period-specific surface form picked from the anchor's cognate
    cluster (or descent edges as fallback). ``None`` when neither
    lookup path produced a match — caller should fall back."""

    fallback: bool
    """True when ``era_form`` is None and the rendered compound used
    a fallback (anchor's canonical_form, or modern usage when no
    anchor exists). Flagged so the user can see which morphemes
    'don't have era data' for the requested cell."""

    is_free_particle: bool = False
    """wyrd-085k: True when this morpheme is a free preposition /
    particle (``on``, ``upon``, ``under``, ``of``) that scribes
    historically wrote as a separate token rather than concatenating
    into the compound (Stratford **on** Avon, Henley **upon**
    Thames). The smart-join renderer uses this to surround the
    morpheme with spaces instead of concatenating it inline.

    Detection: the morpheme's modern_usage has no hyphen markers
    AND its lowercased form is in the particle list. The hyphen-
    absence guard catches the by / by-suffix collision — a
    hyphen-marked usage (``-by``, the Old Norse settlement marker
    in Whitby / Derby) is a positional suffix regardless of whether
    the bare string matches a preposition. ``by`` and ``in`` are
    intentionally NOT in the particle list either, since both have
    common settlement-suffix uses in English place names; treating
    them as particles would mis-join the Norse-origin names even
    when the hyphen guard didn't catch them."""


@dataclass
class EraStop:
    """A single era's rewind result for the full input name."""

    family: str
    cell: str
    morphemes: list[MorphemeRewind]
    rendered: str
    """wyrd-085k: composed surface form rendered as a medieval-style
    place name. Adjacent non-particle morphemes concatenate directly;
    free particles (``on``, ``upon``, ``under``, ``of``) sit as their
    own space-separated tokens. Each non-particle token gets its
    first letter title-cased. See ``_render_morphemes_as_name`` for
    the join algorithm. Pre-wyrd-085k this was hyphen-separated
    (``elm-tūn``); the analytical view is still available via
    ``wyrd kenning explain``."""


@dataclass
class Rewind:
    """Result of a rewind call for one input name."""

    name: str
    decomposition: str
    """Human-readable decomposition string ('hwit + cirice')."""

    eras: list[EraStop]
    unaccounted: list[str] = field(default_factory=list)


def rewind_name(
    name: str,
    db: LexiconDB,
    meaning_db: dict[str, list[Meaning]],
    *,
    eras: tuple[tuple[str, str], ...] = DEFAULT_ENGLISH_ERAS,
) -> Rewind:
    """Decompose ``name`` and render its morphemes at each era stop.

    Returns a ``Rewind`` carrying one ``EraStop`` per requested era.
    Morphemes that don't anchor (no etymon row found, or anchor lacks
    cognate_id and descent edges) fall back to their modern usage at
    each era stop with ``fallback=True`` flagged.

    The decomposition uses the existing ``Name.find_meaning`` matcher
    (same as ``KenningExplain``) with ``reduce=True`` to pick the
    canonical decomposition; multi-decomposition surfaces aren't
    relevant here since we only render ONE rewind output per input.

    ``meaning_db`` is the bundle-loaded Meaning index — passed in by
    the caller rather than loaded inside this function so the CLI
    can reuse it across multiple rewinds without re-parsing the bundle.
    """
    text = (name or "").strip()
    if not text:
        raise ValueError("name is required")

    name_obj = Name(text)
    name_obj.find_meaning(meaning_db, reduce=True)
    decomposition_parts: list[str] = []
    # wyrd-t2bh: per-word morpheme groups so the renderer can preserve
    # the operator's input shape — a multi-word input like
    # 'Helon Mortborough' decomposes into two words ([Hel, on] +
    # [Mort, borough]), and the modern-cell renderer concatenates
    # within each word but joins across words with the original space.
    # The flat list (used by the CLI's fallback-flag scan via
    # EraStop.morphemes) gets rebuilt at the end.
    morpheme_anchors_per_word: list[list[tuple[Meaning, _Anchor | None]]] = []
    unaccounted: list[str] = []

    # ``Name.words[word_str]`` is a list of Word objects (alternative
    # canonical decompositions tied for 'best'). v1 takes the first;
    # multi-decomposition surfacing is a follow-up if user demand
    # warrants it. ``Word.word`` is the flat segmented sequence
    # (Meaning | str chunks).
    for word_str in text.split():
        candidates = name_obj.words.get(word_str, [])
        if not candidates:
            unaccounted.append(word_str)
            continue
        decomposition = candidates[0]
        word_anchors: list[tuple[Meaning, _Anchor | None]] = []
        for chunk in decomposition.word:
            if isinstance(chunk, Meaning):
                anchor = _resolve_anchor(chunk, meaning_db, db)
                word_anchors.append((chunk, anchor))
                fragment = chunk.usage.replace("-", "")
                decomposition_parts.append(fragment)
            elif isinstance(chunk, str) and chunk:
                unaccounted.append(chunk)
        if word_anchors:
            morpheme_anchors_per_word.append(word_anchors)

    decomposition = " + ".join(decomposition_parts) or "no morphemes recognized"

    era_stops: list[EraStop] = []
    for family, cell in eras:
        # wyrd-t2bh: per-word render so multi-word inputs like
        # 'Helon Mortborough' preserve the operator's space-shape.
        morphemes_per_word: list[list[MorphemeRewind]] = [
            [
                _render_morpheme_at_era(meaning, anchor, family, cell, db)
                for meaning, anchor in word_anchors
            ]
            for word_anchors in morpheme_anchors_per_word
        ]
        # wyrd-t2bh: modern era stop short-circuits to canonical per
        # wyrd-8qbi; bypass smart-join so the rendered output matches
        # the operator's input shape rather than re-splitting words
        # like 'Helon' into 'Hel on'. OE/ME cells still smart-join
        # (historical scribal practice).
        target_language = canonical_language_for_cell(family, cell)
        smart_join = target_language != "modern-english"
        word_renders = [
            _render_morphemes_as_name(word, smart_join=smart_join) for word in morphemes_per_word
        ]
        rendered = " ".join(r for r in word_renders if r)
        # Flatten morphemes for back-compat with EraStop.morphemes
        # consumers (CLI fallback-flag scan).
        morphemes = [m for word in morphemes_per_word for m in word]
        era_stops.append(EraStop(family=family, cell=cell, morphemes=morphemes, rendered=rendered))

    return Rewind(
        name=text,
        decomposition=decomposition,
        eras=era_stops,
        unaccounted=unaccounted,
    )


def rewind_from_morphemes(
    name: str,
    words: list[list[dict]],
    db: LexiconDB,
    meaning_db: dict[str, list[Meaning]],
    *,
    eras: tuple[tuple[str, str], ...] = DEFAULT_ENGLISH_ERAS,
) -> Rewind:
    """wyrd-cp2d: bypass trie re-decomposition + use pre-picked
    morphemes from a 'kenning generate --json' dump.

    Same per-word grouping + anchor + era-render pipeline as
    ``rewind_name`` but skips the trie matcher entirely. Each
    morpheme dict in ``words`` carries its ``usage`` (bucket key);
    the Meaning is looked up directly in ``meaning_db[usage]``,
    preserving the generator's committed pick rather than the trie's
    re-decomposition (which can land on a DIFFERENT etymological
    cluster mate — the Cranwith → corn / Cornlandian situation).

    Each morpheme in the JSON ``sources`` dict CAN be used to
    pre-select a specific Meaning sibling when meaning_db[usage]
    has multiple entries (different etymological histories under
    the same modern usage). v1 keeps the components()-style "first
    Meaning wins" rule for simplicity; downstream consumers that
    need the exact sibling can filter meaning_db[usage] themselves
    before passing the words list.

    Missing usages (the bundle has dropped a morpheme since the
    JSON was generated) fall back to the unaccounted list with
    the usage string preserved so the rewind output still notes
    the gap.
    """
    text = (name or "").strip()
    if not text:
        raise ValueError("name is required")

    decomposition_parts: list[str] = []
    morpheme_anchors_per_word: list[list[tuple[Meaning, _Anchor | None]]] = []
    unaccounted: list[str] = []

    for word_morphemes in words:
        word_anchors: list[tuple[Meaning, _Anchor | None]] = []
        for morph in word_morphemes:
            usage = morph.get("usage")
            if not usage:
                continue
            candidates = meaning_db.get(usage, [])
            if not candidates:
                # Bundle dropped this morpheme since the JSON was
                # generated; flag in unaccounted with the usage.
                unaccounted.append(usage)
                continue
            # v1: first Meaning wins, matching components() precedent.
            meaning = candidates[0]
            anchor = _resolve_anchor(meaning, meaning_db, db)
            word_anchors.append((meaning, anchor))
            decomposition_parts.append(usage.replace("-", ""))
        if word_anchors:
            morpheme_anchors_per_word.append(word_anchors)

    decomposition = " + ".join(decomposition_parts) or "no morphemes recognized"

    era_stops: list[EraStop] = []
    for family, cell in eras:
        morphemes_per_word: list[list[MorphemeRewind]] = [
            [
                _render_morpheme_at_era(meaning, anchor, family, cell, db)
                for meaning, anchor in word_anchors
            ]
            for word_anchors in morpheme_anchors_per_word
        ]
        target_language = canonical_language_for_cell(family, cell)
        smart_join = target_language != "modern-english"
        word_renders = [
            _render_morphemes_as_name(word, smart_join=smart_join) for word in morphemes_per_word
        ]
        rendered = " ".join(r for r in word_renders if r)
        morphemes = [m for word in morphemes_per_word for m in word]
        era_stops.append(EraStop(family=family, cell=cell, morphemes=morphemes, rendered=rendered))

    return Rewind(
        name=text,
        decomposition=decomposition,
        eras=era_stops,
        unaccounted=unaccounted,
    )


@dataclass
class _Anchor:
    """The etymon row that anchors a morpheme to the lexicon DB."""

    etymon_id: int
    canonical_form: str
    language: str  # etymon.language tag


# Cache of stripped-usage indexes per meaning_db. Keyed by
# ``id(meaning_db)`` rather than the dict itself since dicts aren't
# hashable. The stored value is ``(meaning_db, index)``: the back-reference
# lets the lookup verify the cached entry belongs to THIS meaning_db, so an
# id() recycled after GC (CPython reuses a freed object's id) can never return
# a stale index. Bounded so ad-hoc test meaning_dbs don't accumulate, and
# dropped wholesale on bundle reload via ``_clear_stripped_index_cache`` (wired
# into the kenning coupled cache-clear) — mirrors proportions._SURFACE_INDEX_CACHE.
_STRIPPED_INDEX_CACHE: dict[int, tuple[dict, dict[str, list[Meaning]]]] = {}
_STRIPPED_INDEX_CACHE_MAX = 32


def _clear_stripped_index_cache() -> None:
    """Drop the stripped-usage index cache. Wired into the kenning bundle's
    coupled cache-clear so a meaning_db reload can't read a stale index via an
    id() collision — mirrors ``proportions._clear_surface_index_cache``."""
    _STRIPPED_INDEX_CACHE.clear()


def _get_stripped_index(
    meaning_db: dict[str, list[Meaning]],
) -> dict[str, list[Meaning]]:
    """Return a stripped-usage → list[Meaning] index for ``meaning_db``,
    building it lazily on first lookup and caching by ``id()`` with an identity
    check.

    Stripping the hyphens unifies the bundle's three usage shapes
    (``-ham``, ``ham``, ``ham-``) under a single key so the anchor
    resolver can find OE post-modifier Meanings even when the trie
    picked a no-hyphen Celtic Meaning at the same stripped usage.

    Cache lifetime: keyed by ``id(meaning_db)``. ``_load_meanings`` returns the
    same cached meaning_db across calls, so the index is built once per process
    and reused — O(1) sibling lookup instead of O(N) full-scan. The cached
    ``(meaning_db, index)`` back-reference guards an id() recycled onto a new
    bundle, and the coupled cache-clear drops the whole cache on reload."""
    cache_key = id(meaning_db)
    cached = _STRIPPED_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is meaning_db:
        return cached[1]
    index: dict[str, list[Meaning]] = {}
    for usage, meanings in meaning_db.items():
        stripped = usage.replace("-", "").lower()
        bucket = index.setdefault(stripped, [])
        bucket.extend(meanings)
    # Evict oldest on overflow (matters only for the many small ad-hoc
    # meaning_dbs tests build; production reuses one long-lived bundle).
    if (
        cache_key not in _STRIPPED_INDEX_CACHE
        and len(_STRIPPED_INDEX_CACHE) >= _STRIPPED_INDEX_CACHE_MAX
    ):
        del _STRIPPED_INDEX_CACHE[next(iter(_STRIPPED_INDEX_CACHE))]
    _STRIPPED_INDEX_CACHE[cache_key] = (meaning_db, index)
    return index


def _resolve_anchor(
    meaning: Meaning,
    meaning_db: dict[str, list[Meaning]],
    db: LexiconDB,
) -> _Anchor | None:
    """Pick the source-language etymon that anchors this morpheme.

    Looks across ALL Meaning instances at the same ``usage`` (the
    bundle may carry multiple sense-Meanings — e.g. ``Whit-`` has
    one ON-source and one OE-source instance) and walks
    ``_ANCHOR_LANG_PREFERENCE`` to find the first language whose
    form anchors to an etymon row in the lexicon DB.

    Walking siblings is necessary because the canonical-decomposition
    matcher may have picked the ON-side Meaning for ``Whit-`` while
    the OE-side Meaning is what we want for English-ladder rewind.
    The original ``meaning`` instance is preferred at tied source
    languages (preserves sense fidelity); only when its source langs
    don't anchor do we fall through to siblings.

    Per-language preference reflects the dominant origin pattern for
    British place-name morphemes (OE first, ON second, Celtic third,
    Modern English last-resort).

    Hyphen-variant lookup: the bundle keys morphemes by their full
    hyphen-marked usage (``-ton`` post-modifier, ``Whit-`` pre-
    modifier, ``-en-`` inner-modifier). When the trie matcher picked
    a non-OE Meaning at ``ton`` (Celtic-mix), the OE post-modifier
    Meaning at ``-ton`` carries the right anchor. The
    stripped-usage index built once per ``meaning_db`` (cached on
    the dict via ``_get_stripped_index``) gives O(1) sibling
    lookup; without it this function was O(N) over the full
    meaning_db on every call (~5k entries × every morpheme in every
    rewind).
    """
    stripped = meaning.usage.replace("-", "").lower()
    siblings = list(_get_stripped_index(meaning_db).get(stripped, ()))
    if not siblings:
        siblings = [meaning]
    # Put the resolver's input meaning first so its source-language
    # preference is honoured when tied.
    ordered = [meaning] + [s for s in siblings if s is not meaning]
    for lang_field in _ANCHOR_LANG_PREFERENCE:
        canonical_lang = LANGUAGE_FIELDS.get(lang_field)
        if canonical_lang is None:
            continue
        for sibling in ordered:
            forms = sibling.sources.get(lang_field)
            if not forms:
                continue
            # Pick the FIRST form: bundle ordering puts the canonical
            # lemma first per language (build_forms_by_lang invariant).
            form = forms[0]
            row = db.conn.execute(
                "SELECT id, canonical_form, language FROM etymon "
                "WHERE canonical_form = ? AND language = ? AND merged_into_id IS NULL",
                (form, canonical_lang),
            ).fetchone()
            if row is not None:
                return _Anchor(
                    etymon_id=row["id"],
                    canonical_form=row["canonical_form"],
                    language=row["language"],
                )
    return None


def _render_morpheme_at_era(
    meaning: Meaning,
    anchor: _Anchor | None,
    family: str,
    cell: str,
    db: LexiconDB,
) -> MorphemeRewind:
    """Resolve one morpheme at one era stop.

    Returns a ``MorphemeRewind`` with ``era_form`` set when a cluster
    mate (or descent fallback) matched the requested era, or
    ``era_form=None`` and ``fallback=True`` otherwise. The actual
    *rendered* surface form is decided in ``_pick_form`` — this
    function only resolves the era_form/fallback signals.

    Cases:

    - Anchor present + era-reflex returns at least one mate → pick
      the form per the three-tier preference (canonical match, then
      anchor match, then alphabetical first; see inline comment).
      ``fallback=False``.
    - Anchor present + no era-reflex match → ``era_form=None``,
      ``fallback=True``. ``_pick_form`` falls back to the morpheme's
      modern canonical (NOT the anchor's source form, to keep the
      era ladder from looking reversed at era=modern).
    - No anchor → same fallback path; ``era_form=None``,
      ``fallback=True``, source_form set to the modern canonical so
      downstream consumers can still report 'no anchor for this
      morpheme'.
    - Anchor present but anchor.language already equals the target
      → short-circuit on the anchor's canonical_form (era_form set;
      ``fallback=False``).
    """
    canonical = meaning.usage.replace("-", "")
    particle = _is_free_particle(meaning)
    if anchor is None:
        return MorphemeRewind(
            canonical=canonical,
            source_form=canonical,
            source_language="modern-english",
            source_etymon_id=None,
            era_form=None,
            fallback=True,
            is_free_particle=particle,
        )

    target_language = canonical_language_for_cell(family, cell)
    if target_language is None:
        return MorphemeRewind(
            canonical=canonical,
            source_form=anchor.canonical_form,
            source_language=anchor.language,
            source_etymon_id=anchor.etymon_id,
            era_form=None,
            fallback=True,
            is_free_particle=particle,
        )

    # When the anchor's own language IS the target (era=oe-late and
    # anchor.language='old-english'), the anchor's canonical form is
    # the right answer — short-circuit and skip the cluster query.
    if anchor.language == target_language:
        return MorphemeRewind(
            canonical=canonical,
            source_form=anchor.canonical_form,
            source_language=anchor.language,
            source_etymon_id=anchor.etymon_id,
            era_form=anchor.canonical_form,
            fallback=False,
            is_free_particle=particle,
        )

    # wyrd-8qbi: at the modern-english target, the morpheme's
    # canonical (Meaning.usage with dash markers stripped) IS by
    # construction the modern reflex — that's what 'modern_usage'
    # means in the bundle. Running the cluster-mate picker here
    # adds noise rather than information: it surfaces sibling
    # cognates that happen to share an OE root but aren't the
    # morpheme the operator typed (the demo run produced 'Elmton'
    # -> 'Amherstton' at modern, picking the place-name cognate
    # 'Amherst' over the morpheme 'elm' itself). Short-circuit
    # to canonical so the modern cell round-trips the input.
    if target_language == "modern-english":
        return MorphemeRewind(
            canonical=canonical,
            source_form=anchor.canonical_form,
            source_language=anchor.language,
            source_etymon_id=anchor.etymon_id,
            era_form=canonical,
            fallback=False,
            is_free_particle=particle,
        )

    reflexes: list[EraReflex] = etymon_era_reflexes(
        db, anchor.etymon_id, target_family_cell=(family, cell)
    )
    if not reflexes:
        return MorphemeRewind(
            canonical=canonical,
            source_form=anchor.canonical_form,
            source_language=anchor.language,
            source_etymon_id=anchor.etymon_id,
            era_form=None,
            fallback=True,
            is_free_particle=particle,
        )
    # Three-tier picker preference, falling through alphabetical:
    #
    # Tier 1: morpheme.canonical match (modern usage in cluster)
    #   Handles mining-artifact cases where archaic forms like
    #   'cyning' get co-tagged as modern-english (Wiktionary
    #   cross-reference entries do this). Sorting alphabetically
    #   would surface 'cyning' as the modern reflex of OE 'king',
    #   breaking the era progression. Canonical-match picks 'king'.
    #
    # Tier 2: anchor.canonical_form match (source form in cluster)
    #   Handles morphemes whose orthography didn't shift much across
    #   eras (mynster → mynster → minster) — alphabetical-first
    #   would otherwise return a noise mate ('amounten' for OE
    #   'mynster' at ME). Empirically this affects ~822 OE clusters
    #   on the live corpus where the ME mate set includes
    #   peripheral entries that sort before the actual reflex.
    #
    # Tier 3: alphabetical first
    #   Final tiebreaker; deterministic.
    canonical_lower = canonical.lower()
    anchor_lower = anchor.canonical_form.lower()
    matching = [r for r in reflexes if r.form.lower() == canonical_lower]
    if not matching:
        matching = [r for r in reflexes if r.form.lower() == anchor_lower]
    selected = matching[0] if matching else reflexes[0]
    return MorphemeRewind(
        canonical=canonical,
        source_form=anchor.canonical_form,
        source_language=anchor.language,
        source_etymon_id=anchor.etymon_id,
        era_form=selected.form,
        fallback=False,
        is_free_particle=particle,
    )


def _pick_form(morpheme: MorphemeRewind) -> str:
    """Resolve which surface form to render for the compound.

    Era reflex when present; otherwise fall back to the morpheme's
    modern canonical (rendering the user's familiar reference rather
    than an arbitrary period form). The ``fallback=True`` flag on
    the morpheme tells the caller this rendering is a placeholder
    — the asterisk in the CLI output surfaces this so the user
    knows 'we don't have era data for this morpheme'.

    Why not fall back to the anchor's ``source_form``? At era=modern
    that would render the OE form, making the era ladder look like
    it goes backwards (oe-late: king → me: chinge → modern: cyning).
    Falling back to the modern usage instead keeps the visual
    progression monotone or absent (oe-late: king → me: chinge →
    modern: King*) — the asterisk is the signal, not a reversed
    spelling.

    Leading/trailing hyphens on the picked form are stripped — they
    encode the morpheme's location (pre-, post-, inner-) but the
    compound renderer joins morphemes with explicit hyphens, so
    keeping them on the form would emit double-hyphens like
    ``Had-en--ham``.
    """
    raw = morpheme.era_form or morpheme.canonical
    return raw.strip("-")


def supported_eras_for_family(family: str) -> tuple[str, ...]:  # noqa: V103 — public era-stop accessor pinned by tests
    """Return the cell labels available for ``family`` per ERA_CELLS.
    Lets callers (CLI, future SPA) enumerate valid era stops without
    duplicating the era_cells_for_family lookup."""
    return era_cells_for_family(family)
