"""Weighted random selection over learned morpheme proportions.

Ported from Rando (rando/proportions.py) — Python 2 → 3.

Key change beyond syntax: weighted_choice now takes an explicit Random instance
so generation is reproducible from the seed param.
"""

from __future__ import annotations

import random
from functools import lru_cache

from wyrd.generators.kenning.meaning import _mimic_case


class Generator:
    def __init__(self, tag_db, elements):
        self.tag_db = tag_db
        self.elements = elements

    def add_item(self, key, proportion):
        self.elements[key] = proportion

    def select(self, rng, *tags, novelty: float = 0.0, harshness: float = 0.0):
        """Pick one element, optionally blending toward a uniform marginal
        and/or biasing toward phonologically-harsh keys.

        Per D17, the runtime samples from a mixture
        ``(1-novelty)·empirical + novelty·uniform``: at ``novelty=0`` (default)
        sampling is pure empirical-frequency (current behavior); at
        ``novelty=1`` every key in the bucket is equally likely. Intermediate
        values softly blend, allowing plausible-but-unattested combinations
        without abandoning the corpus. The full D17 design layers a third
        tag-class-prior term, deferred to a follow-up since it requires
        threading neighbor-context through the structure walk.

        Per D6, ``harshness`` re-weights each candidate by its phonological
        harshness score (computed from the key string) — at ``harshness=0``
        every key keeps its empirical weight; at ``harshness=1`` soft-keyed
        items go to zero and harsh-keyed items keep 2× their original weight.
        Composition order: harshness is applied first to the empirical
        weights, then ``_blend_uniform`` blends the result with the uniform
        marginal. So ``--harsh 1 --novelty 1`` yields uniform sampling over
        the entire bucket (because ``_blend_uniform`` ignores the empirical
        side at novelty=1), while ``--harsh 1 --novelty 0`` is pure
        harsh-empirical.
        """
        items = self.elements.items()
        if len(tags) > 0:
            items = self.filter_for_tag(*tags).items()
        if len(items) == 0:
            return None
        items_list = list(items)
        if harshness <= 0 and novelty <= 0:
            return weighted_choice(rng, items_list)
        if harshness > 0:
            items_list = _blend_harsh(items_list, harshness)
        if novelty > 0:
            items_list = _blend_uniform(items_list, novelty)
        return weighted_choice(rng, items_list)

    def has_tag(self, *tags):
        return len(self.filter_for_tag(*tags)) > 0

    def filter_for_tag(self, *tags):
        keys: list[str] = []
        for tag in tags:
            keys.extend(self.tag_db.get(tag, []))
        keys_set = set(keys)
        return {key: self.elements[key] for key in keys_set if key in self.elements}


class MeaningGenerator:
    def __init__(self, meaning_db, tag_db, proportions):
        self.meaning_db = meaning_db
        self.tag_db = tag_db
        self.generators: dict[tuple, Generator] = {}
        self.load_parts(proportions)

    def load_parts(self, proportions, *addkeys):
        for usage, proportion in proportions.items():
            meanings = self.meaning_db[usage]
            keys = {m.key() for m in meanings}
            for key in keys:
                if addkeys:
                    key = tuple(list(key) + list(addkeys))
                gen = self.generators.setdefault(key, Generator(self.tag_db, {}))
                gen.add_item(usage, proportion)

    def select(self, rng, key, *tags, novelty: float = 0.0, harshness: float = 0.0):
        return self.generators[key].select(rng, *tags, novelty=novelty, harshness=harshness)


class NameGenerator:
    def __init__(self, meaning_db, meaning_gen, structs):
        self.meaning_db = meaning_db
        self.meaning_gen = meaning_gen
        self.structs = structs

    def select(
        self,
        rng,
        *tags,
        spelling_variety: float = 0.0,
        novelty: float = 0.0,
        inflection_density: float = 0.0,
        harshness: float = 0.0,
    ):
        """Pick a structure, fill it with morpheme usages, optionally render
        each usage as an attested archaic spelling variant (D18) or an
        inflected form (D8), and blend sampling toward a uniform marginal
        (D17) or harsh-keyed empirical (D6).

        ``spelling_variety`` is the probability *per morpheme* that the
        rendered surface form is drawn from the etymon's variant pool
        rather than the canonical reflex.

        ``novelty`` (D17) blends each morpheme-bucket's empirical-frequency
        distribution with a uniform marginal over the same keys.

        ``inflection_density`` (D8) is the probability *per morpheme* that
        the rendered surface form is drawn from the lemma family's
        inflected-children pool ('cot' → 'cotum', 'cotan', 'cotes'). The
        inflection's grammatical-case label is preserved in
        ``new_name.inflection_labels`` for the explainer.

        ``harshness`` (D6) re-weights each morpheme-bucket toward
        phonologically-harsh keys (stop-final, cluster-heavy). Composes
        orthogonally with ``novelty``: the harsh skew applies to empirical
        weights first, then ``novelty`` blends the result with the uniform
        marginal.

        When both inflection_density and spelling_variety would fire on the
        same morpheme, inflection wins — it carries grammatical meaning
        that the variant axis doesn't.
        """
        items = list(self.structs.items())
        struct = weighted_choice(rng, items)
        if len(tags) == 0:
            new_name = self._select_no_tag(rng, struct, novelty=novelty, harshness=harshness)
        else:
            new_name = self._select_tags(rng, struct, *tags, novelty=novelty, harshness=harshness)
        if spelling_variety > 0 or inflection_density > 0:
            rendered, labels = self._render_substitutions(
                rng, new_name.name, spelling_variety, inflection_density
            )
            new_name.rendered = rendered
            new_name.inflection_labels = labels
        return new_name

    def _render_substitutions(self, rng, name, spelling_variety, inflection_density):
        """Walk the picked usages and produce two parallel lists: surface
        forms (with D8 inflection or D18 variant substitution applied) and
        inflection labels (for the explainer; None where no inflection
        was picked).

        Per-morpheme rolling order: inflection first (more specific
        morphological data), then variant if no inflection fired. Both pass
        gates are independent draws; inflection wins ties because it
        carries grammatical case while the variant axis is purely
        spelling.
        """
        rendered: list[list[str | None]] = []
        labels: list[list[str | None]] = []
        for word in name:
            word_rendered: list[str | None] = []
            word_labels: list[str | None] = []
            for usage in word:
                if usage is None:
                    word_rendered.append(None)
                    word_labels.append(None)
                    continue
                meanings = self.meaning_db.get(usage) or []
                # When a usage maps to multiple Meanings (different senses
                # e.g. -y "district" vs "island"), each sense has its OWN
                # variant + inflection pools because the underlying etymons
                # are distinct. We pick meanings[0] as a deterministic
                # fallback; the picked substitution may not correspond to
                # the sense the structure's tags would have selected.
                # Acceptable trade-off: sampling at the usage level (not
                # the sense level) keeps the runtime simple and seed-stable.
                # A follow-up could thread the structure's tag context to
                # narrow to the matching sense before drawing.
                chosen = meanings[0] if meanings else None
                surface, label = self._pick_surface(
                    rng, usage, chosen, spelling_variety, inflection_density
                )
                word_rendered.append(surface)
                word_labels.append(label)
            rendered.append(word_rendered)
            labels.append(word_labels)
        return rendered, labels

    def _pick_surface(self, rng, usage, chosen, spelling_variety, inflection_density):
        """Decide one element's surface form and inflection label. Returns
        (surface_string, inflection_label_or_None) — the surface is always
        non-None given a non-None usage."""
        canonical = usage.replace("-", "")
        if chosen is None:
            return canonical, None
        if inflection_density > 0:
            picked = chosen.pick_inflection(rng, inflection_density)
            if picked is not None:
                form, label = picked
                return _mimic_case(usage, form), label
        if spelling_variety > 0:
            variant = chosen.pick_variant(rng, spelling_variety)
            if variant is not None:
                return _mimic_case(usage, variant), None
        return canonical, None

    def _select_no_tag(self, rng, struct, *, novelty: float = 0.0, harshness: float = 0.0):
        words = []
        for w in struct:
            keys = []
            for key in w:
                keys.append(self.meaning_gen.select(rng, key, novelty=novelty, harshness=harshness))
            words.append(keys)
        return NewName(struct, self.meaning_db, words)

    def _select_tags(self, rng, struct, *tags, novelty: float = 0.0, harshness: float = 0.0):
        name_pool = [self._select_no_tag(rng, struct, novelty=novelty, harshness=harshness)]
        for tag in tags:
            name_pool.append(
                self._select_tag(rng, struct, tag, novelty=novelty, harshness=harshness)
            )
        words = []
        for i in range(len(struct)):
            keys = []
            for j in range(len(struct[i])):
                pool = []
                for elem in name_pool:
                    e = elem.name[i][j]
                    if e:
                        pool.append(e)
                if not pool:
                    keys.append(None)
                else:
                    keys.append(rng.choice(pool))
            words.append(keys)
        return NewName(struct, self.meaning_db, words)

    def _select_tag(self, rng, struct, tag, *, novelty: float = 0.0, harshness: float = 0.0):
        words = []
        for w in struct:
            keys = []
            for key in w:
                keys.append(
                    self.meaning_gen.select(rng, key, tag, novelty=novelty, harshness=harshness)
                )
            words.append(keys)
        return NewName(struct, self.meaning_db, words)


class NewName:
    def __init__(self, struct, meaning_db, name, rendered=None, inflection_labels=None):
        self.struct = struct
        self.meaning_db = meaning_db
        self.name = name
        # ``rendered`` is a list-of-lists parallel to ``name`` carrying the
        # surface form to emit per element, with D18 variant substitution
        # or D8 inflection substitution already applied. None means "fall
        # back to dash-stripped usage" (today's bit-stable behavior).
        self.rendered = rendered
        # ``inflection_labels`` is a list-of-lists parallel to ``name`` carrying
        # the grammatical-case label per element when an inflected form was
        # picked at render time, else None. The explainer reads this to
        # surface 'cot@dative_or_pl' breakdowns (D8). None on the outer
        # list means no inflection rendering was performed at all.
        self.inflection_labels = inflection_labels

    def __str__(self):
        words = []
        for wi, w in enumerate(self.name):
            for ei, e in enumerate(w):
                if e is None:
                    continue
                if self.rendered is not None and self.rendered[wi][ei] is not None:
                    words.append(self.rendered[wi][ei])
                else:
                    words.append(e.replace("-", ""))
            words.append(" ")
        return "".join(words).strip()

    def __repr__(self):
        words = []
        for w in self.name:
            for e in w:
                if e is None:
                    continue
                words.append(self.meaning_db[e])
        return repr(words)

    def description(self):
        results = []
        for wi, word in enumerate(self.name):
            single = len(word) == 1
            for ei, e in enumerate(word):
                if e is None:
                    continue
                lemma = e.replace("-", "") if single else e
                label = self._inflection_label_for(wi, ei)
                head = f"{lemma}@{label}" if label else lemma
                meanings = [self._find_meaning(m) for m in self.meaning_db[e]]
                citations = _collect_citations(self.meaning_db[e])
                gloss_block = " or ".join(meanings)
                if citations:
                    cite_block = f"; cited by {_format_citations_for_description(citations)}"
                else:
                    cite_block = ""
                results.append(f"{head} ({gloss_block}{cite_block})")
        return " ".join(results)

    def _inflection_label_for(self, wi: int, ei: int) -> str | None:
        if self.inflection_labels is None:
            return None
        try:
            return self.inflection_labels[wi][ei]
        except IndexError:
            return None

    def components(self):
        """Structured component breakdown for the API envelope."""
        out = []
        for word in self.name:
            for e in word:
                if e is None:
                    continue
                meanings = self.meaning_db[e]
                first = meanings[0]
                out.append(
                    {
                        "usage": e,
                        "location": first.location,
                        "meanings": list(first.meanings),
                        "tags": list(first.tags),
                        "roots": self._roots(first),
                        "citations": _collect_citations(meanings),
                    }
                )
        return out

    def _find_meaning(self, meaning):
        roots = self._roots_str(meaning)
        return f"{roots} {', '.join(meaning.meanings)}"

    def _roots(self, meaning) -> list[str]:
        keys = [
            ("old_english", "EN"),
            ("old_scandinavian", "SC"),
            ("old_french", "FR"),
            ("celtic_mix", "CL"),
            ("latin", "LA"),
            ("germanic", "GE"),
            ("greek", "GR"),
        ]
        return [code for src, code in keys if src in meaning.sources]

    def _roots_str(self, meaning) -> str:
        return "/".join(self._roots(meaning))


def _collect_citations(meanings):
    """Distinct sorted source_ids across every Meaning sharing one usage and
    every language they cover (wyrd-9kh.1). The runtime explainer surfaces
    this so a GM can see which scholars attest the morpheme. Empty list
    when no Meaning carries citations (rando-port-only legacy entries with
    no scholarly witnesses yet)."""
    seen: set[str] = set()
    for m in meanings:
        for citations in m.citations.values():
            seen.update(citations)
    return sorted(seen)


# Compact display max for description()'s citation block. Above this, the
# explainer renders 'first, second, third (+N more)' rather than a wall of
# 18 source_ids. components() keeps the full list so the SPA can render
# its own disclosure UI.
_DESCRIPTION_CITATION_LIMIT = 3


def _format_citations_for_description(citations: list[str]) -> str:
    """Render a short scholar-ID list for the explainer breakdown line.
    Long lists collapse to 'first, second, third (+N more)'; lists at or
    under the limit display in full. source_ids stay as-is — the SPA
    layer (wyrd-9kh.6) substitutes prettier titles via the source table."""
    if len(citations) <= _DESCRIPTION_CITATION_LIMIT:
        return ", ".join(citations)
    head = ", ".join(citations[:_DESCRIPTION_CITATION_LIMIT])
    extra = len(citations) - _DESCRIPTION_CITATION_LIMIT
    return f"{head} (+{extra} more)"


def word_to_key(word):
    elements = []
    for element in word:
        key = [element["location"]]
        if element.get("name", False):
            key.append("name")
        if element.get("saint", False):
            key.append("saint")
        elements.append(key)
    if len(word) == 1:
        elements[0].append("single")
    return tuple(tuple(e) for e in elements)


def load_proportions(data, meaning_db, tag_db):
    usages = data["usages"]
    mg = MeaningGenerator(meaning_db, tag_db, usages)
    single_usages = data["single_usages"]
    mg.load_parts(single_usages, "single")
    structures = data["structures"]
    struct = {}
    for element in structures:
        proportion = element["proportion"]
        words = tuple(word_to_key(w) for w in element["words"])
        struct[words] = proportion
    return NameGenerator(meaning_db, mg, struct)


def _blend_uniform(items, novelty: float):
    """Blend each (item, weight) pair with a uniform marginal.

    Returns a new list of (item, blended_weight) where blended_weight is
    ``(1-novelty)·normalized_empirical + novelty·uniform_share``. The result
    is a normalized distribution (sums to ~1.0); weighted_choice handles
    fractional weights fine via its uniform draw over the cumulative range.

    novelty must be > 0 (callers fast-path the novelty=0 case to avoid
    re-allocating). Items with weight <= 0 are stripped from the empirical
    side but still get their uniform share — without that, novelty=1 would
    silently undercount keys that happened to have zero empirical weight.
    """
    if not items:
        return items
    n = len(items)
    total = sum(max(w, 0) for _, w in items)
    if total <= 0:
        # Pure uniform if every empirical weight is zero — there's nothing to
        # blend against, so the novelty knob has no meaningful axis. Returning
        # 1/n keeps the result a normalized probability distribution as the
        # docstring promises.
        return [(k, 1 / n) for k, _ in items]
    return [(k, (1 - novelty) * (max(w, 0) / total) + novelty / n) for k, w in items]


# D6 --harsh: phonological harshness scoring. Heuristic, not phonotactic.
# Goal is order-of-magnitude separation between vowel-soft morphemes
# ('ham', 'baron', 'borough') and stop-cluster-heavy ones ('shuck', 'crag',
# 'fork'). The exact ranking isn't load-bearing — only relative ordering
# matters for the per-item multiplier. Common OE/ON long-vowel diacritics
# count as vowels so 'tūn' / 'lēah' don't get falsely classified.
_VOWELS = frozenset("aeiouyæāēīōūȳáéíóúýǣǿǫâêîôûŵŷ")
_VOICED_STOPS = frozenset("bdg")
_VOICELESS_STOPS = frozenset("pktc")
_NASALS = frozenset("nm")


@lru_cache(maxsize=4096)
def _harshness_score(usage: str) -> float:
    """Phonological harshness in [0, 1]. Higher = more menacing-feeling.

    Cached: the bundle has ~1600 unique modern_usages and `_blend_harsh`
    runs the score over every key on every bucket on every sample, so
    the same string is hit thousands of times across one batch of name
    generations. lru_cache pins the per-string cost at one pass.

    Combines three signals on the dash-stripped lowercased usage:

    - Coda harshness: stops at the end (b/d/g, p/t/k, c) score 1.0; nasals
      0.4; vowels 0.0; everything else (l/r/s/h/f/etc.) 0.5.
    - Cluster density: fraction of adjacent-consonant positions.
    - Consonant density: non-vowel chars / total chars.

    Composition is a simple weighted sum (45/35/20) chosen so coda dominates
    perception ('cot' > 'co'), with cluster + density as tiebreakers. This
    is meant for a sampling reweight, not for serious phonological work.
    """
    s = "".join(c for c in usage.lower() if c.isalpha())
    if not s:
        return 0.5
    n = len(s)

    last = s[-1]
    if last in _VOICED_STOPS or last in _VOICELESS_STOPS:
        coda_score = 1.0
    elif last in _NASALS:
        coda_score = 0.4
    elif last in _VOWELS:
        coda_score = 0.0
    else:
        coda_score = 0.5

    cluster_count = 0
    consonant_count = 0
    for i, c in enumerate(s):
        if c not in _VOWELS:
            consonant_count += 1
            if i > 0 and s[i - 1] not in _VOWELS:
                cluster_count += 1
    cluster_score = cluster_count / max(1, n - 1)
    consonant_density = consonant_count / n

    return min(1.0, 0.45 * coda_score + 0.35 * cluster_score + 0.2 * consonant_density)


def _blend_harsh(items, harshness: float):
    """Re-weight items by phonological harshness score.

    Each (key, weight) becomes (key, weight * (1 - harshness + harshness *
    2 * score)) where score ∈ [0, 1] is computed from the key string. At
    ``harshness=0`` every multiplier is 1.0 (bit-stable); at ``harshness=1``
    soft-keyed items go to weight 0 and harsh-keyed items get 2× their
    empirical weight.

    The 2× scaling is deliberate: an absolute multiplier of 0..1 would
    mean even harsh=1 only halves the mean weight, leaving soft items
    competitive. 0..2 puts soft items at 0 at the harshness=1 limit.

    Callers should fast-path ``harshness <= 0`` to avoid re-allocating.
    """
    if not items:
        return items
    return [
        (k, max(w, 0) * (1 - harshness + harshness * 2 * _harshness_score(k))) for k, w in items
    ]


def weighted_choice(rng: random.Random, choices):
    """Like rng.choice, but each element can have a different weight.

    `choices` is an iterable of (item, weight) pairs. Items with weight <= 0
    are skipped. Returns None if no items have positive weight.
    """
    space = {}
    current = 0
    for choice, weight in choices:
        if weight > 0:
            space[current] = choice
            current += weight
    if current == 0:
        return None
    rand = rng.uniform(0, current)
    last_choice = None
    for key in sorted(list(space.keys()) + [current]):
        if rand < key:
            return last_choice
        if key in space:
            last_choice = space[key]
    return last_choice
