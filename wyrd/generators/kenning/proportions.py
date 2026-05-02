"""Weighted random selection over learned morpheme proportions.

Ported from Rando (rando/proportions.py) — Python 2 → 3.

Key change beyond syntax: weighted_choice now takes an explicit Random instance
so generation is reproducible from the seed param.
"""

from __future__ import annotations

import random

from wyrd.generators.kenning.meaning import _mimic_case


class Generator:
    def __init__(self, tag_db, elements):
        self.tag_db = tag_db
        self.elements = elements

    def add_item(self, key, proportion):
        self.elements[key] = proportion

    def select(self, rng, *tags, novelty: float = 0.0):
        """Pick one element, optionally blending toward a uniform marginal.

        Per D17, the runtime samples from a mixture
        ``(1-novelty)·empirical + novelty·uniform``: at ``novelty=0`` (default)
        sampling is pure empirical-frequency (current behavior); at
        ``novelty=1`` every key in the bucket is equally likely. Intermediate
        values softly blend, allowing plausible-but-unattested combinations
        without abandoning the corpus. The full D17 design layers a third
        tag-class-prior term, deferred to a follow-up since it requires
        threading neighbor-context through the structure walk.
        """
        items = self.elements.items()
        if len(tags) > 0:
            items = self.filter_for_tag(*tags).items()
        if len(items) == 0:
            return None
        items_list = list(items)
        if novelty <= 0:
            return weighted_choice(rng, items_list)
        return weighted_choice(rng, _blend_uniform(items_list, novelty))

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

    def select(self, rng, key, *tags, novelty: float = 0.0):
        return self.generators[key].select(rng, *tags, novelty=novelty)


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
    ):
        """Pick a structure, fill it with morpheme usages, optionally render
        each usage as an attested archaic spelling variant (D18) or an
        inflected form (D8), and blend sampling toward a uniform marginal
        (D17).

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

        When both inflection_density and spelling_variety would fire on the
        same morpheme, inflection wins — it carries grammatical meaning
        that the variant axis doesn't.
        """
        items = list(self.structs.items())
        struct = weighted_choice(rng, items)
        if len(tags) == 0:
            new_name = self._select_no_tag(rng, struct, novelty=novelty)
        else:
            new_name = self._select_tags(rng, struct, *tags, novelty=novelty)
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
        if chosen is None:
            return usage.replace("-", ""), None
        if inflection_density > 0:
            picked = chosen.pick_inflection(rng, inflection_density)
            if picked is not None:
                form, label = picked
                return _mimic_case(usage, form), label
        if spelling_variety > 0:
            variant = chosen.pick_variant(rng, spelling_variety)
            if variant is not None:
                return _mimic_case(usage, variant), None
        return usage.replace("-", ""), None

    def _select_no_tag(self, rng, struct, *, novelty: float = 0.0):
        words = []
        for w in struct:
            keys = []
            for key in w:
                keys.append(self.meaning_gen.select(rng, key, novelty=novelty))
            words.append(keys)
        return NewName(struct, self.meaning_db, words)

    def _select_tags(self, rng, struct, *tags, novelty: float = 0.0):
        name_pool = [self._select_no_tag(rng, struct, novelty=novelty)]
        for tag in tags:
            name_pool.append(self._select_tag(rng, struct, tag, novelty=novelty))
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

    def _select_tag(self, rng, struct, tag, *, novelty: float = 0.0):
        words = []
        for w in struct:
            keys = []
            for key in w:
                keys.append(self.meaning_gen.select(rng, key, tag, novelty=novelty))
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
        for word in self.name:
            single = len(word) == 1
            for e in word:
                if e is None:
                    continue
                part = [e.replace("-", "") if single else e, " ("]
                meanings = [self._find_meaning(m) for m in self.meaning_db[e]]
                part.append(" or ".join(meanings))
                part.append(")")
                results.append("".join(part))
        return " ".join(results)

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
