"""Weighted random selection over learned morpheme proportions.

Ported from Rando (rando/proportions.py) — Python 2 → 3.

Key change beyond syntax: weighted_choice now takes an explicit Random instance
so generation is reproducible from the seed param.
"""

from __future__ import annotations

import random


class Generator:
    def __init__(self, tag_db, elements):
        self.tag_db = tag_db
        self.elements = elements

    def add_item(self, key, proportion):
        self.elements[key] = proportion

    def select(self, rng, *tags):
        items = self.elements.items()
        if len(tags) > 0:
            items = self.filter_for_tag(*tags).items()
        if len(items) == 0:
            return None
        return weighted_choice(rng, list(items))

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

    def select(self, rng, key, *tags):
        return self.generators[key].select(rng, *tags)


class NameGenerator:
    def __init__(self, meaning_db, meaning_gen, structs):
        self.meaning_db = meaning_db
        self.meaning_gen = meaning_gen
        self.structs = structs

    def select(self, rng, *tags):
        items = list(self.structs.items())
        struct = weighted_choice(rng, items)
        if len(tags) == 0:
            return self._select_no_tag(rng, struct)
        return self._select_tags(rng, struct, *tags)

    def _select_no_tag(self, rng, struct):
        words = []
        for w in struct:
            keys = []
            for key in w:
                keys.append(self.meaning_gen.select(rng, key))
            words.append(keys)
        return NewName(struct, self.meaning_db, words)

    def _select_tags(self, rng, struct, *tags):
        name_pool = [self._select_no_tag(rng, struct)]
        for tag in tags:
            name_pool.append(self._select_tag(rng, struct, tag))
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

    def _select_tag(self, rng, struct, tag):
        words = []
        for w in struct:
            keys = []
            for key in w:
                keys.append(self.meaning_gen.select(rng, key, tag))
            words.append(keys)
        return NewName(struct, self.meaning_db, words)


class NewName:
    def __init__(self, struct, meaning_db, name):
        self.struct = struct
        self.meaning_db = meaning_db
        self.name = name

    def __str__(self):
        words = []
        for w in self.name:
            for e in w:
                if e is None:
                    continue
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
