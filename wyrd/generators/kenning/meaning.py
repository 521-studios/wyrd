"""Morpheme model: a Meaning is a piece of a word like 'Bre-' or '-combe'.

Ported from Rando (rando/meaning.py) — Python 2 → 3.

Variants (D18) ride along on the Meaning so generation can optionally
emit an attested archaic spelling instead of the modern reflex.
"""

from __future__ import annotations

import random

# Suffix used in meanings.json to mark per-language variant pools, e.g.
# "old_english" canonical forms have their variants in "old_english_variants".
# Matches the export-meanings emit shape in lexicon.py:_emit_variant_list.
_VARIANT_SUFFIX = "_variants"


class Meaning:
    def __init__(self, usage, tags, meanings, sources, variants=None):
        self.usage = usage
        self.tags = tags
        self.meanings = meanings
        self.sources = sources
        # variants is a dict[lang_field, list[(form, weight)]] keyed by the
        # JSON field name (e.g. "old_english", "celtic_mix"). Optional —
        # legacy meanings.json data has no variants; the field is empty.
        self.variants = variants or {}
        self._set_location()

    def _set_location(self):
        if self.usage.startswith("-") and self.usage.endswith("-"):
            self.location = "inner"
        elif self.usage.endswith("-"):
            self.location = "pre"
        else:
            self.location = "post"

    def __str__(self):
        return self.usage.lower().replace("-", "")

    def __repr__(self):
        return (
            f"{{usage: {self.usage}, tags: {self.tags}, meanings: {self.meanings}, "
            f"sources: {self.sources}, location: {self.location}}}"
        )

    def word_has_meaning(self, word):
        return word.find(str(self)) > -1

    def test(self, word):
        # str(self) lowercases the usage, so all matching is case-insensitive.
        # Slice (not str.replace) to strip the matched span without depending on
        # the original word's casing — fixes a bug from the original Rando port
        # where "Hill" matched post-Meaning("-hill") but left "Hill" as residue
        # because lowercase replace() couldn't find uppercase "H".
        token = str(self)
        lower = word.lower()
        n = len(token)
        if self.location == "pre":
            if lower.startswith(token):
                return [self, word[n:]]
        elif self.location == "post":
            if lower.endswith(token):
                return [word[: len(word) - n], self]
        else:
            idx = lower.find(token)
            if idx > -1:
                return [word[:idx], self, word[idx + n :]]
        return None

    def is_name(self):
        return any(tag in ("female name", "male name", "family name") for tag in self.tags)

    def is_saint(self):
        return "saint" in self.tags

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


def load_meanings(data):
    meaning_db: dict[str, list[Meaning]] = {}
    tags_db: dict[str, list[str]] = {}
    for subject in data:
        tags = subject["modifier_tags"]
        meanings = subject["meaning"]
        for word in subject["words"]:
            usage = word["modern_usage"]
            # Split the word's fields into (a) language form arrays the runtime
            # already used, (b) variant pools added by D18. Variants come in
            # under "<lang>_variants" keys and shouldn't bleed into `sources`.
            sources = {
                k: v
                for k, v in word.items()
                if k != "modern_usage" and not k.endswith(_VARIANT_SUFFIX)
            }
            variants = {
                k[: -len(_VARIANT_SUFFIX)]: [(entry["form"], entry["weight"]) for entry in v]
                for k, v in word.items()
                if k.endswith(_VARIANT_SUFFIX)
            }
            meaning = Meaning(usage, tags, meanings, sources, variants=variants)
            for tag in tags:
                tags_db.setdefault(tag, []).append(usage)
            meaning_db.setdefault(usage, []).append(meaning)
            if meaning.is_name() and not usage.endswith("s"):
                plural = f"{usage}s"
                plural_meaning = Meaning(plural, tags, meanings, sources, variants=variants)
                for tag in tags:
                    tags_db.setdefault(tag, []).append(plural)
                meaning_db.setdefault(plural, []).append(plural_meaning)
    return meaning_db, tags_db
