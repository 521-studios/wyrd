"""Morpheme model: a Meaning is a piece of a word like 'Bre-' or '-combe'.

Ported from Rando (rando/meaning.py) — Python 2 → 3.
"""

from __future__ import annotations


class Meaning:
    def __init__(self, usage, tags, meanings, sources):
        self.usage = usage
        self.tags = tags
        self.meanings = meanings
        self.sources = sources
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
        token = str(self)
        if self.location == "pre":
            if word.lower().startswith(token):
                return [self, word.lower().replace(token, "")]
        elif self.location == "post":
            if word.lower().endswith(token):
                return [word.replace(token, ""), self]
        else:
            if word.find(token) > -1:
                parts = word.split(token)
                return [parts[0], self, parts[1]]
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


def load_meanings(data):
    meaning_db: dict[str, list[Meaning]] = {}
    tags_db: dict[str, list[str]] = {}
    for subject in data:
        tags = subject["modifier_tags"]
        meanings = subject["meaning"]
        for word in subject["words"]:
            usage = word["modern_usage"]
            sources = {k: v for k, v in word.items() if k != "modern_usage"}
            meaning = Meaning(usage, tags, meanings, sources)
            for tag in tags:
                tags_db.setdefault(tag, []).append(usage)
            meaning_db.setdefault(usage, []).append(meaning)
            if meaning.is_name() and not usage.endswith("s"):
                plural = f"{usage}s"
                plural_meaning = Meaning(plural, tags, meanings, sources)
                for tag in tags:
                    tags_db.setdefault(tag, []).append(plural)
                meaning_db.setdefault(plural, []).append(plural_meaning)
    return meaning_db, tags_db
