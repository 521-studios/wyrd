"""Kenning — town-name generator. Compounds Old English / Norse / Celtic morphemes."""

from __future__ import annotations

import itertools
import json
from functools import lru_cache
from importlib import resources
from typing import Any

from wyrd.generators.kenning.meaning import Meaning, load_meanings
from wyrd.generators.kenning.name import Name
from wyrd.generators.kenning.proportions import load_proportions
from wyrd.generators.kenning.word import Word
from wyrd.registry import GenerationResult, Generator, register
from wyrd.seed import rng_for

_LEGEND = [
    {"code": "EN", "name": "Old English"},
    {"code": "SC", "name": "Old Scandinavian"},
    {"code": "FR", "name": "Old French"},
    {"code": "CL", "name": "Celtic"},
    {"code": "LA", "name": "Latin"},
    {"code": "GE", "name": "Germanic"},
    {"code": "GR", "name": "Greek"},
]

# Hard ceiling on decomposition results. A short word can produce many
# overlapping morpheme matches, and the SPA renders one item per result; cap
# so a pathological input doesn't dump a hundred near-duplicates.
_MAX_DECOMPOSITIONS = 25

_ROOT_CODES = [
    ("old_english", "EN"),
    ("old_scandinavian", "SC"),
    ("old_french", "FR"),
    ("celtic_mix", "CL"),
    ("latin", "LA"),
    ("germanic", "GE"),
    ("greek", "GR"),
]

CULTURES = ["english", "scottish", "welsh", "irish", "breton"]

# Tags that are filtering primitives, not meaningful selections to expose.
_INTERNAL_TAGS = {"male name", "female name", "saint"}


def _data_path(filename: str):
    return resources.files("wyrd.generators.kenning.data").joinpath(filename)


@lru_cache(maxsize=1)
def _load_meanings():
    with _data_path("meanings.json").open() as f:
        return load_meanings(json.load(f))


@lru_cache(maxsize=len(CULTURES))
def _load_culture(culture: str):
    if culture not in CULTURES:
        raise ValueError(f"unknown culture: {culture}; expected one of {CULTURES}")
    meaning_db, tag_db = _load_meanings()
    with _data_path(f"{culture}_proportions.json").open() as f:
        proportions = json.load(f)
    return load_proportions(proportions, meaning_db, tag_db), tag_db


def available_tags() -> list[str]:
    """User-visible tags from the meaning DB (excludes internal filtering tags)."""
    _, tag_db = _load_meanings()
    return sorted(t for t in tag_db if t not in _INTERNAL_TAGS)


class Kenning(Generator):
    name = "kenning"
    display_name = "Kenning — Town Names"
    description = (
        "Generates British Isles–style town names by composing Old English, Old Norse, "
        "Old French, and Celtic morphemes. Pick a culture; optionally filter morphemes "
        "by tag (e.g. 'tree', 'water', 'religion')."
    )
    details = (
        "<p>"
        "Town names from the British Isles aren't arbitrary — they're stitched from "
        "old <strong>morphemes</strong>, the small meaning-bearing fragments inside "
        "a word. Place-name scholars call the names themselves <strong>toponyms</strong>; "
        'the morphemes are their building blocks. <em>Ashton</em> is "ash" + "-ton" '
        '(Old English for an enclosed settlement). <em>Bridgwater</em> is "bridge" + '
        '"water". The vocabulary is bounded; the combinations are nearly endless.'
        "</p>"
        "<p>"
        "Kenning learned the patterns by analyzing roughly "
        "<strong>66,000 real British Isles place names</strong> "
        "(English, Scottish, Welsh, Irish) against a corpus of about "
        "2,900 morphemes. For each culture it knows which morphemes show up, "
        "in which structures (prefix + root, two words, saint's name, etc.), "
        "and how often. Each rolled name is a fresh sample from those statistics."
        "</p>"
    )
    legend = _LEGEND

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "culture": {
                    "type": "string",
                    "enum": CULTURES,
                    "default": "english",
                    "description": "Linguistic culture to draw morphemes and structures from.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": available_tags()},
                    "default": [],
                    "description": (
                        "Optional tag filters. Each tag biases the name toward morphemes "
                        "with that meaning category."
                    ),
                },
                "count": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "How many names to generate (1–10).",
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional 64-bit seed for reproducible output.",
                },
                "spelling_variety": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Per-morpheme probability of substituting an attested archaic "
                        "spelling variant for the canonical reflex (D18). 0 keeps the "
                        "modern surface form; higher values mix in 19th-century "
                        "scholarly spellings (e.g. 'Brycg' for 'Bridg-') for archaic "
                        "feel. Variant pool is empty for most morphemes today, so the "
                        "knob has limited reach until more mining lands."
                    ),
                },
                "novelty": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Mixture between empirical-frequency sampling and a uniform "
                        "marginal (D17). 0 keeps today's bit-stable behavior, 1 makes "
                        "every in-bucket morpheme equally likely — plausible-but-"
                        "unattested combinations become possible without abandoning "
                        "the corpus. Intermediate values softly blend."
                    ),
                },
            },
            "required": [],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        culture = params.get("culture", "english")
        raw_tags = params.get("tags", []) or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        tags = tuple(raw_tags)
        spelling_variety = float(params.get("spelling_variety", 0.0) or 0.0)
        novelty = float(params.get("novelty", 0.0) or 0.0)

        name_gen, _ = _load_culture(culture)
        rng = rng_for(seed)
        new_name = name_gen.select(
            rng,
            *tags,
            spelling_variety=spelling_variety,
            novelty=novelty,
        )
        return GenerationResult(
            result=str(new_name),
            explanation=new_name.description(),
            components=new_name.components(),
        )


register(Kenning())


def _roots(meaning: Meaning) -> list[str]:
    return [code for src, code in _ROOT_CODES if src in meaning.sources]


def _all_roots(meanings: list[Meaning]) -> list[str]:
    """Union of root codes across every Meaning sharing one usage. Order
    follows _ROOT_CODES so output is stable."""
    seen: set[str] = set()
    for m in meanings:
        seen.update(_roots(m))
    return [code for _, code in _ROOT_CODES if code in seen]


def _all_senses(meanings: list[Meaning]) -> list[str]:
    """Distinct sense strings across every Meaning sharing one usage,
    preserving first-seen order."""
    return list(dict.fromkeys(sense for m in meanings for sense in m.meanings))


def _build_explanation_part(chunk, meaning_db: dict[str, list[Meaning]]) -> str:
    if isinstance(chunk, Meaning):
        siblings = meaning_db.get(chunk.usage, [chunk])
        fragment = chunk.usage.replace("-", "")
        roots = _all_roots(siblings)
        roots_str = "/".join(roots) if roots else "?"
        senses = " / ".join(_all_senses(siblings))
        return f'{fragment} "{senses}" ({roots_str})'
    return f"[{chunk}]"


def _build_component_part(chunk, meaning_db: dict[str, list[Meaning]]) -> dict[str, Any]:
    if isinstance(chunk, Meaning):
        siblings = meaning_db.get(chunk.usage, [chunk])
        tags = list(dict.fromkeys(tag for m in siblings for tag in m.tags))
        return {
            "type": "matched",
            "fragment": chunk.usage.replace("-", ""),
            "usage": chunk.usage,
            "location": chunk.location,
            "meanings": _all_senses(siblings),
            "tags": tags,
            "roots": _all_roots(siblings),
        }
    return {"type": "unaccounted", "fragment": chunk}


def _decomposition_signature(words: tuple[Word, ...]) -> tuple:
    """Structural fingerprint: usages and unaccounted strings, in order. Used
    to dedupe decompositions that differ only by which Meaning instance was
    selected for a usage with multiple senses."""
    sig: list[tuple[str, str]] = []
    for word in words:
        for chunk in word.word:
            if isinstance(chunk, Meaning):
                sig.append(("M", chunk.usage))
            elif isinstance(chunk, str) and chunk:
                sig.append(("S", chunk))
        sig.append(("|", ""))
    return tuple(sig)


def _build_decomposition_result(
    name_str: str, words, meaning_db: dict[str, list[Meaning]]
) -> GenerationResult:
    explanation_parts: list[str] = []
    components: list[dict[str, Any]] = []
    for word in words:
        word_parts: list[dict[str, Any]] = []
        for chunk in word.word:
            if isinstance(chunk, str) and not chunk:
                continue
            explanation_parts.append(_build_explanation_part(chunk, meaning_db))
            word_parts.append(_build_component_part(chunk, meaning_db))
        components.append({"word": str(word), "parts": word_parts})
    return GenerationResult(
        result=name_str,
        explanation=" + ".join(explanation_parts) or "no morphemes recognized",
        components=components,
    )


class KenningExplain(Generator):
    # Co-located in the `kenning` package rather than `wyrd/generators/kenning_explain/`
    # because the explainer shares the meanings DB and decomposition machinery with
    # the main Kenning generator. The SPA path / API name `kenning-explain` does not
    # match a package directory, so `wyrd/cli.py:_mount_generator_clis()` cannot
    # locate a matching `cli.py` for it. This is intentional: the `explain`
    # subcommand on `wyrd kenning ...` is the CLI surface; the silent miss in the
    # mounter is the documented trade-off for sharing data with `Kenning`.
    name = "kenning-explain"
    display_name = "Kenning — Explain a Name"
    description = (
        "Decompose a real or invented British Isles place name into the morphemes "
        "Kenning recognizes. Returns every matching reading; unrecognized fragments "
        "are flagged."
    )
    details = (
        "<p>"
        "Paste any town name from the British Isles — or invent your own — and "
        "Kenning will break it apart into the <strong>morphemes</strong> it "
        "recognizes: small meaning-bearing fragments like <em>aber-</em>, "
        "<em>-ton</em>, <em>-combe</em>. Multiple readings? You'll see them all."
        "</p>"
        "<p>"
        "Pieces flagged as <strong>unrecognized</strong> are gaps — fragments "
        "that real names use but our morpheme corpus doesn't yet know. They're "
        "a research target for expanding the dataset."
        "</p>"
    )
    legend = _LEGEND
    multi_result = True

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Town name to decompose, e.g. 'Bridgwater'.",
                },
            },
            "required": ["name"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        # multi_result generators are dispatched through generate_all; this
        # exists only to satisfy the abstract method.
        results = self.generate_all(params, seed)
        return results[0]

    def generate_all(self, params: dict[str, Any], seed: int) -> list[GenerationResult]:
        text = (params.get("name") or "").strip()
        if not text:
            raise ValueError("name is required")
        meaning_db, _ = _load_meanings()
        name_obj = Name(text)
        # reduce=False keeps every alternative decomposition instead of
        # collapsing to the "best" one.
        name_obj.find_meaning(meaning_db, reduce=False)
        per_word = [name_obj.words[word] for word in text.split()]
        if not per_word:
            return [GenerationResult(result=text, explanation="no morphemes recognized")]

        # Dedupe by structural signature: when one usage has multiple Meanings
        # (different senses, e.g. -y as both "district" and "island"), the raw
        # cartesian product produces N copies of the same structural break with
        # only the Meaning identity differing. We collapse those into one
        # result and combine senses inside _build_explanation_part.
        seen: set[tuple] = set()
        candidates: list[tuple[int, int, tuple]] = []
        sig_to_words: dict[tuple, Any] = {}
        for words in itertools.product(*per_word):
            sig = _decomposition_signature(words)
            if sig in seen:
                continue
            seen.add(sig)
            sig_to_words[sig] = words
            unaccounted = sum(1 for w in words for c in w.word if isinstance(c, str) and c)
            total = sum(1 for w in words for c in w.word if not (isinstance(c, str) and not c))
            candidates.append((unaccounted, total, sig))

        # Best readings first: fewer unaccounted fragments, then simpler.
        candidates.sort()
        return [
            _build_decomposition_result(text, sig_to_words[sig], meaning_db)
            for _, _, sig in candidates[:_MAX_DECOMPOSITIONS]
        ]


register(KenningExplain())
