"""The `kenning-explain` Generator — decompose a name into morphemes."""

from __future__ import annotations

import itertools
from typing import Any

from wyrd.generators.kenning import (
    _LEGEND,
    _MAX_DECOMPOSITIONS,
    _build_decomposition_result,
    _canonical_signature_for_words,
    _decomposition_matcher,
    _decomposition_signature,
    _load_canonical_decompositions,
    _load_meanings,
    _required_name,
)
from wyrd.generators.kenning.runtime.name import Name
from wyrd.registry import GenerationResult, Generator


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
        text = _required_name(params)
        meaning_db, _ = _load_meanings()
        name_obj = Name(text)
        # reduce=False keeps every alternative decomposition instead of
        # collapsing to the "best" one. wyrd-04oi: reuse the pickled matcher
        # published beside the runtime DB (cold-start: skip the trie rebuild).
        name_obj.find_meaning(meaning_db, reduce=False, trie=_decomposition_matcher(meaning_db))
        per_word = [name_obj.words[word] for word in text.split()]
        if not per_word:
            return [GenerationResult(result=text, explanation="no morphemes recognized")]

        # wyrd-h8k1: when the bundle carries a canonical signature for
        # this toponym, the matching reading floats to the top of the
        # candidates list and gets marked ``canonical=True`` so the SPA
        # can render it distinctly. Bundle map is empty for legacy
        # list-shape bundles + dict-shape bundles missing the
        # ``canonical_decompositions`` field — this codepath is
        # transparent to legacy data.
        canonical_map = _load_canonical_decompositions()
        canonical_entry = canonical_map.get(text)
        canonical_signature = canonical_entry["signature"] if canonical_entry else None
        canonical_source = canonical_entry["source"] if canonical_entry else None

        # Dedupe by structural signature: when one usage has multiple Meanings
        # (different senses, e.g. -y as both "district" and "island"), the raw
        # cartesian product produces N copies of the same structural break with
        # only the Meaning identity differing. We collapse those into one
        # result and combine senses inside _build_explanation_part.
        seen: set[tuple] = set()
        candidates: list[tuple[int, int, int, tuple]] = []
        sig_to_words: dict[tuple, Any] = {}
        sig_is_canonical: dict[tuple, bool] = {}
        for words in itertools.product(*per_word):
            sig = _decomposition_signature(words)
            if sig in seen:
                continue
            seen.add(sig)
            sig_to_words[sig] = words
            unaccounted = sum(1 for w in words for c in w.word if isinstance(c, str) and c)
            total = sum(1 for w in words for c in w.word if not (isinstance(c, str) and not c))
            is_canonical = (
                canonical_signature is not None
                and _canonical_signature_for_words(words) == canonical_signature
            )
            sig_is_canonical[sig] = is_canonical
            # Canonical sorts first by carrying a leading-zero rank;
            # within-rank order falls back to the heuristic
            # (lowest unaccounted, then min-complexity).
            rank = 0 if is_canonical else 1
            candidates.append((rank, unaccounted, total, sig))

        # Best readings first: canonical (when present), then fewer
        # unaccounted fragments, then simpler.
        candidates.sort()
        return [
            _build_decomposition_result(
                text,
                sig_to_words[sig],
                meaning_db,
                canonical=sig_is_canonical[sig],
                canonical_source=canonical_source if sig_is_canonical[sig] else None,
            )
            for _, _, _, sig in candidates[:_MAX_DECOMPOSITIONS]
        ]
