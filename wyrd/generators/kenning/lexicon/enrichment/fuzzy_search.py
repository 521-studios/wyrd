"""Levenshtein-1 spelling-variant scan with gloss-anchored confirmation (D15).

Extends the exact-form ``reverse_search`` with edit-distance-1 matches
on body tokens. The Levenshtein implementation is the operation hot
path (~100M-cell scans across the live corpus), so the helper avoids
allocations where possible — the early-exit ``max_distance`` cap is
the load-bearing pruning.

D15 requires a gloss anchor: an edit-distance-1 match only counts as
evidence when a gloss string from ``etymon_gloss`` appears within
~100 characters of the matched form in the source body. Without the
anchor the edit-distance match is too permissive (``bere`` / ``bera``
— barley vs bear — are one edit apart and mean unrelated things).

The fuzzy scan and ``reverse_search`` are independent CLI steps; the
orchestrator runs reverse-search first by convention. Within fuzzy,
the pre-filter that avoids the expensive edit-distance scan on
already-attested etymons lives inside
``_select_rando_only_candidates_with_glosses`` — its NOT EXISTS
subquery excludes etymons that already carry an
``edit_distance = 0`` row in ``etymon_text_match`` (the canonical
output of ``reverse_search``).
"""

from __future__ import annotations

import re
from pathlib import Path

from wyrd.generators.kenning.lexicon.constants import normalize_ocr_form
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.enrichment.reverse_search import (
    _TEXT_MATCH_SNIPPET_RADIUS,
    _load_normalized_source_texts,
    _select_rando_only_candidates_with_glosses,
)

_FUZZY_TOKEN_RE = re.compile(r"[a-z]{3,}")


def _build_source_vocab(source_texts: dict[str, str]) -> dict[str, list[str]]:
    """Per-source unique alphabetic-token vocabulary (length 3+),
    sorted. Lets fuzzy_search compare each etymon against a bounded
    candidate set per source instead of scanning the body for every
    etymon."""
    return {
        source_id: sorted(set(_FUZZY_TOKEN_RE.findall(text)))
        for source_id, text in source_texts.items()
    }


def _all_canonical_forms_normalized(db: LexiconDB) -> set[str]:
    """Set of OCR-normalized + lowercased canonical_forms across the
    full (un-merged) etymon table. Used by fuzzy_search to suppress
    fuzzy-match claims where the body token is itself an independent
    canonical etymon (per wyrd-c3x — OCR variants between two etymons
    should be merged by normalize-ocr upstream, not connected through
    fuzzy-search's gloss-anchor heuristic)."""
    return {
        normalize_ocr_form(r["canonical_form"])
        for r in db.conn.execute("SELECT canonical_form FROM etymon WHERE merged_into_id IS NULL")
    }


def levenshtein(a: str, b: str, *, max_distance: int | None = None) -> int:
    """Compute Levenshtein edit distance between two strings.

    Pure-python DP implementation. If max_distance is given and the early-
    bound row minimum exceeds it, returns a value > max_distance without
    finishing — the caller can use it as a fast 'too far' rejection.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > (max_distance if max_distance is not None else max(la, lb)):
        return abs(la - lb)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        if max_distance is not None and min(curr) > max_distance:
            return max_distance + 1
        prev = curr
    return prev[lb]


def fuzzy_search_attestations(
    db: LexiconDB,
    sources_dir: Path | str,
    *,
    apply: bool = False,
    min_form_length: int = 5,
    max_distance: int = 1,
    gloss_window: int = 100,
) -> dict:
    """Fuzzy variant of reverse_search_attestations.

    For each rando-only etymon WITHOUT an exact text match, search the
    source-book texts for tokens within Levenshtein distance N of the
    canonical form. To guard against meaningless edit-distance collisions
    (e.g. rando 'bere' ≠ source 'bera' even though they're 1 edit apart),
    require that one of the etymon's glosses appears within ±gloss_window
    chars of the candidate's first occurrence.

    Conservative defaults:
      max_distance=1 — distance=2 produces too many false positives
      min_form_length=5 — short forms have too many fuzzy neighbors
      gloss_window=100 chars — scholarly entries usually gloss inline

    Writes to etymon_text_match with edit_distance > 0 so app/queries can
    distinguish fuzzy from exact matches. Returns a dict of stats and
    sample matches.
    """
    sources_path = Path(sources_dir)
    if not sources_path.is_dir():
        raise ValueError(f"sources_dir not found: {sources_path}")

    candidates = _select_rando_only_candidates_with_glosses(db, min_form_length=min_form_length)
    source_texts = _load_normalized_source_texts(sources_path)
    vocab_by_source = _build_source_vocab(source_texts)
    other_canonicals = _all_canonical_forms_normalized(db)

    matches: dict[int, list[tuple[str, str, int, int, str]]] = {}
    # value: (source_id, matched_form, distance, count, snippet)
    for etymon_id, form, glosses in candidates:
        norm_form = normalize_ocr_form(form)
        for source_id, vocab in vocab_by_source.items():
            text = source_texts[source_id]
            # Fast filter: only consider tokens within ±2 length and starting
            # with the same character (cheap heuristic, good for OE-style
            # short morphemes that vary in their tail).
            for tok in vocab:
                if abs(len(tok) - len(norm_form)) > max_distance:
                    continue
                if tok == norm_form:
                    continue  # exact match — handled by reverse_search
                # If `tok` is itself a canonical etymon, it's not a fuzzy
                # variant — it's its own thing. See wyrd-c3x. Cheap O(1)
                # lookup gates the expensive Levenshtein call.
                if tok in other_canonicals:
                    continue
                d = levenshtein(norm_form, tok, max_distance=max_distance)
                if d > max_distance or d == 0:
                    continue
                # Found a fuzzy candidate. Now verify meaning: does any gloss
                # appear within ±gloss_window chars of this token's first
                # occurrence?
                pattern = re.compile(r"\b" + re.escape(tok) + r"\b")
                m = pattern.search(text)
                if not m:
                    continue
                start = max(0, m.start() - gloss_window)
                end = min(len(text), m.end() + gloss_window)
                window_text = text[start:end]
                if not any(g in window_text for g in glosses):
                    continue  # meaning didn't anchor — skip
                # Record. snippet shows the matched form with marker.
                snip_start = max(0, m.start() - _TEXT_MATCH_SNIPPET_RADIUS)
                snip_end = min(len(text), m.end() + _TEXT_MATCH_SNIPPET_RADIUS)
                snippet = text[snip_start:snip_end].strip().replace(tok, f"«{tok}»", 1)
                count = len(pattern.findall(text))
                matches.setdefault(etymon_id, []).append((source_id, tok, d, count, snippet))
                break  # one fuzzy match per source per etymon is enough

    forms_by_id = {eid: f for eid, f, _ in candidates}
    written = 0
    if apply:
        for etymon_id, hits in matches.items():
            for source_id, matched_form, distance, count, snippet in hits:
                db.conn.execute(
                    """
                    INSERT INTO etymon_text_match
                        (etymon_id, source_id, matched_form, match_count,
                         edit_distance, snippet, method)
                    VALUES (?, ?, ?, ?, ?, ?, 'fuzzy-search-v1')
                    ON CONFLICT(etymon_id, source_id, matched_form)
                    DO UPDATE SET
                        match_count = excluded.match_count,
                        edit_distance = excluded.edit_distance,
                        snippet = excluded.snippet,
                        method = excluded.method
                    """,
                    (etymon_id, source_id, matched_form, count, distance, snippet),
                )
                written += 1
        db.commit()

    return {
        "candidates_with_gloss": len(candidates),
        "etymons_with_fuzzy_match": len(matches),
        "total_match_records": sum(len(v) for v in matches.values()),
        "written": written,
        "sample": [
            {
                "etymon_id": eid,
                "form": forms_by_id.get(eid),
                "matches": [(s, m, d, c) for s, m, d, c, _ in hits[:5]],
            }
            for eid, hits in list(matches.items())[:25]
        ],
    }
