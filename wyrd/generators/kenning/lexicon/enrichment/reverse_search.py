"""Body-text reverse search for canonical-form attestations (D12).

Scans every source body text in ``sources_dir`` for word-boundary
matches of every etymon's canonical form (and known variants), then
writes the matches to ``etymon_text_match``. This is the loose
evidence layer ``etymon_consensus`` does NOT count, but downstream
analysis (period-form projection, attestation-year scan) reads.

Two related entry points share the body-loading + normalization
infrastructure:

* ``reverse_search_attestations`` — the canonical-form scan, with
  the score/select pipeline that emits ``method='reverse-search-v1'``
  rows.
* ``annotate_fragments_with_corpus_evidence`` — used by ``wyrd kenning
  unaccounted`` to surface body-text occurrences of unaccounted
  fragments + a heuristic flag for whether the snippet looks like an
  etymology body. Same source-text load path; output is annotation,
  not lexicon writes.

The Rando-port candidate-selection helpers
(``_select_rando_only_candidates`` + the ``_with_glosses`` variant)
gate the scan to etymons that lack scholarly attestation, so the
scan focuses on adding new evidence rather than re-confirming what
mining already wrote.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.lexicon.constants import normalize_ocr_form
from wyrd.generators.kenning.lexicon.db import LexiconDB

_TEXT_MATCH_SNIPPET_RADIUS = 100


def _find_reverse_matches(
    candidates: list[tuple[int, str, str]],
    source_texts: dict[str, str],
    *,
    min_form_length: int,
) -> dict[int, list[tuple[str, int, str]]]:
    """For each rando-only candidate, word-boundary-search every source body
    for its normalized form. Returns ``{etymon_id: [(source_id, count,
    snippet), ...]}`` — snippet is ±_TEXT_MATCH_SNIPPET_RADIUS chars around the
    FIRST match in that source (with the matched form marked), enough for the
    SPA citation panel without storing every occurrence."""
    matches: dict[int, list[tuple[str, int, str]]] = {}
    for etymon_id, form, _language in candidates:
        norm = normalize_ocr_form(form)
        if len(norm) < min_form_length:
            continue
        pattern = re.compile(r"\b" + re.escape(norm) + r"\b")
        for source_id, text in source_texts.items():
            hits = list(pattern.finditer(text))
            if not hits:
                continue
            first = hits[0]
            start = max(0, first.start() - _TEXT_MATCH_SNIPPET_RADIUS)
            end = min(len(text), first.end() + _TEXT_MATCH_SNIPPET_RADIUS)
            snippet = text[start:end].strip()
            # Mark the matched form within the snippet for app display.
            snippet = snippet.replace(norm, f"«{norm}»", 1)
            matches.setdefault(etymon_id, []).append((source_id, len(hits), snippet))
    return matches


def _write_reverse_matches(
    db: LexiconDB,
    matches: dict[int, list[tuple[str, int, str]]],
    forms_by_id: dict[int, str],
) -> int:
    """Persist reverse-search matches to the etymon_text_match table (kept
    separate from etymon_citation so the consensus view stays an
    extraction-witness count only; search-evidence is loose-confidence). The
    ``source_id`` points at the REAL source row — its being search-evidence is
    encoded by which table the row lives in. Some books in sources/ aren't
    LLM-mined yet, so a stub source row is upserted first for the FK. Returns
    rows written; caller commits + gates on apply."""
    all_source_ids = set()
    for hits in matches.values():
        for sid, _count, _snip in hits:
            all_source_ids.add(sid)
    for sid in all_source_ids:
        db.upsert_source(
            id=sid,
            title=sid.replace("_", " ").title(),
            notes=(
                "Source row created by reverse-search; the book exists "
                "in sources/ but may not have been LLM-mined yet. Full "
                "metadata fills in when the book is formally mined."
            ),
        )

    written = 0
    for etymon_id, hits in matches.items():
        form = forms_by_id.get(etymon_id, "")
        matched_form = normalize_ocr_form(form)
        for source_id, count, snippet in hits:
            db.conn.execute(
                """
                INSERT INTO etymon_text_match
                    (etymon_id, source_id, matched_form, match_count, edit_distance, snippet)
                VALUES (?, ?, ?, ?, 0, ?)
                ON CONFLICT(etymon_id, source_id, matched_form)
                DO UPDATE SET
                    match_count = excluded.match_count,
                    snippet = excluded.snippet
                """,
                (etymon_id, source_id, matched_form, count, snippet),
            )
            written += 1
    return written


def reverse_search_attestations(
    db: LexiconDB,
    sources_dir: Path | str,
    *,
    apply: bool = False,
    min_form_length: int = 4,
) -> dict:
    """Reverse-direction verification: for each rando-port etymon with no
    scholarly citation, search the bundled source-book texts for the form
    as a word-boundary substring. Mentions of the form (in any book) count
    as a 'search-attested' citation — looser evidence than extraction but
    still meaningful confirmation that the form appears in published
    philology.

    The function is conservative:
    - Only operates on etymons with citations from rando-port AND no other
      source. Etymons already cross-corroborated are skipped.
    - Requires the form to be at least `min_form_length` characters
      (default 4). Shorter forms produce too many false-positive substring
      matches in normal English text.
    - Word-boundary regex match (\\b<form>\\b on a normalized version of
      the body) — won't match "ham" inside "hammer" or "hamlet."

    With apply=False (default) reports candidates without writing.
    With apply=True inserts an etymon_citation row for each match,
    pointing at a synthetic source `search-attested:<book>` so we can
    distinguish search-evidence from extraction-evidence.

    Returns a dict of {etymon_id: [(source_id, count), ...]} for matches.
    """
    sources_path = Path(sources_dir)
    if not sources_path.is_dir():
        raise ValueError(f"sources_dir not found: {sources_path}")

    candidates = _select_rando_only_candidates(db, min_form_length=min_form_length)
    source_texts = _load_normalized_source_texts(sources_path)

    # Build a quick lookup so the sample report can name the etymons.
    forms_by_id = {eid: form for eid, form, _ in candidates}

    matches = _find_reverse_matches(candidates, source_texts, min_form_length=min_form_length)
    written = _write_reverse_matches(db, matches, forms_by_id) if apply else 0
    if apply:
        db.commit()

    parser_bug_suspects = _score_extraction_gaps(db, matches, forms_by_id)

    return {
        "rando_only_candidates": len(candidates),
        "etymons_with_match": len(matches),
        "total_match_records": sum(len(v) for v in matches.values()),
        "written": written,
        "parser_bug_suspects": parser_bug_suspects,
        "sample": [
            {
                "etymon_id": eid,
                "form": forms_by_id.get(eid),
                "matches": hits[:5],
            }
            for eid, hits in list(matches.items())[:25]
        ],
    }


def _select_rando_only_candidates(
    db: LexiconDB, *, min_form_length: int
) -> list[tuple[int, str, str]]:
    """Find rando-only etymons (cited by rando-port and ONLY rando-port).

    Excludes modern-english etymons by default — those are real English
    words like 'with', 'north', 'great', 'long', 'bishop' that appear
    thousands of times in normal prose and produce vast amounts of
    noise. They're also not 'unverified' in any meaningful linguistic
    sense; they're modern vocabulary used as place-name modifiers.

    Filters by ``min_form_length`` (forms shorter than that produce too
    many false-positive substring matches).
    """
    cur = db.conn.execute(
        """
        SELECT e.id, e.canonical_form, e.language
        FROM etymon e
        WHERE e.language != 'modern-english'
          AND EXISTS (
            SELECT 1 FROM etymon_citation c
            WHERE c.etymon_id = e.id AND c.source_id = 'rando-port'
          )
          AND NOT EXISTS (
            SELECT 1 FROM etymon_citation c
            WHERE c.etymon_id = e.id AND c.source_id != 'rando-port'
          )
        """
    )
    return [
        (row["id"], row["canonical_form"], row["language"])
        for row in cur.fetchall()
        if len(row["canonical_form"]) >= min_form_length
    ]


def annotate_fragments_with_corpus_evidence(
    fragments: list[str],
    sources_path: Path | str,
    *,
    snippets_per_fragment: int = 2,
    snippet_window: int = 60,
) -> dict[str, dict[str, Any]]:
    """wyrd-bvp: for each fragment in ``fragments``, scan all
    ``sources/*.txt`` files for word-boundary matches and return
    per-fragment evidence:

    * ``corpus_hits`` — count of distinct source files where the
      fragment appears at a word boundary.
    * ``snippets`` — up to ``snippets_per_fragment`` ``{source,
      snippet, in_etym_body}`` entries; ``snippet`` carries
      ``snippet_window`` chars of surrounding context.
    * ``in_etym_body`` — heuristic bool: True when the snippet sits
      inside what looks like an etymology body (preceding text
      contains a year-citation OR a source marker like ``A.S.`` /
      ``O.E.`` / ``M.E.`` / ``cf.`` / ``from``). Lets gap-triage
      prioritise evidence that's actually authoritative rather
      than running prose.
    * ``strong_hits`` — count of in_etym_body snippets in the
      sample. Surfaces fragments with multiple etymology-body
      witnesses for promotion to actual etymon authoring.

    Reuses ``_load_normalized_source_texts`` for the lowercased +
    OCR-normalized source corpus. Lossy on case (per the existing
    convention) since lemma-vs-source case-mismatches are common.

    Returns ``{fragment: evidence_dict}``. Fragments with zero hits
    still get an entry with ``corpus_hits=0`` and empty snippets,
    so callers can render a 'no scholarly evidence' marker
    uniformly.
    """
    sources_path = Path(sources_path)
    if not sources_path.is_dir():
        raise ValueError(f"sources_dir not found: {sources_path}")
    source_texts = _load_normalized_source_texts(sources_path)
    out: dict[str, dict[str, Any]] = {}
    for fragment in fragments:
        frag_lower = fragment.lower()
        if not frag_lower:
            continue
        frag_re = re.compile(r"\b" + re.escape(frag_lower) + r"\b")
        hits = 0
        snippets: list[dict[str, Any]] = []
        # Stable ordering for deterministic output across runs.
        for source_id in sorted(source_texts):
            text = source_texts[source_id]
            m = frag_re.search(text)
            if m is None:
                continue
            hits += 1
            if len(snippets) < snippets_per_fragment:
                lo = max(0, m.start() - snippet_window)
                hi = min(len(text), m.end() + snippet_window)
                snippet = text[lo:hi].strip().replace("\n", " ")
                # Heuristic: an etymology body typically has a
                # year-citation (3-4 digit number, 700-1700 era
                # range) or a source-marker like 'a.s.' / 'o.e.'
                # / 'm.e.' / 'cf.' / 'from' in the immediately-
                # preceding text. Probe the LEFT half of the
                # snippet so we don't flag prose where the year
                # is FOLLOWING the fragment (citation prose
                # typically puts year-marker before the form).
                left = text[lo : m.start()]
                in_etym_body = bool(
                    re.search(r"\b(7\d{2}|[89]\d{2}|1[0-6]\d{2})\b", left)
                    or re.search(r"\b(a\.s\.|o\.e\.|m\.e\.|cf\.|from)\b", left)
                )
                snippets.append(
                    {
                        "source": source_id,
                        "snippet": snippet,
                        "in_etym_body": in_etym_body,
                    }
                )
        out[fragment] = {
            "corpus_hits": hits,
            "strong_hits": sum(1 for s in snippets if s["in_etym_body"]),
            "snippets": snippets,
        }
    return out


def _load_normalized_source_texts(sources_path: Path) -> dict[str, str]:
    """Load every ``*.txt`` under ``sources_path``, lowercased + OCR-
    normalized, keyed by file stem.

    Lowercase + ASCII-fold (via ``normalize_ocr_form``) because rando
    lemmas are often stored ASCII while sources have macrons / æ / ð
    — we want the matcher to see the same surface from both sides.
    """
    source_texts: dict[str, str] = {}
    for f in sources_path.glob("*.txt"):
        text = f.read_text(errors="replace", encoding="utf-8").lower()
        source_texts[f.stem] = normalize_ocr_form(text)
    return source_texts


def _score_extraction_gaps(
    db: LexiconDB,
    matches: dict[int, list[tuple[str, int, str]]],
    forms_by_id: dict[int, str],
) -> list[dict[str, Any]]:
    """PARSER-BUG DIAGNOSTIC: for each etymon found in source text,
    count how often it actually appeared in extracted etymology rows.
    A large gap (lots of text appearances, few extractions) is
    evidence the LLM extractor is systematically missing that
    morpheme — a candidate for prompt tuning.

    Returns the top-30 suspects sorted by descending text-count.
    """
    extraction_counts: dict[int, int] = {}
    if matches:
        # One GROUP BY pass per chunk instead of N separate COUNT(*)
        # queries. Chunked at 999 to stay under SQLite's
        # SQLITE_MAX_VARIABLE_NUMBER default. Etymons with no row in
        # toponym_etymology_element don't appear in the result; the
        # zero is implicit (the call site uses ``.get(etymon_id, 0)``).
        ids = list(matches.keys())
        for i in range(0, len(ids), 999):
            chunk = ids[i : i + 999]
            placeholders = ",".join("?" * len(chunk))
            cur = db.conn.execute(
                f"SELECT etymon_id, COUNT(*) AS n FROM toponym_etymology_element "
                f"WHERE etymon_id IN ({placeholders}) GROUP BY etymon_id",
                chunk,
            )
            for row in cur:
                extraction_counts[row["etymon_id"]] = row["n"]
    suspects: list[dict[str, Any]] = []
    for etymon_id, hits in matches.items():
        text_count = sum(c for _, c, _ in hits)
        ext_count = extraction_counts.get(etymon_id, 0)
        if text_count >= 10 and ext_count == 0:
            suspects.append(
                {
                    "etymon_id": etymon_id,
                    "form": forms_by_id.get(etymon_id),
                    "text_count": text_count,
                    "extraction_count": ext_count,
                    "books": [s for s, _, _ in hits],
                }
            )
    suspects.sort(key=lambda x: -x["text_count"])
    return suspects[:30]


def _select_rando_only_candidates_with_glosses(
    db: LexiconDB, *, min_form_length: int
) -> list[tuple[int, str, list[str]]]:
    """Variant of ``_select_rando_only_candidates`` that joins the
    etymon's glosses (needed for the fuzzy-match gloss-anchor check).
    Skips etymons with no gloss — without one there's no way to anchor
    meaning, and per D15 the fuzzy match is gloss-window-gated.

    Also additionally filters out etymons that already have an exact
    text-match row (those have been resolved by reverse-search and
    don't need fuzzy follow-up).
    """
    cur = db.conn.execute(
        """
        SELECT e.id, e.canonical_form, e.language,
               GROUP_CONCAT(g.gloss, '|') AS glosses
        FROM etymon e
        LEFT JOIN etymon_gloss g ON g.etymon_id = e.id
        WHERE e.language != 'modern-english'
          AND EXISTS (
            SELECT 1 FROM etymon_citation c
            WHERE c.etymon_id = e.id AND c.source_id = 'rando-port'
          )
          AND NOT EXISTS (
            SELECT 1 FROM etymon_citation c
            WHERE c.etymon_id = e.id AND c.source_id != 'rando-port'
          )
          AND NOT EXISTS (
            SELECT 1 FROM etymon_text_match m
            WHERE m.etymon_id = e.id AND m.edit_distance = 0
          )
        GROUP BY e.id
        """
    )
    out: list[tuple[int, str, list[str]]] = []
    for row in cur.fetchall():
        if not row["glosses"]:
            continue
        form = row["canonical_form"]
        if len(form) < min_form_length:
            continue
        glosses = [g.lower() for g in row["glosses"].split("|") if g]
        out.append((row["id"], form, glosses))
    return out
