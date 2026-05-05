"""wyrd-ami Phase 1: fantasy-name etymology research pipeline.

The pipeline takes a (name, description) pair — typically a gaming or
literary creature name plus a paragraph describing it — and routes it
through:

1. A descent-walking pre-filter against the wiktionary-derived etymon
   corpus. Walks etymon_descent.parent_id chains back to the deepest
   ancestor in the approved language families. Resolves ~50% of common
   inputs without an LLM call (harpy → ancient-greek ἅρπυια, troll →
   old-norse trǫll, etc.).

2. An LLM full-research fallback for inputs the pre-filter misses
   entirely or where the candidate ancestor's gloss doesn't match the
   input description (homograph collision: "drow" the D&D dark-elf vs
   ME `truwien` "to trust").

Every input produces one row in fantasy_morpheme, with `usable=1` when
we resolved to an attested etymon and `usable=0` otherwise (with a
`bar_reason` recording why). The bar_reason vocabulary leaves room for
wyrd-0ab's Phase 2 constructed-etymology rescue.

The `APPROACH_VERSION` constant is the user-requested pipeline-version
stamp: when this module's logic changes substantively (e.g. after
wyrd-gpif lands etymon_variant ingestion), bump the constant and
re-process earlier rows.
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Bump when the pipeline logic changes substantively (e.g. after
# wyrd-gpif lands alt-form ingestion, after we add Etymonline citation
# lookup, after we change the LLM prompt). Existing rows with an older
# version are stale and re-processable.
APPROACH_VERSION = "fantasy-v1"

# Language families approved as etymological sources for fantasy
# morphemes. A pre-filter ancestor in any of these counts as "real
# attested in our corpus." Anything else is bar-worthy as
# `outside_language_family`. Per user 2026-05-05: includes Greek
# (most monsters are Greek), Old Saxon, Frisian, Gothic, plus the
# core OE/ON/Celtic/Latin/OFr set we already use for toponym mining.
APPROVED_LANGUAGES: frozenset[str] = frozenset(
    {
        # Old Germanic
        "old-english",
        "old-saxon",
        "old-frisian",
        "old-norse",
        "icelandic",  # ON daughter; often the actual attestation locus
        "gothic",
        "proto-germanic",
        "proto-west-germanic",
        # Romance
        "latin",
        "old-french",
        # Hellenic
        "ancient-greek",
        # Celtic
        "celtic-mix",
        "irish",
        "welsh",
        "scottish-gaelic",
        "old-irish",
        "middle-irish",
        "breton",
        "cornish",
        "manx",
        # Middle English (Chaucer-era English; bridge between OE and modE)
        "middle-english",
    }
)


# Bar reasons enum (free-form text for forward compat, but stick to
# this set when writing rows so queries / dashboards stay clean).
BAR_REASON_NO_ETYMOLOGY = "no_etymology_found"
BAR_REASON_OUTSIDE_FAMILY = "outside_language_family"
BAR_REASON_UNCERTAIN = "uncertain_attestation"
BAR_REASON_MODERN_COINAGE = "modern_coinage"
BAR_REASON_PROPER_NOUN = "proper_noun_only"
BAR_REASON_HOMOGRAPH = "homograph_collision"


@dataclass(frozen=True)
class AncestorMatch:
    """One approved-family ancestor surfaced by the descent-walking lookup."""

    etymon_id: int
    canonical_form: str
    language: str
    edge_type: str  # 'inheritance', 'borrowing', 'derivation', '(direct)', etc.


@dataclass(frozen=True)
class Resolution:
    """Result of routing one (name, description) through the pipeline."""

    usable: bool
    etymon_id: int | None
    bar_reason: str | None
    resolution_method: str  # 'descent_lookup' | 'llm_full_research' | 'manual'
    confidence: str | None
    citation: str | None
    reasoning: str | None


def descent_walking_lookup(
    db_path: Path | str,
    name: str,
    *,
    approved: frozenset[str] = APPROVED_LANGUAGES,
) -> list[AncestorMatch]:
    """Find approved-family ancestors of `name` via the etymon_descent chain.

    1. Match input form (case-insensitive) against any etymon_canonical_form.
    2. For each match, BFS-walk parent_id ancestors until exhausted.
    3. Collect any ancestor whose language is in the approved set.

    Returns ancestors deduped by (canonical_form, language). Direct
    matches (where the input form is itself an approved-family etymon)
    take precedence and appear first with edge_type='(direct)'.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        seed_rows = conn.execute(
            "SELECT id, canonical_form, language FROM etymon "
            "WHERE LOWER(canonical_form) = LOWER(?)",
            (name,),
        ).fetchall()
        if not seed_rows:
            return []

        seen_etymon_ids: set[int] = {r["id"] for r in seed_rows}
        frontier: list[int] = list(seen_etymon_ids)
        approved_hits: list[AncestorMatch] = []

        # Direct hits in approved langs come first.
        for r in seed_rows:
            if r["language"] in approved:
                approved_hits.append(
                    AncestorMatch(
                        etymon_id=r["id"],
                        canonical_form=r["canonical_form"],
                        language=r["language"],
                        edge_type="(direct)",
                    )
                )

        # BFS through descent edges (frontier is the set of etymon_ids
        # whose parents we haven't yet visited). One DB round-trip per
        # depth; corpus depth rarely exceeds ~5.
        while frontier:
            next_frontier: list[int] = []
            placeholders = ",".join("?" * len(frontier))
            edges = conn.execute(
                f"""SELECT e.id, e.canonical_form, e.language, d.edge_type
                    FROM etymon_descent d
                    JOIN etymon e ON e.id = d.parent_id
                    WHERE d.child_id IN ({placeholders})""",
                frontier,
            ).fetchall()
            for e in edges:
                if e["id"] in seen_etymon_ids:
                    continue
                seen_etymon_ids.add(e["id"])
                next_frontier.append(e["id"])
                if e["language"] in approved:
                    approved_hits.append(
                        AncestorMatch(
                            etymon_id=e["id"],
                            canonical_form=e["canonical_form"],
                            language=e["language"],
                            edge_type=e["edge_type"],
                        )
                    )
            frontier = next_frontier

        return _dedupe_by_form_lang(approved_hits)
    finally:
        conn.close()


def _dedupe_by_form_lang(matches: list[AncestorMatch]) -> list[AncestorMatch]:
    seen: set[tuple[str, str]] = set()
    out: list[AncestorMatch] = []
    for m in matches:
        k = (m.canonical_form, m.language)
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    return out


def _ancestor_glosses(db_path: Path | str, etymon_id: int) -> list[str]:
    """Pull etymon_gloss rows for an etymon (best-effort semantic check helper)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (etymon_id,)
        ).fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        conn.close()


# ---------------------------------------------------------------------
# LLM full-research fallback (Gemini Flash via direct REST).
# ---------------------------------------------------------------------

_RESEARCH_PROMPT_TEMPLATE = """You are an etymological researcher working on a name-generation lexicon. Determine whether the following gaming/literary creature name has an attested historical form in any of these language families: {langs}.

Name: {name}
Description: {description}

Respond with ONE JSON object:
{{
  "attested_in": "<language code from the approved list>" OR "none" (no etymology found) OR "outside_family" (real etymology exists but lies outside the approved set, e.g. modern coinage, Sanskrit, Hebrew),
  "historical_form": "<the attested form in the source language>" OR null,
  "gloss": "<short meaning>" OR null,
  "citation": "<dictionary source you're confident about: 'Bosworth-Toller', 'Cleasby-Vigfusson', 'LSJ', 'Etymonline', etc.>" OR null,
  "confidence": "high" | "medium" | "low",
  "bar_reason": "<one of: no_etymology_found / outside_language_family / uncertain_attestation / modern_coinage / proper_noun_only>" OR null,
  "reasoning": "<1-2 sentences>"
}}

Be CONSERVATIVE. If you're not confident the name maps to a real word in the approved list, prefer setting attested_in="none" or "outside_family" with appropriate bar_reason. Modern fantasy authors (Tolkien, Dunsany, Gygax) coined many such names; do not retrofit etymology onto coinages. Proper-noun-only attestations (literary characters whose name became a common noun only later) should be barred with bar_reason=proper_noun_only.

Output ONLY the JSON, no preamble."""


def _llm_full_research(
    name: str,
    description: str,
    *,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    approved: frozenset[str] = APPROVED_LANGUAGES,
    timeout_s: float = 60.0,
) -> dict:
    """Call Gemini Flash with the etymology-research prompt; return parsed JSON."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    )
    prompt = _RESEARCH_PROMPT_TEMPLATE.format(
        name=name,
        description=description,
        langs=", ".join(sorted(approved)),
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        resp = json.loads(r.read())
    text = resp["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def _resolve_via_llm(
    db_path: Path | str,
    name: str,
    description: str,
    *,
    llm_caller=_llm_full_research,
) -> Resolution:
    """LLM full-research path. Maps the LLM's classification to a Resolution
    and (when usable) anchors to an existing etymon row if one exists for
    the (form, language) pair the LLM identified."""
    try:
        result = llm_caller(name, description)
    except (urllib.error.URLError, RuntimeError, json.JSONDecodeError, KeyError) as e:
        return Resolution(
            usable=False,
            etymon_id=None,
            bar_reason=BAR_REASON_UNCERTAIN,
            resolution_method="llm_full_research",
            confidence="low",
            citation=None,
            reasoning=f"LLM call failed: {e}",
        )

    attested_in = result.get("attested_in") or "none"
    confidence = result.get("confidence") or "low"
    bar_reason = result.get("bar_reason")
    historical = result.get("historical_form")
    citation = result.get("citation")
    reasoning = result.get("reasoning")
    gloss = result.get("gloss")

    # Conservative: low-confidence answers are barred even if attested_in
    # names a real language. Per user 2026-05-05.
    if confidence == "low" and attested_in not in ("none", "outside_family"):
        return Resolution(
            usable=False,
            etymon_id=None,
            bar_reason=BAR_REASON_UNCERTAIN,
            resolution_method="llm_full_research",
            confidence=confidence,
            citation=citation,
            reasoning=f"low confidence on {attested_in}/{historical}: {reasoning}",
        )

    if attested_in in APPROVED_LANGUAGES and historical:
        # Try to anchor to an existing etymon row.
        etymon_id = _lookup_etymon_id(db_path, historical, attested_in)
        return Resolution(
            usable=True,
            etymon_id=etymon_id,
            bar_reason=None,
            resolution_method="llm_full_research",
            confidence=confidence,
            citation=citation,
            reasoning=(
                f"{attested_in}/{historical}"
                + (f" '{gloss}'" if gloss else "")
                + (f"; {reasoning}" if reasoning else "")
            ),
        )

    # Either "none" or "outside_family" → barred. Use LLM-supplied
    # bar_reason, fall back to inference.
    if not bar_reason:
        bar_reason = (
            BAR_REASON_OUTSIDE_FAMILY
            if attested_in == "outside_family"
            else BAR_REASON_NO_ETYMOLOGY
        )
    return Resolution(
        usable=False,
        etymon_id=None,
        bar_reason=bar_reason,
        resolution_method="llm_full_research",
        confidence=confidence,
        citation=citation,
        reasoning=reasoning,
    )


def _lookup_etymon_id(db_path: Path | str, form: str, language: str) -> int | None:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = ? AND language = ?",
            (form, language),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Top-level resolve.
# ---------------------------------------------------------------------


def resolve(
    db_path: Path | str,
    name: str,
    description: str,
    *,
    skip_llm: bool = False,
    llm_caller=_llm_full_research,
) -> Resolution:
    """Resolve a (name, description) into a Resolution.

    Routing:
    - Pre-filter via descent_walking_lookup. If exactly one approved
      ancestor, return usable. If multiple ancestors and they all share
      a language family, prefer the deepest (closest to source).
    - If pre-filter misses entirely OR returns ambiguous candidates
      across multiple language families OR `skip_llm=False` and the
      caller wants a confidence boost, fall through to LLM research.
    - `skip_llm=True` short-circuits the LLM path entirely (used in
      tests and dry-runs).

    Homograph detection (e.g. "drow" matching ME `truwien` "to trust"
    while the description says "subterranean dark elf") is handled by
    the LLM path: when pre-filter has no clear winner, we just hand the
    description to the LLM and trust its disambiguation.
    """
    pre = descent_walking_lookup(db_path, name)
    if pre:
        # Pick the best pre-filter ancestor: prefer direct matches, then
        # the first-listed approved-family ancestor (BFS walks
        # parent-first so this is the closest ancestor in the chain).
        winner = pre[0]
        return Resolution(
            usable=True,
            etymon_id=winner.etymon_id,
            bar_reason=None,
            resolution_method="descent_lookup",
            confidence="high",
            citation="wiktionary-descent-chain",
            reasoning=f"matched {winner.language}/{winner.canonical_form} via {winner.edge_type}",
        )

    if skip_llm:
        return Resolution(
            usable=False,
            etymon_id=None,
            bar_reason=BAR_REASON_NO_ETYMOLOGY,
            resolution_method="descent_lookup",
            confidence="low",
            citation=None,
            reasoning="pre-filter found no approved-family ancestors; LLM skipped",
        )

    return _resolve_via_llm(db_path, name, description, llm_caller=llm_caller)


def write_resolution(
    db_conn: sqlite3.Connection,
    *,
    input_name: str,
    input_description: str | None,
    resolution: Resolution,
    approach_version: str = APPROACH_VERSION,
) -> int:
    """Insert one fantasy_morpheme row from a Resolution.

    Returns the inserted row id. Idempotent on (input_name,
    approach_version): re-running with the same name+version updates
    the existing row in place rather than failing the UNIQUE.
    """
    now = datetime.now(UTC).isoformat()
    cur = db_conn.execute(
        """INSERT INTO fantasy_morpheme (
            input_name, input_description, usable, etymon_id, bar_reason,
            resolution_method, approach_version, confidence, citation,
            reasoning, processed_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (input_name, approach_version) DO UPDATE SET
            input_description = excluded.input_description,
            usable            = excluded.usable,
            etymon_id         = excluded.etymon_id,
            bar_reason        = excluded.bar_reason,
            resolution_method = excluded.resolution_method,
            confidence        = excluded.confidence,
            citation          = excluded.citation,
            reasoning         = excluded.reasoning,
            processed_at      = excluded.processed_at
           RETURNING id""",
        (
            input_name,
            input_description,
            1 if resolution.usable else 0,
            resolution.etymon_id,
            resolution.bar_reason,
            resolution.resolution_method,
            approach_version,
            resolution.confidence,
            resolution.citation,
            resolution.reasoning,
            now,
        ),
    )
    return cur.fetchone()[0]


def tag_etymon_as_fantasy(db_conn: sqlite3.Connection, etymon_id: int) -> None:
    """Add the 'fantasy' tag to an etymon (idempotent on the etymon_tag PK)."""
    db_conn.execute(
        "INSERT OR IGNORE INTO etymon_tag (etymon_id, tag) VALUES (?, ?)",
        (etymon_id, "fantasy"),
    )
