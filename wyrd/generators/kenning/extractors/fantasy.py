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
import logging
import os
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Gemini REST endpoint shared by both _llm_full_research and
# _llm_semantic_check. Format with .format(model=...) at call time.
_GEMINI_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Bump when the pipeline logic changes substantively (e.g. after
# wyrd-gpif lands alt-form ingestion, after we add Etymonline citation
# lookup, after we change the LLM prompt). Existing rows with an older
# version are stale and re-processable.
APPROACH_VERSION = "fantasy-v1"

# The CANONICAL register: the user-facing approved-language set we
# describe to the LLM in `_RESEARCH_PROMPT_TEMPLATE`. These are the
# languages a reader would call out as the kenning fantasy register.
# Wave-1 IE + Celtic + wave-2 world-religion set. Per user 2026-05-05:
# includes Greek (most monsters are Greek), Old Saxon, Frisian, Gothic,
# plus the core OE/ON/Celtic/Latin/OFr set we already use for toponym
# mining. Codes use the etymon-table conventions (a mix of descriptive
# names like "old-english" and ISO short codes like "got"/"osx"/"sco"
# that wiktextract preserves as-is). Verified against corpus: every
# entry below has >0 rows in the etymon table at the time of writing.
CANONICAL_LANGUAGES: frozenset[str] = frozenset(
    {
        # Old Germanic
        "old-english",
        "osx",  # Old Saxon (wiktextract uses ISO 639-3 code)
        "ofs",  # Old Frisian (ISO code)
        "old-norse",
        "icelandic",  # ON daughter; often the actual attestation locus
        "got",  # Gothic (ISO code)
        "proto-germanic",
        "gmw-pro",  # Proto-West-Germanic (Wiktionary code)
        # German tree (kobold/wyrm-cousin morphemes; user-approved 2026-05-05)
        "old-high-german",
        "middle-high-german",
        "german",
        # Scots + Norn (drow/trow chain → ON troll via Orkney/Shetland Norn).
        # `sco` = Scots, the Germanic lowland-Scotland language.
        # `nrn` = Norn, the extinct Norse-derived language of Orkney/Shetland.
        # User-approved 2026-05-05 (note: distinct from Scots GAELIC = the
        # Celtic `scottish-gaelic`, which is also approved separately below).
        "sco",
        "nrn",
        # Romance
        "latin",
        "old-french",
        # Hellenic
        "ancient-greek",
        # Celtic (specific languages; the seed-only "celtic-mix" pseudo-code
        # from meanings.json doesn't appear in the etymon table)
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
        # World-religion source material — wave 2, user-approved
        # 2026-05-06. Each foundational to creature etymology in
        # canonical fantasy: Hebrew (golem, behemoth, leviathan,
        # lilith, cherub, seraph), Arabic (djinn/ifrit/marid/shaitan/
        # ghoul), Persian (pari, div, simurgh — many Arabic creature
        # names entered English via Persian translations of folktales),
        # Sanskrit (rakshasa, naga, garuda, deva, yaksha — D&D
        # canonical), Akkadian (tiamat, lamassu — Mesopotamian
        # cosmology), Egyptian (sphinx, mummy origins). Plus useful
        # intermediates: Aramaic (biblical bridge between Hebrew and
        # Arabic), Middle Persian / Pahlavi (Persian descent chain).
        # Codes are wiktextract's ISO short codes; descriptive names
        # like 'arabic'/'sanskrit'/etc. map to these via
        # _LANGUAGE_ALIAS_MAP.
        "he",
        "ar",
        "fa",
        "sa",
        "akk",
        "egy",
        "arc",
        "pal",
    }
)


# Wave-2 precursor / postcursor codes — user-approved 2026-05-06
# ("you may add the precursors or postcursors of any language we've
# approved so we can do the full history"). These are pipeline-internal
# additions that let `descent_walking_lookup` walk the FULL chain from
# a modern reflex back to its deepest ancestor without bailing at the
# language gate, but they are NOT exposed in the LLM prompt: the model
# has weak training signal on obscure codes like `iir-pro`/`afa-pro`
# and would just get confused. The descriptive aliases in
# `_LANGUAGE_ALIAS_MAP` ("Old Persian", "Proto-Semitic", "Coptic",
# "Pali", ...) absorb anything the LLM might say in plain prose.
#
# Per-code etymon-row counts at the time of approval are listed in
# INGESTION.md ("Approved languages" section under "Mining the
# wyrd-ami fantasy-name corpus") and in PR #94's body — not in this
# comment block, since those counts grow whenever wiktextract is
# re-ingested.
_PRECURSOR_POSTCURSOR_STACK: frozenset[str] = frozenset(
    {
        # Semitic precursors:
        "hbo",  # Biblical Hebrew (precursor to he)
        "sem-pro",  # Proto-Semitic (root of he/ar/akk/arc/hbo)
        "sem-wes-pro",  # Proto-West-Semitic (intermediate Semitic node)
        "afa-pro",  # Proto-Afroasiatic (deep root of egy + Semitic)
        "sux",  # Sumerian (Mesopotamian substrate of akk loanwords)
        # Aramaic descendant:
        "syc",  # Classical Syriac (late-ancient descendant of arc)
        # Iranian precursors / postcursors:
        "peo",  # Old Persian (precursor of fa)
        "fa-cls",  # Classical Persian (intermediate fa stage)
        "xpr",  # Parthian (Iranian, sister of pal)
        "ira-pro",  # Proto-Iranian (root of fa/pal/peo/xpr)
        # Indo-Aryan precursors / postcursors:
        "iir-pro",  # Proto-Indo-Iranian (root of fa+sa branches)
        "inc-pro",  # Proto-Indo-Aryan (root of sa)
        "pra",  # Prakrit (descendant of sa)
        "pi",  # Pali (Buddhist canonical Indo-Aryan, descendant of sa)
        # Egyptian descendant:
        "cop",  # Coptic (late-ancient descendant of egy)
        # Armenian (ancient register; appears in Iranian / Greek descent paths):
        "axm",  # Old / Classical Armenian
    }
)


# The full set the descent walker uses as its language gate. A
# pre-filter ancestor in any of these counts as "real attested in
# our corpus"; anything else is bar-worthy as `outside_language_family`.
# Includes both the canonical register AND the precursor/postcursor
# stack so BFS-walks of full chains don't dead-end at intermediate
# nodes. The LLM prompt deliberately uses CANONICAL_LANGUAGES (smaller
# set) — proto-codes and ancient intermediates are pipeline machinery,
# not a register the LLM should classify against.
APPROVED_LANGUAGES: frozenset[str] = CANONICAL_LANGUAGES | _PRECURSOR_POSTCURSOR_STACK


# Map descriptive language names from LLM output to the ISO codes our
# etymon table uses. Applied in _classify_llm_result after lowercase +
# space-to-dash normalization. The descent_walking_lookup pre-filter
# doesn't need this — it queries the etymon table by ISO code directly
# — but the LLM commonly returns descriptive names. Without this map,
# 'sanskrit' from the LLM mis-classifies as outside_language_family
# even though `sa` is approved.
_LANGUAGE_ALIAS_MAP: dict[str, str] = {
    "sanskrit": "sa",
    "arabic": "ar",
    "classical-arabic": "ar",
    "hebrew": "he",
    # 'biblical-hebrew' keeps mapping to 'he' because the Hebrew
    # wiktextract slice files biblical attestations under the modern
    # he lemma (15k rows) rather than under hbo (229 rows). The
    # standalone hbo code is approved separately so descent walks
    # from he back through hbo don't dead-end at the language gate.
    "biblical-hebrew": "he",
    "persian": "fa",
    # 'classical-persian' keeps mapping to 'fa' for the same reason
    # 'biblical-hebrew' maps to 'he': modern Persian wiktextract
    # carries the literary-classical lemmas under fa (~20k rows)
    # rather than under fa-cls (~211 rows). fa-cls is approved
    # separately so descent walks through it succeed.
    "classical-persian": "fa",
    "farsi": "fa",
    "akkadian": "akk",
    "egyptian": "egy",
    "ancient-egyptian": "egy",
    "aramaic": "arc",
    "middle-persian": "pal",
    # 'pahlavi' is a synonym for Middle Persian; both alias to 'pal'.
    # The Iranian-stack precursors (xpr Parthian, ira-pro Proto-Iranian)
    # are approved separately for descent-chain coverage.
    "pahlavi": "pal",
    # Wave-2 precursor / postcursor descriptive names → ISO codes
    # (user-approved 2026-05-06). Mirrors the corresponding
    # APPROVED_LANGUAGES additions; without these aliases an LLM
    # response of "Old Persian" / "Coptic" / "Pali" would mis-classify
    # as outside_language_family even though the etymon table now
    # carries those rows under the ISO codes.
    "old-persian": "peo",
    "parthian": "xpr",
    "sumerian": "sux",
    "syriac": "syc",
    "classical-syriac": "syc",
    "coptic": "cop",
    "prakrit": "pra",
    "pali": "pi",
    "old-armenian": "axm",
    "classical-armenian": "axm",
    "proto-semitic": "sem-pro",
    "proto-west-semitic": "sem-wes-pro",
    "proto-afroasiatic": "afa-pro",
    "proto-iranian": "ira-pro",
    "proto-indo-iranian": "iir-pro",
    "proto-indo-aryan": "inc-pro",
}


# Bar reasons enum (free-form text for forward compat, but stick to
# this set when writing rows so queries / dashboards stay clean).
BAR_REASON_NO_ETYMOLOGY = "no_etymology_found"
BAR_REASON_OUTSIDE_FAMILY = "outside_language_family"
BAR_REASON_UNCERTAIN = "uncertain_attestation"
BAR_REASON_MODERN_COINAGE = "modern_coinage"
BAR_REASON_PROPER_NOUN = "proper_noun_only"
BAR_REASON_HOMOGRAPH = "homograph_collision"
# LLM identified an attested historical form in an approved language,
# but our etymon table doesn't yet have a row for that (form, language)
# pair. The morpheme is real; it just isn't in our corpus, so it can't
# be used for generation until we ingest it. The fantasy_morpheme row
# records the LLM finding so a future backfill pass can add the etymon.
BAR_REASON_NOT_IN_CORPUS = "attested_but_not_in_corpus"

# Sentinel values the LLM uses for "no real attestation" in the
# attested_in field (distinct from naming a real-but-unapproved
# language). Module-level frozenset for O(1) lookups in resolve().
_NO_REAL_ATTESTATION_SENTINELS: frozenset[str] = frozenset(
    {"none", "modern_coinage", "outside_family"}
)


def _connect_ro(db_path: Path | str) -> sqlite3.Connection:
    """Open a read-only SQLite connection on `db_path`.

    Uses Path.as_uri() so paths with spaces, non-ASCII chars, or other
    URI-unsafe characters are encoded correctly. Falls back to a string
    path verbatim if the input is already URI-shaped (rare in practice).
    """
    p = Path(db_path).resolve()
    return sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True)


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
    # When bar_reason='outside_language_family', the LLM may know
    # WHICH unapproved language the etymology is in; record it so we
    # can aggregate a "languages to consider approving" report later.
    unapproved_language: str | None = None
    unapproved_form: str | None = None


def _seed_rows_for_name(
    conn: sqlite3.Connection, name: str
) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    """Return (direct_lemma_rows, variant_seeded_rows) — the two
    BFS-seed sources used by descent_walking_lookup, deduped against
    each other so the same etymon never appears in both lists.

    (a) direct: etymon.canonical_form matches the input (case-insens).
    (b) variant: etymon_variant.form matches (COLLATE NOCASE on column),
        and the variant's parent etymon isn't already in (a).
    """
    direct = conn.execute(
        "SELECT id, canonical_form, language FROM etymon "
        "WHERE LOWER(canonical_form) = LOWER(?) "
        "ORDER BY id",
        (name,),
    ).fetchall()
    direct_ids = {r["id"] for r in direct}
    if direct_ids:
        placeholders = ",".join("?" * len(direct_ids))
        variant = conn.execute(
            f"""SELECT DISTINCT e.id, e.canonical_form, e.language
                FROM etymon_variant v
                JOIN etymon e ON e.id = v.etymon_id
                WHERE v.form = ?
                  AND e.id NOT IN ({placeholders})
                ORDER BY e.id""",
            (name, *direct_ids),
        ).fetchall()
    else:
        variant = conn.execute(
            """SELECT DISTINCT e.id, e.canonical_form, e.language
               FROM etymon_variant v
               JOIN etymon e ON e.id = v.etymon_id
               WHERE v.form = ?
               ORDER BY e.id""",
            (name,),
        ).fetchall()
    return direct, variant


def _seed_direct_hits(
    direct: list[sqlite3.Row],
    variant: list[sqlite3.Row],
    approved: frozenset[str],
) -> list[AncestorMatch]:
    """Collect the seed rows whose language is in `approved`, tagged
    with the appropriate edge_type. Direct lemma matches come first
    as edge_type='(direct)'; variant-mediated matches follow with
    edge_type='(direct-via-variant)' so callers can tell a match was
    through an alt-form rather than the lemma."""
    hits: list[AncestorMatch] = []
    for rows, edge in ((direct, "(direct)"), (variant, "(direct-via-variant)")):
        for r in rows:
            if r["language"] in approved:
                hits.append(
                    AncestorMatch(
                        etymon_id=r["id"],
                        canonical_form=r["canonical_form"],
                        language=r["language"],
                        edge_type=edge,
                    )
                )
    return hits


# Max bind parameters per descent-edge IN-clause. SQLite's
# SQLITE_MAX_VARIABLE_NUMBER is 999 on older builds (raised to 32766 in 3.32),
# so a BFS frontier wider than that overflows an unchunked IN-list with
# "too many SQL variables". 900 leaves margin under the conservative 999 and
# matches the etymon-id chunking elsewhere (canonicalization_projection,
# reverse_search, toponym_etymology_canonical).
_BFS_PARAM_CHUNK = 900


def descent_walking_lookup(
    db_path: Path | str,
    name: str,
    *,
    approved: frozenset[str] = APPROVED_LANGUAGES,
) -> list[AncestorMatch]:
    """Find approved-family ancestors of `name` via the etymon_descent chain.

    1. Seed the BFS by matching the input form against:
       (a) etymon.canonical_form (case-insensitive) — the lemma row
       (b) etymon_variant.form (COLLATE NOCASE on the column) — alt
           spellings, inflected forms, romanizations from wiktextract's
           `forms` arrays. The variant row's parent etymon becomes the
           seed; resolves Chaucer-era `harpie` → modern-english `harpy`
           (via variant) → ancient-greek ἅρπυια (via descent edge).
    2. For each seed etymon, BFS-walk parent_id ancestors until exhausted.
    3. Collect any ancestor whose language is in the approved set.

    Direct seed-language hits in approved langs come first; variant
    seeds carry edge_type='(direct-via-variant)' so callers can tell
    that the match was through an alt-form rather than the lemma.
    """
    conn = _connect_ro(db_path)
    conn.row_factory = sqlite3.Row
    try:
        direct, variant = _seed_rows_for_name(conn, name)
        if not direct and not variant:
            return []

        all_seed_ids = {r["id"] for r in direct} | {r["id"] for r in variant}
        seen_etymon_ids: set[int] = set(all_seed_ids)
        frontier: list[int] = list(all_seed_ids)
        approved_hits = _seed_direct_hits(direct, variant, approved)

        # BFS through descent edges (frontier is the set of etymon_ids
        # whose parents we haven't yet visited). One DB round-trip per
        # depth; corpus depth rarely exceeds ~5.
        while frontier:
            next_frontier: list[int] = []
            # Chunk the IN-list so a frontier wider than the SQLite variable
            # limit doesn't raise "too many SQL variables" (frontiers stay tiny
            # in practice, so this is a single chunk on the hot path).
            edges: list[sqlite3.Row] = []
            for i in range(0, len(frontier), _BFS_PARAM_CHUNK):
                chunk = frontier[i : i + _BFS_PARAM_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                edges.extend(
                    conn.execute(
                        f"""SELECT e.id, e.canonical_form, e.language, d.edge_type
                            FROM etymon_descent d
                            JOIN etymon e ON e.id = d.parent_id
                            WHERE d.child_id IN ({placeholders})""",
                        chunk,
                    ).fetchall()
                )
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
    """Dedupe by (canonical_form, language), then move reconstructed
    forms to the back. With the wave-2 precursor/postcursor stack now
    in APPROVED_LANGUAGES, the BFS can return both an attested ancestor
    (e.g. OE `tūn`) AND a reconstructed proto-form (e.g. proto-germanic
    `*tūnaz`) at the same depth; without a tiebreaker, SQL row-insertion
    order picks for us, which can put `*tūnaz` ahead of `tūn` and feed
    the semantic-check pass a reconstruction when an attested form is
    available. The starred-form convention (linguistic standard for
    reconstructed lemmas) gives us a cheap, content-stable tiebreaker:
    sort `*` keys after non-`*` keys, stable within each group so BFS
    layer order (closer ancestors first) is preserved."""
    seen: set[tuple[str, str]] = set()
    out: list[AncestorMatch] = []
    for m in matches:
        k = (m.canonical_form, m.language)
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    out.sort(key=lambda m: m.canonical_form.startswith("*"))
    return out


def _ancestor_glosses(db_path: Path | str, etymon_id: int) -> list[str]:
    """Pull etymon_gloss rows for an etymon (best-effort semantic check helper)."""
    conn = _connect_ro(db_path)
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
  "attested_in": "<language code if attested in any real language>" OR "none" (no etymology found) OR "modern_coinage" (Tolkien/Dunsany/Gygax/etc invention, no historical attestation),
  "historical_form": "<the attested form in the source language>" OR null,
  "gloss": "<short meaning>" OR null,
  "citation": "<dictionary source you're confident about: 'Bosworth-Toller', 'Cleasby-Vigfusson', 'LSJ', 'Etymonline', etc.>" OR null,
  "confidence": "high" | "medium" | "low",
  "bar_reason": "<one of: no_etymology_found / outside_language_family / uncertain_attestation / modern_coinage / proper_noun_only>" OR null,
  "reasoning": "<1-2 sentences>"
}}

IMPORTANT: when the etymology IS attested but the language lies OUTSIDE the approved list, still name the SPECIFIC language code in `attested_in` (e.g., "sanskrit", "hebrew", "modern-french", "japanese"). Do NOT collapse to a generic "outside_family" string — the caller wants to track which non-approved languages the corpus is missing so the approved list can be expanded over time. Use `attested_in: "modern_coinage"` only when the name is genuinely a 20th-century author's invention with no historical root.

Be CONSERVATIVE. If you're not confident the name maps to a real word in the approved list, prefer setting attested_in="none" or "outside_family" with appropriate bar_reason. Modern fantasy authors (Tolkien, Dunsany, Gygax) coined many such names; do not retrofit etymology onto coinages. Proper-noun-only attestations (literary characters whose name became a common noun only later) should be barred with bar_reason=proper_noun_only.

Output ONLY the JSON, no preamble."""


def _call_gemini(
    prompt: str,
    *,
    api_key: str | None = None,
    model: str,
    timeout_s: float,
) -> dict:
    """Send `prompt` to Gemini and return the parsed JSON response.

    Shared transport for _llm_full_research and _llm_semantic_check —
    both expect `responseMimeType: application/json` with a JSON object
    in the first candidate's text. Key passed in `x-goog-api-key`
    header to keep it out of URL-logging paths.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    url = _GEMINI_API_URL_TEMPLATE.format(model=model)
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
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        resp = json.loads(r.read())
    text = resp["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def _llm_full_research(
    name: str,
    description: str,
    *,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    approved: frozenset[str] = CANONICAL_LANGUAGES,
    timeout_s: float = 60.0,
) -> dict:
    """Call Gemini with the etymology-research prompt; return parsed JSON.

    `approved` defaults to `CANONICAL_LANGUAGES` (not `APPROVED_LANGUAGES`)
    on purpose: we don't want the LLM scoring against the precursor /
    postcursor stack (proto-codes like `iir-pro` and ancient intermediates
    like `xpr`/`fa-cls`/`syc`) where it has weak training signal. The
    descent walker still uses the full `APPROVED_LANGUAGES` gate, and
    `_classify_llm_result` accepts any `attested_in` that lands in
    `APPROVED_LANGUAGES` (post-alias-normalization), so an LLM that
    happens to identify a Pali or Coptic attestation directly still
    works — we just don't *advertise* those codes in the prompt."""
    prompt = _RESEARCH_PROMPT_TEMPLATE.format(
        name=name,
        description=description,
        langs=", ".join(sorted(approved)),
    )
    return _call_gemini(prompt, api_key=api_key, model=model, timeout_s=timeout_s)


# ---------------------------------------------------------------------
# LLM semantic-check (cheap verification of pre-filter resolutions).
# ---------------------------------------------------------------------

_SEMANTIC_CHECK_PROMPT_TEMPLATE = """You are reviewing a candidate match between a fantasy/gaming creature name and an attested historical word from an etymological corpus.

Candidate name: {name}
Description: {description}

Proposed ancestor (from a wiktionary descent chain):
- form: {form}
- language: {language}
- gloss: {gloss}

Question: is this proposed ancestor the GENUINE etymological source of the creature, or do they merely share a form (homograph collision / modern coinage from a real word)?

Return ONE JSON object:
{{
  "verdict": "match" | "mismatch" | "uncertain",
  "reasoning": "<1-2 sentences>"
}}

Examples to calibrate:
- Harpy + ancient-greek ἅρπυια ("snatcher"): MATCH. Greek harpyiai are exactly the same creature concept.
- Troll + old-norse trǫll ("monster, giant"): MATCH. Same folkloric creature.
- Cloaker (a D&D 1981 monster that disguises as a cloak) + old-french cloque ("bell-cape"): MISMATCH. The CREATURE is a 1981 Gygax coinage even though the form derives from a real garment word; Cloaker is not historically attested as a beast in OFr or any other approved-family corpus.
- Drow (a D&D dark elf) + scots `drow`/`trow` (a local Orkney/Shetland Norn-derived word for a troll-like sprite): MATCH. The Scots `drow`/`trow` IS a real folkloric creature (a Shetland/Orcadian fairy or troll), and Gygax's "Drow" was named from this Scots word; the etymological chain is genuine. (Caution: do NOT match Drow against ME `truwien` "to trust" — that's a homograph in the same form.)
- Goblin + old-french gobelin ("evil sprite"): MATCH. Same folkloric being.

Be CONSERVATIVE: when in doubt prefer "mismatch" or "uncertain". A modern fantasy author naming a new creature after an existing word does not make the creature etymologically descended from that word.

Output ONLY the JSON."""


def _llm_semantic_check(
    name: str,
    description: str,
    ancestor: AncestorMatch,
    ancestor_glosses: list[str],
    *,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    timeout_s: float = 30.0,
) -> dict:
    """Cheap LLM yes/no on whether a pre-filter ancestor is the actual
    etymology of the creature, or just a homograph collision."""
    gloss_text = "; ".join(ancestor_glosses) if ancestor_glosses else "(no gloss in corpus)"
    prompt = _SEMANTIC_CHECK_PROMPT_TEMPLATE.format(
        name=name,
        description=description,
        form=ancestor.canonical_form,
        language=ancestor.language,
        gloss=gloss_text,
    )
    return _call_gemini(prompt, api_key=api_key, model=model, timeout_s=timeout_s)


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
    except urllib.error.HTTPError as e:
        return _llm_http_error_resolution(name, e)
    except (
        # OSError covers urllib.error.URLError (subclass), plus
        # socket-level TimeoutError / ConnectionError that can escape
        # urllib's wrapping.
        OSError,
        RuntimeError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
    ) as e:
        # IndexError can occur when Gemini's safety filter blocks the
        # prompt and returns an empty `candidates` list.
        logger.warning("LLM full-research call failed for %s: %s", name, e)
        return Resolution(
            usable=False,
            etymon_id=None,
            bar_reason=BAR_REASON_UNCERTAIN,
            resolution_method="llm_full_research",
            confidence="low",
            citation=None,
            reasoning=f"LLM call failed: {e}",
        )
    return _classify_llm_result(db_path, result)


def _llm_http_error_resolution(name: str, e: urllib.error.HTTPError) -> Resolution:
    """Build an UNCERTAIN-bar Resolution from an HTTP error, capturing
    the response body for diagnostics. Gemini returns useful detail
    in the body (quota-exceeded, safety-block reason, malformed-prompt)
    that the generic str() representation drops."""
    try:
        body = e.read().decode("utf-8", errors="replace")[:800]
    except Exception:
        body = "(could not read response body)"
    logger.warning("LLM full-research HTTP error for %s: status=%s body=%s", name, e.code, body)
    return Resolution(
        usable=False,
        etymon_id=None,
        bar_reason=BAR_REASON_UNCERTAIN,
        resolution_method="llm_full_research",
        confidence="low",
        citation=None,
        reasoning=f"LLM HTTP error {e.code}: {body[:200]}",
    )


def _classify_llm_result(db_path: Path | str, result: dict) -> Resolution:
    """Map the LLM's full-research result dict to a Resolution.

    Five distinct paths:
    1. confidence=low + a real-attestation language → UNCERTAIN bar
    2. attested_in approved + historical form found → USABLE (anchored
       to existing etymon if present; otherwise NOT_IN_CORPUS bar)
    3. attested_in real language but NOT approved → OUTSIDE_FAMILY bar
       with unapproved_language/form recorded for later prioritization
    4. attested_in='modern_coinage' / 'outside_family' / 'none' →
       barred with the appropriate sentinel reason
    """
    # Lowercase + space-to-dash normalize attested_in so 'Latin',
    # 'Old French', and 'old-french' all match APPROVED_LANGUAGES (which
    # uses dashed-lowercase). Without this, the empirical 2026-05-06
    # pfsrd2 mining run mis-classified ~80 morphemes as
    # outside_language_family because the LLM returned capitalized /
    # space-separated language names. Underscores are preserved because
    # the sentinel values in _NO_REAL_ATTESTATION_SENTINELS use them
    # ('modern_coinage', 'outside_family').
    raw_attested = result.get("attested_in") or "none"
    normalized = raw_attested.strip().lower().replace(" ", "-")
    # Map descriptive names ('sanskrit', 'arabic', ...) to ISO codes
    # ('sa', 'ar', ...) so they match etymon-table conventions and
    # APPROVED_LANGUAGES. Pass-through if no alias defined.
    attested_in = _LANGUAGE_ALIAS_MAP.get(normalized, normalized)
    confidence = result.get("confidence") or "low"
    historical = result.get("historical_form")
    citation = result.get("citation")
    reasoning = result.get("reasoning")
    gloss = result.get("gloss")
    bar_reason = result.get("bar_reason")

    # Conservative: low-confidence answers are barred even if attested_in
    # names a real language. Per user 2026-05-05.
    if confidence == "low" and attested_in not in _NO_REAL_ATTESTATION_SENTINELS:
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
        return _anchor_to_etymon(
            db_path, attested_in, historical, gloss, confidence, citation, reasoning
        )

    # LLM named a real language but NOT in APPROVED_LANGUAGES:
    # bar with outside_language_family, but RECORD what language and form
    # so we can aggregate "candidate languages to approve" later. Per user
    # 2026-05-05: tracking these gives us a prioritization list.
    if attested_in not in _NO_REAL_ATTESTATION_SENTINELS:
        return Resolution(
            usable=False,
            etymon_id=None,
            bar_reason=BAR_REASON_OUTSIDE_FAMILY,
            resolution_method="llm_full_research",
            confidence=confidence,
            citation=citation,
            reasoning=(
                f"attested in {attested_in} as '{historical}'"
                + (f" '{gloss}'" if gloss else "")
                + " (not in APPROVED_LANGUAGES)"
                + (f"; {reasoning}" if reasoning else "")
            ),
            unapproved_language=attested_in,
            unapproved_form=historical,
        )

    # "none" / "modern_coinage" / "outside_family" sentinel → barred with
    # no specific language to record. Use LLM-supplied bar_reason, fall
    # back to inference.
    if not bar_reason:
        bar_reason = (
            BAR_REASON_MODERN_COINAGE
            if attested_in == "modern_coinage"
            else BAR_REASON_OUTSIDE_FAMILY
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


def _anchor_to_etymon(
    db_path: Path | str,
    attested_in: str,
    historical: str,
    gloss: str | None,
    confidence: str,
    citation: str | None,
    reasoning: str | None,
) -> Resolution:
    """LLM identified an approved-language attestation. Anchor it to
    an existing etymon row when one exists; otherwise bar with
    NOT_IN_CORPUS (so a future backfill can ingest the etymon)."""
    etymon_id = _lookup_etymon_id(db_path, historical, attested_in)
    if etymon_id is None:
        return Resolution(
            usable=False,
            etymon_id=None,
            bar_reason=BAR_REASON_NOT_IN_CORPUS,
            resolution_method="llm_full_research",
            confidence=confidence,
            citation=citation,
            reasoning=(
                f"attested in {attested_in} as '{historical}'"
                + (f" '{gloss}'" if gloss else "")
                + " but no matching etymon row exists in the corpus"
                + (f"; {reasoning}" if reasoning else "")
            ),
            unapproved_language=attested_in,
            unapproved_form=historical,
        )
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


def _lookup_etymon_id(db_path: Path | str, form: str, language: str) -> int | None:
    """Anchor an LLM-named (form, language) pair to an existing etymon row.

    Case-insensitive on form to absorb any case variation in the LLM
    response (e.g. "Trǫll" vs "trǫll"); language is matched exactly
    since codes are stable.
    """
    conn = _connect_ro(db_path)
    try:
        row = conn.execute(
            "SELECT id FROM etymon WHERE LOWER(canonical_form) = LOWER(?) AND language = ?",
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
    semantic_check_caller=_llm_semantic_check,
) -> Resolution:
    """Resolve a (name, description) into a Resolution.

    Routing:
    - Pre-filter via descent_walking_lookup, then semantic-check each
      candidate ancestor with a cheap LLM call. The check rejects
      homograph collisions: a D&D 'Cloaker' has a real OFr ancestor
      `cloque` but the creature is a 1981 Gygax coinage, not an
      attested OFr beast — semantic check returns 'mismatch' and we
      fall through to full research (which will likely bar it as
      modern_coinage).
    - First semantic-check 'match' wins.
    - All semantic-check 'mismatch' or pre-filter miss → full LLM
      research.
    - `skip_llm=True` short-circuits both LLM steps. In skip mode we
      accept pre-filter resolutions blindly (legacy behavior, used in
      tests / offline-coverage estimation). The caller is on the hook
      for verifying separately.
    """
    pre = descent_walking_lookup(db_path, name)
    if pre:
        if skip_llm:
            return _skip_llm_resolution(pre)
        return _semantic_check_candidates(
            db_path, name, description, pre, semantic_check_caller, llm_caller
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


def _skip_llm_resolution(pre: list[AncestorMatch]) -> Resolution:
    """skip_llm=True branch: trust the pre-filter's first hit blindly
    without semantic verification. Used by tests and offline coverage
    estimation; the caller is on the hook for verifying separately."""
    winner = pre[0]
    return Resolution(
        usable=True,
        etymon_id=winner.etymon_id,
        bar_reason=None,
        resolution_method="descent_lookup",
        confidence="high",
        citation="wiktionary-descent-chain",
        reasoning=(
            f"matched {winner.language}/{winner.canonical_form} "
            f"via {winner.edge_type} (skip_llm=True; not semantically verified)"
        ),
    )


def _semantic_check_candidates(
    db_path: Path | str,
    name: str,
    description: str,
    pre: list[AncestorMatch],
    semantic_check_caller,
    llm_caller,
) -> Resolution:
    """Verify each pre-filter candidate semantically; first 'match'
    wins. If every candidate fails, fall through to full research,
    re-tagging a homograph-collision bar reason if the full-research
    bar would otherwise have been generic no-etymology / outside-family
    / modern-coinage (pre-filter DID find a form match, so it's
    homograph-shaped, not 'no etymology')."""
    for cand in pre:
        glosses = _ancestor_glosses(db_path, cand.etymon_id)
        check = _safe_semantic_check(name, description, cand, glosses, semantic_check_caller)
        if check is None:
            continue
        if (check.get("verdict") or "uncertain") == "match":
            return Resolution(
                usable=True,
                etymon_id=cand.etymon_id,
                bar_reason=None,
                resolution_method="descent_lookup",
                confidence="high",
                citation="wiktionary-descent-chain",
                reasoning=(
                    f"matched {cand.language}/{cand.canonical_form} "
                    f"via {cand.edge_type}; semantic-check: {check.get('reasoning') or ''}"
                ),
            )
    full = _resolve_via_llm(db_path, name, description, llm_caller=llm_caller)
    if not full.usable and full.bar_reason in (
        BAR_REASON_NO_ETYMOLOGY,
        BAR_REASON_OUTSIDE_FAMILY,
        BAR_REASON_MODERN_COINAGE,
    ):
        return Resolution(
            usable=False,
            etymon_id=None,
            bar_reason=BAR_REASON_HOMOGRAPH,
            resolution_method=full.resolution_method,
            confidence=full.confidence,
            citation=full.citation,
            reasoning=(
                f"pre-filter matched {pre[0].language}/{pre[0].canonical_form} "
                f"but semantic check rejected (homograph collision); "
                f"full research: {full.reasoning}"
            ),
        )
    return full


def _safe_semantic_check(
    name: str,
    description: str,
    cand: AncestorMatch,
    glosses: list[str],
    semantic_check_caller,
) -> dict | None:
    """Wrap a semantic-check call with the same HTTP / URL / parse
    error handling as _llm_full_research. On any failure, log + return
    None so the outer loop falls through to the next candidate."""
    try:
        return semantic_check_caller(name, description, cand, glosses)
    except urllib.error.HTTPError as http_exc:
        try:
            body = http_exc.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            body = "(could not read response body)"
        logger.warning(
            "semantic check HTTP error for %s -> %s/%s: status=%s body=%s; falling through",
            name,
            cand.language,
            cand.canonical_form,
            http_exc.code,
            body,
        )
        return None
    except (
        # OSError covers urllib.error.URLError (subclass), plus
        # TimeoutError / ConnectionError.
        OSError,
        RuntimeError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
    ) as exc:
        logger.warning(
            "semantic check failed for %s -> %s/%s (%s); falling through",
            name,
            cand.language,
            cand.canonical_form,
            exc,
        )
        return None


def fetch_resolved_input_names(
    db_conn: sqlite3.Connection,
    approach_version: str = APPROACH_VERSION,
) -> set[str]:
    """Return the set of `input_name` values that already have a row in
    `fantasy_morpheme` for the given `approach_version`.

    Used by `lexicon mine-fantasy-name --skip-resolved` to avoid paying
    for re-resolution (each costs one Gemini call) on inputs that
    already have a result. Names are returned with their original case
    as stored — callers MUST compare case-insensitively to match
    `write_resolution`'s UNIQUE constraint, which uses
    `COLLATE NOCASE` on the `input_name` column (see
    `_create_fantasy_morpheme_table`). The canonical pattern:

        already = {n.lower() for n in fetch_resolved_input_names(conn)}
        if input_name.lower() in already: skip
    """
    cur = db_conn.execute(
        "SELECT input_name FROM fantasy_morpheme WHERE approach_version = ?",
        (approach_version,),
    )
    return {row[0] for row in cur.fetchall()}


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
            reasoning, unapproved_language, unapproved_form, processed_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (input_name, approach_version) DO UPDATE SET
            input_description    = excluded.input_description,
            usable               = excluded.usable,
            etymon_id            = excluded.etymon_id,
            bar_reason           = excluded.bar_reason,
            unapproved_language  = excluded.unapproved_language,
            unapproved_form      = excluded.unapproved_form,
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
            resolution.unapproved_language,
            resolution.unapproved_form,
            now,
        ),
    )
    return cur.fetchone()[0]


# Tags applied to every etymon resolved usable through the fantasy
# pipeline. 'fantasy' is the register marker (filter realistic-mode
# generation OUT of fantasy entries); 'monster' is added because the
# canonical input source is a creature-name corpus (pdsrd-data
# monsters/, etc.) and a unified `monster` tag makes querying the
# fantasy-creature inventory straightforward — the existing seed
# entries (wyrm, elf, thyrs, ...) already carry it.
FANTASY_TAGS: tuple[str, ...] = ("fantasy", "monster")


def tag_etymon_as_fantasy(
    db_conn: sqlite3.Connection,
    etymon_id: int,
    *,
    tags: tuple[str, ...] = FANTASY_TAGS,
) -> None:
    """Apply fantasy-pipeline tags to an etymon (idempotent on etymon_tag PK)."""
    db_conn.executemany(
        "INSERT OR IGNORE INTO etymon_tag (etymon_id, tag) VALUES (?, ?)",
        [(etymon_id, t) for t in tags],
    )


def backfill_fantasy_tag_from_monster_tag(
    db_conn: sqlite3.Connection,
) -> tuple[int, int]:
    """Add 'fantasy' tag to every etymon that already has 'monster'.

    The seed (meanings.json) tags wyrm/elf/thyrs/grīma/sceocca/etc. as
    `monster` but doesn't know about `fantasy` as a register filter.
    This helper re-tags the existing inventory so register-aware
    generation can find them.

    Idempotent. Returns (n_etymons_processed, n_tags_added).
    """
    rows = db_conn.execute(
        """SELECT DISTINCT et.etymon_id
           FROM etymon_tag et
           WHERE et.tag = 'monster'
             AND NOT EXISTS (
               SELECT 1 FROM etymon_tag et2
               WHERE et2.etymon_id = et.etymon_id AND et2.tag = 'fantasy'
             )"""
    ).fetchall()
    ids = [r[0] for r in rows]
    if ids:
        db_conn.executemany(
            "INSERT OR IGNORE INTO etymon_tag (etymon_id, tag) VALUES (?, 'fantasy')",
            [(eid,) for eid in ids],
        )
    return (len(ids), len(ids))
