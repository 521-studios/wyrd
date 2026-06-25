"""Toponym-attestation mining (wyrd-skm Phase 3.0a).

Scans ``toponym_etymology.notes`` for ``(form, year)`` pairs and writes
them to ``toponym_attestation``. Per-toponym dated historical spellings
are the raw input the wyrd-unuo Phase 3.3 ``project_period_forms``
projector consumes to derive per-etymon period forms.

Four compiled regex patterns cover the common scholarly shapes:

* ``_ATTEST_FORM_YEAR_RE`` — ``FORM in YEAR`` / ``FORM, YEAR`` / ``FORM, in YEAR``
* ``_ATTEST_CHAIN_FORM_YEAR_RE`` — ``;FORM YEAR`` (chain-anchored
  bare connector for trailing chain elements)
* ``_ATTEST_FORM_DOMESDAY_RE`` — ``FORM in Domesday[ Book]`` / ``FORM, D.B.``
* ``_ATTEST_DOMESDAY_HAS_FORM_RE`` — ``Domesday has FORM`` /
  ``D.B. has FORM``

Three precision gates:
* Connector requirement (comma, semicolon, or ``in``) between form and
  year — drops "After 1066" / "Source 1234" flow-text false positives.
* Lowercase-required form filter — drops scholarly source
  abbreviations (LPR / LI / LF / DB / MS) without an explicit
  per-token list.
* Source-attribution-chain detector — suppresses ``<year> <source>,
  <year>`` shapes like ``Chevington 1535 VE, 1539 Wills, 1544 LP``
  where ``Wills`` is the source name in a multi-source chain, not a
  place form.

Year range 700-1700 (post-Roman through pre-modern). Welsh + Norman +
OE diacritics covered in the form-character class.

LLM-free, idempotent, reversible via
``clear_enrichment(stage="attestations")``.
"""

from __future__ import annotations

import re

from wyrd.generators.kenning.lexicon.attestation_years import (
    _ATTESTED_YEAR_MAX_LOOKUP,
    _ATTESTED_YEAR_MIN_LOOKUP,
    _TOPONYM_NOTE_PAGE_MARKER_RE,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB

# Domesday Book (1086) is the load-bearing dated reference in English
# toponym scholarship — almost every Mawer / Skeat / Ekwall entry cites
# its Domesday spelling. We canonicalize it to the year so a single
# regex captures both "Cestretone in Domesday Book" and "Domesday Book
# has Chingestone". 'D.B.' is the same source, abbreviated.
_DOMESDAY_YEAR = 1086

# Form character class — leading capital plus letters / OE specials /
# Welsh + Norman diacritics / hyphens.
#
# The base segment is bounded at 4-30 chars (lead + {3,29} continuation);
# hyphenated suffixes can extend the total to ~43 chars (Hædan-ham at 9,
# Bedingafelda at 12, Llanfaên-y-bryn at 15 are typical). The lower
# bound excludes 3-letter sentence connectives ("The", "But", "And",
# "Of") that would otherwise leak through the year-anchor patterns; the
# upper bound is generous enough that real toponym variants haven't
# tripped it on the production corpus.
#
# The diacritic set covers the production toponym_etymology.notes
# corpus: Welsh ŵâêôûŷ + macron-vowel ē (Welsh-stratum work landed in
# PR #105), Norman çéè (early French/Anglo-Norman charter spellings),
# OE specials æðþœǣ + macron set āīōū. Both cases of each diacritic so
# capitalised lemmas (``Hēafod``, ``Ŷrwyrne``, ``Llanfaên``) match.
#
# The follow-up ``_form_passes_filter`` check additionally requires at
# least one lowercase letter so pure-uppercase source abbreviations
# (LPR, LI, LF, DB) don't slip through.
_FORM_CHARSET = r"A-Za-zÆÐÞŒæðþœǣĒĀĪŌŪēāīōūȳŴÂÊÔÛŶŵâêôûŷÇÉÈçéè"
_FORM_PATTERN = (
    rf"[A-ZÆÐÞŒĒĀĪŌŪŴÂÊÔÛŶÇÉÈ][{_FORM_CHARSET}]{{3,29}}(?:[-][{_FORM_CHARSET}]{{1,12}})*"
)


# wyrd-hcd9: source-attribution chain FP detector.
#
# Mawer / Skeat / Ekwall use the convention `<year> <source>` to
# attribute SAME-FORM attestations to multiple sources, chained by
# COMMAS:
#
#   Chevington 1535 VE, 1539 Wills, 1544 LP
#                       ↑ source name, NOT a place form
#
# The regex would otherwise match ``Wills, 1544`` as form=Wills
# year=1544 because it follows the FORM-comma-YEAR shape. The
# distinguishing structural signal is two-fold:
#   (a) the form is preceded by ``<year> `` (a 3-4 digit year + space)
#   (b) the form is followed by ``, <year>`` (next source-attribution)
#
# COMMA-after-form is the load-bearing distinguisher from
# semicolon-separated multi-form chains
# (``Edreston ; 1242 Cl. ...``) where the form IS a real attestation:
# scholarly convention puts commas between same-form sources and
# semicolons between different-form attestations. Both conditions
# together identify the source-attribution-chain FP without
# false-suppressing real chained attestations.
_SOURCE_CHAIN_PRECEDING_YEAR_RE = re.compile(r"\d{3,4}\s+$")
# Note: no ``^`` anchor — ``re.match(string, pos)`` already anchors
# at ``pos``, and the pattern's ``^`` would only match at string
# start (position 0), not at ``pos``. ``\s*,\s*\d{3,4}\b`` covers
# the post-form `, 1544` lookahead with optional whitespace either
# side of the comma.
_SOURCE_CHAIN_FOLLOWING_YEAR_RE = re.compile(r"\s*,\s*\d{3,4}\b")


def _form_passes_filter(form: str) -> bool:
    """Final form-quality gate beyond the regex character class.

    Place-name attestations always carry at least one lowercase letter
    (``Cestretone``, ``Hædan-ham``, ``Wyntewurthe``). Pure-uppercase
    runs that match the regex are scholarly source abbreviations
    (``LPR``, ``LI``, ``LF``, ``DB``, ``MS``, ``FA``) and never
    legitimate forms — those exist in citation suffixes and sometimes
    pick up year-shaped digits nearby that survive the page-marker
    guard. Requiring a lowercase letter gives us a single check that
    handles the entire abbreviation false-positive class.
    """
    if not any(c.islower() for c in form):
        return False
    return form.lower() not in _ATTEST_FORM_BLACKLIST


# Year pattern — same range as _earliest_year_in_notes (post-Roman
# 700-1700) so we share its filter on page-references / publication
# years.
_YEAR_PATTERN = r"7\d{2}|[89]\d{2}|1[0-6]\d{2}|1700"

# "FORM in YEAR" or "FORM, YEAR" or "FORM, in YEAR" — the canonical
# citation shapes in Mawer / Skeat / Ekwall. Matches "Cestretone in
# 1210", "Iselham, 1302", "Tadelowe, in 1302", "Spelt Knesworthe in
# 1316" (the Spelt prefix doesn't need its own pattern — it leaves the
# form right where this one anchors).
#
# Two branches in the connector group — punctuation-then-optional-"in"
# OR bare-"in" — keeps an explicit anchor between form and year. A
# pure-whitespace separator (``"After 1066"``) would generate too many
# false positives when a sentence happens to have a capitalized word
# followed by a year-shaped digit run.
#
# Trailing ``(?!\s*\(p+\.)`` negative lookahead suppresses
# academic-citation shape "Author, 1086 (p. 59)" — when ``(p.`` /
# ``(pp.`` immediately follows the year-paren, the year is a
# publication date and the preceding capitalised word is a scholar
# name, not a place-name attestation. Real attestations cite the
# SOURCE in the parens (``"Cestretone, 1086 (D.B.)"``,
# ``"Iselham, 1302 (F.A.)"``); page references show up as the LATER
# component of a parenthetical (``"(D.B., p. 102)"``), which the
# lookahead ignores because the ``(`` is followed by a non-``p`` char.
_ATTEST_FORM_YEAR_RE = re.compile(
    rf"\b(?P<form>{_FORM_PATTERN})"
    r"(?:\s*[,;]\s*(?:in\s+)?|\s+in\s+)"
    rf"(?P<year>{_YEAR_PATTERN})\b"
    r"(?!\s*\(p+\.)"
)

# Chain-element shape — the LAST item of a comma/semicolon-separated
# chain often drops the explicit connector between form and year:
# ``"Cestretone in 1210; Cestrede, 1218; Chestreton 1242"``. The
# leading ``;`` (or ``.``) plus optional whitespace anchors this
# pattern to chain positions; ``After 1066 the conquest came`` won't
# match because no ``;`` precedes ``After``. Bare-whitespace connector
# is still safe within this anchor since chain context already implies
# we're inside a citation list.
_ATTEST_CHAIN_FORM_YEAR_RE = re.compile(
    rf"[;]\s*(?P<form>{_FORM_PATTERN})\s+(?P<year>{_YEAR_PATTERN})\b"
)

# Domesday-anchored patterns. Spelled-out (``Domesday Book``) and
# abbreviated (``D.B.``) shapes each cover the form-before-marker and
# marker-before-form orderings; the four-pattern set folds to two
# regex alternations on the marker side.
#
# ``D\.\s*B\.`` carries an explicit trailing ``(?:\b|,|;|\s)`` — the
# ``\b`` alone fails to match between ``.`` and a space (both are
# non-word) so without the alternation a sentence like ``"Cestretone
# D.B. has more"`` would mis-match. Pinned by a regression test.
_ATTEST_FORM_DOMESDAY_RE = re.compile(
    rf"\b(?P<form>{_FORM_PATTERN})\s+in\s+Domesday(?:\s+Book)?\b"
    rf"|\b(?P<form2>{_FORM_PATTERN})\s*,?\s+D\.\s*B\.(?:\b|,|;|\s)"
)

# Marker-before-form variants of the same data.
_ATTEST_DOMESDAY_HAS_FORM_RE = re.compile(
    rf"\bDomesday(?:\s+Book)?\s+has\s+(?P<form>{_FORM_PATTERN})"
    rf"|\bD\.\s*B\.\s+has\s+(?P<form2>{_FORM_PATTERN})"
)

# Domesday-anchored regex set, paired with the year that should be
# stamped onto every match. The driver loop in
# ``_extract_attestation_pairs`` iterates this tuple — adding a new
# Domesday phrasing means adding one regex here, not a fresh code path.
_DOMESDAY_RES: tuple[tuple[re.Pattern[str], int], ...] = (
    (_ATTEST_FORM_DOMESDAY_RE, _DOMESDAY_YEAR),
    (_ATTEST_DOMESDAY_HAS_FORM_RE, _DOMESDAY_YEAR),
)


# Forms that match the regex character class but are never legitimate
# place-name attestations. Most false positives filter via the
# year-anchor (a year must follow IMMEDIATELY) plus
# ``_form_passes_filter``'s lowercase-letter requirement; this list
# handles the leftover mixed-case false positives that survive both
# filters. Lowercased entries; membership lookup is case-insensitive.
_ATTEST_FORM_BLACKLIST = frozenset(
    {
        # 1. Domesday-citation noise — "Domesday Book has Foo" would
        #    otherwise match "Domesday" as a form on its own under the
        #    year-anchored pattern when a year happens to be nearby.
        "domesday",
        "book",
        # 2. Citation-prefix words — Mawer / Skeat lead with these
        #    before naming the form. The regex captures them when the
        #    next year-shaped digit is close enough.
        "spelt",
        "formerly",
        "apparently",
        # 3. Generic English connectives that survive the lowercase-
        #    required filter (mixed-case at sentence start).
        "the",
        "from",
        "where",
        "there",
        "these",
        "this",
        "that",
        "and",
        "but",
        "with",
        "here",
        # 4. Scholar surnames — appear inline in citation prose
        #    ("according to Kemble"). Never the toponym itself.
        "kemble",
        "skeat",
        "ekwall",
        "mawer",
        "joyce",
        "thorpe",
        "kelly",
        # 5. Source / archival names that read proper-noun-ish.
        "pipe",
        "roll",
        "red",
        "inquisitio",
        # 6. Mixed-case source abbreviations that the lowercase-required
        #    filter doesn't catch. Pure-uppercase abbreviations (LPR,
        #    LI, DB) are filtered in `_form_passes_filter` directly.
        "cod",
        "dipl",
        "vol",
        "ipm",
        "ipms",
        # EPNS roll/record-series names that read like a Title-case place form
        # and otherwise leak as dated attestations (e.g. "Orig, 1325",
        # "Banco, 1399", "...Fees, 1283"): Originalia Rolls, De Banco Rolls,
        # Book of Fees. Not historical spellings of any toponym.
        "orig",
        "banco",
        "fees",
    }
)


def _matched_form(m: re.Match[str]) -> str | None:
    """Resolve which named group fired in a Domesday alternation.

    Each compiled regex carries either a ``form`` group (single branch) or
    both ``form`` and ``form2`` (two branches sharing one regex). Returning
    the first non-None capture, stripped of trailing punctuation, gives one
    accessor for both shapes.
    """
    for key in ("form", "form2"):
        captured = m.groupdict().get(key)
        if captured is not None:
            return captured.rstrip(",.;:")
    return None


def _admit_year_match(
    m: re.Match[str],
    notes: str,
    pairs: set[tuple[str, int]],
    *,
    context_is_body: bool,
    telemetry: dict[str, int] | None,
) -> None:
    """Validate and absorb a ``(form, year)`` match from the year-anchored or
    chain-anchored regex into ``pairs``.

    Shared guards:
    * form-quality filter (lowercase-required, blacklist),
    * year-range bounds,
    * immediate-predecessor page-marker check (rejects ``"Bedinga feld,
      p. 1086"`` shape where the year is actually a page reference; probe
      scope is narrow ``0:year_start`` since pages cited AFTER the year are
      caught by the ``(?!\\s*\\(p+\\.)`` lookahead on ``_ATTEST_FORM_YEAR_RE``),
    * source-attribution-chain check (rejects ``"1539 Wills, 1544 LP"`` shape
      where the form is a SOURCE name in a multi-source chain — see
      _SOURCE_CHAIN_*_RE comments). Both the preceding-``<year> `` and
      following-``, <year>`` conditions are required so real attestation
      chains (``Edreston ; 1242 ...``) aren't suppressed. Calibrated for
      short notes strings; body-text callers (``context_is_body=True``)
      disable suppression because in body text the same pattern is
      dominantly legitimate (wyrd-9ekl). Telemetry bumps on every pattern
      match — even in body mode — so callers see how often the shape occurs,
      decoupled from whether the guard suppressed.
    """
    form = m.group("form").rstrip(",.;:")
    if not _form_passes_filter(form):
        return
    year = int(m.group("year"))
    if year < _ATTESTED_YEAR_MIN_LOOKUP or year > _ATTESTED_YEAR_MAX_LOOKUP:
        return
    if _TOPONYM_NOTE_PAGE_MARKER_RE.search(notes, 0, m.start("year")):
        return
    form_start = m.start("form")
    form_end = m.end("form")
    if _SOURCE_CHAIN_PRECEDING_YEAR_RE.search(
        notes, 0, form_start
    ) and _SOURCE_CHAIN_FOLLOWING_YEAR_RE.match(notes, form_end):
        if telemetry is not None:
            telemetry["suppressed_by_source_chain"] = (
                telemetry.get("suppressed_by_source_chain", 0) + 1
            )
        if not context_is_body:
            return
    pairs.add((form, year))


def _extract_attestation_pairs(
    notes: str | None,
    *,
    context_is_body: bool = False,
    telemetry: dict[str, int] | None = None,
) -> list[tuple[str, int]]:
    """Extract ``(form, year)`` attestation pairs from a
    ``toponym_etymology.notes`` value or a long source-body string.

    Returns deduped tuples ordered by year ascending, with ties broken
    by form (alphabetical). The ordering is deterministic so callers
    can rely on first-row-per-key idempotency under the unique-index
    DB constraint.

    ``context_is_body``: when False (the default — appropriate for
    short toponym_etymology.notes strings), the
    _SOURCE_CHAIN_PRECEDING_YEAR_RE + _SOURCE_CHAIN_FOLLOWING_YEAR_RE
    guard fires to suppress source-name FPs like ``1539 Wills, 1544 LP``
    where ``Wills`` is the FP. When True (toponym_reverse_search loading
    a multi-page scholar body), the guard is disabled because the
    ``<year> FORM, <year>`` pattern is COMMON in body text and represents
    legitimate attestation chains like ``1086 Suttone, 1210``.

    ``telemetry``: optional mutable dict the function bumps when the
    source-chain pattern matches, regardless of whether the guard
    actually suppresses. Keys: ``suppressed_by_source_chain``. The
    counter measures how often the ``<year> FORM, <year>`` shape
    occurs in the input — body-mode callers see the prevalence even
    though the guard doesn't fire for them. Pass ``{}`` to surface;
    pass ``None`` (default) for callers that don't care. wyrd-9ekl
    observability seam.

    Pattern set (highest precision first):

    1. ``FORM in YEAR`` / ``FORM, YEAR`` / ``FORM; YEAR`` — the
       dominant scholarly shape. Year is filtered to 700-1700 via the
       same range used by ``_earliest_year_in_notes``. The
       ``(?!\\s*\\(p+\\.)`` negative lookahead rejects
       academic-citation shape ``"Author, 1086 (p. 59)"``.
    2. ``;FORM YEAR`` — chain-element bare connector. The LAST item
       of a citation chain often drops the explicit comma/in
       connector; the leading semicolon anchors this to chain
       positions so sentence-flow false positives don't leak.
    3. ``FORM in Domesday[ Book]`` — Domesday-anchored citation;
       year=1086.
    4. ``Domesday[ Book] has FORM`` — inverted shape for the same.
    5. ``FORM, D.B.`` / ``FORM D.B.`` — the abbreviated form.

    Page-reference false positives are guarded TWO ways: first the
    same ``_earliest_year_in_notes`` shared filter (a digit run
    preceded IMMEDIATELY by ``p.`` / ``pp.`` / ``vol.`` is rejected;
    catches ``"Bedinga feld, p. 59"`` shape), and second the
    parenthetical-page lookahead in pattern 1 above. The two combined
    cover both pre-year (``p. 1086``) and post-year (``1086 (p. 59)``)
    page-marker placements.

    Forms that match the regex syntactically but represent scholarly
    metadata (``Domesday``, ``Spelt``, source-author surnames) are
    filtered via ``_ATTEST_FORM_BLACKLIST``.
    """

    if not notes:
        return []
    pairs: set[tuple[str, int]] = set()

    # Year-anchored pattern: explicit connector between form and year.
    for m in _ATTEST_FORM_YEAR_RE.finditer(notes):
        _admit_year_match(m, notes, pairs, context_is_body=context_is_body, telemetry=telemetry)

    # Chain-element pattern: ``;FORM YEAR`` — the last item of a
    # comma/semicolon-separated citation chain often drops the
    # explicit connector. The leading ``;`` anchors this to chain
    # positions so sentence flow ("After 1066") can't slip through.
    for m in _ATTEST_CHAIN_FORM_YEAR_RE.finditer(notes):
        _admit_year_match(m, notes, pairs, context_is_body=context_is_body, telemetry=telemetry)

    # Domesday-anchored patterns (year fixed to 1086 per
    # _DOMESDAY_RES). The driver tuple folds the four shape variants
    # (form-before / marker-before / spelled / abbreviated) into two
    # regexes; adding a new Domesday phrasing means appending one
    # regex to the tuple.
    for pattern, year in _DOMESDAY_RES:
        for m in pattern.finditer(notes):
            form = _matched_form(m)
            if form is None or not _form_passes_filter(form):
                continue
            pairs.add((form, year))

    return sorted(pairs, key=lambda p: (p[1], p[0]))


def mine_toponym_attestations(
    db: LexiconDB,
    *,
    apply: bool = False,
    progress_every: int = 500,
) -> dict:
    """Populate ``toponym_attestation`` from ``toponym_etymology.notes``
    (wyrd-skm Phase 3.0a). For every etymology row carrying inline
    scholarly date citations like ``"Cestretone in 1210; Hadenham in
    Domesday Book"`` we extract ``(form, year)`` pairs and INSERT them
    into ``toponym_attestation`` keyed on the etymology's
    ``toponym_id``. Source attribution rides on the original
    ``toponym_etymology.source_id`` so a Skeat-derived Cestretone(1210)
    is distinguishable from a Mawer-derived one.

    LLM-free, idempotent, reversible (clear-enrichment
    --stage=attestations). Re-runs are no-ops thanks to the unique
    index ``idx_attestation_unique`` on
    ``(toponym_id, form, date_year, source_doc)``.

    Progress lines emit to stderr every ``progress_every`` rows and
    once at completion (CLAUDE.md mining-progress shape).
    ``progress_every`` is clamped to ≥1 so a caller passing 0 doesn't
    trip a modulo-zero (or, on the rate computation, divide-by-zero).

    Returns a result dict with row-counts so callers can report.
    """
    import sys
    import time

    def _emit_progress(scanned: int, total: int | None = None) -> None:
        """Print one CLAUDE.md-shape progress line to stderr.

        ``total=None`` is the mid-loop case (we don't know the final
        count yet); ``total=scanned`` is the post-loop closer that
        guarantees the final partial chunk surfaces. The rate clamps
        to a small floor so a sub-tick scan doesn't blow up.
        """
        elapsed = max(time.monotonic() - started, 1e-6)
        denom = scanned if scanned else 1
        head = f"[{scanned}/{total}]" if total is not None else f"[{scanned}]"
        print(
            f"  {head}  rows_with_pairs={rows_with_pairs} "
            f"candidates={len(candidate_inserts)} "
            f"rows_written={rows_written} "
            f"({elapsed / denom:.4f}s/entry)",
            file=sys.stderr,
            flush=True,
        )

    progress_every = max(progress_every, 1)
    cur = db.conn.execute(
        "SELECT toponym_id, source_id, notes FROM toponym_etymology "
        "WHERE notes IS NOT NULL AND notes != ''"
    )

    rows_scanned = 0
    rows_written = 0
    candidate_inserts: list[tuple[int, str, int, str]] = []
    rows_with_pairs = 0
    started = time.monotonic()
    for row in cur:
        rows_scanned += 1
        pairs = _extract_attestation_pairs(row["notes"])
        if pairs:
            rows_with_pairs += 1
            for form, year in pairs:
                candidate_inserts.append((row["toponym_id"], form, year, row["source_id"]))
        if rows_scanned % progress_every == 0:
            _emit_progress(rows_scanned)

    if apply and candidate_inserts:
        # INSERT OR IGNORE relies on the unique index added by
        # _create_toponym_attestation_unique_index. Without it, re-runs
        # would duplicate every row.
        result = db.conn.executemany(
            "INSERT OR IGNORE INTO toponym_attestation "
            "(toponym_id, form, date_year, source_doc) VALUES (?, ?, ?, ?)",
            candidate_inserts,
        )
        rows_written = result.rowcount
        db.commit()

    _emit_progress(rows_scanned, total=rows_scanned)

    return {
        "rows_scanned": rows_scanned,
        "rows_with_pairs": rows_with_pairs,
        "candidates": len(candidate_inserts),
        "rows_written": rows_written,
        "applied": apply,
    }
