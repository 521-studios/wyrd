"""De-dash a stored morpheme surface — ``etymon.canonical_form`` (wyrd-aicu.8, D45).

A morpheme's stored identity is its BARE surface; position is an explicit axis
(toponym-breakdown-derived), never dash-encoded (D45, holistic scope). The
``etymon.canonical_form`` store historically carried affix-position decoration
as leading/trailing dashes — ``-ach`` (bound suffix), ``-ar-`` (infix), ``ton-``
(prefix) — which merely re-encode position as identity noise and fork one
morpheme into duplicate nodes (``celtic:-ach`` vs ``celtic:ach``).

``normalize_morpheme_surface`` strips ONLY boundary (leading/trailing) dashes,
the same TRIM rationale as ``reflex.surface_form`` (wyrd-aicu.3, migration
0017): position markers are always at a boundary, so trimming removes them while
preserving any *legitimate interior* hyphen in a form — ``al-Quadim``,
``'s-Hertogenbosch``, the PIE proto-segmentation hyphens in ``*(H)réh₁-ti-s`` —
which a blanket ``replace('-', '')`` would corrupt.

The leading reconstruction sigil ``*`` (``*tūn``, ``*-at-``) is PRESERVED: it is
identity, not decoration, and its own folding (``*`` → a ``reconstructed``
boolean, with the recon/attested collision handling) is deferred to wyrd-qoy8.
So a boundary dash *between* the sigil and the stem is still trimmed
(``*-at-`` → ``*at``) while the sigil survives.

A form that is empty after trimming (a bare ``-``, ``--``, or ``*-``) is junk —
``None`` is returned so the ingest boundary can DROP the record. The de-dash
itself (non-junk forms) is a pure, idempotent transform and belongs at the
central write choke (``LexiconDB.upsert_etymon``); the drop is a record-level
decision and belongs at the ingest call sites (cf. ``controlled_vocab`` — the
junk guard sits at the ingest boundary, not in the upsert primitive).
"""

from __future__ import annotations


def normalize_morpheme_surface(raw: str | None) -> str | None:
    """Return the BARE morpheme surface for ``raw``, or ``None`` if it is junk.

    Strips surrounding whitespace and boundary (leading/trailing) dashes,
    preserving the leading ``*`` reconstruction sigil and any interior hyphen.
    Returns ``None`` when nothing but a sigil/dashes remains (strip-to-empty
    junk) so the caller can drop the record.

    Examples::

        "-ach"      -> "ach"          # bound-suffix marker trimmed
        "ton-"      -> "ton"          # prefix marker trimmed
        "-ar-"      -> "ar"           # infix markers trimmed
        "al-Quadim" -> "al-Quadim"    # interior hyphen preserved
        "*tūn"      -> "*tūn"         # reconstruction sigil preserved
        "*-at-"     -> "*at"          # sigil kept, boundary dash trimmed
        "-"         -> None           # junk → drop
        "*-"        -> None           # junk → drop
        "-*"        -> None           # dash + bare sigil → junk
    """
    if not isinstance(raw, str):
        return None
    # Strip boundary spaces AND dashes FIRST, so a leading dash before the sigil
    # ("-*ach-") or a nested-space form ("-  ach  -") still resolves: the sigil
    # is then at position 0 and the stem carries no stray boundary chars.
    s = raw.strip(" -")
    reconstructed = s.startswith("*")
    if reconstructed:
        # Re-strip the stem after peeling the sigil ("* -ach" / "*-ach-").
        s = s[1:].strip(" -")
    # Junk if nothing survives, or the stem is only sigils/dashes ("-*", "**") —
    # a bare sigil is not a morpheme. (`replace` is the junk TEST only; the
    # returned `s` keeps its interior hyphens.)
    if not s or not s.replace("*", "").replace("-", ""):
        return None
    return f"*{s}" if reconstructed else s
