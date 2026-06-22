"""wyrd-vm8t: backfill etymon ``pronunciation_ipa`` from the G2P.

The grapheme→IPA converter (:func:`registers.phonology.to_ipa`) is correct and
covers Old English, Old Norse, and Welsh — but nothing applied it, so all IPA
came from Wiktionary, which is uneven (ON ~1.8% covered, wyrd-7qq3) and renders
OE ``h`` as the velar fricative /x/ even word-initially (the wrong ``-ham`` →
/xɑːm/ the inspector surfaced).

This pass derives IPA deterministically from ``canonical_form``:

* FILL — set ``pronunciation_ipa`` for in-table etymons that have none.
* FIX-INITIAL-H — for OE forms that begin h + vowel (onset [h]) but whose
  existing Wiktionary IPA starts with /x/, replace it with the G2P value (now
  positional: onset [h], coda [x]).

Free + deterministic → runs in ``run_full_enrichment`` (re-derived on every
rebuild, like english-shaped / stratum; no jsonl needed). ``reader_pronunciation``
follows automatically from the IPA at export.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from wyrd.generators.kenning.lexicon.morpheme_surface import normalize_morpheme_surface
from wyrd.generators.kenning.registers.phonology import _VOWELS, to_ipa

# LLM transcriptions occasionally leak non-IPA artifacts (an ASCII capital like
# /troP/, a stray Greek λ in /pλym/). IPA uses lowercase ASCII + phonetic
# extensions, never ASCII A–Z or Greek letters, so reject those rows.
_IPA_ARTIFACT = re.compile("[A-Z\u0370-\u03ff]")

# Confidence tiers from the LLM pronunciation mining that we trust enough to
# ship. "low" rows stay recorded in the jsonl but are not applied (operator
# upgrades them on review).
_ACCEPT_CONFIDENCE = ("high", "medium")

# Etymon language → the G2P table to run. The lexicon's "celtic" place-name
# elements are Brittonic (rhyd, bedd, llan, tref), so the Welsh table fits them
# (bedd → /bɛð/, llan → /ɬan/); Goidelic (irish / gaelic) is too orthographically
# opaque for a clean table and is handled separately (LLM, lower-confidence).
_G2P_LANG: dict[str, str] = {
    "old-english": "old-english",
    "old-norse": "old-norse",
    "welsh": "welsh",
    "celtic": "welsh",
}

# Bundle json_field (underscore form name) → G2P table language, for the
# export-side surface-form fallback (wyrd-vm8t Loop 4). Many bundle forms are
# variants / reflexes that are not etymon canonical_forms and so carry no
# etymon IPA; for the table languages we can still derive one deterministically.
_JSON_FIELD_G2P: dict[str, str] = {
    "old_english": "old-english",
    "old_norse": "old-norse",
    "welsh": "welsh",
    "celtic": "welsh",
}


def surface_ipa(form: str, json_field: str) -> str | None:
    """Deterministic G2P IPA for a bundle surface form whose etymon carried
    none. Table languages only (``json_field`` is the bundle's underscore name,
    e.g. ``old_english``); returns None when the form isn't derivable."""
    g2p_lang = _JSON_FIELD_G2P.get(json_field)
    if g2p_lang is None:
        return None
    cf = _clean_form(form)
    if cf is None:
        return None
    ipa = to_ipa(cf, g2p_lang)
    return ipa if _is_real_ipa(ipa, cf) else None


def _clean_form(form: str | None) -> str | None:
    """The G2P-ready surface, or None when the form isn't a single clean word
    (reconstruction ``*`` marker, spaces, digits — skip those)."""
    f = (form or "").strip().lstrip("*").strip()
    if not f or " " in f or any(ch.isdigit() for ch in f):
        return None
    return f


def _is_real_ipa(derived: str, form: str) -> bool:
    """True when the G2P actually converted something (not a bare passthrough
    of unmapped graphemes)."""
    return derived.strip("/").lower() != form.lower()


def _initial_h_is_x(form: str, existing_ipa: str) -> bool:
    """The OE fix-target: form is onset-h (h + vowel) but Wiktionary rendered
    it with the velar fricative /x/."""
    return (
        len(form) > 1
        and form[0].lower() == "h"
        and form[1].lower() in _VOWELS
        and existing_ipa.lstrip("/").startswith("x")
    )


def collect_pronunciation(path: str | None) -> dict[tuple[str, str], str]:
    """Load the LLM pronunciation jsonl → ``{(language, form): ipa}`` for the
    rows we trust enough to ship (see :data:`_ACCEPT_CONFIDENCE`). The jsonl is
    the L2 record for the no-G2P-table languages (irish/gaelic/french/…);
    ``derive_pronunciation_ipa`` replays it deterministically at enrichment."""
    state: dict[tuple[str, str], str] = {}
    if not path or not os.path.exists(path):
        return state
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ipa = (rec.get("ipa") or "").strip()
            lang, form = rec.get("language"), rec.get("form")
            if (
                ipa
                and lang
                and form
                and rec.get("confidence") in _ACCEPT_CONFIDENCE
                and not _IPA_ARTIFACT.search(ipa)
            ):
                state[(lang, form)] = ipa  # last write wins (jsonl is append-log)
    return state


def derive_pronunciation_ipa(
    db, *, apply: bool = True, llm_state: dict[tuple[str, str], str] | None = None
) -> dict[str, Any]:
    """Fill + fix etymon ``pronunciation_ipa``: deterministic G2P for the
    table languages (OE/ON/welsh/celtic), then the LLM jsonl (``llm_state``,
    from :func:`collect_pronunciation`) for the no-table languages. Both only
    fill etymons that still lack IPA (G2P additionally fixes OE onset-h)."""
    conn = db.conn
    langs = tuple(_G2P_LANG)  # languages we can derive IPA for
    placeholders = ",".join("?" for _ in langs)
    rows = conn.execute(
        f"SELECT id, language, canonical_form, pronunciation_ipa "
        f"FROM etymon WHERE language IN ({placeholders})",
        langs,
    ).fetchall()

    summary: Counter = Counter()
    for eid, lang, form, existing in rows:
        cf = _clean_form(form)
        if cf is None:
            summary["skipped_unclean"] += 1
            continue
        derived = to_ipa(cf, _G2P_LANG[lang])
        if not _is_real_ipa(derived, cf):
            summary["skipped_passthrough"] += 1
            continue
        existing = (existing or "").strip()
        if not existing:
            summary["filled"] += 1
        elif lang == "old-english" and _initial_h_is_x(cf, existing):
            summary["fixed_initial_h"] += 1
        else:
            summary["kept_existing"] += 1
            continue
        if apply:
            conn.execute("UPDATE etymon SET pronunciation_ipa=? WHERE id=?", (derived, eid))

    # LLM jsonl replay — no-G2P-table languages, gaps only (never overrides).
    for (lang, form), ipa in (llm_state or {}).items():
        # De-dash the form (wyrd-aicu.8, D45) so a dashed L2 pronunciation key
        # resolves to the bare-stored etymon.
        form = normalize_morpheme_surface(form) or form
        for (eid,) in conn.execute(
            "SELECT id FROM etymon WHERE language=? AND canonical_form=? "
            "AND (pronunciation_ipa IS NULL OR pronunciation_ipa='')",
            (lang, form),
        ).fetchall():
            summary["filled_llm"] += 1
            if apply:
                conn.execute("UPDATE etymon SET pronunciation_ipa=? WHERE id=?", (ipa, eid))
    return dict(summary)


def format_pronunciation_run(counts: dict[str, Any]) -> str:
    """Render :func:`derive_pronunciation_ipa` output as a markdown block."""
    return (
        "### IPA backfill (wyrd-vm8t, G2P)\n"
        f"- Filled from G2P (was empty): {counts.get('filled', 0)}\n"
        f"- Filled from LLM jsonl (no-table langs): {counts.get('filled_llm', 0)}\n"
        f"- Fixed OE initial-h (/x/ → /h/): {counts.get('fixed_initial_h', 0)}\n"
        f"- Kept existing: {counts.get('kept_existing', 0)}; "
        f"skipped: {counts.get('skipped_unclean', 0) + counts.get('skipped_passthrough', 0)}"
    )
