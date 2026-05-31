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

from collections import Counter
from typing import Any

from wyrd.generators.kenning.registers.phonology import _VOWELS, PHONOLOGY, to_ipa


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


def derive_pronunciation_ipa(db, *, apply: bool = True) -> dict[str, Any]:
    """Fill + fix etymon ``pronunciation_ipa`` from the G2P. Returns a summary."""
    conn = db.conn
    langs = tuple(PHONOLOGY)  # languages with a real phonology table
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
        derived = to_ipa(cf, lang)
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
    return dict(summary)


def format_pronunciation_run(counts: dict[str, Any]) -> str:
    """Render :func:`derive_pronunciation_ipa` output as a markdown block."""
    return (
        "### IPA backfill (wyrd-vm8t, G2P)\n"
        f"- Filled (was empty): {counts.get('filled', 0)}\n"
        f"- Fixed OE initial-h (/x/ → /h/): {counts.get('fixed_initial_h', 0)}\n"
        f"- Kept existing: {counts.get('kept_existing', 0)}; "
        f"skipped: {counts.get('skipped_unclean', 0) + counts.get('skipped_passthrough', 0)}"
    )
