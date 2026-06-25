"""LLM IPA mining for the no-G2P-table generation languages (wyrd-vm8t Loop 3).

Goidelic (irish/gaelic), Romance (old/norman French, latin), and Middle English
have no clean grapheme→phoneme table, so a fast local LLM provides a broad IPA
transcription for the generation-reachable etymons that still lack one. Output
is the LOWER-CONFIDENCE ``data/mining/_pronunciation.jsonl`` — replayed
deterministically at enrichment (see
:func:`pronunciation_backfill.collect_pronunciation`) and operator-reviewable.

This module holds the pure, network-free pieces (target selection, prompt
construction, response parsing) so they are unit-testable; the Ollama loop lives
in the ``mine-pronunciation-llm`` CLI.
"""

from __future__ import annotations

import json
import sqlite3

# Etymon language → the human label the LLM is primed with. These are exactly
# the generation-reachable languages with no entry in phonology.PHONOLOGY and no
# Brittonic-via-Welsh fallback (those go through the deterministic G2P).
LLM_LANGUAGES: dict[str, str] = {
    "irish": "Irish (Gaeilge)",
    "old-irish": "Old Irish",
    "middle-irish": "Middle Irish",
    "scottish-gaelic": "Scottish Gaelic",
    "old-french": "Old French",
    "norman-french": "Norman French",
    "middle-english": "Middle English",
    "latin": "Latin",
    "breton": "Breton",
}

METHOD = "llm-ipa-v1"


def select_targets(db_path: str, languages: tuple[str, ...]) -> list[tuple[str, str]]:
    """Generation-reachable etymons (those with reflex links) in ``languages``
    that still lack ``pronunciation_ipa`` and have a clean single-word form.
    Read-only; returns ``[(language, canonical_form), …]``.

    Merged OCR-loser etymons (``merged_into_id`` set) are excluded: a reflex link
    can still point at a tombstone, but generation reroutes through its merge winner
    (``_family.py``), so mining an IPA onto the tombstone is unreachable (mirrors
    ``tag_mining.select_targets``)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" for _ in languages)
        return conn.execute(
            f"SELECT DISTINCT e.language, e.canonical_form FROM etymon e "
            f"JOIN reflex_etymon re ON re.etymon_id = e.id "
            f"WHERE e.language IN ({placeholders}) "
            f"AND (e.pronunciation_ipa IS NULL OR e.pronunciation_ipa = '') "
            f"AND e.merged_into_id IS NULL "
            f"AND e.canonical_form NOT LIKE '%*%' AND e.canonical_form NOT LIKE '% %' "
            f"ORDER BY e.language, e.canonical_form",
            languages,
        ).fetchall()
    finally:
        conn.close()


def build_messages(language: str, form: str) -> list[dict[str, str]]:
    """The chat messages priming a broad IPA transcription for one element."""
    label = LLM_LANGUAGES.get(language, language)
    system = (
        f"You are an expert in {label} historical phonology. Give a BROAD IPA "
        f"transcription for the place-name element, reflecting its standard "
        f"scholarly pronunciation. Reply strict JSON: "
        '{"ipa":"/.../","confidence":"high|medium|low"}. Use low confidence if '
        "the form is ambiguous or you are unsure."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Element: {form!r}"},
    ]


def parse_response(content: str) -> tuple[str, str] | None:
    """Extract ``(ipa, confidence)`` from a model reply, or None if unusable.
    IPA must be slash-delimited; confidence defaults to ``low``."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    ipa = (data.get("ipa") or "").strip()
    if not ipa.startswith("/"):
        return None
    confidence = str(data.get("confidence") or "low").lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    return ipa, confidence


def record(language: str, form: str, parsed: tuple[str, str] | None, model: str) -> dict:
    """Build the jsonl record for one mined element (with or without IPA)."""
    rec = {
        "_type": "pronunciation",
        "language": language,
        "form": form,
        "ref": f"{language}:{form}",
        "model": model,
        "method": METHOD,
    }
    if parsed is not None:
        rec["ipa"], rec["confidence"] = parsed
    else:
        rec["skipped"] = "no-ipa"
    return rec
