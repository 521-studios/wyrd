"""pfsrd2-data → wyrd-ami JSONL extractor (wyrd-ld5x).

The wyrd-ami fantasy-name pipeline takes JSONL input where each line
is `{"name": ..., "description": ...}`. This module walks the
pfsrd2-data monster JSON corpus (`~/521Studios/pfsrd2-data/monsters/`)
and produces that JSONL — one entry per distinct creature morpheme.

Extraction strategy (verified against creature.schema.1.4.json and
spot-checked across bugbear_thug / harpy / dragons / trolls):

- **Input_name preference**: family root over variant name. A
  "Bugbear Thug" entry has `stat_block.creature_type.family.name =
  "Bugbear"`; we want to feed the LLM the morpheme "Bugbear", not the
  multi-word variant name. Compound family names like "Dragon, Red"
  are split on `,` to get the base ("Dragon").

- **Single-word top-level fallback**: a creature like Harpy has no
  family (`family is None`). Its `name = "Harpy"` is itself the
  morpheme.

- **Multi-word no-family creatures are skipped**: per the user
  2026-05-06, "Ancient Black Dragon" with no family field is
  problematic — there's no clean morpheme to extract. We bail rather
  than guess.

- **Description = family lore + monster sections** (recursive). The
  family.text field is often the richest source (Troll's per-variant
  sections[] is empty; family.text has the full lore). We concatenate
  family.text + all family.sections.text (recursive walk) + all
  monster.sections.text (recursive walk).

- **Dedupe by lowercased input_name** so "Bugbear" doesn't get
  emitted three times across thug/tormentor/lurker variants. First
  occurrence wins; later monsters with the same family contribute
  nothing.

The extractor doesn't HTTP-fetch — it's a pure JSON walker. Run it
out-of-band, pipe the output JSONL into `wyrd kenning lexicon
mine-fantasy-name --batch`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _collect_section_texts(sections: list[dict[str, Any]] | None) -> list[str]:
    """Walk the recursive sections tree, collect every node's `text`
    field in document order. Each node may have its own children
    under `sections`."""
    out: list[str] = []
    for s in sections or []:
        text = s.get("text")
        if text:
            out.append(text)
        out.extend(_collect_section_texts(s.get("sections")))
    return out


def extract_input_name(monster: dict[str, Any]) -> str | None:
    """Pick the morpheme-of-interest for the wyrd-ami pipeline.

    Returns None when the monster doesn't yield a clean single-WORD
    morpheme — caller should skip those rather than feed ambiguous
    multi-word names to the LLM (per the wyrd-ami design directive
    that "Ancient Black Dragon" is problematic input).

    Cases handled:
    - family.name with subtype suffix ('Dragon, Red') → split on comma,
      take the base ('Dragon'). Compound base names ('Hell Hound')
      are still rejected as multi-word.
    - family is None and top-level name is single-word → use the name.
    - everything else → None (skip).
    """
    family = monster.get("stat_block", {}).get("creature_type", {}).get("family")
    if family and family.get("name"):
        # "Dragon, Red" → "Dragon"; "Bugbear" stays "Bugbear".
        candidate = family["name"].split(",")[0].strip()
    else:
        candidate = (monster.get("name") or "").strip()
    if not candidate or " " in candidate:
        return None
    return candidate


def extract_description(monster: dict[str, Any]) -> str:
    """Build the LLM-input description for a monster.

    Concatenates (in order):
      1. `family.text` if present (high-value lore).
      2. All texts under `family.sections` (recursive).
      3. All texts under `monster.sections` (recursive — variant
         flavor for the specific creature).

    Joined with blank-line separators so the LLM gets paragraph
    structure. Returns empty string if no descriptive text was found.
    """
    parts: list[str] = []
    family = monster.get("stat_block", {}).get("creature_type", {}).get("family")
    if family:
        family_text = family.get("text")
        if family_text:
            parts.append(family_text)
        parts.extend(_collect_section_texts(family.get("sections")))
    parts.extend(_collect_section_texts(monster.get("sections")))
    return "\n\n".join(p for p in parts if p)


def extract_monster(
    monster: dict[str, Any],
) -> dict[str, str] | None:
    """Extract one monster JSON into a wyrd-ami input record. Returns
    None when the monster is unusable (multi-word name, no family).
    """
    name = extract_input_name(monster)
    if name is None:
        return None
    description = extract_description(monster)
    return {"name": name, "description": description}


def iter_monster_files(corpus_root: Path) -> Iterator[Path]:
    """Yield each monster JSON file under `corpus_root/monsters/` in
    sorted order. Sorted so the dedupe (first-occurrence-wins) is
    deterministic across runs."""
    monsters_dir = corpus_root / "monsters"
    if not monsters_dir.is_dir():
        return
    yield from sorted(monsters_dir.rglob("*.json"))


def extract_corpus(
    corpus_root: Path,
    *,
    limit: int | None = None,
) -> Iterator[dict[str, str]]:
    """Walk pfsrd2-data, dedupe by lowercased input_name, yield
    {name, description} dicts ready for JSONL output. First occurrence
    of each name wins — sibling variants of the same family don't
    produce duplicate entries."""
    seen: set[str] = set()
    emitted = 0
    for path in iter_monster_files(corpus_root):
        try:
            monster = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        record = extract_monster(monster)
        if record is None:
            continue
        key = record["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        yield record
        emitted += 1
        if limit is not None and emitted >= limit:
            return
