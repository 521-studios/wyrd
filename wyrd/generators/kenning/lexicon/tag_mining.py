"""wyrd-xz3g — LLM tag classification: gloss → controlled tag vocabulary.

~31% of generation-reachable morphemes carry a gloss but NO semantic tag, so
they contribute zero signal to the generator's cohesion / mood / theme
filtering (wyrd-3iln closed: the cohesion knob is fine; the gap is tag
COVERAGE, not the algorithm). This module classifies each untagged gloss into
the EXISTING tag vocabulary plus the one operator-approved new tag, ``kinship``
(2026-06-12: family relations have no existing home). A mandatory "none"
outcome keeps genuinely-abstract glosses ("very", "to judge", a pronoun) blank
— operator constraint: NO bad matches, NO tag sprawl.

LLM-only by directive: deterministic gloss→tag rules fail because glosses are
noisy multi-sense concatenations with distractor words ("personal name element
spear, arrow, dart" — 'element' is a distractor). A local LLM (gemma4:26b on
the macbook Ollama) reads the gloss as a whole.

Pure, network-free pieces live here (vocab, prompt, parse, target selection) so
they are unit-testable; the Ollama loop lives in the ``mine-tags-llm`` CLI.
Output is the lower-confidence L2 file ``data/mining/_tags.jsonl``, replayed
deterministically by the ``apply-tag-additions`` enrichment pass (INSERT OR
IGNORE on ``etymon_tag``) so a from-scratch rebuild never re-pays for the LLM.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

# The controlled vocabulary the LLM may use (45 tags). It is 44 of the repo's
# 45 existing etymon_tag values — dropping the typo-variant 'topology'
# (canonical is 'topography') — plus the one operator-approved new tag
# 'kinship' (wyrd-xz3g, 2026-06-12). The other ls13 candidates fold into
# existing tags by operator decision: body→anatomy, conflict→military,
# food/drink→food. No sprawl: the model picks FROM this set or returns nothing.
TAG_VOCAB: tuple[str, ...] = (
    "action",
    "agriculture",
    "anatomy",
    "animal",
    "architecture",
    "bird",
    "color",
    "death",
    "descriptive",
    "direction",
    "economic",
    "element",
    "fantasy",
    "female name",
    "food",
    "gender",
    "geography",
    "geology",
    "grain",
    "insect",
    "item",
    "kinship",
    "male name",
    "mammal",
    "manorial",
    "material",
    "measurement",
    "military",
    "monster",
    "norman",
    "number",
    "personal-name",
    "plant",
    "possession",
    "profession",
    "religious",
    "saint",
    "size",
    "social",
    "temperature",
    "topography",
    "tree",
    "undead",
    "water",
    "weather",
)
_VOCAB_SET = frozenset(TAG_VOCAB)

# Specific tag → parents it always co-occurs with (matches the data convention:
# tree⊂plant, faunal subtypes ⊂ animal). Applied to the LLM's picks too.
TAG_PARENTS: dict[str, tuple[str, ...]] = {
    "tree": ("plant",),
    "bird": ("animal",),
    "mammal": ("animal",),
    "insect": ("animal",),
}

# Suspected OCR/mapping typo-variants the model might echo → canonical tag.
# Existing 'topology' rows (41) are left untouched; a separate normalization
# pass is out of scope for the backfill.
_TAG_ALIASES: dict[str, str] = {"topology": "topography", "body": "anatomy"}

METHOD = "llm-tags-v1"

# The synthetic L2 source ref. Every _tags.jsonl carries one canonical
# `source` row (its first line) so rebuild-from-jsonl's per-file
# `_extract_source_id` contract is met — mirrors _reflexes.jsonl /
# _merge_audit.jsonl / _fantasy_morphemes.jsonl. Without it, the rebuild's
# glob over data/mining/*.jsonl raises BuildError on this file.
SOURCE_REF = "tag-backfill"


def source_row() -> dict:
    """The mandatory canonical `source` row written as the first line of every
    `_tags.jsonl`. The tag rows themselves replay INERT (no build helper
    consumes the `tags` LIST_TYPE accumulation; `collect_tags` reads the file
    directly for `apply_tag_additions`)."""
    return {
        "_type": "source",
        "ref": SOURCE_REF,
        "title": "LLM tag backfill (wyrd-xz3g)",
        "notes": (
            "Synthetic source for the etymon_tag L2 round-trip (gemma4:26b "
            "gloss → controlled-vocab tags). Tag rows replay inert; "
            "collect_tags reads the file directly for apply_tag_additions."
        ),
    }


# wyrd-jrwc: a breakdown morpheme is "admitted" to the bundle (wyrd-oth3) when it
# appears as an element in scholarly toponym_etymology breakdowns. The realistic
# admit cohort is the elements used in >= this many DISTINCT toponyms (the epic's
# gap-profile threshold; the single-occurrence tail is wyrd-myv4's junk-pruning
# concern, not the tag cohort's).
_BREAKDOWN_ADMIT_MIN_TOPONYMS = 2


def select_targets(db_path: str) -> list[tuple[str, str, str]]:
    """Generation-reachable etymons that have a gloss but no tags. Read-only;
    returns ``[(language, canonical_form, glosses), …]``.

    Reachability matches what the bundle actually ships, which is TWO channels:
    (1) reflex_etymon — a morpheme that reaches a generation surface via a reflex;
    (2) wyrd-jrwc/oth3 — an admitted breakdown morpheme: an element appearing in
    the scholarly breakdowns of >= ``_BREAKDOWN_ADMIT_MIN_TOPONYMS`` DISTINCT
    toponyms (counted by toponym_id, so multiple breakdowns of one toponym count
    once). Channel (2) is the largely-reflex-less admit cohort the reflex-only
    gate used to miss — untagged, those morphemes are matchable for decomposition
    but never SELECTED in generation (the >=1-tag gate skips them). Dead etymons
    (neither channel) are still excluded, so we don't pay to tag them — as are
    merged OCR-loser etymons (``merged_into_id`` set): a tombstone can still carry a
    reflex/breakdown edge, but the bundle reroutes generation through its merge
    winner (``_family.py``), so a tag minted onto the tombstone is unreachable, and
    the empirical-priors extract that joins ``etymon_tag`` on a breakdown element
    would read it as a live tag (wyrd-ckro). Every other reachability reader filters
    ``merged_into_id IS NULL``; this one must too."""
    # Path.as_uri() handles spaces / '?' / '#' / Windows backslashes robustly
    # (matches runtime_db.py's read-only open), unlike bare f-string interp.
    uri = f"{Path(db_path).absolute().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        # Two query shapes matter here:
        # - Reachability is a correlated subquery (EXISTS on reflex_etymon; a
        #   DISTINCT-toponym count on the breakdown channel), NOT a JOIN: joining
        #   the one-to-many reflex_etymon AND the one-to-many etymon_gloss would
        #   Cartesian-product and GROUP_CONCAT would duplicate each gloss once
        #   per reflex ('spear || spear'), inflating the prompt + wasting tokens.
        # - Glosses are ORDER BY'd inside a subquery before GROUP_CONCAT so the
        #   concatenation is DETERMINISTIC (SQLite's GROUP_CONCAT order is
        #   otherwise unspecified). The mine is meant to be reproducible — a
        #   stable gloss order ⇒ a stable prompt ⇒ a stable _tags.jsonl.
        return conn.execute(
            "SELECT language, canonical_form, GROUP_CONCAT(gloss, ' || ') "
            "FROM ("
            "  SELECT e.id AS id, e.language AS language, "
            "         e.canonical_form AS canonical_form, g.gloss AS gloss "
            "  FROM etymon e "
            "  JOIN etymon_gloss g ON g.etymon_id = e.id "
            "  WHERE ("
            "         EXISTS (SELECT 1 FROM reflex_etymon re WHERE re.etymon_id = e.id) "
            "      OR ("
            "           SELECT count(DISTINCT te.toponym_id) "
            "           FROM toponym_etymology_element el "
            "           JOIN toponym_etymology te ON te.id = el.toponym_etymology_id "
            "           WHERE el.etymon_id = e.id "
            f"         ) >= {_BREAKDOWN_ADMIT_MIN_TOPONYMS} "
            "  ) "
            "    AND NOT EXISTS (SELECT 1 FROM etymon_tag t WHERE t.etymon_id = e.id) "
            "    AND e.merged_into_id IS NULL "
            "  ORDER BY e.id, g.gloss"
            ") GROUP BY id ORDER BY language, canonical_form"  # noqa: S608 — interpolated int constant
        ).fetchall()
    finally:
        conn.close()


def build_messages(gloss: str) -> tuple[str, str]:
    """(system, user) prompt classifying one gloss into the controlled vocab.

    Returns a tuple matching ``OllamaClient.chat_json(system, user, schema)``.
    """
    vocab = ", ".join(TAG_VOCAB)
    system = (
        "You classify a historical place-name element by its meaning (gloss) "
        "into a FIXED set of semantic tags. Use ONLY these tags:\n"
        f"{vocab}\n\n"
        "Rules:\n"
        "1. Pick EVERY tag the gloss clearly denotes; multiple tags are fine. "
        "Glosses are noisy concatenations of several senses — read the whole "
        "thing and tag each real sense, but ignore filler words like "
        "'element', 'place', 'name of', 'used in'.\n"
        "2. PRECISION OVER RECALL. Tag ONLY what the element actually MEANS, "
        "not things merely associated with it. A martin is a bird, NOT an "
        "insect. A drinking-horn is an 'item' (vessel), NOT 'anatomy'. The word "
        "'mother' is 'kinship', NOT a 'female name'. When unsure, leave a tag "
        "OFF.\n"
        "3. If the gloss is abstract, grammatical, relational, or has NO "
        "concrete semantic category (e.g. 'very', 'good', 'other', 'to judge', "
        "a number word, a pronoun, 'a part/share/section of something'), return "
        "an EMPTY list. NEVER force a tag when none fits.\n"
        "4. Hierarchy: a tree is also 'plant'; a bird/mammal/insect is also "
        "'animal'.\n"
        "5. Tag mapping you MUST follow:\n"
        "   - landscape features (hill, mountain, valley, ridge, slope, moor, "
        "down, cliff) → 'topography' (this is the most common tag — do not "
        "forget it for hills/mountains/valleys)\n"
        "   - body parts (hair, horn-as-body-part, skin, bone, head, hand) → "
        "'anatomy'\n"
        "   - weapons / war / battle / army (spear, sword, shield) → 'military'\n"
        "   - food and drink (bread, ale, mead, meat) → 'food'\n"
        "   - family relations (father, mother, son, brother, kin) → 'kinship'\n"
        "   - rank / office (king, servant, noble, lord) → 'social'\n"
        "   - ghosts / spectres / wraiths → 'undead'\n"
        '6. Reply STRICT JSON only: {"tags": ["..."], '
        '"confidence": "high|medium|low"}. No prose, no markdown.'
    )
    user = f"Gloss: {gloss!r}"
    return system, user


def parse_response(content: str | dict | None) -> tuple[frozenset[str], str] | None:
    """Extract ``(tags, confidence)`` from a reply. Tags are alias-normalized,
    filtered to the vocab, and expanded with parents; an empty set is a valid
    "none" outcome. Returns None only when the reply is unparseable."""
    if isinstance(content, dict):
        data = content
    else:
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(data, dict):
        # json.loads can yield a list / primitive (e.g. the model replied
        # `["military"]`); `.get` would AttributeError-crash. Treat as
        # unparseable.
        return None
    raw = data.get("tags")
    if not isinstance(raw, list):
        return None
    norm = (
        _TAG_ALIASES.get(t.strip().lower(), t.strip().lower()) for t in raw if isinstance(t, str)
    )
    tags = set(norm) & _VOCAB_SET
    for t in list(tags):
        # Parents are intersected with the vocab too, so a removed / typo'd
        # TAG_PARENTS entry can never inject an out-of-vocab tag (the set is
        # otherwise only ever filtered through _VOCAB_SET above).
        tags.update(set(TAG_PARENTS.get(t, ())) & _VOCAB_SET)
    confidence = str(data.get("confidence") or "low").lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    return frozenset(tags), confidence


def record(language: str, form: str, parsed: tuple[frozenset[str], str] | None, model: str) -> dict:
    """The L2 jsonl record for one classified element (tags may be empty =
    'none' outcome, which is a real recorded decision, not an error)."""
    rec = {
        "_type": "tags",
        "ref": f"{language}:{form}",
        "language": language,
        "form": form,
        "model": model,
        "method": METHOD,
    }
    if parsed is not None:
        tags, confidence = parsed
        rec["tags"] = sorted(tags)
        rec["confidence"] = confidence
    else:
        rec["error"] = "unparseable"
    return rec


def existing_refs(path: str) -> set[str]:
    """Refs already recorded in ``_tags.jsonl`` — the skip-set that makes
    ``mine-tags-llm`` resumable without re-paying for classified glosses."""
    refs: set[str] = set()
    if not path or not os.path.exists(path):
        return refs
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a truncated trailing line from a crash-interrupted
                # flush — skip it rather than abort the resume skip-set.
                continue
            # Only `tags` rows are resume keys; the canonical `source` row (and
            # any future non-tags row) is not an etymon ref and is excluded.
            if isinstance(row, dict) and row.get("_type") == "tags" and row.get("ref"):
                refs.add(row["ref"])
    return refs


def collect_tags(path: str) -> dict[str, dict[str, list[str]]]:
    """Load accepted tag assignments from the L2 ``_tags.jsonl`` record into a
    replay state ``{etymon_ref: {"tags": [...]}}`` for
    :func:`enrichment.apply_tag_additions`.

    Rows with an empty tag list (the 'none' outcome) or an ``error`` are
    SKIPPED — they carry no tag to add but stay in the file as the recorded
    decision (so a re-mine with ``--skip-resolved`` doesn't re-pay for them).
    Last-write-wins per ref. Mirrors
    :func:`element_gloss_backfill.collect_element_glosses` (direct read, no
    replay registry); ``run_full_enrichment`` replays it for FREE on a
    from-scratch rebuild, so the paid LLM pass never re-runs (D21 / REBUILD.md).
    """
    state: dict[str, dict[str, list[str]]] = {}
    if not path or not os.path.exists(path):
        return state
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # A corrupt/truncated line must not abort the whole enrichment
                # replay — skip it (the file is append-only, flushed per row).
                continue
            if not isinstance(r, dict):
                continue
            ref, tags = r.get("ref"), r.get("tags")
            # Skips the canonical `source` row (no `tags`) and 'none'/error
            # rows; guards against a non-list `tags` value.
            if not ref or not isinstance(tags, list) or not tags:
                continue
            state[ref] = {"tags": sorted({t for t in tags if isinstance(t, str)})}
    return state
