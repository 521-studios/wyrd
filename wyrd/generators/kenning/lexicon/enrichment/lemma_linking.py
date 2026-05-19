"""Lemma linkage — inflection-stripping + Celtic mutations (D8).

Two enrichment passes that share the same goal: each etymon row that
LOOKS like an inflected/mutated form of an existing canonical etymon
gets its ``lemma_id`` / ``inflection`` columns set so the consensus
view rolls citations up to the lemma instead of fragmenting them.

* **Suffix stripping** (``INFLECTION_RULES``) — per-language
  conservative suffix lists. ``derive_lemma_candidates`` enumerates
  every candidate stem; ``derive_lemma_candidate`` picks the first
  that survives the linker's lookup.
* **Celtic initial mutations** (``MUTATION_RULES``) — Irish eclipsis
  + lenition + Scottish-Gaelic lenition, applied via
  ``derive_mutation_lemma_candidate``.

``link_lemmas`` is the orchestrator: walks every etymon, tries both
strategies, writes back ``lemma_id`` / ``inflection`` when a lemma
is found. Idempotent and reversible via
``clear_enrichment(stage="lemmas")``.

The rule tables encode hard-won precision choices documented inline
(see the long comments next to INFLECTION_RULES and MUTATION_RULES);
adding a new rule needs the same precision-over-recall scrutiny.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.db import LexiconDB

# Per-language inflection-stripping rules. Each entry is a (suffix, label)
# pair: the suffix to strip from the inflected form, and the label that
# describes the inflection. Order matters — longer/more-specific suffixes
# are tried first so we don't match a substring of a longer suffix.
#
# We keep these conservative: only widely-attested inflectional suffixes,
# and we still require the stem to match an existing etymon (we don't
# fabricate lemmas in this pass — that's a follow-up).
INFLECTION_RULES: dict[str, list[tuple[str, str]]] = {
    "old-english": [
        ("ena", "genitive_pl"),
        ("ana", "genitive_pl"),
        ("um", "dative_pl"),
        ("an", "weak_oblique"),  # gen sg, dat sg, acc sg, nom/acc pl of weak nouns
        ("as", "plural_strong"),
        ("es", "genitive_strong"),
        ("ar", "plural"),
        # NOTE: "-u" plural-neuter is DELIBERATELY OMITTED. OE has many
        # u-stem nouns where -u is the lemma (wudu "wood", sunu "son", lagu
        # "law", duru "door", felu "many"). The strip rule produced too many
        # false-positive linkages. We accept losing a small number of real
        # neuter plurals (scipu "ships") rather than corrupt u-stem lemmas.
        ("e", "dative_or_pl"),
    ],
    "old-norse": [
        ("ar", "genitive_or_pl"),
        ("ir", "plural"),
        ("um", "dative_pl"),
        ("ur", "nominative"),
        ("r", "nominative"),
        ("i", "dative_or_weak"),
    ],
    # wyrd-rc0b Phase 1: Old French / Middle English / Norman French.
    # Same conservative-precision approach as the Welsh entry above.
    # All three languages use case-marking + plural -s suffixes per
    # Old French's two-case system (nom/oblique) inherited into ME via
    # Anglo-Norman influence. Only the LONGER form (-es feminine plural
    # / genitive plural) is registered here; the bare -s is too generic
    # — many OF/ME lemmas end in -s naturally (OF 'pres' 'near', 'fois'
    # 'time'; ME 'is' / 'thus' / 'bras'), and the strip would corrupt
    # those families.
    "old-french": [
        # 336 candidates; OF feminine plural / oblique case.
        ("es", "feminine_plural"),
    ],
    "middle-english": [
        # 5863 candidates; weak plural (oxen, children, brethren).
        # Listed before -es so the longer ending wins on tied stems.
        ("en", "plural"),
        # 2574 candidates; ME plural / genitive singular (cotes / kynges).
        ("es", "genitive_or_plural"),
    ],
    "norman-french": [
        # 5 candidates (NF corpus is small — 70 etymons total). Same
        # shape as Old French.
        ("es", "feminine_plural"),
    ],
    # Modern English — conservative set chosen to minimize false-
    # positive linkages. The wyrd-r6bk precision-over-recall principle
    # applies x10 to ModE because the corpus is 1.4M etymons (most via
    # English wiktextract slice), and many natural English words end in
    # the same surfaces as inflectional suffixes (e.g. 'bed' → 'be'+'d'
    # would mislink to 'be'; 'string' → 'str'+'ing' wrong).
    #
    # Rules to consider but DEFERRED:
    # - "-s" plural / "-es" plural: too generic. Words like 'is', 'this',
    #   'thus', 'pass', 'class' end in -s naturally; linking 'pass' to
    #   'pas' or 'class' to 'clas' would corrupt. Defer to a richer
    #   rule with frequency-of-stem context.
    # - "-er" comparative: conflicts with agent-noun -er (baker, walker
    #   would link to verbs). Defer.
    # - "-ly" adverb: conflicts with adjective-final -ly (ugly, lovely,
    #   silly are NOT 'ug'/'love'/'sill' + ly). Too risky.
    #
    # The v1 subset stays narrow. Inflection coverage for ModE relies
    # on Wiktionary's form-tables (mined via mine-wiktextract-forms
    # which writes etymon_text_match rows — not lemma_id links).
    "modern-english": [
        # Past tense / past participle. Silent-e collision (hoped →
        # hope, not hop) handled via the optional 3rd-tuple element:
        # try (stem + 'e') first, fall back to plain stem. So 'hoped'
        # → ['hope', 'hop']: 'hope' wins if both exist, 'hop' wins if
        # only 'hop' exists.
        ("ed", "past", "e"),
        # Gerund / present participle. Same silent-e handling:
        # 'hoping' → ['hope', 'hop']; 'walking' → ['walke', 'walk']
        # (where 'walke' won't exist so 'walk' wins).
        ("ing", "gerund", "e"),
    ],
    # wyrd-r6bk Phase 1: Welsh suffix-strippable subset, conservative.
    # Initial-mutation morphology (lenition / nasal / aspirate) is
    # NOT suffix-strippable and stays unhandled — wyrd-jott tracks
    # the Goidelic+Brythonic-shared problem.
    #
    # Rules ordered specific-before-general. Several plausibly-Welsh
    # suffixes are DELIBERATELY OMITTED for false-positive risk:
    # - ``-af``: the superlative reading conflicts with common nouns
    #   ending in -af (gaeaf 'winter', gwarthaf 'summit') that aren't
    #   superlatives. Linking 'gaeaf' to 'gae' would corrupt 'gae's
    #   family rollup. Defer to a richer rule with positional / part-
    #   of-speech context.
    # - ``-au``: too generic — many Welsh place-name words coincidentally
    #   end in -au (Llysau / Tegau / Trau as proper nouns). The 1301
    #   candidates would mostly be false positives. Specific plural
    #   variants (-iau, -oedd, -ion) capture the safe subset.
    # - ``-ydd``: derivational nominal suffix as often as plural marker
    #   (gweithydd 'worker'). Too risky for v1.
    # - ``-aint``: rare (43 candidates) and mostly archaic.
    #
    # The Phase 1 subset prioritises precision over recall.
    "welsh": [
        ("iau", "plural"),  # 327 candidates; specific plural subtype
        ("oedd", "plural"),  # 170 candidates; e.g. blynydd + -oedd
        ("ion", "plural"),  # 402 candidates; e.g. dyn + -ion
        ("ach", "comparative"),  # 262 candidates; consistent adj. use
    ],
}


MUTATION_RULES: dict[str, list[tuple[str, str, str]]] = {
    "irish": [
        # Eclipsis (urú) — unambiguous, always grammatical mutation.
        ("bhf", "f", "eclipsis"),  # longest first; bhf- before bh- below
        ("mb", "b", "eclipsis"),
        ("gc", "c", "eclipsis"),
        ("nd", "d", "eclipsis"),
        ("ng", "g", "eclipsis"),
        ("bp", "p", "eclipsis"),
        ("dt", "t", "eclipsis"),
        # Lenition (séimhiú) — safe subset (skip ch-/th-).
        ("mh", "m", "lenition"),
        ("bh", "b", "lenition"),
        ("dh", "d", "lenition"),
        ("gh", "g", "lenition"),
        ("ph", "p", "lenition"),
        ("fh", "f", "lenition"),
        ("sh", "s", "lenition"),
    ],
    "old-irish": [
        ("bhf", "f", "eclipsis"),
        ("mb", "b", "eclipsis"),
        ("gc", "c", "eclipsis"),
        ("nd", "d", "eclipsis"),
        ("ng", "g", "eclipsis"),
        ("bp", "p", "eclipsis"),
        ("dt", "t", "eclipsis"),
        ("mh", "m", "lenition"),
        ("bh", "b", "lenition"),
        ("dh", "d", "lenition"),
        ("gh", "g", "lenition"),
        ("ph", "p", "lenition"),
        ("fh", "f", "lenition"),
        ("sh", "s", "lenition"),
    ],
    "middle-irish": [
        ("bhf", "f", "eclipsis"),
        ("mb", "b", "eclipsis"),
        ("gc", "c", "eclipsis"),
        ("nd", "d", "eclipsis"),
        ("ng", "g", "eclipsis"),
        ("bp", "p", "eclipsis"),
        ("dt", "t", "eclipsis"),
        ("mh", "m", "lenition"),
        ("bh", "b", "lenition"),
        ("dh", "d", "lenition"),
        ("gh", "g", "lenition"),
        ("ph", "p", "lenition"),
        ("fh", "f", "lenition"),
        ("sh", "s", "lenition"),
    ],
    "scottish-gaelic": [
        ("bhf", "f", "eclipsis"),
        ("mb", "b", "eclipsis"),
        ("gc", "c", "eclipsis"),
        ("nd", "d", "eclipsis"),
        ("ng", "g", "eclipsis"),
        ("bp", "p", "eclipsis"),
        ("dt", "t", "eclipsis"),
        ("mh", "m", "lenition"),
        ("bh", "b", "lenition"),
        ("dh", "d", "lenition"),
        ("gh", "g", "lenition"),
        ("ph", "p", "lenition"),
        ("fh", "f", "lenition"),
        ("sh", "s", "lenition"),
    ],
}


def derive_mutation_lemma_candidate(inflected_form: str, language: str) -> tuple[str, str] | None:
    """wyrd-jott: Goidelic-specific lemma-derivation via initial-
    mutation prefix stripping. Mirror of ``derive_lemma_candidate``
    but for prefix-shaped morphology rather than suffix.

    Returns ``(lemma_form, mutation_label)`` or None. Conservative:
    requires the resulting lemma to be at least 3 characters (same
    floor as the suffix path).

    Walks ``MUTATION_RULES[language]`` in declared order; the
    longest-prefix-first ordering means 'bhf' (eclipsis of f-) wins
    over 'bh' (lenition of b-) when both could match. Returns the
    first hit; if no rule fires, returns None.
    """
    if not inflected_form:
        return None
    rules = MUTATION_RULES.get(language, [])
    s = inflected_form.lower()
    for prefix, replacement, label in rules:
        if not s.startswith(prefix):
            continue
        stem = replacement + s[len(prefix) :]
        if len(stem) < 3:
            continue
        return stem, label
    return None


def derive_lemma_candidates(inflected_form: str, language: str) -> list[tuple[str, str]]:
    """Try to identify lemma forms for an inflected etymon by stripping
    a known inflectional suffix.

    Returns a list of (lemma_form, inflection_label) candidates in
    preferred order — caller picks the first one that matches an
    existing etymon row. Conservative: only suggests stems ≥3 chars.

    Each ``INFLECTION_RULES`` entry is either ``(suffix, label)`` or
    ``(suffix, label, restore_suffix)``. The 3-tuple form lets a rule
    produce TWO candidates: the stripped stem plus ``restore_suffix``
    appended (e.g. silent-e for modern English: 'hoped' → 'hope'),
    and the bare stripped stem ('hoped' → 'hop'). Restore-suffix
    candidates are preferred — silent-e is the common case for modern
    English -ed/-ing inflections.
    """
    if not inflected_form:
        return []
    rules = INFLECTION_RULES.get(language, [])
    s = inflected_form.lower()
    out: list[tuple[str, str]] = []
    for rule in rules:
        # Extended unpacking handles both the 2-tuple (legacy) and
        # 3-tuple (wyrd-gf28 restore-suffix) rule shapes. ``*rest``
        # is empty for 2-tuples; restore defaults to ''.
        suffix, label, *rest = rule
        restore = rest[0] if rest else ""
        if not s.endswith(suffix) or len(s) - len(suffix) < 3:
            continue
        stem = s[: -len(suffix)]
        if restore:
            # Prefer restored-suffix candidate (silent-e etc.) before
            # bare stem so a real lemma like 'hope' wins over a
            # coincidentally-existing 'hop'.
            out.append((stem + restore, label))
        out.append((stem, label))
    return out


def derive_lemma_candidate(inflected_form: str, language: str) -> tuple[str, str] | None:
    """Back-compat single-candidate wrapper around ``derive_lemma_candidates``.

    Returns the FIRST candidate or None. Existing callers (and the
    tests pinned against this signature) keep working unchanged;
    link-lemmas itself uses the plural form to leverage the silent-e
    alternates.
    """
    candidates = derive_lemma_candidates(inflected_form, language)
    return candidates[0] if candidates else None


def link_lemmas(db: LexiconDB, *, apply: bool = False) -> dict:
    """Walk unlinked etymons and link them to a matching lemma row.

    For each etymon with `lemma_id IS NULL`, try every applicable
    inflection-strip rule. If the resulting stem matches an EXISTING etymon
    row in the same language, set lemma_id and inflection. Skips:
      - etymons that already have a lemma_id
      - cases where the stripped stem doesn't match any existing etymon
        (we don't fabricate new lemmas in this pass)
      - self-linkage (an etymon already serving as a lemma)

    With apply=False (default) reports candidates without writing.
    """
    # Pull every unlinked, non-tombstoned etymon once. The same set is
    # both the candidate-lemma source (potential link targets) and the
    # candidate-inflected-row source (rows that might need a link). We
    # exclude OCR-merge tombstones (merged_into_id IS NOT NULL) from
    # both sides: linking an inflected variant to a tombstone would
    # create a child → tombstone → canonical chain that the consensus
    # rollup would split, and tombstones themselves don't need linking.
    candidate_rows = list(
        db.conn.execute(
            "SELECT id, canonical_form, language FROM etymon "
            "WHERE lemma_id IS NULL AND merged_into_id IS NULL"
        )
    )
    lemmas_by_key: dict[tuple[str, str], int] = {
        (row["canonical_form"].lower(), row["language"]): row["id"] for row in candidate_rows
    }
    proposed: list[dict] = []
    inflected_rows = candidate_rows
    for row in inflected_rows:
        # Try suffix-strip first (INFLECTION_RULES); always APPEND the
        # mutation prefix-strip candidate (MUTATION_RULES) so it acts
        # as a fallback when suffix candidates don't match a DB lemma.
        # Pre-wyrd-gf28 the mutation path only ran when suffix
        # returned no candidates at all — but with the new plural-
        # candidates form a suffix rule can produce 2+ candidates
        # that ALL fail to resolve, and the mutation path needs to
        # still get a shot. Suffix priority preserved because
        # candidates are tried in list order.
        candidates = derive_lemma_candidates(row["canonical_form"], row["language"])
        mut = derive_mutation_lemma_candidate(row["canonical_form"], row["language"])
        if mut is not None:
            candidates.append(mut)
        lemma_id = None
        inflection = None
        lemma_form = None
        for candidate_form, candidate_label in candidates:
            candidate_id = lemmas_by_key.get((candidate_form, row["language"]))
            if candidate_id is not None and candidate_id != row["id"]:
                lemma_id = candidate_id
                inflection = candidate_label
                lemma_form = candidate_form
                break
        if lemma_id is None:
            continue
        proposed.append(
            {
                "inflected_id": row["id"],
                "inflected_form": row["canonical_form"],
                "lemma_id": lemma_id,
                "lemma_form": lemma_form,
                "inflection": inflection,
                "language": row["language"],
            }
        )

    if apply:
        for p in proposed:
            db.conn.execute(
                "UPDATE etymon SET lemma_id = ?, inflection = ?, "
                "lemma_method = 'link-lemmas-v1' WHERE id = ?",
                (p["lemma_id"], p["inflection"], p["inflected_id"]),
            )
        db.commit()

    return {
        "candidates": len(proposed),
        "sample": proposed[:25],
        "applied": apply,
    }
