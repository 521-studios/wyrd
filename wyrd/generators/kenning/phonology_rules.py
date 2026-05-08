"""Sound-change rules library — Phase 1 (wyrd-4i6).

Encodes phonological transforms per ``(language, era_from, era_to)``
cell. Phase 1 ships the framework + ONE well-attested cell (Old English
→ Middle English, ~25 rules covering place-name morphology). Phase 2
extends to ME → EModE, EModE → ModE, and Welsh OW → ModW; Phase 3 may
add per-culture overrides (e.g. Scots-specific reflexes).

Public API:

    apply_rules(form, language, from_era, to_era, mode='forward'|'inverse')
        -> list[tuple[str, float]]

Returns ``[(derived_form, probability), ...]``. Forward mode applies
the rules for ``(language, from_era, to_era)`` in order; inverse mode
reverses the cell direction and inverts each rule's pattern ↔
replacement. Universal rules (``weight == 1.0``) fire deterministically.
Sporadic rules (``weight < 1.0``) produce two candidates per branch
(the rule fires with prob=weight, doesn't fire with prob=1-weight) and
the candidate list expands accordingly. Phase 1 rules are all universal
to keep the curated set small + auditable; Phase 2 mining adds
sporadic rules.

Unknown cells (no rules registered) are a no-op: the input form is
returned unchanged with probability 1.0. Callers can pre-check
``has_rules(language, from_era, to_era)`` if they need to distinguish
'no rules registered' from 'rules ran, no changes'.

Rule encoding choices:

- **Patterns are literal strings**, not regex. Phase 1 covers place-
  name morphology where literal substring rules ('-tūn' → '-ton',
  'sċ' → 'sh') express the transitions cleanly. Regex (with
  environment constraints like 'before front vowel') is a Phase 2
  refinement; the framework here is small enough to upgrade later
  without API breakage.
- **Rules apply in declared order**. Order matters: 'sċ' → 'sh' must
  run before 'ċ' → 'ch' or the digraph case would mis-match. Each
  cell's rule list documents the ordering rationale inline.
- **Inverse swaps pattern ↔ replacement** and reverses iteration order.
  A forward rule 'ū' → 'ou' inverses to 'ou' → 'ū'. Multi-candidate
  semantics naturally surface when two forward rules collapse
  different OE forms into the same ME form: inverse from the ME
  produces both candidates.

Documentation requirements per rule (enforced by tests):

- ``description``: human-readable summary of the change.
- ``exemplar``: ``(input, output)`` pair the rule should produce. Test
  for each rule asserts the exemplar round-trips through the rule.
- ``source``: bibliographic citation (Campbell OE Grammar §X, Smith
  Place-Names §Y, Hogg & Fulk grammar reference, etc.).

Composition with the higher-level downstream features (homophone
mutation, time-warp, calque/foreignize) lives in their respective
modules; this one is the rule data + transform engine only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SoundChangeRule:
    """One phonological transition rule.

    Forward direction: every literal occurrence of ``pattern`` in the
    input form is replaced with ``replacement`` in the output form.

    Inverse direction: every literal occurrence of ``replacement`` in
    the input form is replaced with ``pattern``. Multiple forward
    rules collapsing onto the same replacement produce multiple
    inverse candidates — that's the point.
    """

    pattern: str
    replacement: str
    weight: float
    description: str
    exemplar: tuple[str, str]
    source: str


# --- Old English → Middle English ----------------------------------------
#
# Sources: Campbell, *Old English Grammar* (1959, §§ 379–428 'changes
# in Late OE'); Smith, *English Place-Name Elements* (EPNS vol 25–26,
# 1956, the canonical reference for place-name suffix transitions);
# Hogg & Fulk, *A Grammar of Old English* (vols I–II, 1992–2011 — the
# standard modern treatment, used for cross-checking exemplars). Three
# standard handbook treatments converge on the rules below; cited per-
# rule for the most specific authority.
#
# Rules ordered specific-before-general so a multi-char place-name
# suffix ('-tūn', '-byriġ', '-lēah') fires intact before a constituent
# single-char rule (ū, ġ, ē) would otherwise consume part of it.
# Within the single-char block, digraphs (sċ, ċġ, hl, hr, hn) precede
# their constituents for the same reason.

OE_TO_ME_RULES: tuple[SoundChangeRule, ...] = (
    # --- Place-name suffixes (well-attested transitions; Smith EPNS
    # is the canonical reference for each). Run FIRST so the suffix's
    # internal vowels / consonants don't get rewritten by char-level
    # rules before the suffix can match.
    SoundChangeRule(
        pattern="-tūn",
        replacement="-ton",
        weight=1.0,
        description="Habitative suffix -tūn → -ton in ME spelling.",
        exemplar=("Hædan-tūn", "Hædan-ton"),
        source="Smith EPNS s.v. 'tūn'",
    ),
    SoundChangeRule(
        pattern="-hām",
        replacement="-ham",
        weight=1.0,
        description="Habitative suffix -hām → -ham (vowel shortening + macron loss).",
        exemplar=("Hēah-hām", "Hēah-ham"),
        source="Smith EPNS s.v. 'hām'",
    ),
    SoundChangeRule(
        pattern="-dūn",
        replacement="-don",
        weight=1.0,
        description="Topographic suffix -dūn → -don (ū → o variant of the place-name reflex).",
        exemplar=("Hēah-dūn", "Hēah-don"),
        source="Smith EPNS s.v. 'dūn'",
    ),
    SoundChangeRule(
        pattern="-lēah",
        replacement="-ley",
        weight=1.0,
        description="Topographic suffix -lēah → -ley (clearing/woodland glade).",
        exemplar=("Wēod-lēah", "Wēod-ley"),
        source="Smith EPNS s.v. 'lēah'",
    ),
    SoundChangeRule(
        pattern="-byriġ",
        replacement="-bury",
        weight=1.0,
        description="Habitative suffix -byriġ → -bury (dative singular of burh, fortified place).",
        exemplar=("Cant-byriġ", "Cant-bury"),
        source="Smith EPNS s.v. 'burh'",
    ),
    SoundChangeRule(
        pattern="-burh",
        replacement="-burgh",
        weight=1.0,
        description="Habitative suffix -burh → -burgh (Northern ME variant).",
        exemplar=("Eden-burh", "Eden-burgh"),
        source="Smith EPNS s.v. 'burh'",
    ),
    SoundChangeRule(
        pattern="-wīc",
        replacement="-wich",
        weight=1.0,
        description="Suffix -wīc → -wich (specialized building / dairy farm).",
        exemplar=("Sand-wīc", "Sand-wich"),
        source="Smith EPNS s.v. 'wīc'",
    ),
    SoundChangeRule(
        pattern="-hyrst",
        replacement="-hurst",
        weight=1.0,
        description="Topographic suffix -hyrst → -hurst (wooded hill / copse).",
        exemplar=("Mid-hyrst", "Mid-hurst"),
        source="Smith EPNS s.v. 'hyrst'",
    ),
    # --- Palatalization digraphs (precede single-char rules) ---
    SoundChangeRule(
        pattern="sċ",
        replacement="sh",
        weight=1.0,
        description="Palatalized sċ → sh (West Germanic *sk fronted by Old English).",
        exemplar=("fisċ", "fish"),
        source="Campbell §428; Smith EPNS s.v. 'fisc'",
    ),
    SoundChangeRule(
        pattern="ċġ",
        replacement="dg",
        weight=1.0,
        description="Geminate palatal ċġ → dg (later spelt 'dge').",
        exemplar=("bryċġ", "brydg"),
        source="Campbell §428",
    ),
    SoundChangeRule(
        pattern="ċ",
        replacement="ch",
        weight=1.0,
        description="Palatalized ċ → ch (before front vowels in OE).",
        exemplar=("ċild", "child"),
        source="Campbell §428",
    ),
    SoundChangeRule(
        pattern="ġ",
        replacement="y",
        weight=1.0,
        description="Palatalized ġ → y (initial / between front vowels).",
        exemplar=("ġēar", "yēar"),
        source="Campbell §427",
    ),
    # --- Thorn / eth normalization ---
    SoundChangeRule(
        pattern="þ",
        replacement="th",
        weight=1.0,
        description="Runic thorn þ → digraph th (orthographic shift in ME).",
        exemplar=("þorp", "thorp"),
        source="Smith EPNS s.v. 'þorp'",
    ),
    SoundChangeRule(
        pattern="ð",
        replacement="th",
        weight=1.0,
        description="Eth ð → th (OE used þ and ð interchangeably; ME prefers th).",
        exemplar=("worðiġ", "worthiġ"),
        source="Smith EPNS s.v. 'worð'",
    ),
    # --- Long vowels: macrons largely lost orthographically; specific
    # quality changes carried per-vowel where the change is real (not
    # just a notation collapse).
    SoundChangeRule(
        pattern="ǣ",
        replacement="e",
        weight=1.0,
        description="Long æ → e (Late OE; merger with native ē).",
        exemplar=("dǣg", "deg"),
        source="Campbell §379; Hogg & Fulk vol I §5.2",
    ),
    SoundChangeRule(
        pattern="ā",
        replacement="o",
        weight=1.0,
        description="Long ā → o in Southern ME (the famous 'stān → stoon → stone' shift).",
        exemplar=("stān", "ston"),
        source="Campbell §379; Smith EPNS s.v. 'stān'",
    ),
    SoundChangeRule(
        pattern="ē",
        replacement="e",
        weight=1.0,
        description="Long ē → e (orthographic; vowel quality preserved).",
        exemplar=("hēah", "heah"),
        source="Campbell §379",
    ),
    SoundChangeRule(
        pattern="ī",
        replacement="i",
        weight=1.0,
        description="Long ī → i (orthographic macron loss).",
        exemplar=("hwīt", "hwit"),
        source="Campbell §379",
    ),
    SoundChangeRule(
        pattern="ō",
        replacement="o",
        weight=1.0,
        description="Long ō → o (orthographic macron loss; quality preserved).",
        exemplar=("bōc", "boc"),
        source="Campbell §379",
    ),
    SoundChangeRule(
        pattern="ū",
        replacement="ou",
        weight=1.0,
        description="Long ū → ou (Anglo-Norman scribal convention adopted in ME).",
        exemplar=("hūs", "hous"),
        source="Campbell §379; Smith EPNS s.v. 'hūs'",
    ),
    SoundChangeRule(
        pattern="ȳ",
        replacement="i",
        weight=1.0,
        description="Long ȳ → i (Southern ME unrounding; West Saxon ȳ merges with Anglian ī).",
        exemplar=("brȳd", "brid"),
        source="Campbell §380; Hogg & Fulk vol I §5.5",
    ),
    # --- Short vowels ---
    SoundChangeRule(
        pattern="æ",
        replacement="a",
        weight=1.0,
        description="Short æ → a (Late OE merger with PWGmc *a in many dialects).",
        exemplar=("æt", "at"),
        source="Campbell §380",
    ),
    SoundChangeRule(
        pattern="y",
        replacement="i",
        weight=1.0,
        description="Short y → i (Southern unrounding; cf. Anglian where y → e).",
        exemplar=("hyll", "hill"),
        source="Campbell §380",
    ),
    # --- Initial consonant cluster simplification ---
    SoundChangeRule(
        pattern="hl",
        replacement="l",
        weight=1.0,
        description="Initial hl- → l- (loss of pre-consonantal h in Late OE).",
        exemplar=("hlīw", "līw"),
        source="Campbell §471",
    ),
    SoundChangeRule(
        pattern="hr",
        replacement="r",
        weight=1.0,
        description="Initial hr- → r- (loss of pre-consonantal h).",
        exemplar=("hrōf", "rōf"),
        source="Campbell §471",
    ),
    SoundChangeRule(
        pattern="hn",
        replacement="n",
        weight=1.0,
        description="Initial hn- → n- (loss of pre-consonantal h).",
        exemplar=("hnutu", "nutu"),
        source="Campbell §471",
    ),
)


# --- dispatch table ------------------------------------------------------
#
# Keyed by ``(language, from_era, to_era)``. Add a tuple here and the
# rules become reachable via ``apply_rules``. Forward direction lookup
# uses the tuple as-is; inverse direction looks up
# ``(language, to_era, from_era)`` (the same forward cell, then
# inverts each rule's pattern ↔ replacement at apply time).

_RULES: dict[tuple[str, str, str], tuple[SoundChangeRule, ...]] = {
    ("english", "old-english", "middle-english"): OE_TO_ME_RULES,
}


def has_rules(language: str, from_era: str, to_era: str) -> bool:
    """True iff Phase 1+ ships rules for the requested cell direction
    (forward). Inverse availability is the same check with eras
    swapped — callers needing inverse should query
    ``has_rules(language, to_era, from_era)``."""
    return (language, from_era, to_era) in _RULES


def apply_rules(
    form: str,
    language: str,
    from_era: str,
    to_era: str,
    mode: str = "forward",
) -> list[tuple[str, float]]:
    """Apply the rule cell for ``(language, from_era, to_era)`` to
    ``form`` and return a list of ``(derived_form, probability)``
    candidates.

    Forward mode walks rules in declared order; each rule replaces
    every occurrence of its ``pattern`` with ``replacement``.

    Inverse mode walks the SAME cell in REVERSE rule order with
    pattern↔replacement swapped — undoing the forward chain rule by
    rule. When two forward rules collapse different inputs onto the
    same replacement (e.g. both ``ē`` and ``ǣ`` → ``e``), inverse from
    the shared output produces multiple candidates, each branched at
    the offending rule.

    Universal rules (``weight == 1.0``) fire deterministically (1
    candidate per input). Sporadic rules (``weight < 1.0``) branch:
    one candidate where the rule fired (probability *= weight), one
    where it didn't (probability *= (1 - weight)). Phase 1 cells are
    all universal; sporadic-rule branching is shipped now so Phase 2
    additions land without API change.

    Unknown cells (no rules registered for the requested direction)
    return ``[(form, 1.0)]`` unchanged. Callers that need to
    distinguish 'no rules registered' from 'rules ran, no changes'
    should call ``has_rules`` first.
    """
    if mode not in ("forward", "inverse"):
        raise ValueError(f"mode must be 'forward' or 'inverse', got {mode!r}")

    # Both directions consult the SAME forward-registered cell. Forward
    # mode walks the rules in declared order; inverse mode walks them
    # in reverse with pattern↔replacement swapped at apply time.
    forward_rules = _RULES.get((language, from_era, to_era), ())
    if mode == "forward":
        rules = forward_rules
    else:
        rules = tuple(reversed(forward_rules))

    candidates: list[tuple[str, float]] = [(form, 1.0)]
    inverse = mode == "inverse"
    for rule in rules:
        pattern = rule.pattern if not inverse else rule.replacement
        replacement = rule.replacement if not inverse else rule.pattern
        if not pattern:
            continue
        candidates = _apply_one_rule(
            candidates, pattern, replacement, rule.weight, always_branch=inverse
        )
    return _dedupe_candidates(candidates)


def _apply_one_rule(
    candidates: list[tuple[str, float]],
    pattern: str,
    replacement: str,
    weight: float,
    *,
    always_branch: bool = False,
) -> list[tuple[str, float]]:
    """Apply one rule's transformation across every candidate.

    Forward (``always_branch=False``): a universal rule (weight=1.0)
    collapses each input into one output; a sporadic rule branches
    into (fired with prob=weight, didn't fire with prob=1-weight).

    Inverse (``always_branch=True``): always branches, even for
    universal forward rules — the inverse direction is inherently
    non-deterministic (the ME form 'fish' could have come from OE
    'fisċ' OR from a hypothetical 'fish' that already had 'sh').
    Probability splits 50/50 in inverse mode regardless of forward
    weight; refinement (attestation-weighted priors) is Phase 2+.

    Candidates not containing ``pattern`` pass through unchanged.
    """
    new_candidates: list[tuple[str, float]] = []
    for cand_form, cand_weight in candidates:
        if pattern not in cand_form:
            new_candidates.append((cand_form, cand_weight))
            continue
        replaced = cand_form.replace(pattern, replacement)
        if always_branch:
            new_candidates.append((replaced, cand_weight * 0.5))
            new_candidates.append((cand_form, cand_weight * 0.5))
        elif weight >= 1.0:
            new_candidates.append((replaced, cand_weight))
        else:
            new_candidates.append((replaced, cand_weight * weight))
            new_candidates.append((cand_form, cand_weight * (1.0 - weight)))
    return new_candidates


def _dedupe_candidates(
    candidates: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Merge candidates with identical forms by summing probabilities.
    The order of first occurrence is preserved so test assertions
    against the candidate list are stable."""
    seen: dict[str, float] = {}
    order: list[str] = []
    for form, prob in candidates:
        if form not in seen:
            seen[form] = 0.0
            order.append(form)
        seen[form] += prob
    return [(form, seen[form]) for form in order]
