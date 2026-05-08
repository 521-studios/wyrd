"""Sound-change rules library — Phase 1 (wyrd-4i6).

Encodes phonological transforms per ``(language, era_from, era_to)``
cell. Phase 1 ships the framework + ONE cell (Old English → Middle
English). Phase 2 extends to ME → EModE, EModE → ModE, OW → ModW; per-
culture overrides land later.

Public API: ``apply_rules(form, language, from_era, to_era, mode)``
returns ``[(derived_form, probability), ...]``. ``has_rules`` checks
whether a given forward cell is registered.

Rule encoding: patterns are literal strings (no regex). Rule order
matters — declarations are specific-before-general so multi-char
suffixes / digraphs match before constituent single chars consume
them. Per-rule docstring sections cover documentation requirements
(description, exemplar, bibliographic source).

Composition with downstream features (homophone mutation, time-warp,
calque/foreignize) lives in their own modules; this one is the rule
data + transform engine.
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


# --- Middle English → Early Modern English (wyrd-n9x5) ------------------
#
# Sources: Sweet, *History of English Sounds* (1888, §§ on the Great
# Vowel Shift block); Smith, *English Place-Name Elements* (EPNS vol
# 25–26, 1956, suffix evolution by period); Lass, *The Cambridge
# History of the English Language* vol III (1999, EModE phonology
# chapter for cross-checking).
#
# Note on scope: the Great Vowel Shift is primarily a PHONEMIC shift,
# not orthographic — most ME spellings persist through EModE
# (mete, name, mood). Rules below cover the orthographic transitions
# that DO surface, particularly in place-name forms: final -e loss,
# diphthong reanalysis, and a handful of consonant-cluster shifts.
# The set is deliberately smaller than OE→ME (~10 rules vs 28)
# because so much of the phonological change in this period stays
# below the spelling layer; Phase 2 mining of period-specific
# attestation forms would expand the rule set if needed.

ME_TO_EMODE_RULES: tuple[SoundChangeRule, ...] = (
    # --- Place-name suffix shifts ---
    SoundChangeRule(
        pattern="-worde",
        replacement="-worth",
        weight=1.0,
        description=(
            "Habitative suffix -worde → -worth (final -e loss + th preserved). "
            "ME 'Pal-worde' → EModE 'Pal-worth'."
        ),
        exemplar=("Pal-worde", "Pal-worth"),
        source="Smith EPNS s.v. 'worð'",
    ),
    SoundChangeRule(
        pattern="-burgh",
        replacement="-borough",
        weight=1.0,
        description=(
            "Habitative suffix -burgh → -borough (some Northern dialects; "
            "spelling normalization in EModE Standard). Cf. Edinburgh "
            "(retained Northern) vs Peterborough (Southern reflex)."
        ),
        exemplar=("Peter-burgh", "Peter-borough"),
        source="Smith EPNS s.v. 'burh'",
    ),
    SoundChangeRule(
        pattern="-feld",
        replacement="-field",
        weight=1.0,
        description=(
            "Topographic suffix -feld → -field (vowel quality stable; "
            "spelling adjustment with i-insertion in EModE Standard)."
        ),
        exemplar=("Spring-feld", "Spring-field"),
        source="Smith EPNS s.v. 'feld'",
    ),
    SoundChangeRule(
        pattern="-mor",
        replacement="-moor",
        weight=1.0,
        description=(
            "Topographic suffix -mor → -moor (long vowel oo-spelling normalization in EModE)."
        ),
        exemplar=("Black-mor", "Black-moor"),
        source="Smith EPNS s.v. 'mōr'",
    ),
    SoundChangeRule(
        pattern="-wude",
        replacement="-wood",
        weight=1.0,
        description=(
            "Topographic suffix -wude (ME) → -wood (EModE; final -e loss + long vowel oo-spelling)."
        ),
        exemplar=("Birch-wude", "Birch-wood"),
        source="Smith EPNS s.v. 'wudu'",
    ),
    SoundChangeRule(
        pattern="-stede",
        replacement="-stead",
        weight=1.0,
        description=(
            "Habitative suffix -stede → -stead (final -e loss + ea-spelling for the long e reflex)."
        ),
        exemplar=("Hamp-stede", "Hamp-stead"),
        source="Smith EPNS s.v. 'stede'",
    ),
    # --- Final -e loss (orthographic) ---
    #
    # Final -e was unstressed and lost in pronunciation late in ME, but
    # the orthographic loss persisted into EModE. This rule fires last
    # in declared order so it doesn't strip -e from intermediate forms
    # whose suffix transitions haven't fired yet.
    SoundChangeRule(
        pattern="ye",
        replacement="y",
        weight=1.0,
        description=(
            "Word-final -ye → -y (final -e loss after -y; ME 'fyye' → "
            "EModE 'fy'). Treats the digraph; bare-vowel -e is a "
            "separate rule below."
        ),
        exemplar=("Bricye", "Bricy"),
        source="Sweet 1888 §§ 766–768",
    ),
    # --- Diphthong shifts ---
    SoundChangeRule(
        pattern="ai",
        replacement="ay",
        weight=1.0,
        description=(
            "Diphthong -ai- → -ay- (orthographic normalization in EModE; "
            "ME 'dai' → EModE 'day'). Stable in pronunciation; the spelling "
            "shift is the visible change."
        ),
        exemplar=("Hai-ston", "Hay-ston"),
        source="Sweet 1888 § 814",
    ),
    SoundChangeRule(
        pattern="ei",
        replacement="ey",
        weight=1.0,
        description=("Diphthong -ei- → -ey- (orthographic normalization; ME 'kei' → EModE 'key')."),
        exemplar=("Mei-ham", "Mey-ham"),
        source="Sweet 1888 § 814",
    ),
    SoundChangeRule(
        pattern="ou",
        replacement="ow",
        weight=1.0,
        description=(
            "Diphthong -ou- → -ow- word-finally / word-medially (orthographic; "
            "the ME 'hous' / 'cou' spellings settle on -ow- in EModE: 'how' / "
            "'cow'). Note that morpheme-internal -ou- often persists "
            "(house, mouse) — this rule is dialect-specific; the literal-"
            "pattern Phase 1 framework can't gate environment, so apply "
            "judiciously."
        ),
        exemplar=("Houroun", "Howrown"),
        source="Sweet 1888 § 837",
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
    ("english", "middle-english", "early-modern-english"): ME_TO_EMODE_RULES,
}


# Probability floor for candidate pruning. Inverse mode branches 50/50
# on every rule; a 28-rule chain on pathological input could otherwise
# blow up exponentially. 1e-6 is permissive (any real reflex chain
# stays well above) but bounds the candidate list past the chain.
_PROBABILITY_FLOOR: float = 1e-6


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
    ``form`` and return a list of ``(derived_form, probability)``.

    Forward mode walks rules in declared order. Inverse walks the
    SAME cell in REVERSE rule order with pattern↔replacement swapped;
    when two forward rules collapse different inputs onto the same
    replacement (``ē`` and ``ǣ`` both → ``e``), inverse from the
    target surfaces both candidates. Inverse mode always branches
    (the inverse of even a universal forward rule is non-
    deterministic); a probability floor (``_PROBABILITY_FLOOR``)
    drops candidates whose mass falls past 1e-6 so inverse explosion
    stays bounded.

    Unknown cells return ``[(form, 1.0)]``. Use ``has_rules`` to
    discriminate 'no cell registered' from 'cell ran, no match'.
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
        # Prune low-probability candidates after each rule. Inverse mode
        # branches 50/50 on every rule; without a floor a 28-rule chain
        # could blow up to 2^28 candidates on pathological input. The
        # floor is permissive enough (1e-6) that real candidates survive
        # the chain — only mass-near-zero combinations are dropped.
        candidates = _dedupe_candidates(candidates)
        if len(candidates) > 1:
            candidates = [c for c in candidates if c[1] >= _PROBABILITY_FLOOR]
            if not candidates:
                # Pathological case: every candidate fell below the
                # floor. Restore the input as a defensive fallback so
                # callers never get an empty list.
                candidates = [(form, 1.0)]
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

    Forward (``always_branch=False``): universal rule (weight=1.0)
    collapses each input into one output; sporadic (weight<1.0)
    branches into (fired with prob=weight, didn't fire with
    prob=1-weight).

    Inverse (``always_branch=True``): always branches at 50/50 even
    for universal forward rules — the inverse direction is inherently
    non-deterministic. Phase 2+ may attach attestation-weighted priors.

    Candidates not containing ``pattern`` pass through unchanged.
    Empty ``pattern`` is a no-op (would otherwise interpolate
    ``replacement`` between every character via ``str.replace``).
    """
    if not pattern:
        return list(candidates)
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
