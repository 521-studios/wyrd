"""Sound-change rules library — Phases 1 + 2.1 + 2.2 + 2.3 (wyrd-4i6 / wyrd-n9x5).

Encodes phonological transforms per ``(language, era_from, era_to)``
cell. Phase 1 (PR #131) shipped the framework + Old English → Middle
English. Phase 2.1 (PR #135) added Middle English → Early Modern
English. Phase 2.3 (PR #141) added Old Welsh → Modern Welsh. Phase
2.2 (this commit) adds Early Modern English → Modern English. Phase
2.4 (sporadic-weight rules + env constraints + restoration of dropped
rules) remains.

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
            "Topographic suffix -feld → -field (16th-17th c. printer-driven "
            "orthographic standardization; vowel quality phonologically stable)."
        ),
        exemplar=("Spring-feld", "Spring-field"),
        source="Smith EPNS s.v. 'feld'",
    ),
    SoundChangeRule(
        pattern="-mor",
        replacement="-moor",
        weight=1.0,
        description=(
            "Topographic suffix -mor → -moor (orthographic standardization of "
            "the long vowel as oo in EModE printer-era)."
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
    # --- Diphthong shifts (printer-era orthographic standardization) ---
    #
    # ai/ei → ay/ey is a stable spelling normalization in the 16th-c.
    # printer era; ME 'ai'/'ei' digraphs WERE consistently respelled in
    # EModE, so the literal-pattern framework handles them cleanly with
    # no false-fires on real ME forms. Rules NOT in this cell:
    # - 'ye → y' (final-e loss after y) — substring-overreach risk
    #   (would mangle 'Wyeham', 'yeoman'). Defer to Phase 2.4 with
    #   word-final regex anchor.
    # - 'ou → ow' — environment-sensitive (word-final OK, morpheme-
    #   internal NOT — would corrupt ME 'hous'/'mous'). Defer to
    #   Phase 2.4 with env-constraint regex.
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
)


# --- Early Modern English → Modern English (wyrd-n9x5 Phase 2.2) --------
#
# Sources: Smith, *English Place-Name Elements* (EPNS vol 25–26, 1956,
# suffix evolution); Mills, *A Dictionary of British Place Names* (rev.
# ed. 2011, per-toponym attestation chains showing EModE intermediate
# forms). Each rule cites the EPNS s.v. for the suffix root and the
# Mills s.v. for the canonical exemplar attestation.
#
# Note on scope: most British place-name orthography stabilized by
# EModE — the dominant phonological changes (Great Vowel Shift,
# consonant cluster shifts) had run their course by 1700, and the
# 18th–19th c. settling is mostly orthographic standardization rather
# than phonological. The rules below cover the surfacing transitions
# that DO appear in attestation chains: final-e drop on EModE-attested
# place-name suffixes (-stocke, -brooke, -grene, -poole, -ditche)
# yielding their ModE forms. The set is deliberately compact (5
# rules); Phase 2 mining of period-specific attestation forms would
# expand it if needed.

EMODE_TO_MODE_RULES: tuple[SoundChangeRule, ...] = (
    SoundChangeRule(
        pattern="-stocke",
        replacement="-stock",
        weight=1.0,
        description=(
            "Habitative suffix -stocke → -stock (final -e loss). EModE "
            "'Tavystocke' / 'Tavistocke' → ModE 'Tavistock'; same "
            "pattern in 'Bemerstocke' → 'Bemerstock'. Stable in "
            "pronunciation; the e drop is the orthographic settling."
        ),
        exemplar=("Tavi-stocke", "Tavi-stock"),
        source="Smith EPNS s.v. 'stoc'; Mills s.v. 'Tavistock'",
    ),
    SoundChangeRule(
        pattern="-brooke",
        replacement="-brook",
        weight=1.0,
        description=(
            "Topographic suffix -brooke → -brook (final -e loss). EModE "
            "'Holbrooke' / 'Hollybrooke' → ModE 'Holbrook'. The "
            "preceding ME→EModE step had already settled the long-o "
            "as -oo- spelling; Phase 2.2 strips the EModE final -e."
        ),
        exemplar=("Hol-brooke", "Hol-brook"),
        source="Smith EPNS s.v. 'brōc'; Mills s.v. 'Holbrook'",
    ),
    SoundChangeRule(
        pattern="-grene",
        replacement="-green",
        weight=1.0,
        description=(
            "Topographic suffix -grene → -green (final -e loss + "
            "ee-spelling for the long-e reflex). EModE 'Bethnal Grene' "
            "→ ModE 'Bethnal Green'. Cf. -wude → -wood (Phase 2.1) for "
            "the parallel oo-spelling on the long-o reflex."
        ),
        exemplar=("Bethnal-grene", "Bethnal-green"),
        source="Smith EPNS s.v. 'grēne'; Mills s.v. 'Bethnal Green'",
    ),
    SoundChangeRule(
        pattern="-poole",
        replacement="-pool",
        weight=1.0,
        description=(
            "Topographic suffix -poole → -pool (final -e loss). EModE "
            "'Liverpoole' / 'Hartelpoole' → ModE 'Liverpool' / "
            "'Hartlepool'. The oo-spelling for long-o was already "
            "stable by EModE; Phase 2.2 strips the trailing -e."
        ),
        exemplar=("Liver-poole", "Liver-pool"),
        source="Smith EPNS s.v. 'pōl'; Mills s.v. 'Liverpool'",
    ),
    SoundChangeRule(
        pattern="-ditche",
        replacement="-ditch",
        weight=1.0,
        description=(
            "Topographic suffix -ditche → -ditch (final -e loss). "
            "EModE 'Houndsditche' → ModE 'Houndsditch' (London "
            "ward / street name); same pattern across other ditch- "
            "compounds with EModE -e tail."
        ),
        exemplar=("Hounds-ditche", "Hounds-ditch"),
        source="Smith EPNS s.v. 'dīc'; Mills s.v. 'Houndsditch'",
    ),
)


# --- Old Welsh → Modern Welsh (wyrd-n9x5 Phase 2.3) ---------------------
#
# Sources: Morris-Jones, *A Welsh Grammar Historical and Comparative*
# (1913, the canonical handbook for OW→MW phonology); Lewis & Pedersen,
# *A Concise Comparative Celtic Grammar* (1937, used for cross-checking
# diphthong reflexes); Charles, *The Place-Names of Pembrokeshire*
# (1992, the EPNS-style reference for Welsh place-name suffix evolution).
#
# Old Welsh attestation is sparse (8th–11th c. glosses + a few short
# texts); most "OW→MW" transitions in the handbooks are reconstructed
# from the gloss corpus + comparative reconstruction. The rules below
# cover orthographic shifts that surface in real Welsh place-names:
# the k→c spelling normalization (OW used 'k' for /k/, MW uses 'c'),
# the au/iu/ui/oi diphthong reflexes, the place-name suffix transitions
# -auc → -og / -iauc → -iog (very common in MW toponyms — Mathryniog,
# Pennog, etc.), and a small handful of consonant cluster shifts. The
# rule set is deliberately compact (10 rules) — Welsh place-name
# orthography stabilized late-medieval and most variation lives in
# stratum-specific spellings rather than systematic OW→MW change.
#
# Rule ordering: multi-char place-name suffixes ('-iauc', '-auc') run
# BEFORE single-char rules so the 'k → c' normalization doesn't consume
# the 'k' in '-iauc' before the suffix can match. Within the diphthong
# block, longer patterns ('kair' which captures both k→c AND ai→ae)
# precede their constituent single-char rules.

OW_TO_MW_RULES: tuple[SoundChangeRule, ...] = (
    # --- Place-name suffixes (multi-char, run first) ---
    SoundChangeRule(
        pattern="-iauc",
        replacement="-iog",
        weight=1.0,
        description=(
            "Adjectival/place-name suffix -iauc → -iog. au-monophthongization "
            "+ k→g voicing in word-final position; the preceding -i- carries "
            "across. OW 'mathryn-iauc' → MW 'Mathryn-iog'."
        ),
        exemplar=("Mathryn-iauc", "Mathryn-iog"),
        source="Morris-Jones 1913 § 21",
    ),
    SoundChangeRule(
        pattern="-auc",
        replacement="-og",
        weight=1.0,
        description=(
            "Adjectival/place-name suffix -auc → -og. au-monophthongization "
            "+ k→g voicing in word-final position. OW 'pennauc' → MW 'pennog'; "
            "common in MW toponym roots."
        ),
        exemplar=("Penn-auc", "Penn-og"),
        source="Morris-Jones 1913 § 21",
    ),
    SoundChangeRule(
        pattern="-niau",
        replacement="-nau",
        weight=1.0,
        description=(
            "Pluralizing/diminutive suffix -niau → -nau (yod-loss + "
            "monophthongization). Less common in toponymy than the -auc "
            "family but well-attested in the gloss corpus."
        ),
        exemplar=("Lloch-niau", "Lloch-nau"),
        source="Morris-Jones 1913 § 22",
    ),
    # --- Multi-char patterns capturing both spelling AND diphthong shifts ---
    SoundChangeRule(
        pattern="kair",
        replacement="caer",
        weight=1.0,
        description=(
            "Lexical 'kair' → 'caer' 'fortress'. Captures both the k→c "
            "spelling normalization AND the ai→ae adjustment specific to "
            "this very common toponym element ('caerwent', 'caernarfon'). "
            "Pattern is lowercase only; capitalized inputs ('Kair-...') "
            "should be lowercased before the cell is applied — same "
            "convention as every other rule in the library."
        ),
        exemplar=("kair-uent", "caer-uent"),
        source="Morris-Jones 1913 § 13",
    ),
    # --- Diphthong reflexes ---
    SoundChangeRule(
        pattern="au",
        replacement="aw",
        weight=1.0,
        description=(
            "Diphthong au → aw. OW 'maur' → MW 'mawr' 'great'; the au→aw "
            "shift is universal in MW orthography for stressed mid-word "
            "diphthongs that don't get captured by the suffix rules."
        ),
        exemplar=("maur", "mawr"),
        source="Morris-Jones 1913 § 70",
    ),
    SoundChangeRule(
        pattern="iu",
        replacement="yw",
        weight=1.0,
        description=(
            "Diphthong iu → yw. OW 'biu' → MW 'byw' 'alive'; the y in MW "
            "is a vowel here, parallel to the i in OW spelling."
        ),
        exemplar=("biu", "byw"),
        source="Morris-Jones 1913 § 75",
    ),
    SoundChangeRule(
        pattern="ui",
        replacement="wy",
        weight=1.0,
        description=(
            "Diphthong ui → wy. Visible in Welsh place-names containing "
            "the 'dwy-' element ('Dwyfor', 'Dwyfach') and any toponym with "
            "the 'wy' diphthong reflex of OW 'ui'."
        ),
        exemplar=("uir", "wyr"),
        source="Morris-Jones 1913 § 78",
    ),
    SoundChangeRule(
        pattern="oi",
        replacement="oe",
        weight=1.0,
        description=(
            "Diphthong oi → oe. The most common reflex; some lexical items "
            "took 'wy' instead, which Phase 2.4 may add as a sporadic-weight "
            "alternative."
        ),
        exemplar=("doisig", "doesig"),
        source="Morris-Jones 1913 § 81",
    ),
    # --- Single-char orthographic shifts (run last so multi-char rules
    # capture their inputs first). ---
    SoundChangeRule(
        pattern="k",
        replacement="c",
        weight=1.0,
        description=(
            "Velar /k/ orthographic shift. Old Welsh 8th–11th c. glosses "
            "use 'k' for /k/; standardized to 'c' in Middle/Modern Welsh "
            "orthography. Universal — any 'k' surviving the multi-char "
            "rules above gets normalized."
        ),
        exemplar=("kil", "cil"),
        source="Morris-Jones 1913 § 9",
    ),
    SoundChangeRule(
        pattern="ph",
        replacement="ff",
        weight=1.0,
        description=(
            "Voiceless labiodental orthographic shift. OW 'ph' → MW 'ff' — "
            "MW 'f' alone represents /v/, so voiceless /f/ is written 'ff'. "
            "Surfaces in toponymic elements like '-ffynnon' (well/spring) "
            "where OW had a 'ph' digraph."
        ),
        exemplar=("phin", "ffin"),
        source="Morris-Jones 1913 § 14",
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
    ("english", "early-modern-english", "modern-english"): EMODE_TO_MODE_RULES,
    ("welsh", "old-welsh", "modern-welsh"): OW_TO_MW_RULES,
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


# --- chain registry + iterated walker (wyrd-98cs, exposed publicly by wyrd-u728) --
#
# Maps a lexicon-language code to the rule-cell family + era-chain it
# lives in. Languages absent from this map have no phonology rules
# registered (Goidelic, Norse, Romance, etc.) — Tier-4 phonology walks
# silently no-op for them.
#
# Rule-cell ``family`` is the first element of the (language, from_era,
# to_era) tuple keys in ``_RULES`` (``english``, ``welsh``). ``chain``
# is the era-tuple in timeline order: walking from index i to j>i
# applies forward cells; j<i applies inverse cells in reverse.

_ENGLISH_CHAIN: tuple[str, ...] = (
    "old-english",
    "middle-english",
    "early-modern-english",
    "modern-english",
)
_WELSH_CHAIN: tuple[str, ...] = ("old-welsh", "modern-welsh")

FAMILY_CHAINS: dict[str, tuple[str, tuple[str, ...]]] = {
    "old-english": ("english", _ENGLISH_CHAIN),
    "middle-english": ("english", _ENGLISH_CHAIN),
    "early-modern-english": ("english", _ENGLISH_CHAIN),
    "modern-english": ("english", _ENGLISH_CHAIN),
    "old-welsh": ("welsh", _WELSH_CHAIN),
    "modern-welsh": ("welsh", _WELSH_CHAIN),
}

# Bundle / lexicon aliases that share a chain with a canonical era.
# Resolved before the chain.index lookup so the alias name itself
# doesn't need to appear in the chain tuple. ``welsh`` as a lexicon
# language code is conventionally treated as modern-welsh for era
# purposes (the lexicon mostly stores Welsh forms with this tag).
LANGUAGE_ALIASES: dict[str, str] = {
    "welsh": "modern-welsh",
}


def resolve_language(language: str) -> str:
    """Alias-resolve a lexicon language code (e.g. ``welsh`` →
    ``modern-welsh``) to its canonical chain-era name. Languages
    absent from ``LANGUAGE_ALIASES`` pass through unchanged.

    Stable public API alongside ``chain_for`` so consumers can find
    their own position in a chain (``chain.index(resolve_language(L))``)
    without reaching into ``LANGUAGE_ALIASES`` directly.
    """
    return LANGUAGE_ALIASES.get(language, language)


def chain_for(language: str) -> tuple[str, tuple[str, ...]] | None:
    """Return the rule-cell ``(family, chain)`` pair for ``language``,
    resolving lexicon-language aliases (e.g. ``welsh`` → ``modern-welsh``)
    along the way. Returns None when the language has no registered
    phonology chain — callers should treat that as 'no Tier-4 reflex
    is reachable from this language'.

    Stable public API for the chain config — consumers should not
    reach into ``FAMILY_CHAINS`` / ``LANGUAGE_ALIASES`` directly so
    future refactors of the registry shape stay internal. Pair with
    ``resolve_language`` when you need the canonical era name for
    chain-position lookup.
    """
    return FAMILY_CHAINS.get(resolve_language(language))


def rule_form(
    canonical_form: str,
    from_language: str,
    to_language: str,
) -> str | None:
    """Derive a ``to_language`` form from ``canonical_form`` (in
    ``from_language``) by iterating ``apply_rules`` across the family
    chain that owns both languages.

    Returns None when:
    - either language is absent from ``FAMILY_CHAINS``
    - the two languages live in different families (no chain bridges
      e.g. Goidelic→Brythonic)
    - both languages map to the same era (no transformation needed;
      caller should prefer the canonical_form directly)
    - the chain walk's final form equals the input (no rule fired)

    Walks forward when the target era is later in the family chain,
    inverse when the target is earlier. At each step picks the
    highest-probability candidate; downstream consumers wanting
    multiple readings would need a richer return shape (deferred
    until KenningRewind explainer surfaces multiple Tier-4 forms —
    Phase 2).
    """
    # Resolve bundle aliases ('welsh' → 'modern-welsh') before chain
    # lookup so the alias name doesn't need to appear in the chain
    # tuple itself.
    from_resolved = resolve_language(from_language)
    to_resolved = resolve_language(to_language)
    if from_resolved == to_resolved:
        return None
    from_chain_info = FAMILY_CHAINS.get(from_resolved)
    to_chain_info = FAMILY_CHAINS.get(to_resolved)
    if from_chain_info is None or to_chain_info is None:
        return None
    family, chain = from_chain_info
    # Cross-family (e.g. english → welsh) has no rules path.
    if family != to_chain_info[0]:
        return None
    i_from = chain.index(from_resolved)
    i_to = chain.index(to_resolved)
    if i_from == i_to:
        return None

    current = canonical_form
    # Note: ``apply_rules`` carries its own probability-floor restore
    # so it never returns []; the ``if not candidates`` branches below
    # are belt-and-suspenders, defensive against a future change in
    # the apply_rules API contract.
    if i_from < i_to:
        # Forward walk: apply cells (chain[i], chain[i+1]) for i in
        # [i_from, i_to). Each step picks the highest-probability
        # candidate produced by the cell.
        for i in range(i_from, i_to):
            candidates = apply_rules(current, family, chain[i], chain[i + 1], mode="forward")
            if not candidates:
                return None
            current = max(candidates, key=lambda c: c[1])[0]
    else:
        # Inverse walk: apply cells (chain[i-1], chain[i]) for i in
        # (i_from, i_to], reading them in inverse mode so the input
        # treated as the to_era form produces the from_era candidate.
        for i in range(i_from, i_to, -1):
            candidates = apply_rules(current, family, chain[i - 1], chain[i], mode="inverse")
            if not candidates:
                return None
            current = max(candidates, key=lambda c: c[1])[0]
    if current == canonical_form:
        # No rule fired across the walk — the form passed through
        # unchanged. Downstream consumers don't need a Tier-4 reflex
        # that's the same as the input; treat as no result so the
        # caller falls back to the canonical.
        return None
    return current
