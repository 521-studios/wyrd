"""Phonological bridging for Old English place-name forms.

Looks up each OE etymon's lowercased canonical_form in the
``_OE_PHONOLOGICAL_BRIDGES`` table (modernized/Anglicized
spelling → scholarly orthography with macrons, æ, þ/ð, etc.) and
links matches to the canonical etymon via ``merged_into_id``. The
shared engine in ``bridges._common._bridge_same_language_phonological``
runs the table; this module only supplies the OE rule set.

Used alongside ``bridge_generic_language`` (cross-language exact-form
lookups) and ``bridge_celtic_forms`` (Anglicized Celtic substrate) as
parallel passes that each close a different slice of the cross-corpus
consensus gap on Old English morphemes.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.bridges._common import _bridge_same_language_phonological
from wyrd.generators.kenning.lexicon.db import LexiconDB

_OE_PHONOLOGICAL_BRIDGES: dict[str, str] = {
    # -tūn / settlement family
    "ton": "tūn",
    "tun": "tūn",
    "tone": "tūn",
    # -lēah / clearing family
    "lea": "lēah",
    "leah": "lēah",
    "leak": "lēah",  # OCR / spelling variant
    "ley": "lēah",
    "ly": "lēah",
    # -cot / -cote (cottage)
    "cote": "cot",
    "cotum": "cot",  # dative-pl
    # -burh / fortified-place family
    "burgh": "burh",
    "bury": "burh",
    "byrig": "burh",  # dative-sg of burh
    # -dæl / valley
    "dale": "dæl",
    # -heall / hall
    "hall": "heall",
    # -ieg / island. Wiktionary's headword is `īeg` but the macron-stripped
    # `ieg` is what the OCR-normalize pass treats as canonical; `īeg`
    # itself is not in the corpus. Snap value to the live form.
    "ey": "ieg",
    "eg": "ieg",
    # -burna / stream
    "burn": "burna",
    "burne": "burna",
    # -healh / nook
    "hale": "healh",
    "halh": "healh",
    # -nīwe / new
    "new": "nīwe",
    # -ōra / shore, edge
    "ore": "ōra",
    # -wudu / wood
    "wood": "wudu",
    "wode": "wudu",
    # -brād / broad
    "brade": "brād",
    # -brycg / bridge
    "bridge": "brycg",
    # -stān / stone
    "stone": "stān",
    # -hyll / hill
    "hill": "hyll",
    # -hyrst / wooded hill
    "hurst": "hyrst",
    # -hlāw / mound
    "low": "hlāw",
    # -mos / moss/marsh
    "moss": "mos",
    # -pōl / pool — bridge value snapped to 'pol' (the live canonical;
    # 'pōl' is itself a tombstone merged into 'pol' by normalize-ocr).
    "pool": "pol",
    # -sealh / willow
    "salh": "sealh",
    # -stede / place
    "stead": "stede",
    # -wella / spring
    "well": "wella",
    # -ing / -ingas (people-of suffix; place-name dicts often drop hyphen)
    "ing": "-ing",
    "ingas": "-ingas",
    # æcer / acre
    "acre": "æcer",
    # æsc / ash
    "ash": "æsc",
    # -hām / homestead
    "ham": "hām",
    # -wella / spring (additional inflected/spelling variants beyond 'well')
    "wella": "welle",
    # -ing-family additions: scholar-spelled variants of the people-of suffix
    "inga": "ing",
    # -cipp / log
    "cippa": "cipp",
    # -hār / grey, hoary
    "hāran": "hār",
    # -clopp / lump, hill
    "cloppa": "clopp",
    # -ticcen / kid (young goat)
    "ticce": "ticcen",
    # -cirice / church (kirk is the Northumbrian / Norse-influenced form)
    "kirk": "cirice",
    # -healh / nook (hala is a common scholarly variant)
    "hala": "healh",
    # -scylfe / shelf, ledge
    "scelf": "scylfe",
    # -ieg / island (alternative scholarly spelling)
    "ēg": "ieg",
    # -hara / hare
    "hare": "hara",
    # -hæsel / hazel (relies on redirect-follow to canonical haesel)
    "hasel": "hæsel",
    # -hlīep / leap (Hartlepool's "harts-leap" element)
    "hlyp": "hlīep",
    # -lacu / stream (lech is also occasionally read as 'lece' = bog;
    # the place-name reading is dominantly the water sense, hence lacu)
    "lech": "lacu",
}


def bridge_phonological_oe(db: LexiconDB, *, apply: bool = False) -> dict:
    """Bridge OE place-name etymons to their Wiktionary canonical
    equivalents via a hand-curated mapping table.

    Place-name dictionaries write modernized OE forms (`ton`, `lea`,
    `burgh`, `dale`); Wiktionary uses scholarly orthography (`tūn`,
    `lēah`, `burh`, `dæl`). normalize-ocr handles macron-strip OCR
    variants but not vowel-weakening / silent-e / gh-spelling shifts.
    This pass uses `_OE_PHONOLOGICAL_BRIDGES` to merge known pairs
    via merged_into_id (D22 non-destructive shape) so the redirect-aware
    cluster-cognates pass rolls the place-name etymons up into the
    Wiktionary cognate clusters.

    Returns:
      - examined: total canonical OE etymons examined
      - bridged: count that found a phonological-bridge target
      - unmatched: count with no entry in the phonological table
      - missing_target: count where the table named a target but no
        OE etymon exists with that canonical form (operator should
        add the target via mining or extend the table)
      - rows_written: actual UPDATE row count when apply=True
    """
    return _bridge_same_language_phonological(
        db, language="old-english", table=_OE_PHONOLOGICAL_BRIDGES, apply=apply
    )
