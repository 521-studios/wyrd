"""Phonological bridging for Old Norse place-name forms.

Looks up each ON etymon's lowercased canonical_form in the
``_ON_PHONOLOGICAL_BRIDGES`` table (modernized/Anglicized spelling
→ scholarly orthography with macrons, þ/ð, etc.) and links matches
to the canonical etymon via ``merged_into_id``. The shared engine
in ``bridges._common._bridge_same_language_phonological`` runs the
table; this module only supplies the ON lookup table for Old Norse
loanwords in Northern English place names.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.bridges._common import _bridge_same_language_phonological
from wyrd.generators.kenning.lexicon.db import LexiconDB

_ON_PHONOLOGICAL_BRIDGES: dict[str, str] = {
    # -býr / settlement, farm
    "by": "býr",
    "byr": "býr",
    # -hólmr / island, holm — bridge value is the macron-stripped 'holmr'
    # since hólmr itself is not in the corpus as a canonical form.
    "holm": "holmr",
    # -kirkja / church (Norse loanword underlying kirk)
    "kirk": "kirkja",
    # -dalr / valley
    "dal": "dalr",
    "dale": "dalr",
    # -garðr / yard, enclosure
    "gardr": "garðr",
    "garth": "garðr",
    # -gríss / pig
    "griss": "gríss",
    # -skógr / wood, forest
    "skogr": "skógr",
    "skégr": "skógr",  # OCR variant
    # -þorp / village (relies on redirect-follow: þorp tombstoned to thorp)
    "thorpe": "þorp",
    "torp": "þorp",
    # -þveit / clearing (relies on redirect-follow: þveit tombstoned to thveit)
    "thwaite": "þveit",
    "thwait": "þveit",
    # -tún / enclosure (Norse cognate of OE tūn)
    "tun": "tún",
    # -vík / bay, inlet
    "vik": "vík",
    # -dýr / animal, deer
    "dyr": "dýr",
    # -hestr / horse
    "hest": "hestr",
    # -kjarr / brushwood, copse
    "kiarr": "kjarr",
    # -krókr / hook, bend
    "krokr": "krókr",
    # -mór / moor
    "mor": "mór",
    # -mýrr / bog
    "myrr": "mýrr",
    # -norðr / north
    "nord": "norðr",
    # -rauðr / red
    "rauthr": "rauðr",
    "raudr": "rauðr",
    # -sauðr / sheep
    "saudr": "sauðr",
    # -vágr / wave, sea-creek
    "vagr": "vágr",
    # -vagn / wagon
    "vogn": "vagn",
    # -vǫllr / field
    "vollr": "vǫllr",
    # -blár / blue
    "bla": "blár",
    # -hǫgg / cut, blow
    "hogg": "hǫgg",
    # -buskr / bush
    "buski": "buskr",
    # -veiðr / hunting (also -veiði)
    "veidi": "veiðr",
    # -flatr / flat
    "flad": "flatr",
    # -hagi / pasture, enclosed grazing
    "hain": "hagi",
}


def bridge_phonological_on(db: LexiconDB, *, apply: bool = False) -> dict:
    """Bridge ON place-name etymons to their Wiktionary canonical
    equivalents via a hand-curated mapping table.

    Place-name dictionaries write Anglicized / modernized ON forms
    (`by`, `holm`, `dale`, `thwaite`, `kirk`, `gardr`); Wiktionary
    uses scholarly orthography with acutes, þ/ð, -r endings, ǫ
    (`býr`, `hólmr`, `dalr`, `þveit`, `kirkja`, `garðr`). normalize-ocr
    handles macron-strip OCR variants but not the Anglicization of
    -r endings, þ → th, ð → d, ǫ → o, etc. This pass uses
    `_ON_PHONOLOGICAL_BRIDGES` to merge known pairs via merged_into_id
    (D22 non-destructive shape) so the redirect-aware cluster-cognates
    pass rolls the place-name etymons up into the Wiktionary cognate
    clusters.

    Returns:
      - examined: total canonical ON etymons examined
      - bridged: count that found a phonological-bridge target
      - unmatched: count with no entry in the phonological table
      - missing_target: count where the table named a target but no
        ON etymon exists with that canonical form (operator should
        add the target via mining or extend the table)
      - rows_written: actual UPDATE row count when apply=True
    """
    return _bridge_same_language_phonological(
        db, language="old-norse", table=_ON_PHONOLOGICAL_BRIDGES, apply=apply
    )
