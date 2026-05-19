"""``wyrd kenning lexicon build`` — initialize the lexicon DB and seed it from meanings.json."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _load_meanings_data
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, seed_from_meanings
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

# Synthetic source row representing the meanings.json data inherited
# from the Rando port. Authority unverified per-entry; entries flagged
# for review as scholarly sources are mined.
_RANDO_SOURCE = {
    "id": "rando-port",
    "title": "Inherited morpheme database from Rando port",
    "author": "Rando project (unknown contributors)",
    "region": "British Isles",
    "language_focus": "mixed",
    "notes": (
        "Original meanings.json ported from Rando. Per-entry authority "
        "unverified; treat as low confidence pending corroboration from "
        "scholarly sources."
    ),
}


@click.command("build")
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Where to write the SQLite DB.",
)
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
def lexicon_build(db_path: Path, meanings: Path | None) -> None:
    """Initialize the lexicon DB and seed it from meanings.json.

    Wipes any existing DB at the path. The seed data is attributed to the
    synthetic 'rando-port' source.
    """
    meanings_data = _load_meanings_data(meanings)

    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.upsert_source(**_RANDO_SOURCE)
        db.commit()
        counts = seed_from_meanings(db, meanings_data, _RANDO_SOURCE["id"])
        stats = db.stats()

    click.echo(f"Built {db_path}", err=True)
    click.echo(f"Seeded as source={_RANDO_SOURCE['id']}: {counts}", err=True)
    click.echo("Tables:", err=True)
    for table, n in stats.items():
        click.echo(f"  {table:<30} {n:>8}", err=True)


# Known public-domain volumes. Used to register a `source` row automatically
# when mining one of these files. Add more as they're targeted.
_KNOWN_SKEAT_BOOKS = {
    "skeat_1901_cambridgeshire": {
        "title": "The Place-Names of Cambridgeshire",
        "author": "Walter W. Skeat",
        "year": 1901,
        "region": "Cambridgeshire",
        "language_focus": "old-english",
    },
    "skeat_1906_bedfordshire": {
        "title": "The Place-Names of Bedfordshire",
        "author": "Walter W. Skeat",
        "year": 1906,
        "region": "Bedfordshire",
        "language_focus": "old-english",
    },
    "skeat_1911_berkshire": {
        "title": "The Place-Names of Berkshire",
        "author": "Walter W. Skeat",
        "year": 1911,
        "region": "Berkshire",
        "language_focus": "old-english",
    },
    "skeat_1913_suffolk": {
        "title": "The Place-Names of Suffolk",
        "author": "Walter W. Skeat",
        "year": 1913,
        "region": "Suffolk",
        "language_focus": "old-english",
    },
    "mawer_1920_northumberland_durham": {
        "title": "The Place-Names of Northumberland and Durham",
        "author": "Allen Mawer",
        "year": 1920,
        "region": "Northumberland and Durham",
        "language_focus": "old-english",
    },
    "mawer_1922_place_names_and_history": {
        "title": "Place-Names and History",
        "author": "Allen Mawer",
        "year": 1922,
        "region": "England",
        "language_focus": "old-english",
    },
    "ekwall_1922_lancashire": {
        "title": "The Place-Names of Lancashire",
        "author": "Eilert Ekwall",
        "year": 1922,
        "region": "Lancashire",
        "language_focus": "old-english",
    },
    "wyld_hirst_1911_lancashire": {
        "title": "The Place-Names of Lancashire: Their Origin and History",
        "author": "H.C. Wyld and T.O. Hirst",
        "year": 1911,
        "region": "Lancashire",
        "language_focus": "old-english",
    },
    "moorman_1910_west_riding_yorkshire": {
        "title": "Place-Names of the West Riding of Yorkshire",
        "author": "F.W. Moorman",
        "year": 1910,
        "region": "West Riding of Yorkshire",
        "language_focus": "old-english",
    },
    "goodall_1914_sw_yorkshire": {
        "title": "Place-Names of South-West Yorkshire",
        "author": "Armitage Goodall",
        "year": 1914,
        "region": "South-West Yorkshire",
        "language_focus": "old-english",
    },
    "duignan_1902_staffordshire": {
        "title": "Notes on Staffordshire Place-Names",
        "author": "W.H. Duignan",
        "year": 1902,
        "region": "Staffordshire",
        "language_focus": "old-english",
    },
    "harrison_1898_liverpool": {
        "title": "The Place-Names of the Liverpool District",
        "author": "Henry Harrison",
        "year": 1898,
        "region": "Lancashire",
        "language_focus": "old-english",
    },
    "johnston_1892_place_names_of_scotland": {
        "title": "Place-Names of Scotland",
        "author": "James B. Johnston",
        "year": 1892,
        "region": "Scotland",
        "language_focus": "celtic",
    },
    "johnston_1904_stirlingshire": {
        "title": "The Place-Names of Stirlingshire",
        "author": "James B. Johnston",
        "year": 1904,
        "region": "Stirlingshire",
        "language_focus": "celtic",
    },
    "johnston_1915_place_names_of_england_and_wales": {
        "title": "The Place-Names of England and Wales",
        "author": "James B. Johnston",
        "year": 1915,
        "region": "England and Wales",
        "language_focus": "old-english",
    },
    "watson_1904_ross_and_cromarty": {
        "title": "Place-Names of Ross and Cromarty",
        "author": "William J. Watson",
        "year": 1904,
        "region": "Ross and Cromarty",
        "language_focus": "celtic",
    },
    "watson_1926_celtic_place_names_of_scotland": {
        "title": "The History of the Celtic Place-Names of Scotland",
        "author": "William J. Watson",
        "year": 1926,
        "region": "Scotland",
        "language_focus": "celtic",
    },
    "macbain_1922_highlands_and_islands": {
        "title": "Place Names, Highlands and Islands of Scotland",
        "author": "Alexander Macbain",
        "year": 1922,
        "region": "Scottish Highlands and Islands",
        "language_focus": "celtic",
    },
    "gillies_1906_argyll": {
        "title": "The Place-Names of Argyll",
        "author": "H.C. Gillies",
        "year": 1906,
        "region": "Argyll",
        "language_focus": "celtic",
    },
    "morgan_1887_wales_monmouthshire": {
        "title": "Handbook of the Origin of Place-Names in Wales and Monmouthshire",
        "author": "Thomas Morgan",
        "year": 1887,
        "region": "Wales and Monmouthshire",
        "language_focus": "celtic",
    },
    "morgan_1912_wales": {
        "title": "The Place-Names of Wales",
        "author": "Thomas Morgan",
        "year": 1912,
        "region": "Wales",
        "language_focus": "celtic",
    },
    "joyce_1875_irish_names_vol1": {
        "title": "The Origin and History of Irish Names of Places, vol. 1",
        "author": "P.W. Joyce",
        "year": 1875,
        "region": "Ireland",
        "language_focus": "celtic",
    },
    "joyce_1898_irish_names_vol3": {
        "title": "The Origin and History of Irish Names of Places, vol. 3",
        "author": "P.W. Joyce",
        "year": 1898,
        "region": "Ireland",
        "language_focus": "celtic",
    },
    "joyce_1913_irish_names_of_places": {
        "title": "Irish Names of Places",
        "author": "P.W. Joyce",
        "year": 1913,
        "region": "Ireland",
        "language_focus": "celtic",
    },
    "moore_1890_isle_of_man": {
        "title": "The Surnames and Place-Names of the Isle of Man",
        "author": "A.W. Moore",
        "year": 1890,
        "region": "Isle of Man",
        "language_focus": "celtic",
    },
    "mcclure_1910_british_place_names": {
        "title": "British Place-Names in their Historical Setting",
        "author": "Edmund McClure",
        "year": 1910,
        "region": "British Isles",
        "language_focus": "old-english",
    },
    "taylor_1893_words_and_places": {
        "title": "Words and Places",
        "author": "Isaac Taylor",
        "year": 1893,
        "region": "Europe",
        "language_focus": "mixed",
    },
    "lindkvist_1912_scandinavian_place_names": {
        "title": "Middle-English Place-Names of Scandinavian Origin",
        "author": "Harald Lindkvist",
        "year": 1912,
        "region": "England (Danelaw)",
        "language_focus": "old-norse",
    },
    "zachrisson_1909_anglo_norman_influence": {
        "title": "A Contribution to the Study of Anglo-Norman Influence on English Place-Names",
        "author": "R.E. Zachrisson",
        "year": 1909,
        "region": "England",
        "language_focus": "norman-french",
    },
    "mawer_stenton_1924_introduction_to_survey": {
        "title": "Introduction to the Survey of English Place-Names",
        "author": "Allen Mawer and F.M. Stenton",
        "year": 1924,
        "region": "England",
        "language_focus": "old-english",
    },
    "mawer_1924_chief_elements": {
        "title": "The Chief Elements Used in English Place-Names",
        "author": "Allen Mawer",
        "year": 1924,
        "region": "England",
        "language_focus": "old-english",
    },
    "bannister_1916_herefordshire": {
        "title": "The Place-Names of Herefordshire",
        "author": "A.T. Bannister",
        "year": 1916,
        "region": "Herefordshire",
        "language_focus": "norman-french",
    },
    "quilgars_1906_loire_inferieure": {
        "title": "Dictionnaire topographique du département de la Loire-Inférieure",
        "author": "Henri Quilgars",
        "year": 1906,
        "region": "Loire-Inférieure (Brittany)",
        "language_focus": "celtic",
    },
}


def add_to(parent: click.Group) -> None:
    """Register ``build`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_build)
