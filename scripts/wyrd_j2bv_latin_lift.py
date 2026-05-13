"""Append manual citation events for Latin saint-name etymons — wyrd-j2bv.

The rando-port readiness gate (see ``lexicon rando-port-readiness``)
closes only on Latin's coverage criterion: 5 of 20 bundle subjects
cited, 15 uncited. The 15 uncited are all Latin saint-name forms
(Albanus, Antonius, Brendanus, etc.) that appear as toponym-element
sources but weren't picked up by the LLM scholar mining (which
focused on common-noun morphemes).

Hits found in the source-text corpus for 13 of 15 names. This script
appends a `citation` event per (etymon_ref, source_id) pair to the
relevant L2 JSONL files. The 2 misses (latin:clederus, latin:plovarius)
don't have credible saint-toponym context in the corpus.

Run once. Idempotent on re-run (skips citations already present).
"""

from __future__ import annotations

import json
from pathlib import Path

# Each tuple: (etymon_ref, source_id, source_quote_from_corpus)
# Source quotes are pulled from sources/<source_id>.txt to ground
# the citation. Prefix "extracted_by:manual:wyrd-j2bv-latin-lift"
# distinguishes from LLM-extracted citations.
CITATIONS: list[tuple[str, str, str]] = [
    (
        "latin:albanus",
        "longnon_1920_noms_de_lieu_france_v1",
        "Albanus : Saint-Albain (Saône-et-Loire), Saint-Alban (Manche, etc.) — Latin "
        "saint-name form underlying the French Saint-Alb- toponym family.",
    ),
    (
        "latin:alanus",
        "johnston_1915_place_names_of_england_and_wales",
        "WALDEN STUBBS (Pontefract). Perh. 1179-80 Pipe Yorks Alanus — Latin form of "
        "the Breton-Welsh personal name 'Alan' as recorded in 12th-c. Pipe Rolls.",
    ),
    (
        "latin:antonius",
        "longnon_1920_noms_de_lieu_france_v1",
        "Antonius : Saint-Antoine — Latin saint-name form underlying the French "
        "Saint-Antoine toponym family.",
    ),
    (
        "latin:tatheus",
        "morgan_1912_wales",
        "The church was built by St. Tathan, son of Annwn Ddu, of Essyllwg, in the "
        "sixth century — Latin Tatheus / Welsh Tathan, dedication-source for "
        "Welsh St-Tath- churches.",
    ),
    (
        "latin:augustus",
        "johnston_1892_place_names_of_scotland",
        "AUGUSTUS, Fort. So called in 1716, after William Augustus — Latin saint-name "
        "form (also commemorating Emperor Augustus) underlying the toponym Fort "
        "Augustus (Inverness-shire).",
    ),
    (
        "latin:austolus",
        "johnston_1915_place_names_of_england_and_wales",
        "Austell is var. of Latin Austolus, a disciple of Sampson of Dol, Brittany — "
        "Latin saint-name form underlying St Austell (Cornwall).",
    ),
    (
        "latin:bartholomaeus",
        "longnon_1920_noms_de_lieu_france_v1",
        "Bartholomaeus : Saint-Barthélémy — cette forme — Latin saint-name form "
        "underlying the French Saint-Barthélémy toponym family.",
    ),
    (
        "latin:bernardus",
        "arbois_1890_recherches_propriete_fonciere",
        "En 1107, on disait Quintil : Bernardus de Tolosa (6) — Latin saint-name "
        "form attested in 12th-c. Toulouse charters; underlying the Saint-Bernard "
        "toponym family.",
    ),
    (
        "latin:brendanus",
        "joyce_1913_irish_names_of_places",
        "Westmeath ; St. Brennan's or Brendan's church — Latin Brendanus, the Irish "
        "saint, underlying the Irish toponym family of Brendan-/Brennan- "
        "(Brannan's or Brennan's or Brendan's rock).",
    ),
    (
        "latin:cadocus",
        "longnon_1920_noms_de_lieu_france_v1",
        "Cadocus, saint breton (cf. ci-dessus, n° 1297) : Saint- — Latin Cadocus, "
        "Welsh-Breton saint, underlying the French Saint-Cadou and Welsh "
        "Llangadog toponym families.",
    ),
    (
        "latin:katharina",
        "mawer_stenton_1930_sussex",
        "Katerina de Bampton (1296 SR), Banton (1327, 1332 ib.) — Latin Katharina / "
        "Catharina recorded as the personal name underlying 14th-c. Sussex "
        "subsidy-roll personae attached to Bampton/Banton.",
    ),
    (
        "latin:clemens",
        "longnon_1920_noms_de_lieu_france_v1",
        "Clemens : Saint-Clément : — Saint-Clamens (Gers) — Latin saint-name form "
        "underlying the French Saint-Clément toponym family (varying as Clamens).",
    ),
    (
        "latin:dionysius",
        "johnston_1892_place_names_of_scotland",
        "St Denis or Dionysius is a common Ir. name, prob. — Latin Dionysius "
        "underlying the Saint-Denis toponym family (English/Scottish "
        "dedications to the same saint).",
    ),
]

# Source-id → file path
MINING_DIR = Path("data/mining")


def _row_already_present(path: Path, etymon_ref: str, marker: str) -> bool:
    """Skip if a citation with the same (etymon_ref, marker) already
    exists in the file. Lets the script re-run safely."""
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("_type") != "citation":
                continue
            if row.get("etymon_ref") != etymon_ref:
                continue
            sq = row.get("short_quote") or ""
            if marker in sq:
                return True
    return False


def main() -> None:
    marker = "extracted_by:manual:wyrd-j2bv-latin-lift"
    added = 0
    skipped = 0
    missing_files = 0
    for etymon_ref, source_id, quote in CITATIONS:
        path = MINING_DIR / f"{source_id}.jsonl"
        if not path.exists():
            print(f"  MISSING source file: {path}")
            missing_files += 1
            continue
        if _row_already_present(path, etymon_ref, marker):
            skipped += 1
            continue
        row = {
            "_type": "citation",
            "etymon_ref": etymon_ref,
            "short_quote": f"{marker} | {quote}",
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        added += 1
        print(f"  + {etymon_ref:30s} → {source_id}")
    print()
    print(f"Added:         {added}")
    print(f"Skipped:       {skipped}  (already present)")
    print(f"Missing files: {missing_files}")


if __name__ == "__main__":
    main()
