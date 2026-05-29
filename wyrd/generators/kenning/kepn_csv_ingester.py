"""KEPN (Key to English Place-Names) county-CSV ingester (wyrd-bc5z).

KEPN's advanced-search exports a per-county CSV: one row per place with
columns ``PlaceName, Etymology, Derivation, ReferenceTitle``. This module
turns that into a self-contained L2 JSONL file the standard
``jsonl.build.build_from_jsonl`` pipeline replays into the lexicon DB —
no LLM, no form-in-body validation (the data is already curated, like the
wiktextract / Briggs paths).

Three payloads land (see DECISIONS / wyrd-bc5z design notes):

1. **Etymons + citations.** Each lexical element of a place's breakdown
   becomes an ``etymon`` (merge-on ``(canonical_form, language)`` →
   corroborates existing Mawer/Skeat/wiktextract rows) plus a ``citation``
   from the single ``kepn`` source. When the place row carries a
   non-empty ``ReferenceTitle`` (KEPN consulted external dictionaries —
   Ekwall/Mills/Watts/etc.), a SECOND citation from the ``kepn_external``
   cited-source is attached, capping any KEPN etymon at 2 witnesses (no
   per-dictionary inflation) while still passing a referenced derivation
   through the ≥2 promotion gate. The raw ReferenceTitle is preserved in
   the citation's ``context_snippet``.

2. **Town names.** Each place → a ``toponym`` (region = county supplied
   out-of-band, country = England — KEPN's CSV does not carry the county).

3. **Breakdowns.** Each place → an ``etymology_element`` carrying the
   ordered element list. Personal-name slots (KEPN's generic
   ``Personal name (X)``) hold no form in the ``Derivation`` — the actual
   name is in the ``Etymology`` prose (``'Earda's wood/clearing'``). We
   extract it, ASCII-fold it (``Æ→Ae``, ``ð/þ→th``, strip macrons), and
   match it against the Briggs personal-name etymons; a confident match
   points the element at the real name etymon (corroborating Briggs). On
   no match the slot is dropped from the breakdown and the raw name is
   recorded in ``toponym_etymology.notes`` (→ review queue).
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

SOURCE_ID = "kepn"
SOURCE_TITLE = "Key to English Place-Names (Institute for Name-Studies, University of Nottingham)"
EXTERNAL_SOURCE_ID = "kepn_external"
EXTERNAL_SOURCE_TITLE = (
    "External dictionaries aggregated by the Key to English Place-Names "
    "(Ekwall, Mills, Watts, county-survey volumes, etc.)"
)
COUNTRY = "England"

# KEPN's eight declared source languages → canonical etymon.language codes.
# 'C' (Celtic) and 'L' (Latin) are the abbreviation forms KEPN uses in the
# Derivation field; 'L' is mapped defensively (no occurrences in Rutland/
# Yorkshire, but parallel to the confirmed 'C' so a county that abbreviates
# Latin can't silently drop it — any unmapped slot is logged, never dropped).
_LANG_MAP: dict[str, str] = {
    "Old English": "old-english",
    "Old Norse": "old-norse",
    "Middle English": "middle-english",
    "Modern English": "modern-english",
    "English": "modern-english",
    "Old French": "old-french",
    "Continental Germanic": "continental-germanic",
    "Latin": "latin",
    "L": "latin",
    "Celtic": "celtic",
    "C": "celtic",
    "Unknown": "unknown",
}
# Longest-first so "Old English" wins over "English", "Modern English" over
# "English", etc. Single-letter abbreviations come last.
_LANG_LABELS = sorted(_LANG_MAP, key=len, reverse=True)
_LANGRE = re.compile(r"(?:" + "|".join(re.escape(k) for k in _LANG_LABELS) + r")$")

# A personal-name slot in the Derivation: KEPN writes a generic placeholder
# rather than a form ("Personal name (Old English)", bare "Name", or the
# folk-group "tribal-name").
_PERSONAL_NAME_RE = re.compile(r"^(?:Personal name\b|Name$|tribal-name$)")

# Possessive in the Etymology prose: the leading "'Earda's …" or "Earda's …".
_POSSESSIVE_RE = re.compile(r"([A-ZÆÐÞ][A-Za-zÆÐÞæðþāēīōūȳȝëéèïô\-]+?)['’]s\b")

_FOLD = str.maketrans(
    {
        "Æ": "Ae", "æ": "ae", "Ǣ": "Ae", "ǣ": "ae",
        "Ð": "Th", "ð": "th", "Þ": "Th", "þ": "th",
        "Ȝ": "G", "ȝ": "g", "Ø": "O", "ø": "o", "Ǫ": "O", "ǫ": "o",
        "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u", "ȳ": "y",
        "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U",
    }
)


def fold_name(s: str) -> str:
    """ASCII-fold an Old-English/Old-Norse name to a match key.

    The systematic transliteration that makes KEPN's ``Aethelstan`` ≡
    Briggs's ``Æthelstan`` (both → ``aethelstan``): expand æ/ð/þ digraphs,
    strip macrons, drop residual combining marks, lowercase. This is the
    Tier-1 matcher's key on both sides. (``english_shaping`` is NOT reused
    — it no-ops on Latin-script source languages, which OE/ON are.)
    """
    s = unicodedata.normalize("NFC", s).strip().strip("'’.­ ")
    s = re.sub(r"['’]s$", "", s)  # defensive: a possessive that slipped past extraction
    s = s.translate(_FOLD)
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s.lower()


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def _edit_distance_le1(a: str, b: str) -> bool:
    """True if Levenshtein(a, b) <= 1 (used by the Tier-2 fuzzy matcher)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b, strict=False)) == 1
    short, long = (a, b) if la < lb else (b, a)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


@dataclass
class KepnStats:
    """Per-county ingest telemetry, incl. the personal-name resolution
    report (the 'how far is our miss' measure, wyrd-bc5z decision 2)."""

    county: str = ""
    places: int = 0
    lexical_elements: int = 0
    etymons_emitted: int = 0
    citations_emitted: int = 0
    external_citations: int = 0
    whitelist_quarantined: int = 0
    quarantine_sample: list[str] = field(default_factory=list)
    unmapped_lang_slots: int = 0
    unmapped_lang_sample: list[str] = field(default_factory=list)
    # personal-name resolution
    pn_slots: int = 0
    pn_extracted: int = 0
    pn_match_t1: int = 0
    pn_match_t2: int = 0
    pn_miss: int = 0
    pn_miss_sample: list[str] = field(default_factory=list)
    # build-side (filled by ingest_kepn_csv)
    etymons_inserted: int = 0
    citations_inserted: int = 0
    citation_orphans: int = 0
    jsonl_path: Path | None = None


@dataclass
class _Element:
    form: str
    lang: str  # canonical code, or "" if unmapped
    gloss: str | None
    is_personal_name: bool
    raw_lang_label: str


def parse_derivation(deriv: str, stats: KepnStats | None = None) -> list[_Element]:
    """Parse KEPN's ``Derivation`` cell into ordered elements.

    The cell is ``"; "``-separated, but glosses themselves contain ``"; "``
    (``stōw Old English - A place; a place of assembly; a holy place.``).
    The split is disambiguated by the closed language set: a chunk starts a
    new element only when its pre-``" - "`` text ends with a known language
    label; otherwise it is a gloss continuation of the previous element and
    is re-joined. Forms are NFC-normalized. Unmapped language slots are
    counted (never silently dropped).
    """
    elements: list[_Element] = []
    for chunk in (c.strip() for c in deriv.split(";") if c.strip()):
        pre = chunk.split(" - ", 1)[0].strip()
        m = _LANGRE.search(pre)
        if not m:
            # gloss continuation → re-join onto the previous element
            if elements and elements[-1].gloss is not None:
                elements[-1].gloss += "; " + chunk
            continue
        label = m.group(0)
        form = _nfc(pre[: m.start()].strip())
        gloss = chunk.split(" - ", 1)[1].strip() if " - " in chunk else None
        is_pn = bool(_PERSONAL_NAME_RE.match(form))
        lang = _LANG_MAP.get(label, "")
        if not lang and not is_pn and stats is not None:
            stats.unmapped_lang_slots += 1
            if len(stats.unmapped_lang_sample) < 10:
                stats.unmapped_lang_sample.append(chunk[:60])
        elements.append(_Element(form, lang, gloss, is_pn, label))
    return elements


def extract_personal_name(etymology: str) -> str | None:
    """Pull the specific personal name out of the Etymology prose via the
    leading possessive (``'Earda's wood'`` → ``Earda``). None if absent."""
    m = _POSSESSIVE_RE.search(etymology or "")
    return m.group(1) if m else None


def load_name_index(briggs_jsonl: Path) -> dict[str, tuple[str, str]]:
    """Build the folded → (canonical_form, language) personal-name index
    from the Briggs L2 JSONL (etymon rows tagged male/female name)."""
    index: dict[str, tuple[str, str]] = {}
    with briggs_jsonl.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("_type") != "etymon":
                continue
            if not any(t in ("male name", "female name") for t in row.get("tags", [])):
                continue
            cf = row.get("canonical_form")
            lang = row.get("language", "old-english")
            if cf:
                index.setdefault(fold_name(cf), (cf, lang))
    return index


def match_name(
    candidate: str, name_index: dict[str, tuple[str, str]]
) -> tuple[tuple[str, str] | None, str]:
    """Resolve an extracted prose name against the Briggs index.

    Returns ((canonical, language) | None, tier) where tier ∈
    {"t1", "t2", "miss"}. Tier 2 (edit≤1) is length-gated (folded key ≥ 4)
    so short names don't fuzz into unrelated ones across ~9k entries.
    """
    fc = fold_name(candidate)
    if fc in name_index:
        return name_index[fc], "t1"
    if len(fc) >= 4:
        for key, val in name_index.items():
            if len(key) >= 4 and _edit_distance_le1(fc, key):
                return val, "t2"
    return None, "miss"


def county_from_filename(path: Path) -> str:
    """``rutland.csv`` → ``Rutland``; ``north-riding-yorkshire.csv`` →
    ``North Riding of Yorkshire``. Override via the CLI ``--county`` flag
    for any stem that doesn't title-case cleanly."""
    overrides = {
        "north-riding-yorkshire": "North Riding of Yorkshire",
        "east-riding-yorkshire": "East Riding of Yorkshire",
        "west-riding-yorkshire": "West Riding of Yorkshire",
    }
    stem = path.stem.lower()
    if stem in overrides:
        return overrides[stem]
    return " ".join(w.capitalize() for w in re.split(r"[-_]", stem))


def _ref(lang: str, form: str) -> str:
    return f"{lang}:{form}"


def emit_kepn_jsonl(
    csv_path: Path,
    jsonl_path: Path,
    *,
    county: str,
    name_index: dict[str, tuple[str, str]],
    element_whitelist: frozenset[str] | None = None,
    on_progress: Callable[[KepnStats], None] | None = None,
) -> KepnStats:
    """Parse one county CSV → self-contained L2 JSONL. Two-pass: collect all
    etymons / toponyms / breakdowns (so each etymon's ``kepn`` /
    ``kepn_external`` citations dedupe to ≤2 witnesses), then stream rows.
    Returns telemetry incl. the personal-name match/miss report."""
    stats = KepnStats(county=county, jsonl_path=jsonl_path)

    # ref → {form, lang, gloss, referenced(bool), sample_ref_title, tags}
    etymons: dict[str, dict] = {}
    # ordered place records for emit
    places: list[dict] = []

    def _note_etymon(ref: str, form: str, lang: str, gloss: str | None, referenced: bool,
                     ref_title: str, tags: list[str]) -> None:
        e = etymons.get(ref)
        if e is None:
            etymons[ref] = {
                "form": form, "lang": lang, "gloss": gloss,
                "referenced": referenced, "ref_title": ref_title if referenced else "",
                "tags": list(tags),
            }
        else:
            if gloss and not e["gloss"]:
                e["gloss"] = gloss
            if referenced and not e["referenced"]:
                e["referenced"], e["ref_title"] = True, ref_title
            for t in tags:
                if t not in e["tags"]:
                    e["tags"].append(t)

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            place = (row.get("PlaceName") or "").strip()
            etym = row.get("Etymology") or ""
            deriv = row.get("Derivation") or ""
            ref_title = (row.get("ReferenceTitle") or "").strip()
            if not place:
                continue
            stats.places += 1
            referenced = bool(ref_title)
            elements = parse_derivation(deriv, stats)
            ordered_refs: list[str] = []
            unresolved_names: list[str] = []
            pn_seen = False
            for el in elements:
                if el.is_personal_name:
                    pn_seen = True
                    continue  # resolved below from prose, once per place
                if not el.form or not el.lang:
                    continue
                if element_whitelist is not None and _nfc(el.form) not in element_whitelist:
                    stats.whitelist_quarantined += 1
                    if len(stats.quarantine_sample) < 12:
                        stats.quarantine_sample.append(f"{el.form!r}[{el.raw_lang_label}]")
                    continue
                stats.lexical_elements += 1
                ref = _ref(el.lang, el.form)
                _note_etymon(ref, el.form, el.lang, el.gloss, referenced, ref_title, [])
                ordered_refs.append(ref)

            if pn_seen:
                stats.pn_slots += 1
                cand = extract_personal_name(etym)
                if cand:
                    stats.pn_extracted += 1
                    matched, tier = match_name(cand, name_index)
                    if tier == "t1":
                        stats.pn_match_t1 += 1
                    elif tier == "t2":
                        stats.pn_match_t2 += 1
                    else:
                        stats.pn_miss += 1
                        if len(stats.pn_miss_sample) < 15:
                            stats.pn_miss_sample.append(cand)
                    if matched:
                        cf, lang = matched
                        ref = _ref(lang, cf)
                        # corroborate the Briggs name etymon; carry its name tag
                        _note_etymon(ref, cf, lang, None, referenced, ref_title, [])
                        ordered_refs.append(ref)
                    else:
                        unresolved_names.append(cand)
                else:
                    unresolved_names.append("(unextracted personal name)")

            notes_parts = []
            quoted = re.search(r"'([^']+)'", etym)
            if quoted:
                notes_parts.append(quoted.group(1).strip())
            if unresolved_names:
                notes_parts.append("unresolved personal name(s): " + ", ".join(unresolved_names))
            places.append({
                "place": place,
                "notes": "; ".join(notes_parts) or None,
                "refs": ordered_refs,
            })
            if on_progress is not None and stats.places % 250 == 0:
                on_progress(stats)

    # ---- emit ----
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as sink:
        _w(sink, {"_type": "source", "ref": SOURCE_ID, "title": SOURCE_TITLE,
                  "notes": "Generated by `wyrd kenning lexicon ingest-kepn`."})
        _w(sink, {"_type": "cited_source", "ref": EXTERNAL_SOURCE_ID, "title": EXTERNAL_SOURCE_TITLE,
                  "notes": "External dictionaries KEPN consulted (per-place ReferenceTitle)."})
        for ref, e in etymons.items():
            row: dict = {"_type": "etymon", "ref": ref,
                         "canonical_form": e["form"], "language": e["lang"]}
            if e["gloss"]:
                row["glosses"] = [e["gloss"]]
            if e["tags"]:
                row["tags"] = e["tags"]
            _w(sink, row)
            stats.etymons_emitted += 1
            # kepn primary citation (always, one per etymon)
            _w(sink, {"_type": "citation", "etymon_ref": ref})
            stats.citations_emitted += 1
            # kepn_external (only when some referencing place carried refs; one per etymon)
            if e["referenced"]:
                _w(sink, {"_type": "citation", "etymon_ref": ref,
                          "source_id": EXTERNAL_SOURCE_ID,
                          "context_snippet": e["ref_title"][:500]})
                stats.external_citations += 1
        for p in places:
            tref = f"{p['place']}@{county}"
            _w(sink, {"_type": "toponym", "ref": tref, "modern_name": p["place"],
                      "country": COUNTRY, "region": county})
            elrows = [{"ordinal": i + 1, "etymon_ref": r} for i, r in enumerate(p["refs"])]
            ee: dict = {"_type": "etymology_element", "toponym_ref": tref, "elements": elrows}
            if p["notes"]:
                ee["notes"] = p["notes"]
            _w(sink, ee)
    if on_progress is not None:
        on_progress(stats)
    return stats


def _w(sink: TextIO, row: dict) -> None:
    sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def ingest_kepn_csv(
    db,
    csv_path: Path,
    *,
    county: str | None = None,
    jsonl_path: Path | None = None,
    briggs_jsonl: Path | None = None,
    element_whitelist_path: Path | None = None,
    on_progress: Callable[[KepnStats], None] | None = None,
) -> KepnStats:
    """Emit one county CSV → JSONL, then replay into ``db`` via the shared
    ``build_from_jsonl`` pipeline. The JSONL is the canonical artifact (DB
    rebuildable from it). ``db`` is a LexiconDB (needs ``.conn``)."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl

    county = county or county_from_filename(csv_path)
    if jsonl_path is None:
        jsonl_path = Path("data/mining") / f"kepn_{csv_path.stem}.jsonl"
    if briggs_jsonl is None:
        briggs_jsonl = Path("data/mining") / "briggs_2024_personal_names_index.jsonl"
    name_index = load_name_index(briggs_jsonl) if briggs_jsonl.exists() else {}
    whitelist = None
    if element_whitelist_path and element_whitelist_path.exists():
        whitelist = frozenset(
            _nfc(ln) for ln in element_whitelist_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        )
    stats = emit_kepn_jsonl(
        csv_path, jsonl_path, county=county, name_index=name_index,
        element_whitelist=whitelist, on_progress=on_progress,
    )
    counts = build_from_jsonl(db.conn, [jsonl_path])
    stats.etymons_inserted = counts.get("etymon", 0)
    stats.citations_inserted = counts.get("citation", 0)
    stats.citation_orphans = counts.get("citation_orphans", 0)
    return stats
