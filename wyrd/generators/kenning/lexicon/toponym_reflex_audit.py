"""LLM per-form toponym-reflex audit (wyrd-1wv2) — clears the era-grid over-merge
long tail.

The cluster modern-reflex audit (``cluster_reflex_audit``) judges a modern reflex
against its cluster's *dominant sense*, which is unreliable on heavily over-merged
clusters (it false-detaches legit reflexes). This judges each distinct modern
form GENERICALLY: *"is this a plausible English place-name element reflex?"* —
keeping legit elements (``ea``, ``ait``, ``holme``, ``vale``, ``clough``) and
detaching unrelated words cluster-merged in (medical/scientific Latin, foreign
words, proper nouns, function words, abstract nouns). Per-FORM + deduped (a form
is noise everywhere or nowhere), so it sidesteps the cluster-sense problem.

A cheap surface pre-screen skips forms that are orthographically similar to one
of their cluster's English-family morphemes (likely genuine reflexes); only the
surface-dissimilar suspects are judged. A detached form is orphaned by cutting
ALL its in-cluster bridging in-edges; the detach round-trips through
``_collapses.jsonl`` (at rebuild ``apply_collapses`` deletes the edges + fresh
``cluster_cognates`` drops the orphan; live deploy = clear cognates + re-cluster).
"""

from __future__ import annotations

import difflib
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from wyrd.generators.kenning.lexicon.cluster_reflex_audit import _bridging_parents
from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK
from wyrd.generators.kenning.lexicon.descent_audit import _load_clustered_corpus

LLM_TOPONYM_REFLEX_DETACH_METHOD = "llm-toponym-reflex-detach-v1"

_MODERN = "modern-english"
_ENGLISH_FAMILY = ("old-english", "old-norse", "old-french", "middle-english")

# wyrd-xdam.5: the fine-grained Celtic languages whose cluster members also get
# judged. The bulk wiktextract ingest pulls whole Welsh/Irish/… dictionaries
# (foreign proper nouns and all) into cognate clusters that cross-link to English
# — e.g. welsh 'Macedonia'/'Libya'/'Philip' clustered with the English forms —
# polluting the era-grid the same way the medical-Latin / proper-noun noise on the
# English side does. Judging Celtic members in English-anchored clusters detaches
# that pollution while leaving real Welsh/Irish elements (which roll up to a
# branch, wyrd-xdam.4) in place.
_CELTIC_LANGS = (
    "welsh",
    "old-welsh",
    "middle-welsh",
    "modern-welsh",
    "cornish",
    "breton",
    "old-breton",
    "middle-breton",
    "irish",
    "old-irish",
    "middle-irish",
    "scottish-gaelic",
    "manx",
)

# Protective whitelist of standard Celtic place-name elements (refs: Owen for
# Welsh; Joyce & Flanagan for Irish; Watson for Scottish) — never detached
# regardless of verdict, like PROTECTED_ELEMENTS on the English side. Bare native
# form, lower-cased.
CELTIC_PROTECTED_ELEMENTS = frozenset(
    [
        # Brittonic (Welsh / Cornish / Breton)
        "aber",
        "llan",
        "caer",
        "pen",
        "tre",
        "tref",
        "nant",
        "bryn",
        "cwm",
        "glan",
        "llyn",
        "afon",
        "coed",
        "rhyd",
        "pant",
        "maes",
        "ynys",
        "porth",
        "eglwys",
        "bod",
        "din",
        "dinas",
        "moel",
        "mynydd",
        "pont",
        "pwll",
        "ros",
        "lan",
        "lann",
        "ker",
        "plou",
        "gwic",
        "carn",
        "men",
        "nans",
        # Goidelic (Irish / Scottish-Gaelic / Manx)
        "bally",
        "baile",
        "dun",
        "kil",
        "cill",
        "inver",
        "inbhear",
        "loch",
        "lough",
        "knock",
        "cnoc",
        "rath",
        "glen",
        "gleann",
        "drum",
        "droim",
        "carrick",
        "carraig",
        "slieve",
        "sliabh",
        "ben",
        "beinn",
        "strath",
        "srath",
        "kin",
        "ceann",
        "auch",
        "achadh",
        "bal",
        "ard",
        "lis",
        "lios",
        "tully",
        "tulach",
        "clon",
        "cluain",
    ]
)

# Protective whitelist of standard English place-name elements (Smith EPNS /
# Gelling & Cole / Mills). The generic judge occasionally rejects a common-word
# element as "just a common noun/adjective" (e.g. home <OE hām, law/low <OE hlāw
# 'mound', new as in Newton, spout) — never detach a known element regardless of
# the verdict. Matched on the bare modern form.
PROTECTED_ELEMENTS = frozenset(
    [
        "home",
        "town",
        "ham",
        "vale",
        "dale",
        "ness",
        "fell",
        "gill",
        "holme",
        "beck",
        "wood",
        "field",
        "ford",
        "bridge",
        "brook",
        "stone",
        "hill",
        "well",
        "mill",
        "green",
        "marsh",
        "moor",
        "clay",
        "sand",
        "stock",
        "bourne",
        "burn",
        "lake",
        "pool",
        "cliff",
        "gate",
        "street",
        "wick",
        "thorpe",
        "worth",
        "bury",
        "barrow",
        "combe",
        "cot",
        "fen",
        "holt",
        "leigh",
        "ley",
        "mere",
        "mouth",
        "stead",
        "hall",
        "church",
        "cross",
        "grove",
        "head",
        "land",
        "lea",
        "ridge",
        "rise",
        "side",
        "water",
        "weir",
        "end",
        "edge",
        "bank",
        "dell",
        "den",
        "dean",
        "knoll",
        "low",
        "law",
        "mound",
        "nook",
        "oak",
        "ash",
        "elm",
        "thorn",
        "birch",
        "willow",
        "reed",
        "rush",
        "salt",
        "shaw",
        "slade",
        "spring",
        "stow",
        "toft",
        "tye",
        "ware",
        "wong",
        "yard",
        "garth",
        "carr",
        "ing",
        "ey",
        "ea",
        "ait",
        "holm",
        "hurst",
        "hythe",
        "wath",
        "scar",
        "slack",
        "rigg",
        "cop",
        "clough",
        "gully",
        "carse",
        "nab",
        "nether",
        "over",
        "under",
        "upper",
        "lower",
        "great",
        "little",
        "new",
        "old",
        "north",
        "south",
        "east",
        "west",
        "mid",
        "fold",
        "cote",
        "booth",
        "borough",
        "burgh",
        "dyke",
        "ditch",
        "wall",
        "barn",
        "croft",
        "close",
        "meadow",
        "heath",
        "common",
        "park",
        "manor",
        "court",
        "castle",
        "tower",
        "keep",
        "port",
        "haven",
        "creek",
        "fleet",
        "moss",
        "bog",
        "mire",
        "tarn",
        "linn",
        "force",
        "foss",
        "spout",
        "howe",
        "pike",
        "rake",
        "brow",
    ]
)


@dataclass(frozen=True)
class ToponymReflexCandidate:
    """One distinct era-grid reflex form — modern-english under --scope english, a
    Celtic language (welsh/irish/…) under --scope celtic — that is NOT
    orthographically similar to any English-family morpheme in its clusters, for
    judging: a plausible place-name element (KEEP) or an unrelated cluster-merged
    word (DETACH the leaf's bridging in-edges)."""

    reflex_ref: str  # "<lang>:<form>"
    reflex_form: str
    bridging_parents: tuple[str, ...]


@dataclass(frozen=True)
class ToponymReflexVerdict:
    toponymic: bool  # plausible English place-name element reflex — KEEP
    confidence: str
    reason: str


def _norm(s: str) -> str:
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip("-")
        .strip()
    )


def _surface_similar(a: str, b: str) -> bool:
    """Cheap orthographic-similarity pre-screen (shared 3-prefix / substring /
    ratio>=0.5) — a modern form similar to one of its cluster morphemes is a
    likely genuine reflex and is auto-kept (not judged)."""
    x, y = _norm(a), _norm(b)
    if not x or not y:
        return False
    if (len(x) >= 3 and len(y) >= 3 and x[:3] == y[:3]) or x in y or y in x:
        return True
    return difflib.SequenceMatcher(None, x, y).ratio() >= 0.5


def looks_like_proper_noun(form: str) -> bool:
    """Deterministic proper-noun screen (wyrd-xdam.5): a capitalized headword whose
    normalized form is not a known place-name element. Catches the foreign
    proper-noun bulk (Macedonia, Libya, Philip, Andreas) the bulk wiktextract
    ingest pulls in, without an LLM call. Place-name elements are lower-case in the
    corpus, so a leading capital is a strong proper-noun signal; the element
    whitelists are a belt-and-braces guard against a capitalized known element."""
    f = form.strip().strip("-")
    if not f or not f[0].isupper():
        return False
    return _norm(f) not in (PROTECTED_ELEMENTS | CELTIC_PROTECTED_ELEMENTS)


def deterministic_proper_noun_verdict(
    c: ToponymReflexCandidate,
) -> ToponymReflexVerdict | None:
    """A high-confidence DETACH verdict for a candidate whose form is a clear
    proper noun (deterministic-first; no LLM). ``None`` → defer to the LLM tail."""
    if looks_like_proper_noun(c.reflex_form):
        return ToponymReflexVerdict(
            toponymic=False,
            confidence="high",
            reason="capitalized non-element headword (proper noun, deterministic)",
        )
    return None


def detect_toponym_reflex_candidates(
    conn: sqlite3.Connection,
    *,
    prescreen: bool = True,
    min_modern_members: int = 1,
    target_langs: tuple[str, ...] = (_MODERN,),
) -> list[ToponymReflexCandidate]:
    """Distinct cluster-member forms (in ``target_langs``) inside clusters that
    contain an English-family morpheme.

    Defaults reproduce the original English behaviour exactly: judge the
    ``modern-english`` members of English-family-anchored clusters. wyrd-xdam.5
    passes ``target_langs=_CELTIC_LANGS`` to instead judge the Celtic members that
    bridge into those English-anchored clusters (the foreign-proper-noun pollution).

    With ``prescreen=True`` (default) a form surface-similar to one of its cluster
    morphemes is auto-kept (likely a real reflex) — catches only the
    surface-DISSIMILAR noise cheaply. With ``prescreen=False`` EVERY form is
    judged, which also catches surface-SIMILAR noise (``vulva``~``val``,
    ``tuna``~``tūn``) that merely looks like the morpheme; the judge keeps the
    legit spelling variants. ``min_modern_members`` restricts to clusters with at
    least that many ``target_langs`` members (use 2 with ``prescreen=False`` to
    target the over-merged clusters where surface-similar noise co-occurs)."""
    by_id = _load_clustered_corpus(conn)
    clusters: dict[int, list[int]] = defaultdict(list)
    for eid, rec in by_id.items():
        clusters[rec["cognate_id"]].append(eid)

    # form -> (etymon_id, set of anchor-family morpheme forms across its clusters)
    candidates: dict[str, int] = {}
    morphemes_for: dict[str, set[str]] = defaultdict(set)
    for _cog, ids in clusters.items():
        fam = [by_id[i]["form"] for i in ids if by_id[i]["lang"] in _ENGLISH_FAMILY]
        if not fam:
            continue
        moderns = [i for i in ids if by_id[i]["lang"] in target_langs]
        if len(moderns) < min_modern_members:
            continue
        for i in moderns:
            ref = by_id[i]["ref"]
            candidates[ref] = i
            morphemes_for[ref].update(fam)

    out: list[ToponymReflexCandidate] = []
    for ref, eid in candidates.items():
        form = by_id[eid]["form"]
        if prescreen and any(_surface_similar(form, m) for m in morphemes_for[ref]):
            continue  # surface-plausible reflex of one of its morphemes — auto-keep
        parents = _bridging_parents(conn, eid, by_id)
        if not parents:
            continue
        out.append(
            ToponymReflexCandidate(
                reflex_ref=ref, reflex_form=form, bridging_parents=tuple(sorted(set(parents)))
            )
        )
    return sorted(out, key=lambda c: c.reflex_ref)


_JUDGE_SYSTEM = (
    "You are auditing modern-English words grouped into the cognate clusters of an ENGLISH "
    "PLACE-NAME (toponym) etymology lexicon. For the given modern word, decide if it is plausibly a "
    "genuine modern reflex/descendant of an English place-name ELEMENT — a word occurring in or derived "
    "from English place-names (e.g. 'vale','ness','dale','ton','ham','holme','beck','fell','rigg','ait',"
    "'ea','wick','thorpe','clough','holt'). Answer NO if it is clearly an UNRELATED word wrongly "
    "cluster-merged: a medical/anatomical or scientific Latin term, a foreign-language word, a proper noun "
    "(person, or a place outside Britain), a function word/pronoun, an acronym, or an abstract noun with no "
    "toponymic use. Judge the word itself, not surface look-alikeness. Reply ONLY with a JSON object: "
    '{"toponymic": true|false, "confidence": "high"|"medium"|"low", "reason": "<one short clause>"}.'
)


_JUDGE_SYSTEM_CELTIC = (
    "You are auditing CELTIC-language words (Welsh, Cornish, Breton, Irish, Scottish "
    "Gaelic, Manx) that the bulk dictionary import has grouped into the cognate clusters "
    "of an ENGLISH PLACE-NAME (toponym) etymology lexicon. For the given Celtic word, "
    "decide if it is plausibly a Celtic place-name ELEMENT — a word occurring in or "
    "derived from Welsh/Irish/Scottish/Cornish/Breton place-names (e.g. 'aber','llan',"
    "'caer','pen','tre','nant','bally','dun','kil','inver','glen','knock','loch'). Answer "
    "NO if it is clearly an UNRELATED word wrongly cluster-merged: a foreign-language word, "
    "a proper noun (a person, or a place outside the Celtic lands — e.g. 'Macedonia','Libya',"
    "'Philip'), a function word/pronoun, an acronym, or an abstract noun with no toponymic "
    "use. Judge the word itself, not surface look-alikeness. Reply ONLY with a JSON object: "
    '{"toponymic": true|false, "confidence": "high"|"medium"|"low", "reason": "<one short clause>"}.'
)


def _is_celtic_ref(ref: str) -> bool:
    return ref.split(":", 1)[0] in _CELTIC_LANGS


def build_judge_prompt(c: ToponymReflexCandidate) -> tuple[str, str]:
    if _is_celtic_ref(c.reflex_ref):
        user = (
            f'Celtic word: "{c.reflex_form}"\n'
            f"Is it a plausible Celtic place-name element (keep), or an unrelated word "
            f"wrongly cluster-merged (detach)?"
        )
        return _JUDGE_SYSTEM_CELTIC, user
    user = (
        f'Modern English word: "{c.reflex_form}"\n'
        f"Is it a plausible English place-name element reflex (keep), or an unrelated word "
        f"wrongly cluster-merged (detach)?"
    )
    return _JUDGE_SYSTEM, user


def parse_verdict(raw: dict) -> ToponymReflexVerdict | None:
    if not isinstance(raw, dict):
        return None
    val = raw.get("toponymic")
    if isinstance(val, str):
        val = val.strip().lower() in {"true", "yes", "y", "1"}
    if not isinstance(val, bool):
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    return ToponymReflexVerdict(
        toponymic=val, confidence=conf, reason=str(raw.get("reason", "")).strip()[:200]
    )


def audit_log_row(c: ToponymReflexCandidate, v: ToponymReflexVerdict) -> dict:
    return {
        "_type": "toponym_reflex_audit",
        "reflex_ref": c.reflex_ref,
        "bridging_parents": list(c.bridging_parents),
        "toponymic": v.toponymic,
        "confidence": v.confidence,
        "reason": v.reason,
    }


def verdict_from_log(row: dict) -> ToponymReflexVerdict | None:
    if not isinstance(row.get("toponymic"), bool):
        return None
    conf = row.get("confidence")
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    return ToponymReflexVerdict(
        toponymic=row["toponymic"], confidence=conf, reason=str(row.get("reason", ""))
    )


def detach_row(row: dict, min_confidence: str) -> dict | None:
    """``_collapses.jsonl`` edge-detach for a NON-toponymic form at/above
    ``min_confidence`` — orphan it by cutting all its in-cluster bridging parents."""
    v = verdict_from_log(row)
    if v is None or v.toponymic or CONFIDENCE_RANK[v.confidence] < CONFIDENCE_RANK[min_confidence]:
        return None
    # never detach a known place-name element, even if the judge said non-toponymic
    # (normalize so a non-lowercase/accented form still matches the whitelist).
    if _norm(row.get("reflex_ref", "").split(":", 1)[-1]) in (
        PROTECTED_ELEMENTS | CELTIC_PROTECTED_ELEMENTS
    ):
        return None
    parents = row.get("bridging_parents") or []
    if not parents:
        return None
    return {
        "_type": "collapse",
        "ref": row["reflex_ref"],
        "detach_parents": sorted(parents),
        "method": LLM_TOPONYM_REFLEX_DETACH_METHOD,
        "confidence": v.confidence,
        "reason": v.reason,
    }
