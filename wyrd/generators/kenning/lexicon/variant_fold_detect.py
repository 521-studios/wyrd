"""Same-language variant-FOLD candidate detection (empty-grid root fix).

The cross-language sibling of ``reflex_link_detect`` (wyrd-rogd.15). Where that
LINKS cross-era reflexes (keeping both), this finds SAME-language path-dependent
DUPLICATES — a BARREN etymon (an anglicized / mis-spelled / accent-variant form
with no cognate cluster: OE ``hill`` id 4632179, 0 mates, gloss just "hill") and
the RICH proper lemma it duplicates (OE ``hyll``, 42 cluster mates, descriptive
gloss) — that are NOT connected by any descent edge. Generation resolves the
morpheme to the barren duplicate, so its era-grid comes up empty even though the
reflexes sit on the unlinked lemma.

Signal: SAME language + a shared gloss CONTENT TOKEN + (form similarity OR a
shared consonant skeleton, the vowel-shift escape hatch) + one side barren
(cluster ≤ ``barren_max``) and the other rich (cluster ≥ ``lemma_min``) + no
existing descent edge. The FOLD op (``enrichment.apply_collapses`` ``into``)
tombstones the barren into the lemma (``merged_into_id``), so the family rollup
unifies them onto one morpheme_id and the grid resolves to the rich reflexes.
"""

from __future__ import annotations

import difflib
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK

LLM_VARIANT_FOLD_METHOD = "llm-variant-fold-v1"
LLM_VARIANT_REJECT_METHOD = "llm-variant-rejected-v1"


@dataclass(frozen=True)
class VariantFoldCandidate:
    """A same-language duplicate for the LLM to judge: is ``barren`` the same
    morpheme as ``lemma`` (spelling/accent variant → FOLD), or a homograph?"""

    barren_ref: str  # the path-dependent duplicate to fold away, "<language>:<form>"
    lemma_ref: str  # the rich proper lemma to fold INTO
    barren_form: str
    lemma_form: str
    shared_glosses: tuple[str, ...]  # barren glosses sharing a content token with the lemma
    similarity: float
    lemma_cluster_size: int
    lemma_glosses: tuple[str, ...] = ()  # the rich lemma's own glosses (for the judge prompt)


def _bare(form: str) -> str:
    """Position/case-insensitive comparison key — dashes are a link, never part
    of the morpheme (wyrd-rogd.10)."""
    return form.strip().strip("-*").lower()


def _consonant_skeleton(form: str) -> str:
    """Diacritic-stripped consonant skeleton — the spine that survives the
    English Great-Vowel-Shift spellings (wood↔wudu, hūs↔house, dūn↔down) whose
    difflib ratio (~0.5) sinks below the form-similarity floor even though they
    are the same morpheme. Used as an ALTERNATIVE match path *inside a shared
    gloss-token group only*, so the loose key can't over-reach across meanings;
    the LLM still adjudicates each pair."""
    bare = _bare(form)
    bare = "".join(
        ch for ch in unicodedata.normalize("NFD", bare) if unicodedata.category(ch) != "Mn"
    )
    # æ/œ/ø are base vowel letters (not NFD-decomposable diacritics), common in
    # Old English / Old Norse — exclude them too or they'd count as consonants
    # and block valid vowel-shift skeletons.
    return "".join(ch for ch in bare if ch.isalpha() and ch not in "aeiouyæœø")


# Common-gloss stopwords: function words + generic dictionary-definition framing
# that would over-group on token match without signalling shared meaning.
#
# Onomastic ENTRY-TYPE framing is the load-bearing addition (wyrd-21p8): a
# bare name gloss like "a feminine personal name" carries no DISTINGUISHING
# meaning — only the name's class — so its only surviving tokens (feminine,
# personal) group every name of that class, pairing genuinely DISTINCT names
# (Ede≠Eve, Godelin≠Godling) by surface similarity alone. The judge then has no
# gloss signal to tell them apart and has mis-folded distinct names (Ede→Eve,
# committed at medium confidence). These class words are NEVER a morpheme's
# meaning, so stripping them is pure precision gain. Toponymic CONTENT
# (hill/place/river/tree/…) is the opposite — the real meaning — and is kept.
_GLOSS_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "used",
        "name",
        "word",
        "form",
        "term",
        "sense",
        "esp",
        "any",
        "one",
        "who",
        "which",
        "whose",
        "been",
        "being",
        "into",
        "out",
        "off",
        "per",
        "via",
        "etc",
        "cf",
        "see",
        "also",
        "have",
        "has",
        "had",
        "was",
        "were",
        "are",
        "not",
        "but",
        "its",
        "his",
        "her",
        "their",
        "our",
        "your",
        "they",
        "them",
        # onomastic entry-type framing (name CLASS, not the name's meaning)
        "personal",
        "feminine",
        "masculine",
        "saint",
        "diminutive",
        "patronymic",
        "matronymic",
        "hypocoristic",
        "cognomen",
        "byname",
        "epithet",
        "surname",
        "forename",
        "nickname",
    ]
)


def _gloss_tokens(gloss: str) -> set[str]:
    """Content tokens of a gloss for loose same-meaning grouping. A bare gloss
    ('stone') and a descriptive one ('A stone, stone, rock.') share the token
    'stone', so a barren duplicate groups with its rich descriptive lemma even
    when neither gloss matches the other verbatim (the exact-gloss miss that
    left -stone / -house empty). Alpha tokens length ≥ 3, minus stopwords."""
    out: set[str] = set()
    tok = ""
    for ch in gloss.lower():
        if ch.isalpha():
            tok += ch
        else:
            if len(tok) >= 3 and tok not in _GLOSS_STOPWORDS:
                out.add(tok)
            tok = ""
    if len(tok) >= 3 and tok not in _GLOSS_STOPWORDS:
        out.add(tok)
    return out


def _index_gloss_rows(rows: list[sqlite3.Row]) -> tuple[dict[int, dict], dict[str, set[int]]]:
    """Index the etymon×gloss rows into per-etymon records keyed by etymon id
    (language, form, bare form, consonant skeleton, cluster id, glosses, gloss
    content tokens) plus an inverted content-token -> etymon-ids map. Bare form
    and skeleton are precomputed once per etymon so the later barren×rich loop
    doesn't re-normalise (incl. unicodedata work) the same forms many times."""
    by_id: dict[int, dict] = {}
    by_token: dict[str, set[int]] = defaultdict(set)  # content token -> etymon ids
    for r in rows:
        eid = r["id"]
        rec = by_id.get(eid)
        if rec is None:
            # Don't use dict.setdefault with a literal default — it would build a
            # throwaway dict (+ two sets) on every row, including the many
            # duplicate-id gloss rows.
            rec = by_id[eid] = {
                "lang": r["language"],
                "form": r["canonical_form"],
                "bare": _bare(r["canonical_form"]),
                "skel": _consonant_skeleton(r["canonical_form"]),
                "clu": r["cognate_id"],
                "gl": set(),
                "tok": set(),
            }
        rec["gl"].add(r["gloss"])
        toks = _gloss_tokens(r["gloss"])
        rec["tok"].update(toks)
        for t in toks:
            by_token[t].add(eid)
    return by_id, by_token


def _cluster_sizes(conn: sqlite3.Connection) -> dict[int, int]:
    """Live (non-tombstoned) cognate-cluster size per cognate_id — the richness
    signal a rich lemma has cluster mates a barren duplicate lacks."""
    return {
        r["cognate_id"]: r["n"]
        for r in conn.execute(
            "SELECT cognate_id, count(*) n FROM etymon "
            "WHERE cognate_id IS NOT NULL AND merged_into_id IS NULL GROUP BY cognate_id"
        )
    }


def _descent_edge_set(conn: sqlite3.Connection) -> set[tuple[int, int]]:
    """All (parent_id, child_id) descent edges — a candidate pair already joined
    by an edge is a known relationship, not a path-dependent duplicate."""
    return {(p, c) for p, c in conn.execute("SELECT parent_id, child_id FROM etymon_descent")}


def _admit_fold_pair(
    brec: dict,
    rrec: dict,
    matcher: difflib.SequenceMatcher,
    skb: str,
    min_similarity: float,
    edges: set[tuple[int, int]],
    bi: int,
    ri: int,
) -> float | None:
    """Return the difflib similarity if (barren ``bi``, rich ``ri``) is an
    admissible same-language fold pair, else None. Admit on form similarity ≥
    ``min_similarity`` OR a shared ≥2-consonant skeleton (the vowel-shift escape
    hatch — wood↔wudu); reject cross-language pairs (cross-language is a LINK,
    wyrd-rogd.15), identical/empty forms, and pairs already joined by a descent
    edge."""
    if brec["lang"] != rrec["lang"]:
        return None
    fr = rrec["bare"]
    if not fr or brec["bare"] == fr:
        return None
    # SequenceMatcher front-loads work on seq a (set once per barren by the
    # caller); set_seq2 only re-analyses the cheap side.
    matcher.set_seq2(fr)
    sim = matcher.ratio()
    skeleton_match = len(skb) >= 2 and skb == rrec["skel"]
    if sim < min_similarity and not skeleton_match:
        return None
    if (bi, ri) in edges or (ri, bi) in edges:
        return None
    return sim


def _better_fold_target(
    cur: tuple[int, float] | None,
    by_id: dict[int, dict],
    ri: int,
    rrec: dict,
    sim: float,
) -> bool:
    """Total-order tie-break: a new (rich ``ri``, ``sim``) beats the current best
    for a barren when (richest cluster, highest similarity, LOWEST etymon id)
    ranks higher. Lowest-id is the final discriminator so the chosen fold target
    is reproducible across runs/operators — etymon ids are content-keyed on
    (canonical_form, language), so there's no set/SQL-iteration-order dependence.
    Reads the per-record ``richness`` precomputed by the caller."""
    if cur is None:
        return True
    return (rrec["richness"], sim, -ri) > (by_id[cur[0]]["richness"], cur[1], -cur[0])


def _fold_targets_for_token(
    ids: set[int],
    by_id: dict[int, dict],
    edges: set[tuple[int, int]],
    best: dict[int, tuple[int, float]],
    *,
    min_similarity: float,
    lemma_min: int,
    barren_max: int,
) -> None:
    """Pair the barren×rich etymons sharing one gloss content token, updating
    ``best`` (barren id -> best (lemma id, sim)) in place. Reads the per-record
    ``richness`` precomputed by the caller. The rich lemma set per token is tiny
    (a few well-attested forms), so even the huge common-token groups (hill /
    stone / house — the very cases that matter) are covered without an O(n²)
    blow-up."""
    rich = [(i, by_id[i]) for i in ids if by_id[i]["richness"] >= lemma_min]
    barren = [(i, by_id[i]) for i in ids if by_id[i]["richness"] <= barren_max]
    if not rich or not barren:
        return
    for bi, brec in barren:
        fb = brec["bare"]
        if not fb:
            continue
        skb = brec["skel"]
        matcher = difflib.SequenceMatcher(None, fb)
        for ri, rrec in rich:
            sim = _admit_fold_pair(brec, rrec, matcher, skb, min_similarity, edges, bi, ri)
            if sim is None:
                continue
            if _better_fold_target(best.get(bi), by_id, ri, rrec, sim):
                best[bi] = (ri, sim)


def _build_fold_candidates(
    best: dict[int, tuple[int, float]],
    by_id: dict[int, dict],
    cluster_size: dict[int, int],
) -> list[VariantFoldCandidate]:
    """Assemble the sorted candidate list from the best barren->lemma picks. The
    shared glosses (barren glosses sharing ≥1 content token with the lemma) are
    the meaning evidence behind the grouping, shown to the judge."""
    out: list[VariantFoldCandidate] = []
    for barren, (lemma, sim) in best.items():
        eb, el = by_id[barren], by_id[lemma]
        shared = tuple(sorted(g for g in eb["gl"] if _gloss_tokens(g) & el["tok"]))
        out.append(
            VariantFoldCandidate(
                barren_ref=f"{eb['lang']}:{eb['form']}",
                lemma_ref=f"{el['lang']}:{el['form']}",
                barren_form=eb["form"],
                lemma_form=el["form"],
                shared_glosses=shared or tuple(sorted(eb["gl"])),
                similarity=round(sim, 3),
                lemma_cluster_size=cluster_size.get(el["clu"], 0),
                lemma_glosses=tuple(sorted(el["gl"]))[:4],
            )
        )
    out.sort(key=lambda c: (c.lemma_ref, c.barren_ref))
    return out


def detect_variant_fold_candidates(
    conn: sqlite3.Connection,
    *,
    min_similarity: float = 0.65,
    barren_max: int = 1,
    lemma_min: int = 5,
    max_per_gloss: int = 600,
) -> list[VariantFoldCandidate]:
    """Same-language barren↔rich variant-fold candidates: SAME-language etymon
    pairs sharing a gloss CONTENT TOKEN, admitted on form similarity (difflib ≥
    ``min_similarity`` on the dash/star-stripped forms) OR a shared ≥2-consonant
    skeleton (the vowel-shift escape hatch — wood↔wudu), one barren (cognate
    cluster ≤ ``barren_max``) and one rich (cluster ≥ ``lemma_min``), not already
    connected by a descent edge. One best lemma per barren (richest cluster, then
    highest similarity, then lowest etymon id) — a TOTAL order, so the chosen
    fold target is reproducible across runs/operators."""
    rows = conn.execute(
        "SELECT e.id, e.language, e.canonical_form, e.cognate_id, g.gloss "
        "FROM etymon e JOIN etymon_gloss g ON g.etymon_id = e.id "
        "WHERE e.merged_into_id IS NULL ORDER BY e.id"
    ).fetchall()
    by_id, by_token = _index_gloss_rows(rows)
    cluster_size = _cluster_sizes(conn)
    edges = _descent_edge_set(conn)

    # Precompute each etymon's richness (live cluster size — the rich-vs-barren
    # signal) once, so the nested barren×rich loop reads a record field instead
    # of recomputing the lookup on every pair comparison.
    for rec in by_id.values():
        rec["richness"] = cluster_size.get(rec["clu"], 0) if rec["clu"] is not None else 0

    # candidates per barren id -> best (lemma id, sim). Group by a shared gloss
    # CONTENT TOKEN (not the whole gloss string), then pair BARREN×RICH per token
    # (see _fold_targets_for_token). Token grouping is what lets a bare barren
    # gloss ('stone') reach a descriptive lemma gloss ('A stone, stone, rock.');
    # ``max_per_gloss`` only guards a pathologic generic-token group.
    best: dict[int, tuple[int, float]] = {}
    for ids in by_token.values():
        if len(ids) < 2 or len(ids) > max_per_gloss:
            continue
        _fold_targets_for_token(
            ids,
            by_id,
            edges,
            best,
            min_similarity=min_similarity,
            lemma_min=lemma_min,
            barren_max=barren_max,
        )
    return _build_fold_candidates(best, by_id, cluster_size)


@dataclass(frozen=True)
class VariantFoldVerdict:
    same_morpheme: bool  # barren is the SAME morpheme as lemma (spelling/accent variant)
    confidence: str  # "high" | "medium" | "low"
    reason: str


_FOLD_JUDGE_SYSTEM = (
    "You are a historical-linguistics editor de-duplicating a place-name "
    "etymology lexicon. You are given two head-words in the SAME language with "
    "the same meaning: a SPARSE form (likely an anglicized spelling, accent or "
    "OCR variant, e.g. 'hill'/'stán'/'stdn') and a RICH canonical form (the "
    "well-attested lemma, e.g. 'hyll'/'stān'). Decide whether the sparse form is "
    "just a SPELLING/ACCENT VARIANT of the same morpheme (→ they should be folded "
    "into one). Answer NO if they are genuinely DIFFERENT morphemes that happen "
    "to share a generic gloss (homographs — e.g. 'hill' the eminence vs 'kiln'), "
    "or different words. Judge by sound/spelling closeness AND meaning. Reply "
    "ONLY with a JSON object: "
    '{"same_morpheme": true|false, "confidence": "high"|"medium"|"low", '
    '"reason": "<one short clause>"}.'
)


def build_fold_judge_prompt(c: VariantFoldCandidate) -> tuple[str, str]:
    """Return (system, user) prompts for judging a variant-fold candidate."""
    lang = c.barren_ref.split(":", 1)[0]
    user = (
        f"Language: {lang}\n"
        f'Sparse form "{c.barren_form}" — gloss(es): {"; ".join(c.shared_glosses) or "(none)"}\n'
        f'Rich lemma  "{c.lemma_form}" ({c.lemma_cluster_size} cluster mates) — '
        f"gloss(es): {'; '.join(c.lemma_glosses) or '(none)'}\n\n"
        f'Is "{c.barren_form}" just a spelling/accent variant of the same '
        f'morpheme as "{c.lemma_form}"?'
    )
    return _FOLD_JUDGE_SYSTEM, user


def parse_fold_verdict(raw: dict) -> VariantFoldVerdict | None:
    """Coerce an LLM JSON response into a :class:`VariantFoldVerdict`, or None if
    unusable (the caller skips rather than crashing the run)."""
    if not isinstance(raw, dict):
        return None
    val = raw.get("same_morpheme")
    if isinstance(val, str):
        val = val.strip().lower() in {"true", "yes", "y", "1"}
    if not isinstance(val, bool):
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    reason = str(raw.get("reason", "")).strip()[:200]
    return VariantFoldVerdict(same_morpheme=val, confidence=conf, reason=reason)


def fold_verdict_to_row(
    c: VariantFoldCandidate, v: VariantFoldVerdict, min_confidence: str
) -> dict[str, str]:
    """Build the ``_collapses.jsonl`` row for a fold verdict. A confident
    same-morpheme verdict becomes a real FOLD (``into`` the rich lemma,
    tombstoning the barren via merged_into_id); anything else is a no-op
    ``into: ""`` row that records the judgment so re-runs skip the candidate.
    ``ref`` is the barren (the form being folded away) — the fold convention.

    Both fold and reject rows carry ``candidate_lemma`` = the lemma this barren
    was judged AGAINST, so re-runs dedup at the (barren, lemma) PAIR level: a
    barren rejected against a weak lemma (stone↔stǣnen, noun vs adjective) can
    still be re-judged against a better one that only surfaces at a looser
    similarity floor (stone↔stān) instead of being permanently shadowed."""
    fold = v.same_morpheme and CONFIDENCE_RANK[v.confidence] >= CONFIDENCE_RANK[min_confidence]
    return {
        "_type": "collapse",
        "ref": c.barren_ref,
        "into": c.lemma_ref if fold else "",
        "candidate_lemma": c.lemma_ref,
        "variant_class": "spelling-variant",
        "method": LLM_VARIANT_FOLD_METHOD if fold else LLM_VARIANT_REJECT_METHOD,
        "confidence": v.confidence,
        "reason": v.reason,
    }
