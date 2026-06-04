"""Same-language variant-FOLD candidate detection (empty-grid root fix).

The cross-language sibling of ``reflex_link_detect`` (wyrd-rogd.15). Where that
LINKS cross-era reflexes (keeping both), this finds SAME-language path-dependent
DUPLICATES — a BARREN etymon (an anglicized / mis-spelled / accent-variant form
with no cognate cluster: OE ``hill`` id 4632179, 0 mates, gloss just "hill") and
the RICH proper lemma it duplicates (OE ``hyll``, 42 cluster mates, descriptive
gloss) — that are NOT connected by any descent edge. Generation resolves the
morpheme to the barren duplicate, so its era-grid comes up empty even though the
reflexes sit on the unlinked lemma.

Signal: SAME language + a shared gloss + form similarity + one side barren
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
    return "".join(ch for ch in bare if ch.isalpha() and ch not in "aeiouy")


# Common-gloss stopwords: function words + generic dictionary-definition framing
# that would over-group on token match without signalling shared meaning.
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


def detect_variant_fold_candidates(
    conn: sqlite3.Connection,
    *,
    min_similarity: float = 0.65,
    barren_max: int = 1,
    lemma_min: int = 5,
    max_per_gloss: int = 600,
) -> list[VariantFoldCandidate]:
    """Same-language barren↔rich variant-fold candidates: SAME-language etymon
    pairs sharing a gloss, form-similar (difflib ≥ ``min_similarity`` on the
    dash/star-stripped forms), one barren (cognate cluster ≤ ``barren_max``) and
    one rich (cluster ≥ ``lemma_min``), not already connected by a descent edge.
    One best lemma per barren (richest cluster, then highest similarity)."""
    rows = conn.execute(
        "SELECT e.id, e.language, e.canonical_form, e.cognate_id, g.gloss "
        "FROM etymon e JOIN etymon_gloss g ON g.etymon_id = e.id "
        "WHERE e.merged_into_id IS NULL"
    ).fetchall()
    by_id: dict[int, dict] = {}
    by_token: dict[str, set[int]] = defaultdict(set)  # content token -> etymon ids
    for r in rows:
        rec = by_id.setdefault(
            r["id"],
            {
                "lang": r["language"],
                "form": r["canonical_form"],
                "clu": r["cognate_id"],
                "gl": set(),
                "tok": set(),
            },
        )
        rec["gl"].add(r["gloss"])
        toks = _gloss_tokens(r["gloss"])
        rec["tok"].update(toks)
        for t in toks:
            by_token[t].add(r["id"])

    # Cluster size per cognate_id = richness signal (a rich lemma has cluster mates).
    cluster_size: dict[int, int] = {}
    for r in conn.execute(
        "SELECT cognate_id, count(*) n FROM etymon "
        "WHERE cognate_id IS NOT NULL AND merged_into_id IS NULL GROUP BY cognate_id"
    ):
        cluster_size[r["cognate_id"]] = r["n"]

    edges: set[tuple[int, int]] = set()
    for p, c in conn.execute("SELECT parent_id, child_id FROM etymon_descent"):
        edges.add((p, c))

    def richness(rec: dict) -> int:
        return cluster_size.get(rec["clu"], 0) if rec["clu"] is not None else 0

    # candidates per barren id -> best (lemma id, sim). Group by a shared gloss
    # CONTENT TOKEN (not the whole gloss string), then pair BARREN×RICH: the rich
    # lemma set per token is tiny (a few well-attested forms), so even the huge
    # common-token groups (hill / stone / house — the very cases that matter) are
    # covered without an O(n²) blow-up. Token grouping is what lets a bare barren
    # gloss ('stone') reach a descriptive lemma gloss ('A stone, stone, rock.');
    # ``max_per_gloss`` only guards a pathologic generic-token group.
    best: dict[int, tuple[int, float]] = {}
    for ids in by_token.values():
        if len(ids) < 2 or len(ids) > max_per_gloss:
            continue
        rich = [(i, by_id[i]) for i in ids if richness(by_id[i]) >= lemma_min]
        barren = [(i, by_id[i]) for i in ids if richness(by_id[i]) <= barren_max]
        if not rich or not barren:
            continue
        for bi, brec in barren:
            fb = _bare(brec["form"])
            if not fb:
                continue
            skb = _consonant_skeleton(brec["form"])
            for ri, rrec in rich:
                if brec["lang"] != rrec["lang"]:
                    continue  # SAME-language only (cross-language is a LINK, wyrd-rogd.15)
                fr = _bare(rrec["form"])
                if not fr or fb == fr:
                    continue
                sim = difflib.SequenceMatcher(None, fb, fr).ratio()
                # Accept on form similarity OR a shared consonant skeleton (the
                # vowel-shift escape hatch — wood↔wudu); the skeleton needs ≥2
                # consonants so single-consonant skeletons can't over-match.
                skeleton_match = len(skb) >= 2 and skb == _consonant_skeleton(rrec["form"])
                if sim < min_similarity and not skeleton_match:
                    continue
                if (bi, ri) in edges or (ri, bi) in edges:
                    continue
                cur = best.get(bi)
                if cur is None or (richness(rrec), sim) > (richness(by_id[cur[0]]), cur[1]):
                    best[bi] = (ri, sim)

    out: list[VariantFoldCandidate] = []
    for barren, (lemma, sim) in best.items():
        eb, el = by_id[barren], by_id[lemma]
        # The barren glosses that share ≥1 content token with the lemma — the
        # meaning evidence behind the grouping (shown to the judge).
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


from dataclasses import dataclass as _dataclass  # noqa: E402

from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK  # noqa: E402


@_dataclass(frozen=True)
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
