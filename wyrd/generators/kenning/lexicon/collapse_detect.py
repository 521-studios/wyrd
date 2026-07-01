"""Deterministic collapse detection — wyrd-y651 (consolidation epic P2b).

Finds the UNAMBIGUOUS form-of / variant etymons that should fold into
their lemma, for emission to ``data/mining/_collapses.jsonl`` (replayed
by :func:`enrichment.apply_collapses`). Two methods, both high-confidence:

(A) **variant-gloss-overlap** — an etymon whose ``(form, language)`` is a
    wiktextract-asserted :data:`etymon_variant` of a DIFFERENT same-
    language lemma AND shares a gloss with that lemma (case/whitespace-
    normalized equality, not generic-substring overlap). The variant
    assertion + the shared meaning together rule out the generic-gloss
    false positive (two different words that merely both gloss "hill").

(B) **pointer-parse** — a PURE form-of etymon: every one of its glosses
    is a cross-reference ("alternative form of burg", "h-prothesized form
    of ea") and the named target resolves to an existing same-language
    lemma. These carry no meaning of their own, so folding them onto the
    target (which holds the real gloss) is lossless.

Ambiguous cases — a doubling that does NOT share a gloss, or a
pointer+real-gloss etymon — are left for the LLM merge pass (P3).
"""

from __future__ import annotations

import re
import sqlite3

from wyrd.generators.kenning.lexicon.morpheme_surface import normalize_morpheme_surface

VARIANT_GLOSS_METHOD = "deterministic-variant-gloss-overlap"
POINTER_PARSE_METHOD = "deterministic-pointer-parse"

# A gloss that is a cross-reference, not a meaning. Anchored to the START
# (allowing ≤3 leading qualifier words like "alternative" / "h-prothesized"
# / "early modern") so a real pointer — "alternative form of burg",
# "h-prothesized form of ea" — matches, but prose that merely CONTAINS the
# phrase does not (e.g. "...holding land by a particular form of free
# tenure", where "form of" is 6 words deep, stays a real definition).
_POINTER_GLOSS_RE = re.compile(
    r"^\s*(?:[a-zà-öø-ÿ0-9'-]+\s+){0,3}(?:form|spelling|variant)\s+of\s+\S",
    re.IGNORECASE,
)

# wyrd-800c: the coarse regex above matches "<≤3 words> form/spelling/variant
# of <X>", which also fits DEFINITIONS that merely open with that shape —
# "early form of writing" (a primitive writing system), "old spelling of the
# name" — not cross-references. Method B AUTO-APPLIES pointer folds (no LLM
# gate), so a false positive tombstones a distinct morpheme (D46: a missed fold
# is harmless, a wrong fold corrupts). Two high-precision semantic guards reject
# the definitional shapes while keeping every real pointer observed in the
# corpus (all 51 live hits: "alternative"/"pet"/"northumbrian"/… form of):
#   1. the word immediately modifying form/spelling/variant is a bare
#      temporal/degree adjective ("early form of", "old spelling of") — real
#      pointers use register/dialect qualifiers, not "how old a form" ones;
#   2. the pointer TAIL opens with an article ("of the name") — a real pointer
#      names a bare lemma, never "the <word>".
# Borderline cases these reject fall through to the LLM-gated merge pass (D46).
_DEFINITIONAL_LEAD_ADJ = frozenset(
    {"early", "earlier", "earliest", "old", "older", "oldest", "primitive", "ancient"}
)
_TAIL_ARTICLES = frozenset({"the", "a", "an"})
_POINTER_KEYWORDS = frozenset({"form", "spelling", "variant"})
_POINTER_WORD_RE = re.compile(r"[a-zà-öø-ÿ0-9'-]+", re.IGNORECASE)


def is_form_of_pointer(gloss: str) -> bool:
    """True if a gloss is a form-of cross-reference ("alternative form of
    burg", "h-prothesized form of ea") rather than a meaning. Used both to
    detect collapse candidates here and to drop pointer glosses from the
    exported bundle (wyrd-6r09) so a folded form-of etymon shows its
    lemma's real meaning, not the pointer.

    Applies the wyrd-800c definitional-shape guards (see ``_POINTER_GLOSS_RE``
    comment) so a genuine definition opening "early form of <X>" / "old
    spelling of the <X>" is NOT read as a pointer."""
    if not _POINTER_GLOSS_RE.search(gloss):
        return False
    words = _POINTER_WORD_RE.findall(gloss.lower())
    for i, w in enumerate(words):
        # First keyword that is followed by "of" — the shape the regex matched.
        if w in _POINTER_KEYWORDS and i + 1 < len(words) and words[i + 1] == "of":
            lead = words[i - 1] if i > 0 else None
            tail_first = words[i + 2] if i + 2 < len(words) else None
            return lead not in _DEFINITIONAL_LEAD_ADJ and tail_first not in _TAIL_ARTICLES
    return False


def _detect_variant_gloss_overlap(conn: sqlite3.Connection) -> list[dict[str, str]]:
    """Method A: variant-doublings that share a gloss with their same-
    language parent lemma (case/whitespace-normalized equality, via the
    ``LOWER(TRIM(...))`` join). One collapse per ``from`` (the
    lexicographically-first parent wins, deterministically)."""
    sql = """
        SELECT e.language || ':' || e.canonical_form AS from_ref,
               parent.language || ':' || parent.canonical_form AS into_ref,
               v.variant_class AS variant_class
        FROM etymon e
        JOIN etymon_variant v
          ON v.form = e.canonical_form AND v.etymon_id != e.id
        JOIN etymon parent
          ON parent.id = v.etymon_id
         AND parent.language = e.language
         AND parent.merged_into_id IS NULL
        WHERE e.merged_into_id IS NULL
          AND EXISTS (SELECT 1 FROM etymon_gloss gp WHERE gp.etymon_id = parent.id)
          AND EXISTS (
                SELECT 1
                  FROM etymon_gloss g1
                  JOIN etymon_gloss g2
                    ON LOWER(TRIM(g1.gloss)) = LOWER(TRIM(g2.gloss))
                 WHERE g1.etymon_id = e.id AND g2.etymon_id = parent.id
          )
        ORDER BY from_ref, into_ref
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for r in conn.execute(sql):
        if r["from_ref"] == r["into_ref"] or r["from_ref"] in seen:
            continue
        seen.add(r["from_ref"])
        out.append(
            {
                "ref": r["from_ref"],
                "into": r["into_ref"],
                "variant_class": r["variant_class"] or "alternative",
                "method": VARIANT_GLOSS_METHOD,
            }
        )
    return out


_POINTER_TAIL_RE = re.compile(r"\b(?:form|spelling|variant) of\s+(.+)", re.IGNORECASE)
# Words in the "...of <tail>" span, case-PRESERVED and split on whitespace /
# punctuation. Case matters: canonical_form is stored bare lower-case, so a
# capitalized language qualifier ("Old", "English") simply fails to resolve.
_TAIL_WORD_RE = re.compile(r"[^\s(,.;:\"']+")


def _resolve_pointer_target(
    conn: sqlite3.Connection, language: str, canonical_form: str, glosses: list[str]
) -> str | None:
    """The lemma a pure-pointer etymon folds INTO: the UNIQUE live same-language
    lemma named in its first form-of gloss tail. A pointer names its target after
    an optional register/dialect qualifier ("Northumbrian form of earon" → earon),
    and a qualifier word can ITSELF be a live lemma — "...form of west saxon burg"
    has both ``west`` and ``burg`` — so trusting the first word after "of" would
    fold onto the qualifier. Instead de-dash EVERY word in the LEMMA REGION and fold
    only when EXACTLY ONE resolves; an ambiguous region (≥2 resolve) is left for the
    LLM merge pass. This fold is AUTO-APPLIED (no LLM gate), so it must never guess —
    D46: a missed fold is harmless, a wrong fold tombstones a distinct morpheme. (A
    capital qualifier like ``Old English`` is filtered for free by the bare-lower-case
    canonical_form, so the unique resolver is the real lemma.)

    The lemma region is the tail UP TO the first ``(`` — a parenthetical sense-gloss
    ('form of wiell ("spring, fountain, well")') holds the lemma's MEANING, never the
    lemma, so its words must not be resolution candidates (else, with the lemma
    absent, the fold lands on a sense word). Does NOT fix two sole-gloss FALSE-pointer
    shapes left to wyrd-800c, because a clean wiktextract form-of template produces
    neither: (a) a real meaning matching the pointer regex ("early form of writing"
    → ``writing``), and (b) trailing bare prose after the lemma ("form of burg used
    widely" — if ``burg`` is absent the region's lone resolver is the prose word).
    Both are hand-written definitions mis-read as pointers — a semantic gap a resolver
    count can't close."""
    for g in glosses:
        m = _POINTER_TAIL_RE.search(g)
        if not m:
            continue
        region = m.group(1).split("(", 1)[0]  # drop a parenthetical sense-gloss
        resolved: set[str] = set()
        for word in _TAIL_WORD_RE.findall(region):
            cand = normalize_morpheme_surface(word) or word
            if not cand or cand == canonical_form:
                continue
            if conn.execute(
                "SELECT 1 FROM etymon "
                "WHERE language = ? AND canonical_form = ? AND merged_into_id IS NULL",
                (language, cand),
            ).fetchone():
                resolved.add(cand)
        return next(iter(resolved)) if len(resolved) == 1 else None
    return None


def _detect_pointer_parse(conn: sqlite3.Connection) -> list[dict[str, str]]:
    """Method B: pure form-of etymons (every gloss is a pointer) whose
    named target resolves to an existing same-language lemma."""
    candidates = conn.execute(
        """
        SELECT e.id, e.language, e.canonical_form
          FROM etymon e
         WHERE e.merged_into_id IS NULL
           AND EXISTS (
                 SELECT 1 FROM etymon_gloss g
                  WHERE g.etymon_id = e.id
                    AND (g.gloss LIKE '% form of %'
                      OR g.gloss LIKE 'alternative %'
                      OR g.gloss LIKE 'variant %'
                      OR g.gloss LIKE '%spelling of %'
                      OR g.gloss LIKE '%prothesi%')
           )
        """
    ).fetchall()
    out: list[dict[str, str]] = []
    for c in candidates:
        glosses = [
            row["gloss"]
            for row in conn.execute(
                "SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (c["id"],)
            )
        ]
        # require EVERY gloss to be a pointer — a pure form-of etymon. A
        # pointer+real-gloss etymon might be a real lemma; that's P3's call.
        # is_form_of_pointer applies the wyrd-800c definitional-shape guards.
        if not glosses or not all(is_form_of_pointer(g) for g in glosses):
            continue
        # The fold target: the UNIQUE live same-language lemma named in the
        # pointer tail (de-dashed, wyrd-aicu.8/D45). None = no target, or an
        # ambiguous tail left for the LLM pass (see _resolve_pointer_target).
        target = _resolve_pointer_target(conn, c["language"], c["canonical_form"], glosses)
        if target is None:
            continue
        out.append(
            {
                "ref": f"{c['language']}:{c['canonical_form']}",
                "into": f"{c['language']}:{target}",
                "variant_class": "alternative",
                "method": POINTER_PARSE_METHOD,
            }
        )
    return out


def _break_collapse_cycles(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop every edge that would close a cycle in the ``merged_into_id`` graph.

    Union-find over the ``ref``/``into`` nodes. Edges are processed in
    deterministic ``(ref, into)`` order; the first direction seen for a pair
    survives, the cycle-closer is skipped — so a wiktextract pair registered
    BIDIRECTIONALLY (``wode`` a variant of ``wood`` AND ``wood`` of ``wode``)
    can't tombstone each node into the other (a ``merged_into_id`` cycle that
    ``_resolve_live_etymon`` could never settle).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    kept: list[dict[str, str]] = []
    for r in sorted(rows, key=lambda e: (e["ref"], e["into"])):
        ra, rb = find(r["ref"]), find(r["into"])
        if ra == rb:
            continue  # already in one cluster — this edge would close a cycle
        parent[ra] = rb
        kept.append(r)
    return kept


def _flatten_collapse_chains(kept: list[dict[str, str]]) -> list[dict[str, str]]:
    """Re-point every collapse at its cluster's TERMINAL survivor.

    The survivor is the node that is never itself a ``ref``, so the ledger is a
    flat forest. Without this a chain ``wella -> well -> wiell`` is apply-order-
    fragile: if ``well`` is tombstoned before the ``wella -> well`` row,
    ``well`` no longer resolves and that fold is silently missed. Transitively
    still correct — a variant of a variant of X is a variant of X.

    The ``survivor(ref) != ref`` filter and the ``cur not in seen`` walk guard
    are purely defensive: :func:`_break_collapse_cycles` runs first and leaves
    ``direct`` acyclic, so ``survivor(ref)`` always reaches a distinct terminal
    node. Were a cycle to reach here anyway, the guard stops the walk and the
    filter drops the self-resolving rows rather than looping forever.
    """
    direct = {r["ref"]: r["into"] for r in kept}

    def survivor(ref: str) -> str:
        cur, seen = ref, set()
        while cur in direct and cur not in seen:
            seen.add(cur)
            cur = direct[cur]
        return cur

    return [{**r, "into": survivor(r["ref"])} for r in kept if survivor(r["ref"]) != r["ref"]]


def detect_deterministic_collapses(conn: sqlite3.Connection) -> list[dict[str, str]]:
    """Return collapse rows for the deterministic cases, with cycles broken.

    A three-step pipeline over the candidate collapse edges:

      1. Gather from both detectors. Method A (variant/gloss overlap, which
         carries the wiktextract ``variant_class``) wins on ``ref`` collisions;
         Method B (pointer-parse) fills only refs Method A didn't claim.
      2. :func:`_break_collapse_cycles` drops edges that would close a
         ``merged_into_id`` cycle.
      3. :func:`_flatten_collapse_chains` re-points each surviving edge at its
         cluster's terminal survivor.

    See those helpers for the cycle-breaking and chain-flattening rationale.
    """
    rows = _detect_variant_gloss_overlap(conn)
    seen = {r["ref"] for r in rows}
    for r in _detect_pointer_parse(conn):
        if r["ref"] not in seen:
            seen.add(r["ref"])
            rows.append(r)

    return _flatten_collapse_chains(_break_collapse_cycles(rows))
