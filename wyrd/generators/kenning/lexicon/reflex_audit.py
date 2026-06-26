"""LLM reflex-link audit (wyrd-862b).

The era-grid town-name pollution ("-ham" / "-ton" / "-ing" showing whole place
names — Burnham, Bolton, Belchamp — as if they were spellings of the morpheme)
comes from bad ``reflex_etymon`` links: the meanings.json seed linked every
``modern_usage`` surface to the etymons on its word, so a place-name subject
linked the WHOLE toponym surface (``-burnham``) — or a fragment of a different
element (``Belch-``) — to the tail morpheme (``hām``). A reflex should be a
SPELLING / phonological descendant of one morpheme, not a toponym containing it.

There is no trustworthy deterministic screen: a real phonological reflex
(``Apple-``→``æppel``, ``-fearn``→``fearna``) and a compound/neighbor
(``-burnham``→``hām``, ``Belch-``→``hām``) are indistinguishable by string
shape — edit-distance and substring tests both misfire (``-ling`` is one edit
from ``ing``; ``apple`` shares no letters with ``acorn`` yet ``-ock`` is
junk). So the deterministic pass only SCREENS (a link whose surface IS a known
form of the etymon — canonical / spelling-variant / cognate-cluster mate /
descent reflex — is kept outright); the LLM is the arbiter on the residual,
exactly like ``merge_audit`` (6hbv) and the fuzzy disambiguator (D25).

Every verdict is logged to ``_reflex_audit.jsonl`` for idempotent re-runs + an
audit trail; the LLM is NEVER re-run at rebuild. The applied prune lives in
``_reflexes.jsonl`` (re-dumped from the pruned DB) — the L2 source of truth for
the reflex layer (wyrd-ned5) — so a rebuild reproduces the clean layer.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

LLM_REFLEX_AUDIT_METHOD = "llm-reflex-audit-v1"

# Gemini responseSchema dialect (uppercase types) — matches GeminiClient.
VERDICT_SCHEMA = {
    "type": "OBJECT",
    "properties": {"valid": {"type": "BOOLEAN"}, "reason": {"type": "STRING"}},
    "required": ["valid", "reason"],
}

SYSTEM_PROMPT = (
    "You audit a historical English place-name etymology lexicon. A 'reflex' "
    "is a single morpheme spelled or phonologically evolved across eras (Old "
    "English 'tūn' has reflexes 'tun', 'toun', 'town'; 'fearna' (fern) has "
    "'fearn', 'farne'). Decide whether a recorded (surface -> morpheme) reflex "
    "link is VALID — the surface is genuinely that ONE morpheme spelled/evolved "
    "differently — or INVALID. INVALID when the surface is really (a) a whole "
    "place-name that merely CONTAINS the morpheme ('Burnham' = burn+ham is NOT a "
    "reflex of 'ham'; 'Bolton' is NOT a reflex of 'tūn'), (b) a fragment of a "
    "DIFFERENT morpheme next to it ('Belch-', the first element of Belchamp, is "
    "NOT a reflex of 'ham'), or (c) a different word sharing letters ('apple' is "
    "NOT a reflex of 'æcern'=acorn). Judge by MEANING + sound, not shared "
    "letters. A genuine spelling/phonological variant of the SAME morpheme is "
    "VALID even if letters differ; a compound or neighbor is INVALID even if it "
    "contains the morpheme verbatim. When genuinely unsure, prefer valid=true."
)


def reflex_audit_user_prompt(canonical: str, language: str, gloss: str, surface: str) -> str:
    return (
        f"Morpheme: {canonical!r} ({language}), meaning: {gloss or '(no gloss)'}\n"
        f"Recorded reflex surface: {surface!r}\n"
        f"Is this surface a genuine spelling/phonological reflex of that ONE morpheme, "
        f"or a place-name / different-morpheme fragment that merely contains or neighbors it?"
    )


def strip_surface(s: str) -> str:
    """Accent-fold + drop dashes/spaces + lowercase — the morpheme's bare form."""
    s = unicodedata.normalize("NFD", s or "")
    return (
        "".join(c for c in s if unicodedata.category(c) != "Mn")
        .replace("-", "")
        .replace(" ", "")
        .lower()
    )


@dataclass(frozen=True)
class ReflexCandidate:
    """One suspicious reflex_etymon link for the LLM to judge."""

    reflex_id: int
    etymon_id: int
    surface: str
    canonical: str
    language: str
    gloss: str

    @property
    def key(self) -> tuple[str, str, str]:
        """Dedup key — one LLM judgment per unique (stripped-surface,
        stripped-canonical, gloss) question; the verdict applies to every link
        sharing it (the OCR/case-variant etymon rows tūn/Tun/tun carry the same
        bad reflexes)."""
        return (strip_surface(self.surface), strip_surface(self.canonical), self.gloss)


# Scope every known-form load to the etymons that actually appear in a reflex
# link — only those can be candidates. Without it the loads full-scan etymon
# (~2.4M) / etymon_variant (~5.8M) into memory (Gemini ![high]). A direct
# subquery (vs a temp table) keeps it simple; reflex_etymon is small so SQLite
# re-evaluating it per query is cheap.
_LINKED = "(SELECT etymon_id FROM reflex_etymon)"


def _known_forms(db: sqlite3.Connection) -> tuple[dict, dict, dict, dict, dict]:
    """Per-etymon valid-surface evidence for the deterministic pre-screen, scoped
    to the etymons present in ``reflex_etymon`` (the only ones that can be
    candidates). Assumes the caller set ``db.row_factory = sqlite3.Row``."""
    et = {}
    for r in db.execute(
        f"SELECT id,canonical_form,language,cognate_id FROM etymon WHERE id IN {_LINKED}"
    ):
        et[r["id"]] = (
            r["canonical_form"],
            strip_surface(r["canonical_form"]),
            r["language"],
            r["cognate_id"],
        )
    # Cluster mates: members of the cognate clusters the linked etymons belong to.
    cluster: dict[int, set[str]] = defaultdict(set)
    for r in db.execute(
        f"SELECT cognate_id,canonical_form FROM etymon WHERE merged_into_id IS NULL "
        f"AND cognate_id IN (SELECT DISTINCT cognate_id FROM etymon "
        f"WHERE id IN {_LINKED} AND cognate_id IS NOT NULL)"
    ):
        cluster[r["cognate_id"]].add(strip_surface(r["canonical_form"]))
    var: dict[int, set[str]] = defaultdict(set)
    try:
        for r in db.execute(
            f"SELECT etymon_id,form FROM etymon_variant WHERE etymon_id IN {_LINKED}"
        ):
            var[r["etymon_id"]].add(strip_surface(r["form"]))
    except sqlite3.OperationalError as exc:
        # etymon_variant is an additive table — tolerate its ABSENCE on an older /
        # partial DB (variants simply don't contribute). Match the table BY NAME so
        # only ITS absence is tolerated: any other operational error — a schema
        # mismatch, a lock, or a missing CORE table (reflex_etymon, referenced in the
        # subquery) — is a real fault and re-raises rather than silently degrading the
        # audit to an empty variant set. (Two substrings, not the exact message, so a
        # schema-qualified "no such table: main.etymon_variant" still matches.)
        err_msg = str(exc).lower()
        if "no such table" not in err_msg or "etymon_variant" not in err_msg:
            raise
    desc: dict[int, set[str]] = defaultdict(set)
    for r in db.execute(
        f"SELECT parent_id,child.canonical_form cf FROM etymon_descent ed "
        f"JOIN etymon child ON ed.child_id=child.id "
        f"WHERE ed.parent_id IN {_LINKED} AND ed.edge_type IN ('inheritance','borrowing')"
    ):
        desc[r["parent_id"]].add(strip_surface(r["cf"]))
    gloss = {}
    for r in db.execute(
        f"SELECT etymon_id,gloss FROM etymon_gloss WHERE etymon_id IN {_LINKED} "
        f"AND gloss IS NOT NULL AND gloss != '' ORDER BY gloss"
    ):
        gloss.setdefault(r["etymon_id"], r["gloss"])
    return et, cluster, var, desc, gloss


def find_candidates(db: sqlite3.Connection) -> list[ReflexCandidate]:
    """Every reflex_etymon link whose surface is NOT a known form of the etymon
    (canonical / spelling-variant / cognate-cluster mate / descent reflex). The
    known-form links are kept outright; only these reach the LLM.

    Saves/restores ``db.row_factory`` so a caller-shared connection isn't left
    mutated (Gemini review — sqlite3's row_factory is connection-level, not
    per-cursor, so a save/restore is the correct way to scope it)."""
    saved_factory = db.row_factory
    db.row_factory = sqlite3.Row
    try:
        et, cluster, var, desc, gloss = _known_forms(db)
        out: list[ReflexCandidate] = []
        for r in db.execute(
            "SELECT re.reflex_id, re.etymon_id, rf.surface_form "
            "FROM reflex_etymon re JOIN reflex rf ON rf.id=re.reflex_id"
        ):
            eid = r["etymon_id"]
            canon_raw, canon, lang, cog = et.get(eid, ("", "", "", None))
            valid = {canon} | var.get(eid, set()) | desc.get(eid, set())
            if cog is not None:
                valid |= cluster.get(cog, set())
            valid.discard("")
            if strip_surface(r["surface_form"]) in valid:
                continue
            out.append(
                ReflexCandidate(
                    r["reflex_id"], eid, r["surface_form"], canon_raw, lang, gloss.get(eid, "")
                )
            )
        return out
    finally:
        db.row_factory = saved_factory


def load_candidates(db_path) -> list[ReflexCandidate]:
    """Open ``db_path`` (read) and return the suspicious reflex links. Owns the
    sqlite3 connection so CLI glue need not import sqlite3 (package boundary)."""
    conn = sqlite3.connect(db_path)
    try:
        return find_candidates(conn)
    finally:
        conn.close()


def apply_prune(db_path, prune_keys: set[tuple[str, str, str]], out_dir) -> tuple[int, int]:
    """Prune PRUNE-verdict links from ``db_path`` and re-dump ``_reflexes.jsonl``
    into ``out_dir`` so the L2 source stays consistent. Returns
    ``(links_deleted, reflex_rows_dumped)``. Owns the connection + the dump
    import so CLI glue stays sqlite3-free."""
    from wyrd.generators.kenning.jsonl.dump import dump_reflexes_to_file

    conn = sqlite3.connect(db_path)
    try:
        # One transaction: if the re-dump fails, the prune rolls back so the DB
        # never gets out of sync with the L2 _reflexes.jsonl (Gemini review).
        with conn:
            n = prune_links(conn, prune_keys)
            conn.row_factory = sqlite3.Row
            _, rows = dump_reflexes_to_file(conn, out_dir)
        return n, rows
    finally:
        conn.close()


def prune_links(db: sqlite3.Connection, prune_keys: set[tuple[str, str, str]]) -> int:
    """Delete reflex_etymon links whose candidate key is a PRUNE verdict. Reflex
    ROWS are left intact — a now-link-less reflex (e.g. '-burnham') is emitted by
    the bundle as a standalone single-word subject (_orphan_reflex_subjects), a
    real toponym with NO morpheme_id that cannot re-pollute any morpheme's
    era-grid. Returns the number of links deleted. Does NOT commit — the caller
    owns the transaction (apply_prune wraps prune + re-dump in one ``with conn:``
    so they're atomic); a direct caller should commit itself."""
    del_links = [(c.reflex_id, c.etymon_id) for c in find_candidates(db) if c.key in prune_keys]
    db.executemany("DELETE FROM reflex_etymon WHERE reflex_id=? AND etymon_id=?", del_links)
    return len(del_links)
