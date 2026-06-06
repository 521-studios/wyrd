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


def _known_forms(db: sqlite3.Connection) -> tuple[dict, dict, dict, dict, dict]:
    """Per-etymon valid-surface evidence used by the deterministic pre-screen."""
    db.row_factory = sqlite3.Row
    et = {}
    for r in db.execute("SELECT id,canonical_form,language,cognate_id FROM etymon"):
        et[r["id"]] = (
            r["canonical_form"],
            strip_surface(r["canonical_form"]),
            r["language"],
            r["cognate_id"],
        )
    cluster: dict[int, set[str]] = defaultdict(set)
    for r in db.execute(
        "SELECT cognate_id,canonical_form FROM etymon "
        "WHERE cognate_id IS NOT NULL AND merged_into_id IS NULL"
    ):
        cluster[r["cognate_id"]].add(strip_surface(r["canonical_form"]))
    var: dict[int, set[str]] = defaultdict(set)
    try:
        for r in db.execute("SELECT etymon_id,form FROM etymon_variant"):
            var[r["etymon_id"]].add(strip_surface(r["form"]))
    except sqlite3.OperationalError:
        pass
    desc: dict[int, set[str]] = defaultdict(set)
    for r in db.execute(
        "SELECT parent_id,child.canonical_form cf FROM etymon_descent ed "
        "JOIN etymon child ON ed.child_id=child.id WHERE ed.edge_type IN ('inheritance','borrowing')"
    ):
        desc[r["parent_id"]].add(strip_surface(r["cf"]))
    gloss = {}
    for r in db.execute("SELECT etymon_id,gloss FROM etymon_gloss ORDER BY gloss"):
        gloss.setdefault(r["etymon_id"], r["gloss"])
    return et, cluster, var, desc, gloss


def find_candidates(db: sqlite3.Connection) -> list[ReflexCandidate]:
    """Every reflex_etymon link whose surface is NOT a known form of the etymon
    (canonical / spelling-variant / cognate-cluster mate / descent reflex). The
    known-form links are kept outright; only these reach the LLM."""
    et, cluster, var, desc, gloss = _known_forms(db)
    db.row_factory = sqlite3.Row
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
    era-grid. Returns the number of links deleted. Caller re-dumps
    _reflexes.jsonl afterward so the L2 source stays consistent."""
    del_links = [(c.reflex_id, c.etymon_id) for c in find_candidates(db) if c.key in prune_keys]
    db.executemany("DELETE FROM reflex_etymon WHERE reflex_id=? AND etymon_id=?", del_links)
    db.commit()
    return len(del_links)
