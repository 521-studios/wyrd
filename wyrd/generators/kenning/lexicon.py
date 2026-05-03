"""Lexicon data store: SQLite-backed authoring DB for place-name etymology.

Authoritative for authoring (mining sources, tracking citations, recording
disagreement). The runtime keeps reading meanings.json, which is exported from
this DB by `wyrd kenning lexicon export-meanings`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wyrd.generators.kenning.skeat_parser import ParsedEntry

# Map meanings.json source-language field names to the lexicon's canonical
# language codes. Keep in sync with _LEGEND in the kenning generator. The
# misspellings ('old_scandanavian', 'old _english') are real entries in the
# bundled meanings.json — normalized here so we don't lose data.
LANGUAGE_FIELDS = {
    "old_english": "old-english",
    "old _english": "old-english",
    "old_scandinavian": "old-norse",
    "old_scandanavian": "old-norse",
    "old_french": "norman-french",
    "celtic_mix": "celtic",
    "latin": "latin",
    "germanic": "germanic",
    "greek": "greek",
    "modern_english": "modern-english",
    "biblical": "biblical",
}

# Fields on a `word` that are not source-language slots and must be skipped
# during ingestion. `source_known` is a per-entry boolean flag; we'll store it
# as etymon notes if it ever matters, but for now we just pass over it.
NON_LANGUAGE_FIELDS = {"modern_usage", "source_known"}


def _schema_sql() -> str:
    return resources.files("wyrd.generators.kenning.data").joinpath("lexicon.sql").read_text()


def init_schema(db_path: Path | str) -> None:
    """Create a fresh lexicon DB at db_path with the schema applied.

    Wipes any existing file at the path.
    """
    path = Path(db_path)
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_schema_sql())
        conn.commit()
    finally:
        conn.close()


# --- OCR ligature normalization ------------------------------------------
#
# OE ligature characters (æ, ð, þ, œ, ȳ, ē, etc.) are routinely mangled by
# OCR engines. The same etymon ends up under many spellings: "Hædan",
# "Hcsdan", "Hsedan", "Haedan", "Hædann"... breaking consensus counts.
#
# We normalize by:
#   1. Mapping common OCR confusions back to ASCII-equivalent OE chars.
#   2. Lowercasing.
#   3. Stripping leading/trailing dashes (which mark position, not form).
#
# This produces a canonical key that's robust to OCR variation.

# OCR-confused digraphs that should map to OE ligatures (in ASCII).
_OCR_LIGATURE_MAP = [
    # Order matters — longer/more-specific first. "cs" inside an OE form is
    # almost always a misread æ; same for "ce" in many positions. We keep the
    # ASCII-friendly equivalents (ae, dh, th, oe) in normalized form.
    ("æ", "ae"),
    ("ð", "dh"),
    ("þ", "th"),
    ("œ", "oe"),
    ("ȳ", "y"),
    ("ē", "e"),
    ("ī", "i"),
    ("ō", "o"),
    ("ū", "u"),
    ("ā", "a"),
    # Common OCR confusions for æ
    ("cs", "ae"),  # "Hcsdan" → "haedan"
    ("ce", "ae"),  # "Hcedan" → "haedan" (only in OE context — risky in modern)
    # Common OCR confusions for ð
    ("§", "dh"),
    ("dh", "dh"),  # idempotent
]


def normalize_ocr_form(form: str) -> str:
    """Normalize an etymon form against common OCR confusions.

    Returns a lowercased, dash-stripped, ligature-normalized form suitable
    as a clustering key. The result is NOT meant to be displayed — it's a
    join key.

    Conservative: we only apply the most reliable mappings. "ce" → "ae"
    is genuinely OE-context-dependent and applied here; if it produces
    false positives in modern-english entries we'd want a per-language
    rule set, but for now the lexicon is dominated by historical forms.
    """
    s = form.strip().strip("-").lower()
    for src, dst in _OCR_LIGATURE_MAP:
        s = s.replace(src, dst)
    return s


def _position_from_usage(modern_usage: str) -> str:
    """Derive a reflex position from the dash markers on a modern_usage string.

    Mirrors Meaning._set_location: starts and ends with '-' → inner, ends with
    '-' → pre, otherwise post.
    """
    if modern_usage.startswith("-") and modern_usage.endswith("-"):
        return "inner"
    if modern_usage.endswith("-"):
        return "pre"
    return "post"


class LexiconDB:
    """Thin convenience wrapper over sqlite3 for lexicon authoring tasks."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        # WAL mode lets readers and a single writer proceed in parallel
        # without blocking each other — important for the multi-session
        # workflow where one Claude is mining (writer) while another is
        # querying corpus state (reader). journal_mode is persistent: set
        # once on the file, applies to all subsequent opens.
        # synchronous=NORMAL is the recommended pairing under WAL; full
        # crash safety is preserved (WAL replays on next open) without
        # the per-transaction fsync cost of synchronous=FULL.
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.row_factory = sqlite3.Row

    def __enter__(self) -> LexiconDB:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    # --- writers ---------------------------------------------------------

    def upsert_source(
        self,
        *,
        id: str,
        title: str,
        author: str | None = None,
        year: int | None = None,
        region: str | None = None,
        language_focus: str | None = None,
        notes: str | None = None,
    ) -> str:
        self.conn.execute(
            """
            INSERT INTO source (id, author, title, year, region, language_focus, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                author = excluded.author,
                title = excluded.title,
                year = excluded.year,
                region = excluded.region,
                language_focus = excluded.language_focus,
                notes = excluded.notes
            """,
            (id, author, title, year, region, language_focus, notes),
        )
        return id

    def upsert_etymon(
        self,
        canonical_form: str,
        language: str,
        *,
        modifier_type: str | None = None,
        position_pref: str | None = None,
        notes: str | None = None,
    ) -> int:
        # ON CONFLICT lets us do this in a single statement: insert if new,
        # otherwise fill in any missing fields without overwriting existing
        # non-null values. RETURNING id avoids a follow-up SELECT.
        cur = self.conn.execute(
            """
            INSERT INTO etymon (canonical_form, language, modifier_type, position_pref, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(canonical_form, language) DO UPDATE SET
                modifier_type = COALESCE(etymon.modifier_type, excluded.modifier_type),
                position_pref = COALESCE(etymon.position_pref, excluded.position_pref),
                notes         = COALESCE(etymon.notes, excluded.notes)
            RETURNING id
            """,
            (canonical_form, language, modifier_type, position_pref, notes),
        )
        return cur.fetchone()[0]

    def add_gloss(self, etymon_id: int, gloss: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (etymon_id, gloss),
        )

    def add_tag(self, etymon_id: int, tag: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO etymon_tag (etymon_id, tag) VALUES (?, ?)",
            (etymon_id, tag),
        )

    def add_citation(
        self,
        etymon_id: int,
        source_id: str,
        *,
        page: str | None = None,
        short_quote: str | None = None,
        context_snippet: str | None = None,
    ) -> None:
        """Insert an etymon_citation row.

        ``context_snippet`` (wyrd-9kh.3) is the surrounding scholarly-prose
        context window for in-app citation display. Populated by siblings
        (.4 reverse-search snippet capture, .5 page-number parser); leave
        None for callers that haven't been updated to capture it yet.
        """
        self.conn.execute(
            """
            INSERT OR IGNORE INTO etymon_citation
                (etymon_id, source_id, page, short_quote, context_snippet)
            VALUES (?, ?, ?, ?, ?)
            """,
            (etymon_id, source_id, page, short_quote, context_snippet),
        )

    def upsert_reflex(self, surface_form: str, position: str) -> int:
        cur = self.conn.execute(
            "SELECT id FROM reflex WHERE surface_form = ? AND position = ?",
            (surface_form, position),
        )
        row = cur.fetchone()
        if row is not None:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO reflex (surface_form, position) VALUES (?, ?)",
            (surface_form, position),
        )
        return cur.lastrowid

    def link_reflex_etymon(self, reflex_id: int, etymon_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)",
            (reflex_id, etymon_id),
        )

    # --- readers / stats -------------------------------------------------

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in (
            "source",
            "etymon",
            "etymon_gloss",
            "etymon_tag",
            "etymon_citation",
            "reflex",
            "reflex_etymon",
            "toponym",
            "toponym_attestation",
            "toponym_etymology",
            "toponym_etymology_element",
        ):
            cur = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = cur.fetchone()["n"]
        return out


# --- seed ingestion --------------------------------------------------------


def seed_from_meanings(
    db: LexiconDB,
    meanings_data: list[dict[str, Any]],
    source_id: str,
) -> dict[str, int]:
    """Ingest a meanings.json document into the lexicon as <source_id>.

    Each subject in the meanings.json contributes:
      - one etymon per (canonical_form, language) tuple that appears across
        its `words[]` entries
      - the subject's `meaning[]` glosses, attached to every etymon
      - the subject's `modifier_tags[]` tags, attached to every etymon
      - the subject's `modifier_type`, set on every etymon
      - one reflex per `modern_usage`, position-tagged from its dash markers
      - reflex→etymon links: a reflex points to every etymon implied by the
        language fields on the originating word
      - one citation row per etymon, pointing at <source_id>

    Returns a counts dict for sanity-checking.
    """
    counts = {
        "etymons": 0,
        "reflexes": 0,
        "links": 0,
        "glosses": 0,
        "tags": 0,
        "citations": 0,
    }

    for subject in meanings_data:
        glosses: list[str] = subject.get("meaning", []) or []
        tags: list[str] = subject.get("modifier_tags", []) or []
        modifier_type: str | None = subject.get("modifier_type")
        words: list[dict[str, Any]] = subject.get("words", []) or []

        # First pass: collect every (form, lang) tuple this subject mentions,
        # so each one becomes a single etymon shared across its reflexes.
        etymons_in_subject: dict[tuple[str, str], int] = {}
        for word in words:
            for json_field, lang_code in LANGUAGE_FIELDS.items():
                forms = word.get(json_field) or []
                for form in forms:
                    key = (form, lang_code)
                    if key in etymons_in_subject:
                        continue
                    etymon_id = db.upsert_etymon(
                        form,
                        lang_code,
                        modifier_type=modifier_type,
                    )
                    etymons_in_subject[key] = etymon_id
                    counts["etymons"] += 1
                    for gloss in glosses:
                        db.add_gloss(etymon_id, gloss)
                        counts["glosses"] += 1
                    for tag in tags:
                        db.add_tag(etymon_id, tag)
                        counts["tags"] += 1
                    db.add_citation(etymon_id, source_id)
                    counts["citations"] += 1

        # Second pass: each word gives one reflex; link it to every etymon
        # implied by the language fields on the same word.
        for word in words:
            modern_usage = word.get("modern_usage")
            if not modern_usage:
                continue
            position = _position_from_usage(modern_usage)
            reflex_id = db.upsert_reflex(modern_usage, position)
            counts["reflexes"] += 1
            for json_field, lang_code in LANGUAGE_FIELDS.items():
                forms = word.get(json_field) or []
                for form in forms:
                    etymon_id = etymons_in_subject.get((form, lang_code))
                    if etymon_id is None:
                        continue
                    db.link_reflex_etymon(reflex_id, etymon_id)
                    counts["links"] += 1

    db.commit()
    return counts


# --- schema migration ------------------------------------------------------


def _add_etymon_columns(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Add lemma_id, inflection, merged_into_id, and lemma_method columns
    if they don't yet exist. Each column is independent and idempotent.
    """
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
    if "lemma_id" not in cols:
        # ON DELETE SET NULL matches lexicon.sql's base schema and keeps
        # the migration / fresh-install paths in sync. Without it, deleting
        # a lemma row would FK-fail when foreign_keys=ON is enabled.
        db.conn.execute(
            "ALTER TABLE etymon ADD COLUMN lemma_id INTEGER "
            "REFERENCES etymon(id) ON DELETE SET NULL"
        )
        applied["etymon.lemma_id"] = True
    if "inflection" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN inflection TEXT")
        applied["etymon.inflection"] = True
    # merged_into_id: non-destructive OCR clustering. When two etymons are
    # merged by cluster_ocr_variants, the loser gets merged_into_id pointing
    # at the canonical winner instead of being DELETE'd. Per D22, this lets
    # callers blow away clustering (UPDATE etymon SET merged_into_id = NULL)
    # and re-run with new heuristics without re-mining.
    if "merged_into_id" not in cols:
        db.conn.execute(
            "ALTER TABLE etymon ADD COLUMN merged_into_id INTEGER "
            "REFERENCES etymon(id) ON DELETE SET NULL"
        )
        applied["etymon.merged_into_id"] = True
    # lemma_method: which version of the lemma-linker produced the
    # lemma_id/inflection attribution on this row. Lets future link-lemmas
    # versions identify their predecessors' work for selective rebuild.
    if "lemma_method" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN lemma_method TEXT")
        applied["etymon.lemma_method"] = True
    # synset_id (D27 / wyrd-0ug): cognate-cluster rollup. Points at the
    # most-ancestral known etymon in this cognate set; populated by the
    # cluster-cognates pass (wyrd-81n) walking etymon_descent inheritance
    # edges. Cross-language by design — distinct from merged_into_id
    # (within-language OCR cluster) and lemma_id (within-language
    # inflection family).
    if "synset_id" not in cols:
        db.conn.execute(
            "ALTER TABLE etymon ADD COLUMN synset_id INTEGER "
            "REFERENCES etymon(id) ON DELETE SET NULL"
        )
        applied["etymon.synset_id"] = True
    # synset_method: which version of cluster-cognates produced this
    # synset_id assignment. Mirrors lemma_method.
    if "synset_method" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN synset_method TEXT")
        applied["etymon.synset_method"] = True


def _create_etymon_indexes(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Idempotent index creation for the etymon table's self-FK columns."""
    existing_indexes = {
        row["name"] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    if "idx_etymon_lemma" not in existing_indexes:
        db.conn.execute("CREATE INDEX idx_etymon_lemma ON etymon(lemma_id)")
        applied["idx_etymon_lemma"] = True
    if "idx_etymon_merged_into" not in existing_indexes:
        db.conn.execute("CREATE INDEX idx_etymon_merged_into ON etymon(merged_into_id)")
        applied["idx_etymon_merged_into"] = True
    if "idx_etymon_synset" not in existing_indexes:
        db.conn.execute("CREATE INDEX idx_etymon_synset ON etymon(synset_id)")
        applied["idx_etymon_synset"] = True


def _rebuild_etymon_views(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Recreate etymon_canonical and etymon_consensus.

    Always rebuilt (DROP IF EXISTS + CREATE) because the definitions change
    when new columns are introduced, and SQLite has no CREATE OR REPLACE
    VIEW. Cheap.
    """
    db.conn.execute("DROP VIEW IF EXISTS etymon_canonical")
    db.conn.execute(
        """
        CREATE VIEW etymon_canonical AS
          SELECT * FROM etymon WHERE merged_into_id IS NULL
        """
    )
    applied["etymon_canonical_view"] = True
    db.conn.execute("DROP VIEW IF EXISTS etymon_consensus")
    db.conn.execute(
        """
        CREATE VIEW etymon_consensus AS
          SELECT lemma_id,
                 canonical_form,
                 language,
                 COUNT(DISTINCT source_id) AS witnesses
          FROM (
            SELECT
              -- Two-step rollup so merged_into_id → lemma_id chains
              -- collapse correctly: target = e's redirect (merged or lemma);
              -- le = target's lemma if target is itself inflected.
              COALESCE(le.id, target.id, e.id) AS lemma_id,
              COALESCE(le.canonical_form, target.canonical_form, e.canonical_form)
                AS canonical_form,
              e.language,
              c.source_id
            FROM etymon e
            LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
            LEFT JOIN etymon le ON le.id = target.lemma_id
            LEFT JOIN etymon_citation c ON c.etymon_id = e.id
          )
          GROUP BY lemma_id
        """
    )
    applied["etymon_consensus_view"] = True


def _create_etymon_descent_table(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Create etymon_descent if missing (D27 / wyrd-0ug).

    Holds directed edges of the etymological descent graph populated by
    Wiktionary mining (wyrd-4rt) and any future scholar-cited chain
    assertions. The cognate cluster column etymon.synset_id is derived
    from this graph by the cluster-cognates pass (wyrd-81n).

    Schema mirrors data/lexicon.sql so fresh-install and migration paths
    stay in lockstep. Idempotent — checks for the table before creating.
    """
    existing = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "etymon_descent" in existing:
        return
    db.conn.executescript(
        """
        CREATE TABLE etymon_descent (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          parent_id   INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          child_id    INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          edge_type   TEXT NOT NULL CHECK (edge_type IN (
                        'inheritance', 'borrowing', 'calque',
                        'compound', 'derivation', 'cognate', 'unknown'
                      )),
          source_id   TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
          confidence  TEXT CHECK (confidence IN ('high', 'medium', 'low')),
          notes       TEXT,
          UNIQUE (parent_id, child_id, edge_type, source_id)
        );
        CREATE INDEX idx_etymon_descent_parent ON etymon_descent(parent_id);
        CREATE INDEX idx_etymon_descent_child  ON etymon_descent(child_id);
        """
    )
    applied["etymon_descent_table"] = True


def _migrate_text_match_table(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Create etymon_text_match if missing; otherwise add the method column
    if it doesn't exist. Tracks which heuristic produced each row
    ('reverse-search-v1', 'fuzzy-search-v1', 'llm-disambiguator-v1', ...)
    so a future rebuild can selectively clear-and-regenerate by method.
    """
    existing_tables = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "etymon_text_match" not in existing_tables:
        db.conn.executescript(
            """
            CREATE TABLE etymon_text_match (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              etymon_id   INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
              source_id   TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
              matched_form  TEXT NOT NULL,
              match_count   INTEGER NOT NULL,
              edit_distance INTEGER NOT NULL DEFAULT 0,
              snippet       TEXT,
              method        TEXT NOT NULL DEFAULT 'reverse-search-v1',
              disambiguator_reason TEXT,
              attested_year INTEGER,
              UNIQUE (etymon_id, source_id, matched_form)
            );
            CREATE INDEX idx_etymon_text_match_etymon ON etymon_text_match(etymon_id);
            CREATE INDEX idx_etymon_text_match_source ON etymon_text_match(source_id);
            CREATE INDEX idx_etymon_text_match_year   ON etymon_text_match(attested_year);
            """
        )
        applied["etymon_text_match_table"] = True
        return
    text_match_cols = {
        row["name"] for row in db.conn.execute("PRAGMA table_info(etymon_text_match)")
    }
    if "method" not in text_match_cols:
        db.conn.execute(
            "ALTER TABLE etymon_text_match ADD COLUMN method TEXT "
            "NOT NULL DEFAULT 'reverse-search-v1'"
        )
        applied["etymon_text_match.method"] = True
    if "disambiguator_reason" not in text_match_cols:
        db.conn.execute("ALTER TABLE etymon_text_match ADD COLUMN disambiguator_reason TEXT")
        applied["etymon_text_match.disambiguator_reason"] = True
    if "attested_year" not in text_match_cols:
        db.conn.execute("ALTER TABLE etymon_text_match ADD COLUMN attested_year INTEGER")
        # The index is only valuable once the column exists; create alongside.
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_etymon_text_match_year "
            "ON etymon_text_match(attested_year)"
        )
        applied["etymon_text_match.attested_year"] = True


# wyrd-9kh.5: pattern for Mawer-style alphabetical-headword running headers
# (e.g. 'BACKWORTH 9' on Mawer 1920, 'ABBEY DORE 1' on Bannister 1916,
# 'ST PETER'S 47' for saint-prefixed places). The leftmost word is the
# first headword on the page; the rightmost integer is the page number.
# Single-word forms require 3+ chars to filter OCR noise; multi-word
# forms accept a 2+ char first word so 'ST PETER'S' / 'DR FOO' admit.
_RUNNING_HEADER_RE = re.compile(
    r"^\s*("
    r"[A-Z][A-Z\-']*\s+[A-Z][A-Z\-']+(?:\s+[A-Z][A-Z\-']+)*"  # multi-word: 2+ char first word
    r"|"
    r"[A-Z][A-Z\-']{2,}"  # single-word: 3+ chars total
    r")\s+(\d+)\s*$",
    re.MULTILINE,
)


# wyrd-8st: pattern for Skeat-style §-section running headers
# (e.g. '§ 2. NAMES ENDING IN -TON. 9' on Skeat 1901 Cambridgeshire).
# Body of every printed page begins with one of these.
#
# Distinguished from non-running TOC + section openers ('§ 2. The suffix
# -ton.' — title case, no trailing page number) by requiring an
# ALL-CAPS title block (≥4 chars including allowed punctuation) AND a
# trailing integer page.
#
# OCR variants tolerated:
# - '§8.' (no space between § and number) and '§ 8.' (variable spacing)
# - 'IX -BRIDGE' (OCR misread of 'IN -BRIDGE')
# - Multi-suffix headers like 'NAMES ENDING IN -BRIDGE, -HITHE.'
_SKEAT_SECTION_HEADER_RE = re.compile(
    r"^\s*§\s*\d+\.\s+"  # § N.
    r"([A-Z\-][A-Z\s,\-\.\']{4,}?)"  # ALL-CAPS title block
    r"\s+(\d+)\s*$",  # trailing page
    re.MULTILINE,
)


def parse_running_header_pages(text: str) -> list[tuple[int, int]]:
    """Extract (offset, page_number) pairs from running headers in OCR'd
    alphabetical-headword books (Mawer / Bannister / Ekwall style).

    Pattern: a line containing one or more all-caps words followed by an
    integer (e.g. 'BACKWORTH 9', 'ABBEY DORE 1'). Returns sorted-by-offset
    pairs. Skeat-style books use a different convention — see
    `parse_skeat_section_header_pages`. Returns an empty list when no
    headers match.
    """
    return [(m.start(), int(m.group(2))) for m in _RUNNING_HEADER_RE.finditer(text)]


def parse_skeat_section_header_pages(text: str) -> list[tuple[int, int]]:
    """Extract (offset, page_number) pairs from Skeat-style running
    headers (`§ N. NAMES ENDING IN -X. <page>`).

    Returns the same shape as `parse_running_header_pages` so
    `page_for_offset` works on either parser's output. Returns an empty
    list when no headers match — caller can then try the Mawer parser
    or report `no_headers`.
    """
    return [(m.start(), int(m.group(2))) for m in _SKEAT_SECTION_HEADER_RE.finditer(text)]


def detect_running_headers(text: str) -> tuple[list[tuple[int, int]], str]:
    """Try both header conventions and return (headers, parser_name)
    for whichever produced more matches. Per-source-book dispatch:
    Mawer-style and Skeat-§ are mutually exclusive in practice, so
    'highest yield wins' is unambiguous. Returns (`[]`, `"none"`) when
    neither convention matches.

    Tie-break: when both parsers return the SAME non-zero count
    (theoretically possible with crafted OCR; not seen in the live
    corpus), Mawer wins. Mawer is the older / more permissive
    pattern; defaulting to it on ties keeps page-resolution behavior
    bit-stable for any source already routed through it."""
    mawer = parse_running_header_pages(text)
    skeat = parse_skeat_section_header_pages(text)
    if not mawer and not skeat:
        return [], "none"
    if len(skeat) > len(mawer):
        return skeat, "skeat-§"
    return mawer, "mawer"


def page_for_offset(headers: list[tuple[int, int]], offset: int) -> int | None:
    """Find the page number for a body character at `offset`. Returns the
    page from the closest preceding running header, or None if no header
    precedes the offset (e.g. the entry sits in the front matter before
    the first numbered page). `headers` must be sorted by offset, which
    is the natural output of `parse_running_header_pages`."""
    if not headers:
        return None
    page = None
    for hdr_offset, hdr_page in headers:
        if hdr_offset > offset:
            break
        page = hdr_page
    return page


def _normalize_for_quote_match(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to a single space. Returns
    (normalized_text, norm_to_orig) where norm_to_orig[i] is the
    original-text offset of normalized character i — so a substring
    found in normalized space can be translated back to an original
    offset (which is the coordinate system parse_running_header_pages
    returns)."""
    out: list[str] = []
    norm_to_orig: list[int] = []
    in_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_ws:
                out.append(" ")
                norm_to_orig.append(i)
                in_ws = True
        else:
            out.append(ch)
            norm_to_orig.append(i)
            in_ws = False
    return "".join(out), norm_to_orig


def _quote_body_excerpt(quote: str) -> str:
    """Strip the provider-attribution prefix the mining writer prepends.

    Format produced by `assemble_extraction_result`:
      `extracted_by:<provider>:<model>; <LLM commentary> | <body excerpt>`

    Body excerpt is what the parser pulled from the source and is the
    only part that exists in source_text. Returns the substring after
    the FIRST ` | ` separator; if absent (legacy citations or
    commentary-only rows), returns the whole quote so the caller can
    still try a literal match.

    First-` | ` (not last) because the prefix `extracted_by:...; <commentary>`
    never contains the separator, while OCR body excerpts can pick up
    spurious ` | ` from table rules / page-number artefacts. A leftmost
    split is unambiguous about where the body actually starts."""
    sep = " | "
    idx = quote.find(sep)
    if idx < 0:
        return quote
    return quote[idx + len(sep) :]


def backfill_citation_pages(
    db: LexiconDB,
    source_id: str,
    source_text: str,
    *,
    apply: bool = False,
) -> dict[str, int]:
    """Backfill etymon_citation.page and toponym_etymology.page from
    running headers in the source body (wyrd-azv, wyrd-8st).

    For each row in ``source_id`` where page IS NULL, strip the provider-
    attribution prefix from the row's quoted excerpt (everything before
    the last ` | `), normalize whitespace (the parser collapses OCR
    multi-spaces before sending to the LLM, so the stored quote uses
    single-spacing while the source has the original OCR runs), find
    the normalized excerpt in normalized source_text, translate the
    match offset back to original coordinates, and look up the page
    from the header sequence. `detect_running_headers` chooses between
    Mawer-style and Skeat-§ conventions per source.

    Returns counts:
      - citations_updated, etymologies_updated: rows whose page was
        successfully resolved (or would be, in dry-run).
      - quote_not_in_text: rows whose excerpt did not appear in
        source_text (OCR drift, commentary-only quotes). Summed across
        both tables.
      - no_quote: rows with NULL/empty excerpt — nothing to anchor on.
      - before_first_page: offset preceded the first running header.
      - no_headers: 1 if neither convention produced any headers; else
        0. On no_headers, the function returns immediately without
        touching any rows.

    Idempotent: only operates on rows where page IS NULL, so a re-run
    is a no-op for already-resolved rows.
    """

    counts = {
        "citations_updated": 0,
        "etymologies_updated": 0,
        "quote_not_in_text": 0,
        "no_quote": 0,
        "before_first_page": 0,
        "no_headers": 0,
    }
    headers, _parser = detect_running_headers(source_text)
    if not headers:
        counts["no_headers"] = 1
        return counts

    norm_text, norm_to_orig = _normalize_for_quote_match(source_text)

    def _resolve_page(quote: str | None) -> tuple[int | None, str]:
        if not quote:
            return None, "no_quote"
        body = _quote_body_excerpt(quote).strip()
        if not body:
            return None, "no_quote"
        norm_quote, _ = _normalize_for_quote_match(body)
        norm_quote = norm_quote.strip()
        if not norm_quote:
            return None, "no_quote"
        norm_offset = norm_text.find(norm_quote)
        if norm_offset < 0:
            return None, "quote_not_in_text"
        orig_offset = norm_to_orig[norm_offset]
        page = page_for_offset(headers, orig_offset)
        if page is None:
            return None, "before_first_page"
        return page, "ok"

    _backfill_table_pages(
        db,
        "etymon_citation",
        "short_quote",
        source_id,
        _resolve_page,
        apply,
        counts,
        "citations_updated",
    )
    _backfill_table_pages(
        db,
        "toponym_etymology",
        "notes",
        source_id,
        _resolve_page,
        apply,
        counts,
        "etymologies_updated",
    )

    if apply:
        db.commit()
    return counts


def _backfill_table_pages(
    db: LexiconDB,
    table: str,
    quote_col: str,
    source_id: str,
    resolve_fn,
    apply: bool,
    counts: dict[str, int],
    updated_key: str,
) -> None:
    """Iterate page-NULL rows of `table` for `source_id`, resolve each
    via `resolve_fn(quote)`, and update `table.page` if apply is set.
    Mutates `counts` in place: bumps `updated_key` on resolution, or the
    status key returned by resolve_fn on skip. Table identifiers are
    code-controlled (callers in this module only), so f-string SQL is
    safe here."""
    rows = db.conn.execute(
        f"SELECT id, {quote_col} AS quote FROM {table} WHERE source_id = ? AND page IS NULL",
        (source_id,),
    ).fetchall()
    for row in rows:
        page, status = resolve_fn(row["quote"])
        if status != "ok":
            counts[status] += 1
            continue
        if apply:
            db.conn.execute(
                f"UPDATE {table} SET page = ? WHERE id = ?",
                (str(page), row["id"]),
            )
        counts[updated_key] += 1


def _migrate_citation_context_snippet(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Add etymon_citation.context_snippet to existing DBs (wyrd-9kh.3).

    Idempotent: PRAGMA-checks the column before adding. Fresh installs
    pick the column up from data/lexicon.sql so this is a migration-only
    path.
    """
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon_citation)")}
    if "context_snippet" not in cols:
        db.conn.execute("ALTER TABLE etymon_citation ADD COLUMN context_snippet TEXT")
        applied["etymon_citation.context_snippet"] = True


def _migrate_toponym_etymology_attested_year(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Add toponym_etymology.attested_year + index to existing DBs
    (wyrd-bag — D5-1 expansion).

    Idempotent: PRAGMA-checks the column before adding. Fresh installs
    pick the column up from data/lexicon.sql so this is a migration-only
    path.
    """
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(toponym_etymology)")}
    if "attested_year" not in cols:
        db.conn.execute("ALTER TABLE toponym_etymology ADD COLUMN attested_year INTEGER")
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_toponym_etymology_year "
            "ON toponym_etymology(attested_year)"
        )
        applied["toponym_etymology.attested_year"] = True


def _create_mining_run_table(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Create mining_run if missing. Per D23, this audit table closes the
    'stop losing accept/decline/reject counts to stdout' gap. One row per
    (book, provider, model, started_at) tuple; the writer is invoked at
    the end of every mine-llm and review run.

    Schema:
      id           - autoincrement
      source_id    - FK to source.id (the book that was mined)
      provider     - 'ollama' | 'gemini' | 'anthropic' | 'unknown'
      model        - free-text model id (e.g. 'qwen3.5:9b', 'gemini-2.5-flash')
      mode         - 'mine' | 'review' — distinguishes Tier 1 from Tier 2
      started_at   - ISO timestamp; null if not recorded
      completed_at - ISO timestamp; null if not recorded
      parsed_count - how many entries were fed to the model
      accepted     - extracted etymology with elements
      declined     - model returned found:false
      rejected     - extraction failed validation (form-in-body, etc.)
      by_failure   - JSON object of {reason: count}; null if not recorded
      notes        - free-text context, e.g. 're-run after parser fix'
    """
    existing = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "mining_run" in existing:
        return
    db.conn.executescript(
        """
        CREATE TABLE mining_run (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          source_id     TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
          provider      TEXT NOT NULL,
          model         TEXT NOT NULL,
          mode          TEXT NOT NULL CHECK (mode IN ('mine', 'review')),
          started_at    TEXT,
          completed_at  TEXT,
          parsed_count  INTEGER NOT NULL DEFAULT 0,
          accepted      INTEGER NOT NULL DEFAULT 0,
          declined      INTEGER NOT NULL DEFAULT 0,
          rejected      INTEGER NOT NULL DEFAULT 0,
          by_failure    TEXT,
          notes         TEXT,
          UNIQUE (source_id, provider, model, mode, completed_at)
        );
        CREATE INDEX idx_mining_run_source ON mining_run(source_id);
        CREATE INDEX idx_mining_run_completed ON mining_run(completed_at);
        """
    )
    applied["mining_run_table"] = True


def record_mining_run(
    db: LexiconDB,
    *,
    source_id: str,
    provider: str,
    model: str,
    mode: str,
    parsed_count: int,
    accepted: int,
    declined: int,
    rejected: int,
    by_failure: dict[str, int] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    notes: str | None = None,
) -> int:
    """Persist one mining-run summary row. Idempotent on
    (source_id, provider, model, mode, completed_at) — a re-run with
    the same completed_at gets ignored. completed_at IS NULL means
    "not yet finished" and won't dedupe (each call gets a new row).
    """
    # sort_keys keeps the serialized form stable across runs so log-style
    # diffs / human inspection of the audit table behave deterministically
    # regardless of dict insertion order.
    by_failure_json = json.dumps(by_failure, sort_keys=True) if by_failure else None
    # ON CONFLICT(...) DO NOTHING (vs INSERT OR IGNORE) limits the silence
    # to UNIQUE conflicts on the dedupe key. CHECK / NOT NULL / FK
    # violations still raise — bad mode values must fail loudly rather
    # than silently dropping the row.
    cur = db.conn.execute(
        """
        INSERT INTO mining_run
          (source_id, provider, model, mode, started_at, completed_at,
           parsed_count, accepted, declined, rejected, by_failure, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, provider, model, mode, completed_at) DO NOTHING
        """,
        (
            source_id,
            provider,
            model,
            mode,
            started_at,
            completed_at,
            parsed_count,
            accepted,
            declined,
            rejected,
            by_failure_json,
            notes,
        ),
    )
    db.commit()
    # rowcount is 1 on insert, 0 on UNIQUE-conflict skip. Returning 0 lets
    # callers (e.g. import-mining-log) distinguish "inserted" from "duplicate".
    return cur.lastrowid if cur.rowcount else 0


def migrate_schema(db: LexiconDB) -> dict[str, bool]:
    """Bring an existing lexicon DB up to the current schema.

    Idempotent — running on an already-migrated DB is a no-op. Returns a
    dict of {migration_name: applied?} so callers can report.

    Each helper is independent and operates on its own slice of the
    schema; reading them in order shows the full migration surface.
    """
    applied = {
        "etymon.lemma_id": False,
        "etymon.inflection": False,
        "etymon.merged_into_id": False,
        "etymon.lemma_method": False,
        "etymon.synset_id": False,
        "etymon.synset_method": False,
        "idx_etymon_lemma": False,
        "idx_etymon_merged_into": False,
        "idx_etymon_synset": False,
        "etymon_text_match.method": False,
        "etymon_text_match.attested_year": False,
        "etymon_consensus_view": False,
        "etymon_canonical_view": False,
        "etymon_text_match_table": False,
        "etymon_descent_table": False,
        "mining_run_table": False,
        "etymon_citation.context_snippet": False,
        "toponym_etymology.attested_year": False,
    }
    _add_etymon_columns(db, applied)
    _create_etymon_indexes(db, applied)
    _rebuild_etymon_views(db, applied)
    _migrate_text_match_table(db, applied)
    _create_etymon_descent_table(db, applied)
    _create_mining_run_table(db, applied)
    _migrate_citation_context_snippet(db, applied)
    _migrate_toponym_etymology_attested_year(db, applied)
    db.commit()
    return applied


# --- lemma linkage ---------------------------------------------------------

# Per-language inflection-stripping rules. Each entry is a (suffix, label)
# pair: the suffix to strip from the inflected form, and the label that
# describes the inflection. Order matters — longer/more-specific suffixes
# are tried first so we don't match a substring of a longer suffix.
#
# We keep these conservative: only widely-attested inflectional suffixes,
# and we still require the stem to match an existing etymon (we don't
# fabricate lemmas in this pass — that's a follow-up).
INFLECTION_RULES: dict[str, list[tuple[str, str]]] = {
    "old-english": [
        ("ena", "genitive_pl"),
        ("ana", "genitive_pl"),
        ("um", "dative_pl"),
        ("an", "weak_oblique"),  # gen sg, dat sg, acc sg, nom/acc pl of weak nouns
        ("as", "plural_strong"),
        ("es", "genitive_strong"),
        ("ar", "plural"),
        # NOTE: "-u" plural-neuter is DELIBERATELY OMITTED. OE has many
        # u-stem nouns where -u is the lemma (wudu "wood", sunu "son", lagu
        # "law", duru "door", felu "many"). The strip rule produced too many
        # false-positive linkages. We accept losing a small number of real
        # neuter plurals (scipu "ships") rather than corrupt u-stem lemmas.
        ("e", "dative_or_pl"),
    ],
    "old-norse": [
        ("ar", "genitive_or_pl"),
        ("ir", "plural"),
        ("um", "dative_pl"),
        ("ur", "nominative"),
        ("r", "nominative"),
        ("i", "dative_or_weak"),
    ],
}


def derive_lemma_candidate(inflected_form: str, language: str) -> tuple[str, str] | None:
    """Try to identify the lemma form for an inflected etymon by stripping
    a known inflectional suffix.

    Returns (lemma_form, inflection_label) or None. Conservative: only
    suggests a lemma if the resulting stem is at least 3 characters.
    """
    if not inflected_form:
        return None
    rules = INFLECTION_RULES.get(language, [])
    s = inflected_form.lower()
    for suffix, label in rules:
        if s.endswith(suffix) and len(s) - len(suffix) >= 3:
            return s[: -len(suffix)], label
    return None


def link_lemmas(db: LexiconDB, *, apply: bool = False) -> dict:
    """Walk unlinked etymons and link them to a matching lemma row.

    For each etymon with `lemma_id IS NULL`, try every applicable
    inflection-strip rule. If the resulting stem matches an EXISTING etymon
    row in the same language, set lemma_id and inflection. Skips:
      - etymons that already have a lemma_id
      - cases where the stripped stem doesn't match any existing etymon
        (we don't fabricate new lemmas in this pass)
      - self-linkage (an etymon already serving as a lemma)

    With apply=False (default) reports candidates without writing.
    """
    # Pull every unlinked, non-tombstoned etymon once. The same set is
    # both the candidate-lemma source (potential link targets) and the
    # candidate-inflected-row source (rows that might need a link). We
    # exclude OCR-merge tombstones (merged_into_id IS NOT NULL) from
    # both sides: linking an inflected variant to a tombstone would
    # create a child → tombstone → canonical chain that the consensus
    # rollup would split, and tombstones themselves don't need linking.
    candidate_rows = list(
        db.conn.execute(
            "SELECT id, canonical_form, language FROM etymon "
            "WHERE lemma_id IS NULL AND merged_into_id IS NULL"
        )
    )
    lemmas_by_key: dict[tuple[str, str], int] = {
        (row["canonical_form"].lower(), row["language"]): row["id"] for row in candidate_rows
    }
    proposed: list[dict] = []
    inflected_rows = candidate_rows
    for row in inflected_rows:
        match = derive_lemma_candidate(row["canonical_form"], row["language"])
        if match is None:
            continue
        lemma_form, inflection = match
        lemma_id = lemmas_by_key.get((lemma_form, row["language"]))
        if lemma_id is None or lemma_id == row["id"]:
            continue
        proposed.append(
            {
                "inflected_id": row["id"],
                "inflected_form": row["canonical_form"],
                "lemma_id": lemma_id,
                "lemma_form": lemma_form,
                "inflection": inflection,
                "language": row["language"],
            }
        )

    if apply:
        for p in proposed:
            db.conn.execute(
                "UPDATE etymon SET lemma_id = ?, inflection = ?, "
                "lemma_method = 'link-lemmas-v1' WHERE id = ?",
                (p["lemma_id"], p["inflection"], p["inflected_id"]),
            )
        db.commit()

    return {
        "candidates": len(proposed),
        "sample": proposed[:25],
        "applied": apply,
    }


# --- OCR variant clustering ------------------------------------------------


# wyrd-9kh.4: snippet window around each match position. ±100 chars gives
# the SPA citation panel enough scholarly prose to surround the matched
# form without bloating etymon_text_match rows. Used by both reverse-
# search (exact form match) and fuzzy-search (Levenshtein-1 with gloss
# anchor); the gloss-anchor window is a separate concern owned by
# fuzzy_search_attestations' `gloss_window` parameter.
_TEXT_MATCH_SNIPPET_RADIUS = 100


def reverse_search_attestations(
    db: LexiconDB,
    sources_dir: Path | str,
    *,
    apply: bool = False,
    min_form_length: int = 4,
) -> dict:
    """Reverse-direction verification: for each rando-port etymon with no
    scholarly citation, search the bundled source-book texts for the form
    as a word-boundary substring. Mentions of the form (in any book) count
    as a 'search-attested' citation — looser evidence than extraction but
    still meaningful confirmation that the form appears in published
    philology.

    The function is conservative:
    - Only operates on etymons with citations from rando-port AND no other
      source. Etymons already cross-corroborated are skipped.
    - Requires the form to be at least `min_form_length` characters
      (default 4). Shorter forms produce too many false-positive substring
      matches in normal English text.
    - Word-boundary regex match (\\b<form>\\b on a normalized version of
      the body) — won't match "ham" inside "hammer" or "hamlet."

    With apply=False (default) reports candidates without writing.
    With apply=True inserts an etymon_citation row for each match,
    pointing at a synthetic source `search-attested:<book>` so we can
    distinguish search-evidence from extraction-evidence.

    Returns a dict of {etymon_id: [(source_id, count), ...]} for matches.
    """
    sources_path = Path(sources_dir)
    if not sources_path.is_dir():
        raise ValueError(f"sources_dir not found: {sources_path}")

    # Find rando-only etymons (cited by rando-port and ONLY rando-port).
    # Exclude modern-english etymons by default — those are real English words
    # like 'with', 'north', 'great', 'long', 'bishop' that appear thousands of
    # times in normal prose and produce vast amounts of noise. They're also
    # not 'unverified' in any meaningful linguistic sense; they're modern
    # vocabulary used as place-name modifiers.
    cur = db.conn.execute(
        """
        SELECT e.id, e.canonical_form, e.language
        FROM etymon e
        WHERE e.language != 'modern-english'
          AND EXISTS (
            SELECT 1 FROM etymon_citation c
            WHERE c.etymon_id = e.id AND c.source_id = 'rando-port'
          )
          AND NOT EXISTS (
            SELECT 1 FROM etymon_citation c
            WHERE c.etymon_id = e.id AND c.source_id != 'rando-port'
          )
        """
    )
    candidates = [
        (row["id"], row["canonical_form"], row["language"])
        for row in cur.fetchall()
        if len(row["canonical_form"]) >= min_form_length
    ]

    # Load and lowercase each source file's text once.
    source_texts: dict[str, str] = {}
    for f in sources_path.glob("*.txt"):
        text = f.read_text(errors="replace").lower()
        # Strip diacritics in the body too — rando lemmas are often stored
        # ASCII but the source has macrons/æ/ð. We lowercase + ASCII-fold
        # via the same normalize_ocr_form pass we use elsewhere.
        text = normalize_ocr_form(text)
        source_texts[f.stem] = text

    # Build a quick lookup so the sample report can name the etymons.
    forms_by_id = {eid: form for eid, form, _ in candidates}

    # matches[etymon_id] = [(source_id, count, sample_snippet), ...]
    # Snippet is ±_TEXT_MATCH_SNIPPET_RADIUS chars around the FIRST match
    # in that source — gives the SPA citation panel enough scholarly prose
    # to display without storing every occurrence.
    matches: dict[int, list[tuple[str, int, str]]] = {}
    for etymon_id, form, _language in candidates:
        norm = normalize_ocr_form(form)
        if len(norm) < min_form_length:
            continue
        pattern = re.compile(r"\b" + re.escape(norm) + r"\b")
        for source_id, text in source_texts.items():
            hits = list(pattern.finditer(text))
            if not hits:
                continue
            first = hits[0]
            start = max(0, first.start() - _TEXT_MATCH_SNIPPET_RADIUS)
            end = min(len(text), first.end() + _TEXT_MATCH_SNIPPET_RADIUS)
            snippet = text[start:end].strip()
            # Mark the matched form within the snippet for app display.
            snippet = snippet.replace(norm, f"«{norm}»", 1)
            matches.setdefault(etymon_id, []).append((source_id, len(hits), snippet))

    written = 0
    if apply:
        # Write to the parallel etymon_text_match table — kept separate from
        # etymon_citation so the consensus view stays a measure of
        # extraction-witness count only. Search-evidence is loose-confidence
        # and should be presented to users as "also mentioned in" rather than
        # equated with peer-reviewed scholarly extraction.
        #
        # source_id points at the REAL source row (e.g. mawer_1920_…), not a
        # synthetic one. The fact that this is search-evidence (not
        # extraction-evidence) is encoded by which table the row lives in.
        #
        # Some books in sources/ haven't been mined yet, so their source row
        # doesn't exist yet. Upsert a stub so the FK constraint passes.
        all_source_ids = set()
        for hits in matches.values():
            for sid, _count, _snip in hits:
                all_source_ids.add(sid)
        for sid in all_source_ids:
            db.upsert_source(
                id=sid,
                title=sid.replace("_", " ").title(),
                notes=(
                    "Source row created by reverse-search; the book exists "
                    "in sources/ but may not have been LLM-mined yet. Full "
                    "metadata fills in when the book is formally mined."
                ),
            )

        for etymon_id, hits in matches.items():
            form = forms_by_id.get(etymon_id, "")
            matched_form = normalize_ocr_form(form)
            for source_id, count, snippet in hits:
                db.conn.execute(
                    """
                    INSERT INTO etymon_text_match
                        (etymon_id, source_id, matched_form, match_count, edit_distance, snippet)
                    VALUES (?, ?, ?, ?, 0, ?)
                    ON CONFLICT(etymon_id, source_id, matched_form)
                    DO UPDATE SET
                        match_count = excluded.match_count,
                        snippet = excluded.snippet
                    """,
                    (etymon_id, source_id, matched_form, count, snippet),
                )
                written += 1
        db.commit()

    # Build a quick lookup so the sample report can name the etymons.
    forms_by_id = {eid: form for eid, form, _ in candidates}

    # PARSER-BUG DIAGNOSTIC: for each etymon found in source text, count
    # how often it actually appeared in extracted etymology rows. A large
    # gap (lots of text appearances, few extractions) is evidence that the
    # LLM extractor is systematically missing that morpheme.
    extraction_counts: dict[int, int] = {}
    for etymon_id in matches:
        cur = db.conn.execute(
            "SELECT COUNT(*) AS n FROM toponym_etymology_element WHERE etymon_id = ?",
            (etymon_id,),
        )
        extraction_counts[etymon_id] = cur.fetchone()["n"]

    # Score each match: ratio of text appearances to extraction appearances.
    # Flag etymons with text >> extractions as potential pipeline gaps.
    parser_bug_suspects = []
    for etymon_id, hits in matches.items():
        text_count = sum(c for _, c, _ in hits)
        ext_count = extraction_counts.get(etymon_id, 0)
        if text_count >= 10 and ext_count == 0:
            parser_bug_suspects.append(
                {
                    "etymon_id": etymon_id,
                    "form": forms_by_id.get(etymon_id),
                    "text_count": text_count,
                    "extraction_count": ext_count,
                    "books": [s for s, _, _ in hits],
                }
            )
    parser_bug_suspects.sort(key=lambda x: -x["text_count"])

    return {
        "rando_only_candidates": len(candidates),
        "etymons_with_match": len(matches),
        "total_match_records": sum(len(v) for v in matches.values()),
        "written": written,
        "parser_bug_suspects": parser_bug_suspects[:30],
        "sample": [
            {
                "etymon_id": eid,
                "form": forms_by_id.get(eid),
                "matches": hits[:5],
            }
            for eid, hits in list(matches.items())[:25]
        ],
    }


def levenshtein(a: str, b: str, *, max_distance: int | None = None) -> int:
    """Compute Levenshtein edit distance between two strings.

    Pure-python DP implementation. If max_distance is given and the early-
    bound row minimum exceeds it, returns a value > max_distance without
    finishing — the caller can use it as a fast 'too far' rejection.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > (max_distance if max_distance is not None else max(la, lb)):
        return abs(la - lb)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        if max_distance is not None and min(curr) > max_distance:
            return max_distance + 1
        prev = curr
    return prev[lb]


def fuzzy_search_attestations(
    db: LexiconDB,
    sources_dir: Path | str,
    *,
    apply: bool = False,
    min_form_length: int = 5,
    max_distance: int = 1,
    gloss_window: int = 100,
) -> dict:
    """Fuzzy variant of reverse_search_attestations.

    For each rando-only etymon WITHOUT an exact text match, search the
    source-book texts for tokens within Levenshtein distance N of the
    canonical form. To guard against meaningless edit-distance collisions
    (e.g. rando 'bere' ≠ source 'bera' even though they're 1 edit apart),
    require that one of the etymon's glosses appears within ±gloss_window
    chars of the candidate's first occurrence.

    Conservative defaults:
      max_distance=1 — distance=2 produces too many false positives
      min_form_length=5 — short forms have too many fuzzy neighbors
      gloss_window=100 chars — scholarly entries usually gloss inline

    Writes to etymon_text_match with edit_distance > 0 so app/queries can
    distinguish fuzzy from exact matches. Returns a dict of stats and
    sample matches.
    """
    sources_path = Path(sources_dir)
    if not sources_path.is_dir():
        raise ValueError(f"sources_dir not found: {sources_path}")

    # Find rando-only etymons that have no existing text match yet, plus
    # their glosses (we need at least one gloss to anchor meaning).
    cur = db.conn.execute(
        """
        SELECT e.id, e.canonical_form, e.language,
               GROUP_CONCAT(g.gloss, '|') AS glosses
        FROM etymon e
        LEFT JOIN etymon_gloss g ON g.etymon_id = e.id
        WHERE e.language != 'modern-english'
          AND EXISTS (
            SELECT 1 FROM etymon_citation c
            WHERE c.etymon_id = e.id AND c.source_id = 'rando-port'
          )
          AND NOT EXISTS (
            SELECT 1 FROM etymon_citation c
            WHERE c.etymon_id = e.id AND c.source_id != 'rando-port'
          )
          AND NOT EXISTS (
            SELECT 1 FROM etymon_text_match m
            WHERE m.etymon_id = e.id AND m.edit_distance = 0
          )
        GROUP BY e.id
        """
    )
    candidates = []
    for row in cur.fetchall():
        if not row["glosses"]:
            continue  # no gloss → can't verify meaning, skip
        form = row["canonical_form"]
        if len(form) < min_form_length:
            continue
        glosses = [g.lower() for g in row["glosses"].split("|") if g]
        candidates.append((row["id"], form, glosses))

    # Load source texts (lowercased + ocr-normalized for matching)
    source_texts: dict[str, str] = {}
    for f in sources_path.glob("*.txt"):
        text = f.read_text(errors="replace").lower()
        source_texts[f.stem] = normalize_ocr_form(text)

    # Build per-source vocabulary (unique alphabetic tokens, length-bucketed)
    # so we can fuzzy-match against the small set of plausibly-similar tokens
    # without scanning the entire text for every etymon.
    token_re = re.compile(r"[a-z]{3,}")
    vocab_by_source: dict[str, list[str]] = {}
    for source_id, text in source_texts.items():
        tokens = set(token_re.findall(text))
        vocab_by_source[source_id] = sorted(tokens)

    # Build the set of canonical forms (OCR-normalized + lowercased) so
    # we can suppress fuzzy claims where the body word is itself a
    # canonical etymon. Per wyrd-c3x: the gloss anchor too easily fires
    # on generic glosses ("land", "area", "district") near body words
    # that are independently attested as their own etymons (herath ↔ heath).
    # OCR variants between two etymons should be merged by normalize-ocr
    # upstream, not connected through fuzzy-search.
    other_canonicals: set[str] = {
        normalize_ocr_form(r["canonical_form"])
        for r in db.conn.execute("SELECT canonical_form FROM etymon WHERE merged_into_id IS NULL")
    }

    matches: dict[int, list[tuple[str, str, int, int, str]]] = {}
    # value: (source_id, matched_form, distance, count, snippet)
    for etymon_id, form, glosses in candidates:
        norm_form = normalize_ocr_form(form)
        for source_id, vocab in vocab_by_source.items():
            text = source_texts[source_id]
            # Fast filter: only consider tokens within ±2 length and starting
            # with the same character (cheap heuristic, good for OE-style
            # short morphemes that vary in their tail).
            for tok in vocab:
                if abs(len(tok) - len(norm_form)) > max_distance:
                    continue
                if tok == norm_form:
                    continue  # exact match — handled by reverse_search
                # If `tok` is itself a canonical etymon, it's not a fuzzy
                # variant — it's its own thing. See wyrd-c3x. Cheap O(1)
                # lookup gates the expensive Levenshtein call.
                if tok in other_canonicals:
                    continue
                d = levenshtein(norm_form, tok, max_distance=max_distance)
                if d > max_distance or d == 0:
                    continue
                # Found a fuzzy candidate. Now verify meaning: does any gloss
                # appear within ±gloss_window chars of this token's first
                # occurrence?
                pattern = re.compile(r"\b" + re.escape(tok) + r"\b")
                m = pattern.search(text)
                if not m:
                    continue
                start = max(0, m.start() - gloss_window)
                end = min(len(text), m.end() + gloss_window)
                window_text = text[start:end]
                if not any(g in window_text for g in glosses):
                    continue  # meaning didn't anchor — skip
                # Record. snippet shows the matched form with marker.
                snip_start = max(0, m.start() - _TEXT_MATCH_SNIPPET_RADIUS)
                snip_end = min(len(text), m.end() + _TEXT_MATCH_SNIPPET_RADIUS)
                snippet = text[snip_start:snip_end].strip().replace(tok, f"«{tok}»", 1)
                count = len(pattern.findall(text))
                matches.setdefault(etymon_id, []).append((source_id, tok, d, count, snippet))
                break  # one fuzzy match per source per etymon is enough

    forms_by_id = {eid: f for eid, f, _ in candidates}
    written = 0
    if apply:
        for etymon_id, hits in matches.items():
            for source_id, matched_form, distance, count, snippet in hits:
                db.conn.execute(
                    """
                    INSERT INTO etymon_text_match
                        (etymon_id, source_id, matched_form, match_count,
                         edit_distance, snippet, method)
                    VALUES (?, ?, ?, ?, ?, ?, 'fuzzy-search-v1')
                    ON CONFLICT(etymon_id, source_id, matched_form)
                    DO UPDATE SET
                        match_count = excluded.match_count,
                        edit_distance = excluded.edit_distance,
                        snippet = excluded.snippet,
                        method = excluded.method
                    """,
                    (etymon_id, source_id, matched_form, count, distance, snippet),
                )
                written += 1
        db.commit()

    return {
        "candidates_with_gloss": len(candidates),
        "etymons_with_fuzzy_match": len(matches),
        "total_match_records": sum(len(v) for v in matches.values()),
        "written": written,
        "sample": [
            {
                "etymon_id": eid,
                "form": forms_by_id.get(eid),
                "matches": [(s, m, d, c) for s, m, d, c, _ in hits[:5]],
            }
            for eid, hits in list(matches.items())[:25]
        ],
    }


def cluster_ocr_variants(db: LexiconDB, *, apply: bool = False) -> dict:
    """Find etymons that differ only in OCR ligature variation, mark them
    merged into a canonical winner.

    Groups every NOT-yet-merged etymon by (language,
    normalize_ocr_form(canonical_form)). Within each group of >1 row, the
    row with the LOWEST id is taken as canonical, and the others get
    `merged_into_id` pointing at the canonical row's lemma_id (if the
    winner is itself an inflected variant) or the winner itself.

    Per D22, the merge is non-destructive: citations, glosses, tags,
    text-match evidence, and reflex links stay attached to the loser
    etymon. The `etymon_consensus` view rolls citation witness counts
    up to the canonical row via the `merged_into_id` column. Gloss,
    tag, and text-match data stay attached to the original etymon ids;
    consumers wanting "all evidence for this canonical group" should
    JOIN through merged_into_id (or wait for wyrd-7lo's per-table
    canonical-rollup views).

    To revert, run `clear_enrichment(stage="ocr")` (or `UPDATE etymon
    SET merged_into_id = NULL`); the underlying mining-evidence rows
    are intact so a subsequent `cluster_ocr_variants` run with new
    heuristics will see fresh state.

    With apply=False (default) we only report what would happen — no
    writes. Pass apply=True to actually mark the merges.

    Returns a dict with keys:
      groups          - count of (lang, normalized) groups with >1 etymon
      etymons_merged  - count of redundant etymons that would be marked merged
      sample_groups   - first 20 example groupings, for human review
    """
    # Skip already-merged rows on re-run — they're tombstones, not merge
    # candidates. New heuristics should clear merged_into_id first
    # (clear_enrichment stage='ocr') before re-clustering.
    cur = db.conn.execute(
        "SELECT id, canonical_form, language, lemma_id FROM etymon "
        "WHERE merged_into_id IS NULL ORDER BY id"
    )
    groups: dict[tuple[str, str], list[tuple[int, str, int | None]]] = {}
    for row in cur:
        key = (row["language"], normalize_ocr_form(row["canonical_form"]))
        groups.setdefault(key, []).append((row["id"], row["canonical_form"], row["lemma_id"]))

    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    sample = []
    for (lang, norm), members in list(duplicate_groups.items())[:20]:
        sample.append(
            {
                "language": lang,
                "normalized": norm,
                "members": [m[1] for m in members],
                "winner": members[0][1],
            }
        )
    counts = {
        "groups": len(duplicate_groups),
        "etymons_merged": sum(len(v) - 1 for v in duplicate_groups.values()),
        "sample_groups": sample,
    }
    if not apply:
        return counts

    # Apply the merges. Lowest id wins; the rest are tombstoned via
    # merged_into_id.
    for members in duplicate_groups.values():
        winner_id, _, winner_lemma_id = members[0]
        loser_ids = [m[0] for m in members[1:]]
        # If the winner is itself an inflected variant of a lemma, point
        # losers directly at the lemma so the rollup chain stays at depth
        # 1 on the lemma_id axis.
        canonical_id = winner_lemma_id if winner_lemma_id is not None else winner_id

        # Self-reference safety: if the canonical destination is another
        # member of this same cluster (i.e., the lemma row sits in the
        # cluster too), that row must be the winner rather than a loser.
        # Otherwise we'd write loser.merged_into_id = self_id. Promote
        # the lemma; demote the previous winner.
        if canonical_id in loser_ids:
            promoted = canonical_id
            loser_ids = [lid for lid in loser_ids if lid != promoted]
            loser_ids.append(winner_id)
            winner_id = promoted
            canonical_id = winner_id

        # Chain-follow: if the canonical destination itself was merged in
        # an earlier pass (e.g., link-lemmas linked the winner to a row
        # that's since been merged), follow merged_into_id to the
        # ultimate canonical. Defends against stale lemma_id pointers
        # that predate the link_lemmas merged-tombstone filter. Tracks
        # visited ids so a malformed cycle (canonical-of-canonical points
        # back) breaks cleanly instead of looping forever.
        visited = {canonical_id}
        while True:
            next_id = db.conn.execute(
                "SELECT merged_into_id FROM etymon WHERE id = ?",
                (canonical_id,),
            ).fetchone()[0]
            if next_id is None or next_id in visited:
                break
            canonical_id = next_id
            visited.add(canonical_id)

        for loser_id in loser_ids:
            # Re-parent any inflected children the loser was acting as a
            # lemma for, so consensus rolls them into the canonical group.
            # The redirect via merged_into_id alone wouldn't suffice
            # because the rollup is single-level on lemma_id.
            db.conn.execute(
                "UPDATE etymon SET lemma_id = ? WHERE lemma_id = ?",
                (canonical_id, loser_id),
            )
            # Mark merged AND flatten any pre-existing redirects so
            # merged_into_id chains can't form. Without the second clause,
            # rows from a prior cluster pass that point at this loser
            # would create a 2-deep chain X → loser → canonical, and the
            # single-level COALESCE rollup would split witnesses.
            # Citations / glosses / tags / text-match / reflex links stay
            # attached to their original etymons; the consensus view
            # rolls them up via merged_into_id. Mining evidence (D21) is
            # preserved exactly as written.
            db.conn.execute(
                "UPDATE etymon SET merged_into_id = ? WHERE id = ? OR merged_into_id = ?",
                (canonical_id, loser_id, loser_id),
            )

    db.commit()
    return counts


# --- D5-1 / wyrd-3ux: attested-year lookup --------------------------------
#
# Post-mining stage that scans source bodies for date-citation patterns
# near each `etymon_text_match.matched_form` and records the EARLIEST
# plausible year into `etymon_text_match.attested_year`. LLM-free,
# idempotent, reversible (clear-enrichment --stage=attested-years).
#
# Year range = [_ATTESTED_YEAR_MIN, _ATTESTED_YEAR_MAX] = [100, 1700]
# (matches llm_extractor's prompt-side capture). Roman-era through pre-
# modern. Filters out scholarly publication years (most are 1800s+),
# page numbers (small integers), and most modern stray digits.
#
# Foundation for D5-2 era-cell sampling.

_ATTESTED_YEAR_MIN_LOOKUP = 100
_ATTESTED_YEAR_MAX_LOOKUP = 1700


# Form-attached year-citation pattern. Three accepted shapes, all
# requiring the year to follow the matched form within a small window of
# punctuation / whitespace:
#
#   1. "c." prefix — explicit "circa" marker (any year in range):
#         Tune c. 950        Tunna, c. 1066        Tune (c. 950)
#
#   2. Parenthesized year — distinctive citation form:
#         Tune (1086)        Tunes (1242)
#
#   3. Bare year ≥ 700 — post-Roman British/OE attestations begin
#      around then, so a 3-4 digit number ≥700 is far more likely to be
#      a real date than a page reference (page numbers in toponym
#      dictionaries cluster in the low hundreds):
#         Tune, 1086 (DB)    Tunes; 1242    Tunna 800
#
# Years <700 only qualify under (1) or (2). This filters the dominant
# false-positive class — common-word reverse-search forms (`with`,
# `wind`, `port`) followed by page references like "form, 134" — while
# admitting bona-fide Roman-era citations when they're explicitly marked
# with "c." or parens.
#
# Built fresh per matched_form because re.escape() makes it form-specific.
def _build_form_year_pattern(form: str) -> re.Pattern[str]:
    return re.compile(
        r"\b" + re.escape(form) + r"\b"
        r"(?:"
        # c./circa prefix — accept any 3-4 digit year (Roman era OK)
        r"[\s,;:.()\[\]]{0,4}c\.?\s*(\d{3,4})\b"
        r"|"
        # parenthesized year — strong citation marker
        r"[\s,;:.]*\((\d{3,4})\b"
        r"|"
        # bare year ≥ 700 — post-Roman / OE attestations. Three-digit
        # range (700-999) plus four-digit range (1000-1700) explicitly
        # listed so a 4-digit 8xxx/9xxx publication ID, ISBN fragment,
        # or page reference can't slip in. The trailing |1700 admits the
        # year-range upper bound that 1[0-6]\d{2} cuts off at 1699.
        r"[\s,;:.]{0,4}(7\d{2}|[89]\d{2}|1[0-6]\d{2}|1700)\b"
        r")"
    )


def _extract_attested_year_from_body(text: str, form: str) -> int | None:
    """Return the EARLIEST plausibly-attested year cited DIRECTLY against
    ``form`` in ``text``, or None if no qualifying citation is found.

    ``text`` is expected lowercased + OCR-normalized (the same shape the
    reverse-search snippets are stored in). ``form`` should match.

    A candidate year qualifies when:
      * The matched_form is followed (within a few chars of intervening
        punctuation / space, and optionally a "c." prefix) by a 3-4
        digit year — see _build_form_year_pattern. This is the dominant
        scholarly toponym citation shape.
      * The year parses to an integer in [_ATTESTED_YEAR_MIN_LOOKUP,
        _ATTESTED_YEAR_MAX_LOOKUP].

    "Earliest" is a deliberate v1 choice: scholarly toponym citations
    often run ascending ("Tune, 1086; Tunes, 1242; Tunne, 1340"), and the
    earliest reflects the earliest known attestation of the form. Future
    work (D5-2) may want all years, not just the earliest.
    """
    if not form:
        return None
    # ``text`` is lowercased + OCR-normalized before reaching this fn
    # (see lookup_attested_years). Lowercase the form so case-mismatched
    # matched_form values still produce hits.
    pattern = _build_form_year_pattern(form.lower())
    earliest: int | None = None
    for m in pattern.finditer(text):
        # Three capture groups in the alternation; exactly one will be
        # populated per match.
        captured = m.group(1) or m.group(2) or m.group(3)
        if captured is None:
            continue
        year = int(captured)
        if year < _ATTESTED_YEAR_MIN_LOOKUP or year > _ATTESTED_YEAR_MAX_LOOKUP:
            continue
        if earliest is None or year < earliest:
            earliest = year
    return earliest


_TOPONYM_NOTE_YEAR_PATTERN = re.compile(r"\b(7\d{2}|[89]\d{2}|1[0-6]\d{2}|1700)\b")

# A digit run preceded by one of these abbreviations is a page or
# volume reference, not a date. Page numbers in long EPNS volumes
# occasionally exceed 700, so filtering on year-range alone leaks
# them through.
#
# Matched as whole words at the END of the preceding slice — `\b`
# prevents false-skips on words that happen to end in "p." (e.g.
# "Bp." for Bishop) by requiring a word boundary before the marker.
# Substring match would also incorrectly fire "p." inside "chap.",
# but that's a no-op (chap is itself a marker); the real FP class
# was words like "Hp." or stray "p" letters at the end of the slice.
_TOPONYM_NOTE_PAGE_MARKER_RE = re.compile(
    r"\b(?:p|pp|vol|vols|ch|chap|no|nr)\.\s*$",
    re.IGNORECASE,
)


def _earliest_year_in_notes(notes: str | None) -> int | None:
    """Find the earliest plausible year (700-1700) in a
    ``toponym_etymology.notes`` value. Skips digit runs preceded by
    page / volume markers (``"p. 755"``, ``"vol. 1244"``) — those are
    bibliographic references, not dates. Returns None when no
    qualifying year appears."""
    if not notes:
        return None
    earliest: int | None = None
    for m in _TOPONYM_NOTE_YEAR_PATTERN.finditer(notes):
        ystart = m.start()
        # Pass the full prefix and rely on the regex's $ anchor to
        # match only when the marker IMMEDIATELY precedes the year.
        # A fixed-size window would miss "p.   1086" with extra spaces;
        # the linear search cost is trivial since notes is already in
        # memory and finditer call frequency is bounded by year-hit
        # count (a few per row in the worst case).
        if _TOPONYM_NOTE_PAGE_MARKER_RE.search(notes, 0, ystart):
            continue
        year = int(m.group(1))
        if earliest is None or year < earliest:
            earliest = year
    return earliest


def _scan_etymon_text_match_for_years(db: LexiconDB, sources_path: Path, *, apply: bool) -> dict:
    """Stream ``etymon_text_match`` rows ordered by source_id; for each
    row, scan the matching source body for a form-attached year
    citation (PR #47 / wyrd-3ux pattern).

    Memory characteristics:
    * ONE source body in memory at a time (the heavy thing — 100s of MB
      for big OCR'd corpora).
    * ONE row's metadata at a time during iteration.
    * candidate_updates accumulates (year, row_id) tuples for every hit
      and flushes via executemany at the end. At ~3% hit rate even a
      million-row text-match table yields ~30k tuples (~1 MB), which
      is fine; chunking the writes would be premature optimisation.
    """
    available_sources = {f.stem: f for f in sources_path.glob("*.txt")}

    cur = db.conn.execute(
        "SELECT id, source_id, matched_form FROM etymon_text_match "
        "WHERE attested_year IS NULL ORDER BY source_id"
    )

    candidate_updates: list[tuple[int, int]] = []  # (year, row_id)
    sources_missing: set[str] = set()
    rows_scanned = 0
    current_source_id: str | None = None
    text: str | None = None
    for row in cur:
        rows_scanned += 1
        if row["source_id"] != current_source_id:
            current_source_id = row["source_id"]
            source_file = available_sources.get(current_source_id)
            if source_file is None:
                sources_missing.add(current_source_id)
                text = None
                continue
            text = source_file.read_text(errors="replace").lower()
            text = normalize_ocr_form(text)
        if text is None:
            continue
        year = _extract_attested_year_from_body(text, row["matched_form"])
        if year is not None:
            candidate_updates.append((year, row["id"]))

    rows_written = 0
    if apply and candidate_updates:
        result = db.conn.executemany(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ? AND attested_year IS NULL",
            candidate_updates,
        )
        rows_written = result.rowcount
        db.commit()

    return {
        "rows_scanned": rows_scanned,
        "candidates": len(candidate_updates),
        "rows_written": rows_written,
        "sources_missing": len(sources_missing),
    }


def _scan_toponym_etymology_for_years(db: LexiconDB, *, apply: bool) -> dict:
    """wyrd-bag: scan ``toponym_etymology.notes`` for the earliest
    plausible year. Unlike the ``etymon_text_match`` scan, this doesn't
    need source-body files — the LLM-extracted notes are stored inline
    in the DB and are densely populated with scholarly citations
    (``"Tune, 1086 (DB); Tunes, 1242"``).

    Memory characteristics:
    * Full match list (candidate_updates) is held in memory before the
      executemany write. At ~80% density on the production corpus
      (~5,200 rows × 0.8 = ~4,200 tuples = ~130 KB) this is fine.
    * If a future corpus pushes this past tens of millions of rows,
      switching to chunked writes (yield + flush per N hits) would
      cap peak memory; not worth the complexity at current scale.
    """
    cur = db.conn.execute(
        "SELECT id, notes FROM toponym_etymology WHERE attested_year IS NULL AND notes IS NOT NULL"
    )

    candidate_updates: list[tuple[int, int]] = []
    rows_scanned = 0
    for row in cur:
        rows_scanned += 1
        year = _earliest_year_in_notes(row["notes"])
        if year is not None:
            candidate_updates.append((year, row["id"]))

    rows_written = 0
    if apply and candidate_updates:
        result = db.conn.executemany(
            "UPDATE toponym_etymology SET attested_year = ? WHERE id = ? AND attested_year IS NULL",
            candidate_updates,
        )
        rows_written = result.rowcount
        db.commit()

    return {
        "rows_scanned": rows_scanned,
        "candidates": len(candidate_updates),
        "rows_written": rows_written,
    }


def lookup_attested_years(
    db: LexiconDB,
    sources_dir: Path | str,
    *,
    apply: bool = False,
) -> dict:
    """Populate ``attested_year`` on the two row sources that carry
    date-citation evidence (D5-1):

    * ``etymon_text_match.attested_year`` (PR #47 / wyrd-3ux) — scans
      source bodies via the form-attached pattern. Lower density (~3%)
      because reverse-search rows are mentions, not citations.
    * ``toponym_etymology.attested_year`` (PR #5x / wyrd-bag) — scans
      LLM-extracted notes for the earliest year ≥700. Higher density
      (~80%) because the notes are scholarly date strings.

    LLM-free, idempotent, reversible. Per D21/D22 this is enrichment —
    operates on already-mined data without touching mining evidence.
    Re-runs are no-ops on rows where ``attested_year`` is already set.
    Reverse via ``clear-enrichment --stage=attested-years --apply``.

    Returns a dict with both per-source breakdowns and aggregate keys
    so existing callers (PR #47's CLI output, tests, etc.) continue
    to read ``rows_scanned`` / ``candidates`` / ``rows_written``
    unchanged while new callers can read the per-source detail.
    """
    sources_path = Path(sources_dir)
    if not sources_path.is_dir():
        raise ValueError(f"sources_dir not found: {sources_path}")

    etm = _scan_etymon_text_match_for_years(db, sources_path, apply=apply)
    te = _scan_toponym_etymology_for_years(db, apply=apply)

    return {
        # Aggregate keys (PR #47 back-compat).
        "rows_scanned": etm["rows_scanned"] + te["rows_scanned"],
        "candidates": etm["candidates"] + te["candidates"],
        "rows_written": etm["rows_written"] + te["rows_written"],
        "sources_missing": etm["sources_missing"],
        "applied": apply,
        # Per-source breakdown for richer CLI output + new tests.
        "etymon_text_match": etm,
        "toponym_etymology": te,
    }


def clear_enrichment(db: LexiconDB, *, stage: str, apply: bool = False) -> dict:
    """Reset one or more enrichment stages so they can be re-run.

    Stages:
      ocr             - UPDATE etymon SET merged_into_id = NULL
                        (un-marks all OCR-cluster merges)
      lemmas          - UPDATE etymon SET lemma_id = NULL,
                                           inflection = NULL,
                                           lemma_method = NULL
                        (un-links every inflected variant from its lemma)
      text-match      - DELETE FROM etymon_text_match
                        (drops every reverse-search / fuzzy-search row)
      cognates        - UPDATE etymon SET synset_id = NULL,
                                           synset_method = NULL
                        (un-clusters every cognate-set assignment from
                        cluster-cognates; D27 / wyrd-81n)
      attested-years  - UPDATE etymon_text_match SET attested_year = NULL
                        AND UPDATE toponym_etymology SET
                                                       attested_year = NULL
                        (drops every lookup-attested-years assignment
                        on both row sources; D5-1 / wyrd-3ux + wyrd-bag)
      all-derived     - all five of the above

    Mining evidence (etymon, etymon_citation, etymon_gloss, etymon_tag,
    etymon_descent, toponym, toponym_etymology, toponym_etymology_element)
    is never touched — that's the load-bearing data and rebuilding it
    costs hours of LLM time. Per D21, only enrichment inferences are
    reversible; the extraction layer is sacred.

    Note: stage='ocr' is *partially* reversible. It un-marks
    merged_into_id but does NOT restore the lemma_id re-parenting that
    cluster_ocr_variants applies to inflected children at merge time.
    For a fully clean revert, use stage='all-derived' (which also
    clears lemma_id) and re-run link-lemmas.

    With apply=False the dry-run reports what would change. Pass
    apply=True to actually write.
    """
    valid = {"ocr", "lemmas", "text-match", "cognates", "attested-years", "all-derived"}
    if stage not in valid:
        raise ValueError(f"unknown stage {stage!r}; must be one of {sorted(valid)}")

    stages = (
        {"ocr", "lemmas", "text-match", "cognates", "attested-years"}
        if stage == "all-derived"
        else {stage}
    )

    counts = {
        "stage": stage,
        "ocr_merges_to_clear": 0,
        "lemma_links_to_clear": 0,
        "text_match_rows_to_clear": 0,
        "synset_assignments_to_clear": 0,
        "attested_years_to_clear": 0,
    }
    if "ocr" in stages:
        counts["ocr_merges_to_clear"] = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE merged_into_id IS NOT NULL"
        ).fetchone()[0]
    if "lemmas" in stages:
        counts["lemma_links_to_clear"] = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE lemma_id IS NOT NULL"
        ).fetchone()[0]
    if "text-match" in stages:
        counts["text_match_rows_to_clear"] = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_text_match"
        ).fetchone()[0]
    if "cognates" in stages:
        counts["synset_assignments_to_clear"] = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE synset_id IS NOT NULL"
        ).fetchone()[0]
    if "attested-years" in stages:
        counts["attested_years_to_clear"] = (
            db.conn.execute(
                "SELECT COUNT(*) FROM etymon_text_match WHERE attested_year IS NOT NULL"
            ).fetchone()[0]
            + db.conn.execute(
                "SELECT COUNT(*) FROM toponym_etymology WHERE attested_year IS NOT NULL"
            ).fetchone()[0]
        )

    if not apply:
        return counts

    if "ocr" in stages:
        db.conn.execute("UPDATE etymon SET merged_into_id = NULL")
    if "lemmas" in stages:
        db.conn.execute("UPDATE etymon SET lemma_id = NULL, inflection = NULL, lemma_method = NULL")
    if "text-match" in stages:
        db.conn.execute("DELETE FROM etymon_text_match")
    if "cognates" in stages:
        db.conn.execute("UPDATE etymon SET synset_id = NULL, synset_method = NULL")
    if "attested-years" in stages:
        # 'text-match' DELETEs etymon_text_match if it's also in stages
        # (notably under 'all-derived'), so guard that UPDATE — the
        # toponym_etymology UPDATE always runs since text-match doesn't
        # touch that table.
        if "text-match" not in stages:
            db.conn.execute("UPDATE etymon_text_match SET attested_year = NULL")
        db.conn.execute("UPDATE toponym_etymology SET attested_year = NULL")
    db.commit()
    return counts


# --- D27 / wyrd-81n: cognate clustering ------------------------------------


_COGNATE_BRIDGING_EDGES = ("inheritance", "borrowing")
_CLUSTER_COGNATES_METHOD = "cluster-cognates-v1"


def cluster_cognates(db: LexiconDB, *, apply: bool = False) -> dict:
    """Walk etymon_descent inheritance + borrowing edges from each root
    and assign synset_id to every reachable descendant (D27 / wyrd-81n).

    A "root" is an etymon that participates in inheritance/borrowing
    descent edges as a parent but never as a child — i.e. the most-
    ancestral known form in its cognate chain (typically a Proto-* form).
    Every descendant reachable via inheritance or borrowing edges from
    that root gets synset_id = root.id, so cross-language cognates
    cluster behind a single canonical pointer.

    Edge type semantics (D27):
      inheritance — direct lineage. Bridges synset.
      borrowing   — borrowed across language lines. Bridges synset (a
                    borrowed word IS part of the borrowing language's
                    cognate set in practice).
      cognate     — peer relation, NOT a chain. Does not bridge — would
                    over-unify, since Wiktionary's cognate sections
                    sometimes cross probable-but-unproven boundaries.
      derivation, calque, compound, unknown — context-specific; treat
                    as non-bridging for v1, refine if mining surfaces
                    a clear case.

    Determinism: when an etymon is reachable from multiple roots (rare;
    happens when scholars disagree on the chain), the smallest root id
    wins. Iteration order is sorted by root id so the assignment is
    bit-stable across runs.

    Operates in CANONICAL space (per D22 / wyrd-223): edges'
    parent_id and child_id are resolved through merged_into_id before
    use, so a descent edge that points at an OCR-merge tombstone is
    treated as if it pointed at the canonical winner. synset_id is
    written on canonical etymons only; tombstones stay NULL (and are
    rolled up at query time via merged_into_id chain). This bridges
    cross-source canonical-form mismatches like wiktextract's
    `tun` vs the place-name dictionaries' `tūn`.

    With apply=False (default) reports candidate counts without writing.
    With apply=True writes synset_id + synset_method='cluster-cognates-v1'
    only on rows whose current (synset_id, synset_method) doesn't already
    match the target — re-runs against unchanged data become real no-ops
    instead of redundant UPDATEs. Reverse with
    `clear-enrichment --stage=cognates --apply`.

    Returns a dict of:
      - roots: number of canonical root etymons walked
      - candidates: total canonical etymons that received (or would
        receive) a synset_id assignment
      - applied: whether writes happened
      - rows_written: count of UPDATE statements that actually changed
        a row (always 0 in dry-run; ≤ candidates when applied)
      - cycle_orphans: count of canonical etymons that participate in
        bridging edges but couldn't be assigned because they sit in a
        cycle with no external root. Healthy data should report 0 here;
        non-zero means the descent graph has at least one closed loop
        with no anchor — worth flagging in operator output.
    """
    # f-string interpolation of `placeholders` is safe here:
    # _COGNATE_BRIDGING_EDGES is a module-level tuple of code-controlled
    # strings, never user input, so no SQL injection risk. Edge values
    # themselves are bound via parameters.
    placeholders = ", ".join(["?"] * len(_COGNATE_BRIDGING_EDGES))

    # Build the canonical edge set: every descent edge with both
    # endpoints resolved through merged_into_id. Returns
    # (parent_canon_id, child_canon_id) tuples. Self-loops introduced
    # by the resolution (parent and child both merge into the same
    # canonical) are filtered out — they'd waste a BFS node and offer
    # zero clustering signal.
    canonical_edges = [
        (row["parent_canon"], row["child_canon"])
        for row in db.conn.execute(
            f"""
            SELECT
              COALESCE(p.merged_into_id, d.parent_id) AS parent_canon,
              COALESCE(c.merged_into_id, d.child_id) AS child_canon
            FROM etymon_descent d
            JOIN etymon p ON p.id = d.parent_id
            JOIN etymon c ON c.id = d.child_id
            WHERE d.edge_type IN ({placeholders})
            """,
            _COGNATE_BRIDGING_EDGES,
        ).fetchall()
        if row["parent_canon"] != row["child_canon"]
    ]

    # Build child-of-parent index for the BFS. Parent → set of children.
    children_by_parent: dict[int, set[int]] = {}
    bridging_participants: set[int] = set()
    parent_set: set[int] = set()
    child_set: set[int] = set()
    for parent_id, child_id in canonical_edges:
        children_by_parent.setdefault(parent_id, set()).add(child_id)
        bridging_participants.add(parent_id)
        bridging_participants.add(child_id)
        parent_set.add(parent_id)
        child_set.add(child_id)

    # Roots: canonical etymons that appear as parent but never as
    # child in the canonical edge set.
    roots = sorted(parent_set - child_set)

    assignments: dict[int, int] = {}
    for root_id in roots:
        if root_id in assignments:
            # Already claimed by an earlier (smaller-id) root via cross-edges.
            continue
        assignments[root_id] = root_id
        frontier: list[int] = [root_id]
        while frontier:
            next_frontier: list[int] = []
            for node_id in frontier:
                for child_id in children_by_parent.get(node_id, ()):
                    if child_id not in assignments:
                        assignments[child_id] = root_id
                        next_frontier.append(child_id)
            frontier = next_frontier

    # Cycle-orphans: canonical etymons that participate in bridging
    # edges but never reached a root. A pure cycle has no anchor.
    cycle_orphans = bridging_participants - set(assignments.keys())

    rows_written = 0
    if apply:
        for etymon_id, synset_id in assignments.items():
            cur = db.conn.execute(
                "UPDATE etymon SET synset_id = ?, synset_method = ? "
                "WHERE id = ? "
                "  AND (synset_id IS NOT ? OR synset_method IS NOT ?)",
                (
                    synset_id,
                    _CLUSTER_COGNATES_METHOD,
                    etymon_id,
                    synset_id,
                    _CLUSTER_COGNATES_METHOD,
                ),
            )
            rows_written += cur.rowcount
        db.commit()

    return {
        "roots": len(roots),
        "candidates": len(assignments),
        "applied": apply,
        "rows_written": rows_written,
        "cycle_orphans": len(cycle_orphans),
    }


# --- generic-language bridging --------------------------------------------


def bridge_generic_language(
    db: LexiconDB,
    *,
    generic_lang: str,
    candidate_langs: tuple[str, ...],
    apply: bool = False,
) -> dict:
    """Bridge a generic-language etymon (e.g. 'celtic') to the matching
    specific-language entry (e.g. 'irish' / 'welsh' / 'old-irish') by
    setting merged_into_id (D22 OCR-cluster style — non-destructive).

    Place-name dictionaries often write generic language tags like
    'celtic' for morphemes whose specific Celtic-family origin isn't
    pinned (or doesn't matter to the place-name analysis). Wiktextract
    entries are language-specific. This pass bridges the lookup
    mismatch by canonicalizing each generic-language etymon onto a
    specific-language match with the same canonical_form.

    For each generic-language etymon (canonical, not already merged):
      - Look up specific-language etymons matching `LOWER(canonical_form)`
        across `candidate_langs`.
      - If matches exist, pick the highest-priority one per
        `candidate_langs` order (typically Proto-* > Old-* > Middle-* >
        modern). Within a language, pick the smallest etymon id for
        determinism.
      - Set the generic-language etymon's `merged_into_id` to the picked
        canonical winner.

    The cluster-cognates pass is already redirect-aware, so any descent
    edges that previously pointed at the specific-language etymon now
    also cluster the merged generic etymon via the merged_into_id rollup
    chain.

    With apply=False (default) reports candidate counts without writing.
    Returns:
      - generic_etymons: total canonical generic-language etymons examined
      - bridged: count that found a specific-language match (would
        merge / did merge)
      - unmatched: count with no specific-language candidate (will
        remain as standalone canonical entries)
      - rows_written: actual UPDATE row count when apply=True (always 0
        in dry-run)
    """
    if not candidate_langs:
        raise ValueError("candidate_langs must be non-empty")

    # Build a (lower(canonical_form), language) → smallest_id lookup over
    # canonical specific-language etymons. One pass over the candidate
    # set keeps the bridging O(N) on the generic-language list.
    placeholders = ", ".join(["?"] * len(candidate_langs))
    candidate_index: dict[tuple[str, str], int] = {}
    for row in db.conn.execute(
        f"""
        SELECT id, canonical_form, language
        FROM etymon
        WHERE merged_into_id IS NULL
          AND language IN ({placeholders})
        ORDER BY id
        """,
        candidate_langs,
    ).fetchall():
        key = (row["canonical_form"].lower(), row["language"])
        # First-seen wins (smallest id) per (form, lang) — within a
        # language the lower id is the older entry, deterministic.
        if key not in candidate_index:
            candidate_index[key] = row["id"]

    # Walk generic-language canonical etymons; pick the first matching
    # candidate language per the priority order in `candidate_langs`.
    generic_rows = db.conn.execute(
        "SELECT id, canonical_form FROM etymon "
        "WHERE language = ? AND merged_into_id IS NULL ORDER BY id",
        (generic_lang,),
    ).fetchall()

    bridges: list[tuple[int, int]] = []  # (generic_id, target_id)
    for gen_row in generic_rows:
        form_lower = gen_row["canonical_form"].lower()
        for cand_lang in candidate_langs:
            target_id = candidate_index.get((form_lower, cand_lang))
            if target_id is not None:
                bridges.append((gen_row["id"], target_id))
                break

    rows_written = 0
    if apply and bridges:
        # Mirror cluster_ocr_variants's chain-flatten (D22): set
        # merged_into_id on the generic row AND re-route any
        # pre-existing redirect that pointed AT the generic row.
        # Without the OR-clause a 2-deep chain X → generic → target
        # would form, and the single-level COALESCE rollup in
        # etymon_consensus / etymon_canonical would split witnesses
        # across two GROUP BY buckets.
        cur = db.conn.executemany(
            "UPDATE etymon SET merged_into_id = ? "
            "WHERE (id = ? OR merged_into_id = ?) "
            "  AND merged_into_id IS NOT ?",
            [(tid, gid, gid, tid) for gid, tid in bridges],
        )
        rows_written = cur.rowcount
        # Re-parent any inflected children the generic row was acting
        # as a lemma for. Mining evidence stays on the original etymon.
        db.conn.executemany(
            "UPDATE etymon SET lemma_id = ? WHERE lemma_id = ?",
            [(tid, gid) for gid, tid in bridges],
        )
        db.commit()

    return {
        "generic_etymons": len(generic_rows),
        "bridged": len(bridges),
        "unmatched": len(generic_rows) - len(bridges),
        "rows_written": rows_written,
        "applied": apply,
    }


# --- phonological bridging for OE place-name forms ------------------------


# Hand-curated mapping from "scholarly OE form as place-name dictionaries
# write it" → "Wiktionary canonical OE form". Wiktionary uses formal
# scholarly orthography (macrons, æ, weak-final consonants); place-name
# dictionaries write modernized / Norman-Anglicized spellings inherited
# from medieval scribal practice. The mapping captures the most common
# pairs that surface in the high-witness end of our place-name corpus.
# Extend the table when new high-witness mismatches surface.
#
# Convention: keys are lowercase scholarly forms; values are the
# Wiktionary canonical (with macrons, æ, etc.). The bridge uses
# merged_into_id (D22 non-destructive) — no mining evidence is
# destroyed.
_OE_PHONOLOGICAL_BRIDGES: dict[str, str] = {
    # -tūn / settlement family
    "ton": "tūn",
    "tun": "tūn",
    "tone": "tūn",
    # -lēah / clearing family
    "lea": "lēah",
    "leah": "lēah",
    "leak": "lēah",  # OCR / spelling variant
    "ley": "lēah",
    "ly": "lēah",
    # -cot / -cote (cottage)
    "cote": "cot",
    "cotum": "cot",  # dative-pl
    # -burh / fortified-place family
    "burgh": "burh",
    "bury": "burh",
    "byrig": "burh",  # dative-sg of burh
    # -dæl / valley
    "dale": "dæl",
    # -heall / hall
    "hall": "heall",
    # -ieg / island. Wiktionary's headword is `īeg` but the macron-stripped
    # `ieg` is what the OCR-normalize pass treats as canonical; `īeg`
    # itself is not in the corpus. Snap value to the live form.
    "ey": "ieg",
    "eg": "ieg",
    # -burna / stream
    "burn": "burna",
    "burne": "burna",
    # -healh / nook
    "hale": "healh",
    "halh": "healh",
    # -nīwe / new
    "new": "nīwe",
    # -ōra / shore, edge
    "ore": "ōra",
    # -wudu / wood
    "wood": "wudu",
    "wode": "wudu",
    # -brād / broad
    "brade": "brād",
    # -brycg / bridge
    "bridge": "brycg",
    # -stān / stone
    "stone": "stān",
    # -hyll / hill
    "hill": "hyll",
    # -hyrst / wooded hill
    "hurst": "hyrst",
    # -hlāw / mound
    "low": "hlāw",
    # -mos / moss/marsh
    "moss": "mos",
    # -pōl / pool — bridge value snapped to 'pol' (the live canonical;
    # 'pōl' is itself a tombstone merged into 'pol' by normalize-ocr).
    "pool": "pol",
    # -sealh / willow
    "salh": "sealh",
    # -stede / place
    "stead": "stede",
    # -wella / spring
    "well": "wella",
    # -ing / -ingas (people-of suffix; place-name dicts often drop hyphen)
    "ing": "-ing",
    "ingas": "-ingas",
    # æcer / acre
    "acre": "æcer",
    # æsc / ash
    "ash": "æsc",
    # -hām / homestead
    "ham": "hām",
    # -wella / spring (additional inflected/spelling variants beyond 'well')
    "wella": "welle",
    # -ing-family additions: scholar-spelled variants of the people-of suffix
    "inga": "ing",
    # -cipp / log
    "cippa": "cipp",
    # -hār / grey, hoary
    "hāran": "hār",
    # -clopp / lump, hill
    "cloppa": "clopp",
    # -ticcen / kid (young goat)
    "ticce": "ticcen",
    # -cirice / church (kirk is the Northumbrian / Norse-influenced form)
    "kirk": "cirice",
    # -healh / nook (hala is a common scholarly variant)
    "hala": "healh",
    # -scylfe / shelf, ledge
    "scelf": "scylfe",
    # -ieg / island (alternative scholarly spelling)
    "ēg": "ieg",
    # -hara / hare
    "hare": "hara",
    # -hæsel / hazel (relies on redirect-follow to canonical haesel)
    "hasel": "hæsel",
    # -hlīep / leap (Hartlepool's "harts-leap" element)
    "hlyp": "hlīep",
    # -lacu / stream (lech is also occasionally read as 'lece' = bog;
    # the place-name reading is dominantly the water sense, hence lacu)
    "lech": "lacu",
}


def bridge_phonological_oe(db: LexiconDB, *, apply: bool = False) -> dict:
    """Bridge OE place-name etymons to their Wiktionary canonical
    equivalents via a hand-curated mapping table.

    Place-name dictionaries write modernized OE forms (`ton`, `lea`,
    `burgh`, `dale`); Wiktionary uses scholarly orthography (`tūn`,
    `lēah`, `burh`, `dæl`). normalize-ocr handles macron-strip OCR
    variants but not vowel-weakening / silent-e / gh-spelling shifts.
    This pass uses `_OE_PHONOLOGICAL_BRIDGES` to merge known pairs
    via merged_into_id (D22 non-destructive shape) so the redirect-aware
    cluster-cognates pass rolls the place-name etymons up into the
    Wiktionary cognate clusters.

    Algorithm:
      1. For each canonical OE etymon, lowercase its canonical_form.
      2. Look up the form in `_OE_PHONOLOGICAL_BRIDGES`.
      3. If a target form is named, find the canonical OE etymon with
         that form (case-insensitive), and merge the place-name entry
         into it.

    With apply=False (default) reports candidate counts without writing.
    Returns:
      - examined: total canonical OE etymons examined
      - bridged: count that found a phonological-bridge target
      - unmatched: count with no entry in the phonological table
      - missing_target: count where the table named a target but no
        OE etymon exists with that canonical form (operator should
        add the target via mining or extend the table)
      - rows_written: actual UPDATE row count when apply=True
    """
    # Build target_index from ALL OE rows (canonical + tombstones), with
    # each form key resolving to its LIVE canonical id by following the
    # merged_into_id chain. Without redirect-following, the bridge map
    # cannot name a target like 'dæl' or 'pōl' that has been merged into
    # a macron-stripped canonical (dael, pol) by an earlier OCR pass —
    # the lookup would silently miss. Walking the chain in Python here
    # keeps the bridge robust against the current and future state of
    # OCR-clustering / normalize-ocr results.
    all_oe_rows = db.conn.execute(
        "SELECT id, canonical_form, merged_into_id FROM etymon "
        "WHERE language = 'old-english' ORDER BY id"
    ).fetchall()
    chain: dict[int, int | None] = {r["id"]: r["merged_into_id"] for r in all_oe_rows}

    def _resolve_canonical(start_id: int) -> int:
        cid = start_id
        visited: set[int] = set()
        while (next_id := chain.get(cid)) is not None and cid not in visited:
            visited.add(cid)
            cid = next_id
        return cid

    target_index: dict[str, int] = {}
    for row in all_oe_rows:
        key = row["canonical_form"].lower()
        if key not in target_index:
            target_index[key] = _resolve_canonical(row["id"])

    # Iteration set: only the canonical (un-merged) rows are bridge
    # candidates. A row that is itself a tombstone is already redirected
    # and bridging from it would be a no-op against the COALESCE rollup.
    examined_rows = [r for r in all_oe_rows if r["merged_into_id"] is None]

    bridges: list[tuple[int, int]] = []
    missing_target = 0
    for row in examined_rows:
        form = row["canonical_form"].lower()
        wiktionary_form = _OE_PHONOLOGICAL_BRIDGES.get(form)
        if wiktionary_form is None:
            continue
        target_id = target_index.get(wiktionary_form.lower())
        if target_id is None:
            missing_target += 1
            continue
        if target_id == row["id"]:
            continue
        bridges.append((row["id"], target_id))

    rows_written = 0
    if apply and bridges:
        # Chain-flatten + lemma-reparent in batch (mirrors
        # cluster_ocr_variants's D22 pattern). The OR-clause re-routes
        # any pre-existing redirect that pointed AT this place-name
        # etymon onto the canonical target, preventing a 2-deep chain
        # that would split witnesses in the single-level COALESCE
        # rollup used by etymon_consensus / etymon_canonical.
        cur = db.conn.executemany(
            "UPDATE etymon SET merged_into_id = ? "
            "WHERE (id = ? OR merged_into_id = ?) "
            "  AND merged_into_id IS NOT ?",
            [(tid, sid, sid, tid) for sid, tid in bridges],
        )
        rows_written = cur.rowcount
        db.conn.executemany(
            "UPDATE etymon SET lemma_id = ? WHERE lemma_id = ?",
            [(tid, sid) for sid, tid in bridges],
        )
        db.commit()

    return {
        "examined": len(examined_rows),
        "bridged": len(bridges),
        "unmatched": len(examined_rows) - len(bridges) - missing_target,
        "missing_target": missing_target,
        "rows_written": rows_written,
        "applied": apply,
    }


# --- ingest from parsed corpus entries -------------------------------------


def _upsert_toponym(db: LexiconDB, modern_name: str, region: str | None) -> int:
    cur = db.conn.execute(
        """
        SELECT id FROM toponym
        WHERE modern_name = ?
          AND COALESCE(country, '') = ''
          AND COALESCE(region, '') = COALESCE(?, '')
        """,
        (modern_name, region),
    )
    row = cur.fetchone()
    if row is not None:
        return row["id"]
    cur = db.conn.execute(
        "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
        (modern_name, region),
    )
    return cur.lastrowid


def ingest_parsed_entries(
    db: LexiconDB,
    parsed_entries: list[ParsedEntry],
    source_id: str,
    *,
    region: str | None = None,
) -> dict[str, int]:
    """Persist ParsedEntry rows from a Skeat-style parser into the DB.

    For each entry:
      - Upsert toponym row (region carries the source's coverage area).
      - For HIGH/MEDIUM confidence: insert a toponym_etymology row pointing
        at <source_id>, plus one toponym_etymology_element per parsed
        element. Each element's etymon is upserted, glosses/tags attached,
        and a citation row added.
      - For LOW confidence: only the toponym row is written, so we still
        record that the source covers this place even when we couldn't
        recover a breakdown.

    Returns a counts dict for sanity-checking.
    """
    counts = {
        "toponyms": 0,
        "etymologies": 0,
        "elements": 0,
        "etymons_touched": 0,
    }
    for entry in parsed_entries:
        toponym_id = _upsert_toponym(db, entry.toponym, region)
        counts["toponyms"] += 1

        if entry.confidence == "low" or not entry.elements:
            continue

        confidence = entry.confidence if entry.confidence in ("high", "medium", "low") else "low"
        cur = db.conn.execute(
            """
            INSERT INTO toponym_etymology
                (toponym_id, source_id, historical_form, confidence, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                toponym_id,
                source_id,
                entry.historical_form,
                confidence,
                entry.source_quote or None,
            ),
        )
        etymology_id = cur.lastrowid
        counts["etymologies"] += 1

        for ordinal, elem in enumerate(entry.elements):
            etymon_id = db.upsert_etymon(elem.form, elem.language)
            counts["etymons_touched"] += 1
            if elem.gloss:
                db.add_gloss(etymon_id, elem.gloss)
            db.add_citation(etymon_id, source_id, short_quote=entry.source_quote or None)
            db.conn.execute(
                """
                INSERT INTO toponym_etymology_element
                    (toponym_etymology_id, ordinal, etymon_id, inflection, surface_in_modern)
                VALUES (?, ?, ?, ?, ?)
                """,
                (etymology_id, ordinal, etymon_id, elem.inflection, None),
            )
            counts["elements"] += 1

    db.commit()
    return counts


# --- meanings.json export --------------------------------------------------
#
# Inverse of `seed_from_meanings`: walk the lexicon DB, apply the D4 promotion
# rule, and emit a meanings.json document the runtime can load. The runtime
# shape is unchanged (D1) — generation still consumes meanings.json — so the
# export is the bridge that lets new mining reach users.
#
# Inflections (D8) and OCR-cluster losers (D22) ride along in the lemma's
# language form list rather than getting their own subjects, so the consensus
# rollup the runtime sees matches the rollup the lexicon enforces.

# Reverse of LANGUAGE_FIELDS: lexicon language code → meanings.json field name.
# Only the canonical key is emitted; the misspelled aliases ('old _english',
# 'old_scandanavian') are accepted on ingestion but not regenerated on export.
_LANG_CODE_TO_JSON_FIELD = {
    "old-english": "old_english",
    "old-norse": "old_scandinavian",
    "norman-french": "old_french",
    "celtic": "celtic_mix",
    "latin": "latin",
    "germanic": "germanic",
    "greek": "greek",
    "modern-english": "modern_english",
    "biblical": "biblical",
}

# Per-language witness thresholds calibrated against corpus availability and
# spot-checked quality at w=2 (analysis 2026-05-02). Languages absent from
# this map fall back to the global ``min_witnesses``. Rationale per language:
#   old-english : 3 — well-mined (32 sources); strict gate keeps Tier-1
#                     prose-extraction noise out (~20% noise at w=2).
#   celtic      : 2 — Qwen weak on Celtic per D13; corpus structurally
#                     thinner per yield. Quality at w=2 ~90% clean.
#   old-norse   : 2 — only one ON-focused dictionary in the corpus; w=3
#                     filters out 93% of ON purely on corpus thinness.
#   modern-english,
#   norman-french,
#   latin,
#   biblical    : 2 — small populations at w≥2 (≤20 each); spot-check 100%
#                     clean. The cost of admitting them is negligible and
#                     the gain (modern English placename elements like mill,
#                     stone, head; NF castle, monte; Latin ecclesia,
#                     ceaster) is meaningful for generation breadth.
#   germanic, greek: nothing reaches w≥2 in the current corpus, so the
#                     threshold is moot — they ride the rando-port path only.
RECOMMENDED_LANG_THRESHOLDS: dict[str, int] = {
    "old-english": 3,
    "celtic": 2,
    "old-norse": 2,
    "modern-english": 2,
    "norman-french": 2,
    "latin": 2,
    "biblical": 2,
}


def export_meanings(
    db: LexiconDB,
    *,
    min_witnesses: int = 3,
    lang_thresholds: dict[str, int] | None = None,
    include_rando: bool = True,
) -> list[dict[str, Any]]:
    """Walk the lexicon and emit a meanings.json structure.

    Promotion rule (D4): a family root is included if either
    (a) any etymon in the family is cited by 'rando-port' AND ``include_rando``
        is true (legacy seed kept until corroborated), OR
    (b) the family's witness count (``etymon_consensus.witnesses``) is at
        least the threshold for that family's language.

    The threshold per language is taken from ``lang_thresholds`` (defaults to
    ``RECOMMENDED_LANG_THRESHOLDS``); languages absent from the map use
    ``min_witnesses``. Pass ``lang_thresholds={}`` to apply a uniform
    ``min_witnesses`` threshold across all languages.

    Family roots are computed by the same two-step rollup the
    ``etymon_consensus`` view uses (``merged_into_id`` then ``lemma_id``), so
    OCR-cluster losers and inflected variants don't get their own subjects —
    their canonical_forms ride along in the lemma's language form list (D8,
    D18, D22).

    Subjects are reconstructed by grouping family roots that share the same
    (modifier_type, glosses, tags) signature — the natural inverse of how
    seed_from_meanings shaped the original rando-port subjects. The
    reconstruction is lossy where the original meanings.json had two distinct
    subjects with identical signatures, but the runtime ``load_meanings``
    keys by ``modern_usage`` regardless of subject boundary.
    """
    if lang_thresholds is None:
        lang_thresholds = RECOMMENDED_LANG_THRESHOLDS
    families = _collect_families(
        db,
        min_witnesses=min_witnesses,
        lang_thresholds=lang_thresholds,
        include_rando=include_rando,
    )
    subjects = _group_families_into_subjects(families)
    if include_rando:
        subjects.extend(_orphan_reflex_subjects(db))
    return subjects


def _build_witness_filter(
    lang_thresholds: dict[str, int],
    min_witnesses: int,
) -> tuple[str, list[Any]]:
    """Build a SQL WHERE fragment + bind params that gate etymon_consensus
    rows by per-language thresholds with ``min_witnesses`` as the fallback.

    Returns ``(sql_fragment, params)`` where the fragment is parenthesized
    and references ``language`` / ``witnesses`` columns.
    """
    if not lang_thresholds:
        return "(witnesses >= ?)", [min_witnesses]
    clauses: list[str] = []
    params: list[Any] = []
    sorted_langs = sorted(lang_thresholds)
    for lang in sorted_langs:
        clauses.append("(language = ? AND witnesses >= ?)")
        params.extend([lang, lang_thresholds[lang]])
    placeholders = ",".join("?" * len(sorted_langs))
    clauses.append(f"(language NOT IN ({placeholders}) AND witnesses >= ?)")
    params.extend(sorted_langs)
    params.append(min_witnesses)
    return f"({' OR '.join(clauses)})", params


def _collect_families(
    db: LexiconDB,
    *,
    min_witnesses: int,
    lang_thresholds: dict[str, int],
    include_rando: bool,
) -> list[dict[str, Any]]:
    """Build per-family-root data: forms-by-language, glosses, tags, reflexes."""
    witness_sql, witness_params = _build_witness_filter(lang_thresholds, min_witnesses)
    cur = db.conn.execute(
        f"""
        WITH rollup AS (
            SELECT
                e.id AS etymon_id,
                COALESCE(le.id, target.id, e.id) AS root_id
            FROM etymon e
            LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
            LEFT JOIN etymon le ON le.id = target.lemma_id
        ),
        rando_roots AS (
            SELECT DISTINCT r.root_id
            FROM rollup r
            JOIN etymon_citation c ON c.etymon_id = r.etymon_id
            WHERE c.source_id = 'rando-port'
        ),
        promoted AS (
            SELECT lemma_id AS root_id FROM etymon_consensus WHERE {witness_sql}
            UNION
            SELECT root_id FROM rando_roots WHERE ? = 1
        )
        SELECT DISTINCT root_id FROM promoted
        ORDER BY root_id
        """,
        [*witness_params, 1 if include_rando else 0],
    )
    root_ids = [row["root_id"] for row in cur.fetchall()]

    families: list[dict[str, Any]] = []
    for root_id in root_ids:
        family = _gather_family(db, root_id)
        if family is not None and family["forms_by_lang"]:
            families.append(family)
    return families


_FAMILY_MEMBER_SQL = """
SELECT e.id, e.canonical_form, e.language, e.lemma_id, e.merged_into_id, e.inflection
FROM etymon e
LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
LEFT JOIN etymon le ON le.id = target.lemma_id
WHERE COALESCE(le.id, target.id, e.id) = ?
ORDER BY e.language, e.canonical_form
"""


def _gather_family(db: LexiconDB, root_id: int) -> dict[str, Any] | None:
    """Collect forms / glosses / tags / reflexes for one family root.

    Includes the root itself plus all etymons that roll up to it
    (inflected children, OCR-cluster losers, and combinations thereof).
    Pure orchestrator — each per-aspect aggregation lives in a focused
    helper below.
    """
    root_row = db.conn.execute(
        "SELECT canonical_form, language, modifier_type, position_pref FROM etymon WHERE id = ?",
        (root_id,),
    ).fetchone()
    if root_row is None:
        return None

    member_rows = db.conn.execute(_FAMILY_MEMBER_SQL, (root_id,)).fetchall()
    if not member_rows:
        return None

    member_ids = [r["id"] for r in member_rows]
    member_form_by_id = {r["id"]: (r["language"], r["canonical_form"]) for r in member_rows}
    member_inflection_by_id: dict[int, str | None] = {r["id"]: r["inflection"] for r in member_rows}
    canonical_forms_lower = {f.lower() for _lang, f in member_form_by_id.values()}

    reflex_links = _fetch_member_reflex_links(db, member_ids)

    return {
        "root_id": root_id,
        "root_canonical_form": root_row["canonical_form"],
        "root_language": root_row["language"],
        "modifier_type": root_row["modifier_type"],
        "position_pref": root_row["position_pref"],
        "forms_by_lang": _build_forms_by_lang(root_row, member_rows),
        "member_form_by_id": member_form_by_id,
        "member_descendants": _compute_member_descendants(member_rows),
        "member_variants": _fetch_member_variants(db, member_ids, canonical_forms_lower),
        "member_inflection_by_id": member_inflection_by_id,
        "member_citations": _fetch_member_citations(db, member_ids),
        "glosses": _fetch_member_glosses(db, member_ids),
        "tags": _fetch_member_tags(db, member_ids),
        "reflexes": _fetch_member_reflexes(db, member_ids, reflex_links),
    }


def _build_forms_by_lang(root_row: Any, member_rows: list[Any]) -> dict[str, list[str]]:
    """Group canonical forms by language, root-first.

    Root form first per language so the lemma's own canonical_form leads
    the list (predictable order for snapshot tests).
    """
    forms_by_lang: dict[str, list[str]] = {}
    forms_by_lang.setdefault(root_row["language"], []).append(root_row["canonical_form"])
    for r in member_rows:
        bucket = forms_by_lang.setdefault(r["language"], [])
        if r["canonical_form"] not in bucket:
            bucket.append(r["canonical_form"])
    return forms_by_lang


_GLOSS_SPLIT_RE = re.compile(r"\s*[,;]\s*|\s+or\s+", re.IGNORECASE)


def _fetch_member_glosses(db: LexiconDB, member_ids: list[int]) -> list[str]:
    """Distinct sorted glosses across the family's members.

    Drops glosses that are concatenations of already-present sibling
    glosses (mining noise where the LLM emitted a comma/semicolon list as
    a single gloss when the source bullet split it). Concretely, if
    'lake', 'pond' are present and 'lake, pond' also got extracted, drop
    the latter — it adds no information and clutters the explainer.
    Splitter recognizes ',', ';', and ' or '; case-insensitive on the
    'or' connector. Single-token glosses are kept unconditionally.
    """
    placeholders = ",".join("?" * len(member_ids))
    raw = [
        row["gloss"]
        for row in db.conn.execute(
            f"SELECT DISTINCT gloss FROM etymon_gloss "
            f"WHERE etymon_id IN ({placeholders}) ORDER BY gloss",
            member_ids,
        )
    ]
    return _filter_concatenation_glosses(raw)


def _filter_concatenation_glosses(glosses: list[str]) -> list[str]:
    """Drop glosses whose tokens are entirely covered by other singleton
    glosses already in the list. Pure (no DB), so the caller's order
    is preserved for the survivors."""
    singleton_set = {g.strip().lower() for g in glosses if not _GLOSS_SPLIT_RE.search(g)}
    out: list[str] = []
    for g in glosses:
        if g.strip().lower() in singleton_set and not _GLOSS_SPLIT_RE.search(g):
            out.append(g)
            continue
        tokens = [t.strip().lower() for t in _GLOSS_SPLIT_RE.split(g) if t.strip()]
        if len(tokens) > 1 and all(t in singleton_set for t in tokens):
            continue
        out.append(g)
    return out


def _fetch_member_tags(db: LexiconDB, member_ids: list[int]) -> list[str]:
    """Distinct sorted tags across the family's members."""
    placeholders = ",".join("?" * len(member_ids))
    return [
        row["tag"]
        for row in db.conn.execute(
            f"SELECT DISTINCT tag FROM etymon_tag WHERE etymon_id IN ({placeholders}) ORDER BY tag",
            member_ids,
        )
    ]


def _fetch_member_reflex_links(db: LexiconDB, member_ids: list[int]) -> dict[int, list[int]]:
    """Per-reflex linked etymon ids (within this family).

    Lets us narrow the exported language array per word at group-build
    time — without this, a subject grouping a Celtic family and an OE
    family would emit each reflex's word with both languages even when
    the reflex only links to one of them.
    """
    placeholders = ",".join("?" * len(member_ids))
    reflex_links: dict[int, list[int]] = {}
    for row in db.conn.execute(
        f"SELECT re.reflex_id, re.etymon_id "
        f"FROM reflex_etymon re "
        f"WHERE re.etymon_id IN ({placeholders}) "
        f"ORDER BY re.reflex_id, re.etymon_id",
        member_ids,
    ):
        reflex_links.setdefault(row["reflex_id"], []).append(row["etymon_id"])
    return reflex_links


def _fetch_member_reflexes(
    db: LexiconDB, member_ids: list[int], reflex_links: dict[int, list[int]]
) -> list[dict[str, Any]]:
    """Reflex rows (modern surface forms) linked to any family member."""
    placeholders = ",".join("?" * len(member_ids))
    return [
        {
            "id": row["id"],
            "surface_form": row["surface_form"],
            "position": row["position"],
            "linked_member_ids": reflex_links.get(row["id"], []),
        }
        for row in db.conn.execute(
            f"SELECT DISTINCT r.id, r.surface_form, r.position "
            f"FROM reflex r "
            f"JOIN reflex_etymon re ON re.reflex_id = r.id "
            f"WHERE re.etymon_id IN ({placeholders}) "
            f"ORDER BY r.position, r.surface_form",
            member_ids,
        )
    ]


_LOW_CONFIDENCE_METHODS = frozenset({"fuzzy-search-v1", "llm-disambiguated-v1"})


def _fetch_member_variants(
    db: LexiconDB, member_ids: list[int], canonical_forms_lower: set[str]
) -> dict[int, list[tuple[str, int]]]:
    """Spelling variant pool per member_id (D18).

    Pulls from etymon_text_match.matched_form: real attested 19th-c.
    spellings (denu/dene/denū/dená) and post-disambiguator winners. These
    are the surface-form randomization targets for archaic-feel
    generation. Dedupes against canonical_forms_lower so the pool only
    contains FORMS NEW TO THE GENERATOR — emitting "denu" when "denu" is
    also the canonical_form would be redundant.

    Drops "low-confidence singletons": variants whose total match_count
    is 1 AND every contributing row was produced by fuzzy-search or
    llm-disambiguator (vs reverse-search, which scans for canonical-form
    occurrences and is high-precision). A 1-edit-distance fuzzy hit with
    a single attestation in the corpus is the dominant OCR-noise class
    (saearp, drnim, etc.); requiring either ≥2 attestations or any
    high-confidence method removes them without throwing away legitimate
    archaic spellings.
    """
    placeholders = ",".join("?" * len(member_ids))
    member_variants: dict[int, list[tuple[str, int]]] = {}
    for row in db.conn.execute(
        f"SELECT etymon_id, matched_form, "
        f"  SUM(match_count) AS total_count, "
        f"  GROUP_CONCAT(DISTINCT method) AS methods "
        f"FROM etymon_text_match "
        f"WHERE etymon_id IN ({placeholders}) "
        f"GROUP BY etymon_id, LOWER(matched_form) "
        f"ORDER BY etymon_id, total_count DESC, matched_form",
        member_ids,
    ):
        if row["matched_form"].lower() in canonical_forms_lower:
            continue
        methods = set((row["methods"] or "").split(","))
        if row["total_count"] <= 1 and methods.issubset(_LOW_CONFIDENCE_METHODS):
            continue
        member_variants.setdefault(row["etymon_id"], []).append(
            (row["matched_form"], row["total_count"])
        )
    return member_variants


_NON_SCHOLAR_SOURCES = frozenset({"rando-port"})


def _fetch_member_citations(db: LexiconDB, member_ids: list[int]) -> dict[int, list[str]]:
    """Distinct sorted scholarly source_ids per member_id from
    etymon_citation. The runtime explainer surfaces this so a GM holding
    a generated name can see which scholars attest each morpheme.
    Sorted alphabetically for deterministic bundle output.

    Filters out non-scholarly seeds (rando-port — the Wikipedia-derived
    legacy bootstrap that ships every legacy etymon but isn't a real
    citation a GM would recognize). Members with no scholarly citations
    drop out of the result entirely so the emitter omits the
    `<lang>_citations` field on rando-only words.
    """
    placeholders = ",".join("?" * len(member_ids))
    member_citations: dict[int, list[str]] = {}
    for row in db.conn.execute(
        f"SELECT etymon_id, source_id "
        f"FROM etymon_citation "
        f"WHERE etymon_id IN ({placeholders}) "
        f"GROUP BY etymon_id, source_id "
        f"ORDER BY etymon_id, source_id",
        member_ids,
    ):
        if row["source_id"] in _NON_SCHOLAR_SOURCES:
            continue
        member_citations.setdefault(row["etymon_id"], []).append(row["source_id"])
    return member_citations


def _compute_member_descendants(member_rows: list[Any]) -> dict[int, list[int]]:
    """Transitive descendants per member_id (DB-free DFS).

    Lets a reflex linked to a lemma pick up the lemma's inflected children
    + OCR-cluster losers rather than just the directly-linked etymon. The
    graph is shallow (D22 flatten-at-merge-time keeps chains at depth ≤
    2), so a single inversion pass suffices.
    """
    children: dict[int, list[int]] = {}
    for r in member_rows:
        parent_id = r["lemma_id"] or r["merged_into_id"]
        if parent_id is not None:
            children.setdefault(parent_id, []).append(r["id"])
    member_descendants: dict[int, list[int]] = {}
    for r in member_rows:
        member_id = r["id"]
        descendants = [member_id]
        stack = list(children.get(member_id, []))
        while stack:
            d = stack.pop()
            if d in descendants:
                continue
            descendants.append(d)
            stack.extend(children.get(d, []))
        member_descendants[member_id] = descendants
    return member_descendants


def _group_families_into_subjects(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group families by (modifier_type, glosses, tags) into meanings.json subjects.

    Each group becomes one subject. Reflexes linked to any family in the
    group become "words"; families without any reflex contribute a synthesized
    word using the family's canonical_form (for newly-mined etymons that
    haven't been wired into a modern surface form yet).
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for fam in families:
        key = (
            fam["modifier_type"] or "",
            tuple(sorted(fam["glosses"])),
            tuple(sorted(fam["tags"])),
        )
        groups.setdefault(key, []).append(fam)

    subjects: list[dict[str, Any]] = []
    for (mod_type, glosses_tuple, tags_tuple), fams in groups.items():
        words = _build_words_for_group(fams)
        if not words:
            continue
        subject: dict[str, Any] = {
            "meaning": list(glosses_tuple),
            "modifier_tags": list(tags_tuple),
            "modifier_type": mod_type or None,
            "words": words,
        }
        subjects.append(subject)

    # Fully-discriminating sort key: every field that varies across subjects
    # must be in the tuple. Otherwise ties fall back to dict insertion order,
    # which traces back to AUTOINCREMENT root_ids and re-shuffles on rebuild.
    subjects.sort(
        key=lambda s: (
            s.get("modifier_type") or "",
            tuple(s["meaning"]),
            tuple(s["modifier_tags"]),
            tuple(w["modern_usage"] for w in s["words"]),
        )
    )
    return subjects


def _build_words_for_group(fams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble the `words` list for a subject grouping multiple families.

    Sort families deterministically, partition into reflex-linked vs.
    reflex-less, then dispatch each group to a focused word-builder.
    """
    # Sort families by canonical_form/language so the export output is
    # deterministic — root_ids are AUTOINCREMENT and therefore unstable
    # across DB rebuilds, which would otherwise churn the diff in version
    # control even when no semantic content changed.
    fams = sorted(fams, key=lambda f: (f["root_canonical_form"], f["root_language"]))

    reflex_to_links, reflex_meta, families_without_reflex = _partition_families_by_reflex(fams)

    words: list[dict[str, Any]] = []
    for reflex_id in sorted(
        reflex_to_links,
        key=lambda rid: (reflex_meta[rid]["position"], reflex_meta[rid]["surface_form"]),
    ):
        words.append(_word_for_reflex(reflex_meta[reflex_id], reflex_to_links[reflex_id]))

    for fam in families_without_reflex:
        words.append(_synthesize_word_for_family(fam))

    return words


def _partition_families_by_reflex(
    fams: list[dict[str, Any]],
) -> tuple[
    dict[int, list[tuple[dict[str, Any], list[int]]]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Split families into reflex-linked and reflex-less groups.

    Returns ``(reflex_to_links, reflex_meta, families_without_reflex)``.
    ``reflex_to_links[reflex_id]`` is a list of ``(family, linked_member_ids)``
    tuples so each reflex's word entry only includes language forms from
    the etymons it's actually linked to, not from every family in the
    subject. Per the original meanings.json shape (e.g. "Alder tree"),
    reflexes are language-specific: '-farne' carries celtic_mix only,
    'Alder-' carries old_english only. seed_from_meanings preserves this
    in reflex_etymon, and the export must too.
    """
    reflex_to_links: dict[int, list[tuple[dict[str, Any], list[int]]]] = {}
    reflex_meta: dict[int, dict[str, Any]] = {}
    families_without_reflex: list[dict[str, Any]] = []
    for fam in fams:
        if not fam["reflexes"]:
            families_without_reflex.append(fam)
            continue
        for r in fam["reflexes"]:
            reflex_meta[r["id"]] = r
            reflex_to_links.setdefault(r["id"], []).append((fam, r["linked_member_ids"]))
    return reflex_to_links, reflex_meta, families_without_reflex


def _word_for_reflex(
    meta: dict[str, Any], link_pairs: list[tuple[dict[str, Any], list[int]]]
) -> dict[str, Any]:
    """Assemble one word entry for a reflex linked to one or more families.

    Walks each linked etymon's descendants so the reflex picks up the
    lemma's inflected children (D8) and OCR-cluster losers (D22) —
    not just the seeded etymon itself.
    """
    per_lang: dict[str, list[str]] = {}
    per_lang_variants: dict[str, dict[str, int]] = {}
    per_lang_inflections: dict[str, dict[str, str]] = {}
    per_lang_citations: dict[str, set[str]] = {}
    for fam, linked_ids in link_pairs:
        for member_id in linked_ids:
            for descendant_id in fam["member_descendants"][member_id]:
                lang, form = fam["member_form_by_id"][descendant_id]
                bucket = per_lang.setdefault(lang, [])
                if form not in bucket:
                    bucket.append(form)
                _absorb_member_variants(per_lang_variants, fam, descendant_id, lang)
                _absorb_member_inflection(per_lang_inflections, fam, descendant_id, lang, form)
                _absorb_member_citations(per_lang_citations, fam, descendant_id, lang)
    word: dict[str, Any] = {"modern_usage": meta["surface_form"]}
    _emit_word_languages(
        word, per_lang, per_lang_variants, per_lang_inflections, per_lang_citations
    )
    return word


def _synthesize_word_for_family(fam: dict[str, Any]) -> dict[str, Any]:
    """Assemble a synthesized word for a family that has no linked reflex.

    Uses the family's canonical_forms en bloc (no per-reflex narrowing
    applies). The whole family's variants and inflections fold in by
    matching language.
    """
    word: dict[str, Any] = {"modern_usage": _synthesize_modern_usage(fam)}
    per_lang: dict[str, list[str]] = {
        lang: list(fam["forms_by_lang"][lang]) for lang in fam["forms_by_lang"]
    }
    per_lang_variants: dict[str, dict[str, int]] = {}
    per_lang_inflections: dict[str, dict[str, str]] = {}
    per_lang_citations: dict[str, set[str]] = {}
    for member_id, (member_lang, member_form) in fam["member_form_by_id"].items():
        _absorb_member_variants(per_lang_variants, fam, member_id, member_lang)
        _absorb_member_inflection(per_lang_inflections, fam, member_id, member_lang, member_form)
        _absorb_member_citations(per_lang_citations, fam, member_id, member_lang)
    _emit_word_languages(
        word, per_lang, per_lang_variants, per_lang_inflections, per_lang_citations
    )
    return word


def _absorb_member_variants(
    per_lang_variants: dict[str, dict[str, int]],
    fam: dict[str, Any],
    member_id: int,
    lang: str,
) -> None:
    """Aggregate D18 spelling variants for one (member, language) into the
    per-language pool, summing weights on collision. Caller guarantees
    `lang` matches the member's language so callers don't accidentally
    cross-pollinate across languages."""
    for variant_form, weight in fam.get("member_variants", {}).get(member_id, []):
        lang_variants = per_lang_variants.setdefault(lang, {})
        lang_variants[variant_form] = lang_variants.get(variant_form, 0) + weight


def _absorb_member_inflection(
    per_lang_inflections: dict[str, dict[str, str]],
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """Record a member's D8 inflection label (if any) keyed by its surface
    form. Lemmas have inflection=None and are skipped — only inflected
    children carry a grammatical-case label worth surfacing."""
    inflection = fam.get("member_inflection_by_id", {}).get(member_id)
    if inflection:
        per_lang_inflections.setdefault(lang, {})[form] = inflection


def _absorb_member_citations(
    per_lang_citations: dict[str, set[str]],
    fam: dict[str, Any],
    member_id: int,
    lang: str,
) -> None:
    """Aggregate scholarly source_ids for one member into the per-language
    citation set (wyrd-9kh.1). Set semantics dedupe across the descendant
    walk; emit-time sorts deterministically. Caller guarantees `lang`
    matches the member's language."""
    citations = fam.get("member_citations", {}).get(member_id, [])
    if citations:
        per_lang_citations.setdefault(lang, set()).update(citations)


def _emit_word_languages(
    word: dict[str, Any],
    per_lang: dict[str, list[str]],
    per_lang_variants: dict[str, dict[str, int]],
    per_lang_inflections: dict[str, dict[str, str]],
    per_lang_citations: dict[str, set[str]],
) -> None:
    """Stamp per-language form arrays + sibling _variants / _inflections /
    _citations metadata onto the word dict. Per D26, the metadata fields
    are sibling keys (`<lang>_variants`, `<lang>_inflections`,
    `<lang>_citations`) so legacy loaders that ignore unknown fields keep
    working."""
    for lang in sorted(per_lang):
        json_field = _LANG_CODE_TO_JSON_FIELD.get(lang)
        if not json_field:
            continue
        word[json_field] = per_lang[lang]
        if lang in per_lang_variants:
            word[f"{json_field}_variants"] = _emit_variant_list(per_lang_variants[lang])
        if lang in per_lang_inflections:
            word[f"{json_field}_inflections"] = _emit_inflection_list(per_lang_inflections[lang])
        if lang in per_lang_citations:
            word[f"{json_field}_citations"] = sorted(per_lang_citations[lang])


def _emit_variant_list(variants: dict[str, int]) -> list[dict[str, Any]]:
    """Serialize {form: weight} into the meanings.json variant entry shape.
    Output is sorted by descending weight (most-attested first), with the
    form string as a stable secondary key — matters because two variants
    can share a weight after aggregation across the family."""
    return [
        {"form": form, "weight": weight}
        for form, weight in sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _emit_inflection_list(inflections: dict[str, str]) -> list[dict[str, Any]]:
    """Serialize {form: inflection_label} into the meanings.json inflection
    entry shape. Sorted by form so the output is deterministic regardless of
    aggregation order."""
    return [{"form": form, "inflection": label} for form, label in sorted(inflections.items())]


def _synthesize_modern_usage(family: dict[str, Any]) -> str:
    """Derive a modern_usage for a family that has no linked reflex.

    Uses ``position_pref`` to choose pre/post/inner dash markers; defaults
    to no-dash (the runtime treats undecorated usage as a post-suffix
    via Meaning._set_location).
    """
    form = family["root_canonical_form"]
    position = family.get("position_pref")
    if position == "pre":
        return f"{form}-"
    if position == "inner":
        return f"-{form}-"
    if position == "post":
        return f"-{form}"
    return form


def _orphan_reflex_subjects(db: LexiconDB) -> list[dict[str, Any]]:
    """Emit one subject per reflex that has no linked etymon.

    These come from the rando-port seed where a `word` entry had only
    `modern_usage` (e.g., 'Adam-' as a saint's name) or had a language slot
    with an empty form list (e.g., {'celtic_mix': [], 'modern_usage': 'Bre-'}).
    The runtime needs them in `meaning_db` because per-culture proportions
    JSONs reference them by usage. Emit each as its own subject with empty
    glosses/tags so the runtime can register the usage; promotion to a richer
    subject can land later via the manual seed-source-aware revisit (D7).
    """
    rows = db.conn.execute(
        """
        SELECT r.surface_form
        FROM reflex r
        WHERE NOT EXISTS (
            SELECT 1 FROM reflex_etymon re WHERE re.reflex_id = r.id
        )
        ORDER BY r.position, r.surface_form
        """
    ).fetchall()
    return [
        {
            "meaning": [],
            "modifier_tags": [],
            "modifier_type": None,
            "words": [{"modern_usage": row["surface_form"]}],
        }
        for row in rows
    ]
