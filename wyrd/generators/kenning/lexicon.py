"""Lexicon data store: SQLite-backed authoring DB for place-name etymology.

Authoritative for authoring (mining sources, tracking citations, recording
disagreement). The runtime keeps reading meanings.json, which is exported from
this DB by `wyrd kenning lexicon export-meanings`.
"""

from __future__ import annotations

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
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO etymon_citation (etymon_id, source_id, page, short_quote)
            VALUES (?, ?, ?, ?)
            """,
            (etymon_id, source_id, page, short_quote),
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


def migrate_schema(db: LexiconDB) -> dict[str, bool]:
    """Bring an existing lexicon DB up to the current schema.

    Idempotent — running on an already-migrated DB is a no-op. Currently
    handles only the lemma_id / inflection additions to `etymon`. Returns a
    dict of {migration_name: applied?} so callers can report.
    """
    applied = {
        "etymon.lemma_id": False,
        "etymon.inflection": False,
        "idx_etymon_lemma": False,
        "etymon_consensus_view": False,
        "etymon_text_match_table": False,
    }
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
    if "lemma_id" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN lemma_id INTEGER REFERENCES etymon(id)")
        applied["etymon.lemma_id"] = True
    if "inflection" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN inflection TEXT")
        applied["etymon.inflection"] = True
    # Idempotent index creation.
    existing_indexes = {
        row["name"] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    if "idx_etymon_lemma" not in existing_indexes:
        db.conn.execute("CREATE INDEX idx_etymon_lemma ON etymon(lemma_id)")
        applied["idx_etymon_lemma"] = True
    # Always rebuild the consensus view since its definition changes when
    # lemma_id is introduced. SQLite doesn't have CREATE OR REPLACE VIEW;
    # we DROP IF EXISTS and recreate. Cheap.
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
              COALESCE(e.lemma_id, e.id)              AS lemma_id,
              COALESCE(le.canonical_form, e.canonical_form) AS canonical_form,
              e.language,
              c.source_id
            FROM etymon e
            LEFT JOIN etymon le ON le.id = e.lemma_id
            LEFT JOIN etymon_citation c ON c.etymon_id = e.id
          )
          GROUP BY lemma_id
        """
    )
    applied["etymon_consensus_view"] = True

    # Add the etymon_text_match table for parallel search-evidence storage.
    # See lexicon.sql for the design rationale (extraction vs. search
    # evidence kept separate so consensus counts stay pure).
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
              UNIQUE (etymon_id, source_id, matched_form)
            );
            CREATE INDEX idx_etymon_text_match_etymon ON etymon_text_match(etymon_id);
            CREATE INDEX idx_etymon_text_match_source ON etymon_text_match(source_id);
            """
        )
        applied["etymon_text_match_table"] = True

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
    # Index existing etymons by (form, language) → id, but EXCLUDE etymons
    # that are themselves inflected — a candidate lemma must be unlinked
    # OR a true root form. For simplicity here we say: any etymon with
    # lemma_id IS NULL is a candidate lemma.
    lemmas_by_key: dict[tuple[str, str], int] = {}
    for row in db.conn.execute(
        "SELECT id, canonical_form, language FROM etymon WHERE lemma_id IS NULL"
    ):
        lemmas_by_key[(row["canonical_form"].lower(), row["language"])] = row["id"]

    proposed: list[dict] = []
    inflected_rows = list(
        db.conn.execute("SELECT id, canonical_form, language FROM etymon WHERE lemma_id IS NULL")
    )
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
                "UPDATE etymon SET lemma_id = ?, inflection = ? WHERE id = ?",
                (p["lemma_id"], p["inflection"], p["inflected_id"]),
            )
        db.commit()

    return {
        "candidates": len(proposed),
        "sample": proposed[:25],
        "applied": apply,
    }


# --- OCR variant clustering ------------------------------------------------


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
    # Snippet is ±60 chars around the FIRST match in that source — gives
    # the app something to show without storing every occurrence.
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
            start = max(0, first.start() - 60)
            end = min(len(text), first.end() + 60)
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


def _levenshtein(a: str, b: str, *, max_distance: int | None = None) -> int:
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
                d = _levenshtein(norm_form, tok, max_distance=max_distance)
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
                snip_start = max(0, m.start() - 60)
                snip_end = min(len(text), m.end() + 60)
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
                         edit_distance, snippet)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(etymon_id, source_id, matched_form)
                    DO UPDATE SET
                        match_count = excluded.match_count,
                        edit_distance = excluded.edit_distance,
                        snippet = excluded.snippet
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
    """Find etymons that differ only in OCR ligature variation, merge them.

    Groups every etymon by (language, normalize_ocr_form(canonical_form)).
    Within each group of >1 row, the row with the LOWEST id is taken as
    canonical, and the others are merged in:
      - etymon_citation, etymon_gloss, etymon_tag rows are repointed (or
        deleted if they would duplicate)
      - reflex_etymon, toponym_etymology_element rows are repointed
      - the redundant etymon rows are deleted

    With apply=False (default) we only report what would happen — no
    writes. Pass apply=True to actually perform the merges.

    Returns a dict with keys:
      groups          - count of (lang, normalized) groups with >1 etymon
      etymons_merged  - count of redundant etymons that would be removed
      sample_groups   - first 20 example groupings, for human review
    """
    cur = db.conn.execute("SELECT id, canonical_form, language FROM etymon ORDER BY id")
    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for row in cur:
        key = (row["language"], normalize_ocr_form(row["canonical_form"]))
        groups.setdefault(key, []).append((row["id"], row["canonical_form"]))

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

    # Apply the merges. For each group, the lowest id wins; others get
    # repointed and deleted.
    for members in duplicate_groups.values():
        winner_id = members[0][0]
        loser_ids = [m[0] for m in members[1:]]
        for loser_id in loser_ids:
            # Repoint child rows that have unique constraints first by
            # using INSERT OR IGNORE then DELETE on the loser.
            db.conn.execute(
                "INSERT OR IGNORE INTO etymon_citation "
                "  (etymon_id, source_id, page, short_quote) "
                "SELECT ?, source_id, page, short_quote "
                "FROM etymon_citation WHERE etymon_id = ?",
                (winner_id, loser_id),
            )
            db.conn.execute(
                "DELETE FROM etymon_citation WHERE etymon_id = ?",
                (loser_id,),
            )
            db.conn.execute(
                "INSERT OR IGNORE INTO etymon_gloss (etymon_id, gloss) "
                "SELECT ?, gloss FROM etymon_gloss WHERE etymon_id = ?",
                (winner_id, loser_id),
            )
            db.conn.execute(
                "DELETE FROM etymon_gloss WHERE etymon_id = ?",
                (loser_id,),
            )
            db.conn.execute(
                "INSERT OR IGNORE INTO etymon_tag (etymon_id, tag) "
                "SELECT ?, tag FROM etymon_tag WHERE etymon_id = ?",
                (winner_id, loser_id),
            )
            db.conn.execute(
                "DELETE FROM etymon_tag WHERE etymon_id = ?",
                (loser_id,),
            )
            db.conn.execute(
                "INSERT OR IGNORE INTO reflex_etymon (reflex_id, etymon_id) "
                "SELECT reflex_id, ? FROM reflex_etymon WHERE etymon_id = ?",
                (winner_id, loser_id),
            )
            db.conn.execute(
                "DELETE FROM reflex_etymon WHERE etymon_id = ?",
                (loser_id,),
            )
            # toponym_etymology_element doesn't have a unique constraint on
            # etymon_id so we can just UPDATE.
            db.conn.execute(
                "UPDATE toponym_etymology_element SET etymon_id = ? WHERE etymon_id = ?",
                (winner_id, loser_id),
            )
            db.conn.execute("DELETE FROM etymon WHERE id = ?", (loser_id,))

    db.commit()
    return counts


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
