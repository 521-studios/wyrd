"""wyrd-h6x7: pytest-time package-boundary checks for kenning/.

Enforces the AGENT-REVIEWERS.md lexicon-package-structure rule
('no DB I/O outside lexicon/') at test-collection time, so the
boundary doesn't have to be re-litigated by humans + agents on every
PR. Runs AST-only — no import side effects.

Scope:
- ``test_no_new_top_level_sqlite_or_lexicondb_importers`` flags any
  top-level ``wyrd/generators/kenning/*.py`` file that imports
  ``sqlite3`` or ``LexiconDB``, beyond the documented skip-list of
  legitimate pre-existing violators awaiting subpackage extraction.
- ``test_cli_lexicon_files_do_not_define_db_helpers`` checks that
  ``wyrd/generators/kenning/cli/lexicon/*.py`` doesn't DEFINE
  lexicon-DB-touching helpers (they should import from ``lexicon/``).
- ``test_no_create_table_view_index_outside_lexicon`` flags string
  literals matching the DDL keywords outside the lexicon package.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Absolute root of the kenning package this test runs against.
_KENNING_ROOT = Path(__file__).parent.parent / "wyrd" / "generators" / "kenning"

# Pre-existing files at kenning/*.py that touch sqlite3 or LexiconDB
# and are awaiting extraction into the appropriate subpackage (ingester
# tools, mining orchestrators, audit utilities). Adding NEW entries here
# requires a tracking ticket — the goal is for this list to shrink
# monotonically over time. wyrd-h6x7.
_DB_IMPORT_SKIPLIST: frozenset[str] = frozenset(
    {
        "enrichment.py",
        "short_quote_refill.py",
        "toponym_candidate_review.py",
        "toponym_etymology_canonical.py",
        "toponym_mention_ingest.py",
        "toponym_reverse_search.py",
        "wiktextract_corpus_miner.py",
    }
)

# Same idea but for DDL string literals (CREATE TABLE / INDEX / VIEW).
# Smaller skip-list because the DB-import violators don't all also
# carry DDL strings.
_DDL_SKIPLIST: frozenset[str] = frozenset(set())


def _file_imports_sqlite3_or_lexicondb(path: Path) -> bool:
    """AST-walk ``path``; return True if it imports sqlite3 or
    LexiconDB (from anywhere). Catches both ``import sqlite3``,
    ``from sqlite3 import ...``, and ``from ... import LexiconDB``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sqlite3" or alias.name.startswith("sqlite3."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if node.module == "sqlite3" or node.module.startswith("sqlite3."):
                return True
            # from wyrd...lexicon... import LexiconDB
            for alias in node.names:
                if alias.name == "LexiconDB":
                    return True
    return False


def _file_contains_ddl_literal(path: Path) -> bool:
    """AST-walk ``path``; return True if any string-constant body
    starts with a DDL keyword (CREATE TABLE / CREATE INDEX /
    CREATE VIEW, case-insensitive). String-literal check (not text
    grep) so docstrings still apply but commented-out DDL doesn't."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError):
        return False
    ddl_prefixes = ("CREATE TABLE", "CREATE INDEX", "CREATE VIEW")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.lstrip().upper()
            if any(text.startswith(p) for p in ddl_prefixes):
                return True
    return False


def test_no_new_top_level_sqlite_or_lexicondb_importers() -> None:
    """Kenning's top-level ``*.py`` files must not import sqlite3 or
    LexiconDB — those calls belong inside ``lexicon/`` (or a sibling
    subpackage like ``jsonl/``, ``ingesters/``, ``bundle/``). Pre-
    existing violators are in ``_DB_IMPORT_SKIPLIST`` with a tracking
    note in the module docstring."""
    violators: list[str] = []
    for py in sorted(_KENNING_ROOT.glob("*.py")):
        if py.name in _DB_IMPORT_SKIPLIST:
            continue
        if _file_imports_sqlite3_or_lexicondb(py):
            violators.append(py.name)
    assert not violators, (
        "New top-level kenning/*.py files imported sqlite3 or LexiconDB:\n  "
        + "\n  ".join(violators)
        + "\nMove the DB-touching code into the appropriate subpackage "
        "(lexicon/ for authoring, jsonl/ for replay/build, ingesters/ "
        "for operator-CSV inputs), or — if it has to live at the top "
        "level for a documented reason — add it to "
        "_DB_IMPORT_SKIPLIST in this test file with a follow-up ticket."
    )


def test_skiplist_entries_actually_exist() -> None:
    """Every name in _DB_IMPORT_SKIPLIST must correspond to a file that
    currently exists. Otherwise the skip-list rots silently as files get
    moved / deleted, and the boundary check loses its teeth."""
    missing = [name for name in _DB_IMPORT_SKIPLIST if not (_KENNING_ROOT / name).exists()]
    assert not missing, (
        f"_DB_IMPORT_SKIPLIST references non-existent files: {missing}. "
        "Drop the stale entries so the skip-list reflects actual ground truth."
    )


def test_skiplist_entries_still_violate() -> None:
    """Every name in _DB_IMPORT_SKIPLIST must STILL import sqlite3 /
    LexiconDB. If a file got refactored to not violate anymore, drop
    it from the skip-list so the check tightens automatically."""
    no_longer_violating = [
        name
        for name in _DB_IMPORT_SKIPLIST
        if not _file_imports_sqlite3_or_lexicondb(_KENNING_ROOT / name)
    ]
    assert not no_longer_violating, (
        f"_DB_IMPORT_SKIPLIST entries no longer violate (refactor landed): "
        f"{no_longer_violating}. Remove from the skip-list so a "
        "regression in those files would trip the boundary check."
    )


def test_cli_lexicon_files_do_not_define_db_helpers() -> None:
    """``cli/lexicon/*.py`` files should be argument-parsing + IO glue
    that calls into the ``lexicon/`` package. They should NOT define
    new sqlite3-touching helpers locally (e.g. open a raw connection
    and write queries inline rather than calling a lexicon-package
    helper). Importing the canonical helpers — ``_readonly_lexicon``
    from ``cli.utils``, ``LexiconDB`` — is fine; defining new ones is
    not.

    Enforcement: a cli/lexicon/*.py file that imports sqlite3 directly
    is a smell. The skip-list below lists pre-existing exceptions
    awaiting migration to ``_readonly_lexicon`` (or, for writers, a
    proper LexiconDB context). wyrd-h0u1 already migrated report +
    language-report; the remaining 6 mostly do writes (rebuild,
    ingest, dump) and need slightly heavier refactoring to land on a
    canonical authoring helper. Tracked at wyrd-mewu."""
    cli_lex_dir = _KENNING_ROOT / "cli" / "lexicon"
    # wyrd-mewu COMPLETE: every cli/lexicon file is off direct sqlite3.
    # (review.py now catches lexicon.LexiconError instead of sqlite3.Error
    # and uses _readonly_lexicon for its read connections.)
    skip: frozenset[str] = frozenset()
    violators: list[str] = []
    for py in sorted(cli_lex_dir.glob("*.py")):
        if py.name in skip:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except (UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "sqlite3":
                        violators.append(f"{py.name}: import sqlite3")
                        break
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module == "sqlite3"
            ):
                violators.append(f"{py.name}: from sqlite3 import ...")
    assert not violators, (
        "cli/lexicon/*.py files imported sqlite3 directly (use "
        "_readonly_lexicon from cli.utils or call into the lexicon/ "
        "package instead):\n  " + "\n  ".join(violators)
    )


def test_no_create_table_view_index_outside_lexicon() -> None:
    """CREATE TABLE / CREATE INDEX / CREATE VIEW string literals must
    live inside ``lexicon/`` (the authoring side). A DDL literal at
    the top level or in a sibling subpackage means schema state is
    being mutated outside the canonical seam."""
    violators: list[str] = []
    for py in sorted(_KENNING_ROOT.glob("*.py")):
        if py.name in _DDL_SKIPLIST:
            continue
        if _file_contains_ddl_literal(py):
            violators.append(py.name)
    assert not violators, (
        "Top-level kenning/*.py contained DDL string literal(s):\n  "
        + "\n  ".join(violators)
        + "\nMove schema mutation into lexicon/ (alembic migration + "
        "tables.py) so there's a single seam for schema state."
    )
