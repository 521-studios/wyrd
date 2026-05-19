"""The kenning ``jsonl/`` subpackage — L2 / L3 round-trip infrastructure.

The wyrd-f295 boundary (see ``L2_L3_BOUNDARY.md``) treats per-source
JSONL files at ``data/mining/<source_id>.jsonl`` as the source of
truth and the SQLite lexicon DB at ``~/.wyrd/lexicon.db`` as a
rebuildable build artifact. This subpackage owns the three pieces
that make that boundary work:

* ``log.py`` — event-log primitives (``LogEvent``, ``replay_file``,
  ``append_event``, etc.). The kernel of the L2 round-trip contract.
* ``dump.py`` — DB → canonical-state JSONL emitter. Called by
  ``lexicon dump-jsonl``.
* ``build.py`` — JSONL → DB replay. Called by
  ``lexicon rebuild-from-jsonl``.

CLI consumers live at ``wyrd/generators/kenning/cli/lexicon/`` —
``dump_jsonl.py``, ``rebuild_from_jsonl.py``, ``compact_jsonl.py``,
``diff_rebuild.py``, ``diff_bundle.py``, ``enrich.py`` all import
from this subpackage.
"""
