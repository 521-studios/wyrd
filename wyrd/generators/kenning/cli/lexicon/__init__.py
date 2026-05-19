"""The ``wyrd kenning lexicon`` click sub-group.

This package holds the ``lexicon``-scoped CLI subcommands as part of
the wyrd-g143 split. Slice 2 added the read-only navigation /
reporting family. Slice 3 added the bulk-sources storage trio, the
JSONL dump/rebuild/diff machinery (the L2 → L3 pipeline backbone),
the operator-CSV ingest family, the export-meanings bundle exporter,
and compact-jsonl. Subsequent slices (4-5) move the remaining
``@lexicon.command`` bodies out of the cli back-compat shim into
per-subcommand modules under this subpackage.

Each per-command module exposes an ``add_to(parent: click.Group)``
hook. This package's ``__init__`` calls each module's ``add_to`` to
register its subcommand on the shared ``lexicon`` click group.

Note: this package's name collides at the directory level with the
sibling ``wyrd.generators.kenning.lexicon`` DB-layer package — they
are distinct absolute import paths (``wyrd.generators.kenning.cli.lexicon``
vs ``wyrd.generators.kenning.lexicon``) and Python's absolute-import
resolution keeps them apart. This package holds CLI subcommands; the
DB-layer package holds the SQLite + enrichment surface.
"""

from __future__ import annotations

import click

from wyrd.generators.kenning.cli.lexicon import (
    audit_etymology_alignment as _audit_etymology_alignment_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    audit_short_quotes as _audit_short_quotes_module,
)
from wyrd.generators.kenning.cli.lexicon import browse as _browse_module
from wyrd.generators.kenning.cli.lexicon import compact_jsonl as _compact_jsonl_module
from wyrd.generators.kenning.cli.lexicon import diff_bundle as _diff_bundle_module
from wyrd.generators.kenning.cli.lexicon import diff_rebuild as _diff_rebuild_module
from wyrd.generators.kenning.cli.lexicon import dump_jsonl as _dump_jsonl_module
from wyrd.generators.kenning.cli.lexicon import enrich as _enrich_module
from wyrd.generators.kenning.cli.lexicon import enrichment_status as _enrichment_status_module
from wyrd.generators.kenning.cli.lexicon import era_cell as _era_cell_module
from wyrd.generators.kenning.cli.lexicon import era_coverage as _era_coverage_module
from wyrd.generators.kenning.cli.lexicon import era_reflex as _era_reflex_module
from wyrd.generators.kenning.cli.lexicon import era_timeline as _era_timeline_module
from wyrd.generators.kenning.cli.lexicon import export_meanings as _export_meanings_module
from wyrd.generators.kenning.cli.lexicon import fetch_bulk_sources as _fetch_bulk_sources_module
from wyrd.generators.kenning.cli.lexicon import ingest_hearth_tax as _ingest_hearth_tax_module
from wyrd.generators.kenning.cli.lexicon import (
    ingest_hundred_rolls as _ingest_hundred_rolls_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    ingest_os_open_names as _ingest_os_open_names_module,
)
from wyrd.generators.kenning.cli.lexicon import ingest_speed_1611 as _ingest_speed_1611_module
from wyrd.generators.kenning.cli.lexicon import language_report as _language_report_module
from wyrd.generators.kenning.cli.lexicon import migrate as _migrate_module
from wyrd.generators.kenning.cli.lexicon import path as _path_module
from wyrd.generators.kenning.cli.lexicon import push_bulk_sources as _push_bulk_sources_module
from wyrd.generators.kenning.cli.lexicon import rando_port_readiness as _rando_port_readiness_module
from wyrd.generators.kenning.cli.lexicon import (
    rebuild_from_jsonl as _rebuild_from_jsonl_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    report as _report_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    report_wikipedia_backfill as _report_wikipedia_backfill_module,
)
from wyrd.generators.kenning.cli.lexicon import stats as _stats_module
from wyrd.generators.kenning.cli.lexicon import verify_bulk_sources as _verify_bulk_sources_module


@click.group("lexicon")
def lexicon() -> None:
    """Manage the authoring lexicon DB (etymology data store)."""


# Register each extracted subcommand module on the @lexicon group.
# Adding a new lexicon subcommand means creating cli/lexicon/<name>.py
# (exposing add_to(parent)) and adding a one-line module import + a
# `<name>_module.add_to(lexicon)` call below. The browse sub-group
# follows the same pattern at the package level (cli/lexicon/browse/).
_audit_etymology_alignment_module.add_to(lexicon)
_audit_short_quotes_module.add_to(lexicon)
_browse_module.add_to(lexicon)
_compact_jsonl_module.add_to(lexicon)
_diff_bundle_module.add_to(lexicon)
_diff_rebuild_module.add_to(lexicon)
_dump_jsonl_module.add_to(lexicon)
_enrich_module.add_to(lexicon)
_enrichment_status_module.add_to(lexicon)
_era_cell_module.add_to(lexicon)
_era_coverage_module.add_to(lexicon)
_era_reflex_module.add_to(lexicon)
_era_timeline_module.add_to(lexicon)
_export_meanings_module.add_to(lexicon)
_fetch_bulk_sources_module.add_to(lexicon)
_ingest_hearth_tax_module.add_to(lexicon)
_ingest_hundred_rolls_module.add_to(lexicon)
_ingest_os_open_names_module.add_to(lexicon)
_ingest_speed_1611_module.add_to(lexicon)
_language_report_module.add_to(lexicon)
_migrate_module.add_to(lexicon)
_path_module.add_to(lexicon)
_push_bulk_sources_module.add_to(lexicon)
_rando_port_readiness_module.add_to(lexicon)
_rebuild_from_jsonl_module.add_to(lexicon)
_report_module.add_to(lexicon)
_report_wikipedia_backfill_module.add_to(lexicon)
_stats_module.add_to(lexicon)
_verify_bulk_sources_module.add_to(lexicon)


def add_to(parent: click.Group) -> None:
    """Register the ``lexicon`` sub-group on a parent click group.

    Called once by ``wyrd.generators.kenning.cli.__init__`` while the
    top-level ``@cli`` group is being constructed.
    """
    parent.add_command(lexicon)
