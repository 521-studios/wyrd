"""Guards that the from-scratch rebuild runbook stays complete.

A full wipe (``lexicon rebuild-from-jsonl``) restores only what L2 JSONL
carries plus the enrichment-derived columns. Several data layers are
L3-only and must be restored by a separate free-mining step after the
wipe. Historically those steps were discovered one failing test at a
time (see ``wyrd/generators/kenning/REBUILD.md`` and the May-2026
rebuild). This test makes the runbook self-enforcing:

* **Auto-discovery** — every data-population CLI (``mine-*`` / ``ingest-*``
  / ``backfill-*`` / ``cleanup-*``) must be categorized in
  ``data/mining/_rebuild_layers.json`` as either a rebuild step or an
  explicitly-reasoned non-rebuild command. A newly-added command fails
  this test until someone decides which it is.
* **Manifest currency** — the categorization can't reference commands
  that no longer exist, and the two buckets can't overlap.
* **Runbook currency** — every command the manifest calls a rebuild step,
  and every layer's ``runbook_token``, must literally appear in
  REBUILD.md. So the manifest and the prose can't silently drift apart.

The whole thing reads files + walks the Click group; it never builds or
touches a DB.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from wyrd.generators.kenning.cli.lexicon import lexicon

# tests/ sits directly under the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _REPO_ROOT / "data" / "mining" / "_rebuild_layers.json"
_RUNBOOK_PATH = _REPO_ROOT / "wyrd" / "generators" / "kenning" / "REBUILD.md"

# Leaf-name prefixes that POPULATE data (the silent-drop risk class). Read-only
# / structural commands (report, export-*, enrich, decompose, ...) are not in
# scope here -- the manifest's `layers` + REBUILD.md prose cover those.
_DATA_POPULATION_PREFIXES = ("mine-", "ingest-", "backfill-", "cleanup-")

# Data-population commands whose names don't match the prefixes above and so
# must be enumerated by hand (whole-word or subgroup commands). Name-based
# discovery can't infer these; a genuinely-novel non-prefixed data command
# still needs the reviewer to spot it and add it here. Full space-joined paths.
_EXTRA_DATA_POPULATION_COMMANDS = frozenset(
    {
        "build",  # legacy seed-from-meanings.json path
        "synsets seed",  # seeds meaning_synset from committed JSON
        "synsets assign",  # writes etymon_meaning_synset (D28 Phase 2)
        "detect-collapses",  # generates the _collapses.jsonl ledger (wyrd-y651)
        "merge-collapses",  # non-deterministic LLM (Ollama) merge judgment → _collapses.jsonl (wyrd-6xzs)
    }
)


def _load_manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text())


def _runbook_text() -> str:
    return _RUNBOOK_PATH.read_text()


def _walk_commands(group: click.Group, prefix: str = "") -> list[str]:
    """Yield full space-joined paths of every *leaf* command, recursing into
    subgroups (so ``synsets seed`` / ``browse etymon`` are reached)."""
    paths: list[str] = []
    for name, cmd in group.commands.items():
        path = f"{prefix}{name}"
        if isinstance(cmd, click.Group):
            paths.extend(_walk_commands(cmd, path + " "))
        else:
            paths.append(path)
    return paths


def _all_lexicon_commands() -> set[str]:
    return set(_walk_commands(lexicon))


def _leaf(path: str) -> str:
    return path.rsplit(" ", 1)[-1]


def _data_population_commands() -> set[str]:
    return {
        path
        for path in _all_lexicon_commands()
        if _leaf(path).startswith(_DATA_POPULATION_PREFIXES)
        or path in _EXTRA_DATA_POPULATION_COMMANDS
    }


def test_manifest_and_runbook_exist() -> None:
    assert _MANIFEST_PATH.is_file(), f"missing rebuild manifest: {_MANIFEST_PATH}"
    assert _RUNBOOK_PATH.is_file(), f"missing rebuild runbook: {_RUNBOOK_PATH}"


def test_manifest_shape_is_valid() -> None:
    m = _load_manifest()
    assert isinstance(m.get("rebuild_step_commands"), list)
    assert isinstance(m.get("non_rebuild_commands"), dict)
    assert isinstance(m.get("layers"), list) and m["layers"]
    # Every non_rebuild reason must be a non-empty string (forces a real
    # justification, not a bare "" placeholder).
    for cmd, reason in m["non_rebuild_commands"].items():
        assert isinstance(reason, str) and reason.strip(), (
            f"non_rebuild_commands[{cmd!r}] needs a one-line reason"
        )


def test_every_data_population_command_is_categorized() -> None:
    """The core auto-discovery gate: a new mine-/ingest-/backfill-/cleanup-
    command must be filed in the manifest (as a rebuild step or an
    explicitly-reasoned exclusion) before it can merge."""
    m = _load_manifest()
    categorized = set(m["rebuild_step_commands"]) | set(m["non_rebuild_commands"])
    uncategorized = sorted(_data_population_commands() - categorized)
    assert not uncategorized, (
        "These data-population CLI commands are not categorized in "
        f"data/mining/_rebuild_layers.json: {uncategorized}. "
        "Add each to `rebuild_step_commands` (and document it in REBUILD.md) "
        "if a from-scratch rebuild must run it, or to `non_rebuild_commands` "
        "with a one-line reason if it is not a wipe-restore step."
    )


def test_manifest_has_no_stale_command_references() -> None:
    """The manifest can't name commands that no longer exist (catches
    renames / removals)."""
    m = _load_manifest()
    all_cmds = _all_lexicon_commands()
    listed = set(m["rebuild_step_commands"]) | set(m["non_rebuild_commands"])
    stale = sorted(listed - all_cmds)
    assert not stale, (
        f"data/mining/_rebuild_layers.json references commands that no longer "
        f"exist on `wyrd kenning lexicon`: {stale}"
    )


def test_command_buckets_are_disjoint() -> None:
    m = _load_manifest()
    both = sorted(set(m["rebuild_step_commands"]) & set(m["non_rebuild_commands"]))
    assert not both, (
        f"commands appear in BOTH rebuild_step_commands and non_rebuild_commands: {both}"
    )


def test_rebuild_step_commands_appear_in_runbook() -> None:
    """Every command the manifest calls a rebuild step must be written into
    REBUILD.md -- this is what keeps the instructions complete."""
    m = _load_manifest()
    text = _runbook_text()
    missing = sorted(c for c in m["rebuild_step_commands"] if c not in text)
    assert not missing, (
        "These rebuild-step commands are not documented in "
        f"wyrd/generators/kenning/REBUILD.md: {missing}. "
        "Every free-mining rebuild step must appear in the runbook prose."
    )


def test_layer_runbook_tokens_appear_in_runbook() -> None:
    m = _load_manifest()
    text = _runbook_text()
    missing = sorted(
        layer["runbook_token"]
        for layer in m["layers"]
        if layer.get("runbook_token") and layer["runbook_token"] not in text
    )
    assert not missing, (
        "These layer runbook_tokens from the manifest are absent from "
        f"REBUILD.md: {missing}. Each layer a wipe affects must be traceable "
        "to a documented restore in the runbook."
    )


@pytest.mark.parametrize("key", ["restore_command", "cleanup_command"])
def test_layer_commands_are_real_rebuild_steps(key: str) -> None:
    """A free-mining layer's restore/cleanup command must be a real CLI
    command AND categorized as a rebuild step (so the doc-coverage gate
    above actually covers it)."""
    m = _load_manifest()
    all_cmds = _all_lexicon_commands()
    rebuild_steps = set(m["rebuild_step_commands"])
    for layer in m["layers"]:
        cmd = layer.get(key)
        if cmd is None:
            continue
        assert cmd in all_cmds, (
            f"layer {layer['layer']!r} {key}={cmd!r} is not a real `wyrd kenning lexicon` command"
        )
        assert cmd in rebuild_steps, (
            f"layer {layer['layer']!r} {key}={cmd!r} must be listed in rebuild_step_commands"
        )
