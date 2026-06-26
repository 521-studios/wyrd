"""Pack-bundle loader (wyrd-ecjp.10b Phase 7).

Each scenario pack lives in its own ``packs/<pack_name>/`` directory
alongside the main bundle. The directory contains:

* ``manifest.json`` — :class:`PackManifest` serialized.
* ``meanings.json`` — the pack's lemma set, in the same shape as the
  main bundle's meanings.json (list of subjects with words).

This module loads packs from a base directory + returns
``{pack_name: PackBundle}`` for the runtime dispatch layer to consume
when an operator passes ``--pack <name>`` (CLI flag deferred to
wyrd-ecjp.11; this module is the data-layer half).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from wyrd.generators.kenning.runtime.meaning import Meaning, load_meanings
from wyrd.generators.kenning.vectors.schemas import PackManifest


@dataclass(frozen=True)
class PackBundle:
    """Loaded pack: manifest + meaning_db ready to feed into
    :func:`select_via_vector_scoring(pack_meaning_dbs=...)`.

    The ``tag_db`` mirrors the main bundle's tag → usages mapping so
    the runtime can filter pack lemmas by tag the same way native
    lemmas are filtered.
    """

    manifest: PackManifest
    meaning_db: dict[str, list[Meaning]]
    tag_db: dict[str, list[str]]


def _parse_manifest(payload: dict[str, Any]) -> PackManifest:
    """Validate + construct a :class:`PackManifest` from the parsed
    manifest.json dict. Raises ValueError on missing required fields
    so the operator gets a pointed error rather than a Meaning-load
    crash downstream."""
    required = ("pack_name", "template_donor", "template_recipient")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"pack manifest missing required field(s): {', '.join(missing)}")
    # Explicit-null handling: payload.get("default_weight", 1.0) returns
    # None if the key is present-but-null (a hand-edit gone wrong or a
    # default cleared by tooling). Passing None to float() raises
    # TypeError; the explicit `is not None` guard treats null as
    # "use the default" rather than crashing the bundle load.
    raw_weight = payload.get("default_weight")
    raw_version = payload.get("version")
    return PackManifest(
        pack_name=payload["pack_name"],
        template_donor=payload["template_donor"],
        template_recipient=payload["template_recipient"],
        default_weight=float(raw_weight) if raw_weight is not None else 1.0,
        version=str(raw_version) if raw_version is not None else "unversioned",
    )


def load_packs_from_traversable(packs_root: Any) -> dict[str, PackBundle]:
    """Scan ``packs_root`` for per-pack subdirectories. Each subdir
    that contains both manifest.json and meanings.json becomes a
    :class:`PackBundle` in the returned dict, keyed by the
    manifest's ``pack_name`` (NOT the directory name — operator can
    rename a directory without breaking the registry).

    Tolerant of:
    * Missing packs_root: returns empty dict (legacy bundles have no
      packs/ dir).
    * Subdirs without manifest.json or meanings.json: silently
      skipped (work-in-progress packs, README-only dirs).
    * Subdirs with malformed manifest: raises ValueError so the
      operator notices rather than silently shipping a broken pack.
    * Two subdirs declaring the same manifest ``pack_name``: raises
      ValueError for the same reason — last-dir-wins would silently
      drop one pack from the registry.

    The ``packs_root`` argument is intentionally typed as Any so the
    function accepts both filesystem Path objects (for testing) and
    importlib.resources Traversable objects (for bundle loading at
    runtime).
    """
    out: dict[str, PackBundle] = {}
    if not packs_root.is_dir():
        return out
    for child in sorted(packs_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        manifest_path = child.joinpath("manifest.json")
        meanings_path = child.joinpath("meanings.json")
        if not manifest_path.is_file() or not meanings_path.is_file():
            # Work-in-progress pack or non-pack dir — skip silently.
            continue
        with manifest_path.open(encoding="utf-8") as f:
            manifest_payload = json.load(f)
        manifest = _parse_manifest(manifest_payload)
        with meanings_path.open(encoding="utf-8") as f:
            meanings_payload = json.load(f)
        meaning_db, tag_db = load_meanings(meanings_payload)
        if manifest.pack_name in out:
            # Two pack directories declare the same manifest pack_name (e.g.
            # `cp -r packs/foo packs/foo_v2` without editing pack_name). Since
            # the registry keys on pack_name, last-dir-wins would silently drop
            # one pack — raise instead, consistent with the malformed-manifest
            # ValueError above, so the operator notices rather than shipping a
            # registry that's missing a pack.
            raise ValueError(
                f"duplicate pack_name {manifest.pack_name!r}: more than one pack "
                f"directory declares it; rename one so the registry doesn't "
                f"silently drop a pack"
            )
        out[manifest.pack_name] = PackBundle(
            manifest=manifest,
            meaning_db=meaning_db,
            tag_db=tag_db,
        )
    return out
