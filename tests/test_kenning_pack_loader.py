"""Tests for the scenario-pack loader (wyrd-ecjp.10b Phase 7).

The loader scans a ``packs/`` root directory for per-pack subdirs.
Each subdir with manifest.json + meanings.json becomes a PackBundle.
Used by the bundle-load path (kenning._load_packs) to discover packs
shipped alongside the main meanings.json.

These tests use real tmp_path filesystems so the loader's directory
scan + JSON read paths get exercised end-to-end (the loader itself
just takes a Traversable, so Path objects work).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon.pack_loader import (
    PackBundle,
    _parse_manifest,
    load_packs_from_traversable,
)
from wyrd.generators.kenning.vectors.schemas import PackManifest

# ---- _parse_manifest -----------------------------------------------------


def test_parse_manifest_minimal():
    """Required fields only — defaults kick in for default_weight +
    version."""
    manifest = _parse_manifest(
        {
            "pack_name": "test-pack",
            "template_donor": "old-norse",
            "template_recipient": "old-english",
        }
    )
    assert manifest == PackManifest(
        pack_name="test-pack",
        template_donor="old-norse",
        template_recipient="old-english",
        default_weight=1.0,
        version="unversioned",
    )


def test_parse_manifest_full():
    manifest = _parse_manifest(
        {
            "pack_name": "neo-khuzdul",
            "template_donor": "old-norse",
            "template_recipient": "old-english",
            "default_weight": 0.3,
            "version": "2026-05-20",
        }
    )
    assert manifest.default_weight == 0.3
    assert manifest.version == "2026-05-20"


def test_parse_manifest_missing_required_raises():
    """Missing pack_name surfaces a pointed error rather than crashing
    later during PackOverlay construction."""
    with pytest.raises(ValueError, match="pack_name"):
        _parse_manifest(
            {
                "template_donor": "old-norse",
                "template_recipient": "old-english",
            }
        )


def test_parse_manifest_explicit_null_default_weight_uses_default():
    """A manifest with `"default_weight": null` should fall back to
    1.0, not raise TypeError from float(None). Guards against
    hand-edits or tooling-generated manifests that clear the field
    with null instead of omitting the key."""
    manifest = _parse_manifest(
        {
            "pack_name": "test-pack",
            "template_donor": "old-norse",
            "template_recipient": "old-english",
            "default_weight": None,
            "version": None,
        }
    )
    assert manifest.default_weight == 1.0
    assert manifest.version == "unversioned"


def test_parse_manifest_missing_multiple_required_lists_all():
    """All missing required fields surface in one error so the
    operator doesn't fix-rerun-fix one at a time."""
    with pytest.raises(ValueError, match="pack_name.*template_donor"):
        _parse_manifest({"template_recipient": "old-english"})


# ---- load_packs_from_traversable ----------------------------------------


def _make_pack(
    packs_root: Path,
    pack_name: str,
    *,
    template_donor: str = "old-norse",
    template_recipient: str = "old-english",
    subjects: list | None = None,
    skip_manifest: bool = False,
    skip_meanings: bool = False,
) -> None:
    """Create a synthetic pack on disk for the loader to scan.

    The test fixtures use minimal valid manifest + meanings shapes —
    enough for the loader to parse without exercising the full Meaning
    construction surface.
    """
    pack_dir = packs_root / pack_name
    pack_dir.mkdir(parents=True, exist_ok=True)
    if not skip_manifest:
        manifest = {
            "pack_name": pack_name,
            "template_donor": template_donor,
            "template_recipient": template_recipient,
        }
        (pack_dir / "manifest.json").write_text(json.dumps(manifest))
    if not skip_meanings:
        meanings = {"subjects": subjects or []}
        (pack_dir / "meanings.json").write_text(json.dumps(meanings))


def test_load_packs_returns_empty_when_root_absent(tmp_path: Path):
    """A bundle that ships no packs has no packs/ dir. Loader returns
    empty dict (legacy behavior — runtime treats this as 'no packs
    declared')."""
    nonexistent = tmp_path / "nonexistent_packs"
    assert load_packs_from_traversable(nonexistent) == {}


def test_load_packs_returns_empty_for_empty_dir(tmp_path: Path):
    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    assert load_packs_from_traversable(packs_root) == {}


def test_load_packs_loads_single_pack(tmp_path: Path):
    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    _make_pack(packs_root, "test-pack")
    result = load_packs_from_traversable(packs_root)
    assert "test-pack" in result
    bundle = result["test-pack"]
    assert isinstance(bundle, PackBundle)
    assert bundle.manifest.pack_name == "test-pack"
    assert bundle.manifest.template_donor == "old-norse"
    assert bundle.meaning_db == {}  # empty subjects → empty meaning_db
    assert bundle.tag_db == {}


def test_load_packs_loads_multiple_packs(tmp_path: Path):
    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    _make_pack(packs_root, "pack-a", template_donor="d1", template_recipient="r1")
    _make_pack(packs_root, "pack-b", template_donor="d2", template_recipient="r2")
    result = load_packs_from_traversable(packs_root)
    assert set(result.keys()) == {"pack-a", "pack-b"}
    assert result["pack-a"].manifest.template_donor == "d1"
    assert result["pack-b"].manifest.template_donor == "d2"


def test_load_packs_skips_dir_without_manifest(tmp_path: Path):
    """Operator might have a WIP pack dir with only meanings.json
    so far. Loader skips it silently rather than failing the whole
    bundle load."""
    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    _make_pack(packs_root, "complete-pack")
    _make_pack(packs_root, "wip-pack", skip_manifest=True)
    result = load_packs_from_traversable(packs_root)
    assert set(result.keys()) == {"complete-pack"}


def test_load_packs_skips_dir_without_meanings(tmp_path: Path):
    """Symmetric: manifest exists but meanings.json hasn't been
    authored yet. Skip silently."""
    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    _make_pack(packs_root, "complete-pack")
    _make_pack(packs_root, "draft-pack", skip_meanings=True)
    result = load_packs_from_traversable(packs_root)
    assert set(result.keys()) == {"complete-pack"}


def test_load_packs_skips_non_directory_entries(tmp_path: Path):
    """A README.md file at packs/ root shouldn't be treated as a
    pack. Only subdirectories with both files are loaded."""
    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    (packs_root / "README.md").write_text("# Packs catalog")
    _make_pack(packs_root, "real-pack")
    result = load_packs_from_traversable(packs_root)
    assert set(result.keys()) == {"real-pack"}


def test_load_packs_keys_by_manifest_pack_name_not_dirname(tmp_path: Path):
    """The dict key comes from manifest.pack_name, NOT the directory
    name. An operator can rename a directory without changing the
    pack's registered name."""
    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    # Directory is "tatar-9c-draft" but manifest pack_name is "tatar-9c"
    pack_dir = packs_root / "tatar-9c-draft"
    pack_dir.mkdir()
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pack_name": "tatar-9c",
                "template_donor": "mongol",
                "template_recipient": "old-russian",
            }
        )
    )
    (pack_dir / "meanings.json").write_text(json.dumps({"subjects": []}))
    result = load_packs_from_traversable(packs_root)
    assert "tatar-9c" in result
    assert "tatar-9c-draft" not in result


def test_load_packs_with_real_subjects_populates_meaning_db(tmp_path: Path):
    """End-to-end: a real subject in the pack's meanings.json lands
    on bundle.meaning_db keyed by modern_usage."""
    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    subjects = [
        {
            "meaning": ["Fortress"],
            "modifier_tags": ["fortified"],
            "modifier_type": "Topographical",
            "words": [{"modern_usage": "-keep", "old_english": ["cȳpe"]}],
        }
    ]
    _make_pack(packs_root, "test-pack", subjects=subjects)
    result = load_packs_from_traversable(packs_root)
    bundle = result["test-pack"]
    assert "-keep" in bundle.meaning_db
    assert len(bundle.meaning_db["-keep"]) == 1
    assert "fortified" in bundle.tag_db


def test_load_packs_propagates_malformed_manifest_error(tmp_path: Path):
    """A malformed manifest (missing required field) raises so the
    operator notices rather than silently shipping a broken pack.
    The loader's tolerance is for missing FILES (skip), not for
    malformed PRESENT files (raise)."""
    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    pack_dir = packs_root / "broken-pack"
    pack_dir.mkdir()
    # Missing template_donor + template_recipient
    (pack_dir / "manifest.json").write_text(json.dumps({"pack_name": "broken"}))
    (pack_dir / "meanings.json").write_text(json.dumps({"subjects": []}))
    with pytest.raises(ValueError, match="template_donor"):
        load_packs_from_traversable(packs_root)


def test_load_packs_duplicate_pack_name_raises(tmp_path: Path):
    """Two pack DIRECTORIES declaring the same manifest pack_name must raise,
    not silently drop one. The registry keys on pack_name (dir-name-independent
    by design), so `cp -r packs/foo packs/foo_v2` without editing pack_name
    would otherwise let the second dir clobber the first — a pack vanishes with
    no error. Consistent with the malformed-manifest ValueError so the operator
    notices rather than shipping a registry missing a pack."""
    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    for dir_name in ("foo", "foo_copy"):
        pack_dir = packs_root / dir_name
        pack_dir.mkdir()
        (pack_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "pack_name": "foo",  # SAME pack_name in two different dirs
                    "template_donor": "old-norse",
                    "template_recipient": "old-english",
                }
            )
        )
        (pack_dir / "meanings.json").write_text(json.dumps({"subjects": []}))
    with pytest.raises(ValueError, match="duplicate pack_name"):
        load_packs_from_traversable(packs_root)
