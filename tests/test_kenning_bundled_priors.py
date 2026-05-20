"""Tests for the bundled-priors loader + payload parser
(wyrd-ecjp.10b Phase 7).

The bundle may ship a priors.json sidecar alongside meanings.json.
``kenning._load_empirical_priors`` reads it if present, else returns
an empty EmpiricalPriors so the vector-scoring fallback path stays
intact (degrades to phon + sem + pos when no baseline data is
available).
"""

from __future__ import annotations

import json
from pathlib import Path

from wyrd.generators.kenning.lexicon.empirical_priors import (
    load_empirical_priors_from_json,
    load_empirical_priors_from_payload,
)
from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

# ---- load_empirical_priors_from_payload ---------------------------------


def test_load_empirical_priors_from_payload_empty():
    """Empty payload (no native / no loan cells) returns an empty
    EmpiricalPriors instance, not None."""
    result = load_empirical_priors_from_payload({})
    assert isinstance(result, EmpiricalPriors)
    assert result.native == {}
    assert result.loan_relationship == {}


def test_load_empirical_priors_from_payload_native_cells():
    payload = {
        "native": [
            {
                "culture": "english",
                "position": "post",
                "tag": "fortified",
                "era_midpoint": 1086,
                "lemmas": {"burh": 142.0, "ceaster": 89.0},
            }
        ]
    }
    result = load_empirical_priors_from_payload(payload)
    key = ("english", "post", "fortified", 1086)
    assert key in result.native
    assert result.native[key] == {"burh": 142.0, "ceaster": 89.0}


def test_load_empirical_priors_from_payload_loan_cells():
    payload = {
        "loan": [
            {
                "donor": "old-norse",
                "recipient": "old-english",
                "position": "post",
                "tag": "settlement",
                "era_midpoint": 950,
                "lemmas": {"-by": 412.0},
            }
        ]
    }
    result = load_empirical_priors_from_payload(payload)
    key = ("old-norse", "old-english", "post", "settlement", 950)
    assert key in result.loan_relationship
    assert result.loan_relationship[key] == {"-by": 412.0}


def test_load_empirical_priors_from_payload_version_preserved():
    payload = {"version": "2026-05-20", "native": [], "loan": []}
    result = load_empirical_priors_from_payload(payload)
    assert result.version == "2026-05-20"


def test_load_empirical_priors_from_payload_default_version():
    """Payload without a version key defaults to 'unversioned'."""
    result = load_empirical_priors_from_payload({})
    assert result.version == "unversioned"


# ---- round-trip: from_json delegates to from_payload --------------------


def test_load_empirical_priors_from_json_matches_payload_path(tmp_path: Path):
    """The file-path loader should produce the same EmpiricalPriors
    as the in-memory payload loader. Regression guard against the
    refactor that split the two entry points."""
    payload = {
        "version": "v1",
        "native": [
            {
                "culture": "english",
                "position": "post",
                "tag": "river",
                "era_midpoint": 1000,
                "lemmas": {"-ford": 50.0, "-bridge": 30.0},
            }
        ],
        "loan": [
            {
                "donor": "norman-french",
                "recipient": "old-english",
                "position": "pre",
                "tag": "manorial",
                "era_midpoint": 1086,
                "lemmas": {"Mont-": 12.0},
            }
        ],
    }
    sidecar = tmp_path / "priors.json"
    sidecar.write_text(json.dumps(payload))

    from_file = load_empirical_priors_from_json(sidecar)
    from_payload = load_empirical_priors_from_payload(payload)

    assert from_file.native == from_payload.native
    assert from_file.loan_relationship == from_payload.loan_relationship
    assert from_file.version == from_payload.version


# ---- kenning._load_empirical_priors fallback behavior -------------------


def test_load_empirical_priors_reads_bundled_sidecar_when_present(tmp_path: Path, monkeypatch):
    """Positive-path: when _data_path resolves to a real priors.json
    on disk, the loader reads + parses it into a populated
    EmpiricalPriors. Closes the integration-coverage gap from the
    payload-only + absent-file tests."""
    import wyrd.generators.kenning as kenning_mod

    payload = {
        "version": "test-v1",
        "native": [
            {
                "culture": "english",
                "position": "post",
                "tag": "fortified",
                "era_midpoint": 1086,
                "lemmas": {"burh": 100.0},
            }
        ],
        "loan": [],
    }
    priors_file = tmp_path / "priors.json"
    priors_file.write_text(json.dumps(payload))

    def fake_data_path(filename: str):
        if filename == "priors.json":
            return priors_file
        return kenning_mod._data_path.__wrapped__(filename)  # type: ignore[attr-defined]

    monkeypatch.setattr(kenning_mod, "_data_path", fake_data_path)
    kenning_mod._load_empirical_priors.cache_clear()
    try:
        result = kenning_mod._load_empirical_priors()
        assert result.version == "test-v1"
        key = ("english", "post", "fortified", 1086)
        assert key in result.native
        assert result.native[key] == {"burh": 100.0}
    finally:
        kenning_mod._load_empirical_priors.cache_clear()


def test_load_empirical_priors_returns_empty_when_no_bundled_priors(monkeypatch):
    """When the bundle has no priors.json, the bundled-priors loader
    returns an empty EmpiricalPriors. The vector-scoring fallback
    treats this as 'no baseline axis' and degrades to phon + sem +
    pos — same as legacy behavior.

    This test asserts the no-file branch by monkey-patching
    _data_path to return a path that doesn't exist."""
    import wyrd.generators.kenning as kenning_mod

    # Clear the LRU cache so the monkeypatched _data_path is used
    kenning_mod._load_empirical_priors.cache_clear()

    class _FakeTraversable:
        def is_file(self) -> bool:
            return False

    def fake_data_path(filename: str):
        if filename == "priors.json":
            return _FakeTraversable()
        return kenning_mod._data_path.__wrapped__(filename)  # type: ignore[attr-defined]

    # Monkeypatch the module-level _data_path so the loader picks up
    # the fake. Restore after the test via monkeypatch teardown.
    monkeypatch.setattr(kenning_mod, "_data_path", fake_data_path)
    kenning_mod._load_empirical_priors.cache_clear()
    try:
        result = kenning_mod._load_empirical_priors()
        assert isinstance(result, EmpiricalPriors)
        assert result.native == {}
        assert result.loan_relationship == {}
    finally:
        kenning_mod._load_empirical_priors.cache_clear()
