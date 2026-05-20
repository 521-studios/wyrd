"""Tests for ``load_empirical_priors_from_json`` (wyrd-ecjp.5 prereq).

Counterpart to the existing ``dump_empirical_priors_to_json`` writer
(tested in test_kenning_empirical_priors.py). The vector-scoring
runtime reads the JSON sidecar at Lambda startup; this loader is the
runtime-side reverse of the lexicon-side dumper.

Round-trip: dump → load → byte-identical re-dump. That's the
bit-stability contract the priors-version invariant relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.empirical_priors import (
    dump_empirical_priors_to_json,
    load_empirical_priors_from_json,
)
from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors


def _write_artifact(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def test_load_minimal_artifact(tmp_path: Path):
    """Smallest valid artifact: version + empty native + empty loan."""
    p = tmp_path / "priors.json"
    _write_artifact(p, {"version": "v1", "native": [], "loan": []})
    priors = load_empirical_priors_from_json(p)
    assert priors.version == "v1"
    assert priors.native == {}
    assert priors.loan_relationship == {}


def test_load_native_cell(tmp_path: Path):
    """Native cell key is the 4-tuple (culture, position, tag, era_midpoint)."""
    p = tmp_path / "priors.json"
    _write_artifact(
        p,
        {
            "version": "v1",
            "native": [
                {
                    "culture": "english",
                    "position": "pre",
                    "tag": "animal",
                    "era_midpoint": 1100,
                    "lemmas": {"old-english:wulf": 5, "old-english:bera": 3},
                },
            ],
            "loan": [],
        },
    )
    priors = load_empirical_priors_from_json(p)
    key = ("english", "pre", "animal", 1100)
    assert key in priors.native
    assert priors.native[key] == {"old-english:wulf": 5.0, "old-english:bera": 3.0}


def test_load_loan_cell(tmp_path: Path):
    """Loan cell key is the 5-tuple (donor, recipient, position, tag, era_midpoint)."""
    p = tmp_path / "priors.json"
    _write_artifact(
        p,
        {
            "version": "v1",
            "native": [],
            "loan": [
                {
                    "donor": "old-norse",
                    "recipient": "old-english",
                    "position": "post",
                    "tag": "water",
                    "era_midpoint": 1000,
                    "lemmas": {"old-norse:bekkr": 7},
                },
            ],
        },
    )
    priors = load_empirical_priors_from_json(p)
    key = ("old-norse", "old-english", "post", "water", 1000)
    assert key in priors.loan_relationship
    assert priors.loan_relationship[key] == {"old-norse:bekkr": 7.0}


def test_load_tolerates_missing_optional_keys(tmp_path: Path):
    """An older / partial artifact (missing 'loan' entirely) loads cleanly —
    the missing side defaults to empty dict so the scoring runtime
    degrades gracefully on pack lookup misses."""
    p = tmp_path / "priors.json"
    _write_artifact(p, {"version": "partial-v1", "native": []})
    priors = load_empirical_priors_from_json(p)
    assert priors.loan_relationship == {}


def test_load_tolerates_missing_version(tmp_path: Path):
    """No version field → defaults to 'unversioned' (matches the dumper's
    default when --version is omitted)."""
    p = tmp_path / "priors.json"
    _write_artifact(p, {"native": [], "loan": []})
    priors = load_empirical_priors_from_json(p)
    assert priors.version == "unversioned"


def test_load_coerces_count_strings_to_floats(tmp_path: Path):
    """An artifact whose counts arrive as JSON strings (from a manually-
    edited file) coerces to float rather than failing. Defensive
    against operator hand-edits."""
    p = tmp_path / "priors.json"
    _write_artifact(
        p,
        {
            "version": "v1",
            "native": [
                {
                    "culture": "english",
                    "position": "pre",
                    "tag": "animal",
                    "era_midpoint": 1100,
                    "lemmas": {"old-english:wulf": "5", "old-english:bera": "3"},
                },
            ],
            "loan": [],
        },
    )
    priors = load_empirical_priors_from_json(p)
    assert priors.native[("english", "pre", "animal", 1100)] == {
        "old-english:wulf": 5.0,
        "old-english:bera": 3.0,
    }


def test_load_coerces_era_midpoint_string_to_int(tmp_path: Path):
    """Same defensive coercion for era_midpoint — operator-edited JSON
    may stringify integers."""
    p = tmp_path / "priors.json"
    _write_artifact(
        p,
        {
            "version": "v1",
            "native": [
                {
                    "culture": "english",
                    "position": "pre",
                    "tag": "animal",
                    "era_midpoint": "1100",
                    "lemmas": {"old-english:wulf": 5},
                },
            ],
            "loan": [],
        },
    )
    priors = load_empirical_priors_from_json(p)
    # The int(...) cast in the loader normalizes; tuple-key lookup with
    # int 1100 succeeds.
    assert ("english", "pre", "animal", 1100) in priors.native


def test_returns_empirical_priors_dataclass(tmp_path: Path):
    """Confirms the loader returns the actual dataclass type (not a
    bare dict) so downstream consumers can type-annotate against it."""
    p = tmp_path / "priors.json"
    _write_artifact(p, {"version": "v1", "native": [], "loan": []})
    priors = load_empirical_priors_from_json(p)
    assert isinstance(priors, EmpiricalPriors)


def test_round_trip_dump_load_dump_byte_identical(tmp_path: Path):
    """The bit-stability contract: a dump → load → dump cycle produces
    a byte-identical second artifact. Critical for the priors-version
    invariant (re-loading + re-dumping shouldn't drift the bytes the
    operator committed)."""
    db_path = tmp_path / "test.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    # Hand-insert into the two priors tables. Bypass the mining
    # pipeline since this test is about the dump/load round-trip,
    # not the extraction logic.
    db.conn.execute(
        "INSERT INTO empirical_priors_native "
        "(culture, position, tag, era_midpoint, lemma_ref, count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("english", "pre", "animal", 1100, "old-english:wulf", 5),
    )
    db.conn.execute(
        "INSERT INTO empirical_priors_native "
        "(culture, position, tag, era_midpoint, lemma_ref, count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("english", "post", "water", 1100, "old-english:burna", 4),
    )
    db.conn.execute(
        "INSERT INTO empirical_priors_loan "
        "(donor, recipient, position, tag, era_midpoint, lemma_ref, count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("old-norse", "old-english", "post", "water", 1000, "old-norse:bekkr", 7),
    )
    db.conn.commit()

    first_path = tmp_path / "priors_first.json"
    dump_empirical_priors_to_json(db, first_path, version="v1")

    # Load + re-dump from a different DB seeded from the loader output.
    priors_loaded = load_empirical_priors_from_json(first_path)
    # The dataclass round-trips the cell counts faithfully; assert by
    # re-rendering JSON in the same shape and comparing bytes.
    second_path = tmp_path / "priors_second.json"
    payload = {
        "version": priors_loaded.version,
        "native": [
            {
                "culture": k[0],
                "position": k[1],
                "tag": k[2],
                "era_midpoint": k[3],
                "lemmas": {ref: int(c) for ref, c in v.items()},
            }
            for k, v in sorted(priors_loaded.native.items())
        ],
        "loan": [
            {
                "donor": k[0],
                "recipient": k[1],
                "position": k[2],
                "tag": k[3],
                "era_midpoint": k[4],
                "lemmas": {ref: int(c) for ref, c in v.items()},
            }
            for k, v in sorted(priors_loaded.loan_relationship.items())
        ],
    }
    with second_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    assert first_path.read_bytes() == second_path.read_bytes(), (
        "dump → load → re-emit produced byte-different JSON; bit-stability broken"
    )
    db.close()


def test_load_raises_on_missing_file(tmp_path: Path):
    """Missing file → FileNotFoundError, not silent empty-priors fallback.
    Loud failure so an operator can't accidentally Lambda-deploy with a
    missing priors artifact."""
    p = tmp_path / "does-not-exist.json"
    with pytest.raises(FileNotFoundError):
        load_empirical_priors_from_json(p)


def test_load_raises_on_malformed_json(tmp_path: Path):
    """Malformed JSON → json.JSONDecodeError propagates. Same loud-
    failure rationale as missing-file: an operator shouldn't be able
    to Lambda-deploy a corrupted priors artifact and silently get
    empty priors. JSONDecodeError is a ValueError subclass."""
    p = tmp_path / "broken.json"
    p.write_text("{not valid json")
    with pytest.raises(ValueError):
        load_empirical_priors_from_json(p)
