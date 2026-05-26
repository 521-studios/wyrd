"""Tests for the bundled-priors loader + payload parser
(wyrd-ecjp.10b Phase 7; d90t L4 cutover).

Pre-cutover the bundle shipped a ``priors.json`` sidecar; post-cutover
the priors payload lives in the L4 ``empirical_priors`` singleton row.
``kenning._load_empirical_priors`` reads it via the L4 adapter,
returning an empty ``EmpiricalPriors`` when the row is absent so the
vector-scoring fallback path stays intact (degrades to phon + sem +
pos when no baseline data is available).
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


def test_load_empirical_priors_reads_l4_singleton_when_present(tmp_path: Path, monkeypatch):
    """Positive-path: when the L4 carries an ``empirical_priors``
    singleton row, ``_load_empirical_priors`` reads + parses it into
    a populated ``EmpiricalPriors``. Closes the integration-coverage
    gap from the payload-only + absent-row tests."""
    import wyrd.generators.kenning as kenning_mod
    from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
    from wyrd.generators.kenning.lexicon.runtime_db_export import write_runtime_db
    from wyrd.generators.kenning.runtime.runtime_db import reset_runtime_db_cache

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

    lexicon_path = tmp_path / "lexicon.db"
    init_schema(lexicon_path)
    with LexiconDB(lexicon_path) as db:
        db.upsert_source(id="test", title="test")
        db.commit()

    out_path = tmp_path / "runtime.db"
    proportions_dir = tmp_path / "proportions"
    proportions_dir.mkdir()
    for culture in ("english", "scottish", "welsh", "irish", "breton"):
        (proportions_dir / f"{culture}_proportions.json").write_text(
            json.dumps(
                {
                    "usages": {},
                    "single_usages": {},
                    "structures": [],
                    "tag_marginal": {},
                    "tag_cooccurrence": {},
                }
            )
        )
    write_runtime_db(
        output_path=out_path,
        subjects=[],
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_dir=proportions_dir,
        source_lexicon_db=lexicon_path,
        empirical_priors_payload=payload,
    )

    monkeypatch.setenv("WYRD_RUNTIME_DB", str(out_path))
    reset_runtime_db_cache()
    kenning_mod._load_empirical_priors.cache_clear()
    try:
        result = kenning_mod._load_empirical_priors()
        assert result.version == "test-v1"
        key = ("english", "post", "fortified", 1086)
        assert key in result.native
        assert result.native[key] == {"burh": 100.0}
    finally:
        kenning_mod._load_empirical_priors.cache_clear()
        reset_runtime_db_cache()


def test_load_empirical_priors_returns_empty_when_no_l4_row(tmp_path: Path, monkeypatch):
    """When the L4 has no ``empirical_priors`` row, the loader returns
    an empty ``EmpiricalPriors``. The vector-scoring fallback treats
    this as 'no baseline axis' and degrades to phon + sem + pos —
    same as legacy behavior."""
    import wyrd.generators.kenning as kenning_mod
    from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
    from wyrd.generators.kenning.lexicon.runtime_db_export import write_runtime_db
    from wyrd.generators.kenning.runtime.runtime_db import reset_runtime_db_cache

    lexicon_path = tmp_path / "lexicon.db"
    init_schema(lexicon_path)
    with LexiconDB(lexicon_path) as db:
        db.upsert_source(id="test", title="test")
        db.commit()

    out_path = tmp_path / "runtime.db"
    proportions_dir = tmp_path / "proportions"
    proportions_dir.mkdir()
    for culture in ("english", "scottish", "welsh", "irish", "breton"):
        (proportions_dir / f"{culture}_proportions.json").write_text(
            json.dumps(
                {
                    "usages": {},
                    "single_usages": {},
                    "structures": [],
                    "tag_marginal": {},
                    "tag_cooccurrence": {},
                }
            )
        )
    # No empirical_priors_payload → no row written.
    write_runtime_db(
        output_path=out_path,
        subjects=[],
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_dir=proportions_dir,
        source_lexicon_db=lexicon_path,
    )

    monkeypatch.setenv("WYRD_RUNTIME_DB", str(out_path))
    reset_runtime_db_cache()
    kenning_mod._load_empirical_priors.cache_clear()
    try:
        result = kenning_mod._load_empirical_priors()
        assert isinstance(result, EmpiricalPriors)
        assert result.native == {}
        assert result.loan_relationship == {}
    finally:
        kenning_mod._load_empirical_priors.cache_clear()
        reset_runtime_db_cache()
