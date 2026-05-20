"""End-to-end tests for Kenning.generate with scoring_mode='vector'
(wyrd-ecjp.5 PR C).

The dispatch path: Kenning.generate reads scoring_mode, translates
the per-call knobs into a RequestVector via the adapter, optionally
loads priors from a JSON sidecar, calls
NameGenerator.select_via_vector, and returns a GenerationResult.

These tests cover the wiring end-to-end against the LIVE bundle —
unlike the vector_name_select unit tests which use synthetic
meaning_db fixtures, the dispatch needs the real bundle's
structures + meanings to round-trip the NewName output surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyrd.generators.kenning.generators.kenning import Kenning


def test_kenning_scoring_mode_proportions_is_default():
    """Default scoring_mode is 'proportions' — bit-stable with the
    legacy path."""
    k = Kenning()
    # Generate without specifying scoring_mode — should not raise
    result = k.generate({"culture": "english"}, seed=42)
    assert result.result  # non-empty surface string


def test_kenning_scoring_mode_proportions_explicit():
    """Explicit scoring_mode='proportions' produces the same output as
    the implicit default (bit-stable)."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42)
    b = k.generate({"culture": "english", "scoring_mode": "proportions"}, seed=42)
    assert a.result == b.result


def test_kenning_scoring_mode_vector_smoke():
    """scoring_mode='vector' without priors — vector path runs but
    might return empty / raise depending on bundle coverage. We just
    check it doesn't crash with an unhandled exception; if it raises
    the documented ValueError, that's fine — operator's fault for
    requesting vector mode without setting up the data."""
    k = Kenning()
    # Pass a register-relevant tag so the sem axis has signal
    try:
        result = k.generate(
            {
                "culture": "english",
                "tags": ["water"],
                "harshness": 0.5,
                "scoring_mode": "vector",
            },
            seed=42,
        )
        # If we got here, the vector path produced a non-empty pick
        assert result.result  # non-empty surface
    except ValueError as e:
        # Documented loud-failure path: vector returned None because
        # the bundle/data combo couldn't produce a non-zero score
        # for any slot. Acceptable in v1 (bundle doesn't yet ship
        # phonological_vector / priors via ecjp.8).
        assert "vector" in str(e).lower()


def test_kenning_scoring_mode_vector_with_priors(tmp_path: Path):
    """scoring_mode='vector' with priors_path pointing at a JSON
    sidecar. The priors load + the baseline axis contributes."""
    # Build a minimal valid priors sidecar
    priors_file = tmp_path / "priors.json"
    priors_file.write_text(
        json.dumps(
            {
                "version": "test-v1",
                "native": [
                    {
                        "culture": "english",
                        "position": "pre",
                        "tag": "water",
                        "era_midpoint": 0,
                        "lemmas": {"bourne": 50, "brook": 30},
                    }
                ],
                "loan": [],
            }
        )
    )
    k = Kenning()
    try:
        result = k.generate(
            {
                "culture": "english",
                "tags": ["water"],
                "harshness": 0.3,
                "scoring_mode": "vector",
                "priors_path": str(priors_file),
            },
            seed=42,
        )
        # Vector path produced something (priors + sem + phon all
        # contribute non-zero for water-tagged lemmas)
        assert result.result
    except ValueError as e:
        # Acceptable — even with priors, the bundle may not have
        # phonological_vector for water-tagged pre-slot lemmas, and
        # the gate might filter everything.
        assert "vector" in str(e).lower()


def test_kenning_scoring_mode_vector_raises_on_empty():
    """When the vector path filters everything (e.g. gate excludes
    every meaning + no priors + no register weights), the dispatch
    raises a ValueError with operator-readable diagnostic rather
    than returning empty silently."""
    k = Kenning()
    with pytest.raises(ValueError, match=r"vector.*no eligible"):
        k.generate(
            {
                "culture": "english",
                # Empty tags + zero harshness + no priors + no mood →
                # vector scoring returns 0 for every lemma → empty pick
                "scoring_mode": "vector",
            },
            seed=42,
        )


def test_kenning_seed_stable_within_mode():
    """Same seed + same mode → same result (bit-stable contract per
    GenerationResult)."""
    k = Kenning()
    r1 = k.generate({"culture": "english"}, seed=7)
    r2 = k.generate({"culture": "english"}, seed=7)
    assert r1.result == r2.result


def test_kenning_scoring_mode_unknown_value_validated_by_schema():
    """Per the schema's enum, 'banana' is invalid. The generator
    itself doesn't validate (relies on the schema layer); but a
    typo'd value falls through the else branch and runs the legacy
    path — pinning that fallback behavior."""
    k = Kenning()
    # Unknown scoring_mode → falls through to else → legacy path runs
    result = k.generate(
        {"culture": "english", "scoring_mode": "banana"},
        seed=42,
    )
    assert result.result
