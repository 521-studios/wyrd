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


def test_kenning_scoring_mode_vector_default_uses_uniform_tag_weights():
    """Vector mode with no explicit semantic signal (no tags, no
    mood, no harshness) now falls back to uniform-tag-weights via
    ``build_request_vector``'s no-opinion-default branch — so the
    request still expresses semantic interest (1.0 over the full
    tag universe) and ``baseline_score_native`` returns non-zero
    for any lemma the priors data covers. Pin that the dispatch
    produces a name rather than raising the legacy "no eligible
    name" error.

    Pre-fix contract was: empty register → no eligible name (vector
    mode only worked with a mood). That was the gate on adopting
    vector as a default scoring mode; this test pins the new
    works-by-default contract."""
    k = Kenning()
    result = k.generate(
        {
            "culture": "english",
            "scoring_mode": "vector",
        },
        seed=42,
    )
    assert result.result, "vector mode default-call must produce a non-empty name"


def test_kenning_seed_stable_within_mode():
    """Same seed + same mode → same result (bit-stable contract per
    GenerationResult)."""
    k = Kenning()
    r1 = k.generate({"culture": "english"}, seed=7)
    r2 = k.generate({"culture": "english"}, seed=7)
    assert r1.result == r2.result


def test_kenning_seed_stable_within_vector_mode():
    """Same seed + scoring_mode='vector' → same result. The vector
    path's bit-stability is critical for ecjp.6/7 drift measurement —
    if two runs at the same seed diverge, the drift signal is noise.
    Round-1 reviewer caught the gap that legacy was pinned but the
    vector path wasn't."""
    k = Kenning()
    params = {
        "culture": "english",
        "tags": ["water"],
        "harshness": 0.5,
        "scoring_mode": "vector",
    }
    # Either both raise OR both succeed with identical output. The
    # contract isn't "vector mode succeeds against the live bundle"
    # (that depends on data not yet shipped); it's "same inputs →
    # same outputs whatever those outputs are".
    try:
        r1 = k.generate(params, seed=7)
        r2 = k.generate(params, seed=7)
        assert r1.result == r2.result
    except ValueError:
        # If the first raised, the second must raise identically too.
        with pytest.raises(ValueError):
            k.generate(params, seed=7)


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
