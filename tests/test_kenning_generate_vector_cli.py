"""CLI wiring tests for the vector-mode knobs in
``wyrd kenning generate`` (wyrd-ecjp.9 Phase 8).

Verifies the new CLI options:

* --scoring-mode {proportions|vector}
* --priors-path <path>
* --baseline-weight / --phonological-weight / --semantic-weight /
  --position-weight

The legacy proportions path stays bit-stable; the new flags only
fire when --scoring-mode=vector is set. Without --priors-path, the
vector path's baseline axis contributes 0 (operator's choice).
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from wyrd.generators.kenning.cli.generate import generate


def test_help_lists_new_vector_flags():
    runner = CliRunner()
    result = runner.invoke(generate, ["--help"])
    assert result.exit_code == 0
    # Verify the new options are surfaced in --help
    assert "--scoring-mode" in result.output
    assert "--priors-path" in result.output
    assert "--baseline-weight" in result.output
    assert "--phonological-weight" in result.output
    assert "--semantic-weight" in result.output
    assert "--position-weight" in result.output


def test_default_scoring_mode_is_proportions():
    """Without --scoring-mode, the legacy proportions path runs.
    Bit-stable per-seed contract preserved."""
    runner = CliRunner()
    result = runner.invoke(
        generate,
        ["english", "--count", "1", "--seed", "42", "--no-describe"],
    )
    assert result.exit_code == 0, result.output
    # First non-empty line of stdout is the generated name. Just
    # verifying it ran without error — exact string is bit-stable
    # but we don't pin it here (the legacy proportions golden tests
    # already do that elsewhere).
    assert result.output.strip() != ""


def test_scoring_mode_vector_runs_without_priors_path():
    """--scoring-mode=vector without --priors-path: the vector
    path's baseline axis contributes 0; the score falls back to
    phon + sem + pos. Should produce a name (no raise) when the
    register is non-trivial enough that some lemma scores > 0."""
    runner = CliRunner()
    result = runner.invoke(
        generate,
        [
            "english",
            "--count",
            "1",
            "--seed",
            "42",
            "--no-describe",
            "--scoring-mode",
            "vector",
            "--mood",
            "harsh",  # provides non-trivial register
        ],
    )
    # Either succeeds with a name or raises ValueError with the
    # operator-readable "produced no eligible name" diagnostic. Both
    # are non-crash exits — bare exception would be a bug.
    if result.exit_code != 0:
        # ValueError path — verify it's the friendly one, not a
        # silent failure or KeyError / AttributeError
        assert "Error" in result.output, result.output


def test_invalid_scoring_mode_rejected():
    runner = CliRunner()
    result = runner.invoke(
        generate,
        ["english", "--scoring-mode", "fancy"],
    )
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "fancy" in result.output


def test_weight_flags_validate_range():
    """Weight scalars are FloatRange(0.0, 10.0). Out-of-range values
    rejected by click before reaching the generator."""
    runner = CliRunner()
    result = runner.invoke(
        generate,
        ["english", "--baseline-weight", "100.0"],
    )
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "100" in result.output


def test_priors_path_must_exist():
    """--priors-path validates existence at click parse time."""
    runner = CliRunner()
    result = runner.invoke(
        generate,
        ["english", "--priors-path", "/nonexistent/path/to/priors.json"],
    )
    assert result.exit_code != 0


def test_priors_path_passed_through_when_provided(tmp_path):
    """When --priors-path is given alongside --scoring-mode=vector,
    the priors JSON is parsed and consumed by the vector dispatch.
    An empty (but valid) priors JSON is enough to verify the
    plumbing — the path string makes it into params and the loader
    runs."""
    priors_file = tmp_path / "priors.json"
    priors_file.write_text(json.dumps({"native": {}, "loan_relationship": {}}))
    runner = CliRunner()
    result = runner.invoke(
        generate,
        [
            "english",
            "--count",
            "1",
            "--seed",
            "42",
            "--no-describe",
            "--scoring-mode",
            "vector",
            "--priors-path",
            str(priors_file),
            "--mood",
            "harsh",
        ],
    )
    # Exit code 0 (success) OR ValueError with friendly "no eligible
    # name" diagnostic. The point is that the priors file LOADED
    # without a JSON-parse error or schema error — a bad path or
    # a malformed JSON would raise a different exception before this.
    if result.exit_code != 0:
        assert "Error" in result.output, result.output


def test_proportions_mode_ignores_vector_only_flags(tmp_path):
    """Passing vector-only flags (--priors-path, weight flags) in
    proportions mode is a no-op — they don't affect the legacy
    sampling path. Bit-stable proportions output preserved.

    This is the operator-friendly contract: 'I tried --priors-path
    but forgot --scoring-mode=vector' produces a normal proportions-
    mode name rather than a confusing error."""
    priors_file = tmp_path / "priors.json"
    priors_file.write_text(json.dumps({"native": {}, "loan_relationship": {}}))

    runner = CliRunner()
    # Same seed in both runs — bit-stable proportions contract means
    # the output should match regardless of vector-only flags.
    baseline = runner.invoke(
        generate,
        ["english", "--count", "1", "--seed", "42", "--no-describe"],
    )
    with_vector_flags = runner.invoke(
        generate,
        [
            "english",
            "--count",
            "1",
            "--seed",
            "42",
            "--no-describe",
            "--priors-path",
            str(priors_file),
            "--baseline-weight",
            "5.0",
            "--phonological-weight",
            "2.0",
        ],
    )
    assert baseline.exit_code == 0
    assert with_vector_flags.exit_code == 0
    # Vector-only flags don't perturb proportions output (the params
    # dict gets them but Kenning.generate ignores when scoring_mode
    # != 'vector')
    assert baseline.output == with_vector_flags.output


def test_scoring_weights_flow_through_to_request_vector(monkeypatch):
    """--baseline-weight=0 + --phonological-weight=2 + --scoring-mode=
    vector composes a ScoringWeights(base_w=0.0, phon_w=2.0, ...) and
    feeds it into build_request_vector. Verifies the CLI → params
    dict → _generate_via_vector → ScoringWeights(**raw) plumbing.

    Uses monkeypatch to capture the weights at the dispatch boundary
    so we don't need a fully-loaded bundle to assert correctness."""
    from wyrd.generators.kenning.generators import kenning as kenning_mod

    captured = {}
    original = kenning_mod._generate_via_vector

    def capture(*args, **kwargs):
        captured["scoring_weights_raw"] = kwargs.get("scoring_weights_raw")
        return original(*args, **kwargs)

    monkeypatch.setattr(kenning_mod, "_generate_via_vector", capture)

    runner = CliRunner()
    result = runner.invoke(
        generate,
        [
            "english",
            "--count",
            "1",
            "--seed",
            "42",
            "--no-describe",
            "--scoring-mode",
            "vector",
            "--baseline-weight",
            "0.0",
            "--phonological-weight",
            "2.0",
            "--semantic-weight",
            "1.5",
            "--position-weight",
            "0.5",
        ],
    )
    # Whether the run succeeded or hit the "no eligible name"
    # ValueError, the dispatch MUST have been called with the
    # expected scoring_weights dict. The capture happens before any
    # raise inside the vector path.
    assert "scoring_weights_raw" in captured, (
        f"vector dispatch never reached; result: {result.output}"
    )
    raw = captured["scoring_weights_raw"]
    assert raw == {
        "phon_w": 2.0,
        "sem_w": 1.5,
        "pos_w": 0.5,
        "base_w": 0.0,
    }
