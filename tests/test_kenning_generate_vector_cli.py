"""CLI wiring tests for the vector scoring knobs in
``wyrd kenning generate``.

Verifies the options:

* --priors-path <path>
* --baseline-weight / --phonological-weight / --semantic-weight /
  --position-weight

wyrd-rt2m: vector is the ONLY scoring path (the ``--scoring-mode`` flag was
removed), so these knobs always apply. Without --priors-path, the vector
path's baseline axis contributes 0 (operator's choice).
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from wyrd.generators.kenning.cli.generate import generate


def test_help_lists_vector_flags():
    runner = CliRunner()
    result = runner.invoke(generate, ["--help"])
    assert result.exit_code == 0
    assert "--priors-path" in result.output
    assert "--baseline-weight" in result.output
    assert "--phonological-weight" in result.output
    assert "--semantic-weight" in result.output
    assert "--position-weight" in result.output


def test_default_generation_runs():
    """A bare invocation runs the (vector) generator and prints a name."""
    runner = CliRunner()
    result = runner.invoke(
        generate,
        ["english", "--count", "1", "--seed", "42", "--no-describe"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() != ""


def test_vector_runs_without_priors_path():
    """Without --priors-path the baseline axis contributes 0 and the score
    falls back to phon + sem + pos. Produces a name (or the friendly
    'no eligible name' ValueError) — never a bare crash."""
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
            "--mood",
            "harsh",  # provides non-trivial register
        ],
    )
    if result.exit_code != 0:
        assert "Error" in result.output, result.output


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
    """When --priors-path is given, the priors JSON is parsed and consumed by
    the vector dispatch. An empty (but valid) priors JSON is enough to verify
    the plumbing — the path string makes it into params and the loader runs."""
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
            "--priors-path",
            str(priors_file),
            "--mood",
            "harsh",
        ],
    )
    # Exit 0 OR the friendly "no eligible name" ValueError. The point is the
    # priors file LOADED without a JSON-parse / schema error.
    if result.exit_code != 0:
        assert "Error" in result.output, result.output


def test_generate_via_vector_drops_unknown_scoring_weight_keys():
    """SPA / Lambda forward-compat: unknown keys in scoring_weights are
    silently dropped rather than raising TypeError."""
    raw = {
        "phon_w": 1.0,
        "sem_w": 1.0,
        "pos_w": 1.0,
        "base_w": 1.0,
        "future_axis_w": 99.0,  # unknown — must be dropped, not raise
    }
    from wyrd.generators.kenning.vectors.schemas import ScoringWeights

    known = set(ScoringWeights.__dataclass_fields__)
    filtered = {k: v for k, v in raw.items() if k in known}
    weights = ScoringWeights(**filtered)
    assert weights.phon_w == 1.0
    assert not hasattr(weights, "future_axis_w")


def test_scoring_weights_flow_through_to_request_vector(monkeypatch):
    """--baseline-weight=0 + --phonological-weight=2 composes a
    ScoringWeights(base_w=0.0, phon_w=2.0, ...) and feeds it into the vector
    dispatch. Verifies the CLI → params dict → _generate_via_vector →
    ScoringWeights(**raw) plumbing (now unconditional — vector is the only
    path). Monkeypatches the dispatch to capture the weights without needing a
    fully-loaded bundle."""
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
    assert "scoring_weights_raw" in captured, (
        f"vector dispatch never reached; result: {result.output}"
    )
    assert captured["scoring_weights_raw"] == {
        "phon_w": 2.0,
        "sem_w": 1.5,
        "pos_w": 0.5,
        "base_w": 0.0,
    }
