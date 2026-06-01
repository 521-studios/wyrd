"""Envelope + registry bundle_version stamp (wyrd-dsl5).

The API envelope carries a top-level ``bundle_version`` ONLY when the
generator supplies one via ``runtime_version()`` — keeping the response
shape uniform for generators with no versioned data layer (which return
None by default).
"""

from __future__ import annotations

from typing import Any

from wyrd.envelope import envelope
from wyrd.registry import GenerationResult, Generator


def _results():
    return [GenerationResult(result="Foo")]


def test_envelope_omits_bundle_version_when_none():
    out = envelope(generator="g", parameters={}, seed=1, results=_results())
    assert "bundle_version" not in out


def test_envelope_includes_bundle_version_when_supplied():
    bv = {"built_at": "2026-05-30T00:00:00Z", "schema_version": "2"}
    out = envelope(generator="g", parameters={}, seed=1, results=_results(), bundle_version=bv)
    assert out["bundle_version"] == bv


def test_generator_runtime_version_defaults_to_none():
    class _Dummy(Generator):
        name = "dummy"
        display_name = "Dummy"
        description = "test"

        def input_schema(self) -> dict[str, Any]:
            return {}

        def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
            return GenerationResult(result="x")

    assert _Dummy().runtime_version() is None
