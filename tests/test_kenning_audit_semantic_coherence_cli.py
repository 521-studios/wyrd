"""Tests for ``audit-semantic-coherence`` (wyrd-36ez) — the wyrd-8uvi-extracted
command phases (``_build_audit_entities`` / ``_embed_entities`` /
``_bucket_by_usage`` / ``_compute_cross_rows`` / ``_embed_per_gloss`` /
``_compute_intra_rows``) and the ``_ollama_embed`` helpers (``_post_embed_request``
/ ``_retry_degenerate_vectors``). The embed transport is stubbed — no Ollama.
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from click.testing import CliRunner

_mod = importlib.import_module("wyrd.generators.kenning.cli.lexicon.audit_semantic_coherence")


def _stub_embed(base_url, model, texts, timeout=60.0):
    """Deterministic unit vectors keyed by text (never degenerate, so the
    retry path doesn't fire)."""
    out = []
    for t in texts:
        h = sum(ord(c) for c in t)
        v = [math.sin(h + k) for k in range(4)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def test_audit_writes_both_suspect_csvs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_mod, "_ollama_embed", _stub_embed)
    bundle = {
        "subjects": [
            {"meaning": ["Homestead"], "words": [{"modern_usage": "-ham", "old_english": ["ham"]}]},
            {"meaning": ["Kiln"], "words": [{"modern_usage": "-ham", "old_english": ["cyln"]}]},
            {"meaning": ["Dwelling"], "words": [{"modern_usage": "-ham", "old_english": ["hamm"]}]},
            {
                "meaning": ["Farmstead", "Enclosure"],
                "words": [{"modern_usage": "-ton", "old_english": ["tun"]}],
            },
        ]
    }
    bpath = tmp_path / "meanings.json"
    bpath.write_text(json.dumps(bundle))
    outdir = tmp_path / "audit"
    result = CliRunner().invoke(
        _mod.lexicon_audit_semantic_coherence,
        ["--meanings", str(bpath), "--output-dir", str(outdir), "--batch-size", "2"],
    )
    assert result.exit_code == 0, result.output
    cross = list(csv.DictReader((outdir / "cross-sibling-suspects.csv").open()))
    intra = list(csv.DictReader((outdir / "intra-entry-suspects.csv").open()))
    # The 3-sibling "-ham" bucket yields one cross row per sibling.
    assert {r["source_lemma"] for r in cross} == {"ham", "cyln", "hamm"}
    # Cross rows are sorted ascending by avg cosine to mates.
    avgs = [float(r["avg_cosine_to_mates"]) for r in cross]
    assert avgs == sorted(avgs)
    # The 2-gloss "-ton" entity is the only intra-entry row.
    assert len(intra) == 1
    assert intra[0]["source_lemma"] == "tun"
    assert intra[0]["n_glosses"] == "2"


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._p = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self) -> bytes:
        return self._p


def test_ollama_embed_retries_degenerate_vector(monkeypatch) -> None:
    calls = {"n": 0}

    def _fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:  # batch: middle vector is degenerate (norm ~0)
            return _FakeResp({"embeddings": [[1.0, 0.0], [0.001, 0.001], [0.0, 1.0]]})
        return _FakeResp({"embeddings": [[0.6, 0.8]]})  # the single-input retry returns a good one

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    vecs = _mod._ollama_embed("http://host:11434/", "m", ["a", "b", "c"])
    assert vecs == [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]  # degenerate slot replaced by retry
    assert calls["n"] == 2  # one batch + one retry


def test_ollama_embed_404_model_not_found_message(monkeypatch) -> None:
    def _fake_404(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url,
            404,
            "Not Found",
            {},
            io.BytesIO(b"model 'm' not found, try pulling it first"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", _fake_404)
    with pytest.raises(Exception) as exc:  # ClickException
        _mod._ollama_embed("http://host:11434", "m", ["x"])
    assert "not available at http://host:11434" in str(exc.value)
    assert "ollama pull m" in str(exc.value)
