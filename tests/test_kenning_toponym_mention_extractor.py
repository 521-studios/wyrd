"""Tests for the LLM-driven toponym-mention extractor (wyrd-x82p Phase 2).

The extractor is provider-agnostic — it takes any object with a
``chat_json(system, user, schema=None) -> dict`` method. Tests use a
``FakeClient`` that replays canned responses, so the test suite never
hits a real LLM API. The real provider clients
(AnthropicClient/OllamaClient/GeminiClient) are already covered by
their own extractor's test files; here we focus on the toponym-
extraction-specific logic (validation, chunking, error handling,
range clamping).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.toponym_mention_extractor import (
    MineToponymMentionsReport,
    ToponymMention,
    _DATE_YEAR_MAX,
    _DATE_YEAR_MIN,
    _validated_mentions,
    chunk_source_body,
    extract_toponym_mentions_from_chunk,
    mine_toponym_mentions,
)


# ---------- FakeClient ---------------------------------------------------


class FakeClient:
    """Minimal stand-in for AnthropicClient/OllamaClient/GeminiClient.

    ``responses`` is consumed in order — one response per ``chat_json``
    call. When the list is exhausted, the next call raises StopIteration
    so over-consumption tests cleanly. Pass ``raises`` instead of a dict
    in any slot to simulate a transport-error scenario.
    """

    def __init__(self, responses: list[Any]):
        self.model = "fake-test-model"
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict | None]] = []

    def chat_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        self.calls.append((system, user, schema))
        if not self._responses:
            raise StopIteration("FakeClient ran out of canned responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


# ---------- _validated_mentions ------------------------------------------


def test_validated_mentions_keeps_form_present_in_chunk():
    chunk = "Edlingham is the homestead of the sons of Eadwulf."
    raw = [
        {"form": "Edlingham", "date_year": None, "region_hint": None, "context": "Edlingham is the homestead"},
    ]
    out = _validated_mentions(raw, chunk)
    assert len(out) == 1
    assert out[0].form == "Edlingham"


def test_validated_mentions_drops_hallucinated_form():
    chunk = "Edlingham is the homestead of the sons of Eadwulf."
    raw = [
        {"form": "Edlingham", "context": "valid"},
        # Hallucinated — Birmingham doesn't appear in the chunk.
        {"form": "Birmingham", "context": "made up"},
    ]
    out = _validated_mentions(raw, chunk)
    assert [m.form for m in out] == ["Edlingham"]


def test_validated_mentions_clamps_out_of_range_year_to_none():
    chunk = "Edlingham in 1086"
    raw = [
        # In range.
        {"form": "Edlingham", "date_year": 1086, "context": "valid"},
    ]
    assert _validated_mentions(raw, chunk)[0].date_year == 1086

    # Below range.
    raw_low = [{"form": "Edlingham", "date_year": _DATE_YEAR_MIN - 1, "context": "valid"}]
    assert _validated_mentions(raw_low, chunk)[0].date_year is None

    # Above range.
    raw_hi = [{"form": "Edlingham", "date_year": _DATE_YEAR_MAX + 1, "context": "valid"}]
    assert _validated_mentions(raw_hi, chunk)[0].date_year is None


def test_validated_mentions_handles_non_int_year():
    chunk = "Edlingham mentioned"
    raw = [
        {"form": "Edlingham", "date_year": "ten eighty-six", "context": "x"},
        {"form": "Edlingham", "date_year": None, "context": "x"},
    ]
    out = _validated_mentions(raw, chunk)
    assert all(m.date_year is None for m in out)


def test_validated_mentions_empty_region_hint_becomes_none():
    chunk = "Edlingham mentioned"
    raw = [
        {"form": "Edlingham", "date_year": None, "region_hint": "", "context": "x"},
        {"form": "Edlingham", "date_year": None, "region_hint": "  ", "context": "x"},
        {"form": "Edlingham", "date_year": None, "region_hint": "Northumberland", "context": "x"},
    ]
    out = _validated_mentions(raw, chunk)
    assert out[0].region_hint is None
    assert out[1].region_hint is None
    assert out[2].region_hint == "Northumberland"


def test_validated_mentions_synthesizes_missing_context_from_chunk():
    chunk = "the village of Edlingham was first attested in 1086 as a small homestead"
    raw = [
        {"form": "Edlingham", "date_year": 1086, "region_hint": None, "context": ""},
    ]
    out = _validated_mentions(raw, chunk)
    assert "Edlingham" in out[0].context
    assert out[0].context  # non-empty


def test_validated_mentions_strips_form_whitespace():
    chunk = "the village of Edlingham"
    raw = [{"form": "  Edlingham  ", "context": "x"}]
    out = _validated_mentions(raw, chunk)
    assert out[0].form == "Edlingham"


def test_validated_mentions_drops_empty_form():
    chunk = "anything"
    raw = [
        {"form": "", "context": "x"},
        {"form": "   ", "context": "x"},
        {"form": None, "context": "x"},
    ]
    assert _validated_mentions(raw, chunk) == []


def test_validated_mentions_skips_non_dict_items():
    chunk = "the village of Edlingham"
    raw = ["a string", 42, None, {"form": "Edlingham", "context": "ok"}]
    out = _validated_mentions(raw, chunk)
    assert [m.form for m in out] == ["Edlingham"]


def test_validated_mentions_collapses_newlines_in_context():
    chunk = "Edlingham mentioned"
    raw = [{"form": "Edlingham", "context": "line one\nline two\nline three"}]
    out = _validated_mentions(raw, chunk)
    assert "\n" not in out[0].context
    assert "line one line two line three" in out[0].context


# ---------- chunk_source_body --------------------------------------------


def test_chunk_source_body_empty_returns_empty_list():
    assert chunk_source_body("") == []
    assert chunk_source_body("   \n\n\t  ") == []


def test_chunk_source_body_single_short_body_one_chunk():
    body = "Single paragraph that fits in one chunk."
    chunks = chunk_source_body(body, target_chunk_size=20000)
    assert chunks == ["Single paragraph that fits in one chunk."]


def test_chunk_source_body_respects_paragraph_boundaries():
    body = "para one is short.\n\npara two is also short.\n\npara three completes the trio."
    chunks = chunk_source_body(body, target_chunk_size=30)
    # Each paragraph is short enough on its own that the splitter should
    # flush after each, OR pack two together if they fit.
    # Reassembled chunks should equal the original (modulo whitespace).
    rejoined = "\n\n".join(chunks)
    assert "para one" in rejoined
    assert "para two" in rejoined
    assert "para three" in rejoined


def test_chunk_source_body_single_long_paragraph_kept_as_one_chunk():
    """If a single paragraph exceeds the target, we keep it whole rather
    than mid-paragraph splitting (snap-to-paragraph invariant)."""
    body = "A" * 50000  # one huge paragraph, no \n\n
    chunks = chunk_source_body(body, target_chunk_size=10000)
    assert chunks == [body]


def test_chunk_source_body_packs_paragraphs_up_to_target():
    p_short = "short."
    body = "\n\n".join([p_short] * 100)
    chunks = chunk_source_body(body, target_chunk_size=100)
    # Each chunk should be ≤ ~100 chars (paragraph-snapping may overshoot
    # slightly when a single paragraph exceeds the target, but here all
    # paragraphs are tiny).
    assert all(len(c) <= 150 for c in chunks)
    # Total content preserved.
    rejoined = "\n\n".join(chunks)
    assert rejoined.count("short.") == 100


# ---------- extract_toponym_mentions_from_chunk --------------------------


def test_extract_one_chunk_happy_path():
    chunk = "Edlingham is in Northumberland. Tynemouth is on the coast."
    client = FakeClient(
        [
            {
                "mentions": [
                    {
                        "form": "Edlingham",
                        "date_year": None,
                        "region_hint": "Northumberland",
                        "context": "Edlingham is in Northumberland",
                    },
                    {
                        "form": "Tynemouth",
                        "date_year": None,
                        "region_hint": None,
                        "context": "Tynemouth is on the coast",
                    },
                ]
            }
        ]
    )
    out = extract_toponym_mentions_from_chunk(client, chunk)
    assert [m.form for m in out] == ["Edlingham", "Tynemouth"]
    assert out[0].region_hint == "Northumberland"
    assert out[1].region_hint is None


def test_extract_one_chunk_non_dict_response_returns_empty():
    client = FakeClient(["not a dict"])
    out = extract_toponym_mentions_from_chunk(client, "Edlingham mentioned")
    assert out == []


def test_extract_one_chunk_non_list_mentions_returns_empty():
    client = FakeClient([{"mentions": "should be a list"}])
    out = extract_toponym_mentions_from_chunk(client, "Edlingham mentioned")
    assert out == []


def test_extract_one_chunk_missing_mentions_key_returns_empty():
    client = FakeClient([{}])
    out = extract_toponym_mentions_from_chunk(client, "Edlingham mentioned")
    assert out == []


def test_extract_one_chunk_runtime_error_propagates():
    """Transport-level failures bubble up to the caller (the per-source
    orchestrator) which decides whether to skip the chunk or abort."""
    client = FakeClient([RuntimeError("Anthropic HTTP 429: rate limited")])
    with pytest.raises(RuntimeError, match="rate limited"):
        extract_toponym_mentions_from_chunk(client, "Edlingham")


# ---------- mine_toponym_mentions (orchestrator) -------------------------


def test_mine_toponym_mentions_aggregates_across_chunks():
    body = "Edlingham is here.\n\n" + ("padding " * 2000) + "\n\nTynemouth is on the coast."
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlingham", "context": "Edlingham is here"}]},
            # The "padding" chunk: no mentions.
            {"mentions": []},
            {"mentions": [{"form": "Tynemouth", "context": "Tynemouth is on the coast"}]},
        ]
    )
    report = mine_toponym_mentions(
        client, "test_source", body, target_chunk_size=10000
    )
    assert isinstance(report, MineToponymMentionsReport)
    assert report.source_id == "test_source"
    # Number of chunks depends on how the splitter packs paragraphs;
    # the assertion that matters is that all admitted mentions appear.
    forms = [m.form for m in report.mentions]
    assert "Edlingham" in forms
    assert "Tynemouth" in forms


def test_mine_toponym_mentions_continues_past_chunk_failure():
    """One chunk failing (transport error) does NOT abort the run —
    the orchestrator skips the failed chunk and increments
    chunks_failed. Per-source pilot operators can re-run later or
    pivot providers for failed chunks."""
    body = "Edlingham mentioned.\n\n" + ("padding " * 2000) + "\n\nTynemouth coast."
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlingham", "context": "Edlingham mentioned"}]},
            RuntimeError("Anthropic HTTP 503: service unavailable"),
            {"mentions": [{"form": "Tynemouth", "context": "Tynemouth coast"}]},
        ]
    )
    report = mine_toponym_mentions(
        client, "test_source", body, target_chunk_size=10000
    )
    # First and third chunk succeeded; middle failed.
    assert report.chunks_failed == 1
    assert report.chunks_processed == 2
    # Both successful chunks contributed mentions.
    forms = [m.form for m in report.mentions]
    assert "Edlingham" in forms
    assert "Tynemouth" in forms


def test_mine_toponym_mentions_empty_body_returns_empty_report():
    client = FakeClient([])
    report = mine_toponym_mentions(client, "test_source", "")
    assert report.source_id == "test_source"
    assert report.chunks_processed == 0
    assert report.mentions == []


def test_mine_toponym_mentions_progress_callback_invoked_per_chunk():
    body = "Edlingham.\n\n" + ("p " * 2000) + "\n\nTynemouth."
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlingham", "context": "Edlingham."}]},
            {"mentions": []},
            {"mentions": [{"form": "Tynemouth", "context": "Tynemouth."}]},
        ]
    )
    progress_calls: list[tuple[int, int, int]] = []

    def on_chunk_done(done: int, total: int, mentions: int) -> None:
        progress_calls.append((done, total, mentions))

    mine_toponym_mentions(
        client, "test_source", body, target_chunk_size=10000, on_chunk_done=on_chunk_done
    )
    # One call per SUCCESSFUL chunk; total stays constant; mentions
    # accumulates monotonically.
    assert len(progress_calls) >= 1
    totals = {t for (_, t, _) in progress_calls}
    assert len(totals) == 1  # constant total across calls
    counts = [c for (_, _, c) in progress_calls]
    assert counts == sorted(counts)  # monotonically non-decreasing


# ---------- CLI integration ----------------------------------------------


def test_cli_validates_missing_source_body(tmp_path, monkeypatch):
    """--source pointing at a non-existent .txt fails with a clear error."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "nonexistent_source",
            "--sources-dir",
            str(sources_dir),
        ],
    )
    assert result.exit_code != 0
    assert "source body not found" in result.output


def test_cli_path_traversal_is_neutralized(tmp_path, monkeypatch):
    """source_id with path-traversal characters should be stripped via
    Path(...).name before lookup — file outside the sources_dir must
    not be readable."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    # No file in sources_dir; if traversal worked it could read /etc/passwd
    # via a .txt extension trick, but Path(...).name strips traversal.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "../etc/passwd",
            "--sources-dir",
            str(sources_dir),
        ],
    )
    # We expect "source body not found" — Path(...).name on "../etc/passwd"
    # is "passwd", which doesn't exist in sources_dir. Confirms the
    # traversal didn't reach outside.
    assert result.exit_code != 0
    assert "source body not found" in result.output


def test_cli_jsonl_output_shape(tmp_path, monkeypatch):
    """End-to-end CLI test with a stubbed client emitting deterministic
    JSON. Verifies the JSONL output shape (one mention per line, expected
    keys present)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    body_file = sources_dir / "stub_source.txt"
    body_file.write_text("Edlingham is in Northumberland.\n", encoding="utf-8")

    # Stub the AnthropicClient factory in the CLI module so the test
    # doesn't make a real API call.
    fake = FakeClient(
        [
            {
                "mentions": [
                    {
                        "form": "Edlingham",
                        "date_year": None,
                        "region_hint": "Northumberland",
                        "context": "Edlingham is in Northumberland",
                    }
                ]
            }
        ]
    )

    import wyrd.generators.kenning.anthropic_extractor as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", lambda **kw: fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    output_path = tmp_path / "out.jsonl"
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub_source",
            "--sources-dir",
            str(sources_dir),
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["source_id"] == "stub_source"
    assert parsed["form"] == "Edlingham"
    assert parsed["region_hint"] == "Northumberland"
    assert parsed["date_year"] is None
    assert "Edlingham" in parsed["context"]
