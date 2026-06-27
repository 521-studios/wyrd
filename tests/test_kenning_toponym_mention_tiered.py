"""Tests for the two-tier LLM mention extractor + multi-source CLI
(wyrd-x82p Phase 2b.2).

The tiered orchestrator runs a primary client on every chunk and
retries chunks where the primary failed (transport, JSON parse,
schema mismatch) with a fallback client. The CLI walks all sources
under a directory, runs the tiered extraction per source, and
writes per-source JSONL output ready for the Phase 2b.1 ingester.

Tests use ``FakeClient`` (same shape as the single-tier suite) to
avoid real LLM API calls.
"""

from __future__ import annotations

import json
from typing import Any

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.extractors.toponym_mentions import (
    FailedChunk,
    MineToponymMentionsReport,
    chunk_source_body,
    mine_toponym_mentions_tiered,
)


class FakeClient:
    """Replays canned chat_json responses. Pass an Exception in any
    slot to simulate a transport/parse failure on that call.

    Same contract as the single-tier FakeClient — copied here rather
    than imported to keep the test files independent."""

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


# ---------- _three_chunk_body helper -------------------------------------


def _three_chunk_body() -> str:
    """Deterministically 3-chunk body. Mirrors the single-tier test
    helper so the tier-fallback tests can pin chunk counts."""
    p1 = "A" * 7995 + " Edlin"
    p2 = "B" * 7996 + " Wear"
    p3 = "C" * 7996 + " Tyne"
    return f"{p1}\n\n{p2}\n\n{p3}"


def test_three_chunk_body_indeed_chunks_to_three():
    assert len(chunk_source_body(_three_chunk_body(), target_chunk_size=10000)) == 3


# ---------- mine_toponym_mentions_tiered --------------------------------


def test_tiered_primary_only_when_primary_succeeds():
    """If the primary client succeeds on every chunk, the fallback
    client is never invoked. chunks_recovered_by_fallback stays 0."""
    primary = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            {"mentions": []},
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    fallback = FakeClient([])  # nothing canned — would StopIteration if called
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
    )
    assert isinstance(report, MineToponymMentionsReport)
    assert report.chunks_processed == 3
    assert report.chunks_recovered_by_fallback == 0
    assert report.chunks_failed == 0
    assert len(fallback.calls) == 0
    assert [m.form for m in report.mentions] == ["Edlin", "Tyne"]


def test_tiered_fallback_rescues_primary_failure():
    """Primary raises on chunk 1 → fallback retries and succeeds.
    chunks_recovered_by_fallback == 1; chunks_failed == 0."""
    primary = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            RuntimeError("primary parse error"),
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    fallback = FakeClient(
        [
            {"mentions": [{"form": "Wear", "context": "Wear"}]},
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
    )
    assert report.chunks_processed == 3
    assert report.chunks_recovered_by_fallback == 1
    assert report.chunks_failed == 0
    # Both clients called: primary 3 times, fallback once.
    assert len(primary.calls) == 3
    assert len(fallback.calls) == 1
    forms = [m.form for m in report.mentions]
    assert "Edlin" in forms
    assert "Wear" in forms  # rescued by fallback
    assert "Tyne" in forms


def test_tiered_both_tiers_fail_records_failed_chunk():
    """Primary raises AND fallback raises → chunks_failed bumps,
    failed_chunks records BOTH errors (with the fallback's error as
    the more diagnostic message)."""
    primary = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            RuntimeError("primary 503"),
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    fallback = FakeClient(
        [
            RuntimeError("fallback 429: rate limited"),
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
    )
    assert report.chunks_processed == 2
    assert report.chunks_recovered_by_fallback == 0
    assert report.chunks_failed == 1
    assert len(report.failed_chunks) == 1
    fc = report.failed_chunks[0]
    assert fc.index == 1
    # Both errors surfaced in the message — operator sees the full
    # diagnostic.
    assert "primary 503" in fc.error
    assert "429" in fc.error


def test_tiered_sanitizes_surrogate_in_chained_failure_message():
    """wyrd-8wa4 round 3: the tiered both-tiers-failed path constructs
    `FailedChunk(error=f"primary={...}; fallback={...}")`. If either
    fragment carries a lone surrogate from a provider SDK that
    stringified an LLM response excerpt, the persistence writer that
    serializes the FailedChunk JSONL hits the same UnicodeEncodeError
    crash class. The single-tier mirror is pinned by
    `test_mine_toponym_mentions_sanitizes_surrogate_in_failure_message`
    in the extractor test file; this is the tiered-path symmetric
    pin (pr-test-analyzer round 3, criticality 5)."""
    primary = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            # Primary's exception message carries a lone surrogate
            # (e.g. provider SDK echoing an LLM token).
            RuntimeError("primary parse near token: bad\udaaa"),
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    fallback = FakeClient(
        [
            # Fallback ALSO fails with a surrogate, so both fragments
            # exercise the sanitizer at the FailedChunk construction
            # site.
            RuntimeError("fallback 429: limit\udbff hit"),
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
    )
    assert report.chunks_failed == 1
    fc = report.failed_chunks[0]
    # Contract: the persistence writer's json.dumps(..., ensure_ascii=False)
    # + utf-8 file write must not raise. Sanitization at FailedChunk
    # construction is what makes this true.
    json.dumps({"error": fc.error}, ensure_ascii=False).encode("utf-8")
    # Both fragments survive (sanitized), so operators get the full
    # diagnostic chain.
    assert "primary" in fc.error
    assert "fallback" in fc.error
    assert "429" in fc.error
    # Lone surrogates were replaced with U+FFFD.
    assert "\udaaa" not in fc.error
    assert "\udbff" not in fc.error


def test_tiered_rolls_up_surrogates_sanitized_across_chunks():
    """wyrd-8wa4 round 3: the tiered orchestrator has its own
    `report.surrogates_sanitized += chunk_counters.surrogates_sanitized`
    rollup site distinct from the single-tier mirror. Both are plain
    `int +=` operations; both have the same drop-the-line regression
    class. The single-tier rollup is pinned by
    `test_mine_toponym_mentions_rolls_up_surrogates_sanitized_across_chunks`
    in the extractor test file; this is the tiered-path symmetric
    pin (pr-test-analyzer round 3, criticality 5)."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "ctx\udaaa chunk0"},
                ]
            },
            {"mentions": []},  # Clean middle chunk — positive control.
            {
                "mentions": [
                    {"form": "Tyne", "context": "ctx\ud800 chunk2"},
                ]
            },
        ]
    )
    # Fallback never invoked (primary succeeds every chunk).
    fallback = FakeClient([])
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
    )
    # Rollup must SUM across chunks; last-chunk-only regression would
    # land here as 1.
    assert report.surrogates_sanitized == 2
    # Positive control: both mentions admitted (the surrogate was in
    # context, not form).
    assert len(report.mentions) == 2


def test_tiered_on_chunk_failed_fires_per_both_tiers_failure():
    """wyrd-3gbx: the tiered orchestrator must invoke on_chunk_failed
    on every chunk where BOTH tiers failed, passing the FailedChunk
    record with the chained primary=...; fallback=... error. Mirrors
    the single-tier mine_toponym_mentions parity that
    mine-toponym-mentions-staged depends on for failure-streaming.
    The callback does NOT fire on primary-only failures the fallback
    recovers — those are reported via chunks_recovered_by_fallback.
    """
    primary = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            RuntimeError("primary 503"),
            RuntimeError("primary 504"),
        ]
    )
    fallback = FakeClient(
        [
            # First primary failure: fallback recovers (mentions
            # returned) — no on_chunk_failed fire.
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
            # Second primary failure: fallback also fails — fires.
            RuntimeError("fallback 429"),
        ]
    )
    captured: list[FailedChunk] = []
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        on_chunk_failed=captured.append,
    )
    # Sanity: primary-recovered AND both-tiers-failed both occurred.
    assert report.chunks_recovered_by_fallback == 1
    assert report.chunks_failed == 1
    # Callback fired EXACTLY once — on the both-tiers-failed chunk,
    # not on the primary-only chunk the fallback rescued.
    assert len(captured) == 1
    fc = captured[0]
    assert fc.index == 2
    assert "primary 504" in fc.error
    assert "fallback 429" in fc.error
    assert fc.chunk_body  # non-empty — operator can resume from this record


def test_tiered_on_chunk_failed_fires_with_hallucination_threshold_set():
    """wyrd-3gbx: the on_chunk_failed callback fires on the both-tiers-
    failed path regardless of whether hallucination_fallback_threshold
    is set. The two paths are orthogonal — Phase 2 (primary-fail-
    fallback) runs when the primary RAISES; Phase 3 (rescue) runs only
    when the primary SUCCEEDS with hallucinations. A primary that
    raises can never enter Phase 3, so the callback contract holds
    even with the rescue flag enabled. Guards against a regression
    that conditionally suppressed on_chunk_failed when the threshold
    flag is set."""
    primary = FakeClient([RuntimeError("primary boom")])
    fallback = FakeClient([RuntimeError("fallback boom")])
    captured: list[FailedChunk] = []
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        limit=1,
        hallucination_fallback_threshold=2,  # rescue is configured but never reached
        on_chunk_failed=captured.append,
    )
    assert report.chunks_failed == 1
    # Rescue did NOT run (primary never produced mentions).
    assert report.chunks_hallucination_rescue_attempted == 0
    # on_chunk_failed STILL fired on the both-tiers-failed chunk.
    assert len(captured) == 1
    assert "primary boom" in captured[0].error
    assert "fallback boom" in captured[0].error


def test_tiered_on_chunk_failed_none_is_silent():
    """Default on_chunk_failed=None must remain valid (the existing CLI
    callers don't pass the callback). Guard against a regression that
    would NoneType-call the callback unconditionally."""
    primary = FakeClient([RuntimeError("primary boom")])
    fallback = FakeClient([RuntimeError("fallback boom")])
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        limit=1,
        # on_chunk_failed not passed — relies on the default of None.
    )
    assert report.chunks_failed == 1
    # The failure was still recorded in the in-memory buffer (the
    # head/tail path remains the source of truth when on_chunk_failed
    # isn't wired).
    assert len(report.failed_chunks) == 1


def test_tiered_counters_from_successful_tier_only():
    """When the fallback rescues a chunk, the report counters reflect
    the FALLBACK's validation (hallucinations_dropped, years_clamped),
    not the primary's. Today this property is trivially satisfied
    because ``extract_toponym_mentions_from_chunk`` only updates
    counters after the LLM call returns — a raise means counters
    stay at their default zero. The explicit reset in
    ``mine_toponym_mentions_tiered`` is defensive against a future
    refactor that might validate-as-it-parses. Test pins the
    observable contract; the reset's necessity is implementation-
    detail (preserved by the docstring)."""
    primary = FakeClient(
        [
            # Primary fails on this chunk before counters could accumulate.
            RuntimeError("primary fail"),
        ]
    )
    fallback = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "ok"},
                    {"form": "Hallucin", "context": "fake"},  # not in chunk
                    {"form": "Edlin", "date_year": "bad", "context": "clamp"},
                ]
            }
        ]
    )
    # Use a single-chunk body for clarity.
    body = "A" * 7995 + " Edlin"
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        body,
        target_chunk_size=10000,
    )
    assert report.chunks_recovered_by_fallback == 1
    # Counters reflect ONLY the fallback's output.
    assert report.hallucinations_dropped == 1
    assert report.years_clamped == 1


def test_tiered_progress_callback_fires_per_processed_chunk():
    """The callback fires once per successfully-processed chunk
    (whether primary or fallback handled it). Failed chunks do NOT
    fire the callback."""
    primary = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            RuntimeError("primary fail"),  # rescued by fallback
            RuntimeError("primary fail 2"),  # fallback also fails
        ]
    )
    fallback = FakeClient(
        [
            {"mentions": [{"form": "Wear", "context": "Wear"}]},  # rescues chunk 1
            RuntimeError("fallback fail"),  # fails on chunk 2
        ]
    )
    calls: list[tuple[int, int, int]] = []
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        on_chunk_done=lambda d, t, m: calls.append((d, t, m)),
    )
    # 2 processed (chunk 0 primary, chunk 1 fallback), 1 failed (chunk 2).
    assert report.chunks_processed == 2
    assert report.chunks_failed == 1
    assert len(calls) == 2


def test_tiered_limit_applies_after_chunking():
    """``limit`` slices the chunk list before processing — same
    semantics as the single-tier function. The fallback is only
    eligible to be called on chunks within the limit."""
    primary = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
        ]
    )
    fallback = FakeClient([])
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        limit=1,
    )
    assert report.chunks_processed == 1
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0


def test_tiered_log_warning_emits_per_failure():
    """When the primary fails, the log_warning hook fires with a
    'retrying with fallback' message. When BOTH tiers fail, a
    'both tiers failed' message fires."""
    primary = FakeClient(
        [
            RuntimeError("primary 503"),
            RuntimeError("primary 503 again"),
        ]
    )
    fallback = FakeClient(
        [
            {"mentions": []},  # rescues chunk 0
            RuntimeError("fallback 429"),  # both fail on chunk 1
        ]
    )
    # 2-chunk body: small enough that one paragraph each.
    body = ("A" * 7990 + " p1") + "\n\n" + ("B" * 7990 + " p2")
    warnings: list[str] = []
    mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        body,
        target_chunk_size=10000,
        log_warning=warnings.append,
    )
    # Chunk 0: primary failed, fallback rescued → one "retrying" warning.
    # Chunk 1: primary failed, fallback also failed → one "retrying" +
    # one "both tiers failed".
    assert sum(1 for w in warnings if "retrying with fallback" in w) == 2
    assert sum(1 for w in warnings if "both tiers failed" in w) == 1


# ---------- CLI: mine-toponym-mentions-tiered ---------------------------


def _stub_anthropic_for_cli(monkeypatch, fake_client: FakeClient) -> None:
    import wyrd.generators.kenning.extractors.anthropic as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", lambda **kw: fake_client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")


def _stub_ollama_for_cli(monkeypatch, fake_client: FakeClient) -> None:
    import wyrd.generators.kenning.extractors.llm as llm_module

    monkeypatch.setattr(llm_module, "OllamaClient", lambda **kw: fake_client)


def test_cli_walks_all_txt_files_when_no_source_specified(tmp_path, monkeypatch):
    """Without --source, the CLI walks every *.txt under --sources-dir
    and writes one JSONL per source under --output-dir."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham in Northumberland.", encoding="utf-8")
    (sources_dir / "src_b.txt").write_text("Tynemouth on the coast.", encoding="utf-8")
    output_dir = tmp_path / "out"

    primary = FakeClient(
        [
            {"mentions": [{"form": "Edlingham", "context": "x"}]},
            {"mentions": [{"form": "Tynemouth", "context": "x"}]},
        ]
    )
    fallback = FakeClient([])  # untouched
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    # Two per-source output files written.
    assert (output_dir / "src_a.jsonl").exists()
    assert (output_dir / "src_b.jsonl").exists()
    # Output content is JSONL with source_id+form per line.
    line_a = json.loads((output_dir / "src_a.jsonl").read_text().strip())
    assert line_a["source_id"] == "src_a"
    assert line_a["form"] == "Edlingham"


def test_cli_invalid_source_does_not_open_failure_sink(tmp_path):
    """A bad --source must fail source validation BEFORE the --capture-failures
    sink is opened. Otherwise open_failure_sink() leaks the file handle and
    leaves a stray empty failure file (+ mkdir'd parent) for a run that never
    processed anything. Regression for the open-then-resolve ordering: resolve
    raises a ClickException, so the sink must be opened only after it succeeds."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "real.txt").write_text("Edlingham.", encoding="utf-8")
    capture = tmp_path / "failures" / "fail.jsonl"

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--source",
            "does-not-exist",
            "--capture-failures",
            str(capture),
        ],
    )
    assert result.exit_code != 0  # source validation failed
    # The sink was never opened: no stray file and no mkdir'd parent dir.
    assert not capture.exists()
    assert not capture.parent.exists()


def test_cli_skip_existing_resumes_without_reprocessing(tmp_path, monkeypatch):
    """--skip-existing: sources whose output JSONL already exists are
    skipped without invoking the LLM. Idempotent resume contract."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")
    (sources_dir / "src_b.txt").write_text("Tynemouth.", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Pre-existing output for src_a — should be skipped.
    (output_dir / "src_a.jsonl").write_text(
        '{"source_id":"src_a","form":"OldRun"}\n', encoding="utf-8"
    )

    # Only one LLM response prepared — if src_a is processed, this would
    # not be enough for both sources.
    primary = FakeClient([{"mentions": [{"form": "Tynemouth", "context": "x"}]}])
    fallback = FakeClient([])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--skip-existing",
        ],
    )
    assert result.exit_code == 0, result.output
    # src_a's pre-existing output untouched.
    assert (
        (output_dir / "src_a.jsonl").read_text().startswith('{"source_id":"src_a","form":"OldRun"}')
    )
    # src_b processed; the new output exists with the fresh content.
    assert (output_dir / "src_b.jsonl").exists()
    line_b = json.loads((output_dir / "src_b.jsonl").read_text().strip())
    assert line_b["form"] == "Tynemouth"
    # src_a was skipped — primary was called exactly once (for src_b).
    assert len(primary.calls) == 1


def test_cli_refuses_to_overwrite_without_force_or_skip(tmp_path, monkeypatch):
    """Without --skip-existing or --force, an existing per-source
    output file is a hard error — prevents silent destruction of
    a multi-hour prior run."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "src_a.jsonl").write_text("PREVIOUS\n", encoding="utf-8")

    _stub_ollama_for_cli(monkeypatch, FakeClient([{"mentions": []}]))
    _stub_anthropic_for_cli(monkeypatch, FakeClient([]))

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code != 0
    assert "exists" in result.output
    # Pre-existing file unchanged.
    assert (output_dir / "src_a.jsonl").read_text() == "PREVIOUS\n"


def test_cli_skip_existing_and_force_are_mutually_exclusive(tmp_path, monkeypatch):
    """The two flags are contradictory — passing both is an error
    (prevents 'I meant to overwrite' / 'I meant to skip' confusion)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("x", encoding="utf-8")

    _stub_ollama_for_cli(monkeypatch, FakeClient([{"mentions": []}]))
    _stub_anthropic_for_cli(monkeypatch, FakeClient([]))

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-existing",
            "--force",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_cli_force_overwrites_existing(tmp_path, monkeypatch):
    """--force replaces existing output. The atomic-rename write
    ensures the previous file isn't truncated mid-write on interrupt."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "src_a.jsonl").write_text("PREVIOUS\n", encoding="utf-8")

    primary = FakeClient([{"mentions": [{"form": "Edlingham", "context": "x"}]}])
    fallback = FakeClient([])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    # Pre-existing content gone; new content in place.
    new_content = (output_dir / "src_a.jsonl").read_text()
    assert "PREVIOUS" not in new_content
    assert "Edlingham" in new_content


def test_cli_specific_sources_via_source_flag(tmp_path, monkeypatch):
    """--source filters the walk to named source_ids only. Sources
    not listed are not touched (no output written for them)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")
    (sources_dir / "src_b.txt").write_text("Tynemouth.", encoding="utf-8")
    (sources_dir / "src_c.txt").write_text("Wearmouth.", encoding="utf-8")
    output_dir = tmp_path / "out"

    primary = FakeClient(
        [
            {"mentions": [{"form": "Edlingham", "context": "x"}]},
            {"mentions": [{"form": "Tynemouth", "context": "x"}]},
        ]
    )
    fallback = FakeClient([])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--source",
            "src_a",
            "--source",
            "src_b",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / "src_a.jsonl").exists()
    assert (output_dir / "src_b.jsonl").exists()
    # src_c was never touched — no output written.
    assert not (output_dir / "src_c.jsonl").exists()


def test_cli_unknown_source_errors_clearly(tmp_path, monkeypatch):
    """--source <name> where <name>.txt doesn't exist is a hard error
    BEFORE any LLM client is invoked."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("x", encoding="utf-8")

    _stub_ollama_for_cli(monkeypatch, FakeClient([{"mentions": []}]))
    _stub_anthropic_for_cli(monkeypatch, FakeClient([]))

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--source",
            "nonexistent",
        ],
    )
    assert result.exit_code != 0
    assert "source body not found" in result.output


def test_cli_path_traversal_is_neutralized(tmp_path, monkeypatch):
    """--source "../planted" should be neutralized via Path(...).name
    to "planted". Both files (one outside sources_dir with secret
    content, one inside with safe content named "planted.txt") are
    planted: the test verifies the INSIDE file is read (proving the
    traversal collapsed to a bare filename, not just failing because
    the outside file didn't exist)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    # File OUTSIDE sources_dir — must not be reachable.
    planted_outside = tmp_path / "planted.txt"
    planted_outside.write_text("SECRET PLAINTEXT", encoding="utf-8")
    # File INSIDE sources_dir with the same basename — Path("../planted").name
    # is "planted", so the .txt suffix appended lands on this one.
    planted_inside = sources_dir / "planted.txt"
    planted_inside.write_text("INSIDE PROSE Edlingham mentioned.", encoding="utf-8")

    primary_calls: list[tuple[str, str, dict | None]] = []

    class _CapturingClient:
        model = "fake"

        def chat_json(self, system: str, user: str, schema: dict | None = None) -> dict:
            primary_calls.append((system, user, schema))
            return {"mentions": []}

    _stub_ollama_for_cli(monkeypatch, _CapturingClient())
    _stub_anthropic_for_cli(monkeypatch, FakeClient([]))

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--source",
            "../planted",
        ],
    )
    # The CLI succeeded — Path(...).name reduced "../planted" to
    # "planted" and read sources_dir/planted.txt (the INSIDE file).
    assert result.exit_code == 0, result.output
    # The chunk passed to the LLM is the INSIDE file's content, NOT
    # the outside one. If traversal had succeeded, "SECRET PLAINTEXT"
    # would have reached the prompt.
    assert len(primary_calls) == 1
    user_arg = primary_calls[0][1]
    assert "INSIDE PROSE" in user_arg
    assert "SECRET PLAINTEXT" not in user_arg


def test_cli_path_traversal_missing_basename_still_errors(tmp_path, monkeypatch):
    """When --source is a path-traversal expression AND no file with
    the stripped basename exists in --sources-dir, error clearly
    (rather than reading some other file by accident)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    # File outside sources_dir with secret content; NO file at
    # sources_dir/planted.txt.
    planted = tmp_path / "planted.txt"
    planted.write_text("SECRET PLAINTEXT", encoding="utf-8")

    _stub_ollama_for_cli(monkeypatch, FakeClient([{"mentions": []}]))
    _stub_anthropic_for_cli(monkeypatch, FakeClient([]))

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--source",
            "../planted",
        ],
    )
    # Path(...).name reduces "../planted" to "planted"; "planted.txt"
    # doesn't exist under sources_dir → error. The OUTSIDE file is
    # never touched.
    assert result.exit_code != 0
    assert "source body not found" in result.output


def test_cli_empty_sources_dir_errors_clearly(tmp_path, monkeypatch):
    """If --sources-dir contains no *.txt files (and no --source given),
    error rather than silently doing nothing."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    output_dir = tmp_path / "out"

    _stub_ollama_for_cli(monkeypatch, FakeClient([]))
    _stub_anthropic_for_cli(monkeypatch, FakeClient([]))

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code != 0
    assert "no sources" in result.output


def test_cli_atomic_write_no_tmp_leftover(tmp_path, monkeypatch):
    """The per-source output uses temp + rename. After a clean run
    the .tmp sibling must not linger."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")
    output_dir = tmp_path / "out"

    primary = FakeClient([{"mentions": [{"form": "Edlingham", "context": "x"}]}])
    fallback = FakeClient([])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / "src_a.jsonl").exists()
    assert not (output_dir / "src_a.jsonl.tmp").exists()


def test_cli_summary_reports_recovery_count(tmp_path, monkeypatch):
    """The TOTAL summary line surfaces recovered_by_fallback so the
    operator can see how often the primary was insufficient — useful
    signal for deciding whether to swap providers entirely."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")
    output_dir = tmp_path / "out"

    # Primary fails; fallback rescues.
    primary = FakeClient([RuntimeError("primary fail")])
    fallback = FakeClient([{"mentions": [{"form": "Edlingham", "context": "x"}]}])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "recovered_by_fallback=1" in result.output


def test_cli_capture_failures_streams_both_tiers_failed_chunks(tmp_path, monkeypatch):
    """wyrd-3gbx: --capture-failures PATH must stream a JSONL record
    for every chunk where BOTH tiers failed, mirroring the single-tier
    mine-toponym-mentions failure-streaming. Each record carries
    source_id, chunk_index, chunk_body, and the chained error string."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham in Northumberland.", encoding="utf-8")

    # One chunk; primary fails, fallback fails — both-tiers-failure path.
    primary = FakeClient([RuntimeError("primary 503")])
    fallback = FakeClient([RuntimeError("fallback 429")])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    capture_path = tmp_path / "failures.jsonl"
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--capture-failures",
            str(capture_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # Failure JSONL has exactly one record for the failed chunk.
    lines = capture_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["source_id"] == "src_a"
    assert record["chunk_index"] == 0
    assert record["chunk_body"]
    assert "primary 503" in record["error"]
    assert "fallback 429" in record["error"]
    # Summary line surfaces the streamed count.
    assert "1 failed chunk(s) appended" in result.output


def test_cli_capture_failures_unchanged_when_no_failures(tmp_path, monkeypatch):
    """wyrd-3gbx: when --capture-failures is given but no chunk hits
    the both-tiers-failed path, the CLI still emits an explicit
    'unchanged this run' status message (matches the staged-cascade
    CLI's observability shape so operators can distinguish 'no
    failures' from 'flag ignored')."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham in Northumberland.", encoding="utf-8")

    primary = FakeClient([{"mentions": [{"form": "Edlingham", "context": "x"}]}])
    fallback = FakeClient([])  # untouched
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    capture_path = tmp_path / "failures.jsonl"
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--capture-failures",
            str(capture_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "0 new failures" in result.output
    # The capture file exists (we opened it for append) but is empty.
    assert capture_path.exists()
    assert capture_path.read_text(encoding="utf-8") == ""


def test_cli_capture_failures_multi_source_stamps_correct_source_id(tmp_path, monkeypatch):
    """wyrd-3gbx: when multiple sources are mined and each emits a
    both-tiers-failure, every captured record must carry that
    source's id — NOT the last loop iteration's. Pins the B023
    closure-over-loop-var defense via the ``_sid: str = source_id``
    default-arg trick at the on_fail definition. A regression that
    dropped the default-arg (or hoisted on_fail above the loop)
    would silently emit every record with the final source_id seen.
    """
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham in Northumberland.", encoding="utf-8")
    (sources_dir / "src_b.txt").write_text("Tynemouth on the coast.", encoding="utf-8")

    # Both sources have one chunk; primary fails twice, fallback fails twice.
    primary = FakeClient(
        [RuntimeError("primary 503 on src_a"), RuntimeError("primary 503 on src_b")]
    )
    fallback = FakeClient(
        [RuntimeError("fallback 429 on src_a"), RuntimeError("fallback 429 on src_b")]
    )
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    capture_path = tmp_path / "failures.jsonl"
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--capture-failures",
            str(capture_path),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = capture_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    by_source = {json.loads(ln)["source_id"]: json.loads(ln) for ln in lines}
    # Each record stamped with the CORRECT source_id, not the last one
    # seen in the loop.
    assert set(by_source) == {"src_a", "src_b"}
    # Sanity: error strings also paired correctly with the right source.
    assert "src_a" in by_source["src_a"]["error"]
    assert "src_b" in by_source["src_b"]["error"]


def test_cli_capture_failures_appends_new_records_to_existing_file(tmp_path, monkeypatch):
    """wyrd-3gbx: when --capture-failures targets a pre-existing file
    AND the run produces new failures, both the old records and the
    new ones must end up in the file (append, not overwrite). A
    regression that flipped open('a') to open('w') would still pass
    the warning-message test alone — this test pins the actual
    coexistence."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham in Northumberland.", encoding="utf-8")

    capture_path = tmp_path / "failures.jsonl"
    capture_path.write_text(
        '{"source_id": "src_old", "chunk_index": 0, "chunk_body": "old", "error": "x"}\n'
    )

    primary = FakeClient([RuntimeError("primary 503")])
    fallback = FakeClient([RuntimeError("fallback 429")])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--capture-failures",
            str(capture_path),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = capture_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    sources = [json.loads(ln)["source_id"] for ln in lines]
    # Old record preserved at the top; new record appended below.
    assert sources == ["src_old", "src_a"]


def test_cli_capture_failures_warns_on_existing_non_empty_file(tmp_path, monkeypatch):
    """wyrd-3gbx: append-mode is the chosen idempotent-resume shape
    (matches the staged-cascade CLI). Warn loudly if the file already
    has records so an operator-blind append doesn't silently bury old
    state. The warning surfaces the existing record count so the
    operator can decide whether to `> failures.jsonl` to clear."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham in Northumberland.", encoding="utf-8")

    capture_path = tmp_path / "failures.jsonl"
    capture_path.write_text('{"source_id": "old", "chunk_index": 0, "chunk_body": "x"}\n')

    primary = FakeClient([{"mentions": [{"form": "Edlingham", "context": "x"}]}])
    fallback = FakeClient([])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--capture-failures",
            str(capture_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "already has 1 record(s)" in result.output
    # Pre-existing record is preserved.
    lines = capture_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["source_id"] == "old"


# ---------- round-2 additions: CLI flag pass-through coverage -----------


def test_cli_primary_model_flows_to_client_factory(tmp_path, monkeypatch):
    """--primary-model must reach the primary client factory's
    `model=` kwarg. A regression that dropped the kwarg would land
    the default model and never be caught by the existing tests
    (which all use `lambda **kw: fake_client` and discard kwargs)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")

    captured: list[dict] = []
    fake = FakeClient([{"mentions": []}])

    def recording_factory(**kw):
        captured.append(kw)
        return fake

    import wyrd.generators.kenning.extractors.llm as llm_module

    monkeypatch.setattr(llm_module, "OllamaClient", recording_factory)
    monkeypatch.setattr(
        "wyrd.generators.kenning.extractors.anthropic.AnthropicClient",
        lambda **kw: FakeClient([]),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--primary-model",
            "qwen-test-override",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0]["model"] == "qwen-test-override"


def test_cli_fallback_model_flows_to_client_factory(tmp_path, monkeypatch):
    """--fallback-model reaches the fallback factory. Same shape as
    primary-model test but for the second-tier client."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")

    captured: list[dict] = []
    fake = FakeClient([{"mentions": []}])

    def recording_factory(**kw):
        captured.append(kw)
        return fake

    import wyrd.generators.kenning.extractors.anthropic as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", recording_factory)
    monkeypatch.setattr(
        "wyrd.generators.kenning.extractors.llm.OllamaClient",
        lambda **kw: FakeClient([{"mentions": []}]),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--fallback-model",
            "claude-test-override-v9",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0]["model"] == "claude-test-override-v9"


def test_cli_chunk_size_flows_through(tmp_path, monkeypatch):
    """--chunk-size must affect the number of chunks the extractor
    sees. Without explicit coverage, a regression that ignored the
    flag and used the default 20000 would silently pass every
    existing test (all of them implicitly rely on the default)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    # Two paragraphs that pack into 1 chunk at default 20000 chars but
    # need 2 chunks at --chunk-size 50.
    body = "First paragraph here.\n\nSecond paragraph follows."
    (sources_dir / "src_a.txt").write_text(body, encoding="utf-8")

    primary = FakeClient([{"mentions": []}, {"mentions": []}])
    fallback = FakeClient([])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            # target_chunk_size=25 forces flush after the first
            # paragraph (21 chars + 2 + 25 = 48 > 25). Default 20000
            # would pack both paragraphs into one chunk.
            "--chunk-size",
            "25",
        ],
    )
    assert result.exit_code == 0, result.output
    # Primary called twice — once per chunk. If --chunk-size were
    # ignored, the body would pack into 1 chunk and the primary
    # would be called once.
    assert len(primary.calls) == 2


def test_cli_limit_flows_through(tmp_path, monkeypatch):
    """--limit caps the number of chunks processed per source. A
    regression that dropped the flag would process all chunks
    (the existing tests don't pin this — all use defaults)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    body = "A" * 7990 + " Edlin\n\n" + "B" * 7990 + " Wear\n\n" + "C" * 7990 + " Tyne"
    (sources_dir / "src_a.txt").write_text(body, encoding="utf-8")

    primary = FakeClient([{"mentions": []}])  # only 1 prepared
    fallback = FakeClient([])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--chunk-size",
            "10000",  # 3 chunks total at this size
            "--limit",
            "1",  # process only 1
        ],
    )
    assert result.exit_code == 0, result.output
    # Primary called ONCE — limit=1 cap honored.
    assert len(primary.calls) == 1


def test_cli_gemini_primary_provider_routes_to_gemini_client(tmp_path, monkeypatch):
    """--primary-provider gemini must construct a GeminiClient and
    use it. Without this test, a regression in _build_extractor_client
    that mis-dispatched 'gemini' to AnthropicClient would silently
    pass the ollama+anthropic tests."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")

    gemini_fake = FakeClient([{"mentions": []}])
    import wyrd.generators.kenning.extractors.gemini as gemini_module

    monkeypatch.setattr(gemini_module, "GeminiClient", lambda **kw: gemini_fake)
    monkeypatch.setattr(
        "wyrd.generators.kenning.extractors.anthropic.AnthropicClient",
        lambda **kw: FakeClient([]),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--primary-provider",
            "gemini",
        ],
    )
    assert result.exit_code == 0, result.output
    # GeminiClient was called once (one chunk).
    assert len(gemini_fake.calls) == 1


def test_cli_skip_existing_defers_client_construction(tmp_path, monkeypatch):
    """Client construction must be deferred past --skip-existing. If
    every source is skipped, neither client factory should be
    invoked (no ANTHROPIC_API_KEY validation, no Ollama-URL probe).
    Enables 'dry-run resume verification' without needing API keys."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Pre-existing output → will be skipped under --skip-existing.
    (output_dir / "src_a.jsonl").write_text('{"source_id":"src_a","form":"x"}\n', encoding="utf-8")

    ollama_built = {"n": 0}
    anthropic_built = {"n": 0}

    def ollama_factory(**kw):
        ollama_built["n"] += 1
        return FakeClient([])

    def anthropic_factory(**kw):
        anthropic_built["n"] += 1
        return FakeClient([])

    monkeypatch.setattr("wyrd.generators.kenning.extractors.llm.OllamaClient", ollama_factory)
    monkeypatch.setattr(
        "wyrd.generators.kenning.extractors.anthropic.AnthropicClient",
        anthropic_factory,
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--skip-existing",
        ],
    )
    assert result.exit_code == 0, result.output
    # No client factory was called — every source skipped, deferred
    # construction never triggered.
    assert ollama_built["n"] == 0
    assert anthropic_built["n"] == 0


def test_tiered_propagates_non_runtime_exception_to_fallback():
    """Non-RuntimeError exceptions from the primary (TimeoutError,
    OSError — provider-SDK transport exception classes are similar)
    must trigger the fallback rather than abort the source. Round-1
    silent-failure finding: bare 'except Exception' is intentional
    precisely because anthropic.APIError, httpx.HTTPError, etc. don't
    derive from RuntimeError. ValueError is excluded from this list
    as of wyrd-pe4g round 3 — see the test below pinning that
    behavior."""
    primary = FakeClient(
        [
            TimeoutError("primary timeout"),  # NOT a RuntimeError
        ]
    )
    fallback = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
        ]
    )
    body = "A" * 7995 + " Edlin"
    from wyrd.generators.kenning.extractors.toponym_mentions import (
        mine_toponym_mentions_tiered,
    )

    report = mine_toponym_mentions_tiered(
        primary, fallback, "test_source", body, target_chunk_size=10000
    )
    # Fallback rescued the chunk; chunks_failed stayed 0.
    assert report.chunks_recovered_by_fallback == 1
    assert report.chunks_failed == 0
    assert [m.form for m in report.mentions] == ["Edlin"]


def test_tiered_propagates_unknown_dialect_value_error_does_not_mask_via_fallback():
    """wyrd-pe4g round-4 pr-test-analyzer + silent-failure-hunter:
    a ValueError from the dispatch-time schema_dialect guard on the
    PRIMARY client must propagate as a configuration error rather
    than being silently rescued by the fallback. The whole point of
    the round-2 guard is that a typo'd schema_dialect on the primary
    is a misconfiguration the operator needs to see — masking it via
    a healthy fallback would defeat the design entirely (the run
    "succeeds" via 100% fallback recovery while the primary stays
    broken forever)."""

    class TypoPrimary:
        schema_dialect = "Openapi"  # capitalization typo

        def chat_json(self, system, user, response_schema=None):
            raise AssertionError("guard should fire before chat_json is reached")

    healthy_fallback = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
        ]
    )
    body = "A" * 7995 + " Edlin"
    import pytest

    from wyrd.generators.kenning.extractors.toponym_mentions import (
        mine_toponym_mentions_tiered,
    )

    with pytest.raises(ValueError, match=r"unknown schema_dialect 'Openapi'"):
        mine_toponym_mentions_tiered(
            TypoPrimary(), healthy_fallback, "test_source", body, target_chunk_size=10000
        )


def test_tiered_both_tiers_non_runtime_exception_failed_chunk_recorded():
    """When both tiers raise NON-RuntimeError exceptions, the chunk
    is still recorded as failed (with both error types) — not
    propagated as an unhandled exception."""
    primary = FakeClient([TimeoutError("primary timeout")])
    fallback = FakeClient([OSError("disk full")])
    body = "A" * 7995 + " Edlin"
    from wyrd.generators.kenning.extractors.toponym_mentions import (
        mine_toponym_mentions_tiered,
    )

    report = mine_toponym_mentions_tiered(
        primary, fallback, "test_source", body, target_chunk_size=10000
    )
    assert report.chunks_failed == 1
    assert len(report.failed_chunks) == 1
    fc = report.failed_chunks[0]
    assert "TimeoutError" in fc.error
    assert "OSError" in fc.error


def test_cli_ollama_url_flows_to_ollama_client(tmp_path, monkeypatch):
    """--ollama-url overrides the OllamaClient base URL (matches the
    `mine-llm` flag pattern — Gemini round-3 consistency finding)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham.", encoding="utf-8")

    captured: list[dict] = []
    fake = FakeClient([{"mentions": []}])

    def recording_factory(**kw):
        captured.append(kw)
        return fake

    import wyrd.generators.kenning.extractors.llm as llm_module

    monkeypatch.setattr(llm_module, "OllamaClient", recording_factory)
    monkeypatch.setattr(
        "wyrd.generators.kenning.extractors.anthropic.AnthropicClient",
        lambda **kw: FakeClient([]),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--ollama-url",
            "http://hades:11434",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0]["base_url"] == "http://hades:11434"


def test_tiered_reraises_programmer_errors_from_primary():
    """AttributeError / TypeError / NameError / ImportError from the
    primary client are NOT silently logged as per-chunk failures —
    they propagate as tracebacks so a real bug (typo, missing
    method) doesn't masquerade as a 100%-failed extraction run.

    Without this re-raise the bare `except Exception` would catch
    these alongside legitimate SDK errors. Combined-agent round-2
    HIGH finding."""
    import pytest

    class _BrokenClient:
        model = "broken"

        def chat_json(self, system, user, schema=None):
            raise AttributeError("missing method (simulated programmer bug)")

    body = "A" * 7995 + " Edlin"
    from wyrd.generators.kenning.extractors.toponym_mentions import (
        mine_toponym_mentions_tiered,
    )

    # AttributeError must propagate, NOT be swallowed into chunks_failed.
    with pytest.raises(AttributeError, match="missing method"):
        mine_toponym_mentions_tiered(
            _BrokenClient(),
            FakeClient([{"mentions": []}]),
            "test_source",
            body,
            target_chunk_size=10000,
        )


def test_tiered_reraises_programmer_errors_from_fallback():
    """Same as above but the bug lives in the FALLBACK client. When
    the primary fails with a transport error and the fallback then
    raises a programmer error, the programmer error wins — operator
    sees the traceback."""
    import pytest

    class _BrokenFallback:
        model = "broken"

        def chat_json(self, system, user, schema=None):
            raise TypeError("bad argument (simulated programmer bug)")

    primary = FakeClient([RuntimeError("primary transport fail")])
    body = "A" * 7995 + " Edlin"
    from wyrd.generators.kenning.extractors.toponym_mentions import (
        mine_toponym_mentions_tiered,
    )

    with pytest.raises(TypeError, match="bad argument"):
        mine_toponym_mentions_tiered(
            primary,
            _BrokenFallback(),
            "test_source",
            body,
            target_chunk_size=10000,
        )


def test_single_tier_reraises_programmer_errors():
    """Mirror of the tiered re-raise contract for the single-tier
    function — exact same widened-exception-with-whitelist policy."""
    import pytest

    class _BrokenClient:
        model = "broken"

        def chat_json(self, system, user, schema=None):
            raise NameError("undefined variable (simulated programmer bug)")

    body = "A" * 7995 + " Edlin"
    from wyrd.generators.kenning.extractors.toponym_mentions import (
        mine_toponym_mentions,
    )

    with pytest.raises(NameError):
        mine_toponym_mentions(
            _BrokenClient(),
            "test_source",
            body,
            target_chunk_size=10000,
        )


def test_cli_realistic_mention_shape_preserves_all_fields(tmp_path, monkeypatch):
    """A regression that dropped date_year or region_hint from the
    JSONL output would not be caught by the existing tests (their
    FakeClient responses omit those fields, so the synthesized
    fixture context never has them either). This test feeds a
    realistic shape and verifies all fields make it to the file."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text(
        "Edlingham in Northumberland was attested in 1086.",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    primary = FakeClient(
        [
            {
                "mentions": [
                    {
                        "form": "Edlingham",
                        "date_year": 1086,
                        "region_hint": "Northumberland",
                        "context": "Edlingham in Northumberland was attested in 1086.",
                    }
                ]
            }
        ]
    )
    fallback = FakeClient([])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    line = json.loads((output_dir / "src_a.jsonl").read_text().strip())
    assert line["source_id"] == "src_a"
    assert line["form"] == "Edlingham"
    assert line["date_year"] == 1086
    assert line["region_hint"] == "Northumberland"
    assert "Edlingham in Northumberland" in line["context"]


# ---------- hallucination-triggered rescue (wyrd-z8mq) ------------------
#
# Tests for ``hallucination_fallback_threshold``: when the primary
# succeeds but emits ``hallucinations_dropped >= threshold`` forms, the
# fallback ALSO runs on the same chunk and union-merges mentions by
# (form, date_year, region_hint, context). Designed for gemma4 primary
# + Anthropic fallback so the small model keeps its good mentions while
# the large model catches what it fabricated. Each FakeClient response
# below uses chunk 1's "Edlin" / chunk 2's "Wear" / chunk 3's "Tyne"
# as real forms; other strings (e.g. "Faux") trip the word-boundary
# guard since they don't appear in the chunk body.
#
# Counter contract (round-1 silent-failure-hunter HIGH): the rescue
# tracks attempted + succeeded separately. _attempted bumps when the
# threshold trips (regardless of fallback success); _succeeded bumps
# only when the fallback returns mentions. A gap (succeeded < attempted)
# is the broken-fallback signal (bad API key, network outage) that the
# CLI surfaces as `halluc_rescued=succeeded/attempted` in summary lines.


def test_tiered_hallucination_rescue_triggers_and_dedupes_on_byte_identical():
    """Primary emits a good mention + 2 hallucinated forms; fallback
    emits the SAME good mention with byte-identical context. Rescue
    fires, byte-identical dedup collapses the two Edlin emissions
    (one survives, not two).

    Pins: (a) the attempted+succeeded counters increment, (b)
    byte-identical (form, date_year, region_hint, context) collisions
    drop the fallback's copy. The 'primary wins' canonical-winner
    semantics is structurally true (the code inserts primary's
    mentions before applying the filter to fallback's), but with
    byte-identical attributes the surviving mention is observationally
    indistinguishable from either source; the test doesn't try to
    pin which physical instance survived — test-coverage R1 LOW
    noted this and the resolution is to NOT overclaim. See
    test_tiered_hallucination_rescue_preserves_distinct_contexts
    for the distinct-context case where both mentions survive."""
    primary = FakeClient(
        [
            # chunk 1: 1 real ("Edlin") + 2 fake ("Faux1", "Faux2")
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin shared context"},
                    {"form": "Faux1", "context": "Faux1 context"},
                    {"form": "Faux2", "context": "Faux2 context"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient(
        [
            # Byte-identical context — same form/year/region/context
            # → primary wins on collision, fallback's copy dropped.
            {"mentions": [{"form": "Edlin", "context": "Edlin shared context"}]},
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_processed == 3
    assert report.chunks_hallucination_rescue_attempted == 1
    assert report.chunks_hallucination_rescue_succeeded == 1
    assert report.chunks_recovered_by_fallback == 0
    assert report.hallucinations_dropped == 2
    assert len(fallback.calls) == 1
    # One Edlin survives — both emissions had identical dedup keys.
    assert [m.form for m in report.mentions] == ["Edlin"]
    assert report.mentions[0].context == "Edlin shared context"


def test_tiered_hallucination_rescue_preserves_distinct_contexts():
    """Context-inclusive dedup (silent-failure-hunter R1 MEDIUM): if
    primary's Edlin has context A and fallback's Edlin has context B,
    the two are DIFFERENT attestations and both must be retained.
    Without context in the dedup key the fallback's distinct
    attestation would be silently dropped — real information loss
    since context is downstream-serialized."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin sentence one"},
                    {"form": "Faux1", "context": "Faux1"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin sentence two"}]},
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_hallucination_rescue_attempted == 1
    assert report.chunks_hallucination_rescue_succeeded == 1
    # Both Edlin mentions retained — different contexts = different
    # dedup keys.
    contexts = sorted(m.context for m in report.mentions)
    assert contexts == ["Edlin sentence one", "Edlin sentence two"]


def test_tiered_hallucination_rescue_at_exact_threshold_boundary():
    """threshold=1, hallucinations=1: the >= comparison must fire
    at the boundary (test-coverage R1 LOW). A regression to > would
    silently break the dominant production config (gemma4 primary
    with threshold=1). Fallback emits a non-empty rescue so
    _succeeded increments under the round-2 contract (otherwise the
    test would conflate threshold-boundary with rescue-content-
    delivery)."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1", "context": "Faux1"},  # 1 fake → boundary
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient([{"mentions": [{"form": "Edlin", "context": "Edlin"}]}])
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_hallucination_rescue_attempted == 1
    assert report.chunks_hallucination_rescue_succeeded == 1
    assert len(fallback.calls) == 1


def test_tiered_hallucination_rescue_below_threshold_no_fallback():
    """Primary emits 1 hallucinated form, but threshold=2 — rescue
    does NOT fire. Counters stay 0, fallback isn't called."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin context"},
                    {"form": "Faux1", "context": "Faux1 context"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient([])
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=2,
    )
    assert report.chunks_hallucination_rescue_attempted == 0
    assert report.chunks_hallucination_rescue_succeeded == 0
    assert report.hallucinations_dropped == 1
    assert len(fallback.calls) == 0


def test_tiered_hallucination_rescue_disabled_by_default():
    """Without the threshold (None), primary's hallucinations don't
    trigger the rescue — existing two-tier semantics are preserved."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin context"},
                    {"form": "Faux1", "context": "Faux1 context"},
                    {"form": "Faux2", "context": "Faux2 context"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient([])
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
    )
    assert report.chunks_hallucination_rescue_attempted == 0
    assert report.chunks_hallucination_rescue_succeeded == 0
    assert report.hallucinations_dropped == 2
    assert len(fallback.calls) == 0


def test_tiered_hallucination_rescue_disabled_when_zero():
    """threshold=0 explicitly-disabled. Fallback NOT called even with
    5 hallucinations."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1", "context": "Faux1"},
                    {"form": "Faux2", "context": "Faux2"},
                    {"form": "Faux3", "context": "Faux3"},
                    {"form": "Faux4", "context": "Faux4"},
                    {"form": "Faux5", "context": "Faux5"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient([])
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=0,
    )
    assert report.chunks_hallucination_rescue_attempted == 0
    assert report.chunks_hallucination_rescue_succeeded == 0
    assert len(fallback.calls) == 0


def test_tiered_hallucination_rescue_union_adds_novel_mentions():
    """Primary's chunk 2 hits threshold; fallback emits a NEW form
    primary didn't see (different from primary's dedup keys). Union
    merge keeps both."""
    primary = FakeClient(
        [
            {"mentions": []},
            {
                "mentions": [
                    {"form": "Wear", "context": "Wear context", "date_year": 1086},
                    {"form": "FakeForm", "context": "FakeForm", "date_year": 1086},
                ]
            },
            {"mentions": []},
        ]
    )
    fallback = FakeClient(
        [
            {"mentions": [{"form": "Wear", "context": "Wear alt", "date_year": 1200}]},
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_hallucination_rescue_attempted == 1
    assert report.chunks_hallucination_rescue_succeeded == 1
    forms_years = [(m.form, m.date_year) for m in report.mentions]
    assert forms_years == [("Wear", 1086), ("Wear", 1200)]


def test_tiered_hallucination_rescue_continues_on_fallback_error():
    """Rescue is opportunistic — if the fallback ERRORS on the rescue
    pass, primary's good mentions flow through (no chunks_failed bump).
    _attempted increments since the rescue WAS attempted; _succeeded
    does NOT, exposing the broken-fallback case in the summary line
    (silent-failure-hunter R1 HIGH)."""

    class _FailingFallback:
        model = "fail"

        def chat_json(self, system, user, schema=None):
            raise RuntimeError("simulated fallback transport failure")

    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1", "context": "Faux1"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        _FailingFallback(),
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_hallucination_rescue_attempted == 1
    assert report.chunks_hallucination_rescue_succeeded == 0
    assert report.chunks_failed == 0  # primary's mentions still good
    assert report.chunks_processed == 3
    assert [m.form for m in report.mentions] == ["Edlin"]
    # wyrd-oaq5: the errored-rescue path returns a zero
    # ValidationCounters delta so the orchestrator's ``+=`` fold is a
    # no-op. Pin the operator-observable consequence: primary's
    # hallucination count is recorded once (Faux1 dropped on chunk 1),
    # NOT corrupted by the errored fallback. Catches (a) a buggy
    # __iadd__ that mutates from a non-zero operand, (b) a future
    # refactor that lets the errored-rescue path leak partial
    # counters, (c) orchestrator folding from the wrong accumulator.
    assert report.hallucinations_dropped == 1
    assert report.years_clamped == 0


def test_tiered_hallucination_rescue_counters_attempted_vs_succeeded_diverge():
    """Multi-chunk run with mixed fallback outcomes (one succeeds,
    one fails). Pins that the two counters can legitimately diverge —
    the operationally critical observability case the dual-counter
    design is meant to surface (silent-failure-hunter R1 HIGH)."""

    class _MixedFallback:
        """Returns mentions on first call, errors on second."""

        model = "mixed"

        def __init__(self):
            self.calls = 0

        def chat_json(self, system, user, schema=None):
            self.calls += 1
            if self.calls == 1:
                return {"mentions": [{"form": "Edlin", "context": "rescue context"}]}
            raise RuntimeError("simulated mid-run transient")

    primary = FakeClient(
        [
            # chunk 1: triggers rescue, fallback succeeds
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1a", "context": "Faux1a"},
                ]
            },
            # chunk 2: triggers rescue, fallback fails
            {
                "mentions": [
                    {"form": "Wear", "context": "Wear"},
                    {"form": "Faux2a", "context": "Faux2a"},
                ]
            },
            {"mentions": []},
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        _MixedFallback(),
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_hallucination_rescue_attempted == 2
    assert report.chunks_hallucination_rescue_succeeded == 1
    assert report.chunks_failed == 0
    # Primary's good mentions retained from BOTH chunks regardless
    # of fallback outcome on either. Chunk-1 fallback added a
    # distinct-context Edlin (kept under context-inclusive dedup);
    # chunk-2 fallback errored so no contribution.
    forms = sorted(m.form for m in report.mentions)
    assert forms == ["Edlin", "Edlin", "Wear"]


def test_tiered_hallucination_rescue_aggregates_counters_across_chunks():
    """Counters aggregate across multiple rescued chunks. Both
    counters track the same chunks (attempted=succeeded since both
    fallback calls return mentions); hallucinations_dropped sums
    primary + fallback contributions."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1a", "context": "Faux1a"},
                    {"form": "Faux1b", "context": "Faux1b"},
                ]
            },
            {
                "mentions": [
                    {"form": "Wear", "context": "Wear"},
                    {"form": "Faux2a", "context": "Faux2a"},
                ]
            },
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},  # no rescue
        ]
    )
    fallback = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin alt"}]},
            {
                "mentions": [
                    {"form": "Wear", "context": "Wear alt"},
                    {"form": "Faux2b", "context": "Faux2b"},
                ]
            },
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_hallucination_rescue_attempted == 2
    assert report.chunks_hallucination_rescue_succeeded == 2
    # primary chunk 1 (2) + primary chunk 2 (1) + fallback chunk 2 (1) = 4
    assert report.hallucinations_dropped == 4
    assert report.chunks_processed == 3
    assert report.chunks_failed == 0
    # Context-inclusive dedup keeps the fallback's distinct-context
    # Edlin and Wear too — both contribute novel attestations.
    forms = sorted(m.form for m in report.mentions)
    assert forms == ["Edlin", "Edlin", "Tyne", "Wear", "Wear"]


def test_tiered_hallucination_rescue_all_chunks_triggered():
    """Every chunk trips the threshold (3/3). Boundary test for
    consistent counter aggregation when the rescue fires on the
    first chunk, last chunk, and every chunk in between
    (pr-test-analyzer R1 LOW). Fallback emits non-empty mentions on
    each chunk so the contract-of-_succeeded ('fallback delivered
    content') is satisfied — see test_tiered_hallucination_rescue_empty_fallback_*
    below for the empty-response semantics."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1", "context": "Faux1"},
                ]
            },
            {
                "mentions": [
                    {"form": "Wear", "context": "Wear"},
                    {"form": "Faux2", "context": "Faux2"},
                ]
            },
            {
                "mentions": [
                    {"form": "Tyne", "context": "Tyne"},
                    {"form": "Faux3", "context": "Faux3"},
                ]
            },
        ]
    )
    fallback = FakeClient(
        [
            # Non-empty rescues: each emits the same form the primary
            # found, with same context → dedup collapses, but
            # rescue_mentions itself is non-empty so _succeeded
            # increments under the round-2 contract.
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            {"mentions": [{"form": "Wear", "context": "Wear"}]},
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_hallucination_rescue_attempted == 3
    assert report.chunks_hallucination_rescue_succeeded == 3
    assert len(fallback.calls) == 3


def test_tiered_hallucination_rescue_empty_fallback_response_counts_as_unsucceeded():
    """wyrd-z8mq round 2 silent-failure-hunter HIGH: a fallback that
    returns `{"mentions": []}` (content-policy refusal, schema
    validation glitch, model declining to extract) is operationally
    equivalent to a failed rescue — the primary's hallucinations
    weren't actually caught even though the call mechanically
    succeeded. _attempted increments (rescue WAS triggered);
    _succeeded does NOT (no content delivered). The CLI summary
    will surface this as 'halluc_rescued=0/N', identical to the
    transport-failure case."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1", "context": "Faux1"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient(
        [
            # Mechanically successful call, but empty result —
            # the silent-empty case the round-2 fix is meant to
            # surface.
            {"mentions": []},
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_hallucination_rescue_attempted == 1
    assert report.chunks_hallucination_rescue_succeeded == 0  # empty → no content delivered
    assert report.chunks_failed == 0  # primary's mentions still good
    assert [m.form for m in report.mentions] == ["Edlin"]


def test_tiered_hallucination_rescue_log_warnings_emitted():
    """wyrd-z8mq R2 pr-test-analyzer LOW: the rescue's trigger log
    line + the fallback-failure log line are operator-debugging
    contracts ('which chunks were rescued', 'which rescues failed').
    A refactor that drops chunk index or changes the wording would
    not be caught by other tests. Capture both."""

    class _FailingFallback:
        model = "fail"

        def chat_json(self, system, user, schema=None):
            raise RuntimeError("simulated rescue transport failure")

    primary = FakeClient(
        [
            # chunk 1 (index 0): triggers rescue
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1", "context": "Faux1"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    warnings: list[str] = []
    mine_toponym_mentions_tiered(
        primary,
        _FailingFallback(),
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
        log_warning=warnings.append,
    )
    rescue_trigger_lines = [w for w in warnings if "running fallback for rescue" in w]
    assert len(rescue_trigger_lines) == 1
    assert "chunk 0" in rescue_trigger_lines[0]  # chunk index preserved
    assert "1 forms" in rescue_trigger_lines[0]  # hallucination count preserved

    rescue_fail_lines = [w for w in warnings if "rescue fallback failed" in w]
    assert len(rescue_fail_lines) == 1
    assert "chunk 0" in rescue_fail_lines[0]
    assert "simulated rescue transport failure" in rescue_fail_lines[0]
    assert "primary mentions retained" in rescue_fail_lines[0]


def test_tiered_hallucination_rescue_reraises_programmer_errors_from_rescue():
    """wyrd-z8mq R2 pr-test-analyzer LOW + test-coverage LOW
    (convergent): the rescue helper has its OWN
    _PROGRAMMER_ERROR_EXCEPTIONS re-raise clause (separate from the
    primary-failed-fallback path's). A regression that swallowed
    AttributeError/TypeError/NameError/ImportError/ValueError on
    the rescue path would downgrade real bugs into 'failed rescue'
    log lines, masking them in production.

    Mirrors test_tiered_reraises_programmer_errors_from_fallback
    but for the rescue path specifically."""

    class _ProgrammerErrorOnRescue:
        model = "broken"

        def chat_json(self, system, user, schema=None):
            # NOT a transport error — a real Python bug surface.
            raise AttributeError("simulated missing method bug")

    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1", "context": "Faux1"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    import pytest

    with pytest.raises(AttributeError, match="simulated missing method bug"):
        mine_toponym_mentions_tiered(
            primary,
            _ProgrammerErrorOnRescue(),
            "test_source",
            _three_chunk_body(),
            target_chunk_size=10000,
            hallucination_fallback_threshold=1,
        )


def test_tiered_hallucination_rescue_folds_years_clamped_from_fallback():
    """wyrd-z8mq R2 pr-test-analyzer LOW: years_clamped from the
    rescue's chunk_counters folds into the report alongside
    hallucinations_dropped. The aggregation test only pinned
    hallucinations_dropped; a regression dropping the years_clamped
    fold would silently lose the year-coercion stat from rescued
    chunks.

    A primary mention with an out-of-range date_year triggers
    years_clamped on the primary side; a fallback mention with the
    same triggers years_clamped on the rescue side. Both should
    sum into the report's aggregate."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin", "date_year": 9999},  # clamped
                    {"form": "Faux1", "context": "Faux1"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient(
        [
            # Rescue contributes its own clamped year — different
            # context so it isn't deduped against primary's Edlin.
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin other", "date_year": 5555},
                ]
            },
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_hallucination_rescue_attempted == 1
    assert report.chunks_hallucination_rescue_succeeded == 1
    # primary clamped 1 + fallback rescue clamped 1 = 2
    assert report.years_clamped == 2


def test_tiered_hallucination_rescue_dedup_key_distinguishes_region_hint():
    """wyrd-z8mq R2 test-coverage LOW: dedup-key tests cover form,
    date_year, and context distinctions but not region_hint. A
    regression dropping region_hint from
    _hallucination_rescue_dedup_key would collapse mentions of the
    same place name from different regions (e.g. Newton in
    Northumberland vs Newton in Durham) into one. Pin the slot
    explicitly."""
    primary = FakeClient(
        [
            {"mentions": []},
            {"mentions": []},
            {
                "mentions": [
                    {"form": "Tyne", "context": "Tyne shared", "region_hint": "Northumberland"},
                    {"form": "Faux", "context": "Faux"},
                ]
            },
        ]
    )
    fallback = FakeClient(
        [
            # Same form + same context, different region_hint —
            # distinct dedup key, both must survive.
            {"mentions": [{"form": "Tyne", "context": "Tyne shared", "region_hint": "Durham"}]},
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    regions = sorted(m.region_hint for m in report.mentions)
    assert regions == ["Durham", "Northumberland"]


def test_tiered_hallucination_rescue_dedupes_internal_fallback_duplicates():
    """wyrd-z8mq round 3 Gemini MEDIUM: when the fallback itself
    emits the same mention twice, the union merge should keep only
    one. The base extractor doesn't dedup intra-response duplicates;
    the union-merge pass is the appropriate place to clean them.

    Chunk 1's body contains 'Edlin' (real form, word-boundary
    matches). Primary emits Edlin with one context; fallback emits
    Edlin TWICE with a different (byte-identical-to-itself) context.
    Without internal dedup the result would have 3 Edlins
    (primary + fallback x2); with it, the result has 2 (primary +
    one fallback)."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin primary-context"},
                    {"form": "Faux", "context": "Faux"},
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin fallback-context"},
                    {"form": "Edlin", "context": "Edlin fallback-context"},
                ]
            },
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_hallucination_rescue_attempted == 1
    assert report.chunks_hallucination_rescue_succeeded == 1
    # Two Edlins (primary's + one fallback). Without internal dedup
    # we'd have three.
    edlin_contexts = sorted(m.context for m in report.mentions if m.form == "Edlin")
    assert edlin_contexts == ["Edlin fallback-context", "Edlin primary-context"]


def test_run_hallucination_rescue_failure_path_returns_fresh_list():
    """wyrd-z8mq R5 (Gemini MEDIUM) + R6 (test-coverage LOW): pin
    the list-identity contract on the failure path. The R5 fix
    changed `return primary_mentions, False` to
    `return list(primary_mentions), False` for ownership-contract
    consistency with the success path. A regression deleting the
    `list(...)` wrap would silently alias the failure-path return
    — not a current bug (caller only `.extend()`s the result) but
    a latent hazard the R5 fix was specifically meant to prevent.
    Until now this contract had no test."""
    from wyrd.generators.kenning.extractors.toponym_mentions import (
        ToponymMention,
        ValidationCounters,
        _run_hallucination_rescue,
    )

    class _RaisingFallback:
        model = "raise"

        def chat_json(self, system, user, schema=None):
            raise RuntimeError("simulated rescue transport failure")

    primary = [
        ToponymMention(form="Edlin", date_year=None, region_hint=None, context="Edlin context")
    ]
    result, delta, succeeded = _run_hallucination_rescue(
        _RaisingFallback(),
        chunk="Edlin context",
        chunk_index=0,
        primary_mentions=primary,
        log_warning=None,
    )
    assert succeeded is False
    assert result == primary  # same contents
    assert result is not primary  # ...but a fresh list (the load-bearing assertion)
    # wyrd-oaq5: errored-rescue returns a zero counters delta so the
    # caller's ``+=`` fold is a no-op (primary's counters stay untouched).
    assert delta == ValidationCounters()


def test_tiered_hallucination_rescue_only_when_primary_succeeds():
    """The rescue path is mutually exclusive with the primary-failed
    path. If the primary FAILS and the fallback recovers, the recovery
    path counters increment but the rescue counters do NOT — even if
    the recovery fallback emits its own hallucinations."""
    primary = FakeClient(
        [
            RuntimeError("primary timeout"),
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1", "context": "Faux1"},
                    {"form": "Faux2", "context": "Faux2"},
                ]
            },
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    assert report.chunks_recovered_by_fallback == 1
    assert report.chunks_hallucination_rescue_attempted == 0  # mutually exclusive
    assert report.chunks_hallucination_rescue_succeeded == 0
    assert report.hallucinations_dropped == 2
    assert [m.form for m in report.mentions] == ["Edlin"]


# ---------- CLI integration for the rescue flag (wyrd-z8mq) -------------


def test_cli_hallucination_fallback_threshold_flows_through(tmp_path, monkeypatch):
    """End-to-end: --hallucination-fallback-threshold flag is plumbed
    from the click decorator through the function call site so a
    typo'd kwarg or missing pass-through is caught (pr-test-analyzer
    R1 HIGH). Source body has 1 chunk; primary emits 1 real + 1 fake;
    fallback succeeds → halluc_rescued counters increment."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham mentioned here.", encoding="utf-8")
    output_dir = tmp_path / "out"

    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlingham", "context": "Edlingham mentioned here"},
                    {"form": "FauxPlace", "context": "FauxPlace fabricated"},
                ]
            }
        ]
    )
    fallback = FakeClient([{"mentions": []}])
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--hallucination-fallback-threshold",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    # Fallback was actually called → flag plumbed through.
    assert len(fallback.calls) == 1


def test_cli_hallucination_fallback_threshold_rejects_negative(tmp_path):
    """click.IntRange(min=0) rejects negative threshold values
    (type-design-analyzer R1 MEDIUM). Without the IntRange guard,
    a negative threshold would silently disable the rescue (the
    library checks `> 0`) while the operator might expect a stricter
    config (or some other intent) — better to fail loudly at the CLI."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("body", encoding="utf-8")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--hallucination-fallback-threshold",
            "-1",
        ],
    )
    assert result.exit_code != 0
    # click.IntRange validation message
    assert "is not in the range" in result.output or "Invalid value" in result.output


def test_cli_summary_reports_halluc_rescued_succeeded_over_attempted(tmp_path, monkeypatch):
    """Per-source + TOTAL summary lines include
    `halluc_rescued=succeeded/attempted` (silent-failure-hunter R1
    HIGH + test-coverage R1 MEDIUM). A broken fallback (every rescue
    errors) would show e.g. "0/14" — the gap is the only signal that
    the rescue is structurally failing."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham one.", encoding="utf-8")
    (sources_dir / "src_b.txt").write_text("Tynemouth two.", encoding="utf-8")
    output_dir = tmp_path / "out"

    # Each source: primary emits 1 real + 1 fake (trips threshold=1).
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlingham", "context": "Edlingham one"},
                    {"form": "FauxA", "context": "FauxA"},
                ]
            },
            {
                "mentions": [
                    {"form": "Tynemouth", "context": "Tynemouth two"},
                    {"form": "FauxB", "context": "FauxB"},
                ]
            },
        ]
    )
    # Fallback emits a non-empty rescue per source so succeeded
    # increments under the round-2 'bool(rescue_mentions)' contract.
    fallback = FakeClient(
        [
            {"mentions": [{"form": "Edlingham", "context": "Edlingham one"}]},
            {"mentions": [{"form": "Tynemouth", "context": "Tynemouth two"}]},
        ]
    )
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--hallucination-fallback-threshold",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    # Per-source summary lines: each source had 1 attempted + 1 succeeded
    assert "halluc_rescued=1/1" in result.output
    # TOTAL summary aggregates: 2 attempted + 2 succeeded across sources
    assert "halluc_rescued=2/2" in result.output
    # Gemini wyrd-z8mq R3 MEDIUM: per-source + TOTAL lines now show
    # halluc + clamped aggregates. Each source dropped 1 hallucination
    # (FauxA / FauxB); total = 2 across sources. No year-clamping in
    # this scenario (no date_year fields set).
    assert "halluc=1 " in result.output  # per-source
    assert "halluc=2 " in result.output  # TOTAL
    assert "clamped=0 " in result.output  # appears in both per-source + TOTAL lines


def test_cli_summary_reports_silent_empty_fallback_as_gap(tmp_path, monkeypatch):
    """wyrd-z8mq R5 pr-test-analyzer LOW: the CLI comment about the
    broken-fallback signal covers BOTH 'every fallback errored' (the
    pre-existing test_cli_summary_reports_broken_fallback_as_gap)
    AND 'every fallback returned empty / all-hallucinated' (the
    silent case). The unit-level test pins the latter, but no CLI
    test exercises the silent-empty case until now. Closes the
    test/documentation parity loop the R4 comment opened."""

    class _SilentEmptyFallback:
        """Fallback that returns content that fails the word-boundary
        guard — admit list will be empty, succeeded=0, but the call
        itself doesn't raise."""

        model = "silent-empty"

        def chat_json(self, system, user, schema=None):
            return {"mentions": [{"form": "NotInChunkAnywhere", "context": "fake"}]}

    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham here.", encoding="utf-8")
    output_dir = tmp_path / "out"

    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlingham", "context": "Edlingham here"},
                    {"form": "Faux", "context": "Faux"},
                ]
            }
        ]
    )
    _stub_ollama_for_cli(monkeypatch, primary)
    import wyrd.generators.kenning.extractors.anthropic as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", lambda **kw: _SilentEmptyFallback())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--hallucination-fallback-threshold",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    # Same shape as the errored-fallback case: attempted=1, succeeded=0
    # surfaces as halluc_rescued=0/1. Operator sees this same signal
    # whether the fallback errored OR returned content that failed
    # the word-boundary guard.
    assert "halluc_rescued=0/1" in result.output


def test_cli_summary_reports_years_clamped_non_zero(tmp_path, monkeypatch):
    """wyrd-z8mq R4 pr-test-analyzer LOW: the previous CLI summary
    test asserts `clamped=0` everywhere because no date_year is
    set in the fixture. A regression deleting
    `totals['years_clamped'] += report.years_clamped` would pass
    the all-zero assertion (right-hand side is always 0). Pin
    non-zero aggregation explicitly with a primary that emits
    out-of-range date_year values (clamped by the validator)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham one.", encoding="utf-8")
    (sources_dir / "src_b.txt").write_text("Tynemouth two.", encoding="utf-8")
    output_dir = tmp_path / "out"

    # Each source emits 1 mention with date_year=9999 (out of range,
    # gets clamped to None — years_clamped += 1 per source).
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlingham", "context": "Edlingham one", "date_year": 9999},
                ]
            },
            {
                "mentions": [
                    {"form": "Tynemouth", "context": "Tynemouth two", "date_year": 8888},
                ]
            },
        ]
    )
    fallback = FakeClient([])  # no rescue triggered (no hallucinations)
    _stub_ollama_for_cli(monkeypatch, primary)
    _stub_anthropic_for_cli(monkeypatch, fallback)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    # Per-source line ends with `clamped=N\n` (no trailing space in
    # the f-string, it's the last field), so match without trailing
    # space. TOTAL line has trailing-space form.
    assert "clamped=1" in result.output  # appears on per-source line(s)
    # TOTAL line shows clamped=2 (sum across sources). A regression
    # deleting the totals['years_clamped'] += line would produce
    # 'clamped=0 ' in the TOTAL line and fail this assertion.
    assert "clamped=2 " in result.output


def test_tiered_hallucination_rescue_folds_counters_even_on_empty_admit():
    """wyrd-z8mq R4 comment-analyzer: the helper docstring claims
    fallback counters fold whenever the call returns — including
    the all-hallucinated case where rescue_mentions ends up empty
    but rescue_counters.hallucinations_dropped > 0. No test pinned
    this until now; a regression switching the fold to be
    conditional on rescue_mentions non-empty would not be caught.

    Pin: primary trips the threshold; fallback returns 2 fabricated
    forms (both fail the word-boundary guard); admit list = [] but
    rescue_counters.hallucinations_dropped = 2. The report's
    aggregate hallucinations_dropped reflects BOTH primary's drops
    AND fallback's drops."""
    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1", "context": "Faux1"},  # primary halluc #1
                ]
            },
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient(
        [
            {
                "mentions": [
                    # Both fail the word-boundary guard (not in chunk 1).
                    {"form": "NotInChunk_A", "context": "NotInChunk_A"},
                    {"form": "NotInChunk_B", "context": "NotInChunk_B"},
                ]
            },
        ]
    )
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=1,
    )
    # Rescue attempted (threshold tripped), but fallback delivered
    # no admitted content → succeeded=0 under the bool(rescue_mentions)
    # contract.
    assert report.chunks_hallucination_rescue_attempted == 1
    assert report.chunks_hallucination_rescue_succeeded == 0
    # Counters STILL fold: primary's 1 halluc + fallback's 2 dropped = 3.
    # A regression dropping the fold in the empty-admit case would
    # show hallucinations_dropped == 1 (primary only).
    assert report.hallucinations_dropped == 3
    # Note (pr-test-analyzer R5 LOW investigation, won't-fix):
    # the parallel years_clamped fold can't be non-vacuously tested
    # on the empty-admit path because _validated_mentions drops
    # forms (incrementing hallucinations_dropped) BEFORE running
    # _coerce_year — so any rescue_mentions=[] case has
    # rescue_counters.years_clamped=0 by construction. The
    # years_clamped fold IS tested in the success-path test
    # ``test_tiered_hallucination_rescue_folds_years_clamped_from_fallback``
    # where admit list is non-empty + carries clamped years.


def test_cli_summary_reports_broken_fallback_as_gap(tmp_path, monkeypatch):
    """Companion to the previous test: a broken fallback shows up as
    halluc_rescued=0/N (succeeded < attempted) in the summary. This
    is the signal an operator with a misconfigured API key would
    actually see (silent-failure-hunter R1 HIGH)."""

    class _FailingFallback:
        model = "fail"

        def chat_json(self, system, user, schema=None):
            raise RuntimeError("simulated bad API key")

    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham here.", encoding="utf-8")
    output_dir = tmp_path / "out"

    primary = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlingham", "context": "Edlingham here"},
                    {"form": "Faux", "context": "Faux"},
                ]
            }
        ]
    )
    _stub_ollama_for_cli(monkeypatch, primary)
    # Plug a failing fallback in by monkeypatching AnthropicClient
    # construction to return the failing stub.
    import wyrd.generators.kenning.extractors.anthropic as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", lambda **kw: _FailingFallback())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-tiered",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--hallucination-fallback-threshold",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    # Attempted = 1, succeeded = 0 — the gap is the broken-fallback signal.
    assert "halluc_rescued=0/1" in result.output
