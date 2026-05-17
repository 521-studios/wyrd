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
from wyrd.generators.kenning.toponym_mention_extractor import (
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
    import wyrd.generators.kenning.anthropic_extractor as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", lambda **kw: fake_client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")


def _stub_ollama_for_cli(monkeypatch, fake_client: FakeClient) -> None:
    import wyrd.generators.kenning.llm_extractor as llm_module

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

    import wyrd.generators.kenning.llm_extractor as llm_module

    monkeypatch.setattr(llm_module, "OllamaClient", recording_factory)
    monkeypatch.setattr(
        "wyrd.generators.kenning.anthropic_extractor.AnthropicClient",
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

    import wyrd.generators.kenning.anthropic_extractor as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", recording_factory)
    monkeypatch.setattr(
        "wyrd.generators.kenning.llm_extractor.OllamaClient",
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
    import wyrd.generators.kenning.gemini_extractor as gemini_module

    monkeypatch.setattr(gemini_module, "GeminiClient", lambda **kw: gemini_fake)
    monkeypatch.setattr(
        "wyrd.generators.kenning.anthropic_extractor.AnthropicClient",
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

    monkeypatch.setattr("wyrd.generators.kenning.llm_extractor.OllamaClient", ollama_factory)
    monkeypatch.setattr(
        "wyrd.generators.kenning.anthropic_extractor.AnthropicClient",
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
    from wyrd.generators.kenning.toponym_mention_extractor import (
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

    from wyrd.generators.kenning.toponym_mention_extractor import (
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
    from wyrd.generators.kenning.toponym_mention_extractor import (
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

    import wyrd.generators.kenning.llm_extractor as llm_module

    monkeypatch.setattr(llm_module, "OllamaClient", recording_factory)
    monkeypatch.setattr(
        "wyrd.generators.kenning.anthropic_extractor.AnthropicClient",
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
    from wyrd.generators.kenning.toponym_mention_extractor import (
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
    from wyrd.generators.kenning.toponym_mention_extractor import (
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
    from wyrd.generators.kenning.toponym_mention_extractor import (
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
# (form, date_year, region_hint). Designed for gemma4 primary +
# Anthropic fallback so the small model keeps its good mentions while
# the large model catches what it fabricated. Each FakeClient response
# below uses chunk 1's "Edlin" / chunk 2's "Wear" / chunk 3's "Tyne"
# as real forms; other strings (e.g. "Faux") trip the word-boundary
# guard since they don't appear in the chunk body.


def test_tiered_hallucination_rescue_triggers_when_threshold_met():
    """Primary emits a good mention + 2 hallucinated forms (not in
    chunk body); fallback emits 1 additional good mention. With
    threshold=1, the rescue fires; mentions get union-merged."""
    primary = FakeClient(
        [
            # chunk 1: 1 real ("Edlin") + 2 fake ("Faux1", "Faux2")
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin context"},
                    {"form": "Faux1", "context": "Faux1 context"},
                    {"form": "Faux2", "context": "Faux2 context"},
                ]
            },
            {"mentions": []},  # chunk 2: empty
            {"mentions": []},  # chunk 3: empty
        ]
    )
    fallback = FakeClient(
        [
            # chunk 1 rescue: 1 net-new mention the primary missed
            # (Edlin is in chunk 1 too — verified via the chunk body
            # `"A" * 7995 + " Edlin"`).
            {"mentions": [{"form": "Edlin", "context": "Edlin alt-context"}]},
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
    assert report.chunks_hallucination_rescued == 1
    assert report.chunks_recovered_by_fallback == 0  # primary didn't fail
    assert report.hallucinations_dropped == 2  # only primary's chunk-1 fakes (fallback had none)
    assert len(fallback.calls) == 1
    # Union merge: primary's "Edlin" is canonical (dedup key collision
    # on (form="Edlin", date_year=None, region_hint=None)). Fallback's
    # "Edlin" is dropped as duplicate. Primary's good mention stays.
    assert [m.form for m in report.mentions] == ["Edlin"]


def test_tiered_hallucination_rescue_below_threshold_no_fallback():
    """Primary emits 1 hallucinated form, but threshold=2 — rescue
    does NOT fire. Counter stays 0, fallback isn't called."""
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
    fallback = FakeClient([])  # not called
    report = mine_toponym_mentions_tiered(
        primary,
        fallback,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        hallucination_fallback_threshold=2,
    )
    assert report.chunks_hallucination_rescued == 0
    assert report.hallucinations_dropped == 1
    assert len(fallback.calls) == 0
    assert [m.form for m in report.mentions] == ["Edlin"]


def test_tiered_hallucination_rescue_disabled_by_default():
    """Without the threshold (None), primary's hallucinations don't
    trigger the rescue — existing two-tier semantics are preserved.
    Regression guard against accidentally enabling-on-default."""
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
        # hallucination_fallback_threshold defaults to None
    )
    assert report.chunks_hallucination_rescued == 0
    assert report.hallucinations_dropped == 2
    assert len(fallback.calls) == 0


def test_tiered_hallucination_rescue_disabled_when_zero():
    """threshold=0 is explicit-disabled (matches CLI default behavior
    where the operator passes 0 to mean 'off'). Fallback NOT called
    even with 5 hallucinations."""
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
    assert report.chunks_hallucination_rescued == 0
    assert len(fallback.calls) == 0


def test_tiered_hallucination_rescue_union_adds_novel_mentions():
    """Primary's chunk 2 hits threshold; fallback emits a NEW form
    primary didn't see (different from primary's dedup keys). Union
    merge keeps both."""
    primary = FakeClient(
        [
            {"mentions": []},
            # chunk 2 ("Wear"): 1 real + 1 fake
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
            # rescue on chunk 2: adds a different mention of Wear at
            # a different year (different dedup key) — both kept.
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
    assert report.chunks_hallucination_rescued == 1
    # Primary's (Wear, 1086) + fallback's (Wear, 1200) — different
    # date_year, different dedup keys, both retained.
    forms_years = [(m.form, m.date_year) for m in report.mentions]
    assert forms_years == [("Wear", 1086), ("Wear", 1200)]


def test_tiered_hallucination_rescue_continues_on_fallback_error():
    """Rescue is opportunistic — if the fallback ERRORS on the rescue
    pass, the primary's good mentions still flow through (no
    chunks_failed bump). Counter still increments since the rescue
    WAS attempted on this chunk (budget signal). Without this guard
    the operator would see chunks_failed bump even though gemma4's
    output was perfectly usable."""

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
    assert report.chunks_hallucination_rescued == 1  # attempt counts
    assert report.chunks_failed == 0  # primary's mentions still good
    assert report.chunks_processed == 3
    assert [m.form for m in report.mentions] == ["Edlin"]


def test_tiered_hallucination_rescue_aggregates_counters_across_chunks():
    """Counters aggregate across multiple rescued chunks (rescued
    counter and hallucinations_dropped from both tiers)."""
    primary = FakeClient(
        [
            # chunk 1: 1 real + 2 fake → triggers rescue
            {
                "mentions": [
                    {"form": "Edlin", "context": "Edlin"},
                    {"form": "Faux1a", "context": "Faux1a"},
                    {"form": "Faux1b", "context": "Faux1b"},
                ]
            },
            # chunk 2: 1 real + 1 fake → triggers rescue
            {
                "mentions": [
                    {"form": "Wear", "context": "Wear"},
                    {"form": "Faux2a", "context": "Faux2a"},
                ]
            },
            # chunk 3: 1 real + 0 fake → no rescue
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    fallback = FakeClient(
        [
            # rescue on chunk 1: no hallucinations
            {"mentions": [{"form": "Edlin", "context": "Edlin alt"}]},
            # rescue on chunk 2: 1 hallucination of its own
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
    assert report.chunks_hallucination_rescued == 2  # chunks 1 + 2 rescued
    # Hallucinations: primary chunk 1 (2) + primary chunk 2 (1) +
    # fallback chunk 2 rescue (1) = 4. Fallback chunk 1 rescue had 0.
    assert report.hallucinations_dropped == 4
    assert report.chunks_processed == 3
    assert report.chunks_failed == 0
    # All three primary mentions retained; fallback contributed no
    # new dedup keys (its Edlin/Wear collided with primary's).
    assert [m.form for m in report.mentions] == ["Edlin", "Wear", "Tyne"]


def test_tiered_hallucination_rescue_only_when_primary_succeeds():
    """The rescue path is mutually exclusive with the primary-failed
    path. If the primary FAILS and the fallback recovers (existing
    behavior), chunks_recovered_by_fallback increments but
    chunks_hallucination_rescued does NOT — even if the fallback's
    mentions include hallucinations of its own."""
    primary = FakeClient(
        [
            RuntimeError("primary timeout"),  # chunk 1: primary FAILS
            {"mentions": []},
            {"mentions": []},
        ]
    )
    fallback = FakeClient(
        [
            # chunk 1 recovery: 1 real + 2 fake. Hallucinations on
            # the recovery DON'T trigger a second rescue (mutually
            # exclusive). This is the existing chunks_recovered_by_
            # fallback path; rescue layer is separate.
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
    assert report.chunks_hallucination_rescued == 0  # mutually exclusive
    assert report.hallucinations_dropped == 2
    assert [m.form for m in report.mentions] == ["Edlin"]
