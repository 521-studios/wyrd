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


def test_tiered_counters_use_fallback_only_when_rescued():
    """When the fallback rescues a chunk, the validation counters
    (hallucinations_dropped, years_clamped) reflect the FALLBACK's
    output — not the primary's (which was discarded). Prevents
    double-counting."""
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
    """--source ../etc/passwd should be neutralized via Path(...).name.
    The bare 'passwd' won't exist under --sources-dir, producing the
    standard 'source body not found' error rather than reading outside."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    # File outside sources_dir that we DON'T want to read.
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
