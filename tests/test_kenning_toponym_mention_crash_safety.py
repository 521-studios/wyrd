"""Tests for crash-safe mining (wyrd-w7i3).

The staged miner streams each successful chunk's mentions to an
in-progress log under ``<output_dir>/_inprogress/<source>.chunks.jsonl``
and fsyncs after every chunk. SIGTERM/crash mid-source leaves the
log behind; ``recover-inprogress-chunks`` promotes/merges it into
the canonical ``<source>.jsonl``.

These tests exercise the contract at the extractor level (the
``on_chunk_mentions`` callback) and the CLI level (the recovery
command), avoiding real LLM calls via the FakeClient pattern.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.extractors.toponym_mentions import (
    _FAILED_CHUNKS_HEAD,
    MineToponymMentionsReport,
    ToponymMention,
    _record_chunk_failure,
    chunk_source_body,
    mine_toponym_mentions,
)


def test_record_chunk_failure_buffers_sanitizes_and_notifies():
    """``_record_chunk_failure`` (extracted from the per-chunk loop): counts the
    failure, captures it head-first then into the bounded tail ring buffer,
    UTF-8-sanitizes the STORED error (wyrd-8wa4) while passing the raw exception
    text to the operator log, and fires the on_chunk_failed callback."""
    from collections import deque

    report = MineToponymMentionsReport(source_id="s")
    head: list = []
    tail: deque = deque(maxlen=4)
    failed_cb: list = []
    logs: list = []

    # First _FAILED_CHUNKS_HEAD failures land in head; the next overflows to tail.
    for n in range(_FAILED_CHUNKS_HEAD + 1):
        _record_chunk_failure(
            report,
            head,
            tail,
            index=n,
            chunk="body",
            error=ValueError(f"boom {n} \udce9"),  # lone surrogate exercises sanitize
            on_chunk_failed=failed_cb.append,
            log_warning=logs.append,
        )
    assert report.chunks_failed == _FAILED_CHUNKS_HEAD + 1
    assert len(head) == _FAILED_CHUNKS_HEAD and len(tail) == 1
    assert len(failed_cb) == _FAILED_CHUNKS_HEAD + 1
    # Stored error is sanitized (no lone surrogate); the log line keeps raw text.
    assert "\udce9" not in head[0].error
    assert head[0].error.startswith("ValueError: boom 0")
    assert logs[0] == "chunk 0 failed: ValueError: boom 0 \udce9"


class FakeClient:
    """Replays canned chat_json responses; raise on Exception slots."""

    def __init__(self, responses: list[Any]):
        self.model = "fake-test-model"
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict | None]] = []

    def chat_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        self.calls.append((system, user, schema))
        if not self._responses:
            raise StopIteration("FakeClient ran out of canned responses")
        nxt = self._responses.pop(0)
        # BaseException covers KeyboardInterrupt/SystemExit/GeneratorExit
        # too, which Exception alone misses — necessary so a test can
        # simulate a true crash (SIGTERM-equivalent) and not a routine
        # per-chunk LLM error.
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


def _three_chunk_body() -> str:
    """Deterministically 3-chunk body at target_chunk_size=10000."""
    p1 = "A" * 7995 + " Edlin"
    p2 = "B" * 7996 + " Wear"
    p3 = "C" * 7996 + " Tyne"
    return f"{p1}\n\n{p2}\n\n{p3}"


def test_three_chunk_body_indeed_chunks_to_three():
    assert len(chunk_source_body(_three_chunk_body(), target_chunk_size=10000)) == 3


# ---------- extractor-level on_chunk_mentions callback ------------------


def test_on_chunk_mentions_fires_per_successful_chunk():
    """Callback is invoked once per SUCCESSFUL chunk with that chunk's
    mentions only — not the cumulative report."""
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            {"mentions": []},
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    seen: list[tuple[int, list[str]]] = []

    def _capture(chunk_index: int, mentions: list[ToponymMention]) -> None:
        seen.append((chunk_index, [m.form for m in mentions]))

    report = mine_toponym_mentions(
        client,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        on_chunk_mentions=_capture,
    )
    assert report.chunks_processed == 3
    assert seen == [(0, ["Edlin"]), (1, []), (2, ["Tyne"])]


def test_on_chunk_mentions_skipped_on_failed_chunk():
    """Failed chunks (LLM raised) do NOT invoke on_chunk_mentions —
    only on_chunk_failed. The chunk_index increments are gap-aware."""
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            RuntimeError("transport boom"),
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    seen_ok: list[tuple[int, list[str]]] = []

    def _capture(chunk_index: int, mentions: list[ToponymMention]) -> None:
        seen_ok.append((chunk_index, [m.form for m in mentions]))

    report = mine_toponym_mentions(
        client,
        "test_source",
        _three_chunk_body(),
        target_chunk_size=10000,
        on_chunk_mentions=_capture,
    )
    assert report.chunks_processed == 2
    assert report.chunks_failed == 1
    # The skipped chunk_index (1) is absent from the mentions stream.
    assert seen_ok == [(0, ["Edlin"]), (2, ["Tyne"])]


# ---------- recover-inprogress-chunks CLI -------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_recover_promotes_when_canonical_missing(tmp_path: Path):
    """Fresh-mining crash: in-progress log exists, canonical does NOT.
    Recovery renames the log to the canonical path."""
    output_dir = tmp_path / "phase2"
    inprogress = output_dir / "_inprogress" / "src_a.chunks.jsonl"
    _write_jsonl(
        inprogress,
        [
            {
                "source_id": "src_a",
                "form": "Foo",
                "date_year": None,
                "region_hint": None,
                "context": "Foo",
                "extractor": "x:y",
            },
            {
                "source_id": "src_a",
                "form": "Bar",
                "date_year": 1066,
                "region_hint": "Wessex",
                "context": "Bar in Wessex",
                "extractor": "x:y",
            },
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_root, ["lexicon", "recover-inprogress-chunks", "--output-dir", str(output_dir)]
    )
    assert result.exit_code == 0, result.output

    canonical = output_dir / "src_a.jsonl"
    assert canonical.exists()
    assert not inprogress.exists()
    rows = _read_jsonl(canonical)
    assert [r["form"] for r in rows] == ["Foo", "Bar"]


def test_recover_merges_with_dedup_when_canonical_exists(tmp_path: Path):
    """Resume-from-failures crash: canonical exists (with existing
    rows); in-progress log has some new + some duplicate rows.
    Recovery preserves canonical rows verbatim and adds only the
    novel ones."""
    output_dir = tmp_path / "phase2"
    canonical = output_dir / "src_b.jsonl"
    inprogress = output_dir / "_inprogress" / "src_b.chunks.jsonl"

    _write_jsonl(
        canonical,
        [
            {
                "source_id": "src_b",
                "form": "Alpha",
                "date_year": None,
                "region_hint": None,
                "context": "Alpha",
                "extractor": "stage1:gemma",
            },
        ],
    )
    _write_jsonl(
        inprogress,
        [
            # Duplicate (same form/year/region/context as Alpha above).
            {
                "source_id": "src_b",
                "form": "Alpha",
                "date_year": None,
                "region_hint": None,
                "context": "Alpha",
                "extractor": "stage2:gemini",
            },
            # Novel.
            {
                "source_id": "src_b",
                "form": "Beta",
                "date_year": 1100,
                "region_hint": None,
                "context": "Beta",
                "extractor": "stage2:gemini",
            },
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_root, ["lexicon", "recover-inprogress-chunks", "--output-dir", str(output_dir)]
    )
    assert result.exit_code == 0, result.output

    assert not inprogress.exists()
    rows = _read_jsonl(canonical)
    assert len(rows) == 2
    # Existing row preserved verbatim (extractor field unchanged).
    assert rows[0]["form"] == "Alpha"
    assert rows[0]["extractor"] == "stage1:gemma"
    # Novel row added, keeps its own extractor.
    assert rows[1]["form"] == "Beta"
    assert rows[1]["extractor"] == "stage2:gemini"


def test_recover_dry_run_writes_nothing(tmp_path: Path):
    """--dry-run reports what would happen but leaves all files intact."""
    output_dir = tmp_path / "phase2"
    inprogress = output_dir / "_inprogress" / "src_c.chunks.jsonl"
    _write_jsonl(
        inprogress,
        [
            {
                "source_id": "src_c",
                "form": "Foo",
                "date_year": None,
                "region_hint": None,
                "context": "Foo",
                "extractor": "x:y",
            }
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "recover-inprogress-chunks", "--output-dir", str(output_dir), "--dry-run"],
    )
    assert result.exit_code == 0, result.output

    canonical = output_dir / "src_c.jsonl"
    assert not canonical.exists()
    assert inprogress.exists()
    assert "dry-run" in result.stderr


def test_recover_handles_empty_inprogress_log(tmp_path: Path):
    """Edge case: source was started but no chunk succeeded before the
    crash. The empty log is cleaned up; no canonical file is created."""
    output_dir = tmp_path / "phase2"
    inprogress = output_dir / "_inprogress" / "src_d.chunks.jsonl"
    inprogress.parent.mkdir(parents=True, exist_ok=True)
    inprogress.write_text("", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli_root, ["lexicon", "recover-inprogress-chunks", "--output-dir", str(output_dir)]
    )
    assert result.exit_code == 0, result.output

    assert not inprogress.exists()
    assert not (output_dir / "src_d.jsonl").exists()


def test_recover_no_inprogress_dir_is_no_op(tmp_path: Path):
    """Pristine output_dir with no _inprogress subdir → command emits a
    notice and exits cleanly."""
    output_dir = tmp_path / "phase2"
    output_dir.mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(
        cli_root, ["lexicon", "recover-inprogress-chunks", "--output-dir", str(output_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "No in-progress logs" in result.stderr


# ---------- staged-miner in-progress helpers ----------------------------


def test_inprogress_path_layout(tmp_path: Path):
    """In-progress logs live under <output_dir>/_inprogress/<source>.chunks.jsonl —
    the recovery CLI greps for that layout, so the convention is load-bearing."""
    from wyrd.generators.kenning.cli.lexicon.mine_toponym_mentions_staged import (
        _inprogress_path,
    )

    p = _inprogress_path(tmp_path / "phase2", "ekwall_1922_lancashire")
    assert p == tmp_path / "phase2" / "_inprogress" / "ekwall_1922_lancashire.chunks.jsonl"


def test_check_inprogress_clear_rejects_existing_log(tmp_path: Path):
    """Pre-existing in-progress log blocks mining — silently overwriting
    it would lose the prior crash's recovery candidate."""
    import click
    import pytest

    from wyrd.generators.kenning.cli.lexicon.mine_toponym_mentions_staged import (
        _check_inprogress_clear,
        _inprogress_path,
    )

    output_dir = tmp_path / "phase2"
    inprogress = _inprogress_path(output_dir, "src")
    inprogress.parent.mkdir(parents=True)
    inprogress.write_text("partial-data\n", encoding="utf-8")

    with pytest.raises(click.ClickException) as exc:
        _check_inprogress_clear(inprogress, output_dir)
    assert "recover-inprogress-chunks" in str(exc.value.message)


def test_check_inprogress_clear_passes_when_no_log(tmp_path: Path):
    """No log → mining proceeds without exception."""
    from wyrd.generators.kenning.cli.lexicon.mine_toponym_mentions_staged import (
        _check_inprogress_clear,
        _inprogress_path,
    )

    output_dir = tmp_path / "phase2"
    inprogress = _inprogress_path(output_dir, "src")
    # File does not exist; should not raise.
    _check_inprogress_clear(inprogress, output_dir)


def test_check_inprogress_clear_auto_clears_empty_log(tmp_path: Path):
    """A 0-byte log carries no recovery value (crash hit before chunk 0
    succeeded). Auto-clear so operators aren't blocked by a stale marker
    pointing at nothing. wyrd-w7i3 HIGH-1."""
    from wyrd.generators.kenning.cli.lexicon.mine_toponym_mentions_staged import (
        _check_inprogress_clear,
        _inprogress_path,
    )

    output_dir = tmp_path / "phase2"
    inprogress = _inprogress_path(output_dir, "src")
    inprogress.parent.mkdir(parents=True)
    inprogress.write_text("", encoding="utf-8")
    assert inprogress.exists()

    # Should auto-clear, not raise.
    _check_inprogress_clear(inprogress, output_dir)
    assert not inprogress.exists()


# ---------- _make_chunk_mentions_writer durability invariant ------------


def test_chunk_mentions_writer_flushes_and_fsyncs_per_chunk(tmp_path: Path, monkeypatch):
    """The writer's durability contract — flush + fsync after each chunk —
    is the entire foundation of crash-safety. Pin it directly so a
    refactor that drops or reorders the flush/fsync calls trips a test.
    wyrd-w7i3."""
    import os as os_module

    from wyrd.generators.kenning.cli.lexicon.mine_toponym_mentions_staged import (
        _make_chunk_mentions_writer,
    )
    from wyrd.generators.kenning.extractors.toponym_mentions import ToponymMention

    fsync_calls: list[int] = []

    def _record_fsync(fd: int) -> None:
        fsync_calls.append(fd)

    monkeypatch.setattr(os_module, "fsync", _record_fsync)

    log_path = tmp_path / "src.chunks.jsonl"
    sink = log_path.open("w", encoding="utf-8")
    try:
        writer = _make_chunk_mentions_writer(
            sink,
            "src",
            lambda sid, m: f'{{"source_id":"{sid}","form":"{m.form}"}}',
        )
        # Two successful chunks → two fsync calls.
        writer(0, [ToponymMention(form="Foo", date_year=None, region_hint=None, context="Foo")])
        writer(
            1,
            [
                ToponymMention(form="Bar", date_year=None, region_hint=None, context="Bar"),
                ToponymMention(form="Baz", date_year=None, region_hint=None, context="Baz"),
            ],
        )
    finally:
        sink.close()

    assert len(fsync_calls) == 2
    rows = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(rows) == 3
    assert '"form":"Foo"' in rows[0]
    assert '"form":"Bar"' in rows[1]
    assert '"form":"Baz"' in rows[2]


def test_chunk_mentions_writer_replaces_surrogate_instead_of_crashing(tmp_path: Path):
    """wyrd-klod fault-injection: the staged miner's file sinks open with
    ``errors="replace"`` as a defense-in-depth backstop. The in-band
    sanitizer (wyrd-8wa4 / PR #309) normally maps surrogate code points to
    U+FFFD before they reach a sink, but if a future path constructs a
    ToponymMention that bypasses that guard, a lone surrogate would reach
    the sink and crash the mine with UnicodeEncodeError. The backstop must
    substitute ASCII "?" and keep going instead.

    Drives the production in-progress sink consumer
    (``_make_chunk_mentions_writer``) with a serializer that emits a lone
    surrogate, mirroring a bypassed guard.
    """
    import pytest

    from wyrd.generators.kenning.cli.lexicon.mine_toponym_mentions_staged import (
        _make_chunk_mentions_writer,
    )
    from wyrd.generators.kenning.extractors.toponym_mentions import ToponymMention

    mention = ToponymMention(form="Edlin", date_year=None, region_hint=None, context="Edlin")

    def _surrogate_serializer(sid: str, m: ToponymMention) -> str:
        # A lone surrogate (\udce9) — illegal in UTF-8 — standing in for a
        # mention field that escaped the in-band sanitizer.
        return f'{{"source_id":"{sid}","form":"{m.form}\udce9"}}'

    # Negative control: a strict UTF-8 sink genuinely cannot encode the
    # surrogate — so errors="replace" is load-bearing, not cosmetic.
    with (
        pytest.raises(UnicodeEncodeError),
        (tmp_path / "strict.jsonl").open("w", encoding="utf-8") as strict,
    ):
        strict.write("Edlin\udce9")

    # Production config: errors="replace" turns the surrogate into "?" and
    # the chunk write succeeds (matches the inprogress / tmp_path sinks in
    # mine_toponym_mentions_staged.py).
    safe_path = tmp_path / "src.chunks.jsonl"
    with safe_path.open("w", encoding="utf-8", errors="replace") as sink:
        writer = _make_chunk_mentions_writer(sink, "src", _surrogate_serializer)
        writer(0, [mention])  # must not raise UnicodeEncodeError

    content = safe_path.read_text(encoding="utf-8")
    assert "Edlin?" in content, f"surrogate should be replaced with '?', got: {content!r}"


# ---------- staged-CLI integration: fresh + resume × happy + crash ------


def _stub_ollama_for_cli(monkeypatch, fake_client) -> None:
    """Replace OllamaClient at its construction site so the staged CLI
    uses the FakeClient instead of probing http://10.5.2.31:11434."""
    import wyrd.generators.kenning.extractors.llm as llm_module

    monkeypatch.setattr(llm_module, "OllamaClient", lambda **kw: fake_client)


def test_cli_fresh_happy_path_unlinks_inprogress_log(tmp_path: Path, monkeypatch):
    """Clean completion: canonical file written, in-progress log
    unlinked. Regression-pin for a refactor that misses the unlink.
    wyrd-w7i3 HIGH (pr-test-analyzer)."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_a.txt").write_text("Edlingham in Northumberland.", encoding="utf-8")
    output_dir = tmp_path / "out"

    client = FakeClient(
        [{"mentions": [{"form": "Edlingham", "context": "Edlingham in Northumberland."}]}]
    )
    _stub_ollama_for_cli(monkeypatch, client)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "ollama",
            "--model",
            "fake-test-model",
        ],
    )
    assert result.exit_code == 0, result.output

    canonical = output_dir / "src_a.jsonl"
    inprogress = output_dir / "_inprogress" / "src_a.chunks.jsonl"
    assert canonical.exists()
    assert not inprogress.exists()


def test_cli_fresh_crash_preserves_inprogress_log(tmp_path: Path, monkeypatch):
    """SIGTERM-equivalent: BaseException raised by the LLM client mid-source.
    In-progress log must hold the pre-crash chunk's mentions; canonical
    file must NOT exist. Operator-recovery message must point at the
    recovery CLI. wyrd-w7i3 HIGH (pr-test-analyzer)."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    p1 = "A" * 7995 + " Edlin"
    p2 = "B" * 7996 + " Tyne"
    (sources_dir / "src_b.txt").write_text(f"{p1}\n\n{p2}", encoding="utf-8")
    output_dir = tmp_path / "out"

    # Chunk 0 succeeds; chunk 1 raises KeyboardInterrupt (BaseException
    # subclass — bypasses the chunk-loop's bare-Exception catch so the
    # mine aborts the source). This is the canonical mid-source crash
    # scenario the PR is designed to handle.
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            KeyboardInterrupt(),
        ]
    )
    _stub_ollama_for_cli(monkeypatch, client)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "ollama",
            "--model",
            "fake-test-model",
            "--chunk-size",
            "10000",
        ],
    )
    assert result.exit_code != 0, "mid-source crash must produce non-zero exit"

    canonical = output_dir / "src_b.jsonl"
    inprogress = output_dir / "_inprogress" / "src_b.chunks.jsonl"
    assert not canonical.exists(), "canonical file must not exist after mid-source crash"
    assert inprogress.exists(), "in-progress log must survive for recovery"

    # The surviving log contains chunk 0's mentions in canonical form.
    rows = _read_jsonl(inprogress)
    assert [r["form"] for r in rows] == ["Edlin"]

    # Operator-visible recovery message names the recovery CLI.
    combined = (result.stderr or "") + (result.output or "")
    assert "recover-inprogress-chunks" in combined


def test_cli_pre_existing_inprogress_log_blocks_mining(tmp_path: Path, monkeypatch):
    """The staged CLI surfaces _check_inprogress_clear as a non-zero
    exit. A regression that swallows the ClickException would bypass
    the data-loss guard. wyrd-w7i3 MEDIUM (pr-test-analyzer)."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_c.txt").write_text("Edlingham.", encoding="utf-8")
    output_dir = tmp_path / "out"
    inprogress = output_dir / "_inprogress" / "src_c.chunks.jsonl"
    inprogress.parent.mkdir(parents=True)
    inprogress.write_text(
        '{"source_id":"src_c","form":"OldRun","date_year":null,"region_hint":null,'
        '"context":"OldRun","extractor":"x:y"}\n',
        encoding="utf-8",
    )

    # The client must not be invoked — the guard fires first.
    client = FakeClient([{"mentions": []}])
    _stub_ollama_for_cli(monkeypatch, client)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "ollama",
            "--model",
            "fake-test-model",
        ],
    )
    assert result.exit_code != 0
    combined = (result.stderr or "") + (result.output or "")
    assert "recover-inprogress-chunks" in combined
    # Guard didn't touch the existing log; canonical was never written.
    assert inprogress.exists()
    assert not (output_dir / "src_c.jsonl").exists()


def test_cli_staged_sanitizes_surrogate_to_ufffd_in_persisted_output(tmp_path: Path, monkeypatch):
    """wyrd-4ywt: end-to-end through the staged CLI's REAL file writer (the
    in-progress sink + atomic merge to the per-source JSONL), driven by a mocked
    client that emits a mention whose CONTEXT carries a lone surrogate — the
    wyrd-8wa4 scenario (a provider SDK / surrogatepass JSON decode can yield
    these). The in-band sanitizer maps it to U+FFFD BEFORE the sink, so:

    - the persisted per-source JSONL is valid UTF-8 (a strict re-decode raises
      if any lone surrogate leaked through the writer),
    - the surrogate is U+FFFD in the persisted record (not the errors='replace'
      backstop's ASCII '?', which would mean the in-band guard was bypassed),
    - the run summary surfaces ``surrogates_sanitized``.

    The surrogate rides in ``context`` (not ``form``) so the form-in-chunk
    validation — which runs on the SANITIZED form — still admits the mention
    against the clean ASCII body.
    """
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "src_s.txt").write_text("Edlingham in Northumberland.", encoding="utf-8")
    output_dir = tmp_path / "out"

    client = FakeClient(
        [{"mentions": [{"form": "Edlingham", "context": "Edlingham\udce9 in Northumberland."}]}]
    )
    _stub_ollama_for_cli(monkeypatch, client)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "ollama",
            "--model",
            "fake-test-model",
        ],
    )
    assert result.exit_code == 0, result.output

    canonical = output_dir / "src_s.jsonl"
    assert canonical.exists()
    # Strict re-decode — raises if a lone surrogate leaked into the persisted file.
    text = canonical.read_bytes().decode("utf-8")
    rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    assert rows, "expected one persisted mention"
    ctx = rows[0]["context"]
    assert "�" in ctx, f"surrogate should be sanitized in-band to U+FFFD; got {ctx!r}"
    assert "\udce9" not in ctx, "no lone surrogate may survive to the persisted file"
    assert "?" not in ctx, "U+FFFD (in-band), not the errors='replace' backstop '?'"
    assert rows[0]["form"] == "Edlingham"
    # surrogates_sanitized is surfaced in the run summary (stderr).
    combined = (result.stderr or "") + (result.output or "")
    assert "surrogates=1" in combined, f"summary should report surrogates=1; got: {combined}"
