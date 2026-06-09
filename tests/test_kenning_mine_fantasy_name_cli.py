"""CLI-level tests for ``wyrd kenning lexicon mine-fantasy-name`` (wyrd-ami / D30).

The fantasy_pipeline resolution logic is covered in
test_kenning_fantasy_pipeline.py. These pin the wyrd-8uvi-extracted command
orchestration (``_validate_input_args`` / ``_parse_batch_inputs`` /
``_apply_skip_resolved`` / ``_process_input`` / ``_FantasyRun``) end-to-end via
CliRunner, with the pipeline's resolve + write calls stubbed — no LLM, no real
fantasy_morpheme writes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning import fantasy_pipeline as fp
from wyrd.generators.kenning.cli.lexicon.mine_fantasy_name import lexicon_mine_fantasy_name
from wyrd.generators.kenning.lexicon import init_schema


@pytest.fixture
def _stub_pipeline(monkeypatch):
    """Stub the pipeline: 'harpy*' resolves USABLE (descent-walk), everything
    else is BARRED (modern_coinage). Write/tag calls are no-ops; the
    already-resolved set is empty."""

    def _fake_resolve(db_path, name, desc, *, skip_llm, llm_caller, semantic_check_caller):
        if name.lower().startswith("harpy"):
            return SimpleNamespace(
                usable=True,
                bar_reason=None,
                resolution_method="descent_walk",
                confidence="high",
                citation="ἅρπυια",
                etymon_id=42,
            )
        return SimpleNamespace(
            usable=False,
            bar_reason="modern_coinage",
            resolution_method="llm_full_research",
            confidence="low",
            citation=None,
            etymon_id=None,
        )

    writes: list[str] = []
    monkeypatch.setattr(fp, "resolve", _fake_resolve)
    monkeypatch.setattr(fp, "write_resolution", lambda conn, **kw: writes.append(kw["input_name"]))
    monkeypatch.setattr(fp, "tag_etymon_as_fantasy", lambda conn, eid: None)
    monkeypatch.setattr(fp, "fetch_resolved_input_names", lambda conn: [])
    return writes


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "lex.db"
    init_schema(db)
    return db


def test_mine_fantasy_single_dry_run(tmp_path: Path, _stub_pipeline) -> None:
    db = _db(tmp_path)
    result = CliRunner().invoke(
        lexicon_mine_fantasy_name,
        ["--db", str(db), "--name", "Harpy", "--description", "bird-woman"],
    )
    assert result.exit_code == 0
    assert "USABLE  Harpy" in result.output
    assert "'usable': 1, 'barred': 0" in result.output
    assert "(dry-run; pass --apply to write)" in result.output
    assert _stub_pipeline == []  # dry-run writes nothing


def test_mine_fantasy_batch_routes_each_line(tmp_path: Path, _stub_pipeline) -> None:
    db = _db(tmp_path)
    batch = tmp_path / "names.jsonl"
    batch.write_text(
        '{"name": "Harpy", "description": "bird-woman"}\n'
        '{"name": "Gnoll", "description": "hyena-folk"}\n'
    )
    result = CliRunner().invoke(lexicon_mine_fantasy_name, ["--db", str(db), "--batch", str(batch)])
    assert result.exit_code == 0
    assert "USABLE  Harpy" in result.output
    assert "BARRED  Gnoll" in result.output
    assert "'usable': 1, 'barred': 1" in result.output


def test_mine_fantasy_apply_writes_each_resolution(tmp_path: Path, _stub_pipeline) -> None:
    db = _db(tmp_path)
    result = CliRunner().invoke(
        lexicon_mine_fantasy_name,
        ["--db", str(db), "--name", "Gnoll", "--description", "x", "--apply"],
    )
    assert result.exit_code == 0
    assert _stub_pipeline == ["Gnoll"]  # --apply routes through write_resolution


def test_mine_fantasy_batch_and_name_mutually_exclusive(tmp_path: Path, _stub_pipeline) -> None:
    db = _db(tmp_path)
    batch = tmp_path / "names.jsonl"
    batch.write_text('{"name": "Harpy", "description": "x"}\n')
    result = CliRunner().invoke(
        lexicon_mine_fantasy_name, ["--db", str(db), "--name", "X", "--batch", str(batch)]
    )
    assert result.exit_code != 0
    assert "--batch is mutually exclusive" in result.output


def test_mine_fantasy_batch_missing_field_errors(tmp_path: Path, _stub_pipeline) -> None:
    db = _db(tmp_path)
    batch = tmp_path / "names.jsonl"
    batch.write_text('{"name": "Harpy"}\n')  # missing description
    result = CliRunner().invoke(lexicon_mine_fantasy_name, ["--db", str(db), "--batch", str(batch)])
    assert result.exit_code != 0
    assert "line 1: missing 'name' or 'description'" in result.output
