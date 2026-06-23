"""CliRunner tests for ``wyrd kenning lexicon gloss-campaign`` (wyrd-u6fn.10.1):
the validate/author/park verbs' exit codes, the validate->author refusal gate,
and the ``_load_candidates`` JSON-error path. (The slice/scoreboard logic is unit-
tested in test_kenning_gloss_sense_campaign; this covers the CLI surface.)
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "lexicon.db"
    init_schema(path)
    db = LexiconDB(path)
    eid = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('tūn', 'old-english')"
    ).lastrowid
    for g in ("enclosure", "town"):
        db.conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, g))
    db.commit()
    db.close()
    return path


def _cand_row(glosses: list[str]) -> dict:
    return {
        "_type": "gloss_sense",
        "ref": "old-english:tūn",
        "language": "old-english",
        "canonical_form": "tūn",
        "senses": [{"label": "town", "glosses": glosses}],
        "confidence": "high",
        "method": "opus-gloss-sense-v1",
        "model": "opus",
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def test_cli_validate_ok_then_author_appends(tmp_path):
    db_path = _db(tmp_path)
    mining = tmp_path / "mining"
    mining.mkdir()
    cands = _write(tmp_path / "c.jsonl", [_cand_row(["enclosure", "town"])])
    runner = CliRunner()

    val = runner.invoke(
        cli_root,
        [
            "lexicon",
            "gloss-campaign",
            "validate",
            "--db",
            str(db_path),
            "--mining-dir",
            str(mining),
            "--candidates",
            str(cands),
        ],
    )
    assert val.exit_code == 0, val.output

    auth = runner.invoke(
        cli_root,
        [
            "lexicon",
            "gloss-campaign",
            "author",
            "--db",
            str(db_path),
            "--mining-dir",
            str(mining),
            "--candidates",
            str(cands),
        ],
    )
    assert auth.exit_code == 0, auth.output
    nodes = mining / "canonicalization" / "_canonical_nodes.jsonl"
    assert nodes.is_file() and nodes.read_text().strip()  # assertions were appended


def test_cli_validate_fails_and_author_refuses(tmp_path):
    db_path = _db(tmp_path)
    mining = tmp_path / "mining"
    mining.mkdir()
    # 'enclosure' left unassigned -> incomplete partition -> validate error
    cands = _write(tmp_path / "c.jsonl", [_cand_row(["town"])])
    runner = CliRunner()

    val = runner.invoke(
        cli_root,
        [
            "lexicon",
            "gloss-campaign",
            "validate",
            "--db",
            str(db_path),
            "--mining-dir",
            str(mining),
            "--candidates",
            str(cands),
        ],
    )
    assert val.exit_code == 1
    assert "VALIDATION FAILED" in val.output

    auth = runner.invoke(
        cli_root,
        [
            "lexicon",
            "gloss-campaign",
            "author",
            "--db",
            str(db_path),
            "--mining-dir",
            str(mining),
            "--candidates",
            str(cands),
        ],
    )
    assert auth.exit_code == 1
    assert "REFUSED" in auth.output
    assert not (mining / "canonicalization").exists()  # the gate wrote nothing


def test_cli_author_rejects_invalid_json(tmp_path):
    db_path = _db(tmp_path)
    mining = tmp_path / "mining"
    mining.mkdir()
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not valid json\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "gloss-campaign",
            "author",
            "--db",
            str(db_path),
            "--mining-dir",
            str(mining),
            "--candidates",
            str(bad),
        ],
    )
    assert result.exit_code != 0
    assert "invalid JSON" in result.output  # clean ClickException, not a raw traceback


def test_cli_park_appends_sidecar(tmp_path):
    mining = tmp_path / "mining"
    mining.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "gloss-campaign",
            "park",
            "--mining-dir",
            str(mining),
            "--ref",
            "old-english:lēah",
            "--reason",
            "fragment",
        ],
    )
    assert result.exit_code == 0, result.output
    sidecar = mining / "_sense_parked.jsonl"
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text().strip()) == {
        "ref": "old-english:lēah",
        "reason": "fragment",
    }
