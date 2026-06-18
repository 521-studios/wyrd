"""End-to-end CLI test for mine-cognate-descents-llm (wyrd-zrce.2) with a FAKE Ollama
client — no live LLM. Proves the two-pass judge → audit-log → emit-descends-from
wiring + resumability + the dry-run-emits-from-log path.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from wyrd.generators.kenning.canonicalization import load_assertions
from wyrd.generators.kenning.cli.lexicon import mine_cognate_descents_llm as cli_mod
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


class _FakeClient:
    """Stands in for OllamaClient. Returns ``propose`` for the propose pass and
    ``verify`` for the refute pass (distinguished by the JSON schema the SYSTEM prompt
    requests — stable, not prose-coupled). ``raises`` makes every call fail (abort /
    retry-fallback paths). Defaults confirm the link at medium."""

    def __init__(self, *, propose=None, verify=None, raises=False, **_kw):
        self.propose = propose or {"choice": 1, "confidence": "medium", "reason": "match"}
        self.verify = verify or {"confirmed": True, "confidence": "medium", "reason": "cognate"}
        self.raises = raises
        self.calls = 0

    def chat_json(self, system, user, _schema):
        self.calls += 1
        if self.raises:
            raise RuntimeError("ollama down")
        return dict(self.propose) if '"choice"' in system else dict(self.verify)


def _patch(monkeypatch, **cfg):
    """Monkeypatch the CLI's OllamaClient with a configured fake (factory keeps the
    config while still accepting the CLI's base_url/model/timeout_s kwargs)."""
    monkeypatch.setattr(cli_mod, "OllamaClient", lambda **_kw: _FakeClient(**cfg))


def _seed_db(path):
    init_schema(path)
    db = LexiconDB(path)
    db.conn.execute("PRAGMA foreign_keys = OFF")
    root = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('stanaz', 'proto-germanic')"
    ).lastrowid
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, root))
    bridge = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('stane', 'old-norse')"
    ).lastrowid
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, bridge))
    db.conn.execute("INSERT INTO etymon_gloss VALUES (?, 'stone', NULL)", (bridge,))
    x = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('stant', 'old-english')"
    ).lastrowid
    db.conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
        "VALUES (?, 0, ?)",
        (x, x),
    )
    db.conn.execute("INSERT INTO etymon_gloss VALUES (?, 'stony place', NULL)", (x,))
    db.commit()
    db.close()
    return root, x


def test_cli_judges_logs_and_emits(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "OllamaClient", _FakeClient)
    db_path = tmp_path / "lexicon.db"
    mining = tmp_path / "mining"
    mining.mkdir()
    root, x = _seed_db(db_path)

    result = CliRunner().invoke(
        cli_mod.lexicon_mine_cognate_descents_llm,
        ["--db", str(db_path), "--mining-dir", str(mining), "--apply"],
    )
    assert result.exit_code == 0, result.output

    # Audit log recorded the confirmed judgment.
    log_lines = (mining / "_cognate_descent_audit.jsonl").read_text().strip().splitlines()
    rows = [
        json.loads(ln) for ln in log_lines if json.loads(ln).get("_type") == "cognate_descent_audit"
    ]
    assert len(rows) == 1 and rows[0]["confirmed"] is True and rows[0]["cluster_root"] == root

    # A descends-from assertion was authored (x -> root).
    assertions = [a for a in load_assertions(mining) if a.predicate == "descends-from"]
    assert len(assertions) == 1
    assert assertions[0].subject.ref == str(x) and assertions[0].object.ref == str(root)
    assert assertions[0].confidence == "medium"


def test_cli_resumes_and_dry_run_emits_from_log(tmp_path, monkeypatch):
    # First --apply run judges + logs. A second run finds nothing fresh (resumed), and
    # a dry-run still re-derives edges from the log without any LLM call.
    monkeypatch.setattr(cli_mod, "OllamaClient", _FakeClient)
    db_path = tmp_path / "lexicon.db"
    mining = tmp_path / "mining"
    mining.mkdir()
    _seed_db(db_path)
    runner = CliRunner()
    base = ["--db", str(db_path), "--mining-dir", str(mining)]

    first = runner.invoke(cli_mod.lexicon_mine_cognate_descents_llm, [*base, "--apply"])
    assert first.exit_code == 0
    log_after_first = (mining / "_cognate_descent_audit.jsonl").read_text()

    # Dry-run: no LLM, no new judgments, but reports edges derivable from the log.
    dry = runner.invoke(cli_mod.lexicon_mine_cognate_descents_llm, base)
    assert dry.exit_code == 0
    assert "dry-run" in dry.output
    assert "1 fresh" not in dry.output  # already judged → resumed
    assert (mining / "_cognate_descent_audit.jsonl").read_text() == log_after_first  # unchanged


def _seed_multi(path, n):
    """One cluster (root + glossed bridge 'stane') + ``n`` residue morphemes sharing
    the 'stan' fold-prefix, each glossed + a breakdown element."""
    init_schema(path)
    db = LexiconDB(path)
    db.conn.execute("PRAGMA foreign_keys = OFF")
    root = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('stanaz', 'proto-germanic')"
    ).lastrowid
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, root))
    b = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('stane', 'old-norse')"
    ).lastrowid
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, b))
    db.conn.execute("INSERT INTO etymon_gloss VALUES (?, 'stone', NULL)", (b,))
    ids = []
    for i in range(n):
        x = db.conn.execute(
            "INSERT INTO etymon (canonical_form, language) VALUES (?, 'old-english')",
            (f"stant{i}",),
        ).lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
            "VALUES (?, 0, ?)",
            (x, x),
        )
        db.conn.execute("INSERT INTO etymon_gloss VALUES (?, 'stony place', NULL)", (x,))
        ids.append(x)
    db.commit()
    db.close()
    return root, ids


def _run(monkeypatch, tmp_path, args, **cfg):
    _patch(monkeypatch, **cfg)
    mining = tmp_path / "mining"
    mining.mkdir(exist_ok=True)
    return CliRunner().invoke(
        cli_mod.lexicon_mine_cognate_descents_llm,
        ["--db", str(tmp_path / "lexicon.db"), "--mining-dir", str(mining), *args],
    ), mining


def test_cli_aborts_after_consecutive_failures(tmp_path, monkeypatch):
    _seed_multi(tmp_path / "lexicon.db", 6)
    result, _ = _run(monkeypatch, tmp_path, ["--apply"], raises=True)
    assert result.exit_code != 0
    assert "Aborting" in result.output and "consecutive" in result.output


def test_cli_refuted_writes_no_assertion(tmp_path, monkeypatch):
    root, _ = _seed_multi(tmp_path / "lexicon.db", 1)
    result, mining = _run(
        monkeypatch,
        tmp_path,
        ["--apply"],
        verify={"confirmed": False, "confidence": "high", "reason": "look-alike"},
    )
    assert result.exit_code == 0
    rows = [
        json.loads(ln)
        for ln in (mining / "_cognate_descent_audit.jsonl").read_text().splitlines()
        if json.loads(ln).get("_type") == "cognate_descent_audit"
    ]
    assert len(rows) == 1 and rows[0]["confirmed"] is False
    assert [a for a in load_assertions(mining) if a.predicate == "descends-from"] == []


def test_cli_dedup_on_second_apply(tmp_path, monkeypatch):
    _seed_multi(tmp_path / "lexicon.db", 1)
    first, mining = _run(monkeypatch, tmp_path, ["--apply"])
    assert first.exit_code == 0
    n_after_first = len([a for a in load_assertions(mining) if a.predicate == "descends-from"])
    second, _ = _run(monkeypatch, tmp_path, ["--apply"])
    assert "wrote 0 descends-from" in second.output  # idempotent: nothing new
    assert (
        len([a for a in load_assertions(mining) if a.predicate == "descends-from"]) == n_after_first
    )


def test_cli_min_confidence_high_emits_no_edge(tmp_path, monkeypatch):
    # Confirmed at medium (LLM cap) never clears a high gate.
    _seed_multi(tmp_path / "lexicon.db", 1)
    result, mining = _run(
        monkeypatch,
        tmp_path,
        ["--apply", "--min-confidence", "high"],
        propose={"choice": 1, "confidence": "high", "reason": "m"},
        verify={"confirmed": True, "confidence": "high", "reason": "v"},
    )
    assert result.exit_code == 0
    assert [a for a in load_assertions(mining) if a.predicate == "descends-from"] == []


def test_cli_max_entries_caps_fresh_judged(tmp_path, monkeypatch):
    _seed_multi(tmp_path / "lexicon.db", 3)
    result, mining = _run(monkeypatch, tmp_path, ["--apply", "--max-entries", "1"])
    assert result.exit_code == 0
    rows = [
        json.loads(ln)
        for ln in (mining / "_cognate_descent_audit.jsonl").read_text().splitlines()
        if json.loads(ln).get("_type") == "cognate_descent_audit"
    ]
    assert len(rows) == 1  # only 1 of 3 judged this run


def test_cli_skips_corrupt_log_line(tmp_path, monkeypatch):
    root, ids = _seed_multi(tmp_path / "lexicon.db", 1)
    mining = tmp_path / "mining"
    mining.mkdir()
    # Pre-seed the log with a corrupt line + one valid confirmed row.
    valid = {
        "_type": "cognate_descent_audit",
        "morpheme_ref": str(ids[0]),
        "morpheme_form": "stant0",
        "cluster_root": root,
        "bridge_form": "stane",
        "proposal_confidence": "medium",
        "confirmed": True,
        "verify_confidence": "medium",
        "verify_reason": "ok",
    }
    (mining / "_cognate_descent_audit.jsonl").write_text(
        "{ not valid json\n" + json.dumps(valid) + "\n"
    )
    _patch(monkeypatch)
    dry = CliRunner().invoke(
        cli_mod.lexicon_mine_cognate_descents_llm,
        ["--db", str(tmp_path / "lexicon.db"), "--mining-dir", str(mining)],
    )
    assert dry.exit_code == 0  # corrupt line skipped, not crashed
    assert "1 edges >= medium" in dry.output  # the valid row still derives an edge
