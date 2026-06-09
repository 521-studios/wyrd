"""``wyrd kenning lexicon audit-reflexes`` — LLM audit of reflex_etymon links,
pruning bad town-name / fragment links from suffix morphemes (wyrd-862b).

The era-grid town-name pollution (``-ham`` showing Burnham/Belchamp, ``-ton``
showing ~120 ``-Xton`` towns) comes from reflex links where a whole place-name
or a neighbor fragment was recorded as a SPELLING of the tail morpheme. A
deterministic pre-screen keeps links whose surface IS a known form of the
etymon (canonical / spelling-variant / cognate-cluster mate / descent reflex);
the residual is LLM-judged (Gemini Flash). Verdicts log to
``_reflex_audit.jsonl`` (idempotent re-runs + audit trail); ``--apply`` deletes
the PRUNE links from the DB and re-dumps ``_reflexes.jsonl`` (the L2 reflex
source) so a rebuild reproduces the clean layer. The LLM is never re-run at
rebuild — the prune round-trips through ``_reflexes.jsonl``.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning.lexicon.reflex_audit import (
    SYSTEM_PROMPT,
    VERDICT_SCHEMA,
    apply_prune,
    load_candidates,
    reflex_audit_user_prompt,
)

_AUDIT_FILENAME = "_reflex_audit.jsonl"


def _load_done(ledger: Path) -> dict[tuple, dict]:
    done: dict[tuple, dict] = {}
    if not ledger.exists():
        return done
    with ledger.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # A partial/truncated final line (interrupted write) shouldn't
                # crash resume/apply — skip it; its key re-judges next run.
                continue
            if r.get("valid") is None:  # error rows re-judge on the next run
                continue
            # Skip a malformed/incomplete row (manual edit, interrupted write)
            # rather than KeyError-ing on a missing key.
            if not all(k in r for k in ("surface_strip", "canon", "gloss")):
                continue
            done[(r["surface_strip"], r["canon"], r["gloss"])] = r
    return done


def _run_apply_prune(db_file: Path, ledger: Path, jsonl_dir: str) -> None:
    """``--apply`` path: prune the reflex_etymon links whose ledger verdict is
    PRUNE (``valid is False``) and re-dump _reflexes.jsonl. Error rows
    (``valid is None``) re-judge on a future run, so they aren't pruned."""
    done = _load_done(ledger)
    prune_keys = {k for k, r in done.items() if r.get("valid") is False}
    click.echo(f"PRUNE verdicts: {len(prune_keys)}", err=True)
    n, rows = apply_prune(db_file, prune_keys, jsonl_dir)
    click.echo(
        f"deleted {n} reflex_etymon links; re-dumped _reflexes.jsonl ({rows} rows)", err=True
    )


def _select_pending(
    db_file: Path, ledger: Path, limit: int | None
) -> tuple[list, dict, dict, list]:
    """Load suspicious reflex candidates, dedup to one judgment per key, and
    drop the already-judged (from the ledger). Returns ``(cands, by_key, done,
    pending)`` — pending capped to ``limit`` when set."""
    cands = load_candidates(db_file)
    by_key: dict[tuple, list] = {}
    for c in cands:
        by_key.setdefault(c.key, []).append(c)
    done = _load_done(ledger)
    pending = [(k, cs[0]) for k, cs in by_key.items() if k not in done]
    if limit is not None:
        pending = pending[:limit]
    return cands, by_key, done, pending


@dataclass
class _AuditJudgeContext:
    """Shared state threaded into the per-candidate judge worker: the LLM
    client, the by-key candidate index (for n_links), the open ledger handle +
    its lock, a mutable ``{"n": int}`` counter, the start time, and the total
    pending count (for the progress denominator)."""

    client: Any
    by_key: dict[tuple, list]
    out: Any
    lock: threading.Lock
    counter: dict[str, int]
    start: float
    total: int


def _judge_one(ctx: _AuditJudgeContext, item: tuple) -> None:
    """Judge one suspicious reflex link via the LLM and append a verdict row to
    the ledger. A non-dict / schema-violating response (or any exception) is
    recorded as an error row (``valid=None``, re-judged next run), never a
    silent prune. Ledger write is locked; the every-10 progress echo is outside
    the lock so terminal I/O doesn't serialize the workers."""
    key, c = item
    try:
        v = ctx.client.chat_json(
            SYSTEM_PROMPT,
            reflex_audit_user_prompt(c.canonical, c.language, c.gloss, c.surface),
            VERDICT_SCHEMA,
        )
        if not isinstance(v, dict) or "valid" not in v:
            # responseSchema marks `valid` required, so this is rare — record
            # as an error row, NOT a silent prune. valid=None re-judges next run.
            raise ValueError(f"bad response: {v!r}"[:200])
        valid, reason, err = bool(v["valid"]), str(v.get("reason", ""))[:300], None
    except Exception as e:  # noqa: BLE001 — record + continue, never abort the batch
        valid, reason, err = None, "", str(e)[:200]
    row = {
        "surface": c.surface,
        "surface_strip": key[0],
        "canon": key[1],
        "canon_raw": c.canonical,
        "lang": c.language,
        "gloss": c.gloss,
        "n_links": len(ctx.by_key[key]),
        "valid": valid,
        "reason": reason,
    }
    if err:
        row["error"] = err
    with ctx.lock:
        ctx.out.write(json.dumps(row, ensure_ascii=False) + "\n")
        ctx.out.flush()
        ctx.counter["n"] += 1
        n = ctx.counter["n"]
    # Progress echo OUTSIDE the lock — terminal I/O shouldn't serialize the
    # worker threads (Gemini review).
    if n % 10 == 0:
        rate = (time.time() - ctx.start) / n
        click.echo(
            f"  [{n}/{ctx.total}] {c.surface!r}->{c.canonical!r} valid={valid} "
            f"({rate:.2f}s/judgment)",
            err=True,
        )


@click.command("audit-reflexes")
@click.option("--db", "db_path", type=click.Path(), default=None, help="Lexicon DB path.")
@click.option(
    "--jsonl-dir",
    type=click.Path(),
    default="data/mining",
    help="L2 dir holding _reflex_audit.jsonl + _reflexes.jsonl.",
)
@click.option("--limit", type=int, default=None, help="Cap unique judgments this run.")
@click.option("--concurrency", type=int, default=8, help="Parallel Gemini calls.")
@click.option(
    "--apply", "do_apply", is_flag=True, help="Prune PRUNE-verdict links + re-dump _reflexes.jsonl."
)
def lexicon_audit_reflexes(db_path, jsonl_dir, limit, concurrency, do_apply):
    import os

    from wyrd.generators.kenning.extractors.gemini import GeminiClient

    if db_path:
        db_file = Path(db_path)
    else:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_file = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_file.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_file}")
    ledger = Path(jsonl_dir) / _AUDIT_FILENAME

    if do_apply:
        _run_apply_prune(db_file, ledger, jsonl_dir)
        return

    cands, by_key, done, pending = _select_pending(db_file, ledger, limit)
    click.echo(
        f"suspicious links: {len(cands)}  unique judgments: {len(by_key)}  "
        f"done: {len(done)}  pending: {len(pending)}",
        err=True,
    )
    if not pending:
        return

    out = ledger.open("a", encoding="utf-8")
    start = time.time()
    ctx = _AuditJudgeContext(
        client=GeminiClient(),
        by_key=by_key,
        out=out,
        lock=threading.Lock(),
        counter={"n": 0},
        start=start,
        total=len(pending),
    )
    # try/finally so the ledger handle is flushed/closed even if the pool
    # raises or the run is interrupted (Gemini's leak-on-exception finding).
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(ex.map(lambda it: _judge_one(ctx, it), pending))
    finally:
        out.close()
    click.echo(
        f"done: {ctx.counter['n']} judgments in {time.time() - start:.0f}s -> {ledger}", err=True
    )


def add_to(parent: click.Group) -> None:
    """Register ``audit-reflexes`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_audit_reflexes)
