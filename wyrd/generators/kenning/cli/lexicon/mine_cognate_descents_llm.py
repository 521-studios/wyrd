"""``wyrd kenning lexicon mine-cognate-descents-llm`` — LLM cognate-ancestor proposal
for the residue, adversarially verified (wyrd-zrce.2, uplift 1b.2).

Two passes per residue morpheme (propose → independent refute); only a CONFIRMED
link at/above ``--min-confidence`` becomes a ``descends-from`` assertion (projected +
clustered by zrce.1's wired stage). Every raw judgment is logged to
``_cognate_descent_audit.jsonl`` for idempotent re-runs — the LLM is never re-run at
rebuild; verdicts round-trip through L2.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from wyrd.generators.kenning.canonicalization import append_assertion, load_assertions
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.extractors.llm import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaClient,
)
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.cognate_descent_llm import (
    METHOD,
    build_candidates,
    edge_from_log,
    judge_item,
)
from wyrd.generators.kenning.lexicon.cognate_descent_mining import descent_assertions
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_refs_for
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

_AUDIT_FILE = "_cognate_descent_audit.jsonl"
_SOURCE_ROW = {
    "_type": "source",
    "id": METHOD,
    "title": "LLM cognate-descent proposal audit (wyrd-zrce.2)",
}
_MAX_CONSECUTIVE_FAILS = 5


def _load_log(path: Path) -> dict[str, dict]:
    """Recorded raw judgments, ``morpheme_ref -> row`` (last wins) — lets a re-run
    skip judged morphemes and re-derive edges at a new ``--min-confidence``."""
    log: dict[str, dict] = {}
    if not path.exists():
        return log
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a half-written ledger line (crash mid-append)
            if (
                isinstance(row, dict)
                and row.get("_type") == "cognate_descent_audit"
                and row.get("morpheme_ref")
            ):
                log[row["morpheme_ref"]] = row
    return log


def _make_judge(client: OllamaClient):
    """A ``(system, user) -> dict`` judge with one retry; returns ``{}`` on persistent
    failure (→ parse None → judge_item None → caller skips without recording)."""

    def judge(system: str, user: str) -> dict:
        last = "unparseable"
        for _attempt in range(2):
            try:
                return client.chat_json(system, user, {})
            except Exception as exc:  # transient LLM/transport noise
                last = str(exc) or exc.__class__.__name__
        click.echo(f"  llm call failed: {last}", err=True)
        return {}

    return judge


@click.command("mine-cognate-descents-llm")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--mining-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="L2 mining root; the audit log lives here, descends-from under canonicalization/.",
)
@click.option("--source", default="cognate-descent-uplift", show_default=True)
@click.option("--model", default=DEFAULT_OLLAMA_MODEL, show_default=True)
@click.option("--ollama-url", default=DEFAULT_OLLAMA_URL, show_default=True)
@click.option("--timeout", type=float, default=120.0, show_default=True)
@click.option(
    "--min-confidence",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    show_default=True,
    help="Emit a descends-from edge only when the verified link clears this gate.",
)
@click.option(
    "--max-entries",
    type=int,
    default=0,
    help="Cap fresh morphemes judged this run (0 = no cap). Bounds LLM cost.",
)
@click.option("--apply/--dry-run", default=False, show_default=True)
def lexicon_mine_cognate_descents_llm(
    db_path: Path,
    mining_dir: Path,
    source: str,
    model: str,
    ollama_url: str,
    timeout: float,
    min_confidence: str,
    max_entries: int,
    apply: bool,
) -> None:
    """wyrd-zrce.2 (uplift 1b.2): propose + adversarially verify cognate-descent links
    for the residue (unclustered breakdown morphemes with a clustered stem-mate), then
    author CONFIRMED links as descends-from assertions (Family B). PASS 1 LLM-judges
    fresh morphemes (resumable via the audit log); PASS 2 re-derives edges from the log
    at --min-confidence (no LLM) and appends them to the L2 stream.
    """
    audit_path = mining_dir / _AUDIT_FILE
    log = _load_log(audit_path)
    with LexiconDB(db_path) as db:
        items = [it for it in build_candidates(db) if it.ref not in log]
        if max_entries > 0:
            items = items[:max_entries]
        total_cohort = len(log) + len(items)
        click.echo(
            f"mine-cognate-descents-llm: {total_cohort} residue morphemes with candidates "
            f"({len(log)} already judged, {len(items)} fresh)",
            err=True,
        )

        # PASS 1 — judge fresh morphemes (only when applying; dry-run reports + emits
        # from the existing log without spending LLM calls).
        if apply and items:
            judge = _make_judge(OllamaClient(base_url=ollama_url, model=model, timeout_s=timeout))
            _judge_fresh(items, judge, log, audit_path, ollama_url, model)

    # PASS 2 — derive descends-from from the (updated) log at the gate; author to L2.
    edges = [e for row in log.values() if (e := edge_from_log(row, min_confidence=min_confidence))]
    confirmed = sum(1 for r in log.values() if r.get("confirmed"))
    click.echo(
        f"  log: {len(log)} judged, {confirmed} confirmed, {len(edges)} edges >= {min_confidence}",
        err=True,
    )
    if not apply:
        click.echo("  dry-run — no LLM calls, no edges written (pass --apply).", err=True)
        return

    # wyrd-c6wu: resolve the log's endpoint ids -> stable natural keys against the
    # current DB so the authored L2 assertions are durable. NOTE: the audit log
    # itself still stores endpoint ids (cluster_root/morpheme_ref) — replaying a
    # log mined against a DIFFERENT corpus would resolve stale ids; that verdict-
    # cache durability is deferred (separate follow-up). Fresh runs are correct.
    with LexiconDB(db_path) as db:
        refs = etymon_refs_for(
            db.conn, {e.child_etymon for e in edges} | {e.cluster_root for e in edges}
        )
    assertions = descent_assertions(edges, refs, source=source)
    existing = {a.id for a in load_assertions(mining_dir)}
    written = 0
    for a in assertions:
        if a.id in existing:
            continue
        append_assertion(mining_dir, a)
        written += 1
    click.echo(
        f"  wrote {written} descends-from assertions to {mining_dir}/canonicalization/", err=True
    )


def _judge_fresh(items, judge, log, audit_path, ollama_url, model) -> None:
    """PASS 1: two-pass-judge each fresh morpheme, recording every raw verdict. Aborts
    after consecutive failures (Ollama down) — judged rows are flushed, so re-run resumes."""
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if not audit_path.exists():
        with audit_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(_SOURCE_ROW, ensure_ascii=False) + "\n")
    judged = skipped = consecutive = 0
    start = time.perf_counter()
    with audit_path.open("a", encoding="utf-8") as fh:
        for i, item in enumerate(items, 1):
            row = judge_item(item, judge)
            if row is None:
                skipped += 1
                consecutive += 1
                if consecutive >= _MAX_CONSECUTIVE_FAILS:
                    raise click.ClickException(
                        f"Aborting: {_MAX_CONSECUTIVE_FAILS} consecutive judge failures — is "
                        f"Ollama up at {ollama_url} with model {model}? (curl .../api/tags)"
                    )
            else:
                consecutive = 0
                judged += 1
                log[item.ref] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            # Emit progress on EVERY boundary (incl. a trailing skip) so the final
            # line always prints and a skip-heavy window isn't silent.
            if i % 10 == 0 or i == len(items):
                rate = (time.perf_counter() - start) / i
                confirmed = sum(1 for r in log.values() if r.get("confirmed"))
                click.echo(
                    f"  judged [{i}/{len(items)}] new={judged} confirmed={confirmed} "
                    f"skipped={skipped} ({rate:.1f}s/entry)",
                    err=True,
                )


def add_to(parent: click.Group) -> None:
    """Register ``mine-cognate-descents-llm`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_cognate_descents_llm)
