"""``wyrd kenning lexicon audit-descent`` — LLM audit of cognate-cluster descent
edges, detaching the bad ones that over-merge distinct words (wyrd-rd69).

Phase 2 of the era-grid over-merge cleanup. ``cluster_cognates`` merges via the
``etymon_descent`` graph; over-broad proto-form fan-out + spurious edges pull
unrelated words into one cluster (``val``/``vale`` clustered with ``vulva``),
polluting the era-grid. This screens bridge edges in incoherent clusters, judges
each with an LLM, and appends an edge-DETACH to ``_collapses.jsonl`` for the
wrong ones (applied by ``apply_collapses``; ``cluster_cognates`` re-derives clean
clusters). Verdicts log to ``_descent_audit.jsonl`` for idempotent re-runs; the
LLM is NEVER re-run at rebuild — the detaches round-trip through L2.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click

from wyrd.generators.kenning.cli.lexicon._judge_retry import judge_with_retry

_DESCENT_AUDIT_SOURCE_ROW = {
    "_type": "source",
    "ref": "descent-audit",
    "title": "LLM descent-edge audit verdict log (wyrd-rd69)",
    "notes": (
        "One row per judged descent edge (keep | detach). Idempotency + audit "
        "trail only — the applied detaches live in _collapses.jsonl. Not replayed "
        "at rebuild."
    ),
}

_COLLAPSE_SOURCE_ROW = {
    "_type": "source",
    "ref": "collapse",
    "title": "Etymon collapse ledger (wyrd-y651)",
    "notes": (
        "Form-of / variant etymons folded into their lemma + descent-edge "
        "detaches. Replayed by run_full_enrichment's collapse pass."
    ),
}


def _ensure_source(path: Path, source_row: dict) -> None:
    """Create ``path`` with its synthetic source row if absent."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(source_row, ensure_ascii=False) + "\n")


def _load_log(audit_file: Path) -> dict[str, dict]:
    """Recorded raw verdicts, ``edge_key -> row`` (last wins). Lets a re-run reuse
    prior judgments and re-derive detaches at a new ``--min-confidence``."""
    log: dict[str, dict] = {}
    if not audit_file.exists():
        return log
    with audit_file.open("r", encoding="utf-8") as fh:  # stream — the log can grow large
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a half-written/corrupt ledger line (e.g. crash mid-append)
            if (
                isinstance(row, dict)
                and row.get("_type") == "descent_audit"
                and row.get("edge_key")
                and isinstance(row.get("same_family"), bool)
            ):
                log[row["edge_key"]] = row
    return log


def _existing_detach_pairs(collapse_state: dict[str, dict]) -> set[tuple[str, str]]:
    """``(child_ref, parent_ref)`` pairs already detached in the NET collapse
    state, so a re-run at a lower threshold doesn't re-append. Derived from the
    last-write-wins ``collect_collapses`` state (NOT a raw file scan), so a detach
    that a later row dropped is correctly NOT counted as still-present."""
    return {
        (ref, parent)
        for ref, row in collapse_state.items()
        for parent in row.get("detach_parents") or []
    }


def _judge_with_retry(client, c):
    """Judge one candidate, one retry on transport/parse failure; None if
    unusable (caller skips WITHOUT recording, so a re-run retries). Echoes the
    last failure to stderr so a persistent Ollama outage is diagnosable rather
    than presenting as a silently-climbing skip count."""
    from wyrd.generators.kenning.lexicon.descent_audit import (
        build_descent_judge_prompt,
        parse_descent_verdict,
    )

    return judge_with_retry(
        client,
        c,
        build_prompt=build_descent_judge_prompt,
        parse_verdict=parse_descent_verdict,
        ref=c.edge_key,
    )


def _append(fh, row: dict) -> None:
    """Append one JSONL row and flush (crash-safe as we go). No-op when fh is
    None (dry-run)."""
    if fh is not None:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def _judge_fresh(client, to_judge, log, audit_fh, base_url, model) -> tuple[int, int]:
    """PASS 1 — LLM-judge edges not already in the log; record each raw verdict.
    Aborts after 5 consecutive failures (Ollama down). Returns (judged, skipped)."""
    from wyrd.generators.kenning.lexicon.descent_audit import audit_log_row

    judged_new = skipped = consecutive = 0
    start = time.perf_counter()
    for i, c in enumerate(to_judge, 1):
        verdict = _judge_with_retry(client, c)
        if verdict is None:
            skipped += 1
            consecutive += 1
            if consecutive >= 5:
                # Fail loud + non-zero (Ollama down) — judged verdicts so far are
                # flushed to the log, so a re-run resumes; don't exit 0 on a
                # half-done batch (silent failure in automation).
                raise click.ClickException(
                    f"Aborting: 5 consecutive judge failures — is Ollama up at {base_url} "
                    f"with model {model}? (curl .../api/tags)"
                )
            continue
        consecutive = 0
        judged_new += 1
        row = audit_log_row(c, verdict)
        log[c.edge_key] = row
        _append(audit_fh, row)
        if i % 10 == 0 or i == len(to_judge):
            rate = (time.perf_counter() - start) / i
            click.echo(
                f"  judged [{i}/{len(to_judge)}] new={judged_new} skipped={skipped} "
                f"({rate:.1f}s/entry)",
                err=True,
            )
    return judged_new, skipped


def _merged_detach_row(child_ref: str, parents: set[str], collapse_state: dict[str, dict]) -> dict:
    """ONE ``_collapses.jsonl`` collapse row detaching ALL ``parents`` from
    ``child_ref``. ``collapse`` rows are last-write-wins per ``ref``, so every
    detach for a child MUST live in a single row, and that row must PRESERVE any
    existing fold (``into`` / ``detach_children``) for the child — else the
    replayed net state would drop the earlier detaches or clobber the fold
    (wyrd-rd69 reconstructibility). Unions the new parents into the child's
    existing net collapse payload (from ``collect_collapses``) if present."""
    from wyrd.generators.kenning.lexicon.descent_audit import LLM_DESCENT_AUDIT_DETACH_METHOD

    existing = collapse_state.get(child_ref)
    if existing:
        row = dict(existing)
        row["_type"] = "collapse"
        row["ref"] = child_ref
        row["detach_parents"] = sorted(set(row.get("detach_parents") or []) | parents)
        return row
    return {
        "_type": "collapse",
        "ref": child_ref,
        "detach_parents": sorted(parents),
        "method": LLM_DESCENT_AUDIT_DETACH_METHOD,
        "reason": "descent-audit over-merge detach (wyrd-rd69); per-edge reasons in _descent_audit.jsonl",
    }


def _emit_detaches(
    log, min_confidence, collapse_state, existing_pairs, collapse_fh, dry_run
) -> dict[str, int]:
    """PASS 2 — derive edge-detaches from recorded verdicts at ``min_confidence``
    (no LLM), GROUPED into one collapse row per child (last-write-wins safe) and
    deduped against ``(child, parent)`` detaches already in ``_collapses.jsonl``."""
    from wyrd.generators.kenning.lexicon.descent_audit import detach_row, verdict_from_log

    counts = {"detaches": 0, "kept": 0, "already": 0}
    new_parents: dict[str, set[str]] = {}
    for _edge_key, row in sorted(log.items()):
        v = verdict_from_log(row)
        if v is None:
            continue
        parent_ref, child_ref = row.get("parent_ref"), row.get("child_ref")
        # detach_row encodes the keep / below-threshold gate (returns None to skip).
        if detach_row(parent_ref, child_ref, v, min_confidence) is None:
            counts["kept"] += 1
            continue
        if (child_ref, parent_ref) in existing_pairs:
            counts["already"] += 1
            continue
        new_parents.setdefault(child_ref, set()).add(parent_ref)
    for child_ref, parents in sorted(new_parents.items()):
        counts["detaches"] += len(parents)
        if dry_run:
            click.echo(f"  [DETACH] {child_ref}  detach_parents={sorted(parents)}", err=True)
        else:
            _append(collapse_fh, _merged_detach_row(child_ref, parents, collapse_state))
    return counts


@click.command("audit-descent")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Lexicon DB to scan (default: $WYRD_LEXICON_DB or ~/.wyrd/lexicon.db).",
)
@click.option(
    "--collapse-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_collapses.jsonl"),
    show_default=True,
    help="Collapse ledger: target of edge-detach rows.",
)
@click.option(
    "--audit-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_descent_audit.jsonl"),
    show_default=True,
    help="Verdict log (idempotency + audit trail).",
)
@click.option(
    "--model",
    default="gemma4:26b",
    show_default=True,
    help="Ollama model. gemma4:26b judges this task best; qwen2.5:7b is a JSON fallback.",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Ollama base URL (default: $WYRD_OLLAMA_URL or http://localhost:11434).",
)
@click.option(
    "--min-confidence",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    show_default=True,
    help="Minimum LLM confidence to actually DETACH a wrong-edge verdict.",
)
@click.option(
    "--scope",
    type=click.Choice(["2a", "2b", "both"]),
    default="both",
    show_default=True,
    help="2a=incoherent-cluster bridges; 2b=glossless-proto-conductor fan-out; both.",
)
@click.option("--limit", type=int, default=None, help="Cap edges judged this run.")
@click.option("--limit-clusters", type=int, default=None, help="Cap clusters screened.")
@click.option("--dry-run", is_flag=True, default=False, help="Judge + print; write nothing.")
def lexicon_audit_descent(
    db_path: Path | None,
    collapse_file: Path,
    audit_file: Path,
    model: str,
    ollama_url: str | None,
    min_confidence: str,
    scope: str,
    limit: int | None,
    limit_clusters: int | None,
    dry_run: bool,
) -> None:
    """LLM-audit cognate-cluster descent edges and detach the over-merging ones
    (wyrd-rd69). Screens bridge edges in incoherent clusters; the LLM arbitrates;
    detaches round-trip through ``_collapses.jsonl`` and replay deterministically
    — never re-run at rebuild."""
    from wyrd.generators.kenning.cli.utils import _readonly_lexicon
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.jsonl.build import collect_collapses
    from wyrd.generators.kenning.lexicon.descent_audit import detect_descent_audit_candidates

    if db_path is None:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.is_file():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")
    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")

    click.echo(f"Detecting descent-audit candidates in {db_path}...", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_descent_audit_candidates(
            conn, scope=scope, limit_clusters=limit_clusters
        )

    log = _load_log(audit_file)
    collapse_state = collect_collapses([collapse_file]) if collapse_file.exists() else {}
    existing = _existing_detach_pairs(collapse_state)
    to_judge = [c for c in candidates if c.edge_key not in log]
    if limit is not None:
        to_judge = to_judge[:limit]
    click.echo(
        f"  candidates={len(candidates)} recorded={len(log)} to-judge={len(to_judge)} "
        f"(model={model}, min-confidence={min_confidence})",
        err=True,
    )

    client = OllamaClient(base_url=base_url, model=model, timeout_s=90.0)
    if not dry_run:
        _ensure_source(audit_file, _DESCENT_AUDIT_SOURCE_ROW)
        _ensure_source(collapse_file, _COLLAPSE_SOURCE_ROW)
    audit_fh = collapse_fh = None
    try:
        if not dry_run:
            audit_fh = audit_file.open("a", encoding="utf-8")
            collapse_fh = collapse_file.open("a", encoding="utf-8")
        judged_new, skipped = _judge_fresh(client, to_judge, log, audit_fh, base_url, model)
        counts = _emit_detaches(log, min_confidence, collapse_state, existing, collapse_fh, dry_run)
    finally:
        for fh in (audit_fh, collapse_fh):
            if fh is not None:
                fh.close()

    verb = "would detach" if dry_run else "detached"
    click.echo(
        f"judged {judged_new} new ({skipped} skipped); {verb}: {counts['detaches']} edges, "
        f"kept {counts['kept']}, already-detached {counts['already']}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``audit-descent`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_audit_descent)
