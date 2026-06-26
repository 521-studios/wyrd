"""``wyrd kenning lexicon audit-cluster-reflexes`` — the working era-grid
over-merge fix (wyrd-1wv2).

Judges each modern-english reflex shown on a cluster's era-grid against the
cluster's dominant glossed sense; for an implausible one (``vulva`` on the
'valley' grid) it appends an edge-DETACH to ``_collapses.jsonl`` that orphans
that modern LEAF (cuts its in-cluster bridging parents). At rebuild
``apply_collapses`` deletes the edges + the fresh ``cluster_cognates`` drops the
orphan; on a live deploy run ``clear-enrichment --stage=cognates`` + cluster
again (the incremental path leaves a stale cognate_id, wyrd-hn03). Verdicts log
to ``_cluster_reflex_audit.jsonl``; the LLM is never re-run at rebuild. A gloss
pre-screen auto-keeps on-sense reflexes, so only the suspect/glossless ones are
judged.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click

from wyrd.generators.kenning.cli.lexicon._judge_retry import judge_with_retry

_AUDIT_SOURCE_ROW = {
    "_type": "source",
    "ref": "cluster-reflex-audit",
    "title": "LLM cluster modern-reflex sense-audit verdict log (wyrd-1wv2)",
    "notes": (
        "One row per judged modern reflex (keep | detach). Idempotency + audit "
        "trail only — applied detaches live in _collapses.jsonl. Not replayed at rebuild."
    ),
}
_COLLAPSE_SOURCE_ROW = {
    "_type": "source",
    "ref": "collapse",
    "title": "Etymon collapse ledger (wyrd-y651)",
    "notes": "Folds + descent-edge detaches. Replayed by run_full_enrichment's collapse pass.",
}


def _ensure_source(path: Path, source_row: dict) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(source_row, ensure_ascii=False) + "\n")


def _load_log(audit_file: Path) -> dict[str, dict]:
    """Recorded raw verdicts, ``reflex_ref -> row`` (last wins)."""
    log: dict[str, dict] = {}
    if not audit_file.exists():
        return log
    with audit_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a half-written/corrupt ledger line
            if (
                isinstance(row, dict)
                and row.get("_type") == "cluster_reflex_audit"
                and row.get("reflex_ref")
                and isinstance(row.get("reflex_of_sense"), bool)
            ):
                log[row["reflex_ref"]] = row
    return log


def _judge_with_retry(client, c):
    from wyrd.generators.kenning.lexicon.cluster_reflex_audit import (
        build_judge_prompt,
        parse_verdict,
    )

    return judge_with_retry(
        client, c, build_prompt=build_judge_prompt, parse_verdict=parse_verdict, ref=c.reflex_ref
    )


def _append(fh, row: dict) -> None:
    """Append one JSONL row and flush (crash-safe). No-op when fh is None (dry-run)."""
    if fh is not None:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def _judge_fresh(client, to_judge, log, audit_fh, base_url, model) -> tuple[int, int]:
    from wyrd.generators.kenning.lexicon.cluster_reflex_audit import audit_log_row

    judged = skipped = consecutive = 0
    start = time.perf_counter()
    for i, c in enumerate(to_judge, 1):
        v = _judge_with_retry(client, c)
        if v is None:
            skipped += 1
            consecutive += 1
            if consecutive >= 5:
                raise click.ClickException(
                    f"Aborting: 5 consecutive judge failures — is Ollama up at {base_url} "
                    f"with model {model}? (curl .../api/tags)"
                )
            continue
        consecutive = 0
        judged += 1
        row = audit_log_row(c, v)
        log[c.reflex_ref] = row
        _append(audit_fh, row)
        if i % 10 == 0 or i == len(to_judge):
            rate = (time.perf_counter() - start) / i
            click.echo(
                f"  judged [{i}/{len(to_judge)}] new={judged} skipped={skipped} ({rate:.1f}s/entry)",
                err=True,
            )
    return judged, skipped


def _emit_detaches(log, min_confidence, collapse_state, existing_pairs, collapse_fh, dry_run):
    """Emit one merged detach row per implausible reflex (preserving any existing
    collapse payload for that ref), deduped against detaches already present."""
    from wyrd.generators.kenning.lexicon.cluster_reflex_audit import detach_row

    counts = {"detaches": 0, "kept": 0, "already": 0}
    for ref, row in sorted(log.items()):
        d = detach_row(row, min_confidence)
        if d is None:
            counts["kept"] += 1
            continue
        new_parents = [p for p in d["detach_parents"] if (ref, p) not in existing_pairs]
        if not new_parents:
            counts["already"] += 1
            continue
        counts["detaches"] += 1
        existing = collapse_state.get(ref)
        if existing:
            merged = dict(existing)
            merged["_type"] = "collapse"
            merged["ref"] = ref
            merged["detach_parents"] = sorted(
                set(merged.get("detach_parents") or []) | set(d["detach_parents"])
            )
            out_row = merged
        else:
            out_row = d
        for p in new_parents:
            existing_pairs.add((ref, p))
        if dry_run:
            click.echo(f"  [DETACH] {ref}  detach_parents={out_row['detach_parents']}", err=True)
        else:
            _append(collapse_fh, out_row)
    return counts


@click.command("audit-cluster-reflexes")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Lexicon DB (default: $WYRD_LEXICON_DB or ~/.wyrd/lexicon.db).",
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
    default=Path("data/mining/_cluster_reflex_audit.jsonl"),
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
    help="Minimum confidence to DETACH an implausible reflex.",
)
@click.option("--limit", type=int, default=None, help="Cap reflexes judged this run.")
@click.option("--limit-clusters", type=int, default=None, help="Cap clusters screened.")
@click.option(
    "--broad",
    is_flag=True,
    default=False,
    help="Judge EVERY modern member (no gloss-token auto-keep) — catches noise "
    "over-grouped into the dominant sense via an incidental shared token.",
)
@click.option("--dry-run", is_flag=True, default=False, help="Judge + print; write nothing.")
def lexicon_audit_cluster_reflexes(
    db_path,
    collapse_file,
    audit_file,
    model,
    ollama_url,
    min_confidence,
    limit,
    limit_clusters,
    broad,
    dry_run,
):
    """LLM-audit modern reflexes vs their cluster's dominant sense and detach the
    implausible ones (wyrd-1wv2) — the working era-grid over-merge fix."""
    from wyrd.generators.kenning.cli.utils import _readonly_lexicon
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.jsonl.build import collect_collapses
    from wyrd.generators.kenning.lexicon.cluster_reflex_audit import (
        detect_cluster_reflex_candidates,
    )

    if db_path is None:
        env = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env) if env else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.is_file():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")
    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")

    click.echo(f"Detecting cluster-reflex candidates in {db_path}...", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_cluster_reflex_candidates(
            conn, limit_clusters=limit_clusters, prescreen=not broad
        )

    log = _load_log(audit_file)
    collapse_state = collect_collapses([collapse_file]) if collapse_file.exists() else {}
    existing_pairs = {
        (ref, p) for ref, r in collapse_state.items() for p in r.get("detach_parents") or []
    }
    to_judge = [c for c in candidates if c.reflex_ref not in log]
    if limit is not None:
        to_judge = to_judge[:limit]
    click.echo(
        f"  candidates={len(candidates)} recorded={len(log)} to-judge={len(to_judge)} "
        f"(model={model}, min-confidence={min_confidence})",
        err=True,
    )

    client = OllamaClient(base_url=base_url, model=model, timeout_s=90.0)
    if not dry_run:
        _ensure_source(audit_file, _AUDIT_SOURCE_ROW)
        _ensure_source(collapse_file, _COLLAPSE_SOURCE_ROW)
    audit_fh = collapse_fh = None
    try:
        if not dry_run:
            audit_fh = audit_file.open("a", encoding="utf-8")
            collapse_fh = collapse_file.open("a", encoding="utf-8")
        judged, skipped = _judge_fresh(client, to_judge, log, audit_fh, base_url, model)
        counts = _emit_detaches(
            log, min_confidence, collapse_state, existing_pairs, collapse_fh, dry_run
        )
    finally:
        for fh in (audit_fh, collapse_fh):
            if fh is not None:
                fh.close()

    verb = "would detach" if dry_run else "detached"
    click.echo(
        f"judged {judged} new ({skipped} skipped); {verb}: {counts['detaches']} reflexes, "
        f"kept {counts['kept']}, already-detached {counts['already']}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    parent.add_command(lexicon_audit_cluster_reflexes)
