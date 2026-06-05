"""``wyrd kenning lexicon audit-merges`` — LLM audit of merge decisions
(folds / OCR-cluster tombstones / lemma links), reverting wrongly-merged
homographs (wyrd-6hbv).

The era-grid gloss-blobs (``-ton`` showing the weight unit; ``-ing`` carrying
~45 disjoint glosses) come from bad merges. This judges every merge candidate
with an LLM and routes a revert to the provenance-appropriate L2 ledger:
collapse-folds revert in ``_collapses.jsonl`` (``into: ""``); OCR-cluster and
lemma-link merges revert in ``_curation.jsonl`` (``merged_into_ref`` /
``lemma_ref`` cleared). Every verdict is logged to ``_merge_audit.jsonl`` for
idempotent re-runs + an audit trail. The LLM is NEVER re-run at rebuild — its
verdicts round-trip through L2 and replay deterministically.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click

_MERGE_AUDIT_SOURCE_ROW = {
    "_type": "source",
    "ref": "merge-audit",
    "title": "LLM merge-audit verdict log (wyrd-6hbv)",
    "notes": (
        "One row per judged merge decision (keep | revert). Idempotency + audit "
        "trail only — the applied reverts live in _collapses.jsonl (collapse "
        "folds) and _curation.jsonl (OCR / lemma merges). Not replayed at rebuild."
    ),
}

_CURATION_SOURCE_ROW = {
    "_type": "source",
    "ref": "manual-curation",
    "title": "Operator curation overrides (wyrd-2jhs)",
    "notes": (
        "L3 enrichment overrides applied after the auto-clustering passes. Each "
        "event records a deliberate decision; reverting is appending another "
        "event with the cleared value (lemma_ref: '')."
    ),
}


def _ensure_source(path: Path, source_row: dict) -> None:
    """Create ``path`` with its synthetic source row if absent — every L2 JSONL
    file needs exactly one source row."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(source_row, ensure_ascii=False) + "\n")


def _load_judged(audit_file: Path) -> set[str]:
    """member_refs already carrying a verdict in the audit log (idempotency)."""
    judged: set[str] = set()
    if not audit_file.exists():
        return judged
    for line in audit_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("_type") == "merge_audit" and row.get("member_ref"):
            judged.add(row["member_ref"])
    return judged


def _judge_with_retry(client, c):
    """Judge one candidate, one retry on transport/parse failure; None if
    unusable (caller skips WITHOUT recording, so a re-run retries). Echoes the
    last failure to stderr so a persistent Ollama outage is diagnosable rather
    than presenting as a silently-climbing skip count."""
    from wyrd.generators.kenning.lexicon.merge_audit import (
        build_audit_judge_prompt,
        parse_audit_verdict,
    )

    system, user = build_audit_judge_prompt(c)
    last_err = "unparseable verdict"
    for _attempt in range(2):
        try:
            verdict = parse_audit_verdict(client.chat_json(system, user, {}))
            if verdict is not None:
                return verdict
        except Exception as exc:  # transient LLM/transport noise: skip, don't crash the batch
            last_err = str(exc) or exc.__class__.__name__
    click.echo(f"  skip (judge failed for {c.member_ref}): {last_err}", err=True)
    return None


def _write_rows(rows, audit_fh, collapse_fh, curation_fh) -> None:
    """Append a verdict's rows: the audit-log marker always; the revert (if any)
    to its provenance-routed ledger. Flush per row for crash-safety."""
    for fh, row in (
        (audit_fh, rows.audit_log),
        (collapse_fh, rows.collapse),
        (curation_fh, rows.curation),
    ):
        if row is not None:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()


@click.command("audit-merges")
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
    help="Collapse ledger: source of folds to re-audit AND target of fold reverts.",
)
@click.option(
    "--curation-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_curation.jsonl"),
    show_default=True,
    help="Curation ledger: target of OCR-cluster / lemma-link reverts.",
)
@click.option(
    "--audit-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_merge_audit.jsonl"),
    show_default=True,
    help="Verdict log (idempotency + audit trail; every keep/revert recorded).",
)
@click.option(
    "--scope",
    type=click.Choice(["anchors", "collapses", "affixes", "both"]),
    default="both",
    show_default=True,
    help="Candidate sources: disjoint-sense anchors, collapse folds, affix↔bare, or all.",
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
    help="Minimum LLM confidence to actually REVERT a wrong-merge verdict.",
)
@click.option("--limit", type=int, default=None, help="Cap candidates judged this run.")
@click.option("--dry-run", is_flag=True, default=False, help="Judge + print; write nothing.")
def lexicon_audit_merges(
    db_path: Path | None,
    collapse_file: Path,
    curation_file: Path,
    audit_file: Path,
    scope: str,
    model: str,
    ollama_url: str | None,
    min_confidence: str,
    limit: int | None,
    dry_run: bool,
) -> None:
    """LLM-audit merge decisions and revert wrongly-merged homographs (wyrd-6hbv).

    Catches the ``-ton``→``ton`` (OCR) and ``-ing`` gloss-blob (collapse / lemma)
    conflations. There is no trustworthy deterministic screen (gloss-token
    overlap over-splits coherent clusters and conflates same-sense wordings), so
    the LLM is the arbiter; the deterministic pass only screens candidates.
    Verdicts round-trip through L2 and replay deterministically — never re-run at
    rebuild.
    """
    from wyrd.generators.kenning.cli.utils import _readonly_lexicon
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.jsonl.build import collect_collapses
    from wyrd.generators.kenning.lexicon.merge_audit import (
        audit_verdict_to_rows,
        detect_merge_audit_candidates,
    )

    if db_path is None:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")
    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")

    collapse_state = collect_collapses([collapse_file]) if collapse_file.exists() else {}
    click.echo(f"Detecting merge-audit candidates in {db_path}...", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_merge_audit_candidates(conn, collapse_state, scope=scope)

    judged = _load_judged(audit_file)
    fresh = [c for c in candidates if c.member_ref not in judged]
    todo = fresh[:limit] if limit is not None else fresh
    click.echo(
        f"  candidates={len(candidates)} already-judged={len(candidates) - len(fresh)} "
        f"to-judge={len(todo)} (scope={scope}, model={model}, min-confidence={min_confidence})",
        err=True,
    )
    if not todo:
        click.echo("Nothing to judge.", err=True)
        return

    client = OllamaClient(base_url=base_url, model=model, timeout_s=90.0)
    if not dry_run:
        _ensure_source(audit_file, _MERGE_AUDIT_SOURCE_ROW)
        _ensure_source(curation_file, _CURATION_SOURCE_ROW)
    reverts = kept = skipped = 0
    by_prov: dict[str, int] = {"collapse-fold": 0, "ocr-cluster": 0, "lemma-link": 0}
    start = time.perf_counter()
    audit_fh = collapse_fh = curation_fh = None
    try:
        if not dry_run:
            audit_fh = audit_file.open("a", encoding="utf-8")
            collapse_fh = collapse_file.open("a", encoding="utf-8")
            curation_fh = curation_file.open("a", encoding="utf-8")
        for i, c in enumerate(todo, 1):
            verdict = _judge_with_retry(client, c)
            if verdict is None:
                skipped += 1  # transient/unparseable — don't record, so a re-run retries
                continue
            rows = audit_verdict_to_rows(c, verdict, min_confidence)
            is_revert = rows.collapse is not None or rows.curation is not None
            if is_revert:
                reverts += 1
                by_prov[c.provenance] = by_prov.get(c.provenance, 0) + 1
            else:
                kept += 1
            if dry_run:
                tag = "REVERT" if is_revert else "keep"
                click.echo(
                    f"  [{tag}/{c.provenance}] {c.member_ref} -> {c.anchor_ref} "
                    f"{verdict.confidence} ({verdict.reason})",
                    err=True,
                )
            else:
                _write_rows(rows, audit_fh, collapse_fh, curation_fh)
            if i % 10 == 0 or i == len(todo):
                rate = (time.perf_counter() - start) / i
                click.echo(
                    f"  [{i}/{len(todo)}]  reverts={reverts} keep={kept} "
                    f"skipped={skipped} ({rate:.1f}s/entry)",
                    err=True,
                )
    finally:
        for fh in (audit_fh, collapse_fh, curation_fh):
            if fh is not None:
                fh.close()

    verb = "would revert" if dry_run else "reverted"
    click.echo(
        f"{verb}: {reverts} merges "
        f"(collapse={by_prov['collapse-fold']} ocr={by_prov['ocr-cluster']} "
        f"lemma={by_prov['lemma-link']}), kept {kept} ({skipped} skipped)",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``audit-merges`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_audit_merges)
