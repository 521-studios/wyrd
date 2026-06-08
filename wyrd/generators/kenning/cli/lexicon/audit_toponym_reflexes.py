"""``wyrd kenning lexicon audit-toponym-reflexes`` — per-form era-grid cleanup
(wyrd-1wv2).

Judges each distinct modern-english era-grid reflex form GENERICALLY — "is this a
plausible English place-name element?" — and detaches the unrelated ones
(medical Latin, foreign words, proper nouns, function words, abstract nouns)
cluster-merged in. Detaches round-trip via ``_collapses.jsonl``; verdicts log to
``_toponym_reflex_audit.jsonl``. A surface pre-screen auto-keeps forms similar to
their cluster morphemes, so only the suspect/dissimilar forms are judged.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click

_AUDIT_SOURCE_ROW = {
    "_type": "source",
    "ref": "toponym-reflex-audit",
    "title": "LLM per-form toponym-reflex audit verdict log (wyrd-1wv2)",
    "notes": (
        "One row per judged modern form (keep | detach). Idempotency + audit trail "
        "only — applied detaches live in _collapses.jsonl. Not replayed at rebuild."
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
                and row.get("_type") == "toponym_reflex_audit"
                and row.get("reflex_ref")
                and isinstance(row.get("toponymic"), bool)
            ):
                log[row["reflex_ref"]] = row
    return log


def _judge_with_retry(client, c):
    from wyrd.generators.kenning.lexicon.toponym_reflex_audit import (
        build_judge_prompt,
        parse_verdict,
    )

    system, user = build_judge_prompt(c)
    last_err = "unparseable verdict"
    for _attempt in range(2):
        try:
            v = parse_verdict(client.chat_json(system, user, {}))
            if v is not None:
                return v
            last_err = "unparseable verdict"
        except Exception as exc:  # transient LLM/transport noise: skip, don't crash the batch
            last_err = str(exc) or exc.__class__.__name__
    click.echo(f"  skip (judge failed for {c.reflex_ref}): {last_err}", err=True)
    return None


def _append(fh, row: dict) -> None:
    """Append one JSONL row and flush (crash-safe). No-op when fh is None (dry-run)."""
    if fh is not None:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def _judge_fresh(client, to_judge, log, audit_fh, base_url, model) -> tuple[int, int]:
    from wyrd.generators.kenning.lexicon.toponym_reflex_audit import audit_log_row

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
        if i % 25 == 0 or i == len(to_judge):
            rate = (time.perf_counter() - start) / i
            click.echo(
                f"  judged [{i}/{len(to_judge)}] new={judged} skipped={skipped} ({rate:.1f}s/entry)",
                err=True,
            )
    return judged, skipped


def _emit_detaches(log, min_confidence, collapse_state, existing_pairs, collapse_fh, dry_run):
    from wyrd.generators.kenning.lexicon.toponym_reflex_audit import detach_row

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
            click.echo(f"  [DETACH] {ref}  ({row.get('reason', '')[:50]})", err=True)
        else:
            _append(collapse_fh, out_row)
    return counts


@click.command("audit-toponym-reflexes")
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
)
@click.option(
    "--audit-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_toponym_reflex_audit.jsonl"),
    show_default=True,
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
    help="Minimum confidence to DETACH a non-toponymic form.",
)
@click.option(
    "--over-merged",
    is_flag=True,
    default=False,
    help="Judge EVERY modern form in clusters with >=2 modern members (no surface "
    "auto-keep) — catches surface-SIMILAR noise (vulva~val, tuna~tūn) the default misses.",
)
@click.option("--limit", type=int, default=None, help="Cap forms judged this run.")
@click.option("--dry-run", is_flag=True, default=False, help="Judge + print; write nothing.")
def lexicon_audit_toponym_reflexes(
    db_path,
    collapse_file,
    audit_file,
    model,
    ollama_url,
    min_confidence,
    over_merged,
    limit,
    dry_run,
):
    """LLM-audit modern era-grid reflexes for toponym-plausibility and detach the
    unrelated cluster-merged ones (wyrd-1wv2)."""
    from wyrd.generators.kenning.cli.utils import _readonly_lexicon
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.jsonl.build import collect_collapses
    from wyrd.generators.kenning.lexicon.toponym_reflex_audit import (
        detect_toponym_reflex_candidates,
    )

    if db_path is None:
        env = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env) if env else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.is_file():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")
    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")

    click.echo(f"Detecting toponym-reflex candidates in {db_path}...", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_toponym_reflex_candidates(
            conn, prescreen=not over_merged, min_modern_members=2 if over_merged else 1
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

    client = OllamaClient(base_url=base_url, model=model, timeout_s=60.0)
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
        f"judged {judged} new ({skipped} skipped); {verb}: {counts['detaches']} forms, "
        f"kept {counts['kept']}, already-detached {counts['already']}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    parent.add_command(lexicon_audit_toponym_reflexes)
