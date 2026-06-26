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

from wyrd.generators.kenning.cli.lexicon._judge_retry import judge_with_retry

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


def _load_log(audit_file: Path) -> dict[str, dict]:
    """The recorded raw verdicts, ``member_ref → audit_log row`` (last wins).
    Lets a re-run reuse prior LLM judgments (no re-judging) and re-derive reverts
    at a new ``--min-confidence`` — the "emit high now, emit medium later" path."""
    log: dict[str, dict] = {}
    if not audit_file.exists():
        return log
    for line in audit_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        # Require a raw verdict (same_morpheme bool): a legacy row from the
        # threshold-gated schema carries no raw verdict, so keeping it would
        # exclude the member from re-judging (in `log`) yet skip it in emission
        # (verdict_from_log → None) — losing it. Omitting it re-judges + recovers.
        if (
            row.get("_type") == "merge_audit"
            and row.get("member_ref")
            and isinstance(row.get("same_morpheme"), bool)
        ):
            log[row["member_ref"]] = row
    return log


def _already_reverted(member_ref, provenance, collapse_state, curation_state) -> bool:
    """True when ``member_ref``'s revert is already recorded in the apply-ledgers
    (so a re-run at a lower threshold doesn't append a duplicate). Checked against
    the NET ledger state, not by method attribution."""
    from wyrd.generators.kenning.lexicon.merge_audit import PROV_COLLAPSE, PROV_OCR

    if provenance == PROV_COLLAPSE:
        # a collapse candidate was folding; an empty net `into` means reverted.
        return not (collapse_state.get(member_ref, {}).get("into") or "").strip()
    field = "merged_into_ref" if provenance == PROV_OCR else "lemma_ref"
    cur = curation_state.get(member_ref, {})
    return field in cur and cur[field] is None


def _judge_with_retry(client, c):
    """Judge one candidate, one retry on transport/parse failure; None if
    unusable (caller skips WITHOUT recording, so a re-run retries). Echoes the
    last failure to stderr so a persistent Ollama outage is diagnosable rather
    than presenting as a silently-climbing skip count."""
    from wyrd.generators.kenning.lexicon.merge_audit import (
        build_audit_judge_prompt,
        parse_audit_verdict,
    )

    return judge_with_retry(
        client,
        c,
        build_prompt=build_audit_judge_prompt,
        parse_verdict=parse_audit_verdict,
        ref=c.member_ref,
    )


def _append(fh, row: dict) -> None:
    """Append one JSONL row and flush (crash-safe as we go). No-op when fh is
    None (dry-run)."""
    if fh is not None:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def _judge_fresh(client, to_judge, log, audit_fh, base_url, model) -> tuple[int, int]:
    """PASS 1 — LLM-judge candidates NOT already in the log; append each raw
    verdict to ``log`` (in-memory) and to ``audit_fh``. Aborts after 5 consecutive
    failures (Ollama down). Returns (judged_new, skipped)."""
    from wyrd.generators.kenning.lexicon.merge_audit import audit_log_row

    judged_new = skipped = consecutive = 0
    start = time.perf_counter()
    for i, c in enumerate(to_judge, 1):
        verdict = _judge_with_retry(client, c)
        if verdict is None:
            skipped += 1
            consecutive += 1
            if consecutive >= 5:
                click.echo(
                    f"Aborting: 5 consecutive judge failures — is Ollama up at {base_url} "
                    f"with model {model}? (curl .../api/tags)",
                    err=True,
                )
                break
            continue
        consecutive = 0
        judged_new += 1
        row = audit_log_row(c, verdict)
        log[c.member_ref] = row
        _append(audit_fh, row)
        if i % 10 == 0 or i == len(to_judge):
            rate = (time.perf_counter() - start) / i
            click.echo(
                f"  judged [{i}/{len(to_judge)}] new={judged_new} skipped={skipped} "
                f"({rate:.1f}s/entry)",
                err=True,
            )
    return judged_new, skipped


def _emit_reverts(
    log, min_confidence, collapse_state, curation_state, collapse_fh, curation_fh, dry_run
) -> dict[str, int]:
    """PASS 2 — derive reverts from the recorded raw verdicts at ``min_confidence``
    (no LLM), provenance-routed, deduped against the net ledger state so a re-run
    at a lower threshold only tops up. Iterates the LOG, so verdicts recorded by a
    prior run are re-emittable here."""
    from wyrd.generators.kenning.lexicon.merge_audit import revert_rows, verdict_from_log

    counts = {
        "reverts": 0,
        "kept": 0,
        "already": 0,
        "collapse-fold": 0,
        "ocr-cluster": 0,
        "lemma-link": 0,
    }
    for member_ref, row in sorted(log.items()):
        v = verdict_from_log(row)
        if v is None:
            continue  # legacy row without a raw verdict — re-judge to recover
        provenance = row.get("provenance", "")
        collapse, curation = revert_rows(member_ref, provenance, v, min_confidence)
        if collapse is None and curation is None:
            counts["kept"] += 1
            continue
        if _already_reverted(member_ref, provenance, collapse_state, curation_state):
            counts["already"] += 1
            continue
        counts["reverts"] += 1
        counts[provenance] = counts.get(provenance, 0) + 1
        if dry_run:
            click.echo(
                f"  [REVERT/{provenance}] {member_ref} -> {row.get('anchor_ref')} "
                f"{v.confidence} ({v.reason})",
                err=True,
            )
        else:
            _append(collapse_fh, collapse) if collapse else _append(curation_fh, curation)
    return counts


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
    from wyrd.generators.kenning.jsonl.build import collect_collapses, collect_curation_overrides
    from wyrd.generators.kenning.lexicon.merge_audit import detect_merge_audit_candidates

    if db_path is None:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")
    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")

    collapse_state = collect_collapses([collapse_file]) if collapse_file.exists() else {}
    curation_state = collect_curation_overrides([curation_file]) if curation_file.exists() else {}
    click.echo(f"Detecting merge-audit candidates in {db_path}...", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_merge_audit_candidates(conn, collapse_state, scope=scope)

    log = _load_log(audit_file)  # member_ref → recorded raw verdict (reused, no re-judge)
    to_judge = [c for c in candidates if c.member_ref not in log]
    if limit is not None:
        to_judge = to_judge[:limit]
    click.echo(
        f"  candidates={len(candidates)} recorded={len(log)} to-judge={len(to_judge)} "
        f"(scope={scope}, model={model}, min-confidence={min_confidence})",
        err=True,
    )

    client = OllamaClient(base_url=base_url, model=model, timeout_s=90.0)
    if not dry_run:
        _ensure_source(audit_file, _MERGE_AUDIT_SOURCE_ROW)
        _ensure_source(curation_file, _CURATION_SOURCE_ROW)
    audit_fh = collapse_fh = curation_fh = None
    try:
        if not dry_run:
            audit_fh = audit_file.open("a", encoding="utf-8")
            collapse_fh = collapse_file.open("a", encoding="utf-8")
            curation_fh = curation_file.open("a", encoding="utf-8")
        # PASS 1 — judge fresh candidates (LLM), recording raw verdicts.
        judged_new, skipped = _judge_fresh(client, to_judge, log, audit_fh, base_url, model)
        # PASS 2 — derive + emit reverts at the chosen threshold from ALL recorded
        # verdicts (logged or fresh), so a later lower-threshold run tops up.
        counts = _emit_reverts(
            log, min_confidence, collapse_state, curation_state, collapse_fh, curation_fh, dry_run
        )
    finally:
        for fh in (audit_fh, collapse_fh, curation_fh):
            if fh is not None:
                fh.close()

    verb = "would revert" if dry_run else "reverted"
    click.echo(
        f"judged {judged_new} new ({skipped} skipped); {verb}: {counts['reverts']} merges "
        f"(collapse={counts['collapse-fold']} ocr={counts['ocr-cluster']} "
        f"lemma={counts['lemma-link']}), kept {counts['kept']}, "
        f"already-reverted {counts['already']}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``audit-merges`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_audit_merges)
