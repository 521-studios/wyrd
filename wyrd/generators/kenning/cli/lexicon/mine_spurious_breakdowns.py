"""``wyrd kenning lexicon mine-spurious-breakdowns`` — flag mining-hallucinated
element breakdowns that belong to a different toponym (wyrd-u6fn.6).

Deterministic pre-screen (low surface-overlap with the toponym name) narrows the
candidates; an LLM judge (Ollama) decides whether each breakdown plausibly
decomposes THIS toponym or was contaminated from a neighbour entry. Confirmed
spurious rows get a reversible ``decomposition-spurious`` assertion (D50 Family C);
never deleted. DORMANT until the canonicalization projection wires Family-C
suppression — mirrors mine-same-morpheme-binds.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click

from wyrd.generators.kenning.canonicalization import (
    append_assertion,
    load_assertions,
    score_confidence,
)
from wyrd.generators.kenning.canonicalization.streams import CANONICALIZATION_DIRNAME
from wyrd.generators.kenning.cli.lexicon._judge_retry import judge_with_retry
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

_AUDIT_FILE = "_decomposition_spurious_audit.jsonl"  # idempotency/judged-log


def _audit_path(mining_dir: Path) -> Path:
    # Deliberately under canonicalization/ (not data/mining/ root like the other
    # audit logs): the row _type ('decomposition_spurious_audit') is NOT in
    # ALL_TYPES (jsonl/log.py), so at root it'd be swept into the rebuild replay glob
    # (jsonl_paths_in) and raise ReplayError unless added to REPLAY_EXCLUDED_LEDGERS
    # — but that denylist's drift-guard requires the file to exist, and this log is
    # created lazily on --apply. canonicalization/ keeps it out of the root glob;
    # load_assertions reads canonicalization/*.jsonl but intentionally skips non
    # canonical_assertion rows, so it's inert there too (wyrd-u6fn.6 / wyrd-5qg7).
    return mining_dir / CANONICALIZATION_DIRNAME / _AUDIT_FILE


def _load_judged(mining_dir: Path) -> dict[int, dict]:
    """te_id -> prior judged verdict row, so re-runs skip already-judged rows."""
    path = _audit_path(mining_dir)
    out: dict[int, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            click.echo(
                f"  warning: skipping corrupt line in {path.name} (te_id will be re-judged)",
                err=True,
            )
            continue
        if row.get("_type") == "decomposition_spurious_audit" and isinstance(row.get("te_id"), int):
            out[row["te_id"]] = row
    return out


def _judge_with_retry(client, c):
    from wyrd.generators.kenning.lexicon.spurious_breakdown_mining import (
        build_judge_prompt,
        parse_verdict,
    )

    return judge_with_retry(
        client, c, build_prompt=build_judge_prompt, parse_verdict=parse_verdict, ref=f"te={c.te_id}"
    )


def _judge_loop(
    client,
    to_judge,
    *,
    mining_dir,
    source,
    min_confidence,
    apply,
    existing_ids,
    audit_fh,
    base_url,
    model,
) -> dict[str, int]:
    """Judge each candidate, author decomposition-spurious assertions for confirmed
    spurious rows (≥min-confidence), and record every verdict to the audit log."""
    from wyrd.generators.kenning.lexicon.spurious_breakdown_mining import spurious_assertion

    counts = {"authored": 0, "spurious": 0, "kept": 0, "skipped": 0}
    consecutive = 0
    start = time.perf_counter()
    for i, c in enumerate(to_judge, 1):
        v = _judge_with_retry(client, c)
        if v is None:
            counts["skipped"] += 1
            consecutive += 1
            if consecutive >= 5:
                raise click.ClickException(
                    f"Aborting: 5 consecutive judge failures — is Ollama up at {base_url} "
                    f"with model {model}? (curl .../api/tags)"
                )
            continue
        consecutive = 0
        if v.spurious and score_confidence(v.confidence) >= score_confidence(min_confidence):
            counts["spurious"] += 1
            a = spurious_assertion(c, v, source=source)
            if apply and a.id not in existing_ids:
                append_assertion(mining_dir, a)
                existing_ids.add(a.id)
                counts["authored"] += 1
            elif not apply:
                click.echo(
                    f"  [SPURIOUS] te={c.te_id} {c.modern_name!r}: {v.reason[:60]}", err=True
                )
        else:
            counts["kept"] += 1
        if audit_fh is not None:
            audit_fh.write(
                json.dumps(
                    {
                        "_type": "decomposition_spurious_audit",
                        "te_id": c.te_id,
                        "spurious": v.spurious,
                        "confidence": v.confidence,
                        "reason": v.reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            audit_fh.flush()
        if i % 10 == 0 or i == len(to_judge):
            rate = (time.perf_counter() - start) / i
            click.echo(
                f"  judged [{i}/{len(to_judge)}] spurious={counts['spurious']} "
                f"kept={counts['kept']} skipped={counts['skipped']} ({rate:.1f}s/entry)",
                err=True,
            )
    return counts


@click.command("mine-spurious-breakdowns")
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
    help="L2 mining root; assertions append under <dir>/canonicalization/.",
)
@click.option("--source", default="spurious-breakdown-audit", show_default=True)
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
    help="Minimum verdict confidence to author a decomposition-spurious assertion.",
)
@click.option("--limit", type=int, default=None, help="Cap candidates judged this run.")
@click.option(
    "--apply/--dry-run",
    default=False,
    show_default=True,
    help="--apply authors assertions + records verdicts; --dry-run (default) reports only.",
)
def lexicon_mine_spurious_breakdowns(
    db_path: Path,
    mining_dir: Path,
    source: str,
    model: str,
    ollama_url: str | None,
    min_confidence: str,
    limit: int | None,
    apply: bool,
) -> None:
    """wyrd-u6fn.6: detect toponym_etymology rows whose element breakdown belongs to a
    different word (neighbour contamination that passed D3 form-in-body), and author
    reversible decomposition-spurious assertions for the LLM-confirmed ones."""
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.lexicon.spurious_breakdown_mining import (
        detect_candidates,
    )

    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")
    click.echo(f"Pre-screening spurious-breakdown candidates in {db_path}…", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_candidates(conn)

    judged = _load_judged(mining_dir)
    to_judge = [c for c in candidates if c.te_id not in judged]
    if limit is not None:
        to_judge = to_judge[:limit]
    click.echo(
        f"  candidates={len(candidates)} already-judged={len(judged)} to-judge={len(to_judge)} "
        f"(model={model}, min-confidence={min_confidence})",
        err=True,
    )

    existing_ids = {a.id for a in load_assertions(mining_dir)}
    client = OllamaClient(base_url=base_url, model=model, timeout_s=60.0)
    audit_fh = None
    try:
        if apply:
            _audit_path(mining_dir).parent.mkdir(parents=True, exist_ok=True)
            audit_fh = _audit_path(mining_dir).open("a", encoding="utf-8")
        counts = _judge_loop(
            client,
            to_judge,
            mining_dir=mining_dir,
            source=source,
            min_confidence=min_confidence,
            apply=apply,
            existing_ids=existing_ids,
            audit_fh=audit_fh,
            base_url=base_url,
            model=model,
        )
    finally:
        if audit_fh is not None:
            audit_fh.close()

    verb = "authored" if apply else "would author"
    click.echo(
        f"spurious={counts['spurious']} (≥{min_confidence}) kept={counts['kept']} "
        f"skipped={counts['skipped']}; {verb} "
        f"{counts['authored'] if apply else counts['spurious']} decomposition-spurious assertions"
        + ("" if apply else " (dry-run — nothing written)"),
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``mine-spurious-breakdowns`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_spurious_breakdowns)
