"""``wyrd kenning lexicon mine-duplicate-canonicals`` — find two SEPARATE canonical
etymon nodes that are the same morpheme and author reversible same-morpheme bind +
merge-canonical assertions to collapse them (wyrd-u6fn.5).

Deterministic pre-screen (canonical etymons sharing a gloss within a language)
narrows the pairs; a two-pass LLM judge PROPOSES sameness then an independent
skeptic tries to REFUTE it (a wrong identity-collapse is corruption, a missed
merge is harmless — D46). Confirmed pairs at/above the confidence gate get a
reversible mint-canonical + bind + merge-canonical set (D50 Family A, LIVE — the
L3 projection collapses them); below-gate pairs are queued to the audit log only.
Regression target: OE 'niwe' (671826) ≈ 'ne' (369498), both 'new'.
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
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

_AUDIT_FILE = "_duplicate_canonical_audit.jsonl"  # idempotency/judged-log
_AUDIT_TYPE = "duplicate_canonical_audit"


def _audit_path(mining_dir: Path) -> Path:
    # Under canonicalization/ (not data/mining/ root like the other audit logs):
    # the row _type isn't in ALL_TYPES (jsonl/log.py), so at root it'd be swept into
    # the rebuild replay glob (jsonl_paths_in) and raise ReplayError unless added to
    # REPLAY_EXCLUDED_LEDGERS — whose drift-guard needs the file to exist, but this
    # log is created lazily on --apply. canonicalization/ keeps it out of the root
    # glob; load_assertions reads canonicalization/*.jsonl but intentionally skips
    # non canonical_assertion rows, so it's inert there too (wyrd-u6fn.5 / wyrd-5qg7).
    return mining_dir / CANONICALIZATION_DIRNAME / _AUDIT_FILE


def _pair_key(c) -> str:
    return f"{c.winner_id}:{c.loser_id}"


def _load_judged(mining_dir: Path) -> set[str]:
    """Set of 'winner:loser' pair keys already judged, so re-runs skip them."""
    path = _audit_path(mining_dir)
    out: set[str] = set()
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
                f"  warning: skipping corrupt line in {path.name} (pair will be re-judged)",
                err=True,
            )
            continue
        if row.get("_type") == _AUDIT_TYPE and isinstance(row.get("pair"), str):
            out.add(row["pair"])
    return out


def _call(client, system, user, parse):
    """One judge call with a 2-attempt retry; None (with a stderr note) on failure."""
    last_err = "unparseable response"
    for _attempt in range(2):
        try:
            v = parse(client.chat_json(system, user, {}))
            if v is not None:
                return v
        except Exception as exc:  # transient LLM/transport noise: retry, then give up
            last_err = str(exc) or exc.__class__.__name__
    click.echo(f"  judge call failed: {last_err}", err=True)  # surface why, don't swallow
    return None


def _judge_pair(client, c):
    """Two-pass: propose same-morpheme, then adversarially refute. Returns a
    MergeVerdict (incl. leave-separate), or None if the propose call itself failed."""
    from wyrd.generators.kenning.lexicon.duplicate_canonical_mining import (
        build_propose_prompt,
        build_refute_prompt,
        combine_verdict,
        parse_propose,
        parse_refute,
    )

    prop = _call(client, *build_propose_prompt(c), parse_propose)
    if prop is None:
        return None  # couldn't even get a proposal — treat as a transient failure
    if not prop.same:
        return combine_verdict(prop, None)  # leave-separate; no need to refute
    refute = _call(client, *build_refute_prompt(c), parse_refute)
    return combine_verdict(prop, refute)  # refute None → verify-failed → leave-separate


def _disposition(v, min_confidence: str) -> str:
    """Which bucket a verdict lands in: separate / same (author) / queued (below gate)."""
    if not v.same:
        return "separate"
    if score_confidence(v.confidence) >= score_confidence(min_confidence):
        return "same"
    return "queued"


def _author_pair(c, v, *, mining_dir, source, existing_ids) -> bool:
    """Append this pair's same-morpheme assertions; True if any were newly written."""
    from wyrd.generators.kenning.lexicon.duplicate_canonical_mining import (
        same_morpheme_assertions,
    )

    authored_any = False
    for a in same_morpheme_assertions(c, v, source=source):
        if a.id not in existing_ids:
            append_assertion(mining_dir, a)
            existing_ids.add(a.id)
            authored_any = True
    return authored_any


def _write_audit(audit_fh, c, v) -> None:
    if audit_fh is None:
        return
    audit_fh.write(
        json.dumps(
            {
                "_type": _AUDIT_TYPE,
                "pair": _pair_key(c),
                "a_form": c.a_form,
                "b_form": c.b_form,
                "language": c.language,
                "same": v.same,
                "confidence": v.confidence,
                "reason": v.reason,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    audit_fh.flush()


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
    """Judge each pair (propose → refute), author same-morpheme assertions for
    confirmed pairs at/above min-confidence, queue the rest, log every verdict."""
    counts = {"authored": 0, "same": 0, "separate": 0, "queued": 0, "skipped": 0}
    consecutive = 0
    start = time.perf_counter()
    for i, c in enumerate(to_judge, 1):
        v = _judge_pair(client, c)
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
        bucket = _disposition(v, min_confidence)
        counts[bucket] += 1
        if bucket == "same" and apply:
            counts["authored"] += int(
                _author_pair(c, v, mining_dir=mining_dir, source=source, existing_ids=existing_ids)
            )
        if bucket != "separate":  # echo same + queued (both apply and dry-run) for visibility
            tag = "merge" if bucket == "same" else "queue"
            click.echo(
                f"  [{tag} {v.confidence}] {c.a_form!r}≈{c.b_form!r} ({c.language}): "
                f"{v.reason[:60]}",
                err=True,
            )
        _write_audit(audit_fh, c, v)
        if i % 10 == 0 or i == len(to_judge):
            rate = (time.perf_counter() - start) / i
            click.echo(
                f"  judged [{i}/{len(to_judge)}] same={counts['same']} queued={counts['queued']} "
                f"separate={counts['separate']} skipped={counts['skipped']} ({rate:.1f}s/entry)",
                err=True,
            )
    return counts


@click.command("mine-duplicate-canonicals")
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
@click.option("--source", default="duplicate-canonical-audit", show_default=True)
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
    default="high",
    show_default=True,
    help="Minimum combined confidence to AUTHOR a merge (below → queued to audit log).",
)
@click.option(
    "--min-gloss-overlap",
    type=float,
    default=0.5,
    show_default=True,
    help="Pre-screen: minimum gloss-token Jaccard to pair two canonical etymons.",
)
@click.option("--limit", type=int, default=None, help="Cap pairs judged this run.")
@click.option(
    "--apply/--dry-run",
    default=False,
    show_default=True,
    help="--apply authors merges + records verdicts; --dry-run (default) reports only.",
)
def lexicon_mine_duplicate_canonicals(
    db_path: Path,
    mining_dir: Path,
    source: str,
    model: str,
    ollama_url: str | None,
    min_confidence: str,
    min_gloss_overlap: float,
    limit: int | None,
    apply: bool,
) -> None:
    """wyrd-u6fn.5: find canonical etymons that are the same morpheme under two
    spellings (e.g. OE 'niwe' ≈ 'ne', both 'new'), via a gloss pre-screen + a
    two-pass propose/refute LLM judge, and author reversible same-morpheme merges."""
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.lexicon.duplicate_canonical_mining import detect_candidates

    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")
    click.echo(f"Pre-screening duplicate-canonical pairs in {db_path}…", err=True)
    with _readonly_lexicon(db_path) as conn:
        det = detect_candidates(conn, min_gloss_overlap=min_gloss_overlap)
    if det.dropped_buckets:
        click.echo(
            f"  note: skipped {det.dropped_buckets} oversized gloss buckets (> {det.bucket_cap} "
            f"etymons) — non-discriminative; raise --min-gloss-overlap to tighten",
            err=True,
        )

    judged = _load_judged(mining_dir)
    to_judge = [c for c in det.candidates if _pair_key(c) not in judged]
    if limit is not None:
        to_judge = to_judge[:limit]
    click.echo(
        f"  pairs={len(det.candidates)} already-judged={len(judged)} to-judge={len(to_judge)} "
        f"(model={model}, min-confidence={min_confidence}, min-gloss-overlap={min_gloss_overlap})",
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
        f"same={counts['same']} (≥{min_confidence}) queued={counts['queued']} (below gate) "
        f"separate={counts['separate']} skipped={counts['skipped']}; {verb} "
        f"{counts['authored'] if apply else counts['same']} merges"
        + ("" if apply else " (dry-run — nothing written)"),
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``mine-duplicate-canonicals`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_duplicate_canonicals)
