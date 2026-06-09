"""``wyrd kenning lexicon canonicalize-toponym-etymology`` — canonicalize toponym etymology element refs."""

from __future__ import annotations

import json
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


def _echo_plan_preview(plans: list, source_id: str | None) -> None:
    """Print the dry-run outcome preview (counts computed before any write)."""
    promoted = sum(1 for p in plans if p.decision.promoted_etymology_id is not None)
    no_consensus = sum(
        1 for p in plans if p.decision.promoted_etymology_id is None and not p.all_empty_elements
    )
    no_elements = sum(1 for p in plans if p.all_empty_elements)
    click.echo(
        f"Examining {len(plans)} toponym(s)"
        + (f" (filtered to source_id={source_id!r})" if source_id else "")
        + f": promote={promoted} no_consensus={no_consensus} no_elements={no_elements}",
        err=True,
    )


def _echo_verbose_decisions(plans: list) -> None:
    """Print one per-toponym decision line to stderr (``--verbose``)."""
    for plan in plans:
        d = plan.decision
        if d.promoted_etymology_id is not None:
            click.echo(
                f"  toponym#{d.toponym_id}: promote ety#{d.promoted_etymology_id} "
                f"(consensus={d.consensus_size}, clusters={d.total_clusters}, "
                f"runner_up={d.runner_up_witness_count}) "
                f"key={d.cluster_key!r}",
                err=True,
            )
        else:
            click.echo(
                f"  toponym#{d.toponym_id}: no consensus "
                f"(clusters={d.total_clusters}, "
                f"max_witnesses={d.runner_up_witness_count})",
                err=True,
            )


def _apply_with_audit(db, plans: list, audit_path: Path | None, apply_fn):
    """Apply canonical decisions, optionally streaming one JSONL audit
    row per decision to ``audit_path``. Returns the apply summary.

    The audit sink (when requested) is opened before the apply and
    closed in a ``finally`` so a mid-apply error still flushes the rows
    written so far. ``apply_fn`` is the lazily-imported
    ``apply_canonical_decisions`` (passed in to keep the CLI's
    import-time cost off the hot path).
    """
    audit_sink = None
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_sink = audit_path.open("w", encoding="utf-8")

    def _audit(decision) -> None:
        if audit_sink is None:
            return
        audit_sink.write(
            json.dumps(
                {
                    "toponym_id": decision.toponym_id,
                    "promoted_etymology_id": decision.promoted_etymology_id,
                    "consensus_size": decision.consensus_size,
                    "cluster_key": decision.cluster_key,
                    "runner_up_witness_count": decision.runner_up_witness_count,
                    "total_clusters": decision.total_clusters,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    try:
        return apply_fn(db, plans, on_decision=_audit)
    finally:
        if audit_sink is not None:
            audit_sink.close()


@click.command("canonicalize-toponym-etymology")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon SQLite DB.",
)
@click.option(
    "--source",
    "source_id",
    default=None,
    help="Restrict canonicalization to toponyms with at least one "
    "toponym_etymology row from this source_id. Default: every toponym.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Write to DB. Without this flag, runs in dry-run mode and prints "
    "the predicted promote/no-consensus counts.",
)
@click.option(
    "--audit-out",
    "audit_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write one JSONL row per decision (toponym_id, promoted_etymology_id, "
    "consensus_size, cluster_key, runner_up_witness_count, total_clusters). "
    "Use the audit trail to diff cross-run promotions or feed downstream "
    "consumers a list of newly-canonical toponyms.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite --audit-out if it exists.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-toponym decisions to stderr (suppressed by default — "
    "the corpus has ~thousands of toponyms).",
)
def lexicon_canonicalize_toponym_etymology(
    db_path: Path,
    source_id: str | None,
    apply: bool,
    audit_path: Path | None,
    force: bool,
    verbose: bool,
) -> None:
    """Canonicalize toponym_etymology rows via multi-extractor consensus (wyrd-08qv).

    Three LLM extractor passes (qwen, haiku, gemini) over the same
    source body produce three ``toponym_etymology`` rows per toponym.
    When ≥2 of them agree on the element-tuple (normalized form +
    language per ordinal), the answer is settled. This command:

    1. Clusters rows by (normalized element-tuple, languages) per toponym.
    2. Counts distinct extractor witnesses per cluster (parsing the
       ``extracted_by:provider:model`` prefix from the notes field).
    3. Promotes the largest agreeing cluster (≥2 witnesses) to canonical
       — marks one row ``is_canonical=1`` + stamps ``consensus_size`` +
       ``cluster_key`` on every row for audit visibility.
    4. Re-runs are idempotent: previously-canonical rows that lose
       consensus get demoted; single-witness rows stay
       ``is_canonical=0`` (but still get ``cluster_key`` /
       ``consensus_size`` stamped each pass for audit completeness).

    Downstream consumers read ``toponym_etymology_canonical`` (a VIEW
    over the canonical-only rows). Future LLM re-extraction passes can
    skip toponyms with ``is_canonical = 1`` rows to save cost on
    settled answers.
    """
    from wyrd.generators.kenning.toponym_etymology_canonical import (
        apply_canonical_decisions,
        compute_canonical_plans,
    )

    if audit_path is not None and audit_path.exists() and not force:
        raise click.ClickException(f"--audit-out {audit_path} exists; pass --force to overwrite")
    db = LexiconDB(db_path)
    click.echo(f"Using DB {db_path}", err=True)

    # Use compute_canonical_plans (not compute_canonical_decisions) so
    # the per-toponym row_updates are built once. apply_canonical_decisions
    # consumes the plans directly; this avoids the previous N+1 reload
    # on the apply side (wyrd-08qv R1 — Gemini HIGH).
    plans = compute_canonical_plans(db, source_id=source_id)
    if not plans:
        click.echo("No toponym_etymology rows to canonicalize.", err=True)
        return

    _echo_plan_preview(plans, source_id)
    if verbose:
        _echo_verbose_decisions(plans)

    if not apply:
        click.echo("(dry-run — pass --apply to write)", err=True)
        return

    summary = _apply_with_audit(db, plans, audit_path, apply_canonical_decisions)

    click.echo(
        f"TOTAL toponyms={summary.toponyms_examined} "
        f"promoted={summary.toponyms_promoted} "
        f"no_consensus={summary.toponyms_no_consensus} "
        f"no_elements={summary.toponyms_no_elements} "
        f"rows_updated={summary.rows_updated} (APPLIED)",
        err=True,
    )
    if audit_path is not None:
        click.echo(f"Wrote audit JSONL → {audit_path}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``canonicalize-toponym-etymology`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_canonicalize_toponym_etymology)
