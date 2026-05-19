"""``wyrd kenning lexicon mine-fantasy-name`` — wyrd-ami fantasy-name etymology research pipeline (D30)."""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import click

from wyrd.generators.kenning import fantasy_pipeline
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-fantasy-name")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option("--name", default=None, help="Single creature/morpheme name to research.")
@click.option(
    "--description",
    default=None,
    help="Paragraph describing the creature (passed to the LLM when pre-filter misses).",
)
@click.option(
    "--batch",
    "batch_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        'JSONL file with one {"name": ..., "description": ...} per line. '
        "Mutually exclusive with --name/--description. Resumes correctly: "
        "re-running with the same approach_version updates rows in place."
    ),
)
@click.option(
    "--skip-llm",
    is_flag=True,
    default=False,
    help=(
        "Run only the descent-walking pre-filter; bar pre-filter misses "
        "instead of calling the LLM. Useful for offline tests, smokes, and "
        "estimating how many inputs the pre-filter alone covers."
    ),
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write fantasy_morpheme rows + add 'fantasy' tag on usable etymons. Without this, dry-run.",
)
@click.option(
    "--skip-resolved",
    is_flag=True,
    default=False,
    help=(
        "Skip inputs whose name already has a fantasy_morpheme row for "
        "the current approach_version. Lets you grow the input corpus "
        "(e.g. extractor v2) and re-run without paying for already-"
        "resolved entries. Looks up the existing names once at startup. "
        "Comparison is case-insensitive (matches the table's "
        "COLLATE NOCASE UNIQUE constraint)."
    ),
)
@click.option(
    "--model",
    default=None,
    help=(
        "Override the LLM model used for both the semantic-check pass on "
        "pre-filter resolutions and the full-research fallback. Gemini "
        "default: gemini-2.5-flash. Pass a different Gemini model name "
        "(gemini-2.5-pro, gemini-2.5-flash-lite, etc.) to substitute it."
    ),
)
def lexicon_mine_fantasy_name(
    db_path: Path,
    name: str | None,
    description: str | None,
    batch_path: Path | None,
    skip_llm: bool,
    apply_changes: bool,
    skip_resolved: bool,
    model: str | None,
) -> None:
    """Research the etymology of a fantasy/gaming creature name (wyrd-ami).

    Each input (name + paragraph description) is routed through:
    1. A descent-walking lookup against the wiktionary-derived etymon
       corpus. Resolves common inputs without an LLM call (harpy →
       ancient-greek ἅρπυια, troll → ON trǫll, etc.).
    2. A Gemini-Flash full-research call when the pre-filter misses.

    Every input produces ONE row in `fantasy_morpheme`: usable=1 with
    `etymon_id` set when we resolved to an attested etymon; usable=0
    with `bar_reason` (e.g. modern_coinage for Dunsany's "gnoll") when
    we couldn't. Conservative-by-default: low-confidence LLM answers
    get barred as `uncertain_attestation`.

    Single mode: `--name X --description Y`
    Batch mode:  `--batch path/to/names.jsonl`
    """
    fp = fantasy_pipeline  # local alias matches the helper-style usage below

    if batch_path and (name or description):
        raise click.ClickException("--batch is mutually exclusive with --name/--description")
    if not batch_path and not (name and description):
        raise click.ClickException("provide either --batch or both --name and --description")

    inputs: list[tuple[str, str]] = []
    if batch_path:
        with batch_path.open() as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise click.ClickException(f"line {i}: invalid JSON: {e}") from e
                if "name" not in rec or "description" not in rec:
                    raise click.ClickException(f"line {i}: missing 'name' or 'description'")
                inputs.append((rec["name"], rec["description"]))
    else:
        inputs.append((name, description))  # type: ignore[arg-type]

    skipped_count = 0
    if skip_resolved:
        # Lowercase-fold both sides — fantasy_morpheme.input_name uses
        # COLLATE NOCASE, so 'Harpy' and 'harpy' are the same row.
        # Without folding, varying input casing would re-trigger LLM
        # calls even though the upsert would later collapse them.
        with LexiconDB(db_path) as resolved_db:
            already = {n.lower() for n in fp.fetch_resolved_input_names(resolved_db.conn)}
        before = len(inputs)
        inputs = [(n, d) for (n, d) in inputs if n.lower() not in already]
        skipped_count = before - len(inputs)

    click.echo(
        f"Routing {len(inputs)} fantasy-name input(s)"
        f"{f' (skipped {skipped_count} already-resolved)' if skip_resolved else ''}. "
        f"approach_version={fp.APPROACH_VERSION}. "
        f"{'Applying' if apply_changes else 'Dry-run'}.",
        err=True,
    )

    counts = {"usable": 0, "barred": 0, "by_method": {}, "by_bar": {}}
    # Bind the LLM callers to the user-supplied model (if any). The
    # pipeline calls semantic_check_caller(name, desc, ancestor, glosses)
    # and llm_caller(name, desc); functools.partial pre-binds the model
    # kwarg without changing those signatures.
    if model:
        llm_caller = partial(fp._llm_full_research, model=model)
        semantic_check_caller = partial(fp._llm_semantic_check, model=model)
    else:
        llm_caller = fp._llm_full_research
        semantic_check_caller = fp._llm_semantic_check

    # Mining-progress convention: same shape as `lexicon mine-llm` so
    # operators can read both batches the same way. See repo CLAUDE.md
    # under "Mining progress reporting".
    progress_start = time.time()
    progress_every = 10
    total_inputs = len(inputs)

    write_db: LexiconDB | None = LexiconDB(db_path) if apply_changes else None
    try:
        for completed, (in_name, in_desc) in enumerate(inputs, start=1):
            res = fp.resolve(
                db_path,
                in_name,
                in_desc,
                skip_llm=skip_llm,
                llm_caller=llm_caller,
                semantic_check_caller=semantic_check_caller,
            )
            if res.usable:
                counts["usable"] += 1
                tag = f"USABLE  {in_name:<18}"
            else:
                counts["barred"] += 1
                tag = f"BARRED  {in_name:<18}"
                counts["by_bar"][res.bar_reason or "?"] = (
                    counts["by_bar"].get(res.bar_reason or "?", 0) + 1
                )
            counts["by_method"][res.resolution_method] = (
                counts["by_method"].get(res.resolution_method, 0) + 1
            )
            click.echo(
                f"  {tag} via={res.resolution_method:<22} conf={res.confidence or '-':<7}"
                f"{(' bar=' + res.bar_reason) if res.bar_reason else ''}"
                f"{(' citation=' + res.citation) if res.citation else ''}",
                err=True,
            )
            if write_db is not None:
                fp.write_resolution(
                    write_db.conn,
                    input_name=in_name,
                    input_description=in_desc,
                    resolution=res,
                )
                if res.usable and res.etymon_id is not None:
                    fp.tag_etymon_as_fantasy(write_db.conn, res.etymon_id)
                # Commit per resolution: each input cost an LLM call, so
                # crashing partway through a long batch shouldn't lose
                # already-paid-for results.
                write_db.commit()
            # Periodic progress line (matches mine-llm shape so operators
            # can scan both batches the same way).
            if completed % progress_every == 0 or completed == total_inputs:
                elapsed = time.time() - progress_start
                rate = elapsed / completed
                click.echo(
                    f"  [{completed}/{total_inputs}]  "
                    f"usable={counts['usable']} barred={counts['barred']} "
                    f"({rate:.1f}s/entry)",
                    err=True,
                )
    finally:
        if write_db is not None:
            write_db.close()

    click.echo("", err=True)
    click.echo(f"Summary: {counts}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``mine-fantasy-name`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_fantasy_name)
