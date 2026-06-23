"""``wyrd kenning lexicon gloss-campaign`` — the compressed-gloss loop rail
(wyrd-u6fn.10.1).

Verbs mirror the eni4 enrichment rail so the loop runbook reads the same:

  next-slice   impact-ordered morphemes still needing sense compression (JSON)
  validate     gate candidate rows (grounding + partition + dedup); exit 0/1
  author       validate + append the canonical_sense assertions to the L2 ledger
  status       committed sense-coverage scoreboard (the per-fire number)
  park         drop a fragment / no-consensus / ambiguous morpheme from the queue

The judgment (partition a wall into senses + short labels) is done by the LOOP
AGENT (Opus) or a headless Ollama pass; either writes ``_type="gloss_sense"``
candidate rows that ``validate`` / ``author`` consume. Remainder is always from
the COMMITTED canonicalization ledger, never the live DB (wyrd-1rw4).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import click

from wyrd.generators.kenning.canonicalization import append_assertion
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.lexicon.gloss_sense import (
    GlossSenseCandidate,
    parse_candidate,
    sense_assertions,
    validate_gloss_sense_candidates,
)
from wyrd.generators.kenning.lexicon.gloss_sense_campaign import (
    committed_done,
    next_slice,
    sense_progress,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from wyrd.generators.kenning.cli.utils import Connection

_DEFAULT_MINING_DIR = "data/mining"
_PARKED_FILENAME = "_sense_parked.jsonl"


@contextmanager
def _open_lexicon(db_path: str | Path) -> Iterator[Connection]:
    """Read-only lexicon connection (via the shared cli.utils helper, so this CLI
    file never imports sqlite3 itself — the package-boundary rule). ``Connection``
    is cli.utils's re-export of ``sqlite3.Connection``, so the annotation stays
    sqlite3-free; ``Connection`` / ``Iterator`` are type-only (TYPE_CHECKING)."""
    db_file = Path(db_path)
    if not db_file.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_file}")
    with _readonly_lexicon(db_file) as conn:
        yield conn


def _parked_path(mining_dir: str) -> Path:
    return Path(mining_dir) / _PARKED_FILENAME


def _load_candidates(candidates_path: str) -> list[GlossSenseCandidate]:
    out: list[GlossSenseCandidate] = []
    with Path(candidates_path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise click.ClickException(f"{candidates_path}:{line_no}: invalid JSON") from exc
            c = parse_candidate(data)
            if c is None:
                raise click.ClickException(
                    f"{candidates_path}:{line_no}: malformed gloss_sense row"
                )
            out.append(c)
    return out


@click.group("gloss-campaign")
def gloss_campaign() -> None:
    """Compressed-gloss loop rail (wyrd-u6fn.10)."""


_db_option = click.option(
    "--db", "db_path", type=click.Path(), default=_DEFAULT_LEXICON_PATH, help="Lexicon DB path."
)
_mining_option = click.option(
    "--mining-dir",
    default=_DEFAULT_MINING_DIR,
    help="L2 mining root (canonicalization/ lives under it).",
)


@gloss_campaign.command("next-slice")
@_db_option
@_mining_option
@click.option("--n", type=int, default=20, help="Slice size (runbook uses 40 phase 1, 8 phase 2).")
@click.option(
    "--min-impact",
    type=int,
    default=2,
    help="Impact floor (phase 1/2 = 2; ignored when --full-inventory is set).",
)
@click.option(
    "--unglossed", is_flag=True, help="Phase 2: the UNGLOSSED cohort (needs glossing first)."
)
@click.option(
    "--full-inventory",
    is_flag=True,
    help="Phase 3: every glossed etymon by wall size (overrides --min-impact).",
)
def cmd_next_slice(db_path, mining_dir, n, min_impact, unglossed, full_inventory):
    """Emit the next impact-ordered morphemes still needing sense compression."""
    with _open_lexicon(db_path) as conn:
        slice_ = next_slice(
            conn,
            mining_dir,
            n=n,
            min_impact=min_impact,
            require_gloss=not unglossed,
            parked_path=_parked_path(mining_dir),
            full_inventory=full_inventory,
        )
    click.echo(json.dumps([e.to_dict() for e in slice_], ensure_ascii=False))


@gloss_campaign.command("validate")
@_db_option
@_mining_option
@click.option(
    "--candidates", required=True, type=click.Path(exists=True), help="JSONL of gloss_sense rows."
)
def cmd_validate(db_path, mining_dir, candidates):
    """Gate candidate rows (grounding + partition + dedup). Exit 0 OK, 1 on errors."""
    cands = _load_candidates(candidates)
    with _open_lexicon(db_path) as conn:
        errors = validate_gloss_sense_candidates(conn, committed_done(mining_dir), cands)
    if errors:
        click.echo(f"VALIDATION FAILED — {len(errors)} error(s):", err=True)
        for e in errors:
            click.echo(f"  - {e}", err=True)
        raise SystemExit(1)
    click.echo(f"OK — {len(cands)} candidate(s) valid", err=True)


@gloss_campaign.command("author")
@_db_option
@_mining_option
@click.option(
    "--candidates", required=True, type=click.Path(exists=True), help="JSONL of gloss_sense rows."
)
def cmd_author(db_path, mining_dir, candidates):
    """Validate, then append the canonical_sense assertions for each candidate."""
    cands = _load_candidates(candidates)
    with _open_lexicon(db_path) as conn:
        errors = validate_gloss_sense_candidates(conn, committed_done(mining_dir), cands)
    if errors:
        click.echo(f"REFUSED — {len(errors)} validation error(s); fix or drop and retry:", err=True)
        for e in errors:
            click.echo(f"  - {e}", err=True)
        raise SystemExit(1)
    n_assertions = 0
    for c in cands:
        for a in sense_assertions(c):
            append_assertion(mining_dir, a)
            n_assertions += 1
    click.echo(
        f"authored {len(cands)} morpheme(s), {n_assertions} assertion(s) -> {mining_dir}/canonicalization/",
        err=True,
    )


@gloss_campaign.command("status")
@_db_option
@_mining_option
@click.option("--min-impact", type=int, default=2, help="Cohort impact floor.")
def cmd_status(db_path, mining_dir, min_impact):
    """Committed sense-coverage of the phase-1 cohort (the per-fire scoreboard)."""
    with _open_lexicon(db_path) as conn:
        done, parked, total = sense_progress(
            conn, mining_dir, _parked_path(mining_dir), min_impact=min_impact
        )
    pct = (100.0 * done / total) if total else 0.0
    click.echo(
        f"gloss-sense coverage: {done}/{total} ({pct:.1f}%) — {parked} parked, "
        f"{total - done - parked} remaining",
        err=True,
    )


@gloss_campaign.command("park")
@_mining_option
@click.option("--ref", "ref", required=True, help="Etymon ref 'language:canonical_form' to park.")
@click.option("--reason", required=True, help="One line: fragment / no-consensus / ambiguous / …")
def cmd_park(mining_dir, ref, reason):
    """Park a morpheme (fragment / no consensus / ambiguous) out of future slices."""
    path = _parked_path(mining_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ref": ref, "reason": reason}, ensure_ascii=False) + "\n")
    click.echo(f"parked {ref}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``gloss-campaign`` on the parent ``@lexicon`` group."""
    parent.add_command(gloss_campaign)
