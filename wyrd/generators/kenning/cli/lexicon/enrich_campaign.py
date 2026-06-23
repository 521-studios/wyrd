"""``wyrd kenning lexicon enrich-campaign`` — rail for the scholar-morpheme
enrichment loop (wyrd-eni4.1).

The 5-minute loop drives a family of subcommands across four authored
dimensions — reflex, tag, IPA, gloss — plus parking, collapse validation,
band-status, and identity-reflex emission. The reflex-rail anchors:

  next-slice  — emit the next impact-ordered scholar etymons needing a reflex,
                with grounding evidence (JSON on stdout).
  validate    — gate authored reflex candidates (a JSONL file) before commit;
                exits non-zero with errors if any record fails.
  status      — committed reflex coverage of the scholar etymon set.
  park        — record an un-groundable etymon so it drops out of future slices.

Read-only against the lexicon DB; every write is an append to a JSONL ledger
under ``data/mining/`` (``_reflexes`` / ``_tags`` / ``_pronunciation`` /
``_curation`` / ``_reflex_parked``) — never to the DB itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.lexicon.enrichment_campaign import (
    bands_status,
    gloss_next_slice,
    identity_reflex_rows,
    ipa_next_slice,
    next_slice,
    reflex_progress,
    tag_next_slice,
    tag_progress,
    validate_candidates,
    validate_collapse_candidates,
    validate_gloss_candidates,
    validate_ipa_candidates,
    validate_tag_candidates,
    validate_variant_candidates,
    variant_gap_census,
    variant_gap_next_slice,
    variant_next_slice,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

_DEFAULT_REFLEXES_PATH = Path("data/mining/_reflexes.jsonl")
_DEFAULT_PARKED_PATH = Path("data/mining/_reflex_parked.jsonl")
_DEFAULT_TAGS_PATH = Path("data/mining/_tags.jsonl")
_DEFAULT_IPA_PATH = Path("data/mining/_pronunciation.jsonl")
# _curation.jsonl carries both gloss-add and collapse rows — one ledger, one path.
_DEFAULT_GLOSS_PATH = Path("data/mining/_curation.jsonl")
_DEFAULT_VARIANTS_PATH = Path("data/mining/_variants.jsonl")
_DEFAULT_VARIANT_PARKED_PATH = Path("data/mining/_variant_parked.jsonl")

_impact_option = click.option(
    "--impact", type=int, default=None, help="Restrict to a single impact band (band-by-band loop)."
)
_n_option = click.option("--n", type=int, default=20, show_default=True, help="Etymons to emit.")

_db_option = click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
_reflexes_option = click.option(
    "--reflexes-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_REFLEXES_PATH,
    show_default=True,
    help="Committed reflex ledger — the remainder/dedup source of truth.",
)
_parked_option = click.option(
    "--parked-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_PARKED_PATH,
    show_default=True,
    help="Parked-etymon ledger — un-groundable refs excluded from future slices.",
)


@click.group("enrich-campaign")
def enrich_campaign() -> None:
    """Scholar-morpheme enrichment campaign rail (wyrd-eni4.1)."""


@enrich_campaign.command("next-slice")
@_db_option
@_reflexes_option
@_parked_option
@_impact_option
@_n_option
def cmd_next_slice(
    db_path: Path, reflexes_path: Path, parked_path: Path, impact: int | None, n: int
) -> None:
    """Emit the next N impact-ordered scholar etymons still missing a reflex
    (and not parked), each with grounding evidence, as a JSON array on stdout."""
    with _readonly_lexicon(db_path) as conn:
        slice_ = next_slice(conn, reflexes_path, n=n, parked_path=parked_path, impact=impact)
    click.echo(json.dumps([e.to_dict() for e in slice_], ensure_ascii=False, indent=2))
    click.echo(f"emitted {len(slice_)} etymons (reflexes-path={reflexes_path})", err=True)


def _park_ref(db_path: Path, parked_path: Path, ref: str, reason: str, *, label: str = "") -> None:
    """Shared park primitive (reflex + variant rails): validate the ref resolves
    (a typo'd ref would otherwise sit in the ledger forever, never matching — and
    never grounding — a slice), skip a no-op re-park if already listed, else append
    a ``{ref, reason}`` row."""
    import unicodedata

    from wyrd.generators.kenning.lexicon.enrichment_campaign import parked_refs
    from wyrd.generators.kenning.lexicon.etymon_refs import resolve_etymon_ref

    with _readonly_lexicon(db_path) as conn:
        if resolve_etymon_ref(conn, ref) is None:
            click.echo(f"Error: etymon ref {ref!r} resolves to no etymon", err=True)
            sys.exit(1)
    if unicodedata.normalize("NFC", ref) in parked_refs(parked_path):
        click.echo(f"already parked {ref}", err=True)
        return
    parked_path.parent.mkdir(parents=True, exist_ok=True)
    with parked_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ref": ref, "reason": reason}, ensure_ascii=False) + "\n")
    click.echo(f"parked {ref}{label}", err=True)


@enrich_campaign.command("park")
@_db_option
@_parked_option
@click.option("--ref", "ref", required=True, help="The 'language:canonical_form' ref to park.")
@click.option("--reason", required=True, help="Why it can't be grounded (one line).")
def cmd_park(db_path: Path, parked_path: Path, ref: str, reason: str) -> None:
    """Record an un-groundable etymon so it drops out of future reflex slices."""
    _park_ref(db_path, parked_path, ref, reason)


@enrich_campaign.command("validate")
@_db_option
@_reflexes_option
@click.option(
    "--candidates",
    "candidates_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file of authored reflex candidate rows to validate.",
)
@click.option(
    "--allow-non-scholar",
    is_flag=True,
    default=False,
    help="Permit refs outside the scholar set (default: campaign-scoped).",
)
def cmd_validate(
    db_path: Path, reflexes_path: Path, candidates_path: Path, allow_non_scholar: bool
) -> None:
    """Validate authored reflex candidates. Exit 0 if all pass, 1 otherwise."""
    candidates = []
    for lineno, line in enumerate(candidates_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as exc:
            click.echo(f"line {lineno}: invalid JSON — {exc}", err=True)
            sys.exit(1)

    with _readonly_lexicon(db_path) as conn:
        errors = validate_candidates(
            conn, reflexes_path, candidates, scholar_only=not allow_non_scholar
        )
    if errors:
        click.echo(f"VALIDATION FAILED — {len(errors)} error(s):", err=True)
        for err in errors:
            click.echo(f"  - {err}", err=True)
        sys.exit(1)
    click.echo(f"OK — {len(candidates)} candidate(s) passed all gates.", err=True)


@enrich_campaign.command("status")
@_db_option
@_reflexes_option
@_parked_option
def cmd_status(db_path: Path, reflexes_path: Path, parked_path: Path) -> None:
    """Report committed reflex coverage of the scholar etymon set."""
    with _readonly_lexicon(db_path) as conn:
        done, parked, total = reflex_progress(conn, reflexes_path, parked_path=parked_path)
    pct = (100.0 * done / total) if total else 0.0
    remaining = total - done - parked
    click.echo(
        f"scholar-reflex coverage: {done}/{total} ({pct:.1f}%) "
        f"— {parked} parked, {remaining} remaining to attempt"
    )


@enrich_campaign.command("variant-gap-status")
@_db_option
def cmd_variant_gap_status(db_path: Path) -> None:
    """Diagnostic census (wyrd-eni4.3.1): classify every scholar (toponym,
    morpheme) pair as matched / variant-gap / reflex-less. Sizes the variant-gap
    lever — pairs where the morpheme HAS a reflex but none match this spelling."""
    with _readonly_lexicon(db_path) as conn:
        c = variant_gap_census(conn)
    total = c["total"] or 1
    click.echo(
        f"scholar (toponym,morpheme) pairs: {c['total']}  "
        f"matched={c['matched']} ({100 * c['matched'] / total:.1f}%)  "
        f"variant-gap={c['variant_gap']} ({100 * c['variant_gap'] / total:.1f}%)  "
        f"reflex-less={c['reflex_less']} ({100 * c['reflex_less'] / total:.1f}%)",
        err=True,
    )


@enrich_campaign.command("variant-gap-next-slice")
@_db_option
@_reflexes_option
@_impact_option
@_n_option
def cmd_variant_gap_next_slice(
    db_path: Path, reflexes_path: Path, impact: int | None, n: int
) -> None:
    """Emit the next N variant-gap scholar morphemes (have a reflex, none matching
    some toponym's spelling), flip-CAN-IT-first then by frequency, each with
    evidence toponyms + a residual-span hint, as a JSON array on stdout. Author a
    grounded reflex surface_form for the worn span (it's IN the toponym)."""
    with _readonly_lexicon(db_path) as conn:
        slice_ = variant_gap_next_slice(conn, reflexes_path, n=n, impact=impact)
    click.echo(json.dumps([t.to_dict() for t in slice_], ensure_ascii=False, indent=2))
    click.echo(
        f"emitted {len(slice_)} variant-gap morphemes "
        f"({sum(1 for t in slice_ if t.flips_can_it)} flip CAN-IT)",
        err=True,
    )


_variants_option = click.option(
    "--variants-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_VARIANTS_PATH,
    show_default=True,
    help="Committed variant ledger (_variants.jsonl) — the remainder/dedup source.",
)
_variant_parked_option = click.option(
    "--parked-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_VARIANT_PARKED_PATH,
    show_default=True,
    help="Parked variant ledger — etymons with no groundable variant, excluded from slices.",
)


@enrich_campaign.command("variants-next-slice")
@_db_option
@_variants_option
@_variant_parked_option
@_impact_option
@_n_option
def cmd_variants_next_slice(
    db_path: Path, variants_path: Path, parked_path: Path, impact: int | None, n: int
) -> None:
    """Emit the next N impact-ordered admit-cohort etymons missing a toponymic-
    surface variant (and not parked), each with grounding evidence (toponyms +
    surface_in_modern spans), as a JSON array on stdout. Author a grounded worn
    alt-spelling per the 'barthos' rule; tag it toponymic-surface."""
    with _readonly_lexicon(db_path) as conn:
        slice_ = variant_next_slice(
            conn, variants_path, n=n, parked_path=parked_path, impact=impact
        )
    click.echo(json.dumps([e.to_dict() for e in slice_], ensure_ascii=False, indent=2))
    click.echo(f"emitted {len(slice_)} etymons for variant uplift", err=True)


@enrich_campaign.command("variants-validate")
@_db_option
@_variants_option
@click.option(
    "--candidates",
    "candidates_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file of authored toponymic-surface variant rows to validate.",
)
def cmd_variants_validate(db_path: Path, variants_path: Path, candidates_path: Path) -> None:
    """Validate authored variant rows (grounding guard identical to reflexes).
    Exit 0 if all pass, 1 otherwise."""
    candidates = _load_candidates(candidates_path)
    with _readonly_lexicon(db_path) as conn:
        errors = validate_variant_candidates(conn, variants_path, candidates)
    _report_validation(errors, len(candidates), "variant(s)")


@enrich_campaign.command("variants-park")
@_db_option
@_variant_parked_option
@click.option("--ref", "ref", required=True, help="The 'language:canonical_form' ref to park.")
@click.option("--reason", required=True, help="Why no groundable variant exists (one line).")
def cmd_variants_park(db_path: Path, parked_path: Path, ref: str, reason: str) -> None:
    """Record an etymon with no groundable toponymic-surface variant so it drops
    out of future variant slices (separate ledger file, shared park primitive)."""
    _park_ref(db_path, parked_path, ref, reason, label=" (variant)")


_tags_option = click.option(
    "--tags-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_TAGS_PATH,
    show_default=True,
    help="Committed tag ledger (_tags.jsonl) — the remainder/dedup source.",
)


@enrich_campaign.command("tags-next-slice")
@_db_option
@_tags_option
@_impact_option
@_n_option
def cmd_tags_next_slice(db_path: Path, tags_path: Path, impact: int | None, n: int) -> None:
    """Emit the next N impact-ordered admit-cohort etymons that have a gloss but
    no tag, each with its glosses + the controlled vocab, as JSON."""
    with _readonly_lexicon(db_path) as conn:
        slice_ = tag_next_slice(conn, db_path, tags_path, n=n, impact=impact)
    click.echo(json.dumps(slice_, ensure_ascii=False, indent=2))
    click.echo(f"emitted {len(slice_)} etymons for tagging", err=True)


@enrich_campaign.command("tags-validate")
@_db_option
@_tags_option
@click.option(
    "--candidates",
    "candidates_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file of authored tag-decision rows to validate.",
)
def cmd_tags_validate(db_path: Path, tags_path: Path, candidates_path: Path) -> None:
    """Validate authored tag decisions. Exit 0 if all pass, 1 otherwise."""
    candidates = []
    for lineno, line in enumerate(candidates_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as exc:
            click.echo(f"line {lineno}: invalid JSON — {exc}", err=True)
            sys.exit(1)
    with _readonly_lexicon(db_path) as conn:
        errors = validate_tag_candidates(conn, tags_path, candidates)
    if errors:
        click.echo(f"VALIDATION FAILED — {len(errors)} error(s):", err=True)
        for err in errors:
            click.echo(f"  - {err}", err=True)
        sys.exit(1)
    click.echo(f"OK — {len(candidates)} tag decision(s) passed all gates.", err=True)


@enrich_campaign.command("tags-status")
@_db_option
@_tags_option
def cmd_tags_status(db_path: Path, tags_path: Path) -> None:
    """Report how many admit-cohort etymons still await a tag decision."""
    with _readonly_lexicon(db_path) as conn:
        remaining = tag_progress(conn, db_path, tags_path)
    click.echo(f"tag uplift: {remaining} admit-cohort etymons awaiting a tag decision")


_ipa_option = click.option(
    "--ipa-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_IPA_PATH,
    show_default=True,
    help="Committed pronunciation ledger (_pronunciation.jsonl).",
)
_gloss_option = click.option(
    "--gloss-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_GLOSS_PATH,
    show_default=True,
    help="Committed curation ledger (_curation.jsonl) carrying etymon_gloss_add rows.",
)


def _load_candidates(candidates_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lineno, line in enumerate(candidates_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            click.echo(f"line {lineno}: invalid JSON — {exc}", err=True)
            sys.exit(1)
    return out


def _report_validation(errors: list[str], n: int, label: str) -> None:
    if errors:
        click.echo(f"VALIDATION FAILED — {len(errors)} error(s):", err=True)
        for err in errors:
            click.echo(f"  - {err}", err=True)
        sys.exit(1)
    click.echo(f"OK — {n} {label} passed all gates.", err=True)


@enrich_campaign.command("ipa-next-slice")
@_db_option
@_ipa_option
@_impact_option
@_n_option
def cmd_ipa_next_slice(db_path: Path, ipa_path: Path, impact: int | None, n: int) -> None:
    """Emit the next N target morphemes missing pronunciation_ipa, with glosses."""
    with _readonly_lexicon(db_path) as conn:
        slice_ = ipa_next_slice(conn, ipa_path, n=n, impact=impact)
    click.echo(json.dumps(slice_, ensure_ascii=False, indent=2))
    click.echo(f"emitted {len(slice_)} etymons for IPA", err=True)


@enrich_campaign.command("ipa-validate")
@_db_option
@_ipa_option
@click.option(
    "--candidates",
    "candidates_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def cmd_ipa_validate(db_path: Path, ipa_path: Path, candidates_path: Path) -> None:
    """Validate authored pronunciation rows. Exit 0 if all pass, 1 otherwise."""
    candidates = _load_candidates(candidates_path)
    with _readonly_lexicon(db_path) as conn:
        errors = validate_ipa_candidates(conn, ipa_path, candidates)
    _report_validation(errors, len(candidates), "pronunciation row(s)")


@enrich_campaign.command("gloss-next-slice")
@_db_option
@_gloss_option
@_impact_option
@_n_option
def cmd_gloss_next_slice(db_path: Path, gloss_path: Path, impact: int | None, n: int) -> None:
    """Emit the next N target morphemes with no gloss, with their toponyms."""
    with _readonly_lexicon(db_path) as conn:
        slice_ = gloss_next_slice(conn, gloss_path, n=n, impact=impact)
    click.echo(json.dumps(slice_, ensure_ascii=False, indent=2))
    click.echo(f"emitted {len(slice_)} etymons for gloss", err=True)


@enrich_campaign.command("gloss-validate")
@_db_option
@_gloss_option
@click.option(
    "--candidates",
    "candidates_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def cmd_gloss_validate(db_path: Path, gloss_path: Path, candidates_path: Path) -> None:
    """Validate authored etymon_gloss_add rows. Exit 0 if all pass, 1 otherwise."""
    candidates = _load_candidates(candidates_path)
    with _readonly_lexicon(db_path) as conn:
        errors = validate_gloss_candidates(conn, gloss_path, candidates)
    _report_validation(errors, len(candidates), "gloss row(s)")


@enrich_campaign.command("bands")
@_db_option
@_reflexes_option
@_parked_option
@_tags_option
@_ipa_option
@_gloss_option
@click.option("--top", type=int, default=12, show_default=True, help="Show this many top bands.")
def cmd_bands(
    db_path: Path,
    reflexes_path: Path,
    parked_path: Path,
    tags_path: Path,
    ipa_path: Path,
    gloss_path: Path,
    top: int,
) -> None:
    """Per impact band (descending), morphemes still missing each loop-owned
    dimension. The first row is the current top band for the band-by-band loop."""
    with _readonly_lexicon(db_path) as conn:
        bands = bands_status(
            conn,
            reflexes_path=reflexes_path,
            parked_path=parked_path,
            tags_path=tags_path,
            ipa_path=ipa_path,
            gloss_path=gloss_path,
        )
    if not bands:
        click.echo("ALL BANDS COMPLETE across reflex/tag/IPA/gloss.")
        return
    click.echo(f"{'impact':>6} {'total':>6} {'reflex':>7} {'tag':>5} {'ipa':>5} {'gloss':>6}")
    for b in bands[:top]:
        click.echo(
            f"{b['impact']:>6} {b['total']:>6} {b['reflex']:>7} {b['tag']:>5} "
            f"{b['ipa']:>5} {b['gloss']:>6}"
        )
    click.echo(f"top band with remaining work: impact={bands[0]['impact']}", err=True)


@enrich_campaign.command("identity-reflexes")
@_db_option
@_reflexes_option
@click.option(
    "--min-len",
    type=int,
    default=2,
    show_default=True,
    help="Skip folded surfaces shorter than this.",
)
@click.option(
    "--require-gloss/--no-require-gloss",
    default=True,
    show_default=True,
    help="Only emit for GLOSSED morphemes (legitimacy gate); --no- to include unglossed.",
)
def cmd_identity_reflexes(
    db_path: Path, reflexes_path: Path, min_len: int, require_gloss: bool
) -> None:
    """Emit one *identity reflex* per GLOSSED scholar morpheme whose folded
    canonical form appears in an attested toponym (the morpheme recognized by its
    own spelling), deduped against the ledger, as a JSON array. Bulk/retroactive.
    Unglossed morphemes are excluded by default — they get per-morpheme judgment in
    the normal loop. Pipe through ``validate`` then append to _reflexes.jsonl."""
    with _readonly_lexicon(db_path) as conn:
        rows = identity_reflex_rows(
            conn, reflexes_path, min_len=min_len, require_gloss=require_gloss
        )
    click.echo(json.dumps(rows, ensure_ascii=False))
    click.echo(
        f"emitted {len(rows)} identity reflexes (min_len={min_len}, require_gloss={require_gloss})",
        err=True,
    )


@enrich_campaign.command("collapse-validate")
@_db_option
@click.option(
    "--curation-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_GLOSS_PATH,
    show_default=True,
    help="Curation ledger (_curation.jsonl) carrying committed collapse rows.",
)
@click.option(
    "--candidates",
    "candidates_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def cmd_collapse_validate(db_path: Path, curation_path: Path, candidates_path: Path) -> None:
    """wyrd-eni4.3.2: validate authored ``collapse`` rows (garbled OCR-dupe → its
    correct existing twin). Gate: both refs resolve to distinct etymons, no '#'
    (intentional sense suffix), no duplicate. Exit 0 if all pass, 1 otherwise."""
    candidates = _load_candidates(candidates_path)
    with _readonly_lexicon(db_path) as conn:
        errors = validate_collapse_candidates(conn, curation_path, candidates)
    _report_validation(errors, len(candidates), "collapse row(s)")


def add_to(parent: click.Group) -> None:
    """Register ``enrich-campaign`` on the parent ``@lexicon`` group."""
    parent.add_command(enrich_campaign)
