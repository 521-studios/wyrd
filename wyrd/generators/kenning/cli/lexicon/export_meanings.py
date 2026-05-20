"""``wyrd kenning lexicon export-meanings`` — emit the runtime meanings.json bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import (
    LANGUAGE_FIELDS,
    RECOMMENDED_LANG_THRESHOLDS,
    LexiconDB,
    collect_canonical_decompositions,
    collect_fantasy_morphemes,
    export_meanings,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("export-meanings")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write to this path instead of stdout.",
)
@click.option(
    "--min-witnesses",
    type=int,
    default=3,
    show_default=True,
    help="Promote etymons with this many distinct scholar witnesses (D4). "
    "Used as the fallback for languages not covered by --lang-threshold or the preset.",
)
@click.option(
    "--lang-threshold",
    "lang_threshold_specs",
    multiple=True,
    metavar="LANG=N",
    help="Per-language witness threshold override (repeatable). Example: "
    "--lang-threshold old-norse=2. Merges with the recommended preset; pass "
    "--no-preset to start from an empty map (then --min-witnesses applies uniformly).",
)
@click.option(
    "--preset/--no-preset",
    "use_preset",
    default=True,
    show_default=True,
    help="Start from RECOMMENDED_LANG_THRESHOLDS (calibrated against corpus availability). "
    "--no-preset uses --min-witnesses uniformly across all languages.",
)
@click.option(
    "--include-rando/--no-include-rando",
    default=True,
    show_default=True,
    help="Include rando-port-cited families regardless of witness count (D4 legacy).",
)
@click.option(
    "--include-wiktionary-empirical/--no-include-wiktionary-empirical",
    default=True,
    show_default=True,
    help="Include 'wiktionary-empirical'-cited families regardless of witness "
    "count (wyrd-4hx7 corpus-mined gap-fills, treated like rando-port).",
)
@click.option(
    "--joiners-from",
    "joiners_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional path to a joiners-pool sidecar JSON (wyrd-q0g6 Phase 2). "
        'Shape: {lang_field: [{"form": str, "weight": int}, ...], ...}. '
        "When provided + non-empty, the bundle's ``joiners`` key is "
        "populated; the runtime joiner_density knob then has data to act on."
    ),
)
def lexicon_export_meanings(
    db_path: Path,
    output_path: Path | None,
    min_witnesses: int,
    lang_threshold_specs: tuple[str, ...],
    use_preset: bool,
    include_rando: bool,
    include_wiktionary_empirical: bool,
    joiners_path: Path | None,
) -> None:
    """Export the lexicon DB as a meanings.json document for the runtime.

    Inverse of ``lexicon build``: walks the authoring DB, applies the D4
    promotion rule, and emits the JSON shape the kenning generator expects.
    Inflected variants (D8) and OCR-cluster losers (D22) ride along in their
    lemma's language form list rather than appearing as separate subjects.

    Per-language witness thresholds are calibrated against corpus availability:
    well-mined languages (old-english) gate at 3 witnesses to keep Tier-1
    extraction noise out, while corpus-thin languages (old-norse, latin) gate
    at 2. See RECOMMENDED_LANG_THRESHOLDS in lexicon.py for rationale.
    """
    lang_thresholds = _parse_lang_thresholds(lang_threshold_specs, use_preset=use_preset)
    with LexiconDB(db_path) as db:
        subjects = export_meanings(
            db,
            min_witnesses=min_witnesses,
            lang_thresholds=lang_thresholds,
            include_rando=include_rando,
            include_wiktionary_empirical=include_wiktionary_empirical,
        )
        canonical_decompositions = collect_canonical_decompositions(db)
        fantasy_morphemes = collect_fantasy_morphemes(db)

    joiners: dict[str, list[dict[str, Any]]] = {}
    if joiners_path is not None:
        joiners = _load_joiners_sidecar(joiners_path)

    # wyrd-c1vq: bundle is always dict-shape going forward. The
    # runtime loaders (load_meanings, load_canonical_decompositions,
    # load_joiners, load_fantasy_morphemes) tolerate both list-shape
    # (legacy) and dict-shape input, so flipping the default doesn't
    # break consumers; future field additions just become new keys
    # instead of re-deciding the shape each time. Optional fields
    # only appear when populated so empty-dict noise stays out of the
    # rendered JSON.
    bundle: dict[str, Any] = {"subjects": subjects}
    if canonical_decompositions:
        bundle["canonical_decompositions"] = canonical_decompositions
    if joiners:
        bundle["joiners"] = joiners
    if fantasy_morphemes:
        bundle["fantasy_morphemes"] = fantasy_morphemes

    payload = json.dumps(bundle, ensure_ascii=False, indent=2)
    if output_path is None:
        click.echo(payload)
    else:
        output_path.write_text(payload + "\n")
        notes = []
        if canonical_decompositions:
            notes.append(f"{len(canonical_decompositions)} canonical decompositions")
        if joiners:
            joiner_total = sum(len(entries) for entries in joiners.values())
            notes.append(f"{joiner_total} joiners across {len(joiners)} fields")
        if fantasy_morphemes:
            notes.append(f"{len(fantasy_morphemes)} fantasy morphemes")
        suffix = " + " + " + ".join(notes) if notes else ""
        click.echo(
            f"Wrote {len(subjects)} subjects{suffix} to {output_path}",
            err=True,
        )


def _parse_lang_thresholds(specs: tuple[str, ...], *, use_preset: bool) -> dict[str, int]:
    """Parse a tuple of ``LANG=N`` strings into a {lang_code: threshold}
    dict, starting from the recommended preset when ``use_preset`` is
    True. Used by both ``export-meanings`` and ``diff-bundle`` so the
    flag shape can't drift across them.

    Accepts JSON-field aliases (``old_english`` → ``old-english``,
    ``celtic_mix`` → ``celtic``) via LANGUAGE_FIELDS. Without the
    alias step an override like ``--lang-threshold old_english=2``
    silently fails to match any consensus row and the preset fallback
    applies instead."""
    lang_thresholds: dict[str, int] = dict(RECOMMENDED_LANG_THRESHOLDS) if use_preset else {}
    for spec in specs:
        if "=" not in spec:
            raise click.BadParameter(f"--lang-threshold expects LANG=N, got {spec!r}")
        lang, _, n_str = spec.partition("=")
        lang = lang.strip()
        n_str = n_str.strip()
        if not lang or not n_str:
            raise click.BadParameter(
                f"--lang-threshold {spec!r}: both LANG and N must be non-empty"
            )
        try:
            n = int(n_str)
        except ValueError as exc:
            raise click.BadParameter(f"--lang-threshold {spec!r}: N must be an integer") from exc
        lang = LANGUAGE_FIELDS.get(lang, lang)
        lang_thresholds[lang] = n
    return lang_thresholds


def _load_joiners_sidecar(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read + validate a joiners-pool sidecar JSON.

    Expected shape: ``{lang_field: [{"form": str, "weight": int}, ...], ...}``.
    Empty dict returned if the file is empty or shaped wrong; specific
    issues raised as ``click.ClickException`` so the operator gets a
    pointed error rather than a JSON parse traceback."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"--joiners-from {path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise click.ClickException(
            f"--joiners-from {path}: expected top-level dict keyed by lang_field"
        )
    out: dict[str, list[dict[str, Any]]] = {}
    for lang_field, entries in data.items():
        if not isinstance(entries, list):
            raise click.ClickException(
                f"--joiners-from {path}: entries for {lang_field!r} must be a list"
            )
        validated: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict) or "form" not in entry or "weight" not in entry:
                raise click.ClickException(
                    f"--joiners-from {path}: each entry under {lang_field!r} must "
                    "be a dict with 'form' and 'weight' keys"
                )
            validated.append({"form": entry["form"], "weight": entry["weight"]})
        if validated:
            out[lang_field] = validated
    return out


def add_to(parent: click.Group) -> None:
    """Register ``export-meanings`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_export_meanings)
