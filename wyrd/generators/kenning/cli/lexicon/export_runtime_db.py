"""``wyrd kenning lexicon export-runtime-db`` — emit the L4 runtime SQLite DB.

Thin click wrapper around
:func:`wyrd.generators.kenning.lexicon.runtime_db_export.write_runtime_db`
— see that module for the architecture rationale (D38).
"""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import (
    LexiconDB,
    collect_canonical_decompositions,
    collect_empirical_priors,
    collect_fantasy_morphemes,
    export_meanings,
)
from wyrd.generators.kenning.lexicon.genitive_priors import (
    build_split_probability_map,
    load_genitive_prior_counts,
)
from wyrd.generators.kenning.lexicon.runtime_db_export import (
    DEV_TOP_N_PER_CULTURE,
    EMITTER_VERSION,
    SCHEMA_VERSION,
    write_runtime_db,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

from ._threshold_args import parse_lang_thresholds

# Re-exports kept for tests that import the schema-version constants
# through this module path.
__all__ = ["EMITTER_VERSION", "SCHEMA_VERSION", "lexicon_export_runtime_db"]

# Canonical defaults for the upstream filter flags. Used in BOTH the
# @click.option ``default=`` declarations AND _reject_non_default_filters_under_dev
# so the two can't drift — bumping a default here propagates to both
# sites at once.
_DEFAULT_MIN_WITNESSES = 2
_DEFAULT_USE_PRESET = True
_DEFAULT_INCLUDE_RANDO = True
_DEFAULT_RANDO_MIN_CORROBORATORS = 0
_DEFAULT_INCLUDE_WIKTIONARY_EMPIRICAL = True
_DEFAULT_INCLUDE_WAVE2_ENRICHED = True
# wyrd-oth3: default OFF — the grader-validated re-emit (wyrd-aicu.9, 2026-06-16)
# showed the unfiltered admit regresses scholar agreement (coverage up but
# cluster recall/head down via composite morphemes). Opt-in until wyrd-myv4 +
# wyrd-h5u1 + wyrd-7hbp make it a graded net win.
_DEFAULT_INCLUDE_TOPONYM_BREAKDOWN = False


@click.command("export-runtime-db")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="L3 lexicon DB to read meanings + fantasy morphemes + canonicals from.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to write the L4 SQLite DB to. Overwritten if it exists.",
)
@click.option(
    "--proportions-dir",
    "proportions_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory containing <culture>_proportions.json files (operator "
        "override). Default behavior rebuilds the proportions inline from "
        "the L3 lexicon + bundled <culture>_place_names.json corpora, so "
        "no intermediate JSON artifact is required."
    ),
)
@click.option(
    "--min-witnesses",
    type=int,
    default=_DEFAULT_MIN_WITNESSES,
    show_default=True,
    help="D4 promotion threshold; passed through to export_meanings.",
)
@click.option(
    "--lang-threshold",
    "lang_threshold_specs",
    multiple=True,
    metavar="LANG=N",
    help="Per-language witness override; matches export-meanings semantics.",
)
@click.option(
    "--preset/--no-preset",
    "use_preset",
    default=_DEFAULT_USE_PRESET,
    show_default=True,
    help="Start from RECOMMENDED_LANG_THRESHOLDS (calibrated per language).",
)
@click.option(
    "--include-rando/--no-include-rando",
    default=_DEFAULT_INCLUDE_RANDO,
    show_default=True,
)
@click.option(
    "--rando-min-corroborators",
    type=int,
    default=_DEFAULT_RANDO_MIN_CORROBORATORS,
    show_default=True,
    help=(
        "wyrd-fssn: minimum non-rando-port citation sources a rando-cited "
        "family must have to be admitted via the rando-port path. 0 = "
        "current behavior."
    ),
)
@click.option(
    "--include-wiktionary-empirical/--no-include-wiktionary-empirical",
    default=_DEFAULT_INCLUDE_WIKTIONARY_EMPIRICAL,
    show_default=True,
)
@click.option(
    "--include-wave2-enriched/--no-include-wave2-enriched",
    default=_DEFAULT_INCLUDE_WAVE2_ENRICHED,
    show_default=True,
)
@click.option(
    "--include-toponym-breakdown/--no-include-toponym-breakdown",
    default=_DEFAULT_INCLUDE_TOPONYM_BREAKDOWN,
    show_default=True,
    help="wyrd-oth3: admit any etymon used in a scholarly toponym breakdown.",
)
@click.option(
    "--dev",
    "dev_subset",
    is_flag=True,
    default=False,
    help=(
        "Emit a committed-seed-shaped subset of the L4 (top N usages per "
        "culture + referenced meanings + all fantasy + all canonical, with "
        "fixed built_at + source_lexicon_db sentinels for byte-stability). "
        "Used to regenerate wyrd/generators/kenning/data/seed-runtime.db. "
        "All upstream filtering knobs (--min-witnesses, --lang-threshold, "
        "--include-rando, --include-wiktionary-empirical, "
        "--include-wave2-enriched) are FORCED to their canonical defaults "
        "in this mode so the seed is reproducible across operators — the "
        "CLI errors if you pass non-default values alongside --dev."
    ),
)
@click.option(
    "--dev-top-n",
    "dev_top_n",
    type=int,
    default=DEV_TOP_N_PER_CULTURE,
    show_default=True,
    help="Per-culture cap on usages / single_usages in --dev mode.",
)
@click.option(
    "--generation-subset",
    "generation_subset",
    is_flag=True,
    default=False,
    help=(
        "Emit the production cold-start trim (wyrd-ukq0): keep ALL "
        "proportion-referenced usages + their meanings, dropping the "
        "decomposition-only corpus (~70% of subjects generation never "
        "samples). Cuts the Lambda cold-load ~70%. Unlike --dev this is the "
        "FULL generatable set (no top-N cap) and carries real built_at / "
        "source_lexicon_db (not the dev sentinels). The arbitrary-name "
        "decompose/explain path degrades on this DB (deferred per wyrd-ukq0)."
    ),
)
def lexicon_export_runtime_db(
    db_path: Path,
    output_path: Path,
    proportions_dir: Path | None,
    min_witnesses: int,
    lang_threshold_specs: tuple[str, ...],
    use_preset: bool,
    include_rando: bool,
    rando_min_corroborators: int,
    include_wiktionary_empirical: bool,
    include_wave2_enriched: bool,
    include_toponym_breakdown: bool,
    dev_subset: bool,
    dev_top_n: int,
    generation_subset: bool,
) -> None:
    """Emit the L4 runtime SQLite DB.

    Self-contained snapshot of everything the kenning runtime needs:
    morpheme meanings (blob), fantasy morphemes (blob),
    canonical-decomposition front-loading (blob), and per-culture
    proportions distributions (normalized with precomputed cumulative
    columns so weighted-random sampling is an O(log n) index seek).

    Replaces ``meanings.json`` + 5 ``<culture>_proportions.json`` files
    in a single artifact downloaded from S3 on Lambda cold start (D38).
    """
    if dev_subset and generation_subset:
        # CLI-level guard for a friendly UsageError (parity with the --dev
        # filter rejection below). write_runtime_db also raises ValueError as
        # the defense-in-depth backstop for non-CLI callers.
        raise click.UsageError("--dev and --generation-subset are mutually exclusive")
    if dev_subset:
        _reject_non_default_filters_under_dev(
            min_witnesses=min_witnesses,
            lang_threshold_specs=lang_threshold_specs,
            use_preset=use_preset,
            include_rando=include_rando,
            rando_min_corroborators=rando_min_corroborators,
            include_wiktionary_empirical=include_wiktionary_empirical,
            include_wave2_enriched=include_wave2_enriched,
            include_toponym_breakdown=include_toponym_breakdown,
        )

    lang_thresholds = parse_lang_thresholds(lang_threshold_specs, use_preset=use_preset)

    with LexiconDB(db_path) as db:
        subjects = export_meanings(
            db,
            min_witnesses=min_witnesses,
            lang_thresholds=lang_thresholds,
            include_rando=include_rando,
            rando_min_corroborators=rando_min_corroborators,
            include_wiktionary_empirical=include_wiktionary_empirical,
            include_wave2_enriched=include_wave2_enriched,
            include_toponym_breakdown=include_toponym_breakdown,
        )
        canonical_decompositions = collect_canonical_decompositions(db)
        fantasy_morphemes = collect_fantasy_morphemes(db)
        empirical_priors_payload = collect_empirical_priors(db, version=EMITTER_VERSION)
        # wyrd-aicu.9 (D-3/Phase 4): precompute the genitive-split prob-map from
        # the L3 ``genitive_split_prior`` counts (smoothing + backoff applied
        # here), mirroring the grader's build. Empty when the table is missing /
        # unmined → no row emitted → runtime loader returns {} → connective
        # coverage still lands, homograph tiebreak goes silent.
        genitive_split_map = build_split_probability_map(load_genitive_prior_counts(db))

    counts = write_runtime_db(
        output_path=output_path,
        subjects=subjects,
        fantasy_morphemes=fantasy_morphemes,
        canonical_decompositions=canonical_decompositions,
        empirical_priors_payload=empirical_priors_payload,
        genitive_split_map=genitive_split_map,
        proportions_dir=proportions_dir,
        source_lexicon_db=db_path,
        dev_subset=dev_subset,
        generation_subset=generation_subset,
        dev_top_n_per_culture=dev_top_n,
    )

    proportion_total = (
        counts.get("proportions_usage", 0)
        + counts.get("proportions_single_usage", 0)
        + counts.get("proportions_structure", 0)
        + counts.get("proportions_tag_marginal", 0)
        + counts.get("proportions_tag_cooccurrence", 0)
        + counts.get("proportions_bare_word_position", 0)
    )
    click.echo(
        f"Wrote L4 DB to {output_path}: "
        f"{counts['meanings']} meaning rows, "
        f"{counts['fantasy_morphemes']} fantasy morphemes, "
        f"{counts['canonical_decompositions']} canonical decompositions, "
        f"{counts.get('genitive_split_pairs', 0)} genitive-split prior pairs, "
        f"{proportion_total} proportion rows across {counts['cultures']} cultures.",
        err=True,
    )

    # wyrd-04oi: pickle the decomposition matcher beside the FULL-corpus DB so
    # cold containers load it instead of rebuilding the ~90k-morpheme trie. The
    # trimmed generation / dev subsets never feed decomposition (explain / era-
    # map / rewind load the full corpus), so they get no sidecar.
    if not generation_subset and not dev_subset:
        sidecar = _build_matcher_sidecar(output_path)
        click.echo(f"Wrote decomposition matcher to {sidecar}", err=True)

    # wyrd-x7w4: --dev is the canonical "regenerate the committed seed-runtime.db"
    # invocation, and the committed per-language structures.yaml allowlist is
    # DERIVED from that seed's proportions (wyrd-hzqs). Regenerate it here too so
    # the two never drift (the test_committed_structures_yaml_in_sync_with_bundle
    # drift-guard would otherwise fail until the operator remembered to re-dump).
    # Refresh-merge keeps existing operator `enabled: false`s; the working-tree
    # file is left for the operator to review + commit (NEVER auto-committed).
    if dev_subset:
        structures_path = _regen_structure_allowlist(output_path)
        click.echo(f"Wrote structure allowlist to {structures_path}", err=True)


def _reject_non_default_filters_under_dev(
    *,
    min_witnesses: int,
    lang_threshold_specs: tuple[str, ...],
    use_preset: bool,
    include_rando: bool,
    rando_min_corroborators: int,
    include_wiktionary_empirical: bool,
    include_wave2_enriched: bool,
    include_toponym_breakdown: bool,
) -> None:
    """Hard-fail when --dev is combined with any non-default upstream
    filter. The committed seed-runtime.db is byte-stable only if every
    operator runs the same canonical invocation; silently honoring a
    custom --min-witnesses would produce a seed that differs from the
    one in the repo without the operator noticing.

    The canonical defaults come from the module-level ``_DEFAULT_*``
    constants — same source the @click.option ``default=`` declarations
    use, so a future bump to one of those propagates to both sites at
    once and can't drift.
    """
    # (this filter differs from canonical?, the conflict message) — one row per
    # upstream filter, in report order. Adding a filter is a one-line entry; the
    # message-vs-condition pairing stays visible instead of a wall of `if`s.
    checks: list[tuple[bool, str]] = [
        (
            min_witnesses != _DEFAULT_MIN_WITNESSES,
            f"--min-witnesses={min_witnesses} (canonical: {_DEFAULT_MIN_WITNESSES})",
        ),
        (
            bool(lang_threshold_specs),
            f"--lang-threshold {lang_threshold_specs!r} (canonical: none)",
        ),
        (use_preset != _DEFAULT_USE_PRESET, "--no-preset (canonical: --preset)"),
        (
            include_rando != _DEFAULT_INCLUDE_RANDO,
            "--no-include-rando (canonical: --include-rando)",
        ),
        (
            rando_min_corroborators != _DEFAULT_RANDO_MIN_CORROBORATORS,
            f"--rando-min-corroborators={rando_min_corroborators} "
            f"(canonical: {_DEFAULT_RANDO_MIN_CORROBORATORS})",
        ),
        (
            include_wiktionary_empirical != _DEFAULT_INCLUDE_WIKTIONARY_EMPIRICAL,
            "--no-include-wiktionary-empirical (canonical: --include-wiktionary-empirical)",
        ),
        (
            include_wave2_enriched != _DEFAULT_INCLUDE_WAVE2_ENRICHED,
            "--no-include-wave2-enriched (canonical: --include-wave2-enriched)",
        ),
        (
            include_toponym_breakdown != _DEFAULT_INCLUDE_TOPONYM_BREAKDOWN,
            "--include-toponym-breakdown (canonical: --no-include-toponym-breakdown)",
        ),
    ]
    offenders = [message for differs, message in checks if differs]
    if offenders:
        raise click.UsageError(
            "--dev requires canonical defaults on every upstream filter "
            "(otherwise the committed seed-runtime.db is unreproducible "
            "across operators). Conflicts:\n  - " + "\n  - ".join(offenders)
        )


def _regen_structure_allowlist(output_path: Path) -> Path:
    """wyrd-x7w4: regenerate the per-language ``structures.yaml`` allowlist from the
    just-written ``--dev`` seed DB and write it next to the DB.

    A refresh-merge over the CURRENT committed file (read via importlib.resources,
    so operator ``enabled: false`` curation stays sticky); new structures land
    enabled, retired ones drop, and the ``+new/-dropped`` summary prints to stderr.
    The working-tree file is left for the operator to review + commit — NEVER
    auto-committed.

    Unlike ``_build_matcher_sidecar`` (which must redirect ``WYRD_RUNTIME_DB`` +
    reset runtime caches because it builds the matcher through ``_load_meanings``),
    this reads the L4 ``proportions_*`` tables directly via the canonical
    read-only opener ``runtime_db._open_readonly`` — no global-cache surgery, and
    no raw sqlite3 in cli/lexicon (the package-boundary rule)."""
    from wyrd.generators.kenning.cli.dump_structures import (
        _build,
        _emit_summary,
        _read_merge_base,
        _render,
    )
    from wyrd.generators.kenning.runtime.runtime_db import _open_readonly

    structures_path = output_path.parent / "structures.yaml"
    # Merge base = the file we're about to OVERWRITE (the working-tree curation
    # next to the DB), not the importlib.resources copy. Under a non-editable
    # install those differ, and reading the installed copy would silently drop
    # local working-tree curation on write (Gemini review HIGH). Fall back to the
    # bundled resource (None) only when no sibling exists yet (first dump).
    merge_base_path = structures_path if structures_path.exists() else None
    conn = _open_readonly(output_path)
    try:
        sections, summary = _build(conn, _read_merge_base(merge_base_path))
    finally:
        conn.close()
    structures_path.write_text(_render(sections), encoding="utf-8")
    _emit_summary(summary)
    return structures_path


def _build_matcher_sidecar(output_path: Path) -> Path:
    """wyrd-04oi: build the decomposition ``MorphemeTrie`` from the just-written
    full-corpus DB + sidecars — exactly as the runtime loads it, so the stamp
    matches at cold-load — and pickle it to ``<output_path>.matcher.pkl``.

    Loads via the real runtime path (``WYRD_RUNTIME_DB`` pointed at the new DB)
    so the matcher is built from the SAME ``_load_meanings`` output a container
    sees (DB bundle + the four JSON sidecars). Resets the runtime caches before
    and after so neither this process's prior state nor this temporary
    redirection leaks into a later in-process load.
    """
    import os

    from wyrd.generators.kenning import _load_meanings
    from wyrd.generators.kenning.runtime import runtime_db as _runtime_db
    from wyrd.generators.kenning.runtime.matcher_cache import (
        dump_matcher,
        matcher_sidecar_path,
        matcher_stamp,
    )
    from wyrd.generators.kenning.runtime.trie_matcher import build_morpheme_trie

    def _reset_runtime_caches() -> None:
        _runtime_db.reset_runtime_db_cache()
        _load_meanings.cache_clear()
        _runtime_db_bundle_dict_cache_clear()

    prev = os.environ.get("WYRD_RUNTIME_DB")
    os.environ["WYRD_RUNTIME_DB"] = str(output_path)
    try:
        _reset_runtime_caches()
        meaning_db, _ = _load_meanings()
        trie = build_morpheme_trie(meaning_db)
        sidecar = matcher_sidecar_path(output_path)
        dump_matcher(trie, sidecar, stamp=matcher_stamp(meaning_db))
        return sidecar
    finally:
        if prev is None:
            os.environ.pop("WYRD_RUNTIME_DB", None)
        else:
            os.environ["WYRD_RUNTIME_DB"] = prev
        _reset_runtime_caches()


def _runtime_db_bundle_dict_cache_clear() -> None:
    """Clear the ``_runtime_db_bundle_dict`` lru_cache (the DB→bundle reader
    that ``_load_meanings`` builds on) so a redirected build sees the new DB."""
    from wyrd.generators.kenning import _runtime_db_bundle_dict

    _runtime_db_bundle_dict.cache_clear()


def add_to(parent: click.Group) -> None:
    """Register ``export-runtime-db`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_export_runtime_db)
