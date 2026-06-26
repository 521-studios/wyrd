"""Shared ``--lang-threshold LANG=N`` parser for the export CLIs.

Lives here, imported by ``export-meanings`` / ``diff-bundle`` /
``export-runtime-db``, rather than duplicated per command — the two former
in-file copies were byte-identical and their docstrings each warned that the
flag shape must not drift across the commands. One source of truth makes that
drift impossible (the same derive-from-one-source discipline as the etymon-key
normalization and the runtime-export column derivation).
"""

from __future__ import annotations

import click

from wyrd.generators.kenning.lexicon import LANGUAGE_FIELDS, RECOMMENDED_LANG_THRESHOLDS


def parse_lang_thresholds(specs: tuple[str, ...], *, use_preset: bool) -> dict[str, int]:
    """Parse a tuple of ``LANG=N`` strings into a ``{lang_code: threshold}``
    dict, starting from the recommended preset when ``use_preset`` is True.

    Accepts JSON-field aliases (``old_english`` → ``old-english``,
    ``celtic_mix`` → ``celtic``) via ``LANGUAGE_FIELDS``. Without the alias step
    an override like ``--lang-threshold old_english=2`` silently fails to match
    any consensus row and the preset fallback applies instead.
    """
    lang_thresholds: dict[str, int] = dict(RECOMMENDED_LANG_THRESHOLDS) if use_preset else {}
    for spec in specs:
        if "=" not in spec:
            raise click.BadParameter(f"--lang-threshold expects LANG=N, got {spec!r}")
        lang, _, n_str = spec.partition("=")
        # Lower-case before the LANGUAGE_FIELDS alias lookup: every alias key and
        # every canonical language code is lower-case, so a mixed-case override
        # (``Old_English=2`` / ``OLD_ENGLISH=2``) would otherwise fall through
        # the alias map and silently fail to match any consensus row (the preset
        # fallback applies instead) — the same silent-operator-input gap the
        # alias step itself exists to close.
        lang = lang.strip().lower()
        n_str = n_str.strip()
        if not lang or not n_str:
            raise click.BadParameter(
                f"--lang-threshold {spec!r}: both LANG and N must be non-empty"
            )
        try:
            n = int(n_str)
        except ValueError as exc:
            raise click.BadParameter(f"--lang-threshold {spec!r}: N must be an integer") from exc
        if n < 0:
            # A witness threshold is a count of distinct scholar witnesses; a
            # negative value would silently admit every morpheme (N >= -1 is
            # always true), disabling the filter for that language instead of
            # erroring. 0 stays valid — an explicit "no minimum / admit all".
            raise click.BadParameter(f"--lang-threshold {spec!r}: N must be non-negative, got {n}")
        lang = LANGUAGE_FIELDS.get(lang, lang)
        lang_thresholds[lang] = n
    return lang_thresholds
