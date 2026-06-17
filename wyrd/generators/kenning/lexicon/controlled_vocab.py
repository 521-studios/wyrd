"""Class-A controlled-vocabulary guards for the flat dimensions (wyrd-3q6m.5, D49).

Generalizes the region guard (``region_model.canonicalize_region``, which carries
a containment hierarchy) to the simpler **flat** controlled-vocabulary dimensions
— ``country`` and curated-etymon ``language`` — as a canonical enum + alias map +
fail-closed ``canonicalize_*`` at the ingest boundary.

Scope notes (the other two dimensions in the ticket need no guard here):

- **source-id** is already gated — every source is a registered ``source`` row.
- **era** is runtime-derived (from ``attested_year`` etc.), not a stored L2 field.

The **language** guard applies only to the *curated* place-name etymon path
(``ingest_parsed_entries``), which uses descriptive dashed names. The bulk
wiktextract ingest legitimately uses ISO/wiktextract codes (af, bar, gmw-cfr, …)
via the shared ``upsert_etymon`` primitive, so it is deliberately NOT guarded —
the guard sits at the parser-ingest boundary, not in ``upsert_etymon``.
"""

from __future__ import annotations

# --- country -----------------------------------------------------------------

# The canonical country values (post-wyrd-3q6m.4 the L2 country field is clean).
# A test pins this as a superset of every country regions.py can derive.
COUNTRY_CANONICAL = frozenset(
    {"England", "Scotland", "Wales", "Ireland", "Northern Ireland", "Isle of Man", "France"}
)

# Historical variants → canonical (the folds wyrd-3q6m.4 applied to the data).
COUNTRY_ALIASES = {
    "Republic of Ireland": "Ireland",
    "The Isle of Man": "Isle of Man",
    "Brittany": "France",
}


class CountryValidationError(ValueError):
    """Raised at the ingest boundary for a country value that is neither
    canonical nor a known alias (fail-closed; wyrd-3q6m.5 / D49)."""

    def __init__(self, country: str):
        self.country = country
        super().__init__(
            f"unknown country {country!r}: add it to COUNTRY_CANONICAL or COUNTRY_ALIASES"
        )


def canonicalize_country(country: str | None) -> str | None:
    """Fold a known alias, pass a canonical value, ``None``/empty → ``None``;
    raise :class:`CountryValidationError` on anything else."""
    if not country:
        return None
    if country in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[country]
    if country in COUNTRY_CANONICAL:
        return country
    raise CountryValidationError(country)


# --- language (curated place-name etymons only) ------------------------------

# The descriptive dashed-name vocabulary used by the curated parser-ingest path.
# Fail-closed: a new place-name language (welsh, irish, …) must be added here
# consciously when its sources are first mined (the Class-A discipline) — it is
# not silently accepted, and a stray wiktextract code can't leak into curated L2.
LANGUAGE_CANONICAL = frozenset(
    {
        "old-english",
        "middle-english",
        "modern-english",
        "old-norse",
        "old-french",
        "norman-french",
        "latin",
        "medieval-latin",
        "celtic",
        "germanic",
        "continental-germanic",
        "proto-germanic",
        "greek",
        "ancient-greek",
        "proto-indo-european",
        "biblical",
        "unknown",
    }
)


class LanguageValidationError(ValueError):
    """Raised at the curated-etymon ingest boundary for a language value not in
    the canonical set (fail-closed; wyrd-3q6m.5 / D49)."""

    def __init__(self, language: str):
        self.language = language
        super().__init__(
            f"unknown curated-etymon language {language!r}: add it to LANGUAGE_CANONICAL"
        )


def canonicalize_language(language: str | None) -> str | None:
    """Validate a curated place-name etymon language. ``None``/empty → ``None``;
    a canonical value passes; anything else raises
    :class:`LanguageValidationError`. (Bulk wiktextract ingest does NOT call this
    — see the module docstring.)"""
    if not language:
        return None
    if language in LANGUAGE_CANONICAL:
        return language
    raise LanguageValidationError(language)
