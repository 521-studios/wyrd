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

# Invariant: every alias must resolve to a canonical country, else a typo'd
# target would silently bypass the guard (alias is resolved before the canonical
# check). Enforced at import via an explicit raise (not assert — survives -O).
_BAD_ALIAS_TARGETS = set(COUNTRY_ALIASES.values()) - COUNTRY_CANONICAL
if _BAD_ALIAS_TARGETS:
    raise RuntimeError(f"COUNTRY_ALIASES targets not canonical: {_BAD_ALIAS_TARGETS}")


class CountryValidationError(ValueError):
    """Raised at the ingest boundary for a country value that is neither
    canonical nor a known alias (fail-closed; wyrd-3q6m.5 / D49)."""

    def __init__(self, country: str):
        self.country = country
        super().__init__(
            f"unknown country {country!r}: add it to COUNTRY_CANONICAL or COUNTRY_ALIASES"
        )


def canonicalize_country(country: str | None) -> str | None:
    """Fold a known alias, pass a canonical value, ``None``/empty/whitespace →
    ``None``; raise :class:`CountryValidationError` on anything else. Surrounding
    whitespace is stripped before lookup."""
    country = (country or "").strip()
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
        # Celtic (wyrd-xdam): the fine-grained per-language names ARE the real
        # cultural-boundary DATA and are first-class canonical — Welsh, Cornish and
        # Breton are distinct cultures and must never be flattened into each other
        # in the data (wyrd-xdam.4). 'brittonic'/'goidelic'/'celtic' are kept as
        # PRESENTATION ROLLUP tokens — the generation-pool boundary is the BRANCH
        # (brittonic morphemes never pool with goidelic), and the rollup is derived
        # for display via era.cells.LANGUAGE_TO_FAMILY (wyrd-xdam.3). A rollup token
        # is used as an etymon's own language only when the finer grain isn't known
        # (e.g. wyrd-xdam.2 assigned a branch to flat 'celtic' L2 etymons because
        # the toponym's country only YIELDS a branch — not because curated data must
        # be coarse). 'celtic' is the coarsest rollup (branch unknown).
        "celtic",
        "brittonic",
        "goidelic",
        # Brittonic branch — fine-grained cultural-boundary languages
        "welsh",
        "old-welsh",
        "middle-welsh",
        "modern-welsh",
        "cornish",
        "breton",
        "old-breton",
        "middle-breton",
        # Goidelic branch — fine-grained cultural-boundary languages
        "irish",
        "old-irish",
        "middle-irish",
        "scottish-gaelic",
        "manx",
        # Proto ancestor (mirrors proto-germanic / proto-indo-european below)
        "proto-celtic",
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


def canonicalize_language(language: str | None) -> str:
    """Validate a curated place-name etymon language → a canonical, non-empty
    ``str`` (surrounding whitespace stripped). Anything not in the canonical set
    — including ``None``/empty — raises :class:`LanguageValidationError`.

    Unlike country, language is NEVER ``None``: it is part of the etymon dedup key
    ``(canonical_form, language)``, where a SQLite NULL would defeat the unique
    constraint. A genuinely-unknown etymology uses the canonical ``"unknown"``,
    not a null. (Bulk wiktextract ingest does NOT call this — see module docstring.)"""
    candidate = (language or "").strip()
    if candidate in LANGUAGE_CANONICAL:
        return candidate
    raise LanguageValidationError(candidate)
