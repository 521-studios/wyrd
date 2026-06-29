"""Canonical decompositions export (consumed by the bundle build).

``collect_canonical_decompositions`` walks every
``toponym_decomposition`` row marked ``is_canonical = 1`` and
projects ``{modern_name: {signature, source}}`` — the lookup the
runtime ``KenningExplain`` uses to front-load the canonical
reading among the matcher's alternatives. The Norman manorial-
family token filter (``_load_norman_manorial_family_tokens``)
skips toponyms whose final whitespace-split token names a known
manorial family (``Castle Cary``, ``Stoke Mandeville``) — the
runtime's manorial-affix detector synthesizes a more specific
decomposition there than the build-time tiebreaker can.
"""

from __future__ import annotations

import json
from importlib import resources

from wyrd.generators.kenning.lexicon.db import LexiconDB


def _load_norman_manorial_family_tokens() -> frozenset[str]:
    """Return the surname-only tokens of Anglo-Norman manorial families
    (e.g. 'Cary', 'Lacy', 'Mandeville', 'Zouche'). Loaded from the
    canonical JSON at ``data/norman_manorial_families.json``.

    Used by :func:`collect_canonical_decompositions` to skip canonical
    picks for toponyms whose final whitespace-split token matches a
    known manorial family — those names get a runtime-synthesized
    decomposition by ``_norman_manorial_subjects`` in ``__init__.py``
    that the build-time tiebreaker shouldn't pre-empt.

    The token is the last whitespace-split word (matches the
    surname-only matching policy that ``_norman_manorial_subjects``
    documents).
    """
    data = resources.files("wyrd.generators.kenning.data").joinpath("norman_manorial_families.json")
    families = json.loads(data.read_text(encoding="utf-8"))
    return frozenset(family.split()[-1] for family in families)


def collect_canonical_decompositions(db: LexiconDB) -> dict[str, dict[str, str]]:
    """Project the lexicon's canonical decomposition picks into a
    bundle-shaped lookup keyed by toponym ``modern_name``.

    Used by ``lexicon export-meanings`` to emit a ``canonical_decompositions``
    field that ``KenningExplain`` (Lambda / SPA — no DB access) can read
    at runtime to mark and front-load the canonical reading among the
    matcher's alternatives (wyrd-h8k1).

    Output shape:
    ``{modern_name: {"signature": sha1_hex, "source": canonical_source}}``.

    Multiple toponym rows sharing a ``modern_name`` (one per region)
    collapse to the lex-first ``(region, id)`` ordered row's canonical.
    The Lambda has no region context at decomposition time so this
    matches ``load_names_with_regions``'s dedup policy: deterministic,
    same-name entries get the same canonical pick across re-runs.

    Toponyms whose final whitespace-split token matches a known
    Anglo-Norman manorial family ('Castle Cary', 'Stoke Mandeville',
    'Newton Lacy') are SKIPPED — the runtime manorial-affix detector
    in ``_norman_manorial_subjects`` synthesizes a more specific
    decomposition (e.g. ``castle + Cary (Norman manorial family)``)
    than the build-time tiebreaker can produce against DB-only
    morphemes. Without this guard, an enrichment pass that adds
    short morphemes (``ca``, ``ry``) lets the matcher perfectly
    decompose these names at export time, the canonical pick wins
    the rank-0 sort at runtime, and the manorial UX silently
    regresses. wyrd-j43l deploy-gate caught this on 'Castle Cary'.

    Returns an empty dict when the ``toponym_decomposition`` table
    doesn't exist yet — older DBs predate the wyrd-08m Phase 1 migration
    and shouldn't crash the bundle export. Run ``lexicon migrate`` to
    create the table, then ``lexicon decompose --apply`` to populate
    canonical picks.
    """
    table_exists = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='toponym_decomposition'"
    ).fetchone()
    if not table_exists:
        return {}
    manorial_tokens = _load_norman_manorial_family_tokens()
    rows = db.conn.execute(
        """
        SELECT t.modern_name,
               td.decomposition_signature,
               td.canonical_source
          FROM toponym t
          JOIN toponym_decomposition td ON td.toponym_id = t.id
         WHERE td.is_canonical = 1
         ORDER BY t.modern_name,
                  COALESCE(t.region, ''),
                  t.id,
                  td.id
        """
    ).fetchall()
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["modern_name"]
        if name in out:
            # Same-name multi-region collisions collapse to the
            # lex-first row's canonical; later rows skipped.
            continue
        tokens = name.split()
        if len(tokens) >= 2 and tokens[-1] in manorial_tokens:
            # Skip — runtime manorial-affix detector wins for these.
            # Requires ≥2 tokens because the manorial-affix shape is
            # ``<base> <family>``; a solo "Lacy" toponym would still
            # legitimately need a build-time canonical pick.
            continue
        out[name] = {
            "signature": row["decomposition_signature"],
            "source": row["canonical_source"] or "",
        }
    return out
