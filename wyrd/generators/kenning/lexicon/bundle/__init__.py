"""Meaning bundle build pipeline (the meanings.json producer).

``export_meanings`` is the public entry point — it walks the
lexicon, groups etymons into families via cluster_cognates, picks a
canonical reflex per era cell, and emits the JSON the runtime
generators load.

The pipeline splits across four internal modules:

* ``_export`` — top-level orchestration: family selection, witness
  filtering, progress reporting, the ``export_meanings`` driver
  itself, and ``RECOMMENDED_LANG_THRESHOLDS``.
* ``_family`` — SQL-heavy data loaders: ``_gather_family`` +
  ``_fetch_member_*`` + ``_fetch_family_era_reflexes``. Leaf module;
  no internal deps.
* ``_subject`` — subject + word assembly: groups families into
  subjects, builds the per-word language buckets, and runs the
  per-member absorption pipeline.
* ``_emit`` — output emitters: ``_BucketAccumulator``,
  ``_emit_*_list`` formatters, ``_LANG_CODE_TO_JSON_FIELD`` rollup
  map, and ``_orphan_reflex_subjects``.

This re-export shim keeps existing
``from wyrd.generators.kenning.lexicon.bundle import export_meanings``
callers working and exposes the two public constants that callers
(``language_quality/audits.py``, ``runtime/decomposition.py``) read directly.
"""

from wyrd.generators.kenning.lexicon.bundle._emit import _LANG_CODE_TO_JSON_FIELD
from wyrd.generators.kenning.lexicon.bundle._export import (
    RECOMMENDED_LANG_THRESHOLDS,
    export_meanings,
)

__all__ = [
    "RECOMMENDED_LANG_THRESHOLDS",
    "_LANG_CODE_TO_JSON_FIELD",
    "export_meanings",
]
