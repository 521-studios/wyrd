"""The kenning ``era/`` subpackage — D33 time-aware era-reflex infrastructure.

Owns the temporal-axis machinery: era cells (per-family year ranges),
the runtime era-reflex picker, multi-era name rewinding, and the
toponym-attestation timeline + coverage reports.

Modules:

* ``cells.py`` (was ``era.py``) — ``era_cell`` / ``era_cell_for_input``
  / ``language_family`` / ``era_cells_for_family`` /
  ``resolve_era_input`` / ``canonical_language_for_cell``. The era-cell
  primitive and CLI/API input parser. See D5-3 + D33.
* ``rewind.py`` — ``rewind_name``, ``DEFAULT_ENGLISH_ERAS``. The
  CLI-side era-rewinder (called by ``wyrd kenning rewind`` and the
  ``KenningRewind`` Generator class).
* ``timeline.py`` (was ``era_timeline.py``) — ``fetch_era_timeline`` /
  ``fetch_era_coverage`` / formatters. The per-toponym timeline +
  cross-corpus coverage report (wyrd-ub76).

Lexicon-side era helpers (``etymon_era_reflexes``, the three-tier
picker, ``etymon_period_form`` projection) stay in
``wyrd.generators.kenning.lexicon`` since they read/write the
lexicon DB. This subpackage owns the runtime + CLI surfaces;
``lexicon/`` owns the DB-side data.
"""

# Re-export the cells-module symbols at the package level so callers
# that ``from wyrd.generators.kenning import era`` and then use
# ``era.era_cell(...)`` / ``era.language_family(...)`` keep working
# without per-call-site updates. The runtime API surface defined in
# ``cells.py`` is the long-stable side of D5-2 / D33.
from wyrd.generators.kenning.era.cells import (  # noqa: F401
    CANONICAL_LANGUAGE_FOR_CELL,
    ERA_CELLS,
    LANGUAGE_TO_FAMILY,
    all_families,
    canonical_language_for_cell,
    era_cell,
    era_cell_for_family,
    era_cell_for_input,
    era_cells_for_family,
    era_year_range,
    language_family,
    resolve_era_input,
)
