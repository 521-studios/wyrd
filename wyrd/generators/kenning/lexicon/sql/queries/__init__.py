"""Named SQL queries used by the lexicon DB layer.

One file per table the queries touch. Split this finely so a reviewer
asking "what writes to ``etymon_citation``?" can answer it by reading
a 30-line file, not by grepping a 7,000-line module. The
``LexiconDB`` methods import the named constants directly; new
queries land here first, then a method on the class (or a free
function on the table module) drives them.

Forward direction: more enrichment-pass queries will extract here
from the legacy ``lexicon/__init__.py`` body as the wyrd-67fv split
progresses. New files follow the existing per-table shape.
"""

from wyrd.generators.kenning.lexicon.sql.queries.etymon import (
    INSERT_GLOSS_OR_IGNORE,
    INSERT_TAG_OR_IGNORE,
    UPSERT_ETYMON,
)
from wyrd.generators.kenning.lexicon.sql.queries.etymon_citation import (
    INSERT_CITATION_IF_ABSENT,
    citation_params,
)
from wyrd.generators.kenning.lexicon.sql.queries.reflex import (
    LINK_REFLEX_ETYMON_OR_IGNORE,
    UPSERT_REFLEX,
)
from wyrd.generators.kenning.lexicon.sql.queries.source import UPSERT_SOURCE
from wyrd.generators.kenning.lexicon.sql.queries.stats import (
    STATS_COUNT_TEMPLATE,
    STATS_TABLES,
)

__all__ = [
    "INSERT_CITATION_IF_ABSENT",
    "INSERT_GLOSS_OR_IGNORE",
    "INSERT_TAG_OR_IGNORE",
    "LINK_REFLEX_ETYMON_OR_IGNORE",
    "STATS_COUNT_TEMPLATE",
    "STATS_TABLES",
    "UPSERT_ETYMON",
    "UPSERT_REFLEX",
    "UPSERT_SOURCE",
    "citation_params",
]
