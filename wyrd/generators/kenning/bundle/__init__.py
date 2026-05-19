"""The kenning ``bundle/`` subpackage — deploy-bundle browse + reports.

Owns the operator-facing surface that READS the lexicon DB (or the
shipped meanings.json bundle) to produce browse helpers and
reporting. These are developer-facing — none of this code is consumed
by the Lambda Generator subclasses at request time.

Naming note: ``kenning/bundle/`` (this package) is conceptually
distinct from ``kenning/lexicon/bundle/`` (which lives one level
deeper and owns the AUTHORING side: building the deploy bundle from
the lexicon DB). Both names are correct for their context — this
package consumes the bundle / DB to surface information; the
lexicon-side package builds the bundle in the first place.

Modules:

* ``browse.py`` — read-only DB browse helpers (``parse_toponym_ref``,
  ``fetch_toponym_detail``, ``format_toponym``, etc.). Consumed by
  ``cli/lexicon/browse/*`` click subcommands and by ``era/timeline.py``.
* ``rando_port_readiness.py`` — Rando port readiness reporting
  (toponym coverage analytics for the rando-server adopter).
* ``wikipedia_backfill_report.py`` — Wikipedia-seed retirement
  report (tracks the empirical case for retiring the Wikipedia
  scrape sources as wiktextract + mining-LLM fills the same surface).

Split between ``bundle/`` and ``runtime/`` (wyrd-7cg7): ``runtime/``
owns code the Lambda Generator subclasses consume at request time
(meaning.py, name.py, decomposition.py, trie_matcher.py, etc.);
``bundle/`` owns code that consumes the DB / bundle to produce
developer-facing reports.
"""
