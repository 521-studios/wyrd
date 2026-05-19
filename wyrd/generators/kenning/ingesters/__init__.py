"""The kenning ``ingesters/`` subpackage — operator-side data ingesters.

Owns the CSV-and-gazetteer → JSONL pipeline for the official British
historical records that feed the lexicon's toponym + attestation
surface. Each module here parses a specific source format and emits
canonical-state JSONL events via ``jsonl/log.py``.

Modules:

* ``hundred_rolls.py`` (was ``hundred_rolls_ingester.py``) — 1279/1280
  Hundred Rolls (CSV).
* ``speed_1611.py`` (was ``speed_1611_ingester.py``) — John Speed's
  1611 county-map gazetteer (CSV).
* ``hearth_tax.py`` (was ``hearth_tax_ingester.py``) — 1660s-70s Hearth
  Tax returns (CSV).
* ``os_open_names.py`` (was ``os_open_names_ingester.py``) — Ordnance
  Survey OpenNames (CSV; bulk modern gazetteer).
* ``domesday.py`` (was ``domesday_ingester.py``) — Open Domesday
  (Hull team CC-BY-NC-SA, 1086 survey).

Shared-base extraction (operator-CSV base class) is deferred to
``wyrd-29mn`` per rule-of-N: the three operator-CSV ingesters
(hundred_rolls, speed_1611, hearth_tax) share a clear shape but the
4th instance is the trigger; os_open_names + domesday differ enough
in column-set / row-format that they don't yet pin down the abstraction.

CLI consumers live at ``wyrd/generators/kenning/cli/lexicon/``
(``ingest_hundred_rolls.py``, ``ingest_speed_1611.py``,
``ingest_hearth_tax.py``, ``ingest_os_open_names.py``,
``ingest_domesday.py``) — each is a thin click wrapper around the
corresponding module here.

Note: ``wiktextract_ingester.py`` and ``etymonline_ingester.py`` are
NOT in this subpackage — they live under ``lexicon/`` since they
write directly to the lexicon DB as enrichment passes, not via the
JSONL event log.
"""
