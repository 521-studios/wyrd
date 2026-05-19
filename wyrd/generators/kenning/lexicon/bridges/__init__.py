"""Etymon-form bridging across languages + phonology layers.

Each pass tries to LINK an etymon row to a canonical lemma in
ANOTHER language or another phonological shape — closing the
cross-corpus gap that pure suffix-stripping (lemma_linking) can't
reach.

* ``generic`` — exact-form lookups across the lexicon (e.g. NF
  ``cot`` → OE ``cot``).
* ``celtic`` — Anglicized Celtic-substrate place-name forms
  (``Kil-``, ``Bally-``, ``Glen-``) → Welsh / Irish / Scottish-Gaelic
  canonical Celtic morphemes.
* ``oe`` — Old English Anglicized/modernized → scholarly-orthography
  lookups (``ton`` → ``tūn``, ``lea`` → ``lēah``) so attested toponym
  spellings link to their canonical OE etymons.
* ``on`` — same shape for Old Norse loanword forms in Northern
  English place names (``by`` → ``býr``, ``thwaite`` → ``þveit``).

The OE + ON passes share ``_bridge_same_language_phonological`` in
``_common`` — same flat ``dict.get()`` lookup engine, different
form→form lookup tables.

This re-export shim lets callers keep
``from wyrd.generators.kenning.lexicon.bridges import bridge_generic_language``
or grab the whole family through ``lexicon.__init__``.
"""

from wyrd.generators.kenning.lexicon.bridges.celtic import bridge_celtic_forms
from wyrd.generators.kenning.lexicon.bridges.generic import bridge_generic_language
from wyrd.generators.kenning.lexicon.bridges.oe import bridge_phonological_oe
from wyrd.generators.kenning.lexicon.bridges.on import bridge_phonological_on

__all__ = [
    "bridge_celtic_forms",
    "bridge_generic_language",
    "bridge_phonological_oe",
    "bridge_phonological_on",
]
