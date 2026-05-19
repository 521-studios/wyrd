"""The kenning ``registers/`` subpackage — register/mood/phonology surface.

Owns the "what does this name SOUND like / FEEL like" axis of the
generation system. Three concerns live here today; the wyrd-kq7w /
D36 rip-and-replace will fold them into a single catalog-driven
register-effect surface over time.

Modules:

* ``moods.py`` (extracted from ``kenning/__init__.py`` in wyrd-a83i) —
  ``MOODS`` dict: code-defined D6 mood presets (grim, harsh,
  pastoral, devotional, mortuary). Each entry bundles a semantic-tag
  union and/or a harshness float. Migrates to ``register_effects.yaml``
  entries under wyrd-kq7w.
* ``phonology.py`` — per-language IPA tables and phoneme-class
  membership lookups used by the harshness skew + cluster-density
  scoring.
* ``phonology_rules.py`` — the sound-change rule library used by
  the lexicon's era-reflex projection (ME→OE, NF→ME, etc.) and by
  the language-quality dashboard's rule-coverage metric.
* ``effects.py`` (was ``register_effects.py``) — register-effect
  catalog loader (D36.5 / wyrd-kq7w.2 Phase B). Reads
  ``data/register_effects.yaml`` and returns
  ``dict[str, RegisterEffect]``.

Future modules (deferred): ``catalog.py`` + ``composition.py`` for the
D36 vector-composition pipeline (wyrd-ecjp Phase 2-5).

Back-compat re-export covers ``MOODS`` so
``from wyrd.generators.kenning import MOODS`` keeps working
unchanged. Loaders + helpers from ``effects.py`` /
``phonology.py`` / ``phonology_rules.py`` were never imported
via the bare-module path, so no shim was needed for those.
"""

from wyrd.generators.kenning.registers.moods import MOODS  # noqa: F401
