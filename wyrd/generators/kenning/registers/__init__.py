"""The kenning ``registers/`` subpackage — register/mood/phonology surface.

Owns the "what does this name SOUND like / FEEL like" axis of the
generation system.

Modules:

* ``effects.py`` (was ``register_effects.py``) — register-effect
  catalog loader (D36.5 / wyrd-kq7w.2 Phase B). Reads
  ``data/register_effects.yaml`` and returns
  ``dict[str, RegisterEffect]``. Owns the catalog-driven mood
  resolution (``parse_mood_spec``, ``mood_spec_to_legacy_form``,
  ``available_register_effects``) since wyrd-kq7w.3.
* ``phonology.py`` — per-language IPA tables and phoneme-class
  membership lookups used by the harshness skew + cluster-density
  scoring.
* ``phonology_rules.py`` — the sound-change rule library used by
  the lexicon's era-reflex projection (ME→OE, NF→ME, etc.) and by
  the language-quality dashboard's rule-coverage metric.
* ``phonological_vector_compute.py`` — IPA → PhonologicalVector
  computation (wyrd-kq7w.1). Used by the enrichment pass + by the
  bundle exporter to attach per-lemma vectors.

Historical: ``moods.py`` (extracted in wyrd-a83i, ripped in
wyrd-kq7w.3) once held a code-defined MOODS dict mirroring D6 mood
presets. The catalog at ``data/register_effects.yaml`` is now the
single source of truth for mood resolution; ``effects.parse_mood_spec``
+ ``effects.mood_spec_to_legacy_form`` cover the lookup + legacy-
shape translation.
"""
