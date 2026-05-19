"""The kenning ``runtime/`` subpackage — request-time generation surface.

Owns the bundle-side files the kenning Generator subclasses consume
at request time (Lambda + SPA generation path). NO lexicon DB access
— every module here reads only the deploy bundle (meanings.json +
sidecars) and stdlib primitives.

This is the explicit complement to ``lexicon/`` (authoring side) and
``bundle/`` (developer-facing reports). The runtime/ vs bundle/ vs
lexicon/ split:

* ``lexicon/`` — authoring side; WRITES the lexicon DB (mining,
  enrichment, bundle build).
* ``bundle/`` — developer-facing reports; READS the lexicon DB to
  surface coverage / readiness / browse info.
* ``runtime/`` — request-time generation; reads ONLY the deploy
  bundle (meanings.json), no DB access. The Lambda has no DB
  connection — only this subpackage runs there.

Modules:

* ``meaning.py`` — ``Meaning`` (per-morpheme bundled data) +
  ``load_meanings`` / ``load_canonical_decompositions`` /
  ``load_joiners`` / ``load_fantasy_morphemes``.
* ``name.py`` — ``Name`` (generated-name container).
* ``word.py`` — ``Word`` (single token within a Name).
* ``decomposition.py`` — ``decompose_with_canonical`` + matcher API +
  ``_decomposition_payload`` / ``_signature_for_payload``.
* ``trie_matcher.py`` — trie-indexed segmentation DAG (D29 — the
  trie-matched morpheme decomposition pipeline).
* ``proportions.py`` — per-culture proportions loader
  (``load_proportions``).
* ``respelling.py`` — ``Meaning.respelling_for`` delegation target
  (D34, wyrd-17t; mood/era/lang-driven respelling overlays).
* ``scripts.py`` — alt-script transliteration (D34, wyrd-y10;
  Shavian + future scripts).

Back-compat: the kenning shim ``__init__.py`` re-exports the
public symbols (``Meaning``, ``Word``, ``load_meanings``, etc.) at
the package level so ``from wyrd.generators.kenning import
Meaning`` keeps working unchanged. New code may import from
``wyrd.generators.kenning.runtime.<module>`` directly.
"""
