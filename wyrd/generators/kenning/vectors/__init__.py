"""The kenning ``vectors/`` subpackage — D36 vector-driven generation.

Owned by the wyrd-ecjp epic. Currently:

* ``schemas.py`` — ``PhonologicalVector`` + 14 named feature dims,
  ``SemanticTagWeights``, ``PositionPrior``, ``ScoringWeights`` —
  in-memory representation locked by D36.1.
* ``scoring.py`` — the canonical composition rule (D36.2):
  ``score(lemma) = phon_w * phon + sem_w * sem + pos_w * pos
                 + base_w * baseline(lemma, source, request)``

Phase 2-5 of wyrd-ecjp will add ``priors.py`` (the empirical-baseline
artifact loader), ``eligibility.py`` (the hard-gate runtime; some of
this currently lives at top-level ``eligibility.py`` and will move
in), and ``name_generator.py`` (the slot-walk rewrite). All of those
land as new modules in this package.
"""
