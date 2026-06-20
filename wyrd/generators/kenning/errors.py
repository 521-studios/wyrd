"""Kenning domain exceptions.

A small hierarchy so the web layer (``wyrd/app.py``) can tell a
valid-but-unsatisfiable request apart from a genuinely unexpected internal
error, instead of collapsing both into an opaque 500 (wyrd-hh2m).

These are deliberately NOT subclasses of ``ValueError`` — the dispatcher's
``except (ValueError, KeyError)`` clause maps those to a 400 ``bad_params``
(the contract ``UnknownCultureError(ValueError)`` relies on). An empty pool is
a different shape (well-formed request, nothing to serve), so it gets its own
base + a dedicated 422 mapping.
"""

from __future__ import annotations


class KenningError(Exception):
    """Base for kenning domain errors surfaced through the generate path."""


class EmptyEligiblePool(KenningError):
    """A valid request that the vector scorer can't satisfy: the gate predicates
    (culture / era / stratum / tags) — or thin bundle data — leave it with zero
    eligible morphemes, so the scorer returns no name.

    Distinct from an unexpected internal error: the request was well-formed, it
    just can't be satisfied. The web layer maps this to a 422 (reserving 500 for
    genuine crashes). The loud-fail behavior is preserved — the message still
    names the likely causes; it's only now classifiable.

    Raised for the empty-pool *no-name* outcome (the vector path's ``None``
    return); a genuine crash mid-scoring is NOT this and still surfaces as 500.
    wyrd-hh2m.
    """
