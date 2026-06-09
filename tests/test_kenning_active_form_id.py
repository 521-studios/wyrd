"""wyrd-i4jd: id-first active-form emission.

The generator stamps ``active_form_id`` on each morpheme — the stable grid-cell
id of the form it ACTUALLY rendered — so the SPA highlights by id instead of a
brittle cross-grid surface fold. These tests pin the contract: the id resolves
to a real cell in that morpheme's OWN era_grid, and that cell's form matches the
rendered surface. Fixture-safe: uses the committed seed-runtime.db bundle via
``Kenning()`` (no live DB).
"""

from __future__ import annotations

import unicodedata

from wyrd.generators.kenning import Kenning


def _fold(s: str) -> str:
    """Accent/dash/case fold — mirrors the grid's _grid_surface_key + accent strip."""
    decomposed = unicodedata.normalize("NFD", (s or "").strip("-").replace("*", ""))
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn").lower()


def _cell_by_id(era_grid, cid):
    for section in era_grid or []:
        for stage in section.get("stages", []):
            for cell in stage.get("forms", []):
                if cell.get("id") == cid:
                    return cell
    return None


def _gen(params, seed):
    res = Kenning().generate(params, seed)
    return res[0] if isinstance(res, list) else res


def _check_result(r):
    """Every morpheme that carries an active_form_id must resolve it to a real
    cell in its OWN era_grid, and that cell's form must fold to the rendered
    surface (or, when there is no distinct render, the modern usage)."""
    seen_any = False
    for word in r.morphemes_by_word or []:
        for m in word:
            cid = m.get("active_form_id")
            if cid is None:
                continue
            seen_any = True
            cell = _cell_by_id(m.get("era_grid"), cid)
            assert cell is not None, (
                f"active_form_id {cid!r} not in this morpheme's era_grid "
                f"(usage={m.get('usage')!r})"
            )
            active_surface = m.get("rendered") or m.get("usage")
            assert _fold(cell["form"]) == _fold(active_surface), (
                f"active cell {cid!r} form={cell['form']!r} does not fold to the "
                f"rendered surface {active_surface!r}"
            )
    return seen_any


def test_active_form_id_resolves_to_a_real_cell_native():
    """era='' (native default): the rendered native form is one of the picked
    morpheme's OWN cells, referenced by active_form_id."""
    any_seen = False
    for seed in range(12):
        any_seen |= _check_result(_gen({"culture": "english", "count": 1}, seed))
    assert any_seen, "no morpheme emitted an active_form_id across 12 native rolls"


def test_active_form_id_resolves_to_a_real_cell_era():
    """An explicit era render: the rendered era form resolves to its own grid
    cell by id (the case the SPA highlight used to get wrong via surface fold)."""
    any_seen = False
    for seed in range(12):
        any_seen |= _check_result(_gen({"culture": "english", "count": 1, "era": "oe-late"}, seed))
    assert any_seen, "no morpheme emitted an active_form_id across 12 era rolls"


def test_active_form_id_present_for_morphemes_with_grid():
    """A morpheme that has an era_grid AND a rendered form should carry an
    active_form_id (the highlight target) — the whole point of id-first."""
    misses = []
    for seed in range(20):
        r = _gen({"culture": "english", "count": 1}, seed)
        for word in r.morphemes_by_word or []:
            for m in word:
                if m.get("era_grid") and m.get("rendered") and "active_form_id" not in m:
                    misses.append((r.result, m.get("usage"), m.get("rendered")))
    # Allow the rare no-cell residual (e.g. a render whose source stage isn't in
    # the grid), but the vast majority must resolve.
    assert len(misses) <= 2, f"too many gridded+rendered morphemes without active_form_id: {misses}"
