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
                f"active_form_id {cid!r} not in this morpheme's era_grid (usage={m.get('usage')!r})"
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


# --- focused unit tests for the id-first helpers (wyrd-i4jd) ----------------


def test_reconstruct_picked_ids_shape_and_none_degrade():
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import _reconstruct_picked_ids

    struct = [[("pre",), ("post",)], [("bare",)]]  # 2 + 1 slots
    picked = [
        SimpleNamespace(morpheme_id="old-english:a"),
        None,  # dropped / permissive slot → None
        SimpleNamespace(morpheme_id=None),  # pick without a morpheme_id → None
    ]
    assert _reconstruct_picked_ids(struct, picked) == [["old-english:a", None], [None]]


def test_active_cell_id_lang_scope_then_fold_then_none():
    from wyrd.generators.kenning.runtime.proportions import _active_cell_id

    grid = [
        {
            "family": "english",
            "stages": [
                {"language": "old-english", "forms": [{"id": "old-english:roth", "form": "roth"}]},
                {
                    "language": "middle-english",
                    "forms": [{"id": "middle-english:roth", "form": "roth"}],
                },
            ],
        }
    ]
    # exact match inside the requested lang stage wins (both stages fold equal).
    assert _active_cell_id(grid, "middle-english", "roth") == "middle-english:roth"
    assert _active_cell_id(grid, "old-english", "roth") == "old-english:roth"
    # case differs → fold (not exact) match in the lang stage.
    assert _active_cell_id(grid, "middle-english", "Roth") == "middle-english:roth"
    # no lang hint → first folding cell anywhere (fold_any).
    assert _active_cell_id(grid, None, "roth") == "old-english:roth"
    # no fold match, empty grid, empty surface → None.
    assert _active_cell_id(grid, "old-english", "zzz") is None
    assert _active_cell_id([], "old-english", "roth") is None
    assert _active_cell_id(grid, "old-english", "") is None


def test_set_active_form_id_forward_exact_not_reconstructed_sibling():
    """wyrd-3vju.1: the active id is computed FORWARD from the picked morpheme's
    own form, so it lands on the EXACT reflex the generator rendered — not a
    fold-equal sibling. Here the reconstructed ``*west`` cell and the attested
    ``west`` cell both display form ``west``; the legacy backwards fold-search
    returned the FIRST folding cell (``*west``), but the render chose the
    attested ``west`` — the forward id (built from the picked morpheme's own
    form) picks it exactly, with no fold."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName, _active_cell_id

    grid = [
        {
            "family": "english",
            "stages": [
                {
                    "language": "old-english",
                    "forms": [
                        {"id": "old-english:*west", "form": "west"},  # reconstructed
                        {"id": "old-english:west", "form": "west"},  # attested (rendered)
                    ],
                }
            ],
        }
    ]
    # The legacy backwards search would (wrongly) return the reconstructed sibling.
    assert _active_cell_id(grid, "old-english", "West") == "old-english:*west"

    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    m = {"usage": "West", "rendered": "West", "era_grid": grid}  # native render
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid="old-english:west")
    # Forward-exact: the rendered reflex's own cell, NOT the *west fold collision.
    assert m["active_form_id"] == "old-english:west"


def test_set_active_form_id_falls_back_to_fold_when_surface_not_a_reflex_cell():
    """wyrd-3vju.1: when the rendered surface is NOT an enumerated reflex of the
    picked etymon (the genuine data gap — sparse era_reflexes / synthesized
    surface), the forward id isn't in the grid, so we fall back to the legacy
    fold so SOMETHING still highlights. No regression vs the pre-3vju.1 path."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName

    grid = [
        {
            "family": "english",
            "stages": [
                {"language": "old-english", "forms": [{"id": "old-english:roth", "form": "roth"}]}
            ],
        }
    ]
    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    # grid_mid's native form is 'hyll' (NOT a cell here); rendered folds to 'roth'.
    m = {"usage": "Roth", "rendered": "Roth", "era_grid": grid}
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid="old-english:hyll")
    assert m["active_form_id"] == "old-english:roth"  # fold fallback still works


def test_active_form_id_is_the_exact_rendered_reflex_corpus():
    """wyrd-3vju.1 contract: across rolls, an emitted active_form_id resolves to
    a cell whose form casefold-equals the rendered surface WITH combining marks
    preserved (macron-strict) — i.e. the highlight is the exact reflex rendered,
    not a lossy accent-fold collision (the Output/Inspect macron mismatch). The
    residual (rendered surface genuinely absent from the reflex set) is the
    era-grid data-quality epic; it stays small."""
    strict_hits = total = 0
    for era in ("", "modern-english", "oe-late"):
        for seed in range(60):
            r = _gen({"culture": "english", "count": 1, "era": era}, seed)
            for word in r.morphemes_by_word or []:
                for m in word:
                    cid = m.get("active_form_id")
                    if not cid or not m.get("rendered"):
                        continue
                    total += 1
                    cell = _cell_by_id(m.get("era_grid"), cid)
                    cf = (cell or {}).get("form", "")
                    if (
                        cf.strip().strip("-").casefold()
                        == m["rendered"].strip().strip("-").casefold()
                    ):
                        strict_hits += 1
    assert total > 100, f"too few samples ({total})"
    # Forward-exact dominates; allow a small fold-fallback residue (data gap).
    assert strict_hits / total >= 0.97, f"only {strict_hits}/{total} macron-strict"


def test_set_active_form_id_native_lang_from_grid_mid_not_first():
    """wyrd-i4jd regression: the native-render lang scope must come from the
    PICKED morpheme (grid_mid), not first.morpheme_id (the surface sibling) —
    else the cross-era fold collision the PR removes comes back."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName

    grid = [
        {
            "family": "english",
            "stages": [
                {"language": "old-english", "forms": [{"id": "old-english:roth", "form": "roth"}]},
                {
                    "language": "middle-english",
                    "forms": [{"id": "middle-english:roth", "form": "roth"}],
                },
            ],
        }
    ]
    first = SimpleNamespace(morpheme_id="old-english:hyll")  # surface sibling, WRONG stage
    nn = NewName(struct=None, meaning_db={}, name=[["x"]])

    # grid built from the ME-picked morpheme → ME stage must win.
    m = {"usage": "x", "rendered": "roth", "era_grid": grid}
    nn._set_active_form_id(m, first, grid_mid="middle-english:roth")
    assert m["active_form_id"] == "middle-english:roth"

    # without grid_mid it falls back to first's lang (old-english) — proving the
    # grid_mid threading is what disambiguates.
    m2 = {"usage": "x", "rendered": "roth", "era_grid": grid}
    nn._set_active_form_id(m2, first)
    assert m2["active_form_id"] == "old-english:roth"


def test_morpheme_id_for_guard_and_resolve_repeat_lockstep():
    """_morpheme_id_for None-degrades on bad indices; diversification updates
    picked_ids in lockstep with the override (cross-language synonym)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    hill = Meaning("hill", tags=[], meanings=["hill"], sources={"old_english": ["hill"]})
    hill.morpheme_id = "old-english:hyll"
    norse = Meaning("hill", tags=[], meanings=["Hill"], sources={"old_scandinavian": ["haeth"]})
    norse.morpheme_id = "old-scandinavian:haeth"
    nn = NewName(
        struct=None,
        meaning_db={"hill": [hill, norse]},
        name=[["hill"], ["hill"]],
        picked_ids=[["old-english:hyll"], ["old-english:hyll"]],
    )
    # out-of-range / no picked_ids → None, never raises.
    assert nn._morpheme_id_for(9, 9) is None
    assert str(nn) == "Hill Haeth"  # triggers diversification (override slot 1)
    # the repeat slot's identity now follows the synonym override, in lockstep.
    assert nn._morpheme_id_for(1, 0) == "old-scandinavian:haeth"
    assert nn._morpheme_id_for(0, 0) == "old-english:hyll"  # first slot unchanged
