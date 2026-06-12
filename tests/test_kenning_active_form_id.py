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


def test_set_active_form_id_forward_exact_not_reconstructed_sibling():
    """wyrd-3vju.1: the active id is computed FORWARD from the picked morpheme's
    own form, so it lands on the EXACT reflex the generator rendered — not a
    fold-equal sibling. Here the reconstructed ``*west`` cell and the attested
    ``west`` cell both display form ``west``; the (since-removed) legacy backwards
    fold-search returned the FIRST folding cell (``*west``), but the render chose
    the attested ``west`` — the forward id (built from the picked morpheme's own
    form) picks it exactly, with no fold."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName

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
    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    m = {"usage": "West", "rendered": "West", "era_grid": grid}  # native render
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid="old-english:west")
    # Forward-exact: the rendered reflex's own cell, NOT the *west fold collision.
    assert m["active_form_id"] == "old-english:west"


def test_set_active_form_id_falls_back_to_fold_when_surface_not_a_reflex_cell():
    """wyrd-3vju.1/.3: when the rendered surface is NOT an enumerated reflex of
    the picked etymon (the genuine data gap — sparse era_reflexes / synthesized
    surface), the forward id isn't in the grid, so _ensure_cell resolves it: a
    fold-equal cell in the render stage is reused (no duplicate) so SOMETHING
    still highlights. No regression vs the pre-3vju.1 path."""
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


def test_set_active_form_id_force_modern_prefers_present_day_stage():
    """wyrd-3vju.2: force-modern (era="modern-english") leaves rendered None, so the
    active id is resolved from the usage against each family's PRESENT-DAY stage.
    A same-spelled earlier-era homograph (middle-english "water") must NOT outrank
    the modern cell — the (since-removed) lang=None fold returned the FIRST
    same-spelled cell in grid order (middle-english), the visible mis-highlight in
    defect 6d808d94 (Hesketh Water)."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName

    grid = [
        {
            "family": "english",
            "stages": [
                # middle-english listed FIRST (older→newer); both display "water".
                {
                    "language": "middle-english",
                    "forms": [{"id": "middle-english:water", "form": "water"}],
                },
                {
                    "language": "modern-english",
                    "forms": [{"id": "modern-english:water", "form": "water"}],
                },
            ],
        }
    ]
    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    # Force-modern: rendered absent/None, usage IS the present-day surface.
    m = {"usage": "water", "era_grid": grid}
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid="modern-english:water")
    assert m["active_form_id"] == "modern-english:water"


def test_set_active_form_id_injects_usage_cell_when_reflex_missing():
    """wyrd-3vju.3: the era grid is the SPA's swap menu, so the rendered surface MUST
    be a clickable cell. When force-modern renders the usage but the modern era-reflex
    is missing/garbage (here the modern cell holds 'gowth', not 'gold'), inject a
    source='usage' cell for the usage into the present-day stage and point
    active_form_id at it — so the live form is in the menu (swap away AND back) with a
    stable id. The existing (garbage) reflex stays as another swap option."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName, _grid_has_cell_id

    grid = [
        {
            "family": "english",
            "stages": [
                {
                    "language": "old-english",
                    "forms": [{"id": "old-english:gold", "form": "gold", "source": "cluster"}],
                },
                {
                    "language": "modern-english",
                    "forms": [{"id": "modern-english:gowth", "form": "gowth", "source": "cluster"}],
                },
            ],
        }
    ]
    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    m = {"usage": "gold", "era_grid": grid}  # force-modern: rendered None, usage is modern surface
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid="old-english:gold")

    assert m["active_form_id"] == "modern-english:gold"
    assert _grid_has_cell_id(grid, "modern-english:gold")
    modern = next(st for sec in grid for st in sec["stages"] if st["language"] == "modern-english")
    injected = next(c for c in modern["forms"] if c["id"] == "modern-english:gold")
    assert injected["form"] == "gold"
    assert injected["source"] == "usage"  # injected as the identity anchor
    assert injected["is_usage"] is True
    # the (garbage) 'gowth' reflex is preserved as another swap option, not removed.
    assert any(c["id"] == "modern-english:gowth" for c in modern["forms"])
    # the identity also appears as the OE 'gold' reflex → it too is marked is_usage.
    oe = next(
        c
        for sec in grid
        for st in sec["stages"]
        for c in st["forms"]
        if c["id"] == "old-english:gold"
    )
    assert oe["is_usage"] is True
    assert oe["source"] == "cluster"  # provenance preserved; not overwritten


def test_set_active_form_id_marks_identity_reflex_without_duplicating():
    """wyrd-3vju.3: when the usage already equals a present-day reflex, use that cell
    (no duplicate) and mark it is_usage while preserving its provenance."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName

    grid = [
        {
            "family": "english",
            "stages": [
                {
                    "language": "modern-english",
                    "forms": [{"id": "modern-english:stone", "form": "stone", "source": "cluster"}],
                },
            ],
        }
    ]
    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    m = {"usage": "stone", "era_grid": grid}
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid="old-english:stan")
    assert m["active_form_id"] == "modern-english:stone"
    modern = next(st for sec in grid for st in sec["stages"] if st["language"] == "modern-english")
    assert len(modern["forms"]) == 1  # no duplicate injected
    assert modern["forms"][0]["is_usage"] is True
    assert modern["forms"][0]["source"] == "cluster"  # provenance untouched


def test_set_active_form_id_era_render_anchors_identity_separately():
    """wyrd-3vju.3: in a HISTORICAL-era render the live highlight is the era reflex
    ('-tun' in OE), but the morpheme's IDENTITY (the usage '-ton') is still anchored
    + marked in the present-day stage — so the parent form is always visible and the
    user can swap back to it. The live OE reflex is NOT the identity."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName, _grid_has_cell_id

    grid = [{"family": "english", "stages": [{"language": "old-english", "forms": []}]}]
    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    # Era render: rendered '-tun' at old-english; usage/identity is '-ton'.
    m = {"usage": "ton", "rendered": "tun", "rendered_language": "old-english", "era_grid": grid}
    nn._set_active_form_id(
        m, SimpleNamespace(morpheme_id="old-english:ton"), grid_mid="old-english:ton"
    )

    # live highlight = the OE reflex actually rendered
    assert m["active_form_id"] == "old-english:tun"
    # identity anchored in the present-day stage, marked, but NOT highlighted
    assert _grid_has_cell_id(grid, "modern-english:ton")
    anchor = next(
        c
        for sec in grid
        for st in sec["stages"]
        for c in st["forms"]
        if c["id"] == "modern-english:ton"
    )
    assert anchor["is_usage"] is True
    assert anchor["source"] == "usage"
    tun = next(
        c
        for sec in grid
        for st in sec["stages"]
        for c in st["forms"]
        if c["id"] == "old-english:tun"
    )
    assert tun.get("is_usage") is not True  # the live era reflex is not the identity
    assert tun["source"] == "self"  # injected via the era-render data-gap path, not the anchor


def test_grid_stage_for_inserts_older_stage_in_canonical_order():
    """wyrd-3vju.3: _grid_stage_for must INSERT an older stage BEFORE an existing
    newer one (the SPA renders stages as oldest→newest columns), not append it.
    Every other injection test creates modern-english (the family's last stage),
    so this is the only test that exercises the mid-list insertion branch."""
    from wyrd.generators.kenning.runtime.proportions import _ensure_cell

    grid = [
        {
            "family": "english",
            "stages": [
                {"language": "middle-english", "forms": []},
                {"language": "modern-english", "forms": []},
            ],
        }
    ]
    cid = _ensure_cell(grid, "old-english", "tun", source="self")
    assert cid == "old-english:tun"
    assert [st["language"] for st in grid[0]["stages"]] == [
        "old-english",
        "middle-english",
        "modern-english",
    ]

    # A defensive unmapped tail stage (per _ordered_stage_langs, unmapped tags
    # sit at the end) stays the tail even for a stage LATER than every mapped
    # one: modern-english must insert BEFORE zzz-unmapped, which only the
    # 'other not in order' arm of the insertion condition achieves (the old
    # 'other in order and ...' condition appended it after the tail).
    grid2 = [
        {
            "family": "english",
            "stages": [
                {"language": "middle-english", "forms": []},
                {"language": "zzz-unmapped", "forms": []},
            ],
        }
    ]
    _ensure_cell(grid2, "modern-english", "ton", source="self")
    assert [st["language"] for st in grid2[0]["stages"]] == [
        "middle-english",
        "modern-english",
        "zzz-unmapped",
    ]


def test_ensure_cell_creates_missing_family_section():
    """wyrd-3vju.3: injecting into a family the grid lacks creates a correctly
    shaped {family, stages} section, inserted at its canonical
    _GRID_FAMILY_ORDER position (matching _era_grid's build ordering), not
    blindly appended."""
    from wyrd.generators.kenning.runtime.proportions import _ensure_cell

    grid = [{"family": "english", "stages": [{"language": "modern-english", "forms": []}]}]
    cid = _ensure_cell(grid, "welsh", "tre", source="self")
    assert cid == "welsh:tre"
    assert [s["family"] for s in grid] == ["english", "brythonic"]
    assert grid[1]["stages"] == [
        {"language": "welsh", "forms": [{"id": "welsh:tre", "form": "tre", "source": "self"}]}
    ]

    # Canonical position: english sorts BEFORE norse in _GRID_FAMILY_ORDER, so
    # injecting an english cell into a norse-first grid inserts at the front.
    grid2 = [{"family": "norse", "stages": [{"language": "old-norse", "forms": []}]}]
    _ensure_cell(grid2, "modern-english", "ton", source="usage")
    assert [s["family"] for s in grid2] == ["english", "norse"]


def test_ensure_cell_declines_language_without_era_family():
    """wyrd-3vju.3: a source language with no era family (proto-*, untracked) must
    NOT fabricate a {"family": None} section — _era_grid never builds one, and the
    SPA keys section headers by family. _ensure_cell fails closed (returns None →
    no active_form_id → the SPA keeps its legacy fold fallback)."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName, _ensure_cell

    grid = [{"family": "english", "stages": [{"language": "modern-english", "forms": []}]}]
    assert _ensure_cell(grid, "proto-germanic", "tunaz", source="self") is None
    assert [s["family"] for s in grid] == ["english"]  # no fabricated section

    # End-to-end via the native-render fallback: mid's source lang has no family.
    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    m = {"usage": None, "rendered": "tunaz", "era_grid": grid}
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid="proto-germanic:*tunaz")
    assert "active_form_id" not in m
    assert [s["family"] for s in grid] == ["english"]


def test_set_active_form_id_without_lang_scope_stamps_nothing():
    """wyrd-3vju.3: native render with NO resolvable morpheme id (no grid_mid, no
    first.morpheme_id) has no language to scope by — nothing is stamped and the
    frontend falls back to its own fold. Pins the deliberate behavior change from
    the removed cross-grid fold_any guess (the collision bug this PR removes)."""
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
    m = {"usage": None, "rendered": "Roth", "era_grid": grid}
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid=None)
    assert "active_form_id" not in m  # no lang scope → no cross-grid fold guess


def test_present_day_lang_anchors_in_mids_family_not_grid_order():
    """wyrd-3vju.3: the identity anchor lands in MID's family's present-day stage
    even when that family isn't the grid's first section; with no mid at all the
    grid's primary (first) section is the documented fallback."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName, _present_day_lang

    grid = [
        {"family": "norse", "stages": [{"language": "old-norse", "forms": []}]},
        {"family": "english", "stages": [{"language": "modern-english", "forms": []}]},
    ]
    assert _present_day_lang(grid, "old-english:ton") == "modern-english"
    assert _present_day_lang(grid, None) == "old-norse"  # grid[0] fallback

    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    m = {"usage": "ton", "era_grid": grid}
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid="old-english:ton")
    # anchor injected into the english (second) section's present-day stage
    assert m["active_form_id"] == "modern-english:ton"
    assert grid[1]["stages"][0]["forms"][0]["id"] == "modern-english:ton"
    assert grid[0]["stages"][0]["forms"] == []  # norse stage untouched


def test_ensure_cell_reuses_accented_reflex_for_ascii_surface():
    """wyrd-3vju.3: the dedupe fold is accent-insensitive (mirrors the SPA's
    accentFold) — a lossy ASCII usage ('ra') binds to the genuine accented reflex
    cell ('rá') instead of injecting a bare visual duplicate beside it, and the
    identity mark lands on the genuine cell."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName, _ensure_cell

    grid = [
        {
            "family": "norse",
            "stages": [
                {
                    "language": "old-norse",
                    "forms": [{"id": "old-norse:rá", "form": "rá", "source": "cluster"}],
                }
            ],
        }
    ]
    assert _ensure_cell(grid, "old-norse", "ra", source="usage") == "old-norse:rá"
    assert len(grid[0]["stages"][0]["forms"]) == 1  # reused, not duplicated

    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    m = {"usage": "ra", "era_grid": grid}
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid="old-norse:rá")
    assert m["active_form_id"] == "old-norse:rá"
    cell = grid[0]["stages"][0]["forms"][0]
    assert cell["is_usage"] is True  # identity mark on the genuine accented cell
    assert cell["source"] == "cluster"  # provenance preserved


def test_cell_id_in_stage_fold_tier_beats_accent_tier():
    """wyrd-3vju.3: tier precedence across DIFFERENT cells — a case/dash-fold
    match outranks an accent-insensitive match even when the accent-tier cell
    comes first in the stage. Pins the exact → fold → accent ordering of
    _cell_id_in_stage as a cross-cell contract, not just each tier in
    isolation."""
    from wyrd.generators.kenning.runtime.proportions import _cell_id_in_stage

    grid = [
        {
            "family": "norse",
            "stages": [
                {
                    "language": "old-norse",
                    "forms": [
                        # accent-tier candidate listed FIRST so position can't win
                        {"id": "old-norse:rá", "form": "rá"},  # matches only accent-fold
                        {"id": "old-norse:Ra", "form": "Ra"},  # matches case/dash fold
                    ],
                }
            ],
        }
    ]
    assert _cell_id_in_stage(grid, "old-norse", "ra") == "old-norse:Ra"
    # And an exact form match beats both fold tiers.
    grid[0]["stages"][0]["forms"].append({"id": "old-norse:ra", "form": "ra"})
    assert _cell_id_in_stage(grid, "old-norse", "ra") == "old-norse:ra"


def test_grid_match_key_star_strip_decides_reuse_and_marking():
    """wyrd-3vju.3: _grid_match_key strips the reconstruction '*' marker (plus
    NFD combining marks, dashes, case), so a surface matches a reconstructed
    cell when that cell is the ONLY candidate — the '*'-strip deciding the
    outcome, not an earlier tier on an attested sibling."""
    from wyrd.generators.kenning.runtime.proportions import (
        _ensure_cell,
        _grid_match_key,
        _mark_usage_cells,
    )

    assert _grid_match_key("*Ūr-") == "ur"
    assert _grid_match_key("-rá") == "ra"
    assert _grid_match_key("ra") == "ra"

    grid = [
        {
            "family": "english",
            "stages": [
                {
                    "language": "old-english",
                    # the ONLY candidate carries the raw '*' marker in its form
                    "forms": [{"id": "old-english:*ūr", "form": "*ūr", "source": "cluster"}],
                }
            ],
        }
    ]
    assert _ensure_cell(grid, "old-english", "ur", source="usage") == "old-english:*ūr"
    assert len(grid[0]["stages"][0]["forms"]) == 1  # reused, not duplicated
    _mark_usage_cells(grid, "ur")
    assert grid[0]["stages"][0]["forms"][0]["is_usage"] is True


def test_late_injected_live_cell_folding_to_usage_is_marked():
    """wyrd-3vju.3: is_usage marking runs AFTER the fallback injections, so a live
    cell injected for an era render whose surface folds to the usage still carries
    the identity mark — the invariant is 'EVERY cell folding to the usage is
    marked', and a cell can be the identity, the live form, or both."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName

    grid = [{"family": "english", "stages": [{"language": "middle-english", "forms": []}]}]
    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    # Era render whose rendered surface IS the identity ('ton' at middle-english),
    # not an enumerated reflex — the live cell is injected by the fallback.
    m = {"usage": "ton", "rendered": "ton", "rendered_language": "middle-english", "era_grid": grid}
    nn._set_active_form_id(
        m, SimpleNamespace(morpheme_id="old-english:ton"), grid_mid="old-english:ton"
    )
    assert m["active_form_id"] == "middle-english:ton"
    live = next(
        c
        for sec in grid
        for st in sec["stages"]
        for c in st["forms"]
        if c["id"] == "middle-english:ton"
    )
    assert live["source"] == "self"
    assert live["is_usage"] is True  # late-injected live cell still gets the identity mark


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


def test_era_reflex_raw_choice_attested_over_reconstructed_and_strip_parity():
    """wyrd-3vju.1: _era_reflex_raw_choice keeps the raw reflex (``*`` marker
    preserved, for the cell id) and shares its attested-over-reconstructed
    selection with _era_form_for_meanings (which returns the ``*``-stripped
    surface). Pins both directly (the era forward branch builds the cell id
    from this raw choice)."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import (
        _era_form_for_meanings,
        _era_reflex_raw_choice,
    )

    def mk(forms):
        return SimpleNamespace(
            era_reflex_for=lambda lang, f=forms: f if lang == "old-english" else []
        )

    # attested present → raw is the attested (no star); surface equal.
    assert _era_reflex_raw_choice([mk(["*west", "west"])], "old-english") == "west"
    assert _era_form_for_meanings([mk(["*west", "west"])], "old-english") == "west"
    # all reconstructed → raw KEEPS the star (cell-id distinct); surface strips it.
    assert _era_reflex_raw_choice([mk(["*west"])], "old-english") == "*west"
    assert _era_form_for_meanings([mk(["*west"])], "old-english") == "west"
    # first sense carrying a reflex wins; no reflex anywhere → None.
    assert _era_reflex_raw_choice([mk([]), mk(["ham"])], "old-english") == "ham"
    assert _era_reflex_raw_choice([mk([])], "old-english") is None
    assert _era_form_for_meanings([mk([])], "old-english") is None


def test_set_active_form_id_era_forward_exact(monkeypatch):
    """wyrd-3vju.1: era-render forward branch — active_form_id is the EXACT cell
    of the era reflex chosen (attested 'west' over reconstructed '*west'),
    resolved from the PICKED morpheme, not a fold of the case-projected surface
    (which would land on the first folding cell, '*west')."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime import proportions as P
    from wyrd.generators.kenning.runtime.proportions import NewName

    fake = SimpleNamespace(
        era_reflex_for=lambda lang: ["*west", "west"] if lang == "old-english" else []
    )
    monkeypatch.setattr(P, "_resolve_morpheme", lambda db, mid: fake)
    grid = [
        {
            "family": "english",
            "stages": [
                {
                    "language": "old-english",
                    "forms": [
                        {"id": "old-english:*west", "form": "west"},
                        {"id": "old-english:west", "form": "west"},
                    ],
                }
            ],
        }
    ]
    nn = NewName(struct=None, meaning_db={"x": 1}, name=[["x"]])
    m = {
        "usage": "West",
        "rendered": "West",
        "rendered_language": "old-english",
        "era_grid": grid,
    }
    nn._set_active_form_id(
        m, SimpleNamespace(morpheme_id="old-english:west"), grid_mid="old-english:west"
    )
    assert m["active_form_id"] == "old-english:west"


def test_set_active_form_id_modern_path_and_no_grid():
    """wyrd-3vju.1: the no-distinct-render (modern) path resolves the present-day
    usage seed; and a morpheme with no era_grid stamps nothing."""
    from types import SimpleNamespace

    from wyrd.generators.kenning.runtime.proportions import NewName

    grid = [
        {
            "family": "english",
            "stages": [
                {
                    "language": "modern-english",
                    "forms": [{"id": "modern-english:west", "form": "west"}],
                }
            ],
        }
    ]
    nn = NewName(struct=None, meaning_db={}, name=[["x"]])
    m = {"usage": "West", "era_grid": grid}  # no 'rendered' → modern
    nn._set_active_form_id(m, SimpleNamespace(morpheme_id=None), grid_mid=None)
    assert m["active_form_id"] == "modern-english:west"

    m2 = {"usage": "x"}  # no era_grid → early return, nothing stamped
    nn._set_active_form_id(m2, SimpleNamespace(morpheme_id=None))
    assert "active_form_id" not in m2


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
