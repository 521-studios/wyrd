"""wyrd-rogd.10 Phase 0 — generation parity oracle.

Captures `(culture, seed) -> generated names` plus a compact per-morpheme
era-grid digest for a fixed sample, against the committed seed-runtime.db
fixture. Every phase of the morpheme-id refactor diffs against this snapshot:

  * Generation output (names) MUST stay byte-identical through ALL phases —
    the (generator, params, seed) -> output contract.
  * The era-grid digest MUST stay identical through Phase 1 (morpheme_id is
    emitted but dormant). It is EXPECTED to change in Phase 2 (the
    unfragmenting surfaces more coverage) — at which point regenerate the
    grid half of the snapshot deliberately and review the diff.

Regenerate the snapshot intentionally with:  WYRD_REGEN_PARITY=1 pytest -q tests/test_rogd10_parity.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from wyrd.generators.kenning import CULTURES, Kenning

# Fixed sample — deterministic seeds across every culture. Count>1 exercises
# the multi-result sub-seeding path.
_SEEDS = [1, 42, 1337, 8502015221875428, 7028257956033530, 99, 250, 777, 31415, 271828]
_COUNT = 3
_SNAPSHOT = Path(__file__).parent / "data" / "rogd10_parity.json"


def _grid_digest(morpheme: dict) -> list:
    """Compact, order-stable digest of a morpheme's era-grid: per family, per
    stage language, the cell forms (the thing Phase 2 is expected to grow)."""
    out = []
    for sec in morpheme.get("era_grid") or []:
        stages = [
            [st.get("language"), [c.get("form") for c in st.get("forms") or []]]
            for st in sec.get("stages") or []
        ]
        out.append([sec.get("family"), stages])
    return out


def _capture() -> dict:
    k = Kenning()
    snap: dict = {}
    for culture in sorted(CULTURES):
        per_culture = []
        for seed in _SEEDS:
            res = k.generate({"culture": culture, "count": _COUNT}, seed=seed)
            # name(s) — the byte-identical-through-all-phases invariant
            entry = {"seed": seed, "name": res.result}
            # era-grid digest of the first result's morphemes (the Phase-2 delta)
            grid = []
            for word in res.morphemes_by_word or []:
                for m in word or []:
                    grid.append([m.get("usage"), _grid_digest(m)])
            entry["grid"] = grid
            per_culture.append(entry)
        snap[culture] = per_culture
    return snap


def test_generation_and_era_grid_parity():
    captured = _capture()
    if os.environ.get("WYRD_REGEN_PARITY") == "1":
        _SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT.write_text(json.dumps(captured, indent=2, ensure_ascii=False, sort_keys=True))
        return  # regen run: write + skip the assertion
    assert _SNAPSHOT.exists(), (
        f"parity snapshot missing at {_SNAPSHOT}; regenerate with "
        "WYRD_REGEN_PARITY=1 pytest tests/test_rogd10_parity.py"
    )
    expected = json.loads(_SNAPSHOT.read_text())

    # Names are the hard invariant — assert them separately for a clear failure.
    def names_only(snap):
        return {c: [(e["seed"], e["name"]) for e in rows] for c, rows in snap.items()}

    assert names_only(captured) == names_only(expected), (
        "GENERATION OUTPUT DRIFTED — names changed for a fixed seed. This breaks "
        "the (generator,params,seed)->output contract; it is NOT an expected "
        "morpheme-id refactor delta."
    )
    # Era-grid digest: identical through Phase 1; regenerate intentionally at Phase 2.
    assert captured == expected, (
        "era-grid digest drifted. Through Phase 1 this must not change. If this "
        "is the intended Phase-2 unfragmenting delta, regenerate with "
        "WYRD_REGEN_PARITY=1 and review the diff."
    )
