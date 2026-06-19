"""wyrd-4rp8: thematic-mood overlay — "one mood-appropriate morpheme per name".

A thematic mood (grim/pastoral/…) used to be a no-op: it was applied as a HARD
required_tags gate so restrictive that generation fell back to the ungated pool.
The fix makes a mood a soft overlay — one slot whose pool can carry a mood tag is
restricted to mood-tagged morphemes; every other slot generates normally. These
tests pin that contract: a mood themes ~every name with ONE accent morpheme, not
every morpheme, and no-mood output is untouched. Seed-deterministic → non-flaky.
"""

from __future__ import annotations

from wyrd.generators.kenning.generators.kenning import Kenning

GRIM = {"death", "military", "monster", "undead", "magic"}
# wyrd-yzul: 100 samples keeps every assertion here comfortably non-flaky — the
# binding one is the no-mood baseline ceiling (base < 0.30; true ~0.17, ~3.5σ
# out at n=100), while grim incidence is a flat 1.0 (∞ margin) and the
# means/has-normal checks sit well inside their windows. Halved from 200 to cut
# CI cost; these are seed-deterministic so the reduction only trades a thinner
# (still ample) statistical margin, never reproducibility.
N = 100


def _name_tag_sets(k, mood, n=N):
    """Per generated name: the set of tags across its morphemes, and the count of
    morphemes carrying a grim tag + total morphemes."""
    out = []
    for i in range(n):
        params = {"culture": "english"}
        if mood:
            params["mood"] = mood
        comps = k.generate(params, seed=i).components or []
        grim_morphs = sum(1 for c in comps if GRIM & set(c.get("tags") or []))
        out.append((grim_morphs, len(comps)))
    return out


def test_grim_themes_nearly_every_name():
    """A grim request lands a grim morpheme in almost every name; the no-mood
    baseline is only incidental."""
    k = Kenning()
    base = sum(1 for g, _ in _name_tag_sets(k, None) if g) / N
    grim = sum(1 for g, _ in _name_tag_sets(k, ["grim"]) if g) / N
    assert base < 0.3, f"baseline grim incidence unexpectedly high: {base:.2f}"
    assert grim >= 0.9, f"grim mood themed only {grim:.2f} of names (expected ~all)"


def test_mood_adds_about_one_morpheme_not_recolor_everything():
    """The overlay restricts ONE slot, not the whole name — distinguishes it from
    the old all-morphemes-must-be-grim hard gate. Mean grim morphemes/name ≈ 1,
    while names still average more than one morpheme (so the rest stay normal)."""
    k = Kenning()
    rows = _name_tag_sets(k, ["grim"])
    mean_grim = sum(g for g, _ in rows) / N
    mean_total = sum(t for _, t in rows) / N
    assert 0.8 <= mean_grim <= 1.6, f"mean grim morphemes/name = {mean_grim:.2f}, expected ~1"
    assert mean_total > 1.4, f"names too short to show the 'one accent' effect: {mean_total:.2f}"
    # Most multi-morpheme grim names keep at least one NON-grim morpheme.
    multi = [(g, t) for g, t in rows if t >= 2]
    has_normal = sum(1 for g, t in multi if g < t) / len(multi)
    assert has_normal >= 0.7, (
        f"only {has_normal:.2f} of multi-morpheme grim names keep a normal morpheme"
    )


def test_no_mood_output_carries_no_imposed_theme():
    """The overlay fires only when a mood is requested — no-mood generation is
    untouched (byte-stability of the full no-mood path is the realism suite's
    job; here we pin that plain output isn't silently grim-shifted)."""
    k = Kenning()
    base = sum(1 for g, _ in _name_tag_sets(k, None) if g) / N
    assert base < 0.3
