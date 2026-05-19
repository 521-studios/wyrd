"""D6 mood preset catalog.

Extracted from ``wyrd/generators/kenning/__init__.py`` in wyrd-a83i.
Imported back via the registers/__init__.py re-export so callers
that read ``MOODS`` off ``kenning`` keep working unchanged.

A "mood" bundles one or more effects (semantic tag union, phonological
harshness skew, future axes) under a single GM-facing label so 'I want
grim names' is one decision rather than separate dials. Each entry may
carry a ``tags`` tuple (semantic-tag union) and/or a ``harshness``
float (D6 phonological skew default when the mood is requested with
no value).

CLI surface: ``--mood grim``, ``--mood harsh``, ``--mood harsh:0.5``
(colon-suffix overrides the recipe default for graduated moods like
harshness). Multiple ``--mood`` flags compose by tag-union and by
max-harshness.

The 'grim' tag set substitutes the original D6 spec names ('grim',
'mortuary', 'monstrous', 'battle', 'wilderness') — none of those
exist in the bundle yet; the closest extant tags fill in (death=18
subjects, military=27, monster=8, undead=9, magic=4). Adding the
spec-named tags later folds in without breaking callers.

The wyrd-kq7w / D36 rip-and-replace will migrate each MOODS entry to
an equivalent RegisterEffect in
``data/register_effects.yaml`` (catalog-driven instead of code-
defined). Until that migration lands the MOODS dict is the canonical
mood source; the YAML catalog covers register effects that aren't
mood-shaped.
"""

from __future__ import annotations

from typing import Any

MOODS: dict[str, dict[str, Any]] = {
    "grim": {"tags": ("death", "military", "monster", "undead", "magic")},
    "harsh": {"harshness": 1.0},
    # Picked from the 2026-05-02 bundle audit (≥5 subjects per tag, distinct
    # semantic identity, minimal overlap with existing moods). 'noble' was
    # considered but no 'royalty' tag exists yet — defer until mining
    # surfaces it. 'ominous' was too thin (magic=4 below threshold).
    "pastoral": {"tags": ("plant", "animal", "water", "agriculture", "tree", "bird")},
    "devotional": {"tags": ("saint", "religious")},
    "mortuary": {"tags": ("death", "undead")},
}
