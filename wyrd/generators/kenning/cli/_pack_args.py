"""Parsing + validation for the ``wyrd kenning generate`` pack
CLI flags (wyrd-ecjp.11 Phase 8).

The CLI exposes two pack-related flags:

* ``--pack <name>[:<weight>]`` (repeatable) — declares a scenario
  pack overlay. Optional ``:<weight>`` overrides the pack's manifest
  default_weight.
* ``--pack-tag-filter <pack>:<tag1,tag2>`` (repeatable) — narrows a
  declared pack's lemmas to those carrying at least one matching
  tag (the PackOverlay.allowed_pack_tags axis from ecjp.8).

Validation rules:
* ``--pack`` weights must parse as floats in the same range as
  ScoringWeights (0..10).
* ``--pack-tag-filter`` referencing an undeclared pack errors
  cleanly (operator typo or flag ordering bug).
* Unknown pack names (not present in the bundled pack-catalog)
  error cleanly with the available-packs list — same UX as the
  existing --tag / --mood unknown-value paths.

The runtime dispatch (kenning._generate_via_vector) consumes the
parsed pack specs + the bundle's PackBundle map to construct
PackOverlay + pack_meaning_dbs at request time.
"""

from __future__ import annotations

from dataclasses import dataclass

import click


@dataclass(frozen=True)
class ParsedPackSpec:
    """One ``--pack`` value, post-parse. ``weight_override`` is None
    when the operator didn't pass a colon-weight (the runtime falls
    back to PackManifest.default_weight)."""

    pack_name: str
    weight_override: float | None


def parse_pack_specs(specs: tuple[str, ...]) -> list[ParsedPackSpec]:
    """Parse ``--pack`` flag values. Each value is either ``name`` or
    ``name:<weight>``. Raises ``click.BadParameter`` on:
    * empty name
    * non-float weight
    * weight outside [0, 10]
    * duplicate pack_name across multiple --pack flags (operator typo)

    Order is preserved so the runtime can apply packs in declaration
    order if it ever matters; the ecjp.8 runtime is order-independent
    today but the test suite pins that contract.
    """
    out: list[ParsedPackSpec] = []
    seen: set[str] = set()
    for raw in specs:
        if ":" in raw:
            name, _, weight_str = raw.partition(":")
            name = name.strip()
            weight_str = weight_str.strip()
            if not name:
                raise click.BadParameter(f"--pack {raw!r}: missing pack name before ':'")
            if not weight_str:
                raise click.BadParameter(f"--pack {raw!r}: missing weight after ':'")
            try:
                weight: float | None = float(weight_str)
            except ValueError as exc:
                raise click.BadParameter(f"--pack {raw!r}: weight must be a float") from exc
            if not (0.0 <= weight <= 10.0):
                raise click.BadParameter(f"--pack {raw!r}: weight must be in [0, 10]")
        else:
            name = raw.strip()
            weight = None
            if not name:
                raise click.BadParameter("--pack: empty pack name")
        if name in seen:
            raise click.BadParameter(
                f"--pack {name!r}: declared more than once "
                "(use a single --pack with the desired weight)"
            )
        seen.add(name)
        out.append(ParsedPackSpec(pack_name=name, weight_override=weight))
    return out


def parse_pack_tag_filter_specs(
    specs: tuple[str, ...],
    declared_packs: set[str],
) -> dict[str, frozenset[str]]:
    """Parse ``--pack-tag-filter`` flag values. Each value is
    ``<pack>:<tag1,tag2,...>``. Returns ``{pack_name: frozenset(tags)}``.

    Raises ``click.BadParameter`` on:
    * missing colon separator
    * empty pack name or empty tag list
    * pack name not in ``declared_packs`` (operator forgot the
      matching --pack, or typo'd the name)
    * duplicate pack_name across multiple --pack-tag-filter flags
      (ambiguous: which tag set wins?)

    Note: the tag set itself is NOT validated against
    ``available_tags()`` — pack lemmas can carry pack-specific tags
    that aren't in the main bundle's tag vocabulary. The runtime
    filter is a set-membership check; an unknown tag just admits
    nothing for the pack (degenerate but not an error).
    """
    out: dict[str, frozenset[str]] = {}
    for raw in specs:
        if ":" not in raw:
            raise click.BadParameter(
                f"--pack-tag-filter {raw!r}: expected '<pack>:<tag1,tag2,...>'"
            )
        pack_name, _, tags_str = raw.partition(":")
        pack_name = pack_name.strip()
        tags_str = tags_str.strip()
        if not pack_name:
            raise click.BadParameter(f"--pack-tag-filter {raw!r}: missing pack name before ':'")
        if not tags_str:
            raise click.BadParameter(f"--pack-tag-filter {raw!r}: missing tag list after ':'")
        if pack_name not in declared_packs:
            declared_list = ", ".join(sorted(declared_packs)) or "(none)"
            raise click.BadParameter(
                f"--pack-tag-filter {pack_name!r}: pack not declared via "
                f"--pack. Declared packs: {declared_list}"
            )
        if pack_name in out:
            raise click.BadParameter(
                f"--pack-tag-filter {pack_name!r}: declared more than "
                "once (use a single --pack-tag-filter with the desired tag list)"
            )
        tags = frozenset(t.strip() for t in tags_str.split(",") if t.strip())
        if not tags:
            raise click.BadParameter(
                f"--pack-tag-filter {raw!r}: tag list must have at least one non-empty tag"
            )
        out[pack_name] = tags
    return out


def validate_packs_against_catalog(
    pack_specs: list[ParsedPackSpec],
    available_packs: set[str],
) -> None:
    """Cross-check declared pack names against the bundle's loaded
    pack catalog. Raises ``click.BadParameter`` with the available-
    packs list (matches the --tag / --mood error UX) on any unknown
    pack name.
    """
    unknown = [s.pack_name for s in pack_specs if s.pack_name not in available_packs]
    if unknown:
        available_list = ", ".join(sorted(available_packs)) or "(no packs bundled)"
        raise click.BadParameter(
            f"Unknown pack(s): {', '.join(unknown)}. Available: {available_list}"
        )
