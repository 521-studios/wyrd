"""``wyrd kenning dump-structures`` — dump the structure inventory to YAML for
the operator allowlist (wyrd-c6o1.5).

Walks every culture's mined structures in the runtime DB, labels each
(``struct_key_to_label``), dedupes globally, and emits one
``"<label>": {enabled: <bool>}`` entry per distinct structure. The operator
commits this as ``wyrd/generators/kenning/data/structures.yaml`` and curates by
flipping unwanted structures to ``enabled: false``; the generator filters its
structure pool through that file at load time
(``runtime.structure_allowlist.is_structure_enabled``).

Defaults: every structure is ``enabled: true`` EXCEPT the lone-dictionary-word
structures (a single morpheme total — ``(bare)`` / ``(bare[name])``), which seed
``enabled: false`` to preserve the migrated wyrd-g1hj exclusion. Ungrammatical
structures (wyrd-zzli) are omitted entirely — they're a hard gate, not
operator-curatable. A re-run after a bundle rebuild surfaces any new structures
(enabled by default) for the operator to diff in.

Output goes to stdout (or ``--output PATH``) — never writes the package-data file
directly, so the change stays review-friendly.
"""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning import CULTURES
from wyrd.generators.kenning.runtime.proportions import is_structurally_grammatical, word_to_key
from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
from wyrd.generators.kenning.runtime.runtime_db_adapter import proportions_dict_for_culture
from wyrd.generators.kenning.runtime.structure_allowlist import struct_key_to_label

_HEADER = """\
# wyrd-c6o1.5: structure allowlist. One entry per name structure mined from the
# corpus. Each defaults to `enabled: true`; set `enabled: false` to forbid the
# generator from using that structure. A structure ABSENT from this file is
# enabled, so a bundle rebuild's new structures generate by default — re-run
# `wyrd kenning dump-structures` to diff them in. Regenerated, do not hand-sort.
#
# The lone-dictionary-word structures ship disabled (the migrated wyrd-g1hj
# "<Bare>" special-case). Multi-bare structures like "(bare) (bare) (bare) (bare)"
# are enabled by default — disable the ones that read poorly.
"""


def _single_morpheme(struct_key: tuple) -> bool:
    """The migrated wyrd-g1hj default: a structure of one morpheme total renders
    as a flat lone dictionary word — seed it disabled.

    This is the same predicate the deleted runtime ``_is_single_morpheme_structure``
    used, but it lives HERE as a one-shot default-SEED for the dumped YAML, NOT as a
    runtime filter — the only runtime structure filter is the allowlist itself
    (``is_structure_enabled``). So there's still exactly one filtering path."""
    return sum(len(word) for word in struct_key) <= 1


def _inventory() -> dict[str, bool]:
    """Global ``{label: enabled-default}`` over every culture's grammatical
    structures."""
    conn = get_runtime_db()
    out: dict[str, bool] = {}
    for culture in CULTURES:
        for element in proportions_dict_for_culture(conn, culture)["structures"]:
            words = tuple(word_to_key(w) for w in element["words"])
            if not is_structurally_grammatical(words):
                continue  # hard-gated (wyrd-zzli); not operator-curatable
            out.setdefault(struct_key_to_label(words), not _single_morpheme(words))
    return out


def _render(inventory: dict[str, bool]) -> str:
    lines = [_HEADER]
    for label in sorted(inventory):
        lines.append(f'"{label}": {{enabled: {str(inventory[label]).lower()}}}')
    return "\n".join(lines) + "\n"


@click.command("dump-structures")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write to this path instead of stdout.",
)
def dump_structures(output: Path | None) -> None:
    """Dump the structure inventory as the structures.yaml allowlist."""
    text = _render(_inventory())
    if output is not None:
        output.write_text(text, encoding="utf-8")
        click.echo(f"wrote {output}", err=True)
    else:
        click.echo(text, nl=False)


def add_to(parent: click.Group) -> None:
    """Register ``dump-structures`` on the top-level ``@cli`` group."""
    parent.add_command(dump_structures)
