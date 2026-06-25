"""wyrd command-line entry point.

Each generator owns its own click group; this module mounts them under the
top-level `wyrd` command and adds a few service-level commands (manifest, serve).
"""

from __future__ import annotations

import json

import click

from wyrd import registry
from wyrd.cli_defects import defects as defects_cli


@click.group()
def main() -> None:
    """wyrd — random-generation service for TTRPG."""
    registry.discover()


@main.command()
def manifest() -> None:
    """Print the manifest of registered generators — the same per-generator
    entries as GET /api/manifest (incl. the optional ``era_stages`` axis). The
    API's ``config`` block (env-resolved SPA feature flags) is API-only; the CLI
    doesn't gate any UI, so it's omitted here."""
    out = {
        "generators": [
            {
                "name": g.name,
                "display_name": g.display_name,
                "description": g.description,
                "details": g.details,
                "legend": g.legend,
                "input_schema": g.input_schema(),
                # Parity with /api/manifest: the optional per-family era-stage
                # axis (kenning), null for generators that don't define it.
                "era_stages": getattr(g, "era_stages", None),
            }
            for g in registry.all_generators()
        ]
    }
    click.echo(json.dumps(out, indent=2))


@main.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=5000, type=int)
@click.option("--debug", is_flag=True, default=False)
def serve(host: str, port: int, debug: bool) -> None:
    """Run the Flask dev server."""
    # Deferred import: pulling Flask in only when serve runs keeps `wyrd --help`
    # and other CLI subcommands lightweight.
    from wyrd.app import create_app

    app = create_app()
    app.run(host=host, port=port, debug=debug)


def _mount_generator_clis() -> None:
    """Discover every generator and mount its `cli` group under `wyrd <name>`."""
    registry.discover()
    for g in registry.all_generators():
        try:
            mod = __import__(f"wyrd.generators.{g.name}.cli", fromlist=["cli"])
        except ModuleNotFoundError:
            continue
        sub_cli = getattr(mod, "cli", None)
        if sub_cli is not None:
            main.add_command(sub_cli, name=g.name)


# wyrd-dsl5: cross-generator triage CLI for defective-name reports. Mounted
# at the top level (not under a generator) since reports span all generators.
main.add_command(defects_cli)

_mount_generator_clis()
