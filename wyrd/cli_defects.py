"""``wyrd defects`` — operator triage CLI for defective-name reports
(wyrd-dsl5).

Generator-agnostic: a report can come from any generator, and ``list``
filters by ``--generator``. The commands read/update the per-env DynamoDB
table written by the SPA → Lambda flow.

Auth: each command takes ``--env staging|production`` and resolves an admin
profile (``521-Staging-Admin`` / ``521-Production-Admin`` per the workspace
notes), overridable with ``--profile`` or ``$WYRD_DEFECTS_AWS_PROFILE``. The
table name follows the per-env convention unless ``$WYRD_DEFECTS_TABLE`` is
set. boto3 is imported lazily by the ``wyrd.defects`` module.
"""

from __future__ import annotations

import json
import os

import click

# Default admin profiles per env (workspace AWS profiles). Override with
# --profile or $WYRD_DEFECTS_AWS_PROFILE; an explicit empty value falls back
# to the default credential chain (used by tests under moto).
_ENV_PROFILE = {
    "staging": "521-Staging-Admin",
    "production": "521-Production-Admin",
}

_ENV_CHOICE = click.Choice(["staging", "production"])


def _resolve_profile(env: str, explicit: str | None) -> str | None:
    """Resolve the AWS profile: explicit --profile wins, then
    $WYRD_DEFECTS_AWS_PROFILE, then the per-env admin default. An empty
    string at any layer means 'default credential chain' (None)."""
    from wyrd.defects import ENV_PROFILE

    if explicit is not None:
        return explicit or None
    env_override = os.environ.get(ENV_PROFILE)
    if env_override is not None:
        return env_override or None
    return _ENV_PROFILE.get(env)


def _env_options(fn):
    """Shared --env / --profile options for every defects subcommand."""
    fn = click.option(
        "--env",
        "env",
        type=_ENV_CHOICE,
        default="staging",
        show_default=True,
        help="Which environment's defects table to target.",
    )(fn)
    fn = click.option(
        "--profile",
        "profile",
        default=None,
        help="AWS profile override (default: the env's admin profile).",
    )(fn)
    return fn


@click.group("defects")
def defects() -> None:
    """Triage defective-name reports submitted from the SPA (wyrd-dsl5)."""


@defects.command("list")
@_env_options
@click.option(
    "--status",
    type=click.Choice(["new", "accepted", "dismissed", "all"]),
    default="new",
    show_default=True,
    help="Filter by triage status.",
)
@click.option("--generator", default=None, help="Only reports from this generator.")
@click.option("--limit", type=int, default=100, show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit raw JSON.")
def list_defects_cmd(env, profile, status, generator, limit, as_json) -> None:
    """List defect reports (default: untriaged 'new', newest first)."""
    from wyrd.defects import DefectsError, list_defects

    try:
        reports = list_defects(
            env=env,
            profile=_resolve_profile(env, profile),
            status=status,
            generator=generator,
            limit=limit,
        )
    except DefectsError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(reports, indent=2, ensure_ascii=False))
        return
    if not reports:
        click.echo(f"No '{status}' defect reports.", err=True)
        return
    for r in reports:
        reason = (r.get("reason") or "").replace("\n", " ")
        if len(reason) > 60:
            reason = reason[:57] + "..."
        click.echo(
            f"{r.get('id', '?')}  {r.get('created_at', ''):20}  "
            f"{r.get('generator', '?'):16}  {r.get('status', ''):9}  "
            f"{r.get('name', ''):24}  {reason}"
        )
    click.echo(f"\n{len(reports)} report(s).", err=True)


# Human-readable ``show`` layout: (label, report-key). Header fields always
# print; optional fields print only when their value is truthy. Both render
# with a shared label column (`_LABEL_WIDTH`) so values line up.
_SHOW_HEADER_FIELDS = (
    ("id", "id"),
    ("status", "status"),
    ("created_at", "created_at"),
    ("generator", "generator"),
    ("name", "name"),
    ("reason", "reason"),
)
_SHOW_OPTIONAL_FIELDS = (
    ("etymology", "explanation"),
    ("triaged_at", "triaged_at"),
    ("ticket", "ticket"),
    ("note", "triage_note"),
)

# Longest label + colon + a trailing space; derived so alignment can't drift
# when a longer-labelled field is added to either table above.
_LABEL_WIDTH = max(len(label) for label, _ in (*_SHOW_HEADER_FIELDS, *_SHOW_OPTIONAL_FIELDS)) + 2


def _echo_defect_human(report: dict) -> None:
    """Render one report for a terminal: the aligned header/optional field
    columns, the nested ``bundle_version`` block, then a copy-paste reproduce
    hint. The ``--json`` path bypasses this and emits the raw dict."""
    for label, key in _SHOW_HEADER_FIELDS:
        click.echo(f"{label + ':':<{_LABEL_WIDTH}}{report.get(key)}")
    for label, key in _SHOW_OPTIONAL_FIELDS:
        value = report.get(key)
        if value:
            click.echo(f"{label + ':':<{_LABEL_WIDTH}}{value}")
    bv = report.get("bundle_version")
    if bv:
        click.echo("bundle_version:")
        for k, v in bv.items():
            click.echo(f"  {k}: {v}")
    params = report.get("parameters")
    seed = report.get("seed")
    click.echo("\n--- reproduce ---")
    click.echo(f"  generator: {report.get('generator')}")
    click.echo(f"  seed:      {seed}")
    click.echo(f"  parameters: {json.dumps(params or {}, ensure_ascii=False)}")
    click.echo(
        "  (equivalent to POST /api/"
        f"{report.get('generator')} with the parameters above against the "
        "stamped bundle_version)"
    )


@defects.command("show")
@_env_options
@click.argument("defect_id")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit raw JSON.")
def show_defect_cmd(env, profile, defect_id, as_json) -> None:
    """Show a single report in full, with a reproduction hint."""
    from wyrd.defects import DefectsError, get_defect

    try:
        report = get_defect(defect_id, env=env, profile=_resolve_profile(env, profile))
    except DefectsError as exc:
        raise click.ClickException(str(exc)) from exc
    if report is None:
        raise click.ClickException(f"no defect with id {defect_id!r}")

    if as_json:
        click.echo(json.dumps(report, indent=2, ensure_ascii=False))
        return

    _echo_defect_human(report)


@defects.command("accept")
@_env_options
@click.argument("defect_id")
@click.option("--ticket", default=None, help="bd ticket filed for this defect (e.g. wyrd-xxxx).")
@click.option("--note", default=None, help="Triage note.")
def accept_defect_cmd(env, profile, defect_id, ticket, note) -> None:
    """Mark a report accepted (real defect, triaged)."""
    from wyrd.defects import STATUS_ACCEPTED, DefectsError, update_status

    try:
        updated = update_status(
            defect_id,
            STATUS_ACCEPTED,
            ticket=ticket,
            note=note,
            env=env,
            profile=_resolve_profile(env, profile),
        )
    except DefectsError as exc:
        raise click.ClickException(str(exc)) from exc
    suffix = f" → {ticket}" if ticket else ""
    click.echo(f"accepted {updated.get('id')}{suffix}")


@defects.command("dismiss")
@_env_options
@click.argument("defect_id")
@click.option(
    "--reason", "note", default=None, help="Why it's being dismissed (bogus/dup/wontfix)."
)
def dismiss_defect_cmd(env, profile, defect_id, note) -> None:
    """Mark a report dismissed (bogus / duplicate / wontfix)."""
    from wyrd.defects import STATUS_DISMISSED, DefectsError, update_status

    try:
        updated = update_status(
            defect_id,
            STATUS_DISMISSED,
            note=note,
            env=env,
            profile=_resolve_profile(env, profile),
        )
    except DefectsError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"dismissed {updated.get('id')}")
