"""Tests for the top-level wyrd CLI (wyrd.cli) — the `manifest` command's
parity with GET /api/manifest."""

from __future__ import annotations

import json

from click.testing import CliRunner

from wyrd.app import create_app
from wyrd.cli import main


def _runner() -> CliRunner:
    # Keep stderr (the loaders' INFO/WARNING logging) out of stdout so the
    # captured `manifest` output is clean JSON. Click <8.2 takes mix_stderr;
    # >=8.2 separates the streams by default and dropped the kwarg.
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


def _cli_manifest() -> dict:
    res = _runner().invoke(main, ["manifest"])
    assert res.exit_code == 0, res.output
    out = res.stdout if hasattr(res, "stdout") else res.output
    return json.loads(out)


def test_cli_manifest_includes_era_stages():
    """`wyrd manifest` exposes each generator's optional ``era_stages`` axis —
    the kenning generator defines it, so it must be present and truthy (it was
    silently dropped before, breaking the 'same as /api/manifest' contract)."""
    gens = {g["name"]: g for g in _cli_manifest()["generators"]}
    assert "kenning" in gens
    assert "era_stages" in gens["kenning"]
    assert gens["kenning"]["era_stages"], "kenning era_stages should be populated"


def test_cli_manifest_generator_entries_match_api():
    """The CLI manifest's per-generator entries are byte-for-byte the API's
    (name/display_name/description/details/legend/input_schema/era_stages). The
    API's top-level ``config`` block (env-resolved SPA feature flags) is API-only
    and intentionally absent from the CLI manifest."""
    cli_gens = {g["name"]: g for g in _cli_manifest()["generators"]}

    api_manifest = create_app().test_client().get("/api/manifest").get_json()
    api_gens = {g["name"]: g for g in api_manifest["generators"]}

    assert cli_gens == api_gens
    assert "config" in api_manifest  # API carries it…
    assert "config" not in _cli_manifest()  # …and the CLI deliberately does not.
