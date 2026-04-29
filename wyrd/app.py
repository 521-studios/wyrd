"""Flask app: manifest, dispatcher, error handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from wyrd import registry
from wyrd.envelope import envelope
from wyrd.seed import resolve_seed, rng_for

MAX_COUNT = 10

_SPA_DIR = Path(__file__).resolve().parent.parent / "spa"


def create_app() -> Flask:
    registry.discover()
    app = Flask(__name__)

    if _SPA_DIR.exists():
        # Dev-only SPA serving. In production CloudFront serves the SPA from S3
        # and Lambda only handles /api/*. Source spa/index.html uses __SHA__
        # placeholders for hashed asset names; the deploy.yml templates them at
        # upload time. Here we substitute "dev" and accept any hash on the way out.
        from flask import Response

        @app.get("/")
        def index():
            text = (_SPA_DIR / "index.html").read_text().replace("__SHA__", "dev")
            return Response(text, mimetype="text/html")

        @app.get("/<path:filename>")
        def spa_static(filename: str):
            target = _SPA_DIR / filename
            if target.is_file():
                return send_from_directory(_SPA_DIR, filename)
            # Strip a hash component: app.dev.js → app.js, style.dev.css → style.css
            parts = filename.rsplit(".", 2)
            if len(parts) == 3:
                unhashed = f"{parts[0]}.{parts[2]}"
                if (_SPA_DIR / unhashed).is_file():
                    return send_from_directory(_SPA_DIR, unhashed)
            return index()

    @app.get("/api/manifest")
    def manifest():
        return jsonify(
            {
                "generators": [
                    {
                        "name": g.name,
                        "display_name": g.display_name,
                        "description": g.description,
                        "input_schema": g.input_schema(),
                    }
                    for g in registry.all_generators()
                ]
            }
        )

    @app.get("/api/<generator_name>")
    def generate_get(generator_name: str):
        params = _coerce_query_params(request.args.to_dict(flat=False))
        return _dispatch(generator_name, params)

    @app.post("/api/<generator_name>")
    def generate_post(generator_name: str):
        body = request.get_json(silent=True) or {}
        return _dispatch(generator_name, body)

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "not_found"}), 404

    return app


def _coerce_query_params(args: dict[str, list[str]]) -> dict[str, Any]:
    """Flatten Flask's multi-value dict, preserving lists where the key repeats."""
    out: dict[str, Any] = {}
    for key, values in args.items():
        out[key] = values if len(values) > 1 else values[0]
    return out


def _dispatch(generator_name: str, params: dict[str, Any]):
    generator = registry.get(generator_name)
    if generator is None:
        return jsonify({"error": "unknown_generator", "name": generator_name}), 404

    seed = resolve_seed(params.pop("seed", None))

    try:
        count = _coerce_count(params.pop("count", 5))
        # Sub-seeds derived deterministically from the top-level seed so that the
        # same (seed, count) pair always reproduces the same set of results.
        seed_rng = rng_for(seed)
        results = [generator.generate(params, seed_rng.randrange(2**63)) for _ in range(count)]
    except (ValueError, KeyError) as e:
        return jsonify({"error": "bad_params", "detail": str(e)}), 400

    return jsonify(
        envelope(
            generator=generator.name,
            parameters=params,
            seed=seed,
            results=results,
        )
    )


def _coerce_count(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"count must be an integer, got {value!r}") from e
    if n < 1 or n > MAX_COUNT:
        raise ValueError(f"count must be between 1 and {MAX_COUNT}, got {n}")
    return n
