"""Flask app: manifest, dispatcher, error handling."""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request

from wyrd import registry
from wyrd.envelope import envelope
from wyrd.seed import MAX_SAFE_INTEGER, resolve_seed, rng_for

MAX_COUNT = 10


def create_app() -> Flask:
    registry.discover()
    app = Flask(__name__)

    # wyrd-20pz: SPA-serving moved out of Flask. Production: CloudFront
    # serves the SPA from S3; Lambda handles /api/* only. Dev: run
    # `npm run dev` in spa-next/ for the Vite dev server on :5173,
    # which proxies /api/* back to this Flask app on :5000. The
    # previous Flask SPA-serve code lived here pre-Vite when the
    # SPA was a single bundled file; the Vite-built SPA has hashed-
    # asset shape that Flask isn't a good fit for.

    @app.get("/api/manifest")
    def manifest():
        return jsonify(
            {
                "generators": [
                    {
                        "name": g.name,
                        "display_name": g.display_name,
                        "description": g.description,
                        "details": g.details,
                        "legend": g.legend,
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
        if generator.multi_result:
            # Multi-result generators size their own output from the input
            # (e.g. an explainer returning every matching decomposition), so
            # count and per-result sub-seeds don't apply.
            params.pop("count", None)
            results = generator.generate_all(params, seed)
        else:
            count = _coerce_count(params.pop("count", 5))
            # Sub-seeds derived deterministically from the top-level seed so that
            # the same (seed, count) pair always reproduces the same set of results.
            seed_rng = rng_for(seed)
            # wyrd-aof8: cap sub-seeds at JS Number safe range so
            # they round-trip through copy/paste in the SPA.
            results = [
                generator.generate(params, seed_rng.randrange(MAX_SAFE_INTEGER + 1))
                for _ in range(count)
            ]
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
