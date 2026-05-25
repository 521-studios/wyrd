"""Flask app: manifest, dispatcher, error handling."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from flask import Flask, jsonify, request

from wyrd import registry
from wyrd.envelope import envelope
from wyrd.seed import MAX_SAFE_INTEGER, resolve_seed, rng_for

MAX_COUNT = 10

_logger = logging.getLogger(__name__)

# Lambda-side logging cadence. Set ``LOG_LEVEL=DEBUG`` in the function
# config to surface per-request timing + structured pipeline traces
# (defaults to INFO, which still emits the per-request summary line).
# wyrd-d90t staging diagnostics: needed visibility into the 10s
# timeouts that bare-bone REPORT lines weren't explaining.
_LOG_LEVEL_ENV = "LOG_LEVEL"


def _configure_logging() -> None:
    """Wire root + wyrd-package logger levels from the LOG_LEVEL env
    var. Idempotent: re-calling reconfigures without duplicating
    handlers (Lambda's runtime already installs a handler that
    forwards to CloudWatch; we only set levels)."""
    raw = os.environ.get(_LOG_LEVEL_ENV, "INFO").upper().strip()
    level = getattr(logging, raw, logging.INFO)
    # Set on root + the wyrd package logger; Lambda's preconfigured
    # root handler picks up emitted records and ships them to CW.
    logging.getLogger().setLevel(level)
    logging.getLogger("wyrd").setLevel(level)


def create_app() -> Flask:
    _configure_logging()
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
        _logger.info("dispatch unknown_generator name=%s", generator_name)
        return jsonify({"error": "unknown_generator", "name": generator_name}), 404

    seed = resolve_seed(params.pop("seed", None))
    # Snapshot params before count is popped so the log line carries
    # the actual operator input shape (count, mode, knobs).
    param_snapshot = {k: v for k, v in params.items() if k != "seed"}
    started = time.perf_counter()

    try:
        if generator.multi_result:
            # Multi-result generators size their own output from the input
            # (e.g. an explainer returning every matching decomposition), so
            # count and per-result sub-seeds don't apply.
            params.pop("count", None)
            results = generator.generate_all(params, seed)
            count = len(results) if hasattr(results, "__len__") else 1
        else:
            count = _coerce_count(params.pop("count", 5))
            # Sub-seeds derived deterministically from the top-level seed so that
            # the same (seed, count) pair always reproduces the same set of results.
            seed_rng = rng_for(seed)
            # wyrd-aof8: cap sub-seeds at JS Number safe range so
            # they round-trip through copy/paste in the SPA.
            slot_times: list[float] = []
            results = []
            for _ in range(count):
                sub_started = time.perf_counter()
                results.append(generator.generate(params, seed_rng.randrange(MAX_SAFE_INTEGER + 1)))
                slot_times.append(time.perf_counter() - sub_started)
            # Per-result timing breakdown helps localize a slow path to
            # a specific sub-seed (vs an init-time hot spot).
            if slot_times and _logger.isEnabledFor(logging.DEBUG):
                slot_ms = [f"{t * 1000:.1f}" for t in slot_times]
                _logger.debug(
                    "dispatch generator=%s per_result_ms=[%s]",
                    generator.name,
                    ",".join(slot_ms),
                )
    except (ValueError, KeyError) as e:
        elapsed_ms = (time.perf_counter() - started) * 1000
        _logger.info(
            "dispatch bad_params generator=%s elapsed_ms=%.1f params=%r detail=%s",
            generator_name,
            elapsed_ms,
            param_snapshot,
            e,
        )
        return jsonify({"error": "bad_params", "detail": str(e)}), 400

    elapsed_ms = (time.perf_counter() - started) * 1000
    _logger.info(
        "dispatch ok generator=%s count=%d elapsed_ms=%.1f seed=%d params=%r",
        generator.name,
        count,
        elapsed_ms,
        seed,
        param_snapshot,
    )
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
