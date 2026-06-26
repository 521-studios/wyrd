"""Flask app: manifest, dispatcher, error handling."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from flask import Flask, jsonify, request

from wyrd import registry
from wyrd.envelope import envelope
from wyrd.feature_flags import apply_env_defaults, resolve_feature_config
from wyrd.generators.kenning.errors import EmptyEligiblePool
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
    """Wire the wyrd-package logger level from the LOG_LEVEL env var.
    Idempotent: re-calling reconfigures without duplicating handlers
    (Lambda's runtime already installs a handler that forwards to
    CloudWatch; we only set levels).

    ``LOG_LEVEL`` only affects wyrd's loggers. Root + third-party
    loggers (botocore, urllib3, s3transfer) stay pinned at WARNING
    regardless of the env var — staging caught a 10s timeout when
    LOG_LEVEL=DEBUG let botocore emit hundreds of DEBUG lines per
    request, burning the full CPU budget on string formatting +
    CloudWatch I/O for telemetry the operator doesn't need.
    """
    raw = os.environ.get(_LOG_LEVEL_ENV, "INFO").upper().strip()
    level = getattr(logging, raw, logging.INFO)
    logging.getLogger().setLevel(logging.WARNING)
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
                        # wyrd-rogd.12: optional per-family era-stage axis for
                        # generators that expose one (kenning); the key is always
                        # present, null for generators that don't define it.
                        "era_stages": getattr(g, "era_stages", None),
                    }
                    for g in registry.all_generators()
                ],
                # wyrd-0gou: env-resolved SPA feature flags + default-value
                # overrides. The SPA reads this to decide which advanced
                # config options to render (default off; staging sets
                # WYRD_FF_ALL=true). Resolved per-request so flipping a
                # Lambda env var takes effect without a redeploy.
                "config": resolve_feature_config(),
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

    @app.post("/api/defects")
    def report_defect():
        # wyrd-dsl5: a user flagged a generated name as defective. The body
        # carries the full reproduction context the SPA had on hand plus the
        # REQUIRED free-text reason. Generic across generators — `generator`
        # is just a stored field.
        body = request.get_json(silent=True) or {}
        return _record_defect(body)

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "not_found"}), 404

    # wyrd-g1wp: SnapStart. create_app() runs at Lambda INIT (lambda_handler
    # imports + calls it at module load), so this is the snapshot point. Register
    # the after-restore hook BEFORE warming the caches, then preload the bundle +
    # cultures so the parsed objects are captured in the snapshot. Both no-op off
    # SnapStart (preload gates on AWS_LAMBDA_INITIALIZATION_TYPE==snap-start; the
    # after-restore hook only fires under SnapStart), so dev + tests are unaffected.
    from wyrd.snapstart import preload_runtime, register_snapstart_hooks

    register_snapstart_hooks()
    preload_runtime()

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

    # wyrd-7f22: apply WYRD_DEFAULT_<OPTION> env overrides for options the
    # request omits, BEFORE generation. The SPA omits any field still at its
    # seeded default (wyrd-14hn), so a value-changing env default (e.g.
    # WYRD_DEFAULT_NOVELTY=0.1) would otherwise never reach the generator. No-op
    # when no WYRD_DEFAULT_* is set, so local / CLI / tests stay bit-stable.
    apply_env_defaults(params, generator.input_schema().get("properties", {}))

    # Snapshot params before count/seed are popped so the log line carries the
    # actual operator input shape (count, mode, knobs); the snapshot drops
    # ``seed`` by key whether or not it has been popped yet.
    param_snapshot = {k: v for k, v in params.items() if k != "seed"}
    started = time.perf_counter()

    try:
        # Resolve the seed INSIDE the try: a non-coercible seed (a non-numeric
        # string, a list) raises ValueError → the except below maps it to a 400
        # bad_params. Outside the try (where this used to live) it crashed as an
        # uncaught 500.
        seed = resolve_seed(params.pop("seed", None))
        if generator.multi_result:
            # Multi-result generators produce their whole result set in ONE
            # generate_all call, so the dispatcher neither loops nor derives
            # per-result sub-seeds. ``count`` is LEFT in params for the generator
            # to use as it sees fit: one that sizes from the INPUT (explain —
            # every decomposition) ignores it, while one that sizes to a
            # requested N (era-map — region size) reads + validates it itself.
            # (Popping it here silently pinned era-map to its default count,
            # ignoring the user's value over the API — the CLI, which calls
            # generate_all directly, never hit the pop and already honored it.)
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
    except EmptyEligiblePool as e:
        # wyrd-hh2m: a valid request whose gate (culture/era/stratum/tags) or
        # thin bundle data leaves zero eligible morphemes. Distinct from a
        # malformed request (400) and from a genuine crash (500): the input was
        # well-formed, it just can't be satisfied → 422 Unprocessable Entity.
        elapsed_ms = (time.perf_counter() - started) * 1000
        _logger.info(
            "dispatch empty_eligible_pool generator=%s elapsed_ms=%.1f params=%r detail=%s",
            generator_name,
            elapsed_ms,
            param_snapshot,
            e,
        )
        return jsonify({"error": "empty_eligible_pool", "detail": str(e)}), 422
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
    # wyrd-dsl5: stamp the data/bundle version so the SPA can echo it back on
    # a defect report. A version-read failure must never sink a roll — the
    # stamp is best-effort metadata, so swallow and ship without it.
    try:
        # `or None`: a generator may return an empty dict (e.g. an empty
        # bundle_metadata table) — treat that as "no version" so the envelope
        # omits the key rather than emitting a useless bundle_version: {}.
        bundle_version = generator.runtime_version() or None
    except Exception:
        # WARNING (not DEBUG): a swallowed version-stamp failure should be
        # visible in default INFO deployments — a misbehaving runtime_version()
        # implementation otherwise looks identical to "no version".
        _logger.warning("runtime_version failed for generator=%s", generator.name, exc_info=True)
        bundle_version = None
    return jsonify(
        envelope(
            generator=generator.name,
            # Echo the operator's params, minus internal ``_``-prefixed request-
            # scratch keys (e.g. wyrd-dag4's ``_vector_pool_cache``, which holds
            # non-serializable request-scoped state). The client never sends
            # ``_``-prefixed keys, so this returns exactly what it sent.
            parameters={k: v for k, v in params.items() if not k.startswith("_")},
            seed=seed,
            results=results,
            bundle_version=bundle_version,
        )
    )


def _stripped_str(value: Any) -> str:
    """A required free-text field, normalized to a stripped string. A non-string
    (number, list, null) is treated as absent (``""``) so a malformed report is a
    clean 400 rather than a 500 from ``.strip()`` on a non-string."""
    return value.strip() if isinstance(value, str) else ""


def _record_defect(body: dict[str, Any]):
    """Validate + persist a defect report (wyrd-dsl5). Requires a non-empty
    ``reason`` and a ``generator``; everything else is reproduction context
    stored verbatim. Returns ``201 {id, status}`` on success, ``400`` on a
    bad report, ``503`` when the DynamoDB table is unconfigured / unreachable.
    """
    from wyrd.defects import DefectsError, record_defect

    if not isinstance(body, dict):
        # A non-object JSON body (list / string / number) would AttributeError
        # on .get below → 500. Reject cleanly instead.
        return jsonify({"error": "bad_report", "detail": "body must be a JSON object"}), 400

    # A non-STRING reason/generator (a number, list, or object) is a malformed
    # report, not a valid one: treat it as absent so it falls to the clean 400
    # below. Calling .strip() on a non-string (the old `(5 or "").strip()`)
    # would AttributeError into a 500 — masking a bad request as a server crash.
    reason = _stripped_str(body.get("reason"))
    generator = _stripped_str(body.get("generator"))
    if not reason:
        return jsonify({"error": "bad_report", "detail": "reason is required"}), 400
    if not generator:
        return jsonify({"error": "bad_report", "detail": "generator is required"}), 400

    report = {
        "generator": generator,
        "reason": reason,
        "result": body.get("result"),
        "seed": body.get("seed"),
        "parameters": body.get("parameters"),
        "explanation": body.get("explanation"),
        "components": body.get("components"),
        "morphemes_by_word": body.get("morphemes_by_word"),
        "bundle_version": body.get("bundle_version"),
    }
    try:
        created = record_defect(report)
    except DefectsError as exc:
        _logger.warning("defect record failed: %s", exc)
        return jsonify({"error": "defects_unavailable", "detail": str(exc)}), 503

    _logger.info(
        "defect recorded id=%s generator=%s name=%r",
        created["id"],
        generator,
        body.get("result"),
    )
    return jsonify(created), 201


def _coerce_count(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"count must be an integer, got {value!r}") from e
    if n < 1 or n > MAX_COUNT:
        raise ValueError(f"count must be between 1 and {MAX_COUNT}, got {n}")
    return n
