"""Unit tests for wyrd.app helpers (count coercion, query-param flattening)."""

from __future__ import annotations

import pytest

from wyrd.app import _coerce_count, _coerce_query_params


def test_coerce_count_accepts_int_in_range():
    for n in (1, 5, 10):
        assert _coerce_count(n) == n


def test_coerce_count_accepts_numeric_string():
    assert _coerce_count("3") == 3


def test_coerce_count_below_minimum_rejected():
    with pytest.raises(ValueError, match="between 1 and 10"):
        _coerce_count(0)


def test_coerce_count_above_maximum_rejected():
    with pytest.raises(ValueError, match="between 1 and 10"):
        _coerce_count(11)


def test_coerce_count_non_integer_rejected():
    with pytest.raises(ValueError, match="must be an integer"):
        _coerce_count("five")


def test_coerce_query_params_single_value_flattened():
    assert _coerce_query_params({"culture": ["english"]}) == {"culture": "english"}


def test_coerce_query_params_repeated_key_kept_as_list():
    assert _coerce_query_params({"tags": ["religion", "tree"]}) == {"tags": ["religion", "tree"]}


def test_coerce_query_params_mixed():
    out = _coerce_query_params(
        {"culture": ["english"], "tags": ["religion", "tree"], "seed": ["42"]}
    )
    assert out == {"culture": "english", "tags": ["religion", "tree"], "seed": "42"}
