"""CPU tests: the indexed lookups reproduce upstream's linear scans exactly."""

import random

import pytest

from vllm_lens._worker_ext import (
    _by_external_id,
    _matching_keys,
    _parse_layer_request,
    _prefix_keys,
)


def _scan_keys(data, internal, sentinel):
    """Upstream ``_find_steering_configs`` key order."""
    keys = [k for k in data if internal.startswith(f"{k}-")]
    if sentinel and sentinel in data:
        keys.append(sentinel)
    return keys


def _scan_by_external(internal_ids, external_ids):
    """Upstream ``get_captured_states_batch`` assignment."""
    out = {}
    for internal in internal_ids:
        for ext in external_ids:
            if internal.startswith(f"{ext}-"):
                out.setdefault(ext, []).append(internal)
                break
    return out


def _random_id(rng):
    parts = [rng.choice(["req", "a", "chat", "0", "1", "12", "x-y"]) for _ in range(rng.randint(1, 3))]
    return "-".join(parts)


def test_prefix_keys():
    assert list(_prefix_keys("req-0-ab12cd34")) == ["req", "req-0"]
    assert list(_prefix_keys("nodash")) == []
    assert list(_prefix_keys("a--b")) == ["a", "a-"]


@pytest.mark.parametrize("seed", range(20))
def test_matching_keys_equals_scan(seed):
    rng = random.Random(seed)
    data, order = {}, {}
    for n, key in enumerate({_random_id(rng) for _ in range(rng.randint(0, 12))}):
        data[key] = n
        order[key] = n
    for _ in range(50):
        internal = f"{_random_id(rng)}-{rng.getrandbits(32):08x}"
        sentinel = rng.choice([None, "_steer_0", *list(data)[:2]])
        assert _matching_keys(data, order, internal, sentinel) == _scan_keys(
            data, internal, sentinel
        )


@pytest.mark.parametrize("seed", range(20))
def test_by_external_id_equals_scan(seed):
    rng = random.Random(seed)
    externals = [_random_id(rng) for _ in range(rng.randint(1, 10))]
    internals = [
        f"{rng.choice(externals + [_random_id(rng)])}-{rng.getrandbits(32):08x}"
        for _ in range(rng.randint(0, 30))
    ]
    assert _by_external_id(internals, externals) == _scan_by_external(internals, externals)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, (False, None)),
        (True, (True, None)),
        ([3, 5], (True, frozenset({3, 5}))),
        ("[1, 2]", (True, frozenset({1, 2}))),
        ("all", (True, None)),
        ([1, [2]], (True, frozenset({1}))),
        ([], (True, frozenset())),
    ],
)
def test_parse_layer_request(value, expected):
    assert _parse_layer_request(value) == expected
