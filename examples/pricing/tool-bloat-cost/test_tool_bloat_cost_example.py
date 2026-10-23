from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path

from metergraph_core import cost_from_response, load_catalog

_spec = importlib.util.spec_from_file_location("tool_bloat_cost_example", Path(__file__).with_name("main.py"))
assert _spec is not None and _spec.loader is not None
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)

CATALOG = load_catalog()
RECORDED = example.load(example.RECORDING)
TOTALS = example.price(RECORDED, CATALOG)


def test_the_recording_covers_every_scenario():
    assert list(RECORDED) == list(example.SCENARIOS)
    for runs in RECORDED.values():
        assert len(runs) == example.RUNS
        for run in runs:
            assert run[-1].stop_reason == "end_turn"


def test_each_total_is_the_sum_of_its_priced_responses():
    for scenario, runs in RECORDED.items():
        results = [cost_from_response("anthropic", response, catalog=CATALOG) for run in runs for response in run]
        assert all(result.status == "priced" for result in results)
        assert TOTALS[scenario].cost == sum((result.cost for result in results), Decimal(0))


def test_unused_tools_cost_without_caching_and_little_with_it():
    lean, bloated = TOTALS["2 tools, no caching"], TOTALS["17 tools, no caching"]
    cached_lean, cached_bloated = TOTALS["2 tools, caching"], TOTALS["17 tools, caching"]

    assert bloated.cost > lean.cost * 2
    # The bloated prefix is cached once and read on every later call.
    assert cached_bloated.cache_write > 0 and cached_bloated.cache_read > cached_bloated.cache_write
    assert cached_bloated.cost < bloated.cost / 2
    # The 2-tool prefix is under the model's minimum cacheable length, so
    # asking for caching changes nothing.
    assert cached_lean.cache_write == cached_lean.cache_read == 0
    assert cached_lean.cost == lean.cost


def test_tools_return_synthetic_values():
    assert example.run_tool("get_weather", {"city": "Lisbon"}) == "21.0"
    assert example.run_tool("celsius_to_fahrenheit", {"celsius": 21}) == "69.8"
