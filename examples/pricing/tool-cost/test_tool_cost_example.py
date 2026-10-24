"""Checks the example's setup without calling the API."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("tool_cost_example", Path(__file__).with_name("main.py"))
assert _spec is not None and _spec.loader is not None
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def test_scenarios_differ_only_in_tool_count_and_caching():
    assert {name: (len(tools), caching) for name, (tools, caching) in example.SCENARIOS.items()} == {
        "2 tools, no caching": (2, False),
        "2 tools, caching": (2, True),
        "17 tools, no caching": (17, False),
        "17 tools, caching": (17, True),
    }
    for tools, _ in example.SCENARIOS.values():
        assert tools[:2] == example.NEEDED_TOOLS
    names = [tool["name"] for tool in example.NEEDED_TOOLS + example.UNUSED_TOOLS]
    assert len(names) == len(set(names))


def test_tools_return_synthetic_values():
    assert example.run_tool("get_weather", {"city": "Lisbon"}) == "21.0"
    assert example.run_tool("celsius_to_fahrenheit", {"celsius": 21}) == "69.8"
