from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from metergraph_core import ModelUsage, load_catalog

_spec = importlib.util.spec_from_file_location("agent_cost_budget_example", Path(__file__).with_name("main.py"))
assert _spec is not None and _spec.loader is not None
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)

CATALOG = load_catalog()


def _expected(usage: dict[str, int]) -> Decimal:
    result = CATALOG.calculate_cost(
        model=example.MODEL,
        provider="anthropic",
        at=datetime.now(timezone.utc),
        usage=ModelUsage(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read_tokens=usage["cache_read_input_tokens"],
            cache_write_tokens=usage["cache_creation_input_tokens"],
        ),
    )
    return result.cost


def test_the_agent_prices_every_turn_and_finishes():
    run = example.run_agent(
        example.ScriptedAnthropic(), "weather", budget_usd=Decimal("1"), catalog=CATALOG
    )

    assert run.stopped_by == "end_turn"
    assert run.answer == "It is 21°C (69.8°F) in Lisbon."
    assert [turn.status for turn in run.turns] == ["priced"] * 3
    assert [turn.cost for turn in run.turns] == [_expected(step["usage"]) for step in example.SCRIPT]
    assert run.total == sum(turn.cost for turn in run.turns)


def test_the_agent_stops_at_its_budget():
    run = example.run_agent(
        example.ScriptedAnthropic(), "weather", budget_usd=Decimal("0.000001"), catalog=CATALOG
    )

    assert run.stopped_by == "budget"
    assert len(run.turns) == 1
    assert run.answer is None


def test_tools_return_synthetic_values():
    assert example.run_tool("get_weather", {"city": "Lisbon"}) == "21.0"
    assert example.run_tool("celsius_to_fahrenheit", {"celsius": 21}) == "69.8"
