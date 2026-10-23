"""A tool-using agent that prices every turn and stops at a budget.

Each model response is priced in-process with metergraph-core's
cost_from_response, so the loop can act on what it has spent so far. Without
ANTHROPIC_API_KEY it replays a scripted conversation, so it runs offline.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message
from metergraph_core import LoadedCatalog, cost_from_response, load_catalog

MODEL = "claude-haiku-4-5"

TOOLS = [
    {
        "name": "get_weather",
        "description": "Current temperature in Celsius for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "input_schema": {
            "type": "object",
            "properties": {"celsius": {"type": "number"}},
            "required": ["celsius"],
        },
    },
]

_TEMPERATURES = {"lisbon": 21.0, "oslo": 4.0}


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "get_weather":
        return str(_TEMPERATURES.get(arguments["city"].lower(), 15.0))
    if name == "celsius_to_fahrenheit":
        return str(arguments["celsius"] * 9 / 5 + 32)
    raise ValueError(f"unknown tool {name}")


@dataclass
class Turn:
    cost: Decimal
    status: str


@dataclass
class AgentRun:
    answer: str | None
    stopped_by: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def total(self) -> Decimal:
        return sum((turn.cost for turn in self.turns), Decimal(0))


def show_price(catalog: LoadedCatalog) -> None:
    """Print the rates a call to MODEL is billed at today."""
    found = catalog.lookup_price(model=MODEL, provider="anthropic", at=datetime.now(timezone.utc))
    price = found.price
    if price is None:
        print(f"{MODEL}: no price in the catalog ({found.status})")
        return
    print(
        f"{found.model_id} per 1M tokens: input ${price.input_per_mtok}, "
        f"output ${price.output_per_mtok}, cache read ${price.cache_read_per_mtok} "
        f"(effective {price.effective_from:%Y-%m-%d})"
    )


def run_agent(
    client: Any,
    task: str,
    *,
    budget_usd: Decimal,
    catalog: LoadedCatalog,
    max_turns: int = 8,
) -> AgentRun:
    run = AgentRun(answer=None, stopped_by="max_turns")
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    while len(run.turns) < max_turns:
        response = client.messages.create(
            model=MODEL, max_tokens=512, tools=TOOLS, messages=messages
        )

        # Price the response we already hold. No token counts to extract.
        result = cost_from_response("anthropic", response, catalog=catalog)
        run.turns.append(Turn(result.cost or Decimal(0), result.status))
        print(f"turn {len(run.turns)}: ${result.cost} ({result.status}), total ${run.total}")

        if response.stop_reason != "tool_use":
            run.answer = "".join(block.text for block in response.content if block.type == "text")
            run.stopped_by = "end_turn"
            return run
        if run.total >= budget_usd:
            run.stopped_by = "budget"
            return run

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": block.id, "content": run_tool(block.name, block.input)}
                for block in response.content
                if block.type == "tool_use"
            ],
        })
    return run


class ScriptedAnthropic:
    """Replays synthetic Messages API responses, in the SDK's own types."""

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self._script = iter(script or SCRIPT)
        self.messages = self

    def create(self, **_: Any) -> Message:
        return Message.model_validate(next(self._script))


def _message(number: int, content: list[dict[str, Any]], stop_reason: str, usage: dict[str, int]) -> dict[str, Any]:
    return {
        "id": f"msg_example_{number}",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, **usage},
    }


SCRIPT = [
    _message(
        1,
        [{"type": "tool_use", "id": "toolu_example_1", "name": "get_weather", "input": {"city": "Lisbon"}}],
        "tool_use",
        {"input_tokens": 420, "output_tokens": 60, "cache_creation_input_tokens": 380},
    ),
    _message(
        2,
        [{"type": "tool_use", "id": "toolu_example_2", "name": "celsius_to_fahrenheit", "input": {"celsius": 21}}],
        "tool_use",
        {"input_tokens": 90, "output_tokens": 55, "cache_read_input_tokens": 380},
    ),
    _message(
        3,
        [{"type": "text", "text": "It is 21°C (69.8°F) in Lisbon."}],
        "end_turn",
        {"input_tokens": 120, "output_tokens": 25, "cache_read_input_tokens": 380},
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-usd", type=Decimal, default=Decimal("0.01"))
    parser.add_argument("--task", default="What is the temperature in Lisbon, in Fahrenheit?")
    args = parser.parse_args()

    catalog = load_catalog()
    show_price(catalog)

    wrapped = False
    if os.environ.get("ANTHROPIC_API_KEY"):
        client: Any = Anthropic()
        if os.environ.get("METERGRAPH_INGEST_URL"):
            # Optional: also send each call to MeterGraph for attribution.
            # wrap() returns non-streamed responses unchanged, so pricing
            # them locally works the same way.
            import metergraph

            metergraph.init(repository="owner/repository", environment="example")
            client = metergraph.wrap(client)
            wrapped = True
    else:
        print("ANTHROPIC_API_KEY is not set: replaying a scripted conversation.")
        client = ScriptedAnthropic()

    run = run_agent(client, args.task, budget_usd=args.budget_usd, catalog=catalog)
    turns = len(run.turns)
    print(f"stopped by {run.stopped_by} after {turns} turn{'' if turns == 1 else 's'}, ${run.total} spent")
    if run.answer:
        print(run.answer)
    if wrapped:
        import metergraph

        metergraph.shutdown()


if __name__ == "__main__":
    main()
