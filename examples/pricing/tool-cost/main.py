"""What do tools a task never uses cost, with and without prompt caching?

Runs one small agent task four ways: with only the two tools it needs or with
fifteen more it never uses, each with prompt caching off and on. Every
response is priced with metergraph-core's cost_from_response, and the totals
are printed side by side. Needs ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message
from metergraph_core import LoadedCatalog, cost_from_response, load_catalog

MODEL = "claude-opus-5-5"
RUNS = 3
TASK = "What is the temperature in Lisbon right now, in Fahrenheit?"
SYSTEM = "You are a concise assistant. Use the tools to answer, then reply in one sentence."


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": list(properties)},
    }


def _text(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


NEEDED_TOOLS = [
    _tool("get_weather", "Current temperature in Celsius for a city.", {"city": _text("City name, for example Lisbon.")}),
    _tool(
        "celsius_to_fahrenheit",
        "Convert a Celsius temperature to Fahrenheit.",
        {"celsius": {"type": "number", "description": "Temperature in Celsius."}},
    ),
]

# Tools a real assistant might carry for other tasks. This task never calls them.
UNUSED_TOOLS = [
    _tool("search_orders", "Search a customer's orders by status, date range or product. Returns order IDs, totals and fulfilment status.",
          {"customer_id": _text("Customer ID."), "status": _text("Order status: pending, shipped, delivered or returned."), "since": _text("ISO 8601 start date.")}),
    _tool("refund_order", "Refund all or part of an order to its original payment method. Partial refunds need an amount and a reason.",
          {"order_id": _text("Order ID."), "amount": _text("Amount to refund, in the order currency."), "reason": _text("Reason shown to the customer.")}),
    _tool("create_support_ticket", "Open a support ticket for a customer with a subject, body and priority. Returns the ticket ID.",
          {"customer_id": _text("Customer ID."), "subject": _text("One-line summary."), "body": _text("Full description."), "priority": _text("low, normal, high or urgent.")}),
    _tool("send_email", "Send a plain-text email from the support address. Use only for messages the customer asked for.",
          {"to": _text("Recipient email address."), "subject": _text("Subject line."), "body": _text("Plain-text body.")}),
    _tool("lookup_customer", "Find a customer by email address or phone number and return their profile, plan and account status.",
          {"query": _text("Email address or phone number.")}),
    _tool("update_shipping_address", "Change the shipping address on an order that has not shipped yet.",
          {"order_id": _text("Order ID."), "address": _text("Full postal address, one line.")}),
    _tool("schedule_callback", "Book a phone callback with a support agent in the customer's time zone.",
          {"customer_id": _text("Customer ID."), "time": _text("ISO 8601 date and time."), "topic": _text("What the call is about.")}),
    _tool("check_inventory", "Check stock for a product SKU across warehouses and return quantities and restock dates.",
          {"sku": _text("Product SKU."), "warehouse": _text("Warehouse code, or all.")}),
    _tool("apply_discount_code", "Validate a discount code and apply it to an open cart. Returns the new cart total.",
          {"cart_id": _text("Cart ID."), "code": _text("Discount code as the customer typed it.")}),
    _tool("get_invoice", "Return the invoice for an order as a list of line items with tax and totals.",
          {"order_id": _text("Order ID.")}),
    _tool("translate_text", "Translate text into another language, keeping product names and order IDs unchanged.",
          {"text": _text("Text to translate."), "target_language": _text("ISO 639-1 language code.")}),
    _tool("search_knowledge_base", "Search help-centre articles and return the most relevant titles, URLs and excerpts.",
          {"query": _text("Search terms."), "limit": {"type": "integer", "description": "Maximum number of results."}}),
    _tool("escalate_to_human", "Hand the conversation to a human agent with a short summary of what has been tried.",
          {"summary": _text("What the customer needs and what was tried.")}),
    _tool("cancel_subscription", "Cancel a customer's subscription at the end of the current billing period.",
          {"customer_id": _text("Customer ID."), "reason": _text("Cancellation reason.")}),
    _tool("get_exchange_rate", "Return the current exchange rate between two currencies.",
          {"base": _text("ISO 4217 code to convert from."), "quote": _text("ISO 4217 code to convert to.")}),
]

SCENARIOS = {
    "2 tools, no caching": (NEEDED_TOOLS, False),
    "2 tools, caching": (NEEDED_TOOLS, True),
    "17 tools, no caching": (NEEDED_TOOLS + UNUSED_TOOLS, False),
    "17 tools, caching": (NEEDED_TOOLS + UNUSED_TOOLS, True),
}


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "get_weather":
        return "21.0"
    if name == "celsius_to_fahrenheit":
        return str(float(arguments["celsius"]) * 9 / 5 + 32)
    return "This tool is not available for this task."


def run_task(client: Any, tools: list[dict[str, Any]], caching: bool, model: str) -> list[Message]:
    """One agent run: call the model, run the tools it asks for, repeat."""
    # With caching on, one breakpoint after the system prompt caches the
    # tool definitions and the system prompt, which come first in a request.
    system: Any = [{"type": "text", "text": SYSTEM, **({"cache_control": {"type": "ephemeral"}} if caching else {})}]
    messages: list[dict[str, Any]] = [{"role": "user", "content": TASK}]
    responses = []
    for _ in range(6):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            output_config={"effort": "low"},
            system=system,
            tools=tools,
            messages=messages,
        )
        responses.append(response)
        if response.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": block.id, "content": run_tool(block.name, block.input)}
                for block in response.content
                if block.type == "tool_use"
            ],
        })
    return responses


@dataclass
class Totals:
    calls: int = 0
    input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0
    cost: Decimal = Decimal(0)
    # Responses the catalog could not price fully, as "status: model".
    not_priced: list[str] = field(default_factory=list)

    def add(self, response: Message, catalog: LoadedCatalog) -> None:
        # Price the response as returned by the SDK.
        result = cost_from_response("anthropic", response, catalog=catalog)
        usage = response.usage
        self.calls += 1
        self.input += usage.input_tokens
        self.cache_write += usage.cache_creation_input_tokens or 0
        self.cache_read += usage.cache_read_input_tokens or 0
        self.output += usage.output_tokens
        if result.status == "priced":
            self.cost += result.cost
        else:
            missing = ", ".join(result.missing_dimensions) or ", ".join(result.reasons)
            self.not_priced.append(f"{result.status} ({missing}): {response.model}")


def price(responses: dict[str, list[list[Message]]], catalog: LoadedCatalog) -> dict[str, Totals]:
    totals = {}
    for scenario, runs in responses.items():
        totals[scenario] = Totals()
        for run in runs:
            for response in run:
                totals[scenario].add(response, catalog)
    return totals


def report(totals: dict[str, Totals]) -> None:
    print(f"{'scenario':<22}{'calls':>6}{'input':>8}{'cache write':>13}{'cache read':>12}{'output':>8}{'cost':>12}")
    for scenario, row in totals.items():
        cost = "incomplete" if row.not_priced else "$" + format(row.cost, ".6f")
        print(
            f"{scenario:<22}{row.calls:>6}{row.input:>8}{row.cache_write:>13}"
            f"{row.cache_read:>12}{row.output:>8}{cost:>12}"
        )
    not_priced = sorted({entry for row in totals.values() for entry in row.not_priced})
    if not_priced:
        print("Some responses could not be fully priced, so no comparison is shown:")
        for entry in not_priced:
            print(f"  {entry}")
        return
    for caching in ("no caching", "caching"):
        two, seventeen = totals[f"2 tools, {caching}"], totals[f"17 tools, {caching}"]
        extra = seventeen.cost - two.cost
        share = extra / seventeen.cost * 100 if seventeen.cost else Decimal(0)
        print(f"15 unused tools, {caching}: ${extra:.6f} extra ({share:.0f}% of the 17-tool total)")


def run_scenarios(client: Any, model: str) -> dict[str, list[list[Message]]]:
    return {
        scenario: [run_task(client, tools, caching, model) for _ in range(RUNS)]
        for scenario, (tools, caching) in SCENARIOS.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    print(f"Calling {args.model}: {len(SCENARIOS)} scenarios x {RUNS} runs.")
    responses = run_scenarios(Anthropic(), args.model)
    report(price(responses, load_catalog()))


if __name__ == "__main__":
    main()
