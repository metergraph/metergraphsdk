# Agent cost budget

Use this example when your code needs the cost of a model call while it is
still running: to stop an agent at a budget, pick a cheaper model mid-task, or
log spend next to the result. It is a small Anthropic tool-use agent that
prices each response with
[`metergraph-core`](https://pypi.org/project/metergraph-core/) and stops when
its running total reaches `--budget-usd`.

Pricing happens in your process, from the response you already hold. It does
not need the MeterGraph server, and nothing is sent anywhere to get the price.

## What the example adds

Open [`main.py`](main.py). The agent loop is ordinary Anthropic tool-use code.
The pricing parts are:

1. `load_catalog()` loads the pricing catalog bundled with `metergraph-core`.
2. `catalog.lookup_price(model=..., provider="anthropic", at=...)` prints the
   rates the model is billed at before the run starts.
3. `cost_from_response("anthropic", response, catalog=catalog)` prices each
   response as returned by `client.messages.create(...)`. It reads the model,
   the usage counts (including cache reads and writes) and the service tier,
   and returns the cost, its breakdown and the catalog price row it used.

The same call prices OpenAI Chat Completions and Responses, Gemini
`generateContent` and Bedrock Converse responses. See the
[`metergraph-core` README](https://github.com/metergraph/metergraph/tree/main/core#costing-a-provider-response).

A result's `status` is `priced`, `partial` or `unpriced`. `partial` means part
of the cost depends on something the catalog cannot price, such as a priority
service tier, and `missing_dimensions` says what. The example counts an
unpriced turn as zero; a production budget should decide what to do instead.

## Run it

```bash
pip install metergraph-core anthropic
python examples/pricing/agent-cost-budget/main.py
```

Without `ANTHROPIC_API_KEY`, the example replays a scripted three-turn
conversation in the Anthropic SDK's own response types, so it runs offline:

```text
anthropic/claude-haiku-4.5 per 1M tokens: input $1.0, output $5.0, cache read $0.1 (effective 2025-10-01)
ANTHROPIC_API_KEY is not set: replaying a scripted conversation.
turn 1: $0.00119500 (priced), total $0.00119500
turn 2: $0.00040300 (priced), total $0.00159800
turn 3: $0.00028300 (priced), total $0.00188100
stopped by end_turn after 3 turns, $0.00188100 spent
It is 21°C (69.8°F) in Lisbon.
```

Your figures follow the catalog version you have installed. Set a lower budget
to see the agent stop early:

```bash
python examples/pricing/agent-cost-budget/main.py --budget-usd 0.0005
```

With `ANTHROPIC_API_KEY` set, it calls the API. Set `METERGRAPH_INGEST_URL` and
`METERGRAPH_APP_TOKEN` as well, and install `metergraph`, to also send each call
to MeterGraph through `metergraph.wrap()`. `wrap()` returns non-streamed
responses unchanged, so local pricing works the same either way.

## Limits

- Fees outside token usage, such as web search, are not included.
- Price a streamed call from its final message (`stream.get_final_message()`).
- The bundled catalog is as current as the installed `metergraph-core`.
  Upgrade it to pick up new models and price changes.
