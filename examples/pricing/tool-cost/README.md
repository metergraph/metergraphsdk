# Tool cost and prompt caching

This example measures what tool definitions a task never uses add to the cost
of an agent, with prompt caching off and on.

It runs one agent task ("What is the temperature in Lisbon right now, in
Fahrenheit?") four ways: with the 2 tools it needs, or with those 2 plus 15
it never calls, each with prompt caching off and on. Each scenario runs 3
times. Every response is priced with
[`metergraph-core`](https://pypi.org/project/metergraph-core/)'s
`cost_from_response`, and the totals are printed side by side.

## Run it

```bash
pip install metergraph-core anthropic
export ANTHROPIC_API_KEY=...
python examples/pricing/tool-cost/main.py
```

It makes about 36 requests to Claude Opus 5.5 at effort `low`. The run below
cost about $0.20. `--model` selects another model.

## Output from one run

Claude Opus 5.5, 23 October 2026, priced with metergraph-core 0.7.1:

```text
scenario               calls   input  cache write  cache read  output        cost
2 tools, no caching        9    5487            0           0     408   $0.030108
2 tools, caching           9    5487            0           0     408   $0.030108
17 tools, no caching       9   24495            0           0     408   $0.106140
17 tools, caching          9    1464         2559       20472     408   $0.030905
15 unused tools, no caching: $0.076032 extra (72% of the 17-tool total)
15 unused tools, caching: $0.000797 extra (3% of the 17-tool total)
```

- **Without caching, every call pays for every tool definition.** The 15
  unused tools add about 2,100 input tokens to each request, and an agent
  sends the tool list again on every turn. In this run they are 72% of the
  17-tool run's cost.
- **With caching, the tool definitions are written to the cache once and read
  on later calls.** The write costs 1.25 times the input price. A read costs a
  twentieth of it on Claude Opus 5.5 and a tenth on most other models. In this
  run the 17-tool run with caching costs about the same as the 2-tool run
  without it.
- **The 2-tool prompt was too short to cache.** Requests below the model's
  minimum cacheable length are processed without caching, and the response
  reports `cache_creation_input_tokens: 0`. The minimum depends on the model:
  Claude Haiku 4.5 needs 4,096 tokens, so on that model even the 17-tool
  prompt here may not cache.
- **A change to the tool list starts the cache over.** Tool definitions come
  first in the cached prefix, so adding, removing or reordering a tool
  invalidates the cache.

Results depend on the model, the tools, the number of turns per task, and how
often the cache expires. The default cache expires 5 minutes after its last
use.

## How it works

[`main.py`](main.py) is an ordinary Anthropic tool-use loop. Caching is one
`cache_control` breakpoint on the system prompt, which covers the tool
definitions and the system prompt because tools come first in a request. Each
response is priced with one call:

```python
result = cost_from_response("anthropic", response, catalog=catalog)
```

It reads the model and usage from the response, including cache writes and
reads, and returns the cost, its breakdown and the catalog price it used. The
same call prices OpenAI, Gemini and Bedrock Converse responses; see the
[`metergraph-core` README](https://github.com/metergraph/metergraph/tree/main/core#costing-a-provider-response).
