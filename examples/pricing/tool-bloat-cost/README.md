# Tool bloat and prompt caching

Agents often carry more tools than a given task needs. Does a long tool list
cost money on every call, and does prompt caching take that cost away? This
example measures it instead of guessing.

It runs one small agent task ("What is the temperature in Lisbon, in
Fahrenheit?") four ways: with only the 2 tools it needs, or with 15 more it
never calls, each with prompt caching off and on. Each scenario runs 3 times.
Every response is priced in-process with
[`metergraph-core`](https://pypi.org/project/metergraph-core/)'s
`cost_from_response`, and the totals are printed side by side.

## Result

Recorded on Claude Opus 5.5 at effort `low` (`recorded_run.json`), priced with
the catalog in metergraph-core 0.7.1:

```text
scenario               calls   input  cache write  cache read  output        cost
2 tools, no caching        9    5487            0           0     408   $0.030108
2 tools, caching           9    5487            0           0     408   $0.030108
17 tools, no caching       9   24495            0           0     408   $0.106140
17 tools, caching          9    1464         2559       20472     408   $0.030905
15 unused tools, no caching: $0.076032 extra (72% of the bloated total)
15 unused tools, caching: $0.000797 extra (3% of the bloated total)
```

What it shows:

- **Without caching, unused tools are paid for on every call.** The 15 extra
  tool definitions add about 2,100 input tokens to each request, and an agent
  sends the whole tool list again on every turn. Here that is 72% of the
  bloated run's cost.
- **With caching, they cost little, but not nothing.** The tool list is written
  to the cache once, at 1.25 times the input price, then read on every later
  call at a twentieth of it on Claude Opus 5.5 (a tenth on most other models).
  The bloated run with caching costs about the same as the lean run without
  it.
- **A short prompt may not cache at all.** The 2-tool prompt is under the
  model's minimum cacheable length, so `cache_control` changed nothing: no
  cache write, no cache read, the same cost. The minimum depends on the model,
  and the API does not return an error, so check `cache_read_input_tokens`.
- **Caching only helps while the tool list stays the same.** Tools come first in
  the cached prefix, so adding, removing or reordering a tool starts the cache
  over. Trimming the list per request and caching it work against each other.

Your figures depend on the model, your tools, how many turns a task takes, and
how often the cache goes cold: the default cache lives 5 minutes after its last
use.

## What the example adds

Open [`main.py`](main.py). The agent loop is ordinary Anthropic tool-use code.
The pricing is one call per response:

```python
result = cost_from_response("anthropic", response, catalog=catalog)
```

It reads the model and usage from the response, including cache writes and
reads, and returns the cost, its breakdown and the catalog price row it used.
The same call prices OpenAI, Gemini and Bedrock Converse responses; see the
[`metergraph-core` README](https://github.com/metergraph/metergraph/tree/main/core#costing-a-provider-response).

Caching is one breakpoint after the system prompt, which covers the tool
definitions and the system prompt because tools come first in a request.

## Run it

```bash
pip install metergraph-core anthropic
python examples/pricing/tool-bloat-cost/main.py
```

Without `ANTHROPIC_API_KEY` it replays `recorded_run.json` and prices it with
your installed catalog, so it runs offline and costs nothing.

With `ANTHROPIC_API_KEY` set it makes the calls itself: 4 scenarios times 3
runs, about 36 requests. On Claude Opus 5.5 the recorded run cost about $0.20.
`--model` picks another model, and `--record` saves the run over
`recorded_run.json`. On a model with a higher minimum cacheable length, such as
Claude Haiku 4.5 (4,096 tokens), even the 17-tool prompt may not cache.
