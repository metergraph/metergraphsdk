# Node + OpenRouter

Capture OpenRouter usage and billing evidence with an ordinary `openai` client.
[`main.mjs`](main.mjs) runs one non-streaming and one streaming Chat Completions
call. Comments mark each MeterGraph integration line against existing application
code.

## Setup and run

Running `npm start` makes two billable OpenRouter calls.
This folder is a self-contained package. `npm install` builds the local
`metergraph` package and installs `openai`.

```bash
cd examples/integrations/providers/openrouter/typescript
npm install
export OPENROUTER_API_KEY=sk-or-...            # your OpenRouter key
export METERGRAPH_APP_TOKEN=dev-token          # one of your server's MG_TOKENS
export METERGRAPH_INGEST_URL=http://localhost:8787   # your self-hosted server
npm start
```

Run the offline contract test (no credentials or network) with `npm test`.

Client instrumentation is one line:

```ts
const client = mg.wrap(new OpenAI({ apiKey, baseURL: "https://openrouter.ai/api/v1" }));
```

## Auto-detection and the custom-domain override

An OpenAI-compatible client whose base URL is exactly `https://openrouter.ai` is
detected automatically — no declaration needed. For a trusted custom domain or
reverse proxy, pass the override:

```ts
const client = mg.wrap(openaiClient, { gateway: "openrouter" });
```

The example picks the branch from `OPENROUTER_BASE_URL`, so setting it to a
custom host exercises the override. The existing provider-string form
(`mg.wrap(client, "openai")`) still works; an options object may pass a
consistent `{ provider: "openai", gateway: "openrouter" }`.

## Request attribution

Both calls share user `user-123`, session `session-123`, customer metadata
`customer_id=customer-123`, and one `gateway-comparison` trace. Replace these
synthetic IDs with your application's opaque IDs. The streaming call overrides
only `metadata.mode` to `streaming`; the normal call keeps `nonstreaming`.
Attribution is captured when the call starts, so the stream keeps its tags when
consumed outside that inner scope. Provider requests and billing capture are
unchanged. Use an SDK build containing user context support and compatible
ingestion. See the [focused attribution example](../../../../attribution/user-context/).

## What is captured

- **Requested vs served model.** The row keeps your requested `model` and adds
  `served_model` from the response. After routing or fallback they can differ.
- **Gateway-reported vs catalog cost.** `reported_cost_usd` comes from
  `usage.cost` (the OpenRouter account charge) with source
  `openrouter.usage.cost`; `reported_upstream_cost_usd` from
  `usage.cost_details.upstream_inference_cost`. These are gateway evidence, not a
  MeterGraph catalog price — the server keeps both. The SDK adds no pricing logic.
- **Final streaming usage.** OpenRouter sends the final usage event
  automatically. MeterGraph does not inject `stream_options` and does not hide or
  reorder any chunk, so your `for await` loop still sees the usage event; capture
  reads the cost from it once.

Malformed or absent evidence is omitted, never invented, and never affects the
native return value, exceptions, or stream.

## BYOK limitation

`reported_cost_usd` is the amount charged to the OpenRouter account. Under BYOK
the upstream provider may bill you separately, so this value can be less than the
total economic cost of the call. MeterGraph reports the documented account charge
and does not reconcile BYOK billing.
