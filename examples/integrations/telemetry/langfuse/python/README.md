# Langfuse SDK live capture

Use this integration when your application already traces LLM calls with the
[Langfuse Python SDK](https://langfuse.com/docs/sdk/python) (v3, v4 or v5, all
OpenTelemetry-based) and you want MeterGraph to receive the same completed
generation spans. MeterGraph is added as one more span processor on the
tracer provider Langfuse uses; existing `@observe` decorators and Langfuse
call sites stay unchanged.

**One capture path per call.** A call captured through this exporter must not
also be wrapped with `metergraph.wrap()` (or Langfuse's own OpenAI wrapper on
top of a MeterGraph-wrapped client), or it is counted twice. Pick one path
per client.

## Install

```bash
python -m pip install 'metergraph[otel]' langfuse
```

Configure the normal MeterGraph environment:

```bash
export METERGRAPH_APP_TOKEN=<your-metergraph-token>
export METERGRAPH_INGEST_URL=<your-metergraph-ingest-url>
export METERGRAPH_REPOSITORY=owner/repository
```

## Real-world setup

The exporter must be on the tracer provider that carries Langfuse's spans.
The dependable way, on every supported Langfuse version, is to build that
provider yourself and hand it to Langfuse:

```python
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from metergraph.opentelemetry import MetergraphGenAIExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(MetergraphGenAIExporter()))
langfuse = Langfuse(tracer_provider=provider)  # before any other Langfuse(...)
```

Langfuse keeps exporting to Langfuse as before: it adds its own processor to
the provider you pass. Build this client before anything else calls
`Langfuse(...)` or `get_client()`, because the client is a process-wide
singleton bound to the provider it was first built with. `@observe`,
`langfuse.openai` and the LangChain `CallbackHandler` all use that client, so
they need no further change.

If your application already registers a global provider that Langfuse uses,
adding the processor there works too:

```python
from opentelemetry import trace

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(MetergraphGenAIExporter())
)
```

Make one traced call and confirm a row arrives. If nothing does, the exporter
is on a different provider from Langfuse's spans.

With Langfuse v3 and an explicit provider, the release (`LANGFUSE_RELEASE`)
does not reach the spans, so it is not on the rows either; Langfuse records
none in that setup. v4 and v5 put it on every span.

### Which observations become rows

Generations and embeddings become rows, and so does any other observation
type that carries a model and usage or cost. Agent, tool, chain, retriever,
span and event observations without them are skipped and counted in
`exporter.skipped["ineligible-kind"]`. To restrict the exporter to Langfuse's
spans on a shared provider, pass `include_scopes=["langfuse-sdk"]` (matched
against `span.instrumentation_scope.name`; `exclude_scopes` wins).

## Run the offline demo

```bash
python main.py
```

[`main.py`](main.py) needs no Langfuse keys, no provider key, and no network:
it builds a plain `TracerProvider`, attaches `MetergraphGenAIExporter`, and
emits the same `langfuse.observation.*`-attribute spans the Langfuse SDK
records for one generation. Open it and search for `MeterGraph integration`:
the only integration block is the `add_span_processor` call; everything else
is synthetic demo traffic.

## What MeterGraph captures

Each row keeps what Langfuse recorded for the call:

| Langfuse | Row |
|---|---|
| Model name | `model`. The provider comes from LangChain's `ls_provider`, else from the model name (`gpt-*`, `claude-*`, Bedrock ids such as `us.anthropic.*`, gateway prefixes such as `openai/`, Gemini, Mistral and others), else `unknown`. |
| `usage_details` | Input, output, cache read, cache write (with 5-minute and 1-hour splits) and reasoning tokens. Langfuse counts these as separate buckets; the row restates them in the provider's own convention, so cached input is billed at the cache rate. `langfuse.openai`'s OpenAI usage object is read as is. |
| `cost_details.total` (when you supply it) | `cost_usd` and `reported_cost_usd`, kept as reported evidence beside MeterGraph's own catalog price. Costs Langfuse infers on its server are not on the span. |
| Completion start time | `ttft_ms` |
| Level `ERROR` and status message | `error`, the message in the response record, and `error_type` from the message's leading exception name (for example `RateLimitError`) |
| Trace name | `trace_name`, and the route (else the observation name; a `metergraph.route` attribute overrides) |
| Prompt name and version | Tags `langfuse.prompt.name` / `langfuse.prompt.version` (never the route) |
| User, session, environment | `user_id`, `session_id`, `environment` |
| Tags | One tag per Langfuse tag: `langfuse.tag.<tag>` = `true` |
| Release, version | Tags `langfuse.release`, `langfuse.version` |
| Trace and observation metadata | Tags `langfuse.metadata.<key>` (observation wins over trace). Structured values are compact JSON. |
| Model parameters, input, output | The request and response records |

Tags follow MeterGraph's usual bounds: at most 32 per row, kept in key order,
and keys that look like credentials are dropped. Message content is subject to
the SDK's normal scrub, redact and truncate policy; set
`METERGRAPH_CAPTURE_TEXT=0` for metadata-only capture.

## TypeScript (Langfuse JS v4 and newer)

Langfuse's JavaScript SDK is OpenTelemetry-based too, so a Node application
adds a second OTLP exporter that sends the same spans to MeterGraph's
`/v1/traces` endpoint. No MeterGraph package is needed; MeterGraph's server
reads the `langfuse.*` attributes the same way this exporter does.

```ts
import { NodeSDK } from "@opentelemetry/sdk-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { LangfuseSpanProcessor } from "@langfuse/otel";

const sdk = new NodeSDK({
  spanProcessors: [
    new LangfuseSpanProcessor(), // existing export to Langfuse
    new BatchSpanProcessor(
      new OTLPTraceExporter({
        url: "https://<your-metergraph-host>/v1/traces",
        headers: { Authorization: `Bearer ${process.env.METERGRAPH_APP_TOKEN}` },
      }),
    ),
  ],
});
sdk.start();
```

Use an ingest-scoped key and keep it on the server: never ship it to a
browser. Spans that are not model calls are counted and dropped by the
endpoint, so the second exporter needs no filter.

