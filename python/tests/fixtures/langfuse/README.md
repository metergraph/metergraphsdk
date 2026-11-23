# Langfuse conformance fixtures

Ground truth for the Langfuse dialect of `MetergraphGenAIExporter`. Content is
synthetic; no credentials, project ids or real prompts are present.

| File | What it is |
|---|---|
| `otel_spans_python_v5.json` | The OpenTelemetry spans the Langfuse Python SDK 5.2.0 emits for the conformance flows, captured offline by `capture_spans.py`. Manual generations for seven model shapes, an embedding under agent, chain, retriever, tool and span parents, the LangChain `CallbackHandler`, and `@observe` with `langfuse.openai`. |
| `conformance_manifest.json` | What the same flows sent to a Langfuse test project, and what Langfuse reported back (`langfuse_readback`: `totalCost`, `latency`, `timeToFirstToken`, `usageDetails` per observation). Ids are stable hashes and timestamps start at 2026-01-01. |
| `capture_spans.py` | Regenerates the span file. It needs no network and no keys. |

Span names map to manifest keys: `gen-<key>` is `observations.<key>`,
`embed-query` is `embedding`, `ChatOpenAI` is `langchain_generation` and
`OpenAI-generation` is `observe_generation`.

`tests/integrations/instrumentation/test_langfuse_conformance.py` replays the
spans through the exporter and checks every row against the manifest. When a
Langfuse release changes what it emits, recapture the span file with that
version pinned and review the diff.
