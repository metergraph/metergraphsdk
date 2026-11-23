"""Regenerate otel_spans_python_v5.json: the OpenTelemetry spans the Langfuse
Python SDK emits for the conformance flows, captured offline.

The flows match conformance_manifest.json: manual generations for seven model
shapes, an embedding under agent/chain/retriever/tool/span parents, the
LangChain CallbackHandler, and @observe with langfuse.openai. OpenAI calls go
to an in-process mock transport and Langfuse exports nowhere, so no network,
credentials or provider spend are involved. Ids are replaced by stable
synthetic ones and timestamps start at 2026-01-01T00:00:00Z.

    pip install "langfuse==5.2.0" "openai==3.57.0" "langchain-openai==1.8.5" httpx
    python capture_spans.py otel_spans_python_v5.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import sys
import time
from pathlib import Path

os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "conformance"
os.environ["LANGFUSE_RELEASE"] = "rel-2026.11.22"

import httpx  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

TAG = "mg-conformance-fixture"
VERSION = "app-v1.4.2"
GENERATIONS = [
    ("openai_cached_reasoning", "gpt-4o-mini",
     {"input": 900, "input_cached_tokens": 600, "output": 120, "output_reasoning_tokens": 40}, None, {}),
    ("anthropic_cache_rw", "claude-3-5-haiku-20241022",
     {"input": 200, "cache_read_input_tokens": 800, "cache_creation_input_tokens": 300, "output": 90},
     {"input": 0.00016, "cache_read_input_tokens": 0.000064, "cache_creation_input_tokens": 0.0003,
      "output": 0.00036, "total": 0.000884}, {}),
    ("o4_mini", "o4-mini", {"input": 500, "output": 300, "output_reasoning_tokens": 200}, None, {}),
    ("bedrock_claude", "us.anthropic.claude-3-5-haiku-20241022-v1:0", {"input": 400, "output": 50}, None, {}),
    ("gateway_openai", "openai/gpt-4o", {"input": 300, "output": 60}, {"total": 0.00135}, {}),
    ("unknown_mistral", "mistral-large-latest", {"input": 250, "output": 40},
     {"input": 0.0005, "output": 0.00024, "total": 0.00074}, {}),
    ("error_call", "gpt-4o-mini", {"input": 100, "output": 0}, None,
     {"level": "ERROR", "status_message": "RateLimitError: 429 from upstream"}),
]
MODEL_PARAMETERS = {"temperature": 0.2, "max_tokens": 512}


class _Local(SpanExporter):
    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis=30_000):
        return True


def mock_http():
    def handler(request):
        body = {
            "id": "chatcmpl-mock", "object": "chat.completion", "created": 1767225600,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "mocked"}}],
            "usage": {"prompt_tokens": 700, "completion_tokens": 30, "total_tokens": 730,
                      "prompt_tokens_details": {"cached_tokens": 512},
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        }
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def capture() -> dict:
    """Run the flows against the installed Langfuse SDK and return the payload."""
    from langfuse import Langfuse, observe, propagate_attributes
    from langfuse.api import Prompt_Text
    from langfuse.model import TextPromptClient

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    import inspect

    options = {"public_key": "pk-lf-synthetic", "secret_key": "sk-lf-synthetic",
               "host": "http://127.0.0.1:9", "tracer_provider": provider}
    if "span_exporter" in inspect.signature(Langfuse.__init__).parameters:
        options["span_exporter"] = _Local()  # v4+; v3 retries the dead loopback host and gives up
    lf = Langfuse(**options)
    prompt = TextPromptClient(Prompt_Text(name="mg-conformance-prompt", version=1, prompt="Classify: {{text}}",
                                          config={}, labels=["production"], tags=[], type="text"))
    labels: dict[str, str] = {}
    common = {"user_id": "user-conf-1", "session_id": "sess-" + TAG, "tags": sorted([TAG, "prod"]),
              "metadata": {"feature": "support-triage", "customer_tier": "gold"},
              "version": VERSION, "trace_name": "conformance-workflow"}
    with propagate_attributes(**common):
        with lf.start_as_current_observation(name="agent-root", as_type="agent", input={"q": "hi"}) as root:
            with lf.start_as_current_observation(name="plan-chain", as_type="chain"):
                with lf.start_as_current_observation(name="kb-retriever", as_type="retriever"):
                    with lf.start_as_current_observation(
                        name="embed-query", as_type="embedding", model="text-embedding-3-small", input="hi",
                        usage_details={"input": 1200}, cost_details={"input": 0.000024, "total": 0.000024}):
                        pass
                with lf.start_as_current_observation(name="lookup-tool", as_type="tool", input={"id": 7},
                                                     output={"ok": True}):
                    pass
                with lf.start_as_current_observation(name="llm-step", as_type="span"):
                    for key, model, usage, cost, extra in GENERATIONS:
                        with lf.start_as_current_observation(
                            name=f"gen-{key}", as_type="generation", model=model,
                            input=[{"role": "system", "content": "You are terse."},
                                   {"role": "user", "content": f"case {key}"}],
                            model_parameters=MODEL_PARAMETERS,
                            prompt=prompt if key == "openai_cached_reasoning" else None,
                            metadata={"step": key, "nested": {"region": "us-west-2", "attempt": 1}},
                            **extra,
                        ) as gen:
                            time.sleep(0.05)
                            ttft = dt.datetime.now(dt.timezone.utc)
                            time.sleep(0.05)
                            gen.update(output=None if extra else "ok", usage_details=usage,
                                       cost_details=cost, completion_start_time=ttft)
            root.update(output={"a": "done"})

    # LangChain handler before langfuse.openai patches the OpenAI client.
    from langchain_openai import ChatOpenAI
    from langfuse.langchain import CallbackHandler

    handler = CallbackHandler()
    llm = ChatOpenAI(model="gpt-4o-mini", api_key="sk-mock", http_client=mock_http())
    with propagate_attributes(tags=[TAG, "langchain"], trace_name="conformance-langchain"):
        with lf.start_as_current_observation(name="conformance-langchain", as_type="chain"):
            llm.invoke("hello", config={"callbacks": [handler]})

    @observe(name="observe-decorated")
    def decorated():
        from langfuse.openai import OpenAI

        client = OpenAI(api_key="sk-mock", http_client=mock_http())
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "wrapped"}])
        return "ok"

    with propagate_attributes(tags=[TAG, "observe"], trace_name="conformance-observe", user_id="user-conf-2"):
        decorated()
    lf.flush()
    provider.force_flush()

    spans = sorted(memory.get_finished_spans(), key=lambda s: s.start_time)
    t0 = spans[0].start_time
    epoch = int(dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1_000_000_000)

    def fake(kind: str, value: int, width: int) -> str | None:
        if not value:
            return None
        return hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()[:width]

    def shift_iso(text: str) -> str:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        ns = int(parsed.timestamp() * 1_000_000_000) - t0 + epoch
        return dt.datetime.fromtimestamp(ns / 1e9, dt.timezone.utc).isoformat()

    # Deterministic, synthetic ids: rank by first appearance.
    ids: dict[tuple[str, int], str] = {}

    def sid(kind: str, value: int, width: int) -> str | None:
        if not value:
            return None
        key = (kind, value)
        if key not in ids:
            ids[key] = hashlib.sha256(f"{kind}-{len(ids)}".encode()).hexdigest()[:width]
        return ids[key]

    records = []
    for span in spans:
        attributes = {}
        for key, value in (span.attributes or {}).items():
            if isinstance(value, tuple):
                value = list(value)
            if key == "langfuse.observation.completion_start_time" and isinstance(value, str):
                decoded = json.loads(value)
                value = json.dumps(shift_iso(decoded))
            attributes[key] = value
        records.append({
            "name": span.name,
            "scope": span.instrumentation_scope.name,
            "kind": span.kind.name,
            "trace_id": sid("trace", span.context.trace_id, 32),
            "span_id": sid("span", span.context.span_id, 16),
            "parent_span_id": sid("span", span.parent.span_id, 16) if span.parent else None,
            "start_time_unix_nano": span.start_time - t0 + epoch,
            "end_time_unix_nano": span.end_time - t0 + epoch,
            "status": {"code": span.status.status_code.name, "description": span.status.description},
            "attributes": attributes,
            "resource": {k: v for k, v in span.resource.attributes.items()
                         if k in ("service.name", "telemetry.sdk.language", "telemetry.sdk.name",
                                  "langfuse.release", "langfuse.environment")},
        })
    payload = {
        "schema": "metergraph-langfuse-otel-spans/1",
        "source": "Langfuse Python SDK OpenTelemetry spans for the conformance fixture flows, captured offline",
        "versions": {p: md.version(p) for p in ("langfuse", "opentelemetry-sdk", "openai", "langchain-openai")},
        "spans": records,
    }
    text = json.dumps(payload)
    for forbidden in ("pk-lf-synthetic", "sk-lf-synthetic", "sk-mock"):
        assert forbidden not in text, forbidden
    return payload


def main(out: Path) -> None:
    payload = capture()
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {len(payload['spans'])} spans to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
