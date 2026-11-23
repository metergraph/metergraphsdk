"""Replay real Langfuse SDK spans through the exporter and check every row
against what Langfuse itself recorded for the same flows.

The spans come from ``fixtures/langfuse/otel_spans_python_v5.json`` (Langfuse
Python SDK 5.2.0, captured offline) and the expected values from
``fixtures/langfuse/conformance_manifest.json`` (what the flows sent to a
Langfuse project and what Langfuse reported back). Neither is hand-written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, Status, StatusCode, TraceFlags

import metergraph
from metergraph import _capture
from metergraph._capture import Options, Runtime
from metergraph.opentelemetry import MetergraphGenAIExporter

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "langfuse"
MANIFEST = json.loads((FIXTURES / "conformance_manifest.json").read_text())
SPANS = json.loads((FIXTURES / "otel_spans_python_v5.json").read_text())

# Span name -> manifest observation key, for spans not named "gen-<key>".
_SPAN_KEYS = {
    "embed-query": "embedding",
    "ChatOpenAI": "langchain_generation",
    "OpenAI-generation": "observe_generation",
    "agent-root": "agent_root",
    "plan-chain": "plan_chain",
    "kb-retriever": "kb_retriever",
    "lookup-tool": "lookup_tool",
    "llm-step": "llm_step",
    "conformance-langchain": "langchain_chain",
    "observe-decorated": "observe_span",
}
# Langfuse usage-detail spellings per bucket, as its integrations write them.
_CACHE_READ = ("input_cached_tokens", "input_cache_read", "cache_read_input_tokens")
_CACHE_WRITE = ("cache_creation_input_tokens", "input_cache_creation")
_REASONING = ("output_reasoning_tokens", "output_reasoning")
_INPUT_INCLUDES_CACHE_READ = {"openai", "google", "deepseek", "xai"}
# Which manifest trace each span's trace-level attributes belong to.
_TRACE_OF_STYLE = {"manual": "main", "langchain": "langchain", "observe+langfuse.openai": "observe"}


class _Rows:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def enqueue(self, row: dict) -> bool:
        self.rows.append(row)
        return True


def _key(name: str) -> str:
    return _SPAN_KEYS.get(name) or name.removeprefix("gen-")


def _context(trace_id: str, span_id: str) -> SpanContext:
    return SpanContext(
        trace_id=int(trace_id, 16),
        span_id=int(span_id, 16),
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


def _readable(record: dict) -> ReadableSpan:
    status = record["status"]
    return ReadableSpan(
        name=record["name"],
        context=_context(record["trace_id"], record["span_id"]),
        parent=(
            _context(record["trace_id"], record["parent_span_id"])
            if record["parent_span_id"]
            else None
        ),
        resource=Resource.create(record["resource"]),
        attributes=record["attributes"],
        status=Status(StatusCode[status["code"]], status["description"]),
        start_time=record["start_time_unix_nano"],
        end_time=record["end_time_unix_nano"],
        instrumentation_scope=InstrumentationScope(record["scope"]),
    )


@pytest.fixture(scope="module")
def replay():
    rows = _Rows()
    previous = _capture._get_runtime()
    _capture.set_runtime(Runtime(rows, Options(app_root="")))
    init = metergraph.init
    metergraph.init = lambda *args, **kwargs: None
    try:
        exporter = MetergraphGenAIExporter()
        exporter.export([_readable(record) for record in SPANS["spans"]])
    finally:
        metergraph.init = init
        _capture.set_runtime(previous)
    names = [
        record["name"]
        for record in SPANS["spans"]
        if MANIFEST["observations"][_key(record["name"])].get("cost_bearing")
    ]
    assert len(rows.rows) == len(names)
    return exporter, {_key(name): row for name, row in zip(names, rows.rows)}


def _cost_bearing_keys() -> list[str]:
    return [
        _key(record["name"])
        for record in SPANS["spans"]
        if MANIFEST["observations"][_key(record["name"])].get("cost_bearing")
    ]


def _bucket(usage: dict, keys: tuple[str, ...]) -> int | None:
    return next((usage[key] for key in keys if key in usage), None)


def test_every_cost_bearing_observation_becomes_one_row_and_structure_is_skipped(replay):
    exporter, rows = replay
    structure = [
        record
        for record in SPANS["spans"]
        if not MANIFEST["observations"][_key(record["name"])].get("cost_bearing")
    ]
    assert set(rows) == set(_cost_bearing_keys())
    assert "embedding" in rows
    assert exporter.skipped == {"ineligible-kind": len(structure)}


@pytest.mark.parametrize("key", _cost_bearing_keys())
def test_provider_is_resolved_never_the_telemetry_vendor(replay, key):
    _, rows = replay
    expected = MANIFEST["observations"][key]["expected_providers"]
    assert rows[key]["provider"] in expected


@pytest.mark.parametrize("key", _cost_bearing_keys())
def test_token_buckets_match_langfuse_readback(replay, key):
    """Langfuse's buckets are disjoint; the row restates them natively.

    Output counts reasoning, and input counts cache reads for providers whose
    native input does (OpenAI-family), so the catalog prices the same tokens
    Langfuse priced.
    """
    _, rows = replay
    row = rows[key]
    usage = MANIFEST["langfuse_readback"][key]["usageDetails"]
    cache_read = _bucket(usage, _CACHE_READ)
    cache_write = _bucket(usage, _CACHE_WRITE)
    reasoning = _bucket(usage, _REASONING)
    expected_input = usage.get("input", 0) + (
        cache_read or 0 if row["provider"] in _INPUT_INCLUDES_CACHE_READ else 0
    )

    assert row["input_tokens"] == expected_input
    assert row["cache_read_tokens"] == cache_read
    assert row["cache_write_tokens"] == cache_write
    if "output" in usage:
        assert row["output_tokens"] == usage["output"] + (reasoning or 0)
    else:
        assert row["output_tokens"] is None
    assert row["reasoning_tokens"] == reasoning


@pytest.mark.parametrize("key", _cost_bearing_keys())
def test_caller_supplied_langfuse_cost_is_reported(replay, key):
    _, rows = replay
    supplied = MANIFEST["observations"][key].get("cost_supplied")
    row = rows[key]
    if supplied is None:
        # Langfuse inferred this cost server-side; the span never carries it.
        assert row.get("cost_usd") is None
        return
    total = supplied["total"]
    assert total == pytest.approx(MANIFEST["langfuse_readback"][key]["totalCost"])
    assert row["cost_usd"] == pytest.approx(total)
    assert row["reported_cost_usd"] == pytest.approx(total)
    assert row["reported_cost_source"] == "langfuse.observation.cost_details.total"


@pytest.mark.parametrize("key", _cost_bearing_keys())
def test_time_to_first_token_is_kept_when_langfuse_has_one(replay, key):
    _, rows = replay
    readback = MANIFEST["langfuse_readback"][key]["timeToFirstToken"]
    if readback is None:
        assert rows[key]["ttft_ms"] is None
    else:
        assert rows[key]["ttft_ms"] > 0


def test_error_keeps_langfuse_status_message_and_type(replay):
    _, rows = replay
    expected = MANIFEST["observations"]["error_call"]["status_message"]
    row = rows["error_call"]
    assert row["error"] is True
    assert row["error_type"] == expected.split(":", 1)[0]
    assert expected in row["response_text"]
    for key, other in rows.items():
        if key != "error_call":
            assert other["error"] is False, key


@pytest.mark.parametrize("key", _cost_bearing_keys())
def test_trace_attribution_lands_on_the_row(replay, key):
    _, rows = replay
    row = rows[key]
    observation = MANIFEST["observations"][key]
    trace = MANIFEST["traces"][_TRACE_OF_STYLE[observation.get("style", "manual")]]
    tags = row["tags"]

    assert row["trace_name"] == trace["name"]
    assert row.get("user_id") == trace.get("user_id")
    # The manifest stores the session id hashed; the span carries the raw one.
    assert (row["session_id"] is not None) == ("session_id" in trace)
    assert row["environment"] == MANIFEST["environment"]
    assert tags["langfuse.release"] == MANIFEST["release"]
    # The fixture's run tag replaces the seeded one, so compare the others.
    for tag in trace["tags"]:
        if tag.startswith("mg-conformance-"):
            continue
        assert tags[f"langfuse.tag.{tag}"] == "true"
    assert "langfuse.tag.mg-conformance-fixture" in tags
    if "version" in trace:
        assert tags["langfuse.version"] == trace["version"]
    for meta_key, value in (trace.get("metadata") or {}).items():
        assert tags[f"langfuse.metadata.{meta_key}"] == value
    for meta_key, value in (observation.get("metadata") or {}).items():
        text = tags[f"langfuse.metadata.{meta_key}"]
        # Strings land as themselves; structures as compact JSON.
        assert (text if isinstance(value, str) else json.loads(text)) == value


@pytest.mark.parametrize("key", _cost_bearing_keys())
def test_route_is_the_prompt_or_the_trace_never_the_operation(replay, key):
    _, rows = replay
    row = rows[key]
    prompt = MANIFEST["observations"][key].get("prompt")
    if prompt:
        assert row["route"] == prompt["name"]
        assert row["route_source"] == "explicit"
        assert row["tags"]["langfuse.prompt.name"] == prompt["name"]
        assert row["tags"]["langfuse.prompt.version"] == str(prompt["version"])
    else:
        assert row["route"] == row["trace_name"]
        assert row["route_source"] == "derived"
    assert row["route"] != "inference"


def test_model_parameters_are_kept_on_the_request(replay):
    _, rows = replay
    request = json.loads(rows["o4_mini"]["request_json"])
    assert request["parameters"] == MANIFEST["observations"]["o4_mini"]["model_parameters"]


def test_embedding_is_an_embeddings_call(replay):
    _, rows = replay
    assert rows["embedding"]["endpoint"] == "embeddings"
    assert rows["embedding"]["model"] == MANIFEST["observations"]["embedding"]["model"]
