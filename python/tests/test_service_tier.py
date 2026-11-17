"""Requested delivery and provider-confirmed billing tier stay separate."""
from types import SimpleNamespace

import pytest

from metergraph._capture import Options, Runtime


class Rows:
    def __init__(self):
        self.rows = []

    def enqueue(self, row):
        self.rows.append(row)
        return True


def finish_row(tmp_path, provider, endpoint, request, response, **extra):
    rows = Rows()
    runtime = Runtime(rows, Options(capture_text=False, app_root=str(tmp_path)))
    call = runtime.call_state(provider, endpoint, {"model": "example-model", **request})
    call.finish(response, **extra)
    return rows.rows[0]


@pytest.mark.parametrize("provider,endpoint,params,response,served,requested", [
    ("openai", "responses.create", {"service_tier": "flex"}, {"service_tier": "default"}, "default", "flex"),
    ("openai", "responses.create", {"service_tier": "flex"}, {}, None, "flex"),
    ("anthropic", "messages.create", {"service_tier": "auto"}, {"usage": {"service_tier": "standard"}}, "standard", "auto"),
    ("google", "models.generate_content", {"config": {"http_options": {"headers": {"X-Vertex-AI-LLM-Shared-Request-Type": "flex"}}}}, {"usage_metadata": {"traffic_type": "ON_DEMAND_FLEX"}}, "flex", "flex"),
    ("google", "models.generate_content", {"config": {"service_tier": "flex"}}, {"usage_metadata": {"traffic_type": "ON_DEMAND"}}, "standard", "flex"),
    ("google", "models.generate_content", {"config": {"serviceTier": "flex"}}, {"usageMetadata": {"trafficType": "ON_DEMAND_FLEX"}}, "flex", "flex"),
    ("google", "models.generate_content", {"config": {"service_tier": "flex"}}, {"usage_metadata": {"service_tier": "flex"}}, "flex", "flex"),
    ("openai", "batch.responses", {"batch": True}, {"service_tier": "default"}, "batch", None),
    ("anthropic", "batch.messages", {"batch": True}, {"usage": {"service_tier": "standard"}}, "batch", None),
    ("anthropic", "messages.create", {"service_tier": "batch"}, {}, None, "batch"),
])
def test_provider_tiers(tmp_path, provider, endpoint, params, response, served, requested):
    row = finish_row(tmp_path, provider, endpoint, params, response)
    assert row["service_tier"] == served
    assert row["requested_service_tier"] == requested
    assert row["content_opted_in"] is False


@pytest.mark.parametrize("provider,endpoint,chunks,expected", [
    ("openai", "chat.completions.create", [{"service_tier": "flex"}, {"usage": {"prompt_tokens": 1}}], "flex"),
    ("openai", "responses.create", [{"response": {"service_tier": "flex"}}, {"response": {"service_tier": "default"}}, {"type": "response.done"}], "default"),
    ("anthropic", "messages.create", [{"type": "message_start", "message": {"usage": {"service_tier": "standard"}}}, {"type": "message_stop"}], "standard"),
    ("google", "models.generate_content.stream", [{"usage_metadata": {"traffic_type": "ON_DEMAND_FLEX"}}, {"candidates": []}], "flex"),
])
def test_stream_tier_survives_trailing_event(tmp_path, provider, endpoint, chunks, expected):
    row = finish_row(tmp_path, provider, endpoint, {}, chunks[-1], stream=True, stream_chunks=chunks)
    assert row["service_tier"] == expected


def test_native_objects_and_enum_values(tmp_path):
    from enum import Enum
    class TrafficType(str, Enum):
        ON_DEMAND_FLEX = "ON_DEMAND_FLEX"
    row = finish_row(tmp_path, "google", "models.generate_content", {
        "config": SimpleNamespace(http_options=SimpleNamespace(headers={"x-vertex-ai-llm-shared-request-type": "flex"})),
    }, SimpleNamespace(usage_metadata=SimpleNamespace(traffic_type=TrafficType.ON_DEMAND_FLEX)))
    assert row["service_tier"] == row["requested_service_tier"] == "flex"


def test_malformed_tiers_and_failed_batch_are_unknown(tmp_path):
    row = finish_row(tmp_path, "openai", "responses.create", {"service_tier": ""}, {"service_tier": {"unexpected": True}})
    assert row["service_tier"] is None
    assert row["requested_service_tier"] is None
    row = finish_row(tmp_path, "anthropic", "batch.messages", {"batch": True, "service_tier": "batch"}, {}, status="error")
    assert row["service_tier"] is None


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_batch_capture_does_not_invent_requested_tier(tmp_path, monkeypatch, provider):
    import json
    from metergraph import _capture
    rows = Rows()
    monkeypatch.setattr(_capture, "_runtime", Runtime(rows, Options(app_root=str(tmp_path))))
    message = {"model": "example-model", "service_tier": "default", "usage": {"service_tier": "standard"}}
    if provider == "anthropic":
        item = {"custom_id": "tier-regression", "result": {"type": "succeeded", "message": message}}
        client = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(results=lambda batch_id: [item])))
        _capture.wrap(client, provider=provider)
        assert list(client.messages.batches.results("batch-tier-regression")) == [item]
    else:
        item = {"id": "tier-regression", "custom_id": "tier-regression", "response": {"status_code": 200, "body": {"object": "response", **message}}}
        client = SimpleNamespace(files=SimpleNamespace(content=lambda file_id: SimpleNamespace(content=json.dumps(item).encode())))
        _capture.wrap(client, provider=provider)
        assert json.loads(client.files.content("file-tier-regression").content) == item
    assert len(rows.rows) == 1
    assert rows.rows[0]["service_tier"] == "batch"
    assert rows.rows[0]["requested_service_tier"] is None
