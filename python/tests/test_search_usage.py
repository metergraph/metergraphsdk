from metergraph._capture import Options, Runtime


class Rows:
    def __init__(self):
        self.rows = []

    def enqueue(self, row):
        self.rows.append(row)
        return True


def search(id):
    return {"type": "web_search_call", "id": id, "status": "completed"}


def record(response, **extra):
    rows = Rows()
    runtime = Runtime(rows, Options(capture_text=False, text_max_bytes=32))
    state = runtime.call_state("openai", "responses.create", {
        "model": "example-model", "tools": [{"type": "web_search"}],
    })
    state.finish(response, **extra)
    return rows.rows[0]


def test_counts_executed_searches_without_content_or_declared_tools():
    row = record({"output": [search("ws_1"), {"type": "function_call"}, search("ws_2"), search("ws_3")]})
    assert row["web_search_calls"] == 3
    assert row["content_opted_in"] is False
    assert record({"output": []})["web_search_calls"] == 0
    assert record({"choices": []}).get("web_search_calls") is None


def test_stream_snapshots_and_repeated_items_are_not_double_counted():
    output = [search("ws_1"), search("ws_2")]
    chunks = [
        {"type": "response.output_item.done", "item": output[0], "output_index": 0},
        {"type": "response.output_item.done", "item": output[0], "output_index": 0},
        {"type": "response.completed", "response": {"output": output}},
        {"type": "response.done"},
    ]
    assert record(chunks[-1], stream=True, stream_chunks=chunks)["web_search_calls"] == 2
    assert record(chunks[1], stream=True, stream_chunks=chunks[:2])["web_search_calls"] == 1


def test_wrapped_native_openai_preserves_result_and_records_searches():
    import httpx
    from openai import OpenAI
    import metergraph
    from metergraph._capture import set_runtime

    rows = Rows()
    set_runtime(Runtime(rows, Options(capture_text=False)))
    body = {"id": "resp_search", "object": "response", "model": "example-model", "status": "completed",
            "output": [search("ws_1"), search("ws_2"), search("ws_3")],
            "usage": {"input_tokens": 100, "output_tokens": 20}}
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(200, json=body, headers={"x-request-id": "req_search"})

    try:
        with OpenAI(api_key="test", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(reply))) as client:
            result = metergraph.wrap(client).responses.create(model="example-model", input="Synthetic")
            assert result.id == body["id"]
            assert len(result.output) == 3
        assert len(calls) == len(rows.rows) == 1
        assert rows.rows[0]["request_id"] == "req_search"
        assert rows.rows[0]["web_search_calls"] == 3
    finally:
        set_runtime(None)


def provider_record(provider, response, chunks=None, request=None):
    rows = Rows()
    runtime = Runtime(rows, Options(capture_text=False))
    runtime.call_state(provider, "responses.create", {"model": "example-model", **(request or {})}).finish(response, stream_chunks=chunks)
    return rows.rows[0]


def test_anthropic_keeps_latest_explicit_usage_including_zero():
    chunks = [{"message": {"usage": {"server_tool_use": {"web_search_requests": 0}}}},
              {"usage": {"server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1}}}]
    assert provider_record("anthropic", {}, chunks)["server_tool_use"] == {"web_search_requests": 2, "web_fetch_requests": 1}
    assert provider_record("anthropic", {"usage": {"server_tool_use": {"web_search_requests": 0}}})["server_tool_use"] == {"web_search_requests": 0}


def test_google_deduplicates_nonempty_stream_grounding_queries():
    chunks = [{"candidates": [{"groundingMetadata": {"webSearchQueries": ["one", "", "one"]}}]},
              {"candidates": [{"grounding_metadata": {"web_search_queries": ["one", "two", " "]}}]}]
    row = provider_record("google", {}, chunks)
    assert row["grounding_queries"] == 2 and row["grounded_prompt"] == 1
    assert provider_record("google", {}).get("grounding_queries") is None


def test_perplexity_prefers_invocation_usage_and_preserves_api_and_type():
    response = {"output": [{"type": "search_results"}], "usage": {"tool_calls_details": {"search_web": {"invocation": 3}}}}
    row = provider_record("perplexity", response, request={"tools": [{"type": "web_search", "search_type": "fast"}]})
    assert row["web_search_calls"] == 3 and row["retrieval_api"] == "agent" and row["search_type"] == "fast"
    assert provider_record("perplexity", {"output": [{"type": "search_results"}, {"type": "search_results"}]})["web_search_calls"] == 2
    assert provider_record("perplexity", {"output": [], "usage": {"tool_calls_details": {"search_web": {"invocation": 0}}}})["web_search_calls"] == 0
    assert provider_record("perplexity", {}).get("web_search_calls") is None


def test_perplexity_openai_compatible_client_keeps_provider_identity():
    import httpx
    from openai import OpenAI
    import metergraph
    from metergraph._capture import set_runtime
    rows = Rows()
    set_runtime(Runtime(rows, Options(capture_text=False)))
    body = {"id": "resp_search", "object": "response", "model": "perplexity/sonar", "status": "completed", "output": [{"type": "search_results"}]}
    try:
        with OpenAI(api_key="test", base_url="https://api.perplexity.ai", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))) as client:
            metergraph.wrap(client).responses.create(model="perplexity/sonar", input="Synthetic")
        assert len(rows.rows) == 1 and rows.rows[0]["provider"] == "perplexity"
        assert rows.rows[0]["web_search_calls"] == 1
    finally:
        set_runtime(None)


def test_perplexity_resolves_bare_search_default_without_guessing_managed_configs():
    response = {"output": [{"type": "search_results"}]}

    def record(request):
        return provider_record("perplexity", response, request=request)

    assert record({"tools": [{"type": "web_search"}]})["search_type"] == "web"
    assert record({"tools": [{"type": "web_search"}], "reasoning": {"effort": "high"}})["search_type"] == "web"
    for extra in [{"effort": "low"}, {"profile": "example-profile"}]:
        assert record({"tools": [{"type": "web_search"}], **extra}).get("search_type") is None
        assert record({"tools": [{"type": "web_search", "search_type": "fast"}], **extra})["search_type"] == "fast"
    for tools in [None, [], [{"type": "web_search", "search_type": None}],
                  [{"type": "web_search", "search_type": "unknown"}],
                  [{"type": "web_search"}, {"type": "web_search", "search_type": "fast"}]]:
        assert record({"tools": tools}).get("search_type") is None
