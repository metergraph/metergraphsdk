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
