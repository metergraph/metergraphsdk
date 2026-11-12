import json
from dataclasses import asdict
from pathlib import Path

import pytest
from metergraph._reliability import classify_model_reliability

CASES = json.loads((Path(__file__).parents[2] / "tests/fixtures/reliability.json").read_text())

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_reliability_conformance(case):
    assert asdict(classify_model_reliability(**case["input"])) == case["expected"]


def test_capture_emits_outcome_without_content_or_extra_calls():
    from types import SimpleNamespace
    from metergraph import _capture
    import metergraph

    class Rows:
        def __init__(self): self.rows = []
        def enqueue(self, row): self.rows.append(row); return True
    rows = Rows()
    _capture.set_runtime(_capture.Runtime(rows, _capture.Options(capture_text=False)))
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='content_filter', message=SimpleNamespace(content=None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    try:
        metergraph.wrap(client, provider='openai')
        assert client.chat.completions.create(model='example-model', messages=[]) is response
        assert len(rows.rows) == 1
        row = rows.rows[0]
        assert row['availability'] == 'available'
        assert row['refused'] is True
        assert row['response_text'] is None
    finally:
        _capture.set_runtime(None)


def test_stream_refusal_survives_terminal_usage_chunk():
    from types import SimpleNamespace
    from metergraph import _capture
    import metergraph

    class Rows:
        def __init__(self): self.rows = []
        def enqueue(self, row): self.rows.append(row); return True
    rows = Rows()
    _capture.set_runtime(_capture.Runtime(rows, _capture.Options(capture_text=False)))
    chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(refusal='policy'), finish_reason=None)]),
              SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(), finish_reason='stop')]),
              SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1))]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: iter(chunks))))
    try:
        metergraph.wrap(client, provider='openai')
        list(client.chat.completions.create(model='example-model', messages=[], stream=True))
        assert len(rows.rows) == 1
        assert rows.rows[0]['refused'] is True
        assert rows.rows[0]['availability'] == 'available'
        assert rows.rows[0]['input_tokens'] == 3
    finally:
        _capture.set_runtime(None)
