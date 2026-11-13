def test_capture_emits_native_evidence_without_content_or_extra_calls():
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
        assert 'availability' not in row and 'refused' not in row
        assert row['otel']['finish_reasons'] == ['content_filter']
        assert row['refusal_signal'] is None
        assert row['response_received'] is True
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
        assert rows.rows[0]['refusal_signal'] is True
        assert rows.rows[0]['otel']['finish_reasons'] == ['stop']
        assert 'availability' not in rows.rows[0] and 'refused' not in rows.rows[0]
        assert rows.rows[0]['input_tokens'] == 3
    finally:
        _capture.set_runtime(None)


def test_scoped_refusal_hint_is_inherited_and_native_positive_wins():
    import metergraph
    from metergraph._context import snapshot
    from metergraph._model_evidence import model_evidence
    with metergraph.context(refusal_signal=False):
        assert snapshot().refusal_signal is False
        assert model_evidence(response={'choices': [{'message': {'refusal': 'policy'}, 'finish_reason': 'stop'}]}, refusal_signal=snapshot().refusal_signal)['refusal_signal'] is True
        with metergraph.context(refusal_signal=True):
            assert snapshot().refusal_signal is True
        assert snapshot().refusal_signal is False
    assert snapshot().refusal_signal is None


def test_evidence_preserves_raw_multi_choice_and_prompt_block_facts():
    from metergraph._model_evidence import model_evidence
    evidence = model_evidence(response={'choices': [
        {'finish_reason': 'stop', 'message': {'content': 'private content'}},
        {'finish_reason': 'content_filter', 'message': {}}]},
        error={'status_code': 400, 'body': {'error': {'code': 'structured_code', 'message': 'private error'}}})
    assert evidence == {'http_status_code': 400, 'error_code': 'structured_code',
        'refusal_signal': None, 'response_received': True,
        'otel': {'finish_reasons': ['stop', 'content_filter']}}
    assert model_evidence(provider='google', error={'code': 429, 'status': 'RESOURCE_EXHAUSTED'})['http_status_code'] == 429
    assert model_evidence(response={'promptFeedback': {'blockReason': 'SAFETY'}})['otel']['finish_reasons'] == ['SAFETY']
    assert model_evidence(response={'choices': [{'finish_reason': 'stop', 'message': {'content': 'ok'}}]})['refusal_signal'] is False
    assert model_evidence(stream_chunks=[{'choices': [{'finish_reason': 'stop', 'delta': {}}]}], stream_complete=True)['refusal_signal'] is None
