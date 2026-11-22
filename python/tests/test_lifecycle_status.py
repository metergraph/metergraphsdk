import pytest
from metergraph._capture import Options, Runtime


class Rows:
    def __init__(self):
        self.rows = []

    def enqueue(self, row):
        self.rows.append(row)
        return True


def record(reason, **extra):
    rows = Rows()
    runtime = Runtime(rows, Options(capture_text=False, text_max_bytes=32))
    state = runtime.call_state('openai', 'chat.completions', {'model': 'example-model'})
    state.finish({'choices': [{'finish_reason': reason}],
                  'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}, **extra)
    return rows.rows[0]


@pytest.mark.parametrize('reason', ['', ' \t '])
def test_blank_finish_reason_uses_success_lifecycle_status(reason):
    row = record(reason)
    assert row['status'] == 'success'
    assert row['input_tokens'] == 10
    assert row['output_tokens'] == 2
    if reason.strip() != reason:
        assert row['finish_reason_raw'] == reason


@pytest.mark.parametrize('reason', ['', ' \t '])
def test_blank_finish_reason_preserves_errors_and_abandoned_streams(reason):
    failed = record(reason, error=RuntimeError('synthetic failure'), stream=True)
    assert failed['status'] == 'error'
    assert failed['status_code'] == 'error'
    assert failed['usage_complete'] is False
    abandoned = record(reason, status='abandoned', stream=True)
    assert abandoned['status'] == 'abandoned'
    assert abandoned['usage_complete'] is False


def test_nonblank_finish_reason_retains_existing_lifecycle_status():
    assert record('length')['status'] == 'length'
