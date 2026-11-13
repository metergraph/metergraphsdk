"""Extract content-free native facts; receivers own outcome classification."""
import json
from collections.abc import Mapping
from typing import Any


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _seq(value):
    return value if isinstance(value, (list, tuple)) else ()


def _text(value):
    return value if isinstance(value, str) and value.strip() and len(value) <= 256 else None


def _inspect(value):
    reasons = []
    positive = False
    coverage = []
    received = False
    def reason(raw):
        if _text(raw) is not None:
            reasons.append(raw)
    for key in ('finish_reason', 'stop_reason'):
        reason(_get(value, key))
    if _get(value, 'type') in {'refusal', 'response.refusal.delta', 'response.refusal.done'}:
        positive = True
    for part in (_get(value, 'part'), _get(value, 'content_block')):
        positive = positive or _get(part, 'type') == 'refusal'
    delta = _get(value, 'delta')
    reason(_get(delta, 'stop_reason'))
    positive = positive or bool(_get(delta, 'refusal'))
    choices = _get(value, 'choices')
    if isinstance(choices, (list, tuple)):
        received = True
        for choice in choices:
            finish = _get(choice, 'finish_reason')
            reason(finish)
            message = _get(choice, 'message')
            positive = (positive or bool(_get(message, 'refusal')) or bool(_get(_get(choice, 'delta'), 'refusal'))
                        or any(_get(part, 'type') == 'refusal' for part in _seq(_get(message, 'content'))))
            coverage.append(message is not None and _text(finish) is not None and (
                _get(message, 'content') is not None or _get(message, 'tool_calls') is not None
                or (isinstance(message, Mapping) and 'refusal' in message)
                or (not isinstance(message, Mapping) and hasattr(message, 'refusal'))))
    content = _get(value, 'content')
    if isinstance(content, (list, tuple)):
        positive = positive or any(_get(part, 'type') == 'refusal' for part in content)
        if _text(_get(value, 'stop_reason')):
            received = True
            coverage.append(bool(content) and all(_get(part, 'type') in {'text', 'thinking', 'redacted_thinking', 'tool_use', 'refusal'} for part in content))
    output = _get(value, 'output')
    if isinstance(output, (list, tuple)):
        received = True
        for item in output:
            parts = _seq(_get(item, 'content'))
            positive = positive or _get(item, 'type') == 'refusal' or any(_get(part, 'type') == 'refusal' for part in parts)
            coverage.append(_get(value, 'status') in {'completed', 'incomplete'} and (
                (_get(item, 'type') == 'message' and bool(parts) and all(_get(part, 'type') in {'output_text', 'refusal'} for part in parts))
                or _get(item, 'type') in {'function_call', 'reasoning'}))
    reason(_get(_get(value, 'incomplete_details'), 'reason'))
    feedback = _get(value, 'promptFeedback', _get(value, 'prompt_feedback'))
    if feedback is not None:
        received = True
        reason(_get(feedback, 'blockReason', _get(feedback, 'block_reason')))
    candidates = _get(value, 'candidates')
    if isinstance(candidates, (list, tuple)):
        received = True
        for candidate in candidates:
            finish = _get(candidate, 'finishReason', _get(candidate, 'finish_reason'))
            reason(finish)
            parts = _seq(_get(_get(candidate, 'content'), 'parts'))
            positive = positive or any(_get(part, 'type') == 'refusal' for part in parts)
            coverage.append(_text(finish) is not None and bool(parts))
    return positive, bool(coverage) and all(coverage), received, reasons


def model_evidence(*, provider=None, response: Any = None, error: Any = None, attributes=None,
                   stream_chunks=None, stream_complete=False, refusal_signal=None) -> dict:
    from_attributes = attributes is not None
    attributes = attributes or {}
    statuses = [attributes.get('http.response.status_code'), attributes.get('http.status_code'),
                _get(error, 'status_code'), _get(error, 'status'), _get(error, 'statusCode'),
                _get(_get(error, 'response'), 'status_code'), _get(response, 'status_code')]
    # Google GenAI exposes its HTTP response status as APIError.code.
    if provider in {'google', 'gemini'}:
        statuses.append(_get(error, 'code'))
    status = next((value for value in statuses if type(value) is int and 100 <= value <= 599), None)
    payload = _get(error, 'error') or _get(error, 'body') or error
    payload = _get(payload, 'error', payload)
    code = _text(attributes.get('error.code')) or _text(_get(payload, 'code')) or _text(_get(error, 'code'))
    positive, complete, received, reasons = _inspect(response if not from_attributes else None)
    if from_attributes:
        for key in ('llm.output.value', 'output.value', 'gen_ai.output.messages'):
            native = attributes.get(key)
            if isinstance(native, str):
                try:
                    native = json.loads(native)
                except (ValueError, TypeError):
                    continue
            if isinstance(native, Mapping):
                p, c, r, raw = _inspect(native)
                positive, complete, received = positive or p, complete or c, received or r
                reasons.extend(raw)
            elif isinstance(native, (list, tuple)):
                # OTel messages contain native content parts. Inspect only the
                # documented refusal field/type, never textual responses.
                for message in native:
                    positive = positive or bool(_get(message, 'refusal'))
                    for part in _seq(_get(message, 'parts', _get(message, 'content'))):
                        positive = positive or _get(part, 'type') == 'refusal'

    for chunk in _seq(stream_chunks):
        for value in (chunk, _get(chunk, 'response')):
            p, _, r, raw = _inspect(value)
            positive = positive or p
            received = received or r
            reasons.extend(raw)
    raw_attributes = attributes.get('gen_ai.response.finish_reasons')
    if isinstance(raw_attributes, str):
        try:
            raw_attributes = json.loads(raw_attributes)
        except (ValueError, TypeError):
            raw_attributes = None
    raw_attributes = _seq(raw_attributes)
    reasons = [value for value in raw_attributes if _text(value)] + reasons
    if _text(attributes.get('llm.finish_reason')):
        reasons.append(attributes['llm.finish_reason'])
    hint = attributes.get('metergraph.refusal_signal', refusal_signal)
    signal = True if positive else hint if type(hint) is bool else (
        False if complete and (stream_chunks is None or stream_complete) else None)
    result = dict(http_status_code=status, error_code=code, refusal_signal=signal,
                  response_received=True if received or positive else attributes.get('metergraph.response_received') if type(attributes.get('metergraph.response_received')) is bool else None)
    if reasons:
        result['otel'] = {'finish_reasons': reasons}
    return result
