import assert from 'node:assert/strict';
import test from 'node:test';
import { CaptureRuntime } from '../dist/capture.js';

function record(reason, extra = {}) {
  const rows = [];
  const runtime = new CaptureRuntime({enqueue(row) { rows.push(row); return true; }},
    {captureText: false, appRoot: '', skipFrames: [], textMaxBytes: 32});
  runtime.finish(runtime.start('openai', 'chat.completions.create', {model: 'example-model'}),
    {choices: [{finish_reason: reason}], usage: {prompt_tokens: 10, completion_tokens: 2}}, extra);
  return rows[0];
}

for (const reason of ['', ' \t ']) {
  test(`blank finish reason ${JSON.stringify(reason)} uses success lifecycle status`, () => {
    const row = record(reason);
    assert.equal(row.status, 'success');
    assert.equal(row.input_tokens, 10);
    assert.equal(row.output_tokens, 2);
    if (reason.trim() !== reason) assert.equal(row.finish_reason_raw, reason);
  });
  test(`blank finish reason ${JSON.stringify(reason)} preserves explicit errors and abandoned streams`, () => {
    const failed = record(reason, {error: new Error('synthetic failure'), stream: true});
    assert.equal(failed.status, 'error');
    assert.equal(failed.status_code, 'error');
    assert.equal(failed.usage_complete, false);
    const abandoned = record(reason, {status: 'abandoned', stream: true});
    assert.equal(abandoned.status, 'abandoned');
    assert.equal(abandoned.usage_complete, false);
  });
}

test('nonblank finish reasons retain existing lifecycle status', () => {
  assert.equal(record('length').status, 'length');
});
