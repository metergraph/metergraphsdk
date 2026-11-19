import assert from 'node:assert/strict';
import test from 'node:test';
import Anthropic from '@anthropic-ai/sdk';
import { wrap } from '../dist/wrap.js';
import { installRuntime } from './support/instrumentation-doubles.mjs';

const snapshot = {
  id: 'msg_partial', type: 'message', role: 'assistant', model: 'claude-sonnet-4-6',
  content: [], stop_reason: null,
  usage: { input_tokens: 100, output_tokens: 0,
    cache_read_input_tokens: 20, cache_creation_input_tokens: 10 },
};

function truncatedResponse() {
  return new Response(`event: message_start\ndata: ${JSON.stringify({ type: 'message_start', message: snapshot })}\n\n`, {
    status: 200, headers: { 'content-type': 'text/event-stream', 'request-id': 'req_partial' },
  });
}

test('real Anthropic finalMessage failure retains partial usage without inventing a final charge', async (t) => {
  const rows = installRuntime(t, { captureText: false });
  let requests = 0;
  const native = new Anthropic({ apiKey: 'test', fetch: async () => {
    requests++; return truncatedResponse();
  } });
  const client = wrap(native, 'anthropic');
  await assert.rejects(client.messages.stream({ model: snapshot.model, messages: [], max_tokens: 100 }).finalMessage(),
    /stream ended without producing a Message/);
  assert.equal(requests, 1);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].error, true);
  assert.equal(rows[0].input_tokens, 100);
  assert.equal(rows[0].cache_read_tokens, 20);
  assert.equal(rows[0].cache_write_tokens, 10);
  assert.equal(rows[0].usage_complete, false);
  assert.equal(rows[0].request_id, 'msg_partial');
  assert.equal(rows[0].request_json, undefined);
});

test('failed helper preserves native error identity and HTTP request id', async (t) => {
  const rows = installRuntime(t);
  const failure = new Error('stream ended without producing a Message with role=assistant');
  const stream = { currentMessage: snapshot, request_id: 'req_partial',
    async finalMessage() { throw failure; } };
  const client = wrap({ messages: { stream: () => stream } }, 'anthropic');
  await assert.rejects(client.messages.stream({ model: snapshot.model }).finalMessage(), error => error === failure);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].http_request_id, 'req_partial');
  assert.equal(rows[0].usage_complete, false);
});

test('broken telemetry getters never replace a native stream error', async (t) => {
  const rows = installRuntime(t);
  const failure = new Error('provider stream failed');
  const stream = { get currentMessage() { throw new Error('snapshot unavailable'); },
    get request_id() { throw new Error('request id unavailable'); },
    async finalMessage() { throw failure; } };
  const client = wrap({ messages: { stream: () => stream } }, 'anthropic');
  await assert.rejects(client.messages.stream({ model: snapshot.model }).finalMessage(), error => error === failure);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].error, true);
});
