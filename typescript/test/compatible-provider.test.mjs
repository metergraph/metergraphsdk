import test from 'node:test';
import assert from 'node:assert/strict';
import { wrap } from '../dist/wrap.js';
import { installRuntime } from './support/instrumentation-doubles.mjs';

test('an explicitly named OpenAI-compatible provider retains its identity and native response', async t => {
  const rows = installRuntime(t);
  const response = { id: 'example-id', model: 'example-model', choices: [], usage: { prompt_tokens: 100, completion_tokens: 20 } };
  const request = { model: 'example-model', messages: [] };
  let received;
  const client = { chat: { completions: { create: async value => { received = value; return response; } } } };
  assert.equal(wrap(client, { provider: 'example-provider' }), client);
  assert.equal(await client.chat.completions.create(request), response);
  assert.equal(received, request);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].provider, 'example-provider');
  assert.equal(rows[0].input_tokens, 100);
  assert.equal(rows[0].output_tokens, 20);
});
