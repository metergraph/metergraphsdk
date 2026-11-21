import assert from "node:assert/strict";
import test from "node:test";
import { wrap } from "../dist/wrap.js";
import { installRuntime } from "./support/instrumentation-doubles.mjs";

test("wrapping multiple clients is quiet and still captures their calls", (t) => {
  const rows = installRuntime(t);
  const info = t.mock.method(console, "info", () => {});
  const warn = t.mock.method(console, "warn", () => {});
  const response = { id: "response", usage: { input_tokens: 1, output_tokens: 1 } };
  for (let i = 0; i < 3; i++) {
    const client = { responses: { create: () => response } };
    assert.equal(wrap(client, "openai"), client);
    assert.equal(client.responses.create({ model: "example-model" }), response);
  }
  assert.equal(rows.length, 3);
  assert.equal(info.mock.callCount(), 0);
  assert.equal(warn.mock.callCount(), 0);
});

test("unsupported clients still produce a warning", (t) => {
  installRuntime(t);
  const warn = t.mock.method(console, "warn", () => {});
  const client = {};
  assert.equal(wrap(client, "openai"), client);
  assert.equal(warn.mock.callCount(), 1);
  assert.match(warn.mock.calls[0].arguments[0], /no supported methods/);
});
