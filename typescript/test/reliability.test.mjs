import assert from "node:assert/strict";
import test from "node:test";
import { wrap, withContext } from "../dist/index.js";
import { CaptureRuntime, DEFAULT_TEXT_MAX_BYTES } from "../dist/capture.js";
import { setCaptureRuntime } from "../dist/wrap.js";

test("capture preserves response identity and emits native evidence without content", () => {
  const rows = [];
  setCaptureRuntime(
    new CaptureRuntime(
      {
        enqueue(row) {
          rows.push(row);
          return true;
        },
      },
      {
        captureText: false,
        appRoot: "",
        skipFrames: [],
        textMaxBytes: DEFAULT_TEXT_MAX_BYTES,
      },
    ),
  );
  const response = {
    choices: [{ finish_reason: "content_filter", message: { content: null } }],
  };
  const client = {
    chat: {
      completions: {
        create() {
          return response;
        },
      },
    },
  };
  try {
    wrap(client, { provider: "openai" });
    assert.equal(
      client.chat.completions.create({ model: "example-model", messages: [] }),
      response,
    );
    assert.equal(rows.length, 1);
    assert.equal("availability" in rows[0], false);
    assert.equal("refused" in rows[0], false);
    assert.deepEqual(rows[0].otel.finish_reasons, ["content_filter"]);
    assert.equal(rows[0].refusal_signal, null);
    assert.equal(rows[0].response_received, true);
    assert.equal(rows[0].response_text, undefined);
  } finally {
    setCaptureRuntime();
  }
});

test("stream refusal survives a terminal usage-only chunk", async () => {
  const rows = [];
  setCaptureRuntime(
    new CaptureRuntime(
      {
        enqueue(row) {
          rows.push(row);
          return true;
        },
      },
      {
        captureText: false,
        appRoot: "",
        skipFrames: [],
        textMaxBytes: DEFAULT_TEXT_MAX_BYTES,
      },
    ),
  );
  const chunks = [
    { choices: [{ delta: { refusal: "policy" }, finish_reason: null }] },
    { choices: [{ delta: {}, finish_reason: "stop" }] },
    { choices: [], usage: { prompt_tokens: 3, completion_tokens: 1 } },
  ];
  const client = {
    chat: {
      completions: {
        create() {
          return {
            async *[Symbol.asyncIterator]() {
              yield* chunks;
            },
          };
        },
      },
    },
  };
  try {
    wrap(client, { provider: "openai" });
    for await (const chunk of client.chat.completions.create({
      model: "example-model",
      messages: [],
      stream: true,
    })) {
    }
    assert.equal(rows.length, 1);
    assert.equal(rows[0].refusal_signal, true);
    assert.deepEqual(rows[0].otel.finish_reasons, ["stop"]);
    assert.equal("availability" in rows[0], false);
    assert.equal(rows[0].input_tokens, 3);
  } finally {
    setCaptureRuntime();
  }
});


test("scoped refusal hint is inherited and native positive wins", async () => {
  const { contextSnapshot } = await import("../dist/context.js");
  const { modelEvidence } = await import("../dist/model-evidence.js");
  withContext({ refusalSignal: false }, () => {
    assert.equal(contextSnapshot().refusalSignal, false);
    assert.equal(modelEvidence({ response: { choices: [{ message: { refusal: "policy" }, finish_reason: "stop" }] }, refusalSignal: contextSnapshot().refusalSignal }).refusal_signal, true);
    withContext({ refusalSignal: true }, () => assert.equal(contextSnapshot().refusalSignal, true));
    assert.equal(contextSnapshot().refusalSignal, false);
  });
  assert.equal(contextSnapshot().refusalSignal, undefined);
});


test("evidence preserves raw multi-choice and prompt-block facts", async () => {
  const { modelEvidence } = await import("../dist/model-evidence.js");
  assert.deepEqual(modelEvidence({ response: { choices: [
    { finish_reason: "stop", message: { content: "private content" } },
    { finish_reason: "content_filter", message: {} }] },
    error: { status_code: 400, body: { error: { code: "structured_code", message: "private error" } } } }), {
      http_status_code: 400, error_code: "structured_code", refusal_signal: null, response_received: true,
      otel: { finish_reasons: ["stop", "content_filter"] },
    });
  assert.deepEqual(modelEvidence({ response: { promptFeedback: { blockReason: "SAFETY" } } }).otel.finish_reasons, ["SAFETY"]);
  assert.equal(modelEvidence({ response: { choices: [{ finish_reason: "stop", message: { content: "ok" } }] } }).refusal_signal, false);
  assert.equal(modelEvidence({ stream_chunks: [{ choices: [{ finish_reason: "stop", delta: {} }] }], stream_complete: true }).refusal_signal, null);
  assert.deepEqual(modelEvidence({ stream_chunks: [{ type: "response.refusal.delta" }] }), {
    http_status_code: null, error_code: null, refusal_signal: true, response_received: true,
  });
});
