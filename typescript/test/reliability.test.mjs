import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { classifyModelReliability } from "../dist/reliability.js";
const cases = JSON.parse(
  readFileSync(
    new URL("../../tests/fixtures/reliability.json", import.meta.url),
    "utf8",
  ),
);
for (const item of cases)
  test(item.name, () => {
    assert.deepEqual(classifyModelReliability(item.input), item.expected);
  });

import { wrap } from "../dist/index.js";
import { CaptureRuntime, DEFAULT_TEXT_MAX_BYTES } from "../dist/capture.js";
import { setCaptureRuntime } from "../dist/wrap.js";

test("capture preserves response identity and emits one outcome without content", () => {
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
    assert.equal(rows[0].availability, "available");
    assert.equal(rows[0].refused, true);
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
    assert.equal(rows[0].refused, true);
    assert.equal(rows[0].availability, "available");
    assert.equal(rows[0].input_tokens, 3);
  } finally {
    setCaptureRuntime();
  }
});
