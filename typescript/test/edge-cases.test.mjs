import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import test from "node:test";

import { CaptureRuntime } from "../dist/capture.js";
import { createVercelAISDKMiddleware } from "../dist/vercel-ai.js";
import { setCaptureRuntime, wrap } from "../dist/wrap.js";


function stubRuntime(rows, options = {}) {
  return new CaptureRuntime(
    { enqueue(row) { rows.push(row); return true; } },
    {
      captureText: true,
      appRoot: "",
      skipFrames: [],
      textMaxBytes: 100 * 1024,
      ...options,
    },
  );
}


function capturedResponse(row) {
  return JSON.parse(row.response_text);
}


async function readAll(stream) {
  const reader = stream.getReader();
  const parts = [];
  while (true) {
    const next = await reader.read();
    if (next.done) return parts;
    parts.push(next.value);
  }
}


test("Vercel middleware versions and disabled capture are exact pass-throughs", async (t) => {
  setCaptureRuntime();
  t.after(() => setCaptureRuntime());
  assert.equal(createVercelAISDKMiddleware().specificationVersion, "v3");
  assert.equal(createVercelAISDKMiddleware({ specificationVersion: "v2" }).specificationVersion, "v2");
  assert.equal(createVercelAISDKMiddleware({ specificationVersion: "v4" }).specificationVersion, "v4");
  assert.equal(
    createVercelAISDKMiddleware({ aiSdkVersion: 5 }).specificationVersion,
    "v2",
  );
  assert.throws(
    () => createVercelAISDKMiddleware({
      aiSdkVersion: 5,
      specificationVersion: "v3",
    }),
    /aiSdkVersion.*specificationVersion/,
  );

  const generated = { content: [{ type: "text", text: "unchanged" }] };
  let generateCalls = 0;
  const result = await createVercelAISDKMiddleware().wrapGenerate({
    model: {},
    params: {},
    doGenerate: () => {
      generateCalls += 1;
      return generated;
    },
  });
  assert.equal(result, generated);
  assert.equal(generateCalls, 1);

  const streamed = { stream: new ReadableStream() };
  let streamCalls = 0;
  const streamResult = await createVercelAISDKMiddleware().wrapStream({
    model: {},
    params: {},
    doStream: () => {
      streamCalls += 1;
      return streamed;
    },
  });
  assert.equal(streamResult, streamed);
  assert.equal(streamCalls, 1);
});


test("Vercel provider aliases normalize and request capture excludes transport secrets", async (t) => {
  const rows = [];
  setCaptureRuntime(stubRuntime(rows));
  t.after(() => setCaptureRuntime());
  const middleware = createVercelAISDKMiddleware({ specificationVersion: "v4" });
  const cases = [
    ["gateway", "anthropic/claude-sonnet", "anthropic"],
    ["vercel-ai-gateway", "amazon/nova-pro", "amazon"],
    ["amazon-bedrock", "nova-pro", "bedrock"],
    ["google.genai", "gemini-3", "google"],
    ["openai.responses", "gpt-5", "openai"],
    ["xai", "grok-4", "xai"],
  ];

  for (const [provider, modelId] of cases) {
    const response = {
      content: [{ type: "text", text: "ok" }],
      usage: { inputTokens: "4", outputTokens: 2 },
    };
    const actual = await middleware.wrapGenerate({
      model: { provider, modelId },
      params: {
        prompt: [{ role: "user", content: "safe prompt" }],
        temperature: 0,
        headers: { authorization: "Bearer header-secret" },
        providerOptions: { apiKey: "provider-secret" },
        abortSignal: { secret: "abort-secret" },
        callback: () => "callback-secret",
      },
      doGenerate: () => response,
    });
    assert.equal(actual, response);
  }

  assert.deepEqual(rows.map((row) => row.provider), cases.map((item) => item[2]));
  for (const row of rows) {
    assert.equal(row.input_tokens, 4);
    assert.equal(row.output_tokens, 2);
    assert.doesNotMatch(
      row.request_json,
      /header-secret|provider-secret|abort-secret|callback-secret|authorization|apiKey/,
    );
    const request = JSON.parse(row.request_json);
    assert.equal(request.temperature, 0);
    assert.equal(request.messages[0].content, "safe prompt");
    assert.equal("headers" in request, false);
    assert.equal("providerOptions" in request, false);
    assert.equal("abortSignal" in request, false);
  }
});


test("malformed usage never becomes a token count", async (t) => {
  const rows = [];
  setCaptureRuntime(stubRuntime(rows));
  t.after(() => setCaptureRuntime());
  await createVercelAISDKMiddleware().wrapGenerate({
    model: { provider: "gateway", modelId: "openai/gpt-5" },
    params: { prompt: [] },
    doGenerate: () => ({
      content: [],
      usage: {
        inputTokens: true,
        outputTokens: -1,
        inputTokenDetails: { cacheReadTokens: 1.5 },
        outputTokenDetails: { reasoningTokens: Number.POSITIVE_INFINITY },
        raw: { cache_creation_input_tokens: Number.NaN },
      },
    }),
  });

  for (const field of [
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
  ]) {
    assert.equal(rows[0][field], undefined, field);
  }
});


test("Vercel adapter is fail-open when capture start or finish throws", async (t) => {
  t.after(() => setCaptureRuntime());
  const middleware = createVercelAISDKMiddleware();
  const generated = { content: [{ type: "text", text: "provider result" }] };

  setCaptureRuntime({
    start() { throw new Error("capture start failed"); },
    finish() { assert.fail("finish must not run without state"); },
  });
  assert.equal(await middleware.wrapGenerate({
    model: {},
    params: {},
    doGenerate: () => generated,
  }), generated);

  setCaptureRuntime({
    start() { return { done: false }; },
    finish() { throw new Error("capture finish failed"); },
  });
  assert.equal(await middleware.wrapGenerate({
    model: {},
    params: {},
    doGenerate: () => generated,
  }), generated);

  const failure = new TypeError("provider failed");
  await assert.rejects(
    middleware.wrapGenerate({
      model: {},
      params: {},
      doGenerate() { throw failure; },
    }),
    (error) => error === failure,
  );
});


test("non-readable and getReader-failing streams capture exactly once", async (t) => {
  const rows = [];
  setCaptureRuntime(stubRuntime(rows));
  t.after(() => setCaptureRuntime());
  const middleware = createVercelAISDKMiddleware();

  const nonReadable = { stream: { reason: "not a web stream" } };
  assert.equal(await middleware.wrapStream({
    model: { provider: "openai", modelId: "m" },
    params: { prompt: [] },
    doStream: () => nonReadable,
  }), nonReadable);

  const lockFailure = new Error("stream already locked");
  const locked = {
    stream: {
      getReader() { throw lockFailure; },
    },
  };
  assert.equal(await middleware.wrapStream({
    model: { provider: "openai", modelId: "m" },
    params: { prompt: [] },
    doStream: () => locked,
  }), locked);

  assert.equal(rows.length, 2);
  assert.equal(rows[0].status, "success");
  assert.equal(rows[0].stream, true);
  assert.equal(rows[1].status, "error");
  assert.equal(rows[1].error_type, "Error");
});


test("stream read rejection and cancellation preserve provider behavior and terminal status", async (t) => {
  const rows = [];
  setCaptureRuntime(stubRuntime(rows));
  t.after(() => setCaptureRuntime());
  const middleware = createVercelAISDKMiddleware();
  const readFailure = new RangeError("reader failed");

  const failed = await middleware.wrapStream({
    model: { provider: "openai", modelId: "m" },
    params: { prompt: [] },
    doStream: () => ({
      stream: {
        getReader() {
          return {
            read() { return Promise.reject(readFailure); },
            cancel() { return Promise.resolve(); },
          };
        },
      },
    }),
  });
  await assert.rejects(failed.stream.getReader().read(), (error) => error === readFailure);

  let cancelReason;
  const cancellable = await middleware.wrapStream({
    model: { provider: "anthropic", modelId: "claude" },
    params: { prompt: [] },
    doStream: () => ({
      stream: {
        getReader() {
          return {
            read() { return new Promise(() => {}); },
            cancel(reason) {
              cancelReason = reason;
              return Promise.resolve();
            },
          };
        },
      },
    }),
  });
  await cancellable.stream.getReader().cancel("caller stopped");

  assert.equal(cancelReason, "caller stopped");
  assert.equal(rows.length, 2);
  assert.equal(rows[0].status, "error");
  assert.equal(rows[0].error_type, "RangeError");
  assert.equal(rows[1].status, "abandoned");
});


test("tool-only and in-band error streams capture TTFT, tools, metadata, and errors", async (t) => {
  const rows = [];
  setCaptureRuntime(stubRuntime(rows));
  t.after(() => setCaptureRuntime());
  const middleware = createVercelAISDKMiddleware({ specificationVersion: "v4" });

  const toolParts = [
    { type: "response-metadata", id: "meta-id", modelId: "claude" },
    {
      type: "tool-call",
      toolCallId: "call-1",
      toolName: "lookup",
      input: "{not-json",
    },
    {
      type: "tool-error",
      toolCallId: "call-1",
      toolName: "lookup",
      error: { message: "tool failed" },
    },
    {
      type: "finish",
      finishReason: { unified: "tool-calls", raw: "tool_use" },
      usage: { inputTokens: 8, outputTokens: 3 },
    },
  ];
  const toolResult = await middleware.wrapStream({
    model: { provider: "gateway", modelId: "anthropic/claude" },
    params: { prompt: [] },
    doStream: () => ({
      response: { id: "result-id" },
      stream: new ReadableStream({
        start(controller) {
          for (const part of toolParts) controller.enqueue(part);
          controller.close();
        },
      }),
    }),
  });
  assert.deepEqual(await readAll(toolResult.stream), toolParts);

  const streamFailure = new SyntaxError("bad provider event");
  const errorPart = { type: "error", error: streamFailure };
  const errorResult = await middleware.wrapStream({
    model: { provider: "openai", modelId: "gpt" },
    params: { prompt: [] },
    doStream: () => ({
      stream: new ReadableStream({
        start(controller) {
          controller.enqueue(errorPart);
          controller.close();
        },
      }),
    }),
  });
  assert.deepEqual(await readAll(errorResult.stream), [errorPart]);

  assert.equal(rows.length, 2);
  assert.ok(rows[0].ttft_ms >= 0);
  assert.equal(rows[0].request_id, "result-id");
  assert.equal(rows[0].input_tokens, 8);
  assert.deepEqual(rows[0].tool_names, ["lookup"]);
  assert.deepEqual(rows[0].tool_calls[0], {
    call_id: "call-1",
    name: "lookup",
    arguments: "{not-json",
    result: { message: "tool failed" },
    status: "error",
    idempotency: "non_idempotent",
  });
  assert.equal(rows[1].status, "error");
  assert.equal(rows[1].error_type, "SyntaxError");
  assert.deepEqual(capturedResponse(rows[1]).error, {
    type: "SyntaxError",
    message: "bad provider event",
  });
});


test("provider wrapper sets TTFT for a tool-only stream and captures iterator errors", async (t) => {
  const rows = [];
  setCaptureRuntime(stubRuntime(rows));
  t.after(() => setCaptureRuntime());

  const iteratorFailure = new URIError("provider iterator failed");
  const client = wrap({
    messages: {
      stream(request) {
        if (request.model === "reasoning-only") {
          return {
            async *[Symbol.asyncIterator]() {
              yield {
                type: "content_block_delta",
                delta: { type: "thinking_delta", thinking: "reasoning" },
              };
            },
          };
        }
        return {
          async *[Symbol.asyncIterator]() {
            yield {
              type: "content_block_start",
              index: 0,
              content_block: {
                type: "tool_use",
                id: "tool-1",
                name: "lookup",
                input: {},
              },
            };
            yield {
              type: "content_block_delta",
              index: 0,
              delta: { type: "input_json_delta", partial_json: '{"id":"1"}' },
            };
            throw iteratorFailure;
          },
        };
      },
    },
  }, "anthropic");

  const reasoningStream = client.messages.stream({
    model: "reasoning-only",
    messages: [],
  });
  for await (const _part of reasoningStream) { /* consume */ }

  const stream = client.messages.stream({ model: "claude", messages: [] });
  await assert.rejects(async () => {
    for await (const _part of stream) { /* consume */ }
  }, (error) => error === iteratorFailure);

  assert.equal(rows.length, 2);
  assert.ok(rows[0].ttft_ms >= 0);
  assert.equal(rows[0].status, "success");
  assert.ok(rows[1].ttft_ms >= 0);
  assert.equal(rows[1].status, "error");
  assert.equal(rows[1].error_type, "URIError");
  assert.deepEqual(rows[1].tool_names, ["lookup"]);
  assert.deepEqual(rows[1].tool_calls[0].arguments, { id: "1" });
});


test("a Responses reply that is only a function call records its output, not the text config", async (t) => {
  // A raw Responses body has no output_text, and its `text` is the text
  // *config*; the reply is its `output`.
  const rows = [];
  setCaptureRuntime(stubRuntime(rows));
  t.after(() => setCaptureRuntime());

  const call = {
    type: "function_call", id: "fc_1", call_id: "call_1",
    name: "get_weather", arguments: "{\"city\":\"SF\"}", status: "completed",
  };
  const client = wrap({
    responses: {
      async create() {
        return {
          id: "resp_1", status: "completed", model: "gpt-5",
          text: { format: { type: "text" }, verbosity: "medium" },
          output: [call],
        };
      },
    },
  }, { provider: "openai" });
  await client.responses.create({ model: "gpt-5", input: "weather in SF?" });

  assert.deepEqual(capturedResponse(rows[0]).content, [call]);
});


test("a Responses text reply still records output_text", async (t) => {
  const rows = [];
  setCaptureRuntime(stubRuntime(rows));
  t.after(() => setCaptureRuntime());

  const client = wrap({
    responses: {
      async create() {
        return {
          id: "resp_1", status: "completed", model: "gpt-5", output_text: "hello",
          text: { format: { type: "text" } }, output: [],
        };
      },
    },
  }, { provider: "openai" });
  await client.responses.create({ model: "gpt-5", input: "hi" });

  assert.equal(capturedResponse(rows[0]).content, "hello");
});


const toolWithSensitiveNames = {
  type: "function",
  function: {
    name: "login",
    parameters: {
      type: "object",
      properties: { token: { type: "string" }, password: { type: "string" } },
      required: ["token", "password"],
    },
  },
};


test("request capture strips credentials only where they travel", async (t) => {
  // A tool parameter or schema property sharing a credential's name is part of
  // the replayed request.
  const rows = [];
  setCaptureRuntime(stubRuntime(rows));
  t.after(() => setCaptureRuntime());

  const request = {
    model: "gpt-5",
    apiKey: "sk-top-level",
    authorization: "Bearer top-level",
    extraHeaders: { "x-trace": "keep-out" },
    defaultQuery: { "api-version": "keep-out" },
    messages: [
      { role: "user", content: "log me in" },
      {
        role: "assistant", content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "login", arguments: "{\"token\":\"t\"}" } }],
      },
      { role: "tool", tool_call_id: "call_1", content: "ok" },
    ],
    tools: [toolWithSensitiveNames],
    response_format: {
      type: "json_schema",
      json_schema: { name: "Session", schema: { type: "object", properties: { secret: { type: "string" } } } },
    },
  };
  const client = wrap({
    chat: { completions: { async create() { return { id: "r", choices: [{ message: { content: "ok" } }] }; } } },
  }, { provider: "openai" });
  await client.chat.completions.create(request);

  const raw = rows[0].request_json;
  assert.doesNotMatch(raw, /sk-top-level|Bearer top-level|keep-out/);
  const captured = JSON.parse(raw);
  assert.deepEqual(captured.tools, [toolWithSensitiveNames]);
  assert.deepEqual(captured.response_format, request.response_format);
  assert.deepEqual(captured.messages, request.messages);
});


test("template hash still ignores credential names at any depth", async () => {
  // This hash routes unnamed workloads, so its input must stay stable.
  const { templateHash } = await import("../dist/template.js");
  const without = {
    model: "m",
    tools: [{
      type: "function",
      function: { name: "login", parameters: { type: "object", properties: {}, required: ["token", "password"] } },
    }],
  };
  assert.equal(templateHash({ model: "m", tools: [toolWithSensitiveNames] }), templateHash(without));
});

test("template hash is pinned for representative requests", async () => {
  // This hash routes unnamed workloads, so a change to any value here moves
  // existing traffic to a new workload. Change one only on purpose.
  const { templateHash } = await import("../dist/template.js");
  const pinned = [
    [
      { model: "gpt-4o", messages: [{ role: "user", content: "Reply to (jane.doe+billing@example.co.uk) about ticket 42." }] },
      "e1515e7d0143618a",
    ],
    [
      { model: "gpt-4o", messages: [{ role: "user", content: "Owner: bob@example.com; cc <ann@example.org>. Thanks!" }] },
      "2e23d1c6b3d55cb3",
    ],
    [
      { model: "gpt-4o", messages: [{ role: "user", content: "See https://user@example.com/path?id=7 and https://example.com/x" }] },
      "df6549839e3b7f36",
    ],
    [
      {
        model: "claude-3-5-sonnet",
        system: "Order 1234-5678 for 3f2504e0-4f89-11d3-9a0c-0305e82c3301 sent to a@b.io",
        messages: [{ role: "user", content: "token sk-abcdefghijklmnopqrstuvwxyz0123 price 19.99 at @handle" }],
      },
      "938d33584f011d8f",
    ],
    [
      { model: "gpt-4o", messages: [{ role: "system", content: "You are a helpful assistant." }, { role: "user", content: "hi" }] },
      "a78c28279eb30bd4",
    ],
  ];
  for (const [request, expected] of pinned) assert.equal(templateHash(request), expected);
});

test("template hash is linear on long opaque strings", async () => {
  // The email pattern used to rescan a run with no whitespace or "@" from
  // every word boundary: a 100 KB base64 image took a third of a second and
  // each doubling took four times as long, inside the wrapped call. "@"
  // followed by a long run of dots was worse again, and so is a run of
  // "data:" that is not a data URL. Each of these now takes milliseconds; the
  // bound is loose so a slow runner cannot fail it.
  const { templateHash } = await import("../dist/template.js");
  const image = randomBytes(300_000).toString("base64");
  const texts = [
    image,
    `data:image/png;base64,${image}`,
    `a@b.com${".".repeat(100_000)}`,
    "data:".repeat(200_000),
  ];
  for (const text of texts) {
    const started = process.hrtime.bigint();
    templateHash({ messages: [{ content: text }] });
    assert.ok(Number(process.hrtime.bigint() - started) / 1e9 < 1);
  }
});

test("template hash ignores inline image bytes", async () => {
  const { templateHash } = await import("../dist/template.js");
  const request = (url) => ({
    model: "gpt-4o",
    messages: [{
      role: "user",
      content: [
        { type: "text", text: "Describe this screenshot." },
        { type: "image_url", image_url: { url } },
      ],
    }],
  });
  const inline = (bytes) => `data:image/png;base64,${Buffer.from(bytes).toString("base64")}`;
  const first = templateHash(request(inline(Array.from({ length: 4096 }, (_, i) => i % 256))));
  assert.equal(first, templateHash(request(inline(Array.from({ length: 4096 }, (_, i) => 255 - (i % 256))))));
  assert.equal(first, templateHash(request("https://images.example.com/shot.png")));
});
