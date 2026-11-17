import assert from "node:assert/strict";
import http from "node:http";
import test from "node:test";

import { Transport } from "../dist/transport.js";

async function serve(handler) {
  const server = http.createServer(handler);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return server;
}

class FakeSession {
  constructor(token) {
    this.token = token;
    this.invalidated = 0;
  }

  async getToken() {
    return this.token;
  }

  invalidate() {
    this.invalidated += 1;
    this.token = undefined;
  }
}

test("transport sends the session token and never the app token", async (t) => {
  let authorization;
  const server = await serve((request, response) => {
    authorization = request.headers.authorization;
    request.resume();
    response.writeHead(202);
    response.end();
  });
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const address = server.address();

  const session = new FakeSession("session-abc");
  const transport = new Transport("app-token-secret", `http://127.0.0.1:${address.port}`, {
    mode: "buffered",
    session,
  });
  transport.enqueue({ payload: "x" });
  await transport.flush(2_000);

  assert.equal(authorization, "Bearer session-abc");
});

test("transport drops the batch without a request when session has no token yet", async (t) => {
  let calls = 0;
  const server = await serve((request, response) => {
    calls += 1;
    request.resume();
    response.writeHead(202);
    response.end();
  });
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const address = server.address();

  const session = new FakeSession(undefined);
  const transport = new Transport("app-token-secret", `http://127.0.0.1:${address.port}`, {
    mode: "buffered",
    session,
  });
  transport.enqueue({ payload: "x" });
  await transport.flush(2_000);

  assert.equal(calls, 0);
});

test("transport invalidates the session on 401 instead of going fatal", async (t) => {
  const server = await serve((request, response) => {
    request.resume();
    response.writeHead(401);
    response.end();
  });
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const address = server.address();

  const session = new FakeSession("session-abc");
  const warnings = [];
  const originalWarn = console.warn;
  console.warn = (...args) => warnings.push(args.join(" "));
  const transport = new Transport("app-token-secret", `http://127.0.0.1:${address.port}`, {
    mode: "buffered",
    session,
  });
  transport.enqueue({ payload: "x" });
  await transport.flush(2_000);
  console.warn = originalWarn;

  assert.equal(session.invalidated, 1);
  assert.ok(!warnings.some((w) => w.includes("authentication failed")));
});

test("transport invalidates the session on 403 instead of going fatal", async (t) => {
  const server = await serve((request, response) => {
    request.resume();
    response.writeHead(403);
    response.end();
  });
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const address = server.address();

  const session = new FakeSession("session-abc");
  const transport = new Transport("app-token-secret", `http://127.0.0.1:${address.port}`, {
    mode: "buffered",
    session,
  });
  transport.enqueue({ payload: "x" });
  await transport.flush(2_000);

  assert.equal(session.invalidated, 1);
});

test("shutdown cancels a stalled upload and does not start queued uploads", async (t) => {
  let calls = 0;
  let disconnected;
  const closed = new Promise((resolve) => { disconnected = resolve; });
  let received;
  const started = new Promise((resolve) => { received = resolve; });
  const server = await serve((request, response) => {
    calls += 1;
    request.resume();
    response.on("close", disconnected);
    received();
    // Deliberately never send response headers.
  });
  t.after(() => { server.closeAllConnections(); server.close(); });
  const transport = new Transport("test-token", `http://127.0.0.1:${server.address().port}`, {
    mode: "background", batchSize: 1, flushMs: 60_000,
  });
  t.after(() => transport.shutdown());
  transport.enqueue({ id: 1 });
  await started;
  transport.enqueue({ id: 2 });
  const before = Date.now();
  await transport.shutdown();
  assert.ok(Date.now() - before < 4_000, "shutdown must remain bounded");
  const cancelled = await Promise.race([
    closed.then(() => true),
    new Promise((resolve) => setTimeout(() => resolve(false), 500)),
  ]);
  assert.equal(cancelled, true, "shutdown must close the stalled upload");
  assert.equal(calls, 1, "shutdown must not start another upload after its deadline");
  assert.equal(transport.enqueue({ id: 3 }), false, "closed transport rejects new records");
});
