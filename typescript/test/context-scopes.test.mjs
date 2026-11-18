import assert from "node:assert/strict";
import test from "node:test";

import {
  contextSnapshot,
  setDefaultTags,
  setSession,
  setUser,
  setTags,
  withContext,
  withSession,
  withTags,
} from "../dist/context.js";

test("overlapping jobs keep session and tags isolated", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const seen = [];

  const first = withContext({ userId: "user-a", sessionId: "job-a", metadata: { customer: "a" } }, async () => {
    await gate;
    seen.push(contextSnapshot());
  });
  const second = withContext({ userId: "user-b", sessionId: "job-b", metadata: { customer: "b" } }, async () => {
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
    seen.push(contextSnapshot());
  });
  await Promise.all([first, second]);

  assert.deepEqual(seen.map(({ sessionId, tags }) => [sessionId, tags.customer]).sort(), [
    ["job-a", "a"],
    ["job-b", "b"],
  ]);
  assert.equal(contextSnapshot().sessionId, undefined);
  assert.deepEqual(seen.map(({ userId }) => userId).sort(), ["user-a", "user-b"]);
});

test("user and metadata inherit, override, clear and restore", () => {
  const values = { customer_id: "first", priority: "metadata" };
  setDefaultTags({ service: "backend" });
  try {
    withContext({ userId: " person ", sessionId: "session", tags: { priority: 7 }, metadata: values }, () => {
      values.customer_id = "mutated";
      assert.deepEqual(contextSnapshot().tags, { service: "backend", priority: "metadata", customer_id: "first" });
      withContext({ metadata: { customer_id: 9, new: "yes" } }, () => {
        assert.equal(contextSnapshot().userId, " person ");
        assert.equal(contextSnapshot().tags.customer_id, "first");
        setUser("child");
        setUser(3);
        assert.equal(contextSnapshot().userId, "child");
        setUser();
        setSession();
        assert.equal(contextSnapshot().userId, undefined);
        assert.equal(contextSnapshot().sessionId, undefined);
      });
      assert.equal(contextSnapshot().userId, " person ");
      assert.equal(contextSnapshot().sessionId, "session");
    });
    setUser("outside");
    assert.equal(contextSnapshot().userId, undefined);
  } finally { setDefaultTags({}); }
});

test("attribution bounds preserve strings and Unicode codepoint order", () => {
  const values = Object.fromEntries(Array.from({ length: 31 }, (_, i) => [`a${String(i).padStart(2, "0")}`, "v"]));
  Object.assign(values, { "😀": "astral", "～": "bmp", api_key: "secret" });
  withContext({ userId: "🙂".repeat(512), metadata: values }, () => {
    assert.equal(contextSnapshot().userId, "🙂".repeat(512));
    const tags = contextSnapshot().tags;
    assert.equal(Object.keys(tags).length, 32);
    assert.equal(tags["～"], "bmp");
    assert.equal(tags["😀"], undefined);
    assert.equal(tags.api_key, undefined);
    withContext({ userId: "\0", metadata: { bad: "\ud800", nested: {}, ok: "" } }, () => {
      assert.equal(contextSnapshot().userId, "🙂".repeat(512));
      assert.equal(contextSnapshot().tags.bad, undefined);
      assert.equal(contextSnapshot().tags.nested, undefined);
    });
  });
  withContext({ userId: "\u0085", metadata: Object.fromEntries(Array.from({ length: 32 }, (_, i) => [`k${i}`, "🙂".repeat(1000)])) }, () => {
    assert.equal(contextSnapshot().userId, undefined);
    const tags = contextSnapshot().tags;
    assert.ok(Object.keys(tags).length > 0 && Object.keys(tags).length < 32);
    assert.ok(Buffer.byteLength(JSON.stringify(tags)) <= 65536);
  });
});

test("nested helpers derive child contexts and restore parents after errors", async () => {
  await withSession("parent", async () => {
    await assert.rejects(
      withTags({ layer: "child" }, async () => {
        assert.equal(contextSnapshot().sessionId, "parent");
        assert.equal(contextSnapshot().tags.layer, "child");
        await new Promise((resolve) => setTimeout(resolve, 0));
        throw new Error("boom");
      }),
      /boom/,
    );
    assert.equal(contextSnapshot().sessionId, "parent");
    assert.equal(contextSnapshot().tags.layer, undefined);
  });
  assert.equal(contextSnapshot().sessionId, undefined);
});

test("legacy setters outside a scope warn once and do not create request state", (t) => {
  const warnings = [];
  t.mock.method(console, "warn", (message) => warnings.push(String(message)));

  setSession("unsafe");
  setSession("unsafe-again");
  setTags({ customer: "unsafe" });
  setTags({ customer: "unsafe-again" });

  assert.equal(contextSnapshot().sessionId, undefined);
  assert.equal(contextSnapshot().tags.customer, undefined);
  assert.equal(warnings.filter((message) => message.includes("active Metergraph context")).length, 2);
});

test("scoped setters remain compatible and default tags are explicit", async () => {
  setDefaultTags({ service: "worker" });
  assert.deepEqual(contextSnapshot().tags, { service: "worker" });
  await withContext({}, async () => {
    setSession("scoped");
    setTags({ customer: "acme" });
    assert.equal(contextSnapshot().sessionId, "scoped");
    assert.deepEqual(contextSnapshot().tags, { service: "worker", customer: "acme" });
  });
  assert.equal(contextSnapshot().sessionId, undefined);
  assert.deepEqual(contextSnapshot().tags, { service: "worker" });
  setDefaultTags({});
});
