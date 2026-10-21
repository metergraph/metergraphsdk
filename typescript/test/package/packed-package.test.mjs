import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import test from "node:test";

const execFileAsync = promisify(execFile);
const packageRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

const expectedRepository = "https://github.com/metergraph/metergraphsdk";

async function runNode(cwd, filename) {
  await execFileAsync(process.execPath, [filename], { cwd });
}

test("published tarball supports CommonJS, ESM, and mixed loading", async () => {
  const workspace = await mkdtemp(path.join(tmpdir(), "metergraph-package-"));
  const npmEnv = {
    ...process.env,
    npm_config_cache: path.join(workspace, "npm-cache"),
  };

  try {
    const { stdout } = await execFileAsync(
      "npm",
      ["pack", "--json", "--ignore-scripts", "--pack-destination", workspace],
      { cwd: packageRoot, env: npmEnv },
    );
    const [packResult] = JSON.parse(stdout);
    const packedFiles = new Set(packResult.files.map(({ path: file }) => file));

    for (const expected of [
      "dist/index.js",
      "dist/index.mjs",
      "dist/index.d.ts",
      "dist/package.json",
      "package.json",
    ]) {
      assert.ok(packedFiles.has(expected), `tarball is missing ${expected}`);
    }
    assert.ok(
      [...packedFiles].every((file) => !file.startsWith("src/")),
      "tarball must not publish TypeScript sources",
    );

    const tarball = path.join(workspace, packResult.filename);
    const unpacked = path.join(workspace, "unpacked");
    await mkdir(unpacked);
    await execFileAsync("tar", ["-xzf", tarball, "-C", unpacked]);
    const packedManifest = JSON.parse(
      await readFile(path.join(unpacked, "package", "package.json"), "utf8"),
    );
    assert.equal(packedManifest.version, "0.6.11");
    assert.equal(
      packedManifest.description,
      "Zero-dependency TypeScript SDK for function-level LLM cost attribution across OpenAI, Anthropic, Gemini, and Vercel AI SDK.",
    );
    assert.equal(packedManifest.author, "Metergraph");
    assert.deepEqual(packedManifest.repository, {
      type: "git",
      url: `git+${expectedRepository}.git`,
      directory: "typescript",
    });
    assert.deepEqual(packedManifest.bugs, { url: `${expectedRepository}/issues` });
    assert.equal(packedManifest.homepage, "https://www.metergraph.dev/");
    assert.deepEqual(packedManifest.keywords, [
      "llm",
      "llm-cost",
      "llm-observability",
      "cost-attribution",
      "ai-agents",
      "openai",
      "anthropic",
      "gemini",
      "vercel-ai-sdk",
      "opentelemetry",
      "self-hosted",
    ]);
    assert.deepEqual(packedManifest.dependencies ?? {}, {});

    const consumer = path.join(workspace, "consumer");
    await mkdir(consumer);
    await writeFile(
      path.join(consumer, "package.json"),
      JSON.stringify({ name: "metergraph-package-consumer", private: true }),
    );
    await execFileAsync(
      "npm",
      [
        "install",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
        "--omit=optional",
        "--legacy-peer-deps",
        "--offline",
        tarball,
      ],
      { cwd: consumer, env: npmEnv },
    );

    await writeFile(
      path.join(consumer, "commonjs.cjs"),
      `const assert = require("node:assert/strict");
const metergraph = require("metergraph");
assert.equal(typeof metergraph.init, "function");
assert.equal(typeof metergraph.wrap, "function");
assert.equal(typeof metergraph.vercelAISDKMiddleware, "function");
`,
    );
    await writeFile(
      path.join(consumer, "esm.mjs"),
      `import assert from "node:assert/strict";
import * as metergraph from "metergraph";
assert.equal(typeof metergraph.init, "function");
assert.equal(typeof metergraph.wrap, "function");
assert.equal(typeof metergraph.vercelAISDKMiddleware, "function");
`,
    );
    await writeFile(
      path.join(consumer, "mixed.mjs"),
      `import assert from "node:assert/strict";
import { createRequire } from "node:module";
import * as imported from "metergraph";
const required = createRequire(import.meta.url)("metergraph");
assert.strictEqual(required.init, imported.init);
assert.strictEqual(required.route, imported.route);
assert.deepEqual(Object.keys(required).sort(), Object.keys(imported).sort());
`,
    );

    await runNode(consumer, "commonjs.cjs");
    await runNode(consumer, "esm.mjs");
    await runNode(consumer, "mixed.mjs");

    const installedPackage = JSON.parse(
      await readFile(
        path.join(consumer, "node_modules", "metergraph", "package.json"),
        "utf8",
      ),
    );
    assert.equal(installedPackage.engines.node, ">=18");
    assert.equal(installedPackage.peerDependencies.openai, ">=4 <8");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});
