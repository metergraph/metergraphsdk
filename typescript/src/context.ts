import { AsyncLocalStorage } from "node:async_hooks";
import { randomBytes } from "node:crypto";

export interface CaptureContext {
  route?: string;
  /** "explicit" when the developer named the route, "derived" when filled in. */
  routeSource?: "explicit" | "derived";
  sessionId?: string;
  userId?: string;
  tags: Record<string, string>;
  unitName?: string;
  unitCount?: number;
  captureText?: boolean;
  refusalSignal?: boolean;
  funcName?: string;
  traceId?: string;
  traceName?: string;
  parentSpanId?: string;
}
export interface RouteOptions {
  unit?: string;
  unitCount?: number;
  tags?: Record<string, unknown>;
  captureText?: boolean;
}
export interface TraceOptions {
  traceId?: string;
  parentSpanId?: string;
  captureText?: boolean;
}
export interface ContextOptions {
  /** Scoped observation; an explicit native refusal takes precedence. */
  refusalSignal?: boolean;
  sessionId?: string;
  tags?: Record<string, unknown>;
  userId?: string;
  metadata?: Record<string, string>;
}

const storage = new AsyncLocalStorage<CaptureContext>();
let defaultTags: Record<string, string> = {};
let warnedSessionOutsideScope = false;
let warnedTagsOutsideScope = false;
let warnedUserOutsideScope = false;
let warnedAttribution = false;

const badText = /[\x00\uD800-\uDFFF]/u;
const credentialKey = /(?:accesskey|apikey|auth|authorization|bearer|clientsecret|cookie|credentials?|encryptionkey|jwt|passwd|password|privatekey|secretkey|secret|sessionkey|signingkey|token)$/;
// Python/hosted whitespace semantics, including U+0085 and excluding U+FEFF.
const whitespaceOnly = /^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/u;

function omitted(): void {
  if (!warnedAttribution) {
    warnedAttribution = true;
    console.warn("Metergraph omitted invalid or excess attribution; use bounded string identifiers and metadata.");
  }
}

function user(value: unknown): string | undefined {
  if (typeof value === "string" && [...value].length <= 512 && !whitespaceOnly.test(value) && !badText.test(value)) return value;
  if (value !== undefined) omitted();
  return undefined;
}

function tagValues(values: unknown, legacy = false): Record<string, string> {
  if (values == null) return {};
  if (typeof values !== "object" || Array.isArray(values)) { omitted(); return {}; }
  const entries: [string, string][] = [];
  for (const [key, raw] of Object.entries(values)) {
    let value = raw;
    if (legacy) {
      try { value = String(raw); } catch { omitted(); continue; }
    }
    if (key.length > 0 && [...key].length <= 200 && !badText.test(key)
      && !credentialKey.test(key.toLowerCase().replace(/[^a-z0-9]/g, ""))
      && typeof value === "string" && [...value].length <= 1000 && !badText.test(value)) entries.push([key, value]);
    else omitted();
  }
  return Object.fromEntries(entries);
}

function boundTags(values: Record<string, string>): Record<string, string> {
  const entries: [string, string][] = [];
  let size = 2;
  // UTF-8 byte order equals Unicode codepoint order for valid strings.
  for (const [key, value] of Object.entries(values).sort(([a], [b]) => Buffer.compare(Buffer.from(a), Buffer.from(b)))) {
    const added = Buffer.byteLength(JSON.stringify({ [key]: value })) - 2 + (entries.length ? 1 : 0);
    if (entries.length === 32 || size + added > 65536) { omitted(); break; }
    entries.push([key, value]);
    size += added;
  }
  return Object.fromEntries(entries);
}

function normalizeTags(tags: Record<string, unknown>): Record<string, string> {
  return tagValues(tags, true);
}

export function contextSnapshot(): CaptureContext {
  const value = storage.getStore() ?? { tags: defaultTags };
  return { ...value, tags: { ...value.tags } };
}

export function runWithContext<T>(context: CaptureContext, fn: () => T): T {
  return storage.run(context, fn);
}

export function withContext<T>(options: ContextOptions, fn: () => T): T {
  const parent = contextSnapshot();
  const child: CaptureContext = {
    ...parent,
    sessionId: options.sessionId ?? parent.sessionId,
    userId: options.userId === undefined ? parent.userId : user(options.userId) ?? parent.userId,
    refusalSignal: options.refusalSignal ?? parent.refusalSignal,
    tags: boundTags({ ...parent.tags, ...normalizeTags(options.tags ?? {}), ...tagValues(options.metadata) }),
  };
  return storage.run(child, fn);
}

export function withSession<T>(sessionId: string, fn: () => T): T {
  return withContext({ sessionId }, fn);
}

export function withTags<T>(tags: Record<string, unknown>, fn: () => T): T {
  return withContext({ tags }, fn);
}

export function setDefaultTags(tags: Record<string, unknown>): void {
  defaultTags = boundTags(normalizeTags(tags));
}

export async function route<T>(
  name: string,
  fn: () => T | Promise<T>,
  options: RouteOptions = {},
): Promise<T> {
  const parent = contextSnapshot();
  const child: CaptureContext = {
    ...parent,
    route: name,
    routeSource: "explicit",
    tags: boundTags({
      ...parent.tags,
      ...normalizeTags(options.tags ?? {}),
    }),
    unitName: options.unit ?? parent.unitName,
    unitCount: options.unit ? (options.unitCount ?? 1) : parent.unitCount,
    captureText: options.captureText ?? parent.captureText,
  };
  return storage.run(child, fn);
}

export function trace<T>(
  name: string,
  fn: () => T | Promise<T>,
  options: TraceOptions = {},
): T | Promise<T> {
  const parent = contextSnapshot();
  const requested = options.traceId?.trim();
  const reuse = parent.traceId !== undefined
    && (requested === undefined || requested === parent.traceId);
  const child: CaptureContext = {
    ...parent,
    traceId: reuse ? parent.traceId : requested || randomBytes(16).toString("hex"),
    traceName: reuse ? parent.traceName : String(name),
    parentSpanId: options.parentSpanId ?? (reuse ? parent.parentSpanId : undefined),
    captureText: options.captureText ?? parent.captureText,
  };
  return storage.run(child, fn);
}

export function setUser(userId?: string): void {
  const store = storage.getStore();
  if (!store) {
    if (!warnedUserOutsideScope) {
      warnedUserOutsideScope = true;
      console.warn("Metergraph setUser() requires an active Metergraph context; use withContext().");
    }
    return;
  }
  const value = user(userId);
  if (userId === undefined || value !== undefined) storage.enterWith({ ...store, userId: value, tags: { ...store.tags } });
}

export function setSession(sessionId?: string): void {
  const store = storage.getStore();
  if (!store) {
    if (!warnedSessionOutsideScope) {
      warnedSessionOutsideScope = true;
      console.warn(
        "Metergraph setSession() requires an active Metergraph context; use withSession() or withContext().",
      );
    }
    return;
  }
  storage.enterWith({ ...store, sessionId, tags: { ...store.tags } });
}

export function setTags(tags: Record<string, unknown>): void {
  const store = storage.getStore();
  if (!store) {
    if (!warnedTagsOutsideScope) {
      warnedTagsOutsideScope = true;
      console.warn(
        "Metergraph setTags() requires an active Metergraph context; use withTags() or withContext().",
      );
    }
    return;
  }
  storage.enterWith({ ...store, tags: boundTags({ ...store.tags, ...normalizeTags(tags) }) });
}
