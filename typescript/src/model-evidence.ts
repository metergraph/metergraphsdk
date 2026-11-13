/** Content-free native facts; receivers own outcome classification. */
function get(value: unknown, key: string): unknown {
  return value && typeof value === "object" ? (value as Record<string, unknown>)[key] : undefined;
}
function items(value: unknown): unknown[] { return Array.isArray(value) ? value : []; }
function text(value: unknown): value is string { return typeof value === "string" && value.trim().length > 0 && value.length <= 256; }
function inspect(value: unknown) {
  const reasons: string[] = [], coverage: boolean[] = [];
  let positive = false, received = false;
  const reason = (raw: unknown) => { if (text(raw)) reasons.push(raw); };
  reason(get(value, "finish_reason")); reason(get(value, "stop_reason"));
  const camelFinish = get(value, "finishReason");
  reason(get(camelFinish, "raw") ?? get(camelFinish, "unified") ?? camelFinish);
  if (["refusal", "response.refusal.delta", "response.refusal.done"].includes(String(get(value, "type")))) positive = true;
  for (const part of [get(value, "part"), get(value, "content_block")]) positive ||= get(part, "type") === "refusal";
  const delta = get(value, "delta");
  reason(get(delta, "stop_reason")); positive ||= Boolean(get(delta, "refusal"));
  const choices = get(value, "choices");
  if (Array.isArray(choices)) {
    received = true;
    for (const choice of choices) {
      const finish = get(choice, "finish_reason"), message = get(choice, "message");
      reason(finish);
      positive ||= Boolean(get(message, "refusal")) || Boolean(get(get(choice, "delta"), "refusal")) || items(get(message, "content")).some(part => get(part, "type") === "refusal");
      coverage.push(message != null && text(finish) && (get(message, "content") != null || get(message, "tool_calls") != null || typeof message === "object" && "refusal" in message));
    }
  }
  const content = get(value, "content");
  if (Array.isArray(content)) {
    positive ||= content.some(part => get(part, "type") === "refusal");
    if (text(get(value, "stop_reason"))) {
      received = true;
      coverage.push(content.length > 0 && content.every(part => ["text", "thinking", "redacted_thinking", "tool_use", "refusal"].includes(String(get(part, "type")))));
    }
  }
  const output = get(value, "output");
  if (Array.isArray(output)) {
    received = true;
    for (const item of output) {
      const parts = items(get(item, "content"));
      positive ||= get(item, "type") === "refusal" || parts.some(part => get(part, "type") === "refusal");
      coverage.push(["completed", "incomplete"].includes(String(get(value, "status"))) && (
        get(item, "type") === "message" && parts.length > 0 && parts.every(part => ["output_text", "refusal"].includes(String(get(part, "type")))) ||
        ["function_call", "reasoning"].includes(String(get(item, "type")))));
    }
  }
  reason(get(get(value, "incomplete_details"), "reason"));
  const feedback = get(value, "promptFeedback") ?? get(value, "prompt_feedback");
  if (feedback != null) { received = true; reason(get(feedback, "blockReason") ?? get(feedback, "block_reason")); }
  const candidates = get(value, "candidates");
  if (Array.isArray(candidates)) {
    received = true;
    for (const candidate of candidates) {
      const finish = get(candidate, "finishReason") ?? get(candidate, "finish_reason");
      reason(finish);
      const parts = items(get(get(candidate, "content"), "parts"));
      positive ||= parts.some(part => get(part, "type") === "refusal");
      coverage.push(text(finish) && parts.length > 0);
    }
  }
  return { positive, complete: coverage.length > 0 && coverage.every(Boolean), received, reasons };
}
export function modelEvidence(input: {
  response?: unknown; error?: unknown; stream_chunks?: unknown[]; stream_complete?: boolean;
  refusalSignal?: boolean;
}) {
  const { response, error } = input;
  const raw = get(error, "status_code") ?? get(error, "status") ?? get(error, "statusCode")
    ?? get(get(error, "response"), "status_code") ?? get(response, "status_code");
  const status = typeof raw === "number" && Number.isInteger(raw) && raw >= 100 && raw <= 599 ? raw : null;
  let payload = get(error, "error") ?? get(error, "body") ?? error;
  payload = get(payload, "error") ?? payload;
  const code = get(payload, "code") ?? get(error, "code");
  const safeInspect = (value: unknown) => {
    try { return inspect(value); } catch { return { positive: false, complete: false, received: false, reasons: [] as string[] }; }
  };
  const native = safeInspect(response);
  for (const chunk of input.stream_chunks ?? []) {
    for (const value of [chunk, get(chunk, "response")]) {
      const inspected = safeInspect(value);
      native.positive ||= inspected.positive;
      native.received ||= inspected.received;
      native.reasons.push(...inspected.reasons);
    }
  }
  const signal = native.positive ? true : typeof input.refusalSignal === "boolean" ? input.refusalSignal :
    native.complete && (input.stream_chunks === undefined || input.stream_complete) ? false : null;
  return {
    http_status_code: status, error_code: text(code) ? code : null,
    refusal_signal: signal, response_received: native.received || native.positive ? true : null,
    ...(native.reasons.length ? { otel: { finish_reasons: native.reasons } } : {}),
  };
}
