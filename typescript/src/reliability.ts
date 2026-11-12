/** Structural provider outcomes, kept in conformance with Core reliability.py. */
export interface ModelReliability {
  http_status_code: number | null;
  availability: "available" | "unavailable" | "excluded" | null;
  refused: boolean | null;
}
function get(value: unknown, key: string): unknown {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)[key]
    : undefined;
}
function items(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}
function reason(value: unknown): string {
  return typeof value === "string"
    ? value.toLowerCase().replaceAll("-", "_")
    : "";
}
const refusalReasons = new Set([
  "refusal",
  "content_filter",
  "safety",
  "recitation",
  "blocklist",
  "prohibited_content",
  "spii",
]);
const chatReasons = new Set([
  "stop",
  "length",
  "tool_calls",
  "function_call",
  "content_filter",
]);
const normalReasons = new Set([
  "stop",
  "length",
  "tool_calls",
  "tool_use",
  "max_tokens",
  "end_turn",
]);
function classifyNative(input: {
  provider?: string;
  response?: unknown;
  error?: unknown;
  http_status_code?: unknown;
  finish_reason?: unknown;
}): ModelReliability {
  const { response, error } = input;
  const raw =
    input.http_status_code ??
    get(error, "status_code") ??
    get(error, "status") ??
    get(error, "statusCode") ??
    get(get(error, "response"), "status_code") ??
    get(response, "status_code");
  const status =
    typeof raw === "number" && Number.isInteger(raw) && raw >= 100 && raw <= 599
      ? raw
      : null;
  const result: ModelReliability = {
    http_status_code: status,
    availability: null,
    refused: null,
  };
  const reasons = [
    input.finish_reason,
    get(response, "stop_reason"),
    get(response, "finish_reason"),
  ].map(reason);
  const complete: boolean[] = [];
  let known = false,
    refused = false;
  for (const choice of items(get(response, "choices"))) {
    const finish = reason(get(choice, "finish_reason"));
    reasons.push(finish);
    const message = get(choice, "message");
    complete.push(
      message != null &&
        chatReasons.has(finish) &&
        (get(message, "content") != null ||
          get(message, "tool_calls") != null ||
          (typeof message === "object" && "refusal" in message)),
    );
    if (chatReasons.has(finish)) known = true;
    if (get(message, "refusal")) refused = true;
  }
  const stop = reason(get(response, "stop_reason"));
  if (
    ["end_turn", "stop_sequence", "max_tokens", "tool_use", "refusal"].includes(
      stop,
    ) &&
    get(response, "content") != null
  ) {
    known = true;
    complete.push(true);
  }
  const output = get(response, "output");
  if (Array.isArray(output)) {
    for (const item of output) {
      if (get(item, "type") === "refusal") refused = true;
      if (
        items(get(item, "content")).some(
          (part) => get(part, "type") === "refusal",
        )
      )
        refused = true;
    }
    if (
      output.length &&
      ["completed", "incomplete"].includes(String(get(response, "status")))
    ) {
      const inspected = output.map((item) => {
        const kind = get(item, "type"),
          content = get(item, "content");
        return (
          (kind === "message" &&
            Array.isArray(content) &&
            content.length > 0 &&
            content.every((part) =>
              ["output_text", "refusal"].includes(String(get(part, "type"))),
            )) ||
          ["function_call", "reasoning"].includes(String(kind))
        );
      });
      complete.push(...inspected);
      known = inspected.some(Boolean);
    }
  }
  reasons.push(reason(get(get(response, "incomplete_details"), "reason")));
  const feedback =
    get(response, "promptFeedback") ?? get(response, "prompt_feedback");
  if (
    [
      "safety",
      "blocklist",
      "prohibited_content",
      "spii",
      "image_safety",
    ].includes(
      reason(get(feedback, "blockReason") ?? get(feedback, "block_reason")),
    )
  )
    refused = true;
  for (const candidate of items(get(response, "candidates"))) {
    const finish = reason(
      get(candidate, "finishReason") ?? get(candidate, "finish_reason"),
    );
    reasons.push(finish);
    if (
      [
        "stop",
        "max_tokens",
        "safety",
        "recitation",
        "blocklist",
        "prohibited_content",
        "spii",
      ].includes(finish)
    ) {
      known = true;
      const parts = get(get(candidate, "content"), "parts");
      complete.push(Array.isArray(parts) && parts.length > 0);
    } else complete.push(false);
  }
  if (reasons.some((finish) => refusalReasons.has(finish))) refused = true;
  let payload = get(error, "error") ?? get(error, "body") ?? error;
  payload = get(payload, "error") ?? payload;
  if (
    input.provider?.toLowerCase() === "meta" &&
    ((status === 400 && get(payload, "code") === "content_policy_violation") ||
      (status === 403 &&
        get(payload, "code") === "user_blocked" &&
        get(payload, "type") === "permission_error" &&
        String(get(payload, "message") ?? "")
          .toLowerCase()
          .includes("repeated policy violations")))
  )
    refused = true;
  if (refused) return { ...result, availability: "available", refused: true };
  if (status === 429 || (status !== null && status >= 500))
    return { ...result, availability: "unavailable" };
  if (status !== null && status >= 400)
    return { ...result, availability: "excluded" };
  const errorType = get(error, "name") ?? get(error, "type");
  if (
    [
      "APITimeoutError",
      "TimeoutError",
      "ReadTimeout",
      "ConnectTimeout",
    ].includes(String(errorType))
  )
    return { ...result, availability: "unavailable" };
  if (known || reasons.some((finish) => normalReasons.has(finish)))
    return {
      ...result,
      availability: "available",
      refused: complete.length && complete.every(Boolean) ? false : null,
    };
  if (error) return result;
  if (status !== null && status >= 200 && status < 300)
    result.availability = "available";
  return result;
}

export function classifyModelReliability(
  input: Parameters<typeof classifyNative>[0] & {
    stream_chunks?: unknown[];
    stream_complete?: boolean;
  },
): ModelReliability {
  const result = classifyNative(input);
  let available = false;
  for (const chunk of input.stream_chunks ?? []) {
    const native = classifyNative({
      provider: input.provider,
      response: chunk,
    });
    const nested = classifyNative({
      provider: input.provider,
      response: get(chunk, "response"),
    });
    const finish = reason(get(get(chunk, "delta"), "stop_reason"));
    const explicit =
      native.refused === true ||
      nested.refused === true ||
      get(get(chunk, "content_block"), "type") === "refusal" ||
      get(get(chunk, "part"), "type") === "refusal" ||
      ["response.refusal.delta", "response.refusal.done"].includes(
        String(get(chunk, "type")),
      ) ||
      ["refusal", "content_filter", "safety"].includes(finish) ||
      items(get(chunk, "choices")).some((choice) =>
        Boolean(get(get(choice, "delta"), "refusal")),
      );
    if (explicit)
      return { ...result, availability: "available", refused: true };
    available ||=
      native.availability === "available" ||
      nested.availability === "available" ||
      ["end_turn", "stop_sequence", "max_tokens", "tool_use"].includes(finish);
  }
  if (result.refused === true) return result;
  if (available && result.availability === null)
    return { ...result, availability: "available", refused: null };
  if (
    input.stream_chunks !== undefined &&
    !input.stream_complete &&
    result.refused === false
  )
    return { ...result, refused: null };
  return result;
}
