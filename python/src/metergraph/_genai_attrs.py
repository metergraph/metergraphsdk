"""Cross-dialect GenAI span-attribute translation.

Turns a plain span-attribute mapping into the exact (request dict, response
dict, metadata) shapes ``MetergraphGenAIExporter._export_span`` builds, so the
capture layer re-derives usage and finish_reason the same way for every
telemetry dialect.

This module ships all four dialects Metergraph understands: the OpenTelemetry
``gen_ai.*`` semantic conventions, the Langfuse SDK, LangSmith, and
OpenInference (Arize Phoenix). When a span carries more than one dialect,
every field falls back per-field in the order ``langfuse.*`` > ``langsmith.*``
> ``gen_ai.*`` > OpenInference, which is why the merge exists before there is
anything to merge.

LangSmith is the one dialect that is not its own attribute namespace: it
writes the ``gen_ai.*`` conventions and adds a ``langsmith.*`` sidecar, so its
extractor reads the gen_ai spellings the semantic conventions dropped rather
than a vocabulary of its own.

This module is a stdlib-only leaf: it must not import ``_capture`` or
``opentelemetry``. Usage keys emitted here are deliberately spelled in the
alias vocabulary ``_capture._usage`` already accepts.
"""

from __future__ import annotations

import ast
import enum
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# The merge order is fixed here, ahead of the dialects that populate it, so
# adding one is a new extractor rather than a reshuffle of existing behavior.
DIALECT_LANGFUSE = "langfuse"
DIALECT_GENAI = "gen_ai"
DIALECT_OPENINFERENCE = "openinference"
DIALECT_LANGSMITH = "langsmith"


class SkipReason(enum.Enum):
    """Why a span produced no mapped call."""

    NOT_GENAI = "not-genai"
    INELIGIBLE_KIND = "ineligible-kind"
    NO_MODEL = "no-model"


@dataclass(frozen=True)
class MappedCall:
    """Normalized shapes for one LLM call, ready for the capture layer."""

    provider: str | None
    model: str
    operation: str
    request: dict[str, Any]
    response: dict[str, Any]
    response_text: str | None
    cost: float | None
    cost_source: str | None
    session_id: str | None
    trace_name: str | None
    completion_start_time: str | None
    error_message: str | None
    dialects: tuple[str, ...]
    parse_degraded: bool
    usage_absent: bool
    dropped_usage_keys: tuple[str, ...]
    # Attribution carried by dialects that record it (today: Langfuse).
    user_id: str | None = None
    environment: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    # An explicit workload name the dialect recorded (a Langfuse prompt name).
    route_name: str | None = None
    error_type: str | None = None


# Langfuse usage_details keys, grouped by the bucket they count. Langfuse's
# buckets are disjoint ("input" excludes cached tokens, "output" excludes
# reasoning), which is how its server prices them; the spellings are the ones
# its own integrations and default model prices use. The first key present in
# a group wins, so two spellings of one bucket are never summed.
_LANGFUSE_INPUT_KEYS = ("input", "input_tokens")
_LANGFUSE_OUTPUT_KEYS = ("output", "output_tokens")
_LANGFUSE_CACHE_READ_KEYS = (
    "input_cached_tokens",
    "input_cache_read",
    "cache_read_input_tokens",
    "cache_read_tokens",
)
_LANGFUSE_CACHE_WRITE_KEYS = (
    "input_cache_creation",
    "cache_creation_input_tokens",
    "input_cache_write_tokens",
    "cache_write_tokens",
    "input_cache_write",
)
_LANGFUSE_CACHE_WRITE_5M_KEYS = ("input_cache_creation_5m",)
_LANGFUSE_CACHE_WRITE_1H_KEYS = ("input_cache_creation_1h",)
_LANGFUSE_REASONING_KEYS = ("output_reasoning_tokens", "output_reasoning")
_LANGFUSE_USAGE_IGNORED = frozenset({"total", "total_tokens"})
# Keys only a provider's own usage object uses (OpenAI chat, Gemini), never
# Langfuse's buckets: their presence means the object passes through as is.
_PROVIDER_USAGE_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "prompt_token_count",
        "promptTokenCount",
        "candidates_token_count",
        "candidatesTokenCount",
        "cached_content_token_count",
        "cachedContentTokenCount",
        "thoughts_token_count",
        "thoughtsTokenCount",
        "cacheCreationInputTokens",
    }
)

# Providers whose native input count already includes cache reads, so a row
# in their native convention restates Langfuse's uncached "input" as
# input + cache read. Mirrors the hosted catalog's (publisher, channel) list;
# every other provider, unknown ones included, counts input without cache.
_INPUT_INCLUDES_CACHE_READ = frozenset({"openai", "google", "deepseek", "xai"})

# LangChain's standard token-detail keys, as LangSmith forwards them in
# gen_ai.usage.{input,output}_token_details. "audio" has no home in the alias
# vocabulary and is dropped rather than guessed at.
_LANGSMITH_TOKEN_DETAIL_RENAMES = {
    "cache_read": "cache_read_input_tokens",
    "cache_creation": "cache_creation_input_tokens",
    "reasoning": "reasoning_tokens",
}

# Provider values are emitted by several OpenTelemetry instrumentors before
# the semantic-convention provider vocabulary is stable. Keep this mapping
# deliberately small: unknown values remain visible instead of being guessed
# into a supported provider.
_PROVIDER_ALIASES = {
    "amazon.bedrock": "bedrock",
    "aws": "bedrock",
    "aws.bedrock": "bedrock",
    "aws.bedrock.converse": "bedrock",
    "aws_bedrock": "bedrock",
    "bedrock": "bedrock",
    "azure": "azure",
    "azure.ai.inference": "azure",
    "azure.openai": "azure",
    "azure.openai.chat": "azure",
    "azure_openai": "azure",
}


def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _first_count(
    decoded: Mapping[str, Any], keys: Sequence[str], used: set[str]
) -> int | None:
    for key in keys:
        if key in decoded:
            used.add(key)
            count = _count(decoded[key])
            if count is not None:
                return count
    return None


def _is_provider_usage(decoded: Mapping[str, Any]) -> bool:
    """Whether usage_details is a provider's own usage object, not Langfuse's.

    ``langfuse.openai`` records the OpenAI response's usage verbatim:
    ``prompt_tokens`` (which includes cached tokens) with nested
    ``prompt_tokens_details``. That shape is already in the convention the
    capture layer reads, so it passes through instead of being flattened,
    which used to drop every nested cached and reasoning count.
    """
    return any(key in decoded for key in _PROVIDER_USAGE_KEYS) or any(
        isinstance(value, Mapping) for value in decoded.values()
    )


def map_usage_details(
    decoded: Mapping[str, Any], provider: str | None = None
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Map Langfuse usage details onto the ``_usage`` alias vocabulary.

    Returns ``(usage, dropped_keys)``: usage in ``provider``'s native
    convention, plus the sorted source keys that did not translate. A
    provider's own usage object passes through unchanged. Langfuse's disjoint
    buckets are restated: output counts reasoning (every catalog prices
    reasoning at the output rate), and input counts cache reads for providers
    whose native input does. The derived ``"total"`` is skipped silently.
    """
    if _is_provider_usage(decoded):
        return {
            key: value
            for key, value in decoded.items()
            if key not in _LANGFUSE_USAGE_IGNORED
        }, ()
    used: set[str] = set()
    input_tokens = _first_count(decoded, _LANGFUSE_INPUT_KEYS, used)
    output_tokens = _first_count(decoded, _LANGFUSE_OUTPUT_KEYS, used)
    cache_read = _first_count(decoded, _LANGFUSE_CACHE_READ_KEYS, used)
    cache_write = _first_count(decoded, _LANGFUSE_CACHE_WRITE_KEYS, used)
    cache_write_5m = _first_count(decoded, _LANGFUSE_CACHE_WRITE_5M_KEYS, used)
    cache_write_1h = _first_count(decoded, _LANGFUSE_CACHE_WRITE_1H_KEYS, used)
    reasoning = _first_count(decoded, _LANGFUSE_REASONING_KEYS, used)

    usage: dict[str, Any] = {}
    if input_tokens is not None or cache_read:
        if (provider or "").lower() in _INPUT_INCLUDES_CACHE_READ:
            usage["input_tokens"] = (input_tokens or 0) + (cache_read or 0)
        elif input_tokens is not None:
            usage["input_tokens"] = input_tokens
    if output_tokens is not None or reasoning:
        usage["output_tokens"] = (output_tokens or 0) + (reasoning or 0)
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write is None and (cache_write_5m or cache_write_1h):
        cache_write = (cache_write_5m or 0) + (cache_write_1h or 0)
    if cache_write is not None:
        usage["cache_creation_input_tokens"] = cache_write
    if cache_write_5m is not None or cache_write_1h is not None:
        usage["cache_creation"] = {
            "ephemeral_5m_input_tokens": cache_write_5m,
            "ephemeral_1h_input_tokens": cache_write_1h,
        }
    if reasoning is not None:
        usage["reasoning_tokens"] = reasoning
    dropped = sorted(
        key
        for key in decoded
        if key not in used and key not in _LANGFUSE_USAGE_IGNORED
    )
    return usage, tuple(dropped)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _canonical_provider(value: Any) -> str | None:
    provider = _string(value)
    if provider is None:
        return None
    return _PROVIDER_ALIASES.get(provider.strip().lower(), provider)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_json(value: Any) -> tuple[Any, bool]:
    """Decode a JSON-string attribute. Returns (parsed, ok)."""
    if not isinstance(value, str):
        return None, False
    try:
        return json.loads(value), True
    except (TypeError, ValueError):
        return None, False


def _first_string(value: Any) -> str | None:
    if isinstance(value, str):
        decoded, ok = _load_json(value)
        if ok and isinstance(decoded, list) and decoded:
            return str(decoded[0])
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return str(value[0]) if value else None
    return None


def _text_from_output_messages(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    messages, ok = _load_json(value)
    if not ok or not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        texts: list[str] = []
        for part in parts:
            if (
                isinstance(part, Mapping)
                and part.get("type") == "text"
                and isinstance(part.get("content"), str)
            ):
                texts.append(part["content"])
        if texts:
            return "".join(texts)
    return None


def _genai_request_content(
    attributes: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Split gen_ai request content exactly like the exporter does today."""
    system = attributes.get("gen_ai.system_instructions")
    messages = attributes.get("gen_ai.input.messages")
    if not isinstance(messages, str):
        return system if isinstance(system, str) else None, None
    if isinstance(system, str):
        return system, messages
    decoded, ok = _load_json(messages)
    if not ok or not isinstance(decoded, list):
        return None, messages

    system_parts: list[Any] = []
    conversation: list[Any] = []
    for message in decoded:
        if isinstance(message, Mapping) and message.get("role") == "system":
            parts = message.get("parts")
            if isinstance(parts, list):
                system_parts.extend(parts)
            continue
        conversation.append(message)
    if not system_parts:
        return None, messages
    return (
        json.dumps(system_parts, separators=(",", ":"), ensure_ascii=False),
        json.dumps(conversation, separators=(",", ":"), ensure_ascii=False),
    )


@dataclass
class _Fields:
    """One dialect's contribution, before the per-field precedence merge."""

    model: str | None = None
    response_model: str | None = None
    provider: str | None = None
    operation: str | None = None
    request: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    response_text: str | None = None
    output_structure: Any = None
    cost: float | None = None
    cost_source: str | None = None
    session_id: str | None = None
    trace_name: str | None = None
    completion_start_time: str | None = None
    error_message: str | None = None
    parse_degraded: bool = False
    dropped_usage_keys: tuple[str, ...] = ()
    user_id: str | None = None
    environment: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    route_name: str | None = None
    error_type: str | None = None


def _extract_genai(attributes: Mapping[str, Any]) -> _Fields:
    fields = _Fields()
    fields.model = _string(attributes.get("gen_ai.request.model"))
    fields.provider = _canonical_provider(
        attributes.get("gen_ai.provider.name")
    ) or _canonical_provider(attributes.get("gen_ai.system"))
    fields.operation = _string(attributes.get("gen_ai.operation.name"))
    fields.response_model = _string(attributes.get("gen_ai.response.model"))
    system, messages = _genai_request_content(attributes)
    if system is not None:
        fields.request["system_instructions"] = system
    if messages is not None:
        fields.request["messages"] = messages
    input_tokens = attributes.get("gen_ai.usage.input_tokens")
    if input_tokens is None:
        input_tokens = attributes.get("gen_ai.usage.prompt_tokens")
    output_tokens = attributes.get("gen_ai.usage.output_tokens")
    if output_tokens is None:
        output_tokens = attributes.get("gen_ai.usage.completion_tokens")
    if input_tokens is not None:
        fields.usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        fields.usage["output_tokens"] = output_tokens
    fields.finish_reason = _first_string(
        attributes.get("gen_ai.response.finish_reasons")
    )
    fields.response_text = _text_from_output_messages(
        attributes.get("gen_ai.output.messages")
    )
    return fields


def _flattened_messages(
    attributes: Mapping[str, Any], prefix: str
) -> list[dict[str, Any]]:
    """Reassemble OpenInference index-flattened messages into dicts."""
    flat: dict[int, dict[str, Any]] = {}
    contents: dict[int, dict[int, dict[str, Any]]] = {}
    lead = prefix + "."
    for key, value in attributes.items():
        if not key.startswith(lead):
            continue
        parts = key[len(lead) :].split(".")
        if len(parts) < 3 or not parts[0].isdigit() or parts[1] != "message":
            continue
        index = int(parts[0])
        tail = parts[2:]
        if tail == ["role"] or tail == ["content"]:
            flat.setdefault(index, {})[tail[0]] = value
        elif (
            len(tail) == 4
            and tail[0] == "contents"
            and tail[1].isdigit()
            and tail[2] == "message_content"
        ):
            contents.setdefault(index, {}).setdefault(int(tail[1]), {})[
                tail[3]
            ] = value
    messages: list[dict[str, Any]] = []
    for index in sorted(set(flat) | set(contents)):
        message = dict(flat.get(index, {}))
        if "content" not in message and index in contents:
            parts_list = [contents[index][j] for j in sorted(contents[index])]
            texts = [
                part.get("text")
                for part in parts_list
                if part.get("type") == "text" and isinstance(part.get("text"), str)
            ]
            if texts and len(texts) == len(parts_list):
                message["content"] = "".join(texts)
            else:
                message["content"] = parts_list
        messages.append(message)
    return messages


def _message_text(message: Mapping[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            part.get("text")
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        if texts:
            return "".join(texts)
    return None


def _extract_openinference(attributes: Mapping[str, Any]) -> _Fields:
    fields = _Fields()
    fields.model = _string(attributes.get("llm.request.model_name")) or _string(
        attributes.get("llm.model_name")
    )
    fields.response_model = _string(attributes.get("llm.response.model_name"))
    fields.provider = _canonical_provider(
        attributes.get("llm.provider")
    ) or _canonical_provider(attributes.get("llm.system"))
    fields.finish_reason = _string(attributes.get("llm.finish_reason"))

    usage_pairs = (
        ("llm.token_count.prompt", "input_tokens"),
        ("llm.token_count.completion", "output_tokens"),
        ("llm.token_count.prompt_details.cache_read", "cache_read_input_tokens"),
        (
            "llm.token_count.prompt_details.cache_write",
            "cache_creation_input_tokens",
        ),
        ("llm.token_count.completion_details.reasoning", "reasoning_tokens"),
    )
    for attr, canonical in usage_pairs:
        value = attributes.get(attr)
        if value is not None:
            fields.usage[canonical] = value

    cost = _number(attributes.get("llm.cost.total"))
    if cost is not None:
        fields.cost = cost
        fields.cost_source = "openinference.llm.cost.total"

    messages = _flattened_messages(attributes, "llm.input_messages")
    if messages:
        fields.request["messages"] = messages
    else:
        raw_input = attributes.get("input.value")
        if isinstance(raw_input, str):
            if attributes.get("input.mime_type") == "application/json":
                decoded, ok = _load_json(raw_input)
                if ok:
                    fields.request["input"] = decoded
                else:
                    fields.request["input"] = raw_input
                    fields.parse_degraded = True
            else:
                fields.request["input"] = raw_input

    parameters = attributes.get("llm.invocation_parameters")
    if isinstance(parameters, str):
        decoded, ok = _load_json(parameters)
        if ok and isinstance(decoded, Mapping):
            fields.request["parameters"] = dict(decoded)
        else:
            fields.parse_degraded = True

    output_messages = _flattened_messages(attributes, "llm.output_messages")
    if output_messages:
        for message in output_messages:
            text = _message_text(message)
            if text is not None:
                fields.response_text = text
                break
    else:
        raw_output = attributes.get("output.value")
        if isinstance(raw_output, str):
            if attributes.get("output.mime_type") == "application/json":
                decoded, ok = _load_json(raw_output)
                if not ok:
                    fields.response_text = raw_output
                    fields.parse_degraded = True
                elif isinstance(decoded, str):
                    fields.response_text = decoded
                else:
                    fields.output_structure = decoded
            else:
                fields.response_text = raw_output
    return fields


# Gateway-style "<provider>/<model>" prefixes (OpenRouter, LiteLLM, the
# Langfuse AI gateway) that name the provider outright.
_MODEL_PREFIX_PROVIDERS = {
    "anthropic": "anthropic",
    "azure": "azure",
    "bedrock": "bedrock",
    "deepseek": "deepseek",
    "gemini": "google",
    "google": "google",
    "groq": "groq",
    "mistral": "mistral",
    "mistralai": "mistral",
    "openai": "openai",
    "vertex_ai": "google",
    "x-ai": "xai",
    "xai": "xai",
}
# Bedrock model ids: "<vendor>.<model>", optionally behind a cross-region
# inference profile ("us.", "eu.", "apac.", "global.").
_BEDROCK_VENDORS = ("anthropic.", "amazon.", "meta.", "cohere.", "ai21.", "mistral.")
_BEDROCK_PROFILES = ("us.", "eu.", "apac.", "global.", "us-gov.", "ca.", "jp.", "au.")
# Bare model-name families whose provider is unambiguous.
_MODEL_NAME_PROVIDERS = (
    # Before OpenAI's broad "text-embedding-" prefix.
    (("text-embedding-004", "text-embedding-005", "text-multilingual-embedding", "textembedding-gecko"), "google"),
    (("gpt-", "chatgpt-", "o1", "o3", "o4", "text-embedding-", "davinci", "babbage", "dall-e", "whisper", "tts-", "omni-moderation", "codex-"), "openai"),
    (("claude",), "anthropic"),
    (("gemini",), "google"),
    (("mistral", "mixtral", "codestral", "magistral", "ministral", "pixtral", "open-mistral", "open-mixtral"), "mistral"),
    (("grok",), "xai"),
    (("deepseek",), "deepseek"),
    (("command", "embed-english", "embed-multilingual"), "cohere"),
)


def provider_from_model(model: str | None) -> str | None:
    """The provider a Langfuse model name identifies, or None when it does not.

    Langfuse prices by model name alone and records no provider, so this is
    the only provider evidence a manual or ``@observe`` generation carries.
    Deliberately conservative: a name it cannot place stays unresolved rather
    than being guessed into a priced provider.
    """
    if not model:
        return None
    name = model.strip().lower()
    prefix, slash, rest = name.partition("/")
    if slash and rest:
        return _MODEL_PREFIX_PROVIDERS.get(prefix)
    bare = name
    for profile in _BEDROCK_PROFILES:
        if bare.startswith(profile):
            bare = bare[len(profile) :]
            break
    if bare.startswith(_BEDROCK_VENDORS) and (bare != name or ":" in bare or "-v" in bare):
        return "bedrock"
    for prefixes, provider in _MODEL_NAME_PROVIDERS:
        if name.startswith(prefixes):
            return provider
    return None


# Status messages that open with an exception class name, as Langfuse
# integrations write them ("RateLimitError: 429 ...").
_ERROR_TYPE = re.compile(r"^\s*([A-Za-z_][\w.]*?(?:Error|Exception|Timeout|Exceeded))\b")


def error_type_from_message(message: str | None) -> str | None:
    if not message:
        return None
    match = _ERROR_TYPE.match(message)
    return match.group(1).rsplit(".", 1)[-1] if match else None


def _attribute_text(value: Any) -> str | None:
    """A Langfuse attribute value as tag text.

    Langfuse JSON-encodes metadata values, so a decoded string is the value
    itself and anything else stays compact JSON.
    """
    if isinstance(value, str):
        decoded, ok = _load_json(value)
        if ok and isinstance(decoded, str):
            return decoded
        if ok and decoded is not None:
            return json.dumps(decoded, separators=(",", ":"), ensure_ascii=False)
        return value if value else None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, Mapping)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)
    return None


def _langfuse_tags(attributes: Mapping[str, Any]) -> list[str]:
    raw = attributes.get("langfuse.trace.tags")
    if raw is None:
        raw = attributes.get("langfuse.tags")
    if isinstance(raw, str):
        decoded, ok = _load_json(raw)
        if ok and isinstance(decoded, list):
            raw = decoded
        else:
            raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return sorted({str(tag) for tag in raw if str(tag).strip()})


def _langfuse_metadata(attributes: Mapping[str, Any]) -> dict[str, str]:
    """Trace then observation metadata, one tag per top-level key.

    Langfuse writes each key as its own ``<prefix>.<key>`` attribute (v3+);
    a whole-object JSON value under the bare prefix is the older form.
    Observation metadata wins over trace metadata for the same key.
    """
    merged: dict[str, str] = {}
    for prefix in (
        "langfuse.metadata",
        "langfuse.trace.metadata",
        "langfuse.observation.metadata",
    ):
        whole = attributes.get(prefix)
        if isinstance(whole, str):
            decoded, ok = _load_json(whole)
            if ok and isinstance(decoded, Mapping):
                for key, value in decoded.items():
                    text = _attribute_text(
                        value if isinstance(value, str) else json.dumps(value)
                    )
                    if text is not None:
                        merged[str(key)] = text
        lead = prefix + "."
        for key, value in attributes.items():
            if key.startswith(lead) and len(key) > len(lead):
                text = _attribute_text(value)
                if text is not None:
                    merged[key[len(lead) :]] = text
    return merged


def _extract_langfuse(attributes: Mapping[str, Any]) -> _Fields:
    fields = _Fields()
    fields.model = _string(attributes.get("langfuse.observation.model.name"))
    if _string(attributes.get("langfuse.observation.type")) == "embedding":
        fields.operation = "embeddings"
    # Session and user are trace-level fields the Langfuse SDK spells WITHOUT
    # its own prefix (TRACE_SESSION_ID = "session.id", TRACE_USER_ID =
    # "user.id") in Python v3-v5 and JS v4+. Langfuse's server also accepts
    # the langfuse.-prefixed compat spellings, so they are read second.
    fields.session_id = _string(attributes.get("session.id")) or _string(
        attributes.get("langfuse.session.id")
    )
    fields.user_id = _string(attributes.get("user.id")) or _string(
        attributes.get("langfuse.user.id")
    )
    fields.trace_name = _string(attributes.get("langfuse.trace.name"))
    fields.environment = (
        _string(attributes.get("langfuse.environment"))
        or _string(attributes.get("deployment.environment.name"))
        or _string(attributes.get("deployment.environment"))
    )

    metadata = _langfuse_metadata(attributes)
    # LangChain's handler records the provider it called as ls_provider
    # metadata; that is better evidence than the model name.
    fields.provider = _canonical_provider(metadata.get("ls_provider")) or (
        provider_from_model(fields.model)
    )
    tags: dict[str, str] = {
        f"langfuse.metadata.{key}": value for key, value in metadata.items()
    }
    for tag in _langfuse_tags(attributes):
        tags[f"langfuse.tag.{tag}"] = "true"
    for attribute, tag in (
        ("langfuse.release", "langfuse.release"),
        ("langfuse.version", "langfuse.version"),
    ):
        value = _attribute_text(attributes.get(attribute))
        if value is not None:
            tags[tag] = value
    prompt_name = _string(attributes.get("langfuse.observation.prompt.name")) or _string(
        attributes.get("langfuse.prompt.name")
    )
    prompt_version = attributes.get("langfuse.observation.prompt.version")
    if prompt_version is None:
        prompt_version = attributes.get("langfuse.prompt.version")
    if prompt_name is not None:
        fields.route_name = prompt_name
        tags["langfuse.prompt.name"] = prompt_name
        version_text = _attribute_text(prompt_version)
        if version_text is not None:
            tags["langfuse.prompt.version"] = version_text
    fields.tags = tags

    completion_start = attributes.get("langfuse.observation.completion_start_time")
    if isinstance(completion_start, str):
        # Langfuse JSON-encodes this timestamp, so the attribute value arrives
        # wrapped in literal quote characters. Decode before it reaches the
        # exporter's ISO parser, which rejects the quoted form and would drop
        # time-to-first-token for every Langfuse span without a word.
        decoded, ok = _load_json(completion_start)
        fields.completion_start_time = (
            decoded if ok and isinstance(decoded, str) else completion_start
        )
    level = _string(attributes.get("langfuse.observation.level"))
    if level is not None and level.upper() == "ERROR":
        message = _string(attributes.get("langfuse.observation.status_message"))
        fields.error_message = message or "Langfuse observation reported an error"
        fields.error_type = error_type_from_message(message) or "LangfuseObservationError"

    parameters = attributes.get("langfuse.observation.model.parameters")
    if isinstance(parameters, str):
        decoded, ok = _load_json(parameters)
        if ok and isinstance(decoded, Mapping):
            fields.request["parameters"] = dict(decoded)
        else:
            fields.parse_degraded = True

    usage_details = attributes.get("langfuse.observation.usage_details")
    if isinstance(usage_details, str):
        decoded, ok = _load_json(usage_details)
        if ok and isinstance(decoded, Mapping):
            mapped_usage, fields.dropped_usage_keys = map_usage_details(
                decoded, fields.provider
            )
            fields.usage.update(mapped_usage)
        else:
            fields.parse_degraded = True

    cost_details = attributes.get("langfuse.observation.cost_details")
    if isinstance(cost_details, str):
        decoded, ok = _load_json(cost_details)
        if ok and isinstance(decoded, Mapping):
            total = _number(decoded.get("total"))
            if total is None:
                parts = [_number(value) for value in decoded.values()]
                if parts and all(part is not None for part in parts):
                    total = sum(parts)  # type: ignore[arg-type]
            if total is not None:
                fields.cost = total
                fields.cost_source = "langfuse.observation.cost_details.total"
        else:
            fields.parse_degraded = True

    raw_input = attributes.get("langfuse.observation.input")
    if isinstance(raw_input, str):
        decoded, ok = _load_json(raw_input)
        value = decoded if ok else raw_input
        if isinstance(value, list):
            fields.request["messages"] = value
        elif isinstance(value, Mapping) and isinstance(value.get("messages"), list):
            # langfuse.openai records the request kwargs; the conversation is
            # their messages list.
            fields.request["messages"] = value["messages"]
        else:
            fields.request["input"] = value

    raw_output = attributes.get("langfuse.observation.output")
    if isinstance(raw_output, str):
        decoded, ok = _load_json(raw_output)
        value = decoded if ok else raw_output
        if isinstance(value, str):
            fields.response_text = value
        elif isinstance(value, Mapping) and isinstance(value.get("content"), str):
            fields.response_text = value["content"]
        elif isinstance(value, Mapping) and isinstance(value.get("text"), str):
            fields.response_text = value["text"]
        else:
            fields.output_structure = value
    return fields


def _load_repr_mapping(value: Any) -> tuple[Any, bool]:
    """Decode an attribute holding a Python ``repr`` of a mapping.

    LangSmith stringifies token-detail dicts with ``str()``, not JSON, so the
    value arrives single-quoted -- ``"{'cache_read': 8}"`` -- and json.loads
    rejects it. Try JSON first anyway -- if upstream ever switches to a real
    JSON encoding this keeps working -- then fall back to literal_eval, which
    evaluates only Python literals and cannot execute code.
    """
    decoded, ok = _load_json(value)
    if ok:
        return decoded, True
    if not isinstance(value, str):
        return None, False
    try:
        return ast.literal_eval(value), True
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None, False


def _chat_completion_text(value: Any) -> str | None:
    """Pull the assistant text out of a provider chat-completion payload.

    LangSmith stores whatever the wrapped client returned, so for the common
    wrap_openai case gen_ai.completion is the raw response body rather than a
    string. Anything else keeps its structure and lands on response["output"].
    """
    if not isinstance(value, Mapping):
        return None
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
        return message["content"]
    return None


def _extract_langsmith(attributes: Mapping[str, Any]) -> _Fields:
    """Read the fields LangSmith spells differently, or spells alone.

    Model, provider, response model, finish reason and the flat token counts
    are plain gen_ai.* attributes that ``_extract_genai`` already reads. What
    is left is the content -- which LangSmith writes in the pre-1.37 gen_ai
    spellings the current conventions replaced -- the token-detail breakdown,
    and the two trace fields it puts under its own prefix.
    """
    fields = _Fields()
    fields.session_id = _string(attributes.get("langsmith.trace.session_id"))
    fields.trace_name = _string(attributes.get("langsmith.trace.name"))

    for attribute in ("input_token_details", "output_token_details"):
        raw = attributes.get(f"gen_ai.usage.{attribute}")
        if raw is None:
            continue
        decoded, ok = _load_repr_mapping(raw)
        if not ok or not isinstance(decoded, Mapping):
            fields.parse_degraded = True
            continue
        dropped: list[str] = []
        for key, value in decoded.items():
            canonical = _LANGSMITH_TOKEN_DETAIL_RENAMES.get(key)
            if canonical is None:
                dropped.append(key)
            else:
                fields.usage[canonical] = value
        fields.dropped_usage_keys = tuple(
            sorted({*fields.dropped_usage_keys, *dropped})
        )

    raw_input = attributes.get("gen_ai.prompt")
    if isinstance(raw_input, str):
        decoded, ok = _load_json(raw_input)
        if not ok:
            fields.request["input"] = raw_input
            fields.parse_degraded = True
        elif isinstance(decoded, Mapping) and isinstance(decoded.get("messages"), list):
            # A wrapped chat call: the run inputs are the request kwargs, so
            # the messages list is the conversation the exporter wants.
            fields.request["messages"] = decoded["messages"]
        else:
            fields.request["input"] = decoded

    raw_output = attributes.get("gen_ai.completion")
    if isinstance(raw_output, str):
        decoded, ok = _load_json(raw_output)
        if not ok:
            fields.response_text = raw_output
            fields.parse_degraded = True
        elif isinstance(decoded, str):
            fields.response_text = decoded
        else:
            text = _chat_completion_text(decoded)
            if text is not None:
                fields.response_text = text
            else:
                fields.output_structure = decoded
    return fields


def _detected_dialects(attributes: Mapping[str, Any]) -> dict[str, bool | None]:
    """Detected dialects mapped to their verdict on whether this is an LLM call.

    ``True`` claims the span, ``False`` denies it, and ``None`` means the
    dialect is present but carries no span-kind attribute to judge by. See
    ``_vetoed`` for how the verdicts combine.
    """
    detected: dict[str, bool | None] = {}
    if any(key.startswith("langfuse.observation.") for key in attributes):
        observation_type = _string(attributes.get("langfuse.observation.type"))
        # Generations and embeddings are Langfuse's cost-bearing types. Any
        # other type (agent, tool, chain, span...) counts only when it carries
        # a model and usage or cost, so structure spans stay out while a
        # custom-typed model call is not lost.
        detected[DIALECT_LANGFUSE] = observation_type in {
            "generation",
            "embedding",
        } or (
            attributes.get("langfuse.observation.model.name") is not None
            and (
                attributes.get("langfuse.observation.usage_details") is not None
                or attributes.get("langfuse.observation.cost_details") is not None
            )
        )
    if (
        attributes.get("gen_ai.request.model") is not None
        or attributes.get("gen_ai.operation.name") is not None
    ):
        # LangSmith exports gen_ai.* for chains, tools and retrievers, so its
        # sidecar kind must be allowed to veto the generic dialect. Other
        # producers use gen_ai.* as their complete eligibility signal.
        detected[DIALECT_GENAI] = (
            None
            if any(key.startswith("langsmith.") for key in attributes)
            else True
        )
    if any(key.startswith("langsmith.") for key in attributes):
        # LangSmith exports EVERY run as a gen_ai span -- chains, tools and
        # retrievers included -- and records the real kind here. Without this
        # verdict a chain span that inherits a model name maps as a second
        # complete call carrying its child's tokens.
        span_kind = _string(attributes.get("langsmith.span.kind"))
        detected[DIALECT_LANGSMITH] = (
            span_kind is not None and span_kind.lower() == "llm"
        )
    kind = attributes.get("openinference.span.kind")
    if kind is not None:
        detected[DIALECT_OPENINFERENCE] = isinstance(kind, str) and kind.upper() == "LLM"
    return detected


def _vetoed(detected: Mapping[str, bool | None]) -> bool:
    """Whether a dialect denies this span and no other dialect claims it.

    An abstaining dialect (``None``) does not decide by itself. A concrete
    veto with no concrete claim is decisive, while a bare span described only
    by abstaining dialects stays eligible.
    """
    verdicts = [verdict for verdict in detected.values() if verdict is not None]
    return bool(verdicts) and not any(verdicts)


def _eligible_dialects(detected: Mapping[str, bool | None]) -> list[str]:
    """Return dialects that did not veto, or none when a veto is decisive."""
    if any(verdict is False for verdict in detected.values()) and not any(
        verdict is True for verdict in detected.values()
    ):
        return []
    return [
        name
        for name in _PRECEDENCE
        if name in detected and detected[name] is not False
    ]


_EXTRACTORS = {
    DIALECT_LANGFUSE: _extract_langfuse,
    DIALECT_LANGSMITH: _extract_langsmith,
    DIALECT_GENAI: _extract_genai,
    DIALECT_OPENINFERENCE: _extract_openinference,
}
# Field-level precedence order: langfuse > langsmith > gen_ai > OpenInference.
_PRECEDENCE = (
    DIALECT_LANGFUSE,
    DIALECT_LANGSMITH,
    DIALECT_GENAI,
    DIALECT_OPENINFERENCE,
)


# LiteLLM's OpenTelemetry callback writes this on every call span it produces.
_LITELLM_CALL_ID = "litellm.call_id"
# LiteLLM's tracer, which names its spans after the caller's generation_name.
LITELLM_SCOPE = "litellm"
# Span names LiteLLM uses when the caller did not name the call.
_LITELLM_DEFAULT_SPAN_NAMES = frozenset({"litellm_request", "raw_gen_ai_request"})


def _is_litellm(attributes: Mapping[str, Any]) -> bool:
    return _LITELLM_CALL_ID in attributes


def litellm_route(span_name: Any, attributes: Mapping[str, Any]) -> str | None:
    """The caller's ``generation_name``, which LiteLLM writes as the span name.

    None for a LiteLLM default name: ``litellm_request``, ``raw_gen_ai_request``
    or the semantic-convention ``"<operation> <model>"``.
    """
    if not _is_litellm(attributes) or not isinstance(span_name, str):
        return None
    name = span_name.strip()
    if not name or name in _LITELLM_DEFAULT_SPAN_NAMES:
        return None
    operations = {"chat", _string(attributes.get("gen_ai.operation.name")) or "chat"}
    if any(name == op or name.startswith(op + " ") for op in operations):
        return None
    return name


def _indexed(attributes: Mapping[str, Any], prefix: str) -> dict[int, dict[str, Any]]:
    """Group ``<prefix>.<i>.<field>`` attributes by index."""
    grouped: dict[int, dict[str, Any]] = {}
    lead = prefix + "."
    for key, value in attributes.items():
        if not key.startswith(lead):
            continue
        index, separator, rest = key[len(lead) :].partition(".")
        if separator and index.isdigit() and rest:
            grouped.setdefault(int(index), {})[rest] = value
    return grouped


def _litellm_tools(attributes: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Request tools from ``llm.request.functions.<i>.*``, in OpenAI chat shape."""
    tools: list[dict[str, Any]] = []
    degraded = False
    for _, fields in sorted(_indexed(attributes, "llm.request.functions").items()):
        name = _string(fields.get("name"))
        if name is None:
            continue
        function: dict[str, Any] = {"name": name}
        description = fields.get("description")
        if isinstance(description, str):
            function["description"] = description
        parameters = fields.get("parameters")
        if isinstance(parameters, str):
            decoded, ok = _load_json(parameters)
            function["parameters"] = decoded if ok else parameters
            degraded = degraded or not ok
        tools.append({"type": "function", "function": function})
    return tools, degraded


def _output_message_tool_calls(attributes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Output tool calls from ``tool_call`` parts of ``gen_ai.output.messages``."""
    messages, ok = _load_json(attributes.get("gen_ai.output.messages"))
    if not ok or not isinstance(messages, list):
        return []
    calls: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("parts"), list):
            continue
        for part in message["parts"]:
            if not isinstance(part, Mapping) or part.get("type") != "tool_call":
                continue
            name = _string(part.get("name"))
            if name is None or not name.strip():
                continue
            arguments = part.get("arguments")
            calls.append(
                {
                    "id": str(part.get("id") or part.get("call_id") or f"call-{len(calls)}"),
                    "type": "function",
                    "function": {
                        "name": name.strip(),
                        "arguments": arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments if arguments is not None else {}),
                    },
                }
            )
    return calls


def _litellm_tool_calls(
    attributes: Mapping[str, Any], response_id: str | None
) -> list[dict[str, Any]]:
    """Output tool calls from ``gen_ai.completion.<i>.function_call`` or ``.tool_calls``.

    The legacy attributes carry no call id, so one is derived from the response
    id and completion index exactly as the MeterGraph OTLP ingest does.
    """
    prefix = response_id or "completion"
    calls: list[dict[str, Any]] = []

    def add(call_id: Any, name: Any, arguments: Any) -> None:
        name = _string(name)
        if name is None or not name.strip():
            return
        calls.append(
            {
                "id": str(call_id),
                "type": "function",
                "function": {
                    "name": name.strip(),
                    "arguments": arguments if isinstance(arguments, str) else "",
                },
            }
        )

    for index, fields in sorted(_indexed(attributes, "gen_ai.completion").items()):
        add(
            f"{prefix}-{index}",
            fields.get("function_call.name"),
            fields.get("function_call.arguments"),
        )
        for position, call in sorted(_indexed(fields, "tool_calls").items()):
            add(
                _string(call.get("id")) or f"{prefix}-{index}-{position}",
                call.get("name"),
                call.get("arguments"),
            )
    return calls


def _apply_litellm(
    attributes: Mapping[str, Any],
    request: dict[str, Any],
    response: dict[str, Any],
) -> bool:
    """Fill the request id, request tools and output tool calls from LiteLLM's
    non-standard attributes, only where the standard ones are absent.

    Returns whether any JSON was malformed.
    """
    response_id = _string(attributes.get("gen_ai.response.id"))
    if response_id is not None and not any(
        response.get(key) for key in ("_request_id", "response_id", "responseId", "id")
    ):
        response["id"] = response_id

    degraded = False
    if "tools" not in request and "gen_ai.tool.definitions" not in attributes:
        tools, degraded = _litellm_tools(attributes)
        if tools:
            request["tools"] = tools

    # The standard part carries the real call id; the legacy attributes do not.
    tool_calls = _output_message_tool_calls(attributes) or _litellm_tool_calls(
        attributes, response_id
    )
    if tool_calls:
        choices = response.get("choices") or [{}]
        response["choices"] = [
            {**choices[0], "message": {"role": "assistant", "tool_calls": tool_calls}},
            *choices[1:],
        ]
    return degraded


def _first_value(contributions: list[_Fields], name: str) -> Any:
    for contribution in contributions:
        value = getattr(contribution, name)
        if value is not None:
            return value
    return None


def map_span_attributes(
    attributes: Mapping[str, Any],
) -> MappedCall | SkipReason:
    """Map one span's attributes to capture shapes, or say why not."""
    detected = _detected_dialects(attributes)
    if not detected:
        return SkipReason.NOT_GENAI
    if _vetoed(detected):
        return SkipReason.INELIGIBLE_KIND

    eligible = _eligible_dialects(detected)
    if not eligible:
        return SkipReason.INELIGIBLE_KIND
    contributions = [
        _EXTRACTORS[name](attributes) for name in eligible
    ]
    model = _first_value(contributions, "model")
    if model is None:
        return SkipReason.NO_MODEL

    request: dict[str, Any] = {"model": model}
    usage: dict[str, Any] = {}
    for contribution in contributions:
        for key, value in contribution.request.items():
            request.setdefault(key, value)
        for key, value in contribution.usage.items():
            usage.setdefault(key, value)

    usage_absent = not any(value is not None for value in usage.values())
    reasoning_tokens = usage.pop("reasoning_tokens", None)
    final_usage: dict[str, Any] = {
        "input_tokens": usage.pop("input_tokens", None),
        "output_tokens": usage.pop("output_tokens", None),
    }
    final_usage.update(usage)
    if reasoning_tokens is not None:
        final_usage["completion_tokens_details"] = {
            "reasoning_tokens": reasoning_tokens
        }

    finish_reason = _first_value(contributions, "finish_reason")
    response: dict[str, Any] = {
        "model": _first_value(contributions, "response_model") or model,
        "usage": final_usage,
        "finish_reason": finish_reason,
        "choices": (
            [{"finish_reason": finish_reason}] if finish_reason is not None else []
        ),
    }
    response_text = _first_value(contributions, "response_text")
    output_structure = _first_value(contributions, "output_structure")
    if output_structure is not None and response_text is None:
        response["output"] = output_structure

    litellm_degraded = (
        _apply_litellm(attributes, request, response)
        if _is_litellm(attributes)
        else False
    )

    priced = next((c for c in contributions if c.cost is not None), None)

    dropped: set[str] = set()
    for contribution in contributions:
        dropped.update(contribution.dropped_usage_keys)

    return MappedCall(
        provider=_first_value(contributions, "provider"),
        model=model,
        operation=_first_value(contributions, "operation") or "inference",
        request=request,
        response=response,
        response_text=response_text,
        cost=priced.cost if priced is not None else None,
        cost_source=priced.cost_source if priced is not None else None,
        session_id=_first_value(contributions, "session_id"),
        trace_name=_first_value(contributions, "trace_name"),
        completion_start_time=_first_value(contributions, "completion_start_time"),
        error_message=_first_value(contributions, "error_message"),
        dialects=tuple(eligible),
        parse_degraded=litellm_degraded
        or any(contribution.parse_degraded for contribution in contributions),
        usage_absent=usage_absent,
        dropped_usage_keys=tuple(sorted(dropped)),
        user_id=_first_value(contributions, "user_id"),
        environment=_first_value(contributions, "environment"),
        tags={
            key: value
            for contribution in reversed(contributions)
            for key, value in contribution.tags.items()
        },
        route_name=_first_value(contributions, "route_name"),
        error_type=_first_value(contributions, "error_type"),
    )


__all__ = [
    "DIALECT_LANGFUSE",
    "LITELLM_SCOPE",
    "MappedCall",
    "SkipReason",
    "litellm_route",
    "map_span_attributes",
]
