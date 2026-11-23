from __future__ import annotations

import json

import pytest

from metergraph._capture import _usage
from metergraph._genai_attrs import (
    DIALECT_GENAI,
    MappedCall,
    SkipReason,
    _vetoed,
    map_span_attributes,
)


def test_eligible_span_without_model_is_skipped_with_reason():
    assert (
        map_span_attributes({"gen_ai.operation.name": "chat"})
        is SkipReason.NO_MODEL
    )


def test_empty_usage_yields_usage_absent_record():
    mapped = map_span_attributes(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-5-mini",
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.usage_absent is True
    assert mapped.response["usage"] == {"input_tokens": None, "output_tokens": None}


def test_gen_ai_span_maps_to_the_exact_exporter_shapes():
    system = json.dumps(
        [{"type": "text", "content": "Answer with synthetic data only."}]
    )
    messages = json.dumps(
        [
            {
                "role": "user",
                "parts": [{"type": "text", "content": "First synthetic input"}],
            },
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "Earlier synthetic reply"}],
            },
        ]
    )
    output = json.dumps(
        [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "Synthetic result"}],
            }
        ]
    )
    mapped = map_span_attributes(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "anthropic",
            "gen_ai.request.model": "claude-opus-5",
            "gen_ai.response.model": "claude-opus-5-20260801",
            "gen_ai.system_instructions": system,
            "gen_ai.input.messages": messages,
            "gen_ai.output.messages": output,
            "gen_ai.usage.input_tokens": 41,
            "gen_ai.usage.output_tokens": 7,
            "gen_ai.response.finish_reasons": ("stop",),
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.provider == "anthropic"
    assert mapped.operation == "chat"
    # Golden comparison with what MetergraphGenAIExporter._export_span builds.
    assert mapped.request == {
        "model": "claude-opus-5",
        "system_instructions": system,
        "messages": messages,
    }
    assert mapped.response == {
        "model": "claude-opus-5-20260801",
        "usage": {"input_tokens": 41, "output_tokens": 7},
        "finish_reason": "stop",
        "choices": [{"finish_reason": "stop"}],
    }
    assert mapped.response_text == "Synthetic result"


def test_gen_ai_system_role_split_matches_exporter_behavior():
    messages = json.dumps(
        [
            {
                "role": "system",
                "parts": [{"type": "text", "content": "Synthetic system"}],
            },
            {
                "role": "user",
                "parts": [{"type": "text", "content": "First"}],
            },
        ]
    )
    mapped = map_span_attributes(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-5-mini",
            "gen_ai.input.messages": messages,
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.provider == "openai"
    assert json.loads(mapped.request["system_instructions"]) == [
        {"type": "text", "content": "Synthetic system"}
    ]
    assert [
        message["role"] for message in json.loads(mapped.request["messages"])
    ] == ["user"]
    # No response model on the span: the exporter falls back to the request
    # model in the response envelope.
    assert mapped.response["model"] == "gpt-5-mini"


def test_gen_ai_accepts_legacy_prompt_and_completion_token_spellings():
    mapped = map_span_attributes(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-5-mini",
            "gen_ai.usage.prompt_tokens": 33,
            "gen_ai.usage.completion_tokens": 4,
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.response["usage"] == {"input_tokens": 33, "output_tokens": 4}


def test_bedrock_provider_spellings_are_canonicalized_without_guessing_unknowns():
    for provider in (
        "aws.bedrock",
        "aws",
        "AWS.Bedrock",
        "aws.bedrock.converse",
        "amazon.bedrock",
        "aws_bedrock",
        "bedrock",
    ):
        mapped = map_span_attributes(
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": provider,
                "gen_ai.request.model": "amazon.nova-lite-v1:0",
                "gen_ai.usage.input_tokens": 12,
                "gen_ai.usage.output_tokens": 4,
            }
        )
        assert isinstance(mapped, MappedCall)
        assert mapped.provider == "bedrock"
        assert mapped.operation == "chat"
        assert mapped.response["usage"] == {
            "input_tokens": 12,
            "output_tokens": 4,
        }

    mapped = map_span_attributes(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "future.provider",
            "gen_ai.request.model": "future-model",
        }
    )
    assert isinstance(mapped, MappedCall)
    assert mapped.provider == "future.provider"


def test_azure_provider_spellings_are_canonicalized_for_openai_compatible_paths():
    for provider in (
        "azure.openai",
        "Azure.OpenAI",
        "azure.openai.chat",
        "azure.ai.inference",
        "azure_openai",
    ):
        mapped = map_span_attributes(
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.system": provider,
                "gen_ai.request.model": "gpt-4o-deployment",
                "gen_ai.usage.input_tokens": 8,
                "gen_ai.usage.output_tokens": 3,
            }
        )
        assert isinstance(mapped, MappedCall)
        assert mapped.provider == "azure"
        assert mapped.model == "gpt-4o-deployment"


def test_langfuse_generation_parses_json_string_attributes():
    mapped = map_span_attributes(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": "claude-opus-5",
            "langfuse.observation.model.parameters": json.dumps(
                {"temperature": 0.2, "max_tokens": 512}
            ),
            "langfuse.observation.usage_details": json.dumps(
                {"input": 41, "output": 7, "total": 48, "cache_read_input_tokens": 16}
            ),
            "langfuse.observation.cost_details": json.dumps(
                {"total": 0.0042, "input": 0.003, "output": 0.0012}
            ),
            "langfuse.observation.input": json.dumps(
                [{"role": "user", "content": "Synthetic input"}]
            ),
            "langfuse.observation.output": json.dumps(
                {"role": "assistant", "content": "Synthetic result"}
            ),
            "langfuse.observation.completion_start_time": "2026-08-28T00:00:00+00:00",
            "langfuse.trace.name": "synthetic-trace",
            "user.id": "synthetic-user",
            "session.id": "synthetic-session",
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.dialects == ("langfuse",)
    assert mapped.model == "claude-opus-5"
    # Langfuse records no provider; the model name places it.
    assert mapped.provider == "anthropic"
    assert mapped.user_id == "synthetic-user"
    # Parsed structures, not raw JSON strings, so downstream scrub/redact can
    # walk the keys.
    assert mapped.request["messages"] == [
        {"role": "user", "content": "Synthetic input"}
    ]
    assert mapped.request["parameters"] == {"temperature": 0.2, "max_tokens": 512}
    assert mapped.response_text == "Synthetic result"
    assert mapped.cost == 0.0042
    assert mapped.cost_source == "langfuse.observation.cost_details.total"
    assert mapped.session_id == "synthetic-session"
    assert mapped.trace_name == "synthetic-trace"
    assert mapped.completion_start_time == "2026-08-28T00:00:00+00:00"
    assert mapped.parse_degraded is False
    assert mapped.usage_absent is False
    assert mapped.dropped_usage_keys == ()
    usage = _usage(mapped.response)
    assert usage["input_tokens"] == 41
    assert usage["output_tokens"] == 7
    assert usage["cache_read_tokens"] == 16


def test_langfuse_malformed_json_degrades_to_metadata_only():
    mapped = map_span_attributes(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": "claude-opus-5",
            "langfuse.observation.model.parameters": "{not json",
            "langfuse.observation.usage_details": "{broken",
            "langfuse.observation.cost_details": "[",
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.model == "claude-opus-5"
    assert mapped.parse_degraded is True
    assert mapped.usage_absent is True
    assert mapped.cost is None
    assert mapped.cost_source is None
    assert "parameters" not in mapped.request


def test_langfuse_bare_string_input_stays_a_string():
    mapped = map_span_attributes(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": "claude-opus-5",
            "langfuse.observation.input": "plain synthetic prompt",
            "langfuse.observation.output": "plain synthetic answer",
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.parse_degraded is False
    assert mapped.request["input"] == "plain synthetic prompt"
    assert mapped.response_text == "plain synthetic answer"


def test_non_llm_kinds_are_skipped_with_reason():
    for kind in ("CHAIN", "RETRIEVER", "TOOL", "AGENT", "EMBEDDING"):
        result = map_span_attributes(
            {"openinference.span.kind": kind, "llm.model_name": "gpt-5-mini"}
        )
        assert result is SkipReason.INELIGIBLE_KIND

    assert (
        map_span_attributes(
            {
                "langfuse.observation.type": "span",
                "langfuse.observation.model.name": "claude-opus-5",
            }
        )
        is SkipReason.INELIGIBLE_KIND
    )
    assert (
        map_span_attributes({"http.request.method": "GET"}) is SkipReason.NOT_GENAI
    )


def test_total_only_usage_yields_usage_absent_record():
    mapped = map_span_attributes(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": "claude-opus-5",
            "langfuse.observation.usage_details": json.dumps({"total": 48}),
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.usage_absent is True
    assert mapped.dropped_usage_keys == ()
    usage = _usage(mapped.response)
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None


def test_unrecognized_usage_detail_keys_are_reported_dropped():
    mapped = map_span_attributes(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": "claude-opus-5",
            "langfuse.observation.usage_details": json.dumps(
                {"input": 10, "output": 2, "input_audio_tokens": 4, "custom_metric": 9}
            ),
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.usage_absent is False
    assert mapped.dropped_usage_keys == ("custom_metric", "input_audio_tokens")
    assert "custom_metric" not in mapped.response["usage"]
    assert "input_audio_tokens" not in mapped.response["usage"]


def test_langfuse_error_level_maps_to_error_message():
    mapped = map_span_attributes(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": "claude-opus-5",
            "langfuse.observation.level": "ERROR",
            "langfuse.observation.status_message": "synthetic failure",
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.error_message == "synthetic failure"


def test_openinference_full_token_details_use_usage_alias_spellings():
    mapped = map_span_attributes(
        {
            "openinference.span.kind": "LLM",
            "llm.model_name": "gpt-5-mini",
            "llm.provider": "openai",
            "llm.token_count.prompt": 120,
            "llm.token_count.completion": 30,
            "llm.token_count.total": 150,
            "llm.token_count.prompt_details.cache_read": 64,
            "llm.token_count.prompt_details.cache_write": 16,
            "llm.token_count.completion_details.reasoning": 8,
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "Synthetic input",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Synthetic result",
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.provider == "openai"
    assert mapped.model == "gpt-5-mini"
    assert mapped.dialects == ("openinference",)
    assert mapped.request["messages"] == [
        {"role": "user", "content": "Synthetic input"}
    ]
    assert mapped.response_text == "Synthetic result"
    assert mapped.usage_absent is False
    # The response usage block must speak the exact alias vocabulary the
    # capture layer already understands.
    usage = _usage(mapped.response)
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 30
    assert usage["cache_read_tokens"] == 64
    assert usage["cache_write_tokens"] == 16
    assert usage["reasoning_tokens"] == 8


def test_openinference_prefers_request_and_response_model_names():
    mapped = map_span_attributes(
        {
            "openinference.span.kind": "LLM",
            "llm.model_name": "legacy-name",
            "llm.request.model_name": "gpt-5-mini",
            "llm.response.model_name": "gpt-5-mini-2026-01-01",
        }
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.model == "gpt-5-mini"
    assert mapped.response["model"] == "gpt-5-mini-2026-01-01"


def test_openinference_falls_back_to_llm_model_name():
    mapped = map_span_attributes(
        {"openinference.span.kind": "LLM", "llm.model_name": "legacy-name"}
    )

    assert isinstance(mapped, MappedCall)
    assert mapped.model == "legacy-name"
    assert mapped.response["model"] == "legacy-name"


def test_openinference_provider_precedes_system():
    both = map_span_attributes(
        {
            "openinference.span.kind": "LLM",
            "llm.model_name": "gpt-5-mini",
            "llm.provider": "azure",
            "llm.system": "openai",
        }
    )
    system_only = map_span_attributes(
        {
            "openinference.span.kind": "LLM",
            "llm.model_name": "gpt-5-mini",
            "llm.system": "openai",
        }
    )

    assert isinstance(both, MappedCall)
    assert both.provider == "azure"
    assert isinstance(system_only, MappedCall)
    assert system_only.provider == "openai"


def test_field_level_merge_across_gen_ai_and_openinference():
    mapped = map_span_attributes(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "anthropic",
            "gen_ai.request.model": "claude-opus-5",
            "gen_ai.usage.input_tokens": 12,
            "gen_ai.usage.output_tokens": 5,
            "openinference.span.kind": "LLM",
            "llm.model_name": "other-model",
            "llm.cost.total": 0.0123,
            "llm.token_count.prompt": 999,
        }
    )

    assert isinstance(mapped, MappedCall)
    assert set(mapped.dialects) == {"gen_ai", "openinference"}
    # gen_ai outranks OpenInference for every field it carries...
    assert mapped.model == "claude-opus-5"
    assert mapped.provider == "anthropic"
    assert mapped.operation == "chat"
    assert mapped.response["usage"]["input_tokens"] == 12
    assert mapped.response["usage"]["output_tokens"] == 5
    # ...while fields only OpenInference carries still land.
    assert mapped.cost == 0.0123
    assert mapped.cost_source == "openinference.llm.cost.total"
# --- LangSmith ---------------------------------------------------------------
#
# LangSmith emits gen_ai.* on every run it exports, so these spans always carry
# two dialects. The attribute values below are the shapes the real SDK
# produces; test_upstream_dialects.py pins that against the installed version.


def _langsmith_llm_span() -> dict[str, object]:
    return {
        "langsmith.span.kind": "llm",
        "langsmith.trace.name": "my-chat-call",
        "langsmith.trace.session_id": "sess-1",
        "gen_ai.operation.name": "chat",
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.response.model": "gpt-4o-mini-2024-07-18",
        "gen_ai.response.finish_reasons": "stop",
        "gen_ai.usage.input_tokens": 11,
        "gen_ai.usage.output_tokens": 4,
        # str(dict), not JSON -- see _load_repr_mapping.
        "gen_ai.usage.input_token_details": "{'cache_read': 8}",
        "gen_ai.usage.output_token_details": "{'reasoning': 2}",
        "gen_ai.prompt": json.dumps(
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]}
        ),
        "gen_ai.completion": json.dumps(
            {
                "model": "gpt-4o-mini-2024-07-18",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello there!"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ),
    }


def test_langsmith_llm_span_maps_to_a_complete_call():
    mapped = map_span_attributes(_langsmith_llm_span())

    assert isinstance(mapped, MappedCall)
    assert mapped.dialects == ("langsmith", "gen_ai")
    assert mapped.model == "gpt-4o-mini"
    assert mapped.provider == "openai"
    assert mapped.operation == "chat"
    assert mapped.request["messages"] == [{"role": "user", "content": "Hi"}]
    assert mapped.response["model"] == "gpt-4o-mini-2024-07-18"
    assert mapped.response["usage"] == {
        "input_tokens": 11,
        "output_tokens": 4,
        "cache_read_input_tokens": 8,
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    assert mapped.response["finish_reason"] == "stop"
    assert mapped.response_text == "Hello there!"
    assert mapped.session_id == "sess-1"
    assert mapped.trace_name == "my-chat-call"
    assert not mapped.parse_degraded
    assert not mapped.usage_absent


def test_langsmith_reports_no_cost():
    """LangSmith prices runs server-side; no cost attribute reaches the span."""
    mapped = map_span_attributes(_langsmith_llm_span())

    assert isinstance(mapped, MappedCall)
    assert mapped.cost is None
    assert mapped.cost_source is None


def test_langsmith_chain_span_is_not_a_billable_call():
    """The double-count guard: a chain span carries gen_ai.* and, through
    metadata inheritance, its child's model and tokens. Only langsmith.span.kind
    tells them apart."""
    attributes = _langsmith_llm_span() | {
        "langsmith.span.kind": "chain",
        "gen_ai.operation.name": "chain",
    }

    assert map_span_attributes(attributes) is SkipReason.INELIGIBLE_KIND


def test_langsmith_tool_span_is_not_a_billable_call():
    attributes = _langsmith_llm_span() | {
        "langsmith.span.kind": "tool",
        "gen_ai.operation.name": "execute_tool",
    }

    assert map_span_attributes(attributes) is SkipReason.INELIGIBLE_KIND


def test_langsmith_without_a_span_kind_is_not_counted():
    """A missing kind is not proof that a LangSmith run is an LLM call."""
    attributes = _langsmith_llm_span()
    del attributes["langsmith.span.kind"]

    assert map_span_attributes(attributes) is SkipReason.INELIGIBLE_KIND


def test_langsmith_token_details_drop_keys_with_no_home_in_the_vocabulary():
    attributes = _langsmith_llm_span() | {
        "gen_ai.usage.input_token_details": "{'cache_read': 8, 'audio': 3}",
    }

    mapped = map_span_attributes(attributes)

    assert isinstance(mapped, MappedCall)
    assert mapped.response["usage"]["cache_read_input_tokens"] == 8
    assert "audio" in mapped.dropped_usage_keys
    assert not mapped.parse_degraded


def test_langsmith_token_details_also_accept_json():
    """literal_eval is the fallback, not the only path: a future upstream
    switch to real JSON must keep working."""
    attributes = _langsmith_llm_span() | {
        "gen_ai.usage.output_token_details": '{"reasoning": 2}',
    }

    mapped = map_span_attributes(attributes)

    assert isinstance(mapped, MappedCall)
    assert mapped.response["usage"]["completion_tokens_details"] == {
        "reasoning_tokens": 2
    }


def test_langsmith_unparseable_token_details_degrade_without_dropping_the_call():
    attributes = _langsmith_llm_span() | {
        "gen_ai.usage.input_token_details": "<not a literal>",
    }

    mapped = map_span_attributes(attributes)

    assert isinstance(mapped, MappedCall)
    assert mapped.parse_degraded
    assert "cache_read_input_tokens" not in mapped.response["usage"]
    assert mapped.response["usage"]["input_tokens"] == 11


def test_langsmith_unparseable_content_is_kept_verbatim_and_flagged():
    attributes = _langsmith_llm_span() | {"gen_ai.completion": "{not json"}

    mapped = map_span_attributes(attributes)

    assert isinstance(mapped, MappedCall)
    assert mapped.parse_degraded
    assert mapped.response_text == "{not json"


def test_langsmith_non_chat_output_keeps_its_structure():
    attributes = _langsmith_llm_span() | {
        "gen_ai.completion": json.dumps({"output": ["a", "b"]})
    }

    mapped = map_span_attributes(attributes)

    assert isinstance(mapped, MappedCall)
    assert mapped.response_text is None
    assert mapped.response["output"] == {"output": ["a", "b"]}


# --- span-kind eligibility ---------------------------------------------------
#
# _vetoed is a pure predicate over verdicts, so it is tested directly on
# verdict dicts rather than through span attributes. That keeps these cases
# independent of which dialects happen to be registered: each dialect asserts
# its own end-to-end eligibility against real attributes in its own tests.


def test_abstaining_dialects_alone_do_not_veto():
    assert _vetoed({DIALECT_GENAI: None}) is False


def test_no_dialects_do_not_veto():
    assert _vetoed({}) is False


def test_a_lone_denial_vetoes_the_span():
    assert _vetoed({"some-dialect": False}) is True


def test_a_claim_outvotes_a_denial():
    assert _vetoed({"some-dialect": False, "other-dialect": True}) is False


def test_a_denial_is_not_rescued_by_an_abstention():
    """The bug this guards: an abstaining gen_ai.* must not revive a span its
    own producer labelled a chain."""
    assert _vetoed({DIALECT_GENAI: None, "some-dialect": False}) is True


# --- Langfuse dialect: usage conventions, provider, attribution ------------

from metergraph._genai_attrs import (  # noqa: E402
    error_type_from_message,
    map_usage_details,
    provider_from_model,
)


def _langfuse(**extra):
    return {
        "langfuse.observation.type": "generation",
        "langfuse.observation.model.name": "gpt-4o-mini",
        **extra,
    }


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("gpt-4o-mini", "openai"),
        ("o4-mini", "openai"),
        ("text-embedding-3-small", "openai"),
        ("claude-3-5-haiku-20241022", "anthropic"),
        ("us.anthropic.claude-3-5-haiku-20241022-v1:0", "bedrock"),
        ("anthropic.claude-sonnet-4-v1:0", "bedrock"),
        ("openai/gpt-4o", "openai"),
        ("anthropic/claude-sonnet-4", "anthropic"),
        ("mistral-large-latest", "mistral"),
        ("gemini-2.5-flash", "google"),
        ("grok-4", "xai"),
        ("in-house-model-7", None),
        ("someone/else", None),
        ("", None),
    ],
)
def test_provider_from_model_names(model, provider):
    assert provider_from_model(model) == provider


def test_langchain_ls_provider_beats_the_model_name():
    mapped = map_span_attributes(
        _langfuse(**{"langfuse.observation.metadata.ls_provider": '"azure"'})
    )
    assert mapped.provider == "azure"


def test_openai_usage_object_passes_through_with_nested_cache_and_reasoning():
    usage, dropped = map_usage_details(
        {
            "prompt_tokens": 700,
            "completion_tokens": 30,
            "total_tokens": 730,
            "prompt_tokens_details": {"cached_tokens": 512},
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
        "openai",
    )
    assert dropped == ()
    mapped = map_span_attributes(
        _langfuse(**{"langfuse.observation.usage_details": json.dumps(usage)})
    )
    counts = _usage(mapped.response)
    assert counts["input_tokens"] == 700
    assert counts["cache_read_tokens"] == 512
    assert counts["reasoning_tokens"] == 4


def test_disjoint_buckets_restate_input_for_cache_inclusive_providers():
    details = {"input": 900, "input_cached_tokens": 600, "output": 120, "output_reasoning_tokens": 40}
    openai_usage, _ = map_usage_details(details, "openai")
    anthropic_usage, _ = map_usage_details(details, "anthropic")
    unknown_usage, _ = map_usage_details(details, None)

    assert openai_usage["input_tokens"] == 1500
    assert anthropic_usage["input_tokens"] == 900
    assert unknown_usage["input_tokens"] == 900
    for usage in (openai_usage, anthropic_usage, unknown_usage):
        assert usage["output_tokens"] == 160
        assert usage["cache_read_input_tokens"] == 600
        assert usage["reasoning_tokens"] == 40


def test_cache_write_ttl_buckets_sum_into_cache_write():
    usage, dropped = map_usage_details(
        {"input": 10, "input_cache_creation_5m": 3, "input_cache_creation_1h": 4, "output": 1},
        "anthropic",
    )
    assert dropped == ()
    mapped = map_span_attributes(
        _langfuse(
            **{
                "langfuse.observation.model.name": "claude-sonnet-4",
                "langfuse.observation.usage_details": json.dumps(
                    {"input": 10, "input_cache_creation_5m": 3, "input_cache_creation_1h": 4, "output": 1}
                ),
            }
        )
    )
    counts = _usage(mapped.response)
    assert counts["cache_write_tokens"] == 7
    assert counts["cache_write_5m_tokens"] == 3
    assert counts["cache_write_1h_tokens"] == 4
    assert usage["cache_creation_input_tokens"] == 7


def test_langchain_disjoint_spellings_are_read():
    mapped = map_span_attributes(
        _langfuse(
            **{
                "langfuse.observation.usage_details": json.dumps(
                    {"input": 188, "output": 30, "total": 730, "input_cache_read": 512, "output_reasoning": 0}
                )
            }
        )
    )
    counts = _usage(mapped.response)
    assert counts["input_tokens"] == 700
    assert counts["cache_read_tokens"] == 512
    assert mapped.dropped_usage_keys == ()


def test_attribution_from_trace_and_observation_attributes():
    mapped = map_span_attributes(
        _langfuse(
            **{
                "user.id": "u-1",
                "langfuse.environment": "staging",
                "langfuse.release": "r-9",
                "langfuse.version": "v-2",
                "langfuse.trace.tags": ("beta", "alpha"),
                "langfuse.trace.metadata.team": '"search"',
                "langfuse.trace.metadata.shared": '"trace"',
                "langfuse.observation.metadata.shared": '"observation"',
                "langfuse.observation.metadata.limits": '{"max":3}',
                "langfuse.observation.prompt.name": "triage",
                "langfuse.observation.prompt.version": 4,
            }
        )
    )
    assert mapped.user_id == "u-1"
    assert mapped.environment == "staging"
    assert mapped.route_name == "triage"
    assert mapped.tags == {
        "langfuse.tag.alpha": "true",
        "langfuse.tag.beta": "true",
        "langfuse.release": "r-9",
        "langfuse.version": "v-2",
        "langfuse.metadata.team": "search",
        "langfuse.metadata.shared": "observation",
        "langfuse.metadata.limits": '{"max":3}',
        "langfuse.prompt.name": "triage",
        "langfuse.prompt.version": "4",
    }


@pytest.mark.parametrize(
    "tags", ['["a","b"]', "a, b", ["b", "a"]],
)
def test_tags_accept_every_encoding_langfuse_accepts(tags):
    mapped = map_span_attributes(_langfuse(**{"langfuse.tags": tags}))
    assert {k for k in mapped.tags if k.startswith("langfuse.tag.")} == {
        "langfuse.tag.a",
        "langfuse.tag.b",
    }


def test_compat_spellings_and_whole_object_metadata():
    mapped = map_span_attributes(
        _langfuse(
            **{
                "langfuse.user.id": "u-compat",
                "langfuse.session.id": "s-compat",
                "deployment.environment.name": "prod",
                "langfuse.metadata": '{"tier":"gold","n":2}',
            }
        )
    )
    assert mapped.user_id == "u-compat"
    assert mapped.session_id == "s-compat"
    assert mapped.environment == "prod"
    assert mapped.tags["langfuse.metadata.tier"] == "gold"
    assert mapped.tags["langfuse.metadata.n"] == "2"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("RateLimitError: 429 from upstream", "RateLimitError"),
        ("openai.APITimeoutError: timed out", "APITimeoutError"),
        ("ContextWindowExceeded", "ContextWindowExceeded"),
        ("the model refused", None),
        (None, None),
    ],
)
def test_error_type_from_status_message(message, expected):
    assert error_type_from_message(message) == expected


def test_embedding_and_custom_typed_model_calls_are_eligible():
    embedding = map_span_attributes(
        {
            "langfuse.observation.type": "embedding",
            "langfuse.observation.model.name": "text-embedding-3-small",
            "langfuse.observation.usage_details": '{"input": 12}',
        }
    )
    assert isinstance(embedding, MappedCall)
    assert embedding.operation == "embeddings"

    agent_call = map_span_attributes(
        {
            "langfuse.observation.type": "agent",
            "langfuse.observation.model.name": "gpt-4o-mini",
            "langfuse.observation.cost_details": '{"total": 0.01}',
        }
    )
    assert isinstance(agent_call, MappedCall)
    assert agent_call.cost == 0.01

    structure = map_span_attributes(
        {"langfuse.observation.type": "agent", "langfuse.observation.model.name": "gpt-4o-mini"}
    )
    assert structure is SkipReason.INELIGIBLE_KIND


def test_cost_details_without_total_sums_its_parts():
    mapped = map_span_attributes(
        _langfuse(**{"langfuse.observation.cost_details": '{"input": 0.25, "output": 0.5}'})
    )
    assert mapped.cost == 0.75
