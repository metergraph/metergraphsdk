"""Perplexity Sonar and Agent search usage and request defaults."""
from collections.abc import Mapping

from ._retrieval_common import count, get


def perplexity_retrieval_usage(endpoint: str, request: dict, values: list, items: list | None) -> dict:
    if "responses" not in endpoint:
        return {"retrieval_api": "sonar"}
    searches = None
    for body in values:
        value = count(get(get(get(get(body, "usage"), "tool_calls_details"), "search_web"), "invocation"))
        if value is not None:
            searches = value
    if searches is None and items is not None:
        searches = sum(get(item, "type") == "search_results" for item in items)
    extra = request.get("extra_body")
    request = {**request, **extra} if isinstance(extra, Mapping) else request
    tools = request.get("tools")
    # Active effort bundles use fast search; pinned versions and profiles are not inferred.
    effort = request.get("effort")
    active_fast = isinstance(effort, str) and effort.removesuffix("@latest") in {"fast", "low", "medium", "high", "xhigh"}
    default_type = None if "profile" in request else "fast" if active_fast else "web" if "effort" not in request else None
    types = [get(tool, "search_type", default_type) for tool in tools if get(tool, "type") == "web_search"] if isinstance(tools, list) else []
    if "tools" not in request and default_type == "fast":
        types = ["fast"]
    search_type = types[0] if types and all(value == types[0] for value in types) and types[0] in {"web", "fast"} else None
    return {"retrieval_api": "agent", **({"web_search_calls": searches} if searches is not None else {}), **({"search_type": search_type} if search_type else {})}
