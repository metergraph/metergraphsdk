"""Native retrieval usage. Receivers own billing units and prices."""
from collections.abc import Mapping
from typing import Any


def get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def count(value):
    return value if type(value) is int and value >= 0 else None


def output_items(response, chunks):
    output = get(response, "output")
    if isinstance(output, list):
        return output
    for chunk in reversed(chunks):
        if get(chunk, "type") in {"response.completed", "response.incomplete", "response.failed"}:
            output = get(get(chunk, "response"), "output")
            if isinstance(output, list):
                return output
    items = {}
    for chunk in chunks:
        if get(chunk, "type") == "response.output_item.done":
            item = get(chunk, "item")
            key = get(item, "id")
            if key is None:
                key = get(chunk, "output_index")
            if isinstance(key, (str, int)):
                items[key] = item
    return list(items.values()) if items else None


def retrieval_usage(provider: str, endpoint: str, request: dict, response: Any, chunks=None) -> dict:
    chunks = chunks or []
    values = [value for chunk in chunks for value in (get(chunk, "message"), get(chunk, "response"), chunk)] + [response]
    if provider == "anthropic":
        server = {}
        for body in values:
            usage = get(get(body, "usage"), "server_tool_use")
            for key in ("web_search_requests", "web_fetch_requests"):
                value = count(get(usage, key))
                if value is not None:
                    server[key] = value
        return {"server_tool_use": server} if server else {}
    if provider == "google":
        queries = set()
        observed = False
        for body in values:
            for candidate in get(body, "candidates", None) or []:
                metadata = get(candidate, "grounding_metadata") or get(candidate, "groundingMetadata")
                items = get(metadata, "web_search_queries")
                if items is None:
                    items = get(metadata, "webSearchQueries")
                if not isinstance(items, list):
                    continue
                observed = True
                queries.update(query.strip() for query in items if isinstance(query, str) and query.strip())
        return {"grounding_queries": len(queries), "grounded_prompt": int(bool(queries))} if observed else {}
    if provider not in {"openai", "perplexity"}:
        return {}
    items = output_items(response, chunks)
    if provider == "openai":
        return {"web_search_calls": sum(get(item, "type") == "web_search_call" for item in items)} if items is not None else {}
    if "responses" not in endpoint:
        return {"retrieval_api": "sonar"}
    searches = None
    for body in values:
        value = count(get(get(get(get(body, "usage"), "tool_calls_details"), "search_web"), "invocation"))
        if value is not None:
            searches = value
    if searches is None and items is not None:
        searches = sum(get(item, "type") == "search_results" for item in items)
    tools = request.get("tools")
    # Bare tools default to web. Managed effort/profile configurations can override that default.
    extra = request.get("extra_body")
    managed = any(key in request or isinstance(extra, Mapping) and key in extra for key in ("effort", "profile"))
    default_type = None if managed else "web"
    types = [get(tool, "search_type", default_type) for tool in tools if get(tool, "type") == "web_search"] if isinstance(tools, list) else []
    search_type = types[0] if types and all(value == types[0] for value in types) and types[0] in {"web", "fast"} else None
    return {"retrieval_api": "agent", **({"web_search_calls": searches} if searches is not None else {}), **({"search_type": search_type} if search_type else {})}
