# Attribute calls to users, sessions, and customers

Two concurrent requests belong to different people and conversations in one
customer account. Both examples set context once around each request; nested
trace and route scopes inherit it. Prompt capture is off, but attribution stays.

| Field | Purpose | Example |
|---|---|---|
| `user_id` / `userId` | Stable opaque person ID across sessions | `user-1` |
| `session_id` / `sessionId` | Conversation or workflow grouping | `session-1` |
| `metadata.customer_id` | Your customer/account cost dimension | `customer-123` |
| `metadata.feature` | Another application-defined dimension | `support` |

Use your application's opaque IDs, rather than email addresses or names. The
authenticated Metergraph workspace is independent of these values. Metadata
values must be strings; they travel in the existing `tags` wire field.

## Run

These commands make **two billable OpenAI calls** and send telemetry to your
configured Metergraph endpoint. They are not offline demos. Use an SDK build
containing user context support and compatible ingestion before relying on
user attribution. Replace `owner/repository` in the source with your repository.

From the repository root, install the local Python SDK:

```sh
python -m pip install -e ./python openai
export OPENAI_API_KEY="replace-with-your-key"
export METERGRAPH_APP_TOKEN="replace-with-your-ingest-key"
export METERGRAPH_INGEST_URL="https://your-metergraph-host"
python examples/attribution/user-context/main.py
```

For JavaScript, build the local SDK and install this example's dependencies:

```sh
npm --prefix typescript install
npm --prefix typescript run build
npm --prefix examples/attribution/user-context install
npm --prefix examples/attribution/user-context start
```

Use the same environment variables as Python. In an application consuming the
published SDK, install `metergraph` and `openai` normally instead of a local link.

## What to inspect

Each request produces a separate trace, with its own user and session IDs.
Both calls have route `support.answer` and tags
`customer_id=customer-123, feature=support`. Customer cost grouping uses
`metadata.customer_id`; this example supplies attribution, not deletion or a
new dashboard view.

Nested contexts inherit omitted fields; valid local metadata overrides matching
keys. `set_user()` / `setUser()` clears identity only within an active scope,
and leaving the scope restores its parent. Avoid setting request identity in
process-wide defaults. See the [SDK context documentation](../../../python/README.md)
for bounds and invalid-value behavior, and the
[OpenRouter example](../../integrations/providers/openrouter/) for streaming.
