"""Scoped attribution for concurrent requests. See README.md before running."""

import asyncio

import metergraph


async def answer(client, *, user_id, session_id, customer_id, question):
    with metergraph.context(
        user_id=user_id,
        session_id=session_id,
        metadata={"customer_id": customer_id, "feature": "support"},
    ):
        with metergraph.trace("support-answer"):
            with metergraph.route("support.answer"):
                return await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": question}],
                )


async def main():
    from openai import AsyncOpenAI

    metergraph.init(repository="owner/repository", capture_text=False)
    client = metergraph.wrap(AsyncOpenAI())
    try:
        # Two people in one customer account, with independent conversations.
        results = await asyncio.gather(*[
            answer(client, user_id=f"user-{i}", session_id=f"session-{i}",
                   customer_id="customer-123", question="Say hello in one sentence.")
            for i in (1, 2)
        ])
        for result in results:
            print(result.choices[0].message.content)
    finally:
        await client.close()
        metergraph.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
