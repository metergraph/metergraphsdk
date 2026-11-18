import { pathToFileURL } from "node:url";
import * as mg from "metergraph";

export function answer(client, { userId, sessionId, customerId, question }) {
  return mg.withContext({
    userId, sessionId,
    metadata: { customer_id: customerId, feature: "support" },
  }, () => mg.trace("support-answer", () => mg.route("support.answer", () =>
    client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [{ role: "user", content: question }],
    }),
  )));
}

async function main() {
  const { default: OpenAI } = await import("openai");
  mg.init({ repository: "owner/repository", captureText: false });
  const client = mg.wrap(new OpenAI());
  try {
    // Two people in one customer account, with independent conversations.
    const results = await Promise.all([1, 2].map((i) => answer(client, {
      userId: `user-${i}`, sessionId: `session-${i}`, customerId: "customer-123",
      question: "Say hello in one sentence.",
    })));
    for (const result of results) console.log(result.choices[0].message.content);
  } finally {
    await mg.shutdown();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
