// Capture native usage only. Billing units and prices belong to the receiver.
function get(value: unknown, key: string): any {
  return value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : undefined;
}
function count(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}
function bodies(response: unknown, chunks: unknown[]): unknown[] {
  return [...chunks.flatMap(chunk => [get(chunk, 'message'), get(chunk, 'response'), chunk]), response];
}
function outputItems(response: unknown, chunks: unknown[]): unknown[] | undefined {
  const direct = get(response, 'output');
  if (Array.isArray(direct)) return direct;
  for (const chunk of [...chunks].reverse()) {
    if (['response.completed', 'response.incomplete', 'response.failed'].includes(get(chunk, 'type'))) {
      const output = get(get(chunk, 'response'), 'output');
      if (Array.isArray(output)) return output;
    }
  }
  const items = new Map<unknown, unknown>();
  for (const chunk of chunks) {
    if (get(chunk, 'type') === 'response.output_item.done') {
      const item = get(chunk, 'item');
      const id = get(item, 'id') ?? get(chunk, 'output_index');
      if (id !== undefined) items.set(id, item);
    }
  }
  return items.size ? [...items.values()] : undefined;
}
export function retrievalUsage(provider: string, endpoint: string, request: Record<string, unknown>, response: unknown, chunks: unknown[] = []): Record<string, unknown> {
  const values = bodies(response, chunks);
  if (provider === 'anthropic') {
    const server: Record<string, number> = {};
    for (const body of values) {
      const usage = get(get(body, 'usage'), 'server_tool_use');
      for (const key of ['web_search_requests', 'web_fetch_requests']) {
        const value = count(get(usage, key));
        if (value !== undefined) server[key] = value;
      }
    }
    return Object.keys(server).length ? {server_tool_use: server} : {};
  }
  if (provider === 'google') {
    const queries = new Set<string>();
    let observed = false;
    for (const body of values) for (const candidate of Array.isArray(get(body, 'candidates')) ? get(body, 'candidates') : []) {
      const metadata = get(candidate, 'groundingMetadata') ?? get(candidate, 'grounding_metadata');
      const items = get(metadata, 'webSearchQueries') ?? get(metadata, 'web_search_queries');
      if (!Array.isArray(items)) continue;
      observed = true;
      for (const query of items) if (typeof query === 'string' && query.trim()) queries.add(query.trim());
    }
    return observed ? {grounding_queries: queries.size, grounded_prompt: queries.size > 0 ? 1 : 0} : {};
  }
  if (provider !== 'openai' && provider !== 'perplexity') return {};
  const items = outputItems(response, chunks);
  if (provider === 'openai') {
    return items ? {web_search_calls: items.filter(item => get(item, 'type') === 'web_search_call').length} : {};
  }
  if (!endpoint.includes('responses')) return {retrieval_api: 'sonar'};
  let searches: number | undefined;
  for (const body of values) {
    const usage = get(get(get(get(body, 'usage'), 'tool_calls_details'), 'search_web'), 'invocation');
    searches = count(usage) ?? searches;
  }
  searches ??= items?.filter(item => get(item, 'type') === 'search_results').length;
  const tools = Array.isArray(request.tools) ? request.tools : [];
  const types = tools.filter(tool => get(tool, 'type') === 'web_search').map(tool => get(tool, 'search_type'));
  const searchType = types.length && types.every(type => type === types[0]) && ['web', 'fast'].includes(types[0]) ? types[0] : undefined;
  return {retrieval_api: 'agent', ...(searches !== undefined ? {web_search_calls: searches} : {}), ...(searchType ? {search_type: searchType} : {})};
}
