// Perplexity Sonar and Agent search usage and request defaults.
import { count, get } from './retrieval-common.js';

export function perplexityRetrievalUsage(endpoint: string, request: Record<string, unknown>, values: unknown[], items: unknown[] | undefined): Record<string, unknown> {
  if (!endpoint.includes('responses')) return {retrieval_api: 'sonar'};
  let searches: number | undefined;
  for (const body of values) {
    const usage = get(get(get(get(body, 'usage'), 'tool_calls_details'), 'search_web'), 'invocation');
    searches = count(usage) ?? searches;
  }
  searches ??= items?.filter(item => get(item, 'type') === 'search_results').length;
  const extra = request.extra_body;
  if (extra && typeof extra === 'object' && !Array.isArray(extra)) request = {...request, ...extra};
  const tools = Array.isArray(request.tools) ? request.tools : [];
  // Active effort bundles use fast search; pinned versions and profiles are not inferred.
  const activeFast = typeof request.effort === 'string' && ['fast','low','medium','high','xhigh'].some(level => request.effort === level || request.effort === `${level}@latest`);
  const defaultType = 'profile' in request ? undefined : activeFast ? 'fast' : !('effort' in request) ? 'web' : undefined;
  const types = request.tools === undefined && defaultType === 'fast' ? ['fast'] : tools.filter(tool => get(tool, 'type') === 'web_search')
    .map(tool => get(tool, 'search_type') === undefined ? defaultType : get(tool, 'search_type'));
  const searchType = types.length && types.every(type => type === types[0]) && ['web', 'fast'].includes(types[0]) ? types[0] : undefined;
  return {retrieval_api: 'agent', ...(searches !== undefined ? {web_search_calls: searches} : {}), ...(searchType ? {search_type: searchType} : {})};
}
