// Shared accessors for native retrieval usage.
export function get(value: unknown, key: string): any {
  return value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : undefined;
}
export function count(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}
