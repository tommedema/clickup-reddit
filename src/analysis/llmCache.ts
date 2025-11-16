import type { CacheClient } from '../cache/cache.js';
import { checksumFrom } from '../utils/hash.js';

export class LlmCache {
  constructor(private readonly cache: CacheClient, private readonly namespace: string = 'openai-responses') {}

  async getOrCompute<T>(payload: unknown, compute: () => Promise<T>): Promise<T> {
    const checksum = checksumFrom(payload);
    const cached = await this.cache.read(this.namespace, checksum);
    if (cached) {
      return JSON.parse(cached.body) as T;
    }

    const result = await compute();
    await this.cache.write(this.namespace, {
      checksum,
      body: JSON.stringify(result),
      metadata: {
        payloadType: describePayload(payload),
      },
    });

    return result;
  }
}

function describePayload(payload: unknown): string | undefined {
  if (!payload || typeof payload !== 'object') {
    return undefined;
  }
  const maybeType = (payload as Record<string, unknown>).type;
  return typeof maybeType === 'string' ? maybeType : undefined;
}
