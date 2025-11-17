import { checksumFrom } from '../utils/hash.js';
import { splitIntoMonthlyRanges, type EpochRange } from '../utils/time.js';
import { sleep } from '../utils/sleep.js';
import type { CacheClient } from '../cache/cache.js';
import type { RedditComment } from '../types/index.js';

interface PullpushResponse {
  data: RawRedditComment[];
  error: { message: string } | null;
}

interface RawRedditComment {
  id: string;
  body: string;
  created_utc: number;
  permalink: string;
  subreddit: string;
}

export interface PullpushClientOptions {
  product: string;
    cache: CacheClient;
    monthsPerBatch: number;
    startEpoch: number;
  endEpoch: number;
  subreddit?: string | undefined;
  namespace?: string;
  pageSize?: number;
  baseUrl?: string;
  maxRetries?: number;
  retryBackoffMs?: number;
  requestSpacingMs?: number;
  logger?: (message: string) => void;
}

const DEFAULT_BASE_URL = 'https://api.pullpush.io/reddit/search/comment/';
const MIN_CURSOR_INCREMENT = 1;

export class PullpushClient {
  private readonly product: string;
  private readonly cache: CacheClient;
  private readonly namespace: string;
  private readonly pageSize: number;
  private readonly baseUrl: string;
  private readonly maxRetries: number;
  private readonly retryBackoffMs: number;
  private readonly requestSpacingMs: number;
  private readonly subreddit: string | undefined;
  private readonly logger: ((message: string) => void) | undefined;

  constructor(private readonly options: PullpushClientOptions) {
    this.product = options.product;
    this.cache = options.cache;
    this.namespace = options.namespace ?? 'pullpush-comments';
    this.pageSize = options.pageSize ?? 100;
    this.baseUrl = options.baseUrl ?? DEFAULT_BASE_URL;
    this.maxRetries = options.maxRetries ?? 5;
    this.retryBackoffMs = options.retryBackoffMs ?? 5000;
    this.requestSpacingMs = options.requestSpacingMs ?? 0;
    this.subreddit = options.subreddit?.trim() || undefined;
    this.logger = options.logger;
  }

  async fetchAll(): Promise<RedditComment[]> {
    const ranges = splitIntoMonthlyRanges(
      this.options.startEpoch,
      this.options.endEpoch,
      this.options.monthsPerBatch,
    );
    const dedup = new Map<string, RedditComment>();

    for (const range of ranges) {
      this.logger?.(
        `Fetching comments between ${range.start} (${new Date(range.start * 1000).toISOString()}) and ${range.end} (${new Date(range.end * 1000).toISOString()})`,
      );
      const chunk = await this.fetchRange(range);
      for (const comment of chunk) {
        dedup.set(comment.id, comment);
      }
    }

    return [...dedup.values()].sort((a, b) => a.createdUtc - b.createdUtc);
  }

  private async fetchRange(range: EpochRange): Promise<RedditComment[]> {
    const results: RedditComment[] = [];
    let cursor = range.start;
    let lastRequestTime = 0;

    while (cursor < range.end) {
      const remaining = range.end - cursor;
      if (remaining <= 0) {
        break;
      }

      const after = Math.floor(cursor);
      const before = Math.floor(range.end);
      const query = new URLSearchParams({
        q: this.product,
        size: String(this.pageSize),
        sort_type: 'created_utc',
        sort: 'asc',
        after: after.toString(),
        before: before.toString(),
      });
      if (this.subreddit) {
        query.set('subreddit', this.subreddit);
      }
      const url = `${this.baseUrl}?${query.toString()}`;
      const checksum = checksumFrom({ url, method: 'GET' });

      const cached = await this.cache.read(this.namespace, checksum);
      let payload: PullpushResponse;
      if (cached) {
        payload = JSON.parse(cached.body) as PullpushResponse;
      } else {
        const now = Date.now();
        const elapsed = now - lastRequestTime;
        if (this.requestSpacingMs > 0 && elapsed < this.requestSpacingMs) {
          await sleep(this.requestSpacingMs - elapsed);
        }
        payload = await this.fetchLive(url, checksum);
        lastRequestTime = Date.now();
      }

      if (payload.error) {
        throw new Error(`Pullpush API returned error: ${payload.error.message}`);
      }

      const mapped = payload.data.map((comment) => ({
        id: comment.id,
        body: comment.body,
        createdUtc: Math.floor(comment.created_utc),
        permalink: comment.permalink,
        subreddit: comment.subreddit,
      } satisfies RedditComment));

      if (mapped.length === 0) {
        break;
      }

      results.push(...mapped);
      const lastTimestamp = mapped[mapped.length - 1]?.createdUtc ?? after;
      if (lastTimestamp <= after) {
        cursor = after + MIN_CURSOR_INCREMENT;
      } else {
        cursor = lastTimestamp;
      }
    }

    return results;
  }

  private async fetchLive(url: string, checksum: string): Promise<PullpushResponse> {
    for (let attempt = 0; attempt < this.maxRetries; attempt += 1) {
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'clickup-reddit-cli/0.1.0',
        },
      });

      if (response.status === 429) {
        const fromHeader = Number(response.headers.get('retry-after'));
        const waitMs = Number.isFinite(fromHeader) ? fromHeader * 1000 : this.retryBackoffMs * (attempt + 1);
        this.logger?.(`Hit pullpush.io rate limit (429). Waiting ${Math.round(waitMs / 1000)}s before retry #${attempt + 1}.`);
        await sleep(waitMs);
        continue;
      }

      if (!response.ok) {
        if (attempt < this.maxRetries - 1) {
          const waitMs = this.retryBackoffMs * (attempt + 1);
          this.logger?.(`Request failed with status ${response.status}. Retrying in ${waitMs}ms.`);
          await sleep(waitMs);
          continue;
        }

        throw new Error(`Pullpush API request failed with status ${response.status}`);
      }

      const text = await response.text();
      await this.cache.write(this.namespace, {
        checksum,
        body: text,
        metadata: {
          url,
          status: response.status,
        },
      });

      return JSON.parse(text) as PullpushResponse;
    }

    throw new Error('Exceeded retry budget for pullpush API request.');
  }
}
