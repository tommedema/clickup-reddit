export interface CacheEntry {
  checksum: string;
  storedAt: string;
  metadata?: Record<string, unknown>;
  body: string;
}

export interface CacheWriteInput {
  checksum: string;
  body: string;
  metadata?: Record<string, unknown>;
}

export interface CacheClient {
  read(namespace: string, checksum: string): Promise<CacheEntry | null>;
  write(namespace: string, entry: CacheWriteInput): Promise<void>;
}
