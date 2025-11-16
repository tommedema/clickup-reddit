import { promises as fs } from 'node:fs';
import path from 'node:path';
import type { CacheClient, CacheEntry, CacheWriteInput } from './cache.js';

interface MetadataFile {
  storedAt: string;
  metadata?: Record<string, unknown>;
}

export interface FileCacheOptions {
  baseDir?: string;
}

export class FileCache implements CacheClient {
  private readonly baseDir: string;

  constructor(options: FileCacheOptions = {}) {
    this.baseDir = options.baseDir ?? '.cache';
  }

  async read(namespace: string, checksum: string): Promise<CacheEntry | null> {
    const { bodyPath, metaPath } = this.paths(namespace, checksum);

    try {
      const [body, metaRaw] = await Promise.all([
        fs.readFile(bodyPath, 'utf8'),
        fs.readFile(metaPath, 'utf8'),
      ]);
      const meta: MetadataFile = JSON.parse(metaRaw);
      const entry: CacheEntry = {
        checksum,
        body,
        storedAt: meta.storedAt,
        ...(meta.metadata ? { metadata: meta.metadata } : {}),
      };
      return entry;
    } catch (error: any) {
      if (error?.code === 'ENOENT') {
        return null;
      }

      throw error;
    }
  }

  async write(namespace: string, entry: CacheWriteInput): Promise<void> {
    const { dir, bodyPath, metaPath } = this.paths(namespace, entry.checksum);
    await fs.mkdir(dir, { recursive: true });

    const metadata: MetadataFile = { storedAt: new Date().toISOString() };
    if (entry.metadata) {
      metadata.metadata = entry.metadata;
    }

    await Promise.all([
      fs.writeFile(bodyPath, entry.body, 'utf8'),
      fs.writeFile(metaPath, JSON.stringify(metadata, null, 2), 'utf8'),
    ]);
  }

  private paths(namespace: string, checksum: string) {
    const dir = path.join(this.baseDir, namespace);
    return {
      dir,
      bodyPath: path.join(dir, `${checksum}.body`),
      metaPath: path.join(dir, `${checksum}.meta.json`),
    };
  }
}
