import { createWriteStream, WriteStream } from 'node:fs';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { finished } from 'node:stream/promises';

export interface CsvRow {
  timestamp: number;
  iso_date: string;
  iso_year: string;
  category: string;
  sentiment_keyword: string;
  sentiment_number: number;
  category_confidence: string;
  sentiment_confidence: string;
  summary: string;
  subreddit: string;
  comment_link: string;
}

const HEADER = [
  'timestamp',
  'iso_date',
  'iso_year',
  'category',
  'sentiment_keyword',
  'sentiment_number',
  'category_confidence',
  'sentiment_confidence',
  'summary',
  'subreddit',
  'comment_link',
];

export class CsvStreamWriter {
  private constructor(private readonly destination: string, private readonly stream: WriteStream) {}

  static async create(destination: string): Promise<CsvStreamWriter> {
    await fs.mkdir(path.dirname(destination), { recursive: true });
    const stream = createWriteStream(destination, { encoding: 'utf8' });
    stream.write(`${HEADER.join(',')}\n`);
    return new CsvStreamWriter(destination, stream);
  }

  async writeRow(row: CsvRow): Promise<void> {
    const line = HEADER.map((key) => csvEscape(String(row[key as keyof CsvRow] ?? ''))).join(',');
    if (!this.stream.write(`${line}\n`)) {
      await onceDrain(this.stream);
    }
  }

  async close(): Promise<void> {
    this.stream.end();
    await finished(this.stream);
  }

  get path(): string {
    return this.destination;
  }
}

async function onceDrain(stream: WriteStream): Promise<void> {
  await new Promise<void>((resolve) => stream.once('drain', resolve));
}

function csvEscape(value: string): string {
  const needsQuotes = value.includes(',') || value.includes('\n') || value.includes('"');
  const sanitized = value.replace(/\r?\n/g, ' ').replace(/"/g, '""');
  return needsQuotes ? `"${sanitized}"` : sanitized;
}
