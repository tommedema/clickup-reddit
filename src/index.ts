#!/usr/bin/env node
import path from 'node:path';
import { Command } from 'commander';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import { FileCache } from './cache/fileCache.js';
import { PullpushClient } from './clients/pullpush.js';
import { CommentAnalyzer } from './analysis/commentAnalyzer.js';
import { CsvStreamWriter, type CsvRow } from './csv/writer.js';
import { epochSecondsNow, epochToIso, toUnixSeconds } from './utils/time.js';
import type { Sentiment, CommentClassification, CategoryDefinition } from './types/index.js';
import { LlmCache } from './analysis/llmCache.js';

dotenv.config();

interface CliOptions {
  keyword: string;
  start?: string;
  end?: string;
  startEpoch?: string;
  endEpoch?: string;
  monthsPerBatch?: string;
  pageSize?: string;
  concurrency?: string;
  model?: string;
  contextTokens?: string;
  outputReserve?: string;
  requestSpacing?: string;
  subreddits?: string;
  productDescription?: string;
}

const SENTIMENT_SCORES: Record<Sentiment, number> = {
  'very-negative': -2,
  negative: -1,
  neutral: 0,
  positive: 1,
  'very-positive': 2,
};

async function main() {
  const program = new Command();
  program
    .name('clickup-reddit')
    .description('Fetch Reddit comments for a keyword, analyze them with OpenAI, and export to CSV.')
    .option('-k, --keyword <value>', 'Product keyword to search for.', 'clickup')
    .option('--start <iso>', 'Start date (ISO, default 2017-01-01).', '2017-01-01')
    .option('--end <iso>', 'End date (ISO, default today).')
    .option('--start-epoch <seconds>', 'Start epoch seconds (overrides --start).')
    .option('--end-epoch <seconds>', 'End epoch seconds (overrides --end).')
    .option('--months-per-batch <number>', 'Months per API batch (default 6).')
    .option('--page-size <number>', 'Comments per API call (default 100).')
    .option('--concurrency <number>', 'Concurrent LLM calls (default 10).')
    .option('--model <id>', 'OpenAI model (default gpt-5-nano-2025-08-07).')
    .option('--context-tokens <number>', 'Context window tokens for first pass (default 8000).')
    .option('--output-reserve <number>', 'Tokens reserved for output when discovering categories (default 800).')
    .option('--request-spacing <ms>', 'Minimum delay between HTTP requests in ms (default 0).')
    .option('--subreddits <names>', 'Comma-separated list of subreddit names to process individually.')
    .option('--product-description <text>', 'One-sentence summary of the product to provide additional context to the classifier.');

  program.parse();
  const raw = program.opts<CliOptions>();

  const keyword = raw.keyword?.trim() || 'clickup';
  const monthsPerBatch = parsePositiveInteger(raw.monthsPerBatch, 6, 'months-per-batch');
  const pageSize = parsePositiveInteger(raw.pageSize, 100, 'page-size');
  const concurrency = parsePositiveInteger(raw.concurrency, 10, 'concurrency');
  const contextTokens = parsePositiveInteger(raw.contextTokens, 8000, 'context-tokens');
  const outputReserve = parsePositiveInteger(raw.outputReserve, 800, 'output-reserve');
  const requestSpacingMs = parseNonNegativeInteger(raw.requestSpacing, 0, 'request-spacing');

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is missing from the environment.');
  }

  const model = raw.model?.trim() || 'gpt-5-nano-2025-08-07';
  const cache = new FileCache();
  const openai = new OpenAI({ apiKey });
  const llmCache = new LlmCache(cache);
  const productDescription = normalizeDescription(raw.productDescription, keyword);

  const subreddits = parseCommaList(raw.subreddits);
  const targets = subreddits.length > 0 ? subreddits : [undefined];

  const startEpoch = determineEpoch(raw.startEpoch, raw.start ?? '2017-01-01T00:00:00Z');
  const endEpoch = determineEpoch(raw.endEpoch, raw.end ?? epochToIso(epochSecondsNow()));
  if (endEpoch <= startEpoch) {
    throw new Error('End epoch must be greater than start epoch.');
  }

  for (const subreddit of targets) {
    await processTarget({
      keyword,
      subreddit,
      startEpoch,
      endEpoch,
      cache,
      llmCache,
      openai,
      monthsPerBatch,
      pageSize,
      requestSpacingMs,
      contextTokens,
      outputReserve,
      concurrency,
      model,
      productDescription,
    });
  }
}

interface ProcessTargetArgs {
  keyword: string;
  subreddit?: string | undefined;
  startEpoch: number;
  endEpoch: number;
  cache: FileCache;
  llmCache: LlmCache;
  openai: OpenAI;
  monthsPerBatch: number;
  pageSize: number;
  requestSpacingMs: number;
  contextTokens: number;
  outputReserve: number;
  concurrency: number;
  model: string;
  productDescription: string;
}

async function processTarget(args: ProcessTargetArgs): Promise<void> {
  const {
    keyword,
    subreddit,
    startEpoch,
    endEpoch,
    cache,
    llmCache,
    openai,
    monthsPerBatch,
    pageSize,
    requestSpacingMs,
    contextTokens,
    outputReserve,
    concurrency,
    model,
    productDescription,
  } = args;

  const suffix = subreddit ? ` r/${subreddit}` : '';
  const logger = (scope: string) => (message: string) =>
    console.log(`[${scope}${subreddit ? `:${subreddit}` : ''}] ${message}`);

  const redditClient = new PullpushClient({
    keyword,
    subreddit,
    cache,
    monthsPerBatch,
    startEpoch,
    endEpoch,
    pageSize,
    requestSpacingMs,
    logger: logger('pullpush'),
  });

  console.log(
    `Fetching Reddit comments for "${keyword}"${suffix} from ${new Date(startEpoch * 1000).toISOString()} to ${new Date(endEpoch * 1000).toISOString()}...`,
  );
  const comments = await redditClient.fetchAll();
  const normalizedSubreddit = subreddit?.toLowerCase();
  const filtered = normalizedSubreddit ? comments.filter((c) => c.subreddit.toLowerCase() === normalizedSubreddit) : comments;
  console.log(`Fetched ${filtered.length} comments${subreddit ? ' after subreddit filter' : ''}.`);

  if (filtered.length === 0) {
    console.log(`No matching Reddit comments found${suffix}.`);
    return;
  }

  const analyzer = new CommentAnalyzer(openai, llmCache, {
    keyword,
    model,
    productDescription,
    contextTokens,
    outputReserveTokens: outputReserve,
    concurrency,
    logger: logger('analysis'),
  });

  const safeKeyword = keyword.toLowerCase().replace(/[^a-z0-9-_]+/g, '-');
  const safeSubreddit = subreddit ? `r-${subreddit.toLowerCase().replace(/[^a-z0-9-_]+/g, '-')}` : null;
  const isoStamp = new Date().toISOString().replace(/[:]/g, '-');
  const descriptor = safeSubreddit ? `${safeKeyword}_${safeSubreddit}` : safeKeyword;
  const filename = `${isoStamp}_${descriptor}.csv`;
  const outputPath = path.join('output', filename);
  const writer = await CsvStreamWriter.create(outputPath);
  let written = 0;
  let categoriesLogged = false;

  const logCategories = (categories: CategoryDefinition[]) => {
    if (categories.length === 0) {
      console.log(`No categories identified${suffix}.`);
    } else {
      console.log(`Identified categories${suffix}:`);
      categories.forEach((category) => {
        console.log(` - ${category.slug}: ${category.description}`);
      });
    }
    categoriesLogged = true;
  };

  const analysis = await analyzer.analyze(filtered, {
    collectClassifications: false,
    onClassification: async (classification) => {
      await writer.writeRow(classificationToRow(classification));
      written += 1;
    },
    onCategories: async (categories) => {
      logCategories(categories);
    },
  });

  await writer.close();

  console.log(`Wrote ${written} rows to ${outputPath}`);
  if (!categoriesLogged && analysis.categories.length > 0) {
    logCategories(analysis.categories);
  }
}

function determineEpoch(epochOption: string | undefined, fallbackIso: string): number {
  if (epochOption) {
    const value = Number(epochOption);
    if (Number.isNaN(value)) {
      throw new Error(`Invalid epoch value: ${epochOption}`);
    }
    return Math.floor(value);
  }

  return toUnixSeconds(fallbackIso);
}

function parsePositiveInteger(value: string | undefined, fallback: number, flagName: string): number {
  if (value === undefined) {
    return fallback;
  }

  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Option --${flagName} must be a positive number.`);
  }
  return Math.floor(parsed);
}

function parseNonNegativeInteger(value: string | undefined, fallback: number, flagName: string): number {
  if (value === undefined) {
    return fallback;
  }

  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`Option --${flagName} must be zero or a positive number.`);
  }
  return Math.floor(parsed);
}

function parseCommaList(value: string | undefined): string[] {
  if (!value) {
    return [];
  }
  const seen = new Set<string>();
  const parts: string[] = [];
  for (const rawPart of value.split(',')) {
    const trimmed = rawPart.trim();
    if (!trimmed) {
      continue;
    }
    const key = trimmed.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    parts.push(trimmed);
  }
  return parts;
}

function normalizeDescription(description: string | undefined, keyword: string): string {
  const trimmed = description?.trim();
  if (trimmed && trimmed.length > 0) {
    return trimmed;
  }
  return `The product "${keyword}".`;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});

function classificationToRow(classification: CommentClassification): CsvRow {
  const isoDate = epochToIso(classification.comment.createdUtc);
  return {
    timestamp: classification.comment.createdUtc,
    iso_date: isoDate,
    iso_year: new Date(classification.comment.createdUtc * 1000).getUTCFullYear().toString(),
    category: classification.category,
    sentiment_keyword: classification.sentiment,
    sentiment_number: SENTIMENT_SCORES[classification.sentiment],
    category_confidence: classification.categoryConfidence,
    sentiment_confidence: classification.sentimentConfidence,
    summary: classification.summary,
    subreddit: classification.comment.subreddit,
    comment_link: `https://reddit.com${classification.comment.permalink}`,
  } satisfies CsvRow;
}
