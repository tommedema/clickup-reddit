#!/usr/bin/env node
import path from 'node:path';
import { promises as fs } from 'node:fs';
import { Command } from 'commander';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import { FileCache } from './cache/fileCache.js';
import { PullpushClient } from './clients/pullpush.js';
import { CommentAnalyzer } from './analysis/commentAnalyzer.js';
import { CsvStreamWriter, type CsvRow } from './csv/writer.js';
import { epochSecondsNow, epochToIso, toUnixSeconds } from './utils/time.js';
import type { Sentiment, CommentClassification, CategoryDefinition, RedditComment } from './types/index.js';
import { LlmCache } from './analysis/llmCache.js';

dotenv.config();

const program = new Command();
program.name('clickup-reddit').description('Fetch Reddit comments for a product, analyze them with OpenAI, and export to CSV.');

configureCommonOptions(
  program
    .command('categories')
    .description('Generate product-specific categories and save them to a JSON file.'),
)
  .option('--categories-output <path>', 'Path to write categories JSON.', 'categories.json')
  .action(async (rawOptions) => {
    await handleGenerateCategories(rawOptions);
  });

configureCommonOptions(
  program
    .command('analyze')
    .description('Analyze Reddit comments using an existing categories JSON file.'),
)
  .option('--categories-input <path>', 'Path to read categories JSON.', 'categories.json')
  .action(async (rawOptions) => {
    await handleAnalyze(rawOptions);
  });

program.parseAsync();

interface RawCommonOptions {
  product: string;
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

interface CategoriesCommandOptions extends RawCommonOptions {
  categoriesOutput: string;
}

interface AnalyzeCommandOptions extends RawCommonOptions {
  categoriesInput: string;
}

interface CommonContext {
  product: string;
  productDescription: string;
  startEpoch: number;
  endEpoch: number;
  monthsPerBatch: number;
  pageSize: number;
  concurrency: number;
  model: string;
  contextTokens: number;
  outputReserve: number;
  requestSpacingMs: number;
  subreddits: Array<string | undefined>;
}

function configureCommonOptions(command: Command): Command {
  return command
    .option('-p, --product <value>', 'Product to search for.', 'clickup')
    .option('--start <iso>', 'Start date (ISO, default 2017-01-01).', '2017-01-01')
    .option('--end <iso>', 'End date (ISO, default today).')
    .option('--start-epoch <seconds>', 'Start epoch seconds (overrides --start).')
    .option('--end-epoch <seconds>', 'End epoch seconds (overrides --end).')
    .option('--months-per-batch <number>', 'Months per API batch (default 6).')
    .option('--page-size <number>', 'Comments per API call (default 100).')
    .option('--concurrency <number>', 'Concurrent LLM calls (default 20).')
    .option('--model <id>', 'OpenAI model (default gpt-5-nano-2025-08-07).')
    .option('--context-tokens <number>', 'Context window tokens for first pass (default 400000).')
    .option('--output-reserve <number>', 'Tokens reserved for output when discovering categories (default 128000).')
    .option('--request-spacing <ms>', 'Minimum delay between HTTP requests in ms (default 0).')
    .option('--subreddits <names>', 'Comma-separated list of subreddit names to process individually.')
    .option('--product-description <text>', 'One-sentence summary of the product for prompts.');
}

async function handleGenerateCategories(rawOptions: CategoriesCommandOptions) {
  const context = buildCommonContext(rawOptions);
  const apiKey = requireApiKey();
  const cache = new FileCache();
  const llmCache = new LlmCache(cache);
  const openai = new OpenAI({ apiKey });
  const analyzer = createAnalyzer(openai, llmCache, context, createLogger('analysis'));

  const aggregated = await gatherCommentsAcrossTargets(context, cache);
  if (aggregated.length === 0) {
    console.log('No matching Reddit comments found for the requested range.');
    return;
  }

  console.log(`Generating categories from ${aggregated.length} comments...`);
  const categories = await analyzer.generateCategories(aggregated);
  const outputPath = path.resolve(rawOptions.categoriesOutput ?? 'categories.json');
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(categories, null, 2), 'utf8');
  console.log(`Wrote ${categories.length} categories to ${outputPath}`);
}

async function handleAnalyze(rawOptions: AnalyzeCommandOptions) {
  const context = buildCommonContext(rawOptions);
  const apiKey = requireApiKey();
  const cache = new FileCache();
  const llmCache = new LlmCache(cache);
  const openai = new OpenAI({ apiKey });
  const analyzer = createAnalyzer(openai, llmCache, context, createLogger('analysis'));

  const categoriesPath = path.resolve(rawOptions.categoriesInput ?? 'categories.json');
  const categories = await readCategories(categoriesPath);
  console.log(`Loaded ${categories.length} categories from ${categoriesPath}`);

  for (const subreddit of context.subreddits) {
    await analyzeTarget(context, cache, analyzer, categories, subreddit);
  }
}

function buildCommonContext(raw: RawCommonOptions): CommonContext {
  const product = raw.product?.trim() || 'clickup';
  const monthsPerBatch = parsePositiveInteger(raw.monthsPerBatch, 6, 'months-per-batch');
  const pageSize = parsePositiveInteger(raw.pageSize, 100, 'page-size');
  const concurrency = parsePositiveInteger(raw.concurrency, 20, 'concurrency');
  const contextTokens = parsePositiveInteger(raw.contextTokens, 400000, 'context-tokens');
  const outputReserve = parsePositiveInteger(raw.outputReserve, 128000, 'output-reserve');
  const requestSpacingMs = parseNonNegativeInteger(raw.requestSpacing, 0, 'request-spacing');
  const startEpoch = determineEpoch(raw.startEpoch, raw.start ?? '2017-01-01T00:00:00Z');
  const endEpoch = determineEpoch(raw.endEpoch, raw.end ?? epochToIso(epochSecondsNow()));
  if (endEpoch <= startEpoch) {
    throw new Error('End epoch must be greater than start epoch.');
  }

  const subredditsRaw = parseCommaList(raw.subreddits);
  const subreddits = subredditsRaw.length > 0 ? subredditsRaw : [undefined];
  const productDescription = normalizeDescription(raw.productDescription, product);

  return {
    product,
    productDescription,
    startEpoch,
    endEpoch,
    monthsPerBatch,
    pageSize,
    concurrency,
    model: raw.model?.trim() || 'gpt-5-nano-2025-08-07',
    contextTokens,
    outputReserve,
    requestSpacingMs,
    subreddits,
  };
}

async function gatherCommentsAcrossTargets(context: CommonContext, cache: FileCache) {
  const dedup = new Map<string, RedditComment>();
  for (const subreddit of context.subreddits) {
    const comments = await fetchCommentsForSubreddit(context, cache, subreddit);
    for (const comment of comments) {
      dedup.set(comment.id, comment);
    }
  }
  return [...dedup.values()].sort((a, b) => a.createdUtc - b.createdUtc);
}

async function analyzeTarget(
  context: CommonContext,
  cache: FileCache,
  analyzer: CommentAnalyzer,
  categories: CategoryDefinition[],
  subreddit: string | undefined,
) {
  const comments = await fetchCommentsForSubreddit(context, cache, subreddit);
  if (comments.length === 0) {
    console.log(`No matching Reddit comments found${subreddit ? ` r/${subreddit}` : ''}.`);
    return;
  }

  const safeProduct = context.product.toLowerCase().replace(/[^a-z0-9-_]+/g, '-');
  const safeSubreddit = subreddit ? `r-${subreddit.toLowerCase().replace(/[^a-z0-9-_]+/g, '-')}` : null;
  const isoStamp = new Date().toISOString().replace(/[:]/g, '-');
  const descriptor = safeSubreddit ? `${safeProduct}_${safeSubreddit}` : safeProduct;
  const filename = `${isoStamp}_${descriptor}.csv`;
  const outputPath = path.join('output', filename);
  const writer = await CsvStreamWriter.create(outputPath);
  let written = 0;

  const analysis = await analyzer.analyze(comments, {
    collectClassifications: false,
    seedCategories: categories,
    onClassification: async (classification) => {
      await writer.writeRow(classificationToRow(classification));
      written += 1;
    },
  });

  await writer.close();
  console.log(`Wrote ${written} rows to ${outputPath}`);
  if (analysis.categories.length === 0) {
    console.log('Used pre-defined categories.');
  }
}

async function fetchCommentsForSubreddit(
  context: CommonContext,
  cache: FileCache,
  subreddit: string | undefined,
) {
  const suffix = subreddit ? ` r/${subreddit}` : '';
  console.log(
    `Fetching Reddit comments for "${context.product}"${suffix} from ${new Date(context.startEpoch * 1000).toISOString()} to ${new Date(context.endEpoch * 1000).toISOString()}...`,
  );

  const redditClient = new PullpushClient({
    product: context.product,
    subreddit,
    cache,
    monthsPerBatch: context.monthsPerBatch,
    startEpoch: context.startEpoch,
    endEpoch: context.endEpoch,
    pageSize: context.pageSize,
    requestSpacingMs: context.requestSpacingMs,
    logger: createLogger('pullpush', subreddit),
  });

  const comments = await redditClient.fetchAll();
  const normalized = subreddit?.toLowerCase();
  const filtered = normalized ? comments.filter((c) => c.subreddit.toLowerCase() === normalized) : comments;
  console.log(`Fetched ${filtered.length} comments${suffix ? ' after subreddit filter' : ''}.`);
  return filtered;
}

function createAnalyzer(
  openai: OpenAI,
  llmCache: LlmCache,
  context: CommonContext,
  logger: (message: string) => void,
) {
  return new CommentAnalyzer(openai, llmCache, {
    product: context.product,
    model: context.model,
    productDescription: context.productDescription,
    contextTokens: context.contextTokens,
    outputReserveTokens: context.outputReserve,
    concurrency: context.concurrency,
    logger,
  });
}

async function readCategories(filePath: string): Promise<CategoryDefinition[]> {
  let raw: string;
  try {
    raw = await fs.readFile(filePath, 'utf8');
  } catch (error: any) {
    if (error?.code === 'ENOENT') {
      throw new Error(`Categories file not found at ${filePath}`);
    }
    throw error;
  }

  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    throw new Error('Categories JSON must be an array.');
  }
  return parsed as CategoryDefinition[];
}

function requireApiKey(): string {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is missing from the environment.');
  }
  return apiKey;
}

function createLogger(scope: string, subreddit?: string) {
  return (message: string) => console.log(`[${scope}${subreddit ? `:${subreddit}` : ''}] ${message}`);
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

function normalizeDescription(description: string | undefined, product: string): string {
  const trimmed = description?.trim();
  if (trimmed && trimmed.length > 0) {
    return trimmed;
  }
  return `The product "${product}".`;
}

function classificationToRow(classification: CommentClassification): CsvRow {
  const isoDate = epochToIso(classification.comment.createdUtc);
  return {
    timestamp: classification.comment.createdUtc,
    iso_date: isoDate,
    iso_year: new Date(classification.comment.createdUtc * 1000).getUTCFullYear().toString(),
    category: classification.category,
    sentiment_keyword: classification.sentiment,
    sentiment_number: sentimentScore(classification.sentiment),
    category_confidence: classification.categoryConfidence,
    sentiment_confidence: classification.sentimentConfidence,
    summary: classification.summary,
    subreddit: classification.comment.subreddit,
    comment_link: `https://reddit.com${classification.comment.permalink}`,
  } satisfies CsvRow;
}

function sentimentScore(sentiment: Sentiment): number {
  const mapping: Record<Sentiment, number> = {
    'very-negative': -2,
    negative: -1,
    neutral: 0,
    positive: 1,
    'very-positive': 2,
  };
  return mapping[sentiment];
}
