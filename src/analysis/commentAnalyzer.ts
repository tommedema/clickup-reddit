import OpenAI from 'openai';
import pLimit from 'p-limit';
import type {
  AnalysisResult,
  CategoryDefinition,
  CommentClassification,
  RedditComment,
  Sentiment,
  ConfidenceBand,
} from '../types/index.js';
import { approximateTokenCount, truncate } from '../utils/text.js';
import { epochToIso } from '../utils/time.js';
import { checksumFrom } from '../utils/hash.js';
import { categoryListFormat, classificationFormat, type CategoryListPayload, type ClassificationPayload } from './schemas.js';
import { LlmCache } from './llmCache.js';

const CATEGORY_SAMPLE_RESERVE = 800;
const MAX_CONTEXT_TOKENS = 8000;
const MAX_CATEGORY_SAMPLE_BODY_CHARS = 1500;

export interface CommentAnalyzerOptions {
  keyword: string;
  model: string;
  productDescription: string;
  contextTokens?: number;
  outputReserveTokens?: number;
  concurrency?: number;
  logger?: (message: string) => void;
}

export interface AnalyzeOptions {
  onClassification?: (classification: CommentClassification) => Promise<void> | void;
  collectClassifications?: boolean;
  onCategories?: (categories: CategoryDefinition[]) => Promise<void> | void;
}

export class CommentAnalyzer {
  private readonly keyword: string;
  private readonly model: string;
  private readonly contextTokens: number;
  private readonly reserveTokens: number;
  private readonly concurrency: number;
  private readonly logger: ((message: string) => void) | undefined;
  private readonly productDescription: string;

  constructor(
    private readonly client: OpenAI,
    private readonly llmCache: LlmCache,
    options: CommentAnalyzerOptions,
  ) {
    this.keyword = options.keyword;
    this.model = options.model;
    this.contextTokens = options.contextTokens ?? MAX_CONTEXT_TOKENS;
    this.reserveTokens = options.outputReserveTokens ?? CATEGORY_SAMPLE_RESERVE;
    this.concurrency = options.concurrency ?? 10;
    this.logger = options.logger;
    this.productDescription = options.productDescription;
  }

  async analyze(comments: RedditComment[], options: AnalyzeOptions = {}): Promise<AnalysisResult> {
    if (comments.length === 0) {
      return { categories: [], classifications: [] };
    }

    this.logger?.(`Analyzing ${comments.length} Reddit comments with ${this.model}...`);

    const categories = await this.discoverCategories(comments);
    if (options.onCategories) {
      await options.onCategories(categories);
    }
    const limit = pLimit(this.concurrency);
    const sorted = [...comments].sort((a, b) => b.createdUtc - a.createdUtc);
    const shouldCollect = options.collectClassifications ?? !options.onClassification;
    const collected: CommentClassification[] = [];
    const pending = new Map<number, CommentClassification>();
    let nextEmitIndex = 0;

    const emitIfReady = async () => {
      while (pending.has(nextEmitIndex)) {
        const ready = pending.get(nextEmitIndex)!;
        pending.delete(nextEmitIndex);
        if (shouldCollect) {
          collected.push(ready);
        }
        if (options.onClassification) {
          await options.onClassification(ready);
        }
        nextEmitIndex += 1;
      }
    };

    const tasks = sorted.map((comment, index) =>
      limit(async () => {
        const classification = await this.classifyComment(comment, categories, index + 1, sorted.length);
        pending.set(index, classification);
        await emitIfReady();
      }),
    );
    await Promise.all(tasks);
    await emitIfReady();

    return { categories, classifications: shouldCollect ? collected : [] };
  }

  private async discoverCategories(comments: RedditComment[]): Promise<CategoryDefinition[]> {
    const usableBudget = Math.max(this.contextTokens - this.reserveTokens, 500);
    const ordered = [...comments].sort((a, b) => a.createdUtc - b.createdUtc);

    const samples: string[] = [];
    let usedTokens = 0;
    for (const comment of ordered) {
      const snippet = this.formatSample(comment);
      const tokens = approximateTokenCount(snippet);
      if (usedTokens + tokens > usableBudget && samples.length > 0) {
        break;
      }
      samples.push(snippet);
      usedTokens += tokens;
    }

    const prompt = this.buildCategoryPrompt(samples, comments.length);

    const cachePayload = {
      type: 'category-list',
      version: 1,
      model: this.model,
      keyword: this.keyword,
      productDescription: this.productDescription,
      totalComments: comments.length,
      sampleHash: checksumFrom(samples),
    } as const;

    const parsed = await this.llmCache.getOrCompute(cachePayload, async () => {
      const response = await this.client.responses.parse({
        model: this.model,
        input: [
          {
            role: 'system',
            content: [
              {
                type: 'input_text',
                text: this.buildCategorySystemPrompt(),
              },
            ],
          },
          { role: 'user', content: [{ type: 'input_text', text: `${prompt}\n\nProduct summary: ${this.productDescription}` }] },
        ],
        text: { format: categoryListFormat },
      });

      const parsedPayload = response.output_parsed as CategoryListPayload | null;
      if (!parsedPayload) {
        throw new Error('OpenAI response did not include parsed categories.');
      }
      return parsedPayload;
    });

    if (!parsed) {
      throw new Error('OpenAI response did not include parsed categories.');
    }

    return parsed.categories.map((category) => ({
      slug: category.slug,
      label: category.label,
      description: category.description,
      signals: category.signals,
    }));
  }

  private async classifyComment(
    comment: RedditComment,
    categories: CategoryDefinition[],
    index: number,
    total: number,
  ): Promise<CommentClassification> {
    const categoryText = categories
      .map((cat) => `- ${cat.slug} (${cat.label}): ${cat.description}. Signals: ${cat.signals.join('; ')}`)
      .join('\n');
    const prompt = `Keyword: ${this.keyword}\nComment ${index} of ${total}. Use the category list below, defaulting to "unrelated" for noise. Provide an information-dense summary.\n\nAvailable categories:\n${categoryText}\n\nComment metadata:\n- Subreddit: ${comment.subreddit}\n- Created: ${epochToIso(comment.createdUtc)}\n- Permalink: https://reddit.com${comment.permalink}\n\nBody:\n${comment.body}`;

    this.logger?.(`Classifying comment ${index}/${total}`);

    const cachePayload = {
      type: 'classification',
      version: 1,
      model: this.model,
      keyword: this.keyword,
      productDescription: this.productDescription,
      commentId: comment.id,
      commentBody: comment.body,
      categoriesSnapshot: categories.map((cat) => ({
        slug: cat.slug,
        label: cat.label,
        description: cat.description,
        signals: cat.signals,
      })),
    } as const;

    const parsed = await this.llmCache.getOrCompute(cachePayload, async () => {
      const response = await this.client.responses.parse({
        model: this.model,
        input: [
          {
            role: 'system',
            content: [
              {
                type: 'input_text',
                text: this.buildClassificationSystemPrompt(),
              },
            ],
          },
          { role: 'user', content: [{ type: 'input_text', text: `${prompt}\n\nProduct summary: ${this.productDescription}` }] },
        ],
        text: { format: classificationFormat },
      });

      const parsedPayload = response.output_parsed as ClassificationPayload | null;
      if (!parsedPayload) {
        throw new Error('OpenAI response did not include parsed classification output.');
      }

      return parsedPayload;
    });

    if (!parsed) {
      throw new Error('OpenAI response did not include parsed classification output.');
    }

    return {
      comment,
      category: parsed.category,
      sentiment: parsed.sentiment as Sentiment,
      summary: parsed.summary,
      categoryConfidence: parsed.categoryConfidence as ConfidenceBand,
      sentimentConfidence: parsed.sentimentConfidence as ConfidenceBand,
    } satisfies CommentClassification;
  }

  private formatSample(comment: RedditComment): string {
    const iso = epochToIso(comment.createdUtc);
    const snippet = truncate(comment.body.trim(), MAX_CATEGORY_SAMPLE_BODY_CHARS);
    return `#${comment.subreddit} | ${iso}\n${snippet}`;
  }

  private buildCategoryPrompt(samples: string[], totalComments: number): string {
    const header = `Below are ${samples.length} sampled comments (out of ${totalComments}) mentioning ${this.keyword}. Group them into the smallest useful list of categories (typically 3-8). Each category must have:\n- slug: lowercase words with hyphens\n- label: short readable name\n- description: one sentence\n- signals: 2-4 cues describing when to assign the category\n\nIf any sampled comment is clearly unrelated to the product description provided, ensure one category uses the slug "unrelated".`;
    return `${header}\n\nSamples:\n${samples.join('\n\n')}`;
  }

  private buildCategorySystemPrompt(): string {
    return `You are an insight analyst summarizing real Reddit feedback about ${this.keyword}. Product summary: ${this.productDescription}. Identify distinct categories that marketing, product, or support teams would use. Always include an "unrelated" category whenever a comment is unlikely to be about this product.`;
  }

  private buildClassificationSystemPrompt(): string {
    return `Return structured data only. The product being studied is: ${this.productDescription}. When a comment is unlikely to refer to this product, set the category slug to "unrelated".`;
  }
}
