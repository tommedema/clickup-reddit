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
import { shuffle } from '../utils/shuffle.js';

const CATEGORY_SAMPLE_RESERVE = 128000;
const DEFAULT_CONTEXT_TOKENS = 400000;
const CATEGORY_BUDGET_RATIO = 0.7;
const MIN_CATEGORY_BUDGET = 2000;
const DEFAULT_CATEGORY_ITERATIONS = 10;
const DEFAULT_CATEGORY_SAMPLE_SIZE = 100;
const DEFAULT_CATEGORY_SAMPLE_LENGTH = 500;

export interface CommentAnalyzerOptions {
  product: string;
  model: string;
  productDescription: string;
  contextTokens?: number;
  outputReserveTokens?: number;
  concurrency?: number;
  categoryIterations?: number;
  categorySampleSize?: number;
  categorySampleLength?: number;
  logger?: (message: string) => void;
}

export interface AnalyzeOptions {
  onClassification?: (classification: CommentClassification) => Promise<void> | void;
  collectClassifications?: boolean;
  onCategories?: (categories: CategoryDefinition[]) => Promise<void> | void;
  seedCategories?: CategoryDefinition[];
}

export class CommentAnalyzer {
  private readonly product: string;
  private readonly model: string;
  private readonly contextTokens: number;
  private readonly reserveTokens: number;
  private readonly concurrency: number;
  private readonly logger: ((message: string) => void) | undefined;
  private readonly productDescription: string;
  private readonly categoryIterations: number;
  private readonly categorySampleSize: number;
  private readonly categorySampleLength: number;

  constructor(
    private readonly client: OpenAI,
    private readonly llmCache: LlmCache,
    options: CommentAnalyzerOptions,
  ) {
    this.product = options.product;
    this.model = options.model;
    this.contextTokens = options.contextTokens ?? DEFAULT_CONTEXT_TOKENS;
    this.reserveTokens = options.outputReserveTokens ?? CATEGORY_SAMPLE_RESERVE;
    this.concurrency = options.concurrency ?? 20;
    this.logger = options.logger;
    this.productDescription = options.productDescription;
    this.categoryIterations = options.categoryIterations ?? DEFAULT_CATEGORY_ITERATIONS;
    this.categorySampleSize = options.categorySampleSize ?? DEFAULT_CATEGORY_SAMPLE_SIZE;
    this.categorySampleLength = options.categorySampleLength ?? DEFAULT_CATEGORY_SAMPLE_LENGTH;
  }

  async analyze(comments: RedditComment[], options: AnalyzeOptions = {}): Promise<AnalysisResult> {
    if (comments.length === 0) {
      return { categories: [], classifications: [] };
    }

    this.logger?.(`Analyzing ${comments.length} Reddit comments with ${this.model}...`);

    const categories = options.seedCategories ?? (await this.discoverCategories(comments));
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
    const iterationResults: CategoryDefinition[] = [];

    for (let iteration = 0; iteration < this.categoryIterations; iteration += 1) {
      const sampleComments = this.sampleComments(comments, this.categorySampleSize);
      if (sampleComments.length === 0) {
        break;
      }

      const samples = sampleComments.map((comment) => this.formatSample(comment));
      const prompt = this.buildCategoryPrompt(samples, comments.length);

      const cachePayload = {
        type: 'category-iteration',
        version: 1,
        model: this.model,
        product: this.product,
        productDescription: this.productDescription,
        totalComments: comments.length,
        iteration,
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
          { role: 'user', content: [{ type: 'input_text', text: prompt }] },
        ],
        text: { format: categoryListFormat },
      });

        const parsedPayload = response.output_parsed as CategoryListPayload | null;
        if (!parsedPayload) {
          throw new Error('OpenAI response did not include parsed categories.');
        }
        return parsedPayload;
      });

      iterationResults.push(
        ...parsed.categories.map((category) => ({
          slug: category.slug,
          label: category.label,
          description: category.description,
          signals: category.signals,
        })),
      );

      this.logger?.(
        `Iteration ${iteration + 1}/${this.categoryIterations} categories: ${parsed.categories
          .map((cat) => cat.slug)
          .join(', ')}`,
      );
    }

    return this.consolidateCategories(iterationResults, comments.length);
  }

  async generateCategories(comments: RedditComment[]): Promise<CategoryDefinition[]> {
    return this.discoverCategories(comments);
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
    const prompt = `Product: ${this.product}\nComment ${index} of ${total}. Only assign a category if the comment clearly refers to ${this.product} (otherwise mark it "unrelated" and keep the sentiment neutral). Use the category list below, defaulting to "miscellaneous" only when a product-specific comment doesn't fit another category. Provide an information-dense summary.\n\nAvailable categories:\n${categoryText}\n\nComment metadata:\n- Subreddit: ${comment.subreddit}\n- Created: ${epochToIso(comment.createdUtc)}\n- Permalink: https://reddit.com${comment.permalink}\n\nBody:\n${comment.body}`;

    this.logger?.(`Classifying comment ${index}/${total}`);

    const cachePayload = {
      type: 'classification',
      version: 1,
      model: this.model,
      product: this.product,
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
    const snippet = truncate(comment.body.trim(), this.categorySampleLength);
    return `#${comment.subreddit} | ${iso}\n${snippet}`;
  }

  private buildCategoryPrompt(samples: string[], totalComments: number): string {
    const header = `Below are ${samples.length} sampled comments (out of ${totalComments}) mentioning ${this.product}. Group them into the a comprehensive and exhaustive list of single-word category labels (typically 7-12). Each category must have:\n- slug: lowercase words with hyphens\n- label: short readable name\n- description: one sentence\n- signals: 2-4 cues describing when to assign the category\n\nEnsure the output always includes two special slugs: "unrelated" (for comments unlikely to be about the product description) and "miscellaneous" (for on-topic comments that do not match any other category).`;
    return `${header}\n\nSamples:\n${samples.join('\n\n')}`;
  }

  private buildCategorySystemPrompt(): string {
    return `You are an insight analyst summarizing real Reddit feedback about ${this.product}. Product summary: ${this.productDescription}. Identify distinct single-word category labels that marketing, product, or support teams would use. Always include two special categories: one with slug "unrelated" for comments that are unlikely to be referring to the product "${this.product}", and another with slug "miscellaneous" for relevant comments that do not belong elsewhere.`;
  }

  private buildClassificationSystemPrompt(): string {
    return `Return structured data only. The product being studied is: ${this.productDescription}. When a comment is unlikely to refer to this product, set the category slug to "unrelated" and the sentiment to "neutral". When it is about the product but does not clearly align with another category, set the slug to "miscellaneous".`;
  }

  private sampleComments(comments: RedditComment[], size: number): RedditComment[] {
    return shuffle([...comments]).slice(0, Math.min(size, comments.length));
  }

  private async consolidateCategories(
    candidates: CategoryDefinition[],
    totalComments: number,
  ): Promise<CategoryDefinition[]> {
    const uniqueCandidates = this.dedupCategories(candidates);
    if (uniqueCandidates.length === 0) {
      return this.ensureMandatoryCategories([]);
    }

    const candidateList = uniqueCandidates
      .map((category) => `- ${category.slug} (${category.label}): ${category.description}. Signals: ${category.signals.join('; ')}`)
      .join('\n');

    const prompt = this.buildConsolidationPrompt(candidateList, uniqueCandidates.length, totalComments);

    const consolidationModel = 'gpt-5.1-2025-11-13';
    const cachePayload = {
      type: 'category-consolidation',
      version: 1,
      model: consolidationModel,
      product: this.product,
      productDescription: this.productDescription,
      totalComments,
      candidateHash: checksumFrom(candidateList),
    } as const;

    const parsed = await this.llmCache.getOrCompute(cachePayload, async () => {
      const response = await this.client.responses.parse({
        model: consolidationModel,
        input: [
          {
            role: 'system',
            content: [
              {
                type: 'input_text',
                text: `You are merging candidate categories about ${this.product}. Consolidate them into a guardrail-aware set and output structured JSON matching the schema.`,
              },
            ],
          },
          { role: 'user', content: [{ type: 'input_text', text: `${prompt}\n\nCandidate list:\n${candidateList}` }] },
        ],
        text: { format: categoryListFormat },
      });

      const parsedPayload = response.output_parsed as CategoryListPayload | null;
      if (!parsedPayload) {
        throw new Error('OpenAI response did not include parsed categories.');
      }
      return parsedPayload;
    });

    const mapped = parsed.categories.map((category) => ({
      slug: category.slug,
      label: category.label,
      description: category.description,
      signals: category.signals,
    }));

    const finalCategories = this.ensureMandatoryCategories(mapped);
    this.logger?.(
      `Consolidated categories: ${finalCategories.map((cat) => cat.slug).join(', ')}`,
    );
    return finalCategories;
  }

  private buildConsolidationPrompt(candidateList: string, candidateCount: number, totalComments: number): string {
    return `You have ${candidateCount} candidate categories derived from ${totalComments} Reddit comments about ${this.product}. Consolidate them into a comprehensive deduped list of distinct single-word category labels, merging overlapping or redundant concepts (e.g., "bug", "performance", and "reliability" would roll into a unified stability bucket). Ensure the final set explicitly contains the slugs "unrelated" and "miscellaneous" (mandatory even if synonyms like "other" were suggested). Prefer the clearest descriptions and strongest signal lists when merging, and avoid reusing synonyms.`;
  }

  private dedupCategories(categories: CategoryDefinition[]): CategoryDefinition[] {
    const seen = new Map<string, CategoryDefinition>();
    for (const category of categories) {
      const existing = seen.get(category.slug);
      if (!existing) {
        seen.set(category.slug, category);
        continue;
      }
      if (category.description.length > (existing.description.length ?? 0)) {
        seen.set(category.slug, category);
      }
    }
    return Array.from(seen.values());
  }


  private ensureMandatoryCategories(categories: CategoryDefinition[]): CategoryDefinition[] {
    const existingSlugs = new Set(categories.map((cat) => cat.slug));
    const required = [this.buildUnrelatedCategory(), this.buildMiscCategory()];
    const merged = [...categories];
    for (const mandatory of required) {
      if (!existingSlugs.has(mandatory.slug)) {
        merged.push(mandatory);
      }
    }
    return merged;
  }

  private buildUnrelatedCategory(): CategoryDefinition {
    return {
      slug: 'unrelated',
      label: 'Unrelated',
      description: `Comments unlikely to refer to ${this.product}, or the described product experience.`,
      signals: ['mentions other products', 'gaming slang "click up"', 'spam or jokes not about productivity'],
    };
  }

  private buildMiscCategory(): CategoryDefinition {
    return {
      slug: 'miscellaneous',
      label: 'Miscellaneous',
      description: `On-topic ${this.product} feedback that does not belong to any other category.`,
      signals: ['general praise/complaints', 'mixed or unique workflows', 'edge cases'],
    };
  }
}
