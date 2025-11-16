import { compileJotSchema, jot, type InferJot } from '../jot.js';

const categoryNode = jot.object({
  slug: jot.string({ description: 'Lowercase hyphenated identifier for the category.' }),
  label: jot.string({ description: 'Readable label for the grouping.' }),
  description: jot.string({ description: 'One-sentence explanation of the shared theme.' }),
  signals: jot.array(jot.string({ description: 'Short cues indicating this category.' })),
});

const categoryListNode = jot.object({
  categories: jot.array(categoryNode, {
    description: 'Distinct categories covering every supplied Reddit comment.',
  }),
});

export type CategoryListPayload = InferJot<typeof categoryListNode>;

export const categoryListFormat = compileJotSchema('reddit_comment_categories', categoryListNode);

const confidenceValues = ['very-low', 'low', 'medium', 'high', 'very-high'] as const;
const sentimentValues = ['very-negative', 'negative', 'neutral', 'positive', 'very-positive'] as const;

const classificationNode = jot.object({
  category: jot.string({ description: 'Slug of the best matching category. Use "unrelated" when the comment is off-topic.' }),
  sentiment: jot.enum(sentimentValues, { description: 'Overall sentiment expressed about the product.' }),
  categoryConfidence: jot.enum(confidenceValues, { description: 'Confidence in the category assignment.' }),
  sentimentConfidence: jot.enum(confidenceValues, { description: 'Confidence in the sentiment assignment.' }),
  summary: jot.string({ description: 'One dense sentence summarizing the user point about the product.' }),
});

export type ClassificationPayload = InferJot<typeof classificationNode>;

export const classificationFormat = compileJotSchema('reddit_comment_classification', classificationNode);
