export type Sentiment = 'very-negative' | 'negative' | 'neutral' | 'positive' | 'very-positive';

export type ConfidenceBand = 'very-low' | 'low' | 'medium' | 'high' | 'very-high';

export interface RedditComment {
  id: string;
  body: string;
  createdUtc: number;
  permalink: string;
  subreddit: string;
}

export interface CategoryDefinition {
  slug: string;
  label: string;
  description: string;
  signals: string[];
}

export interface CommentClassification {
  comment: RedditComment;
  category: string;
  sentiment: Sentiment;
  summary: string;
  categoryConfidence: ConfidenceBand;
  sentimentConfidence: ConfidenceBand;
}

export interface AnalysisResult {
  categories: CategoryDefinition[];
  classifications: CommentClassification[];
}
