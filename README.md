# clickup-reddit

CLI that fetches Reddit comments for a product, prompts OpenAI for qualitative analysis with structured outputs (via the custom jot helper), and writes the enriched results to a CSV.

## Prerequisites

- Node.js 20+
- An `.env` file that sets `OPENAI_API_KEY`

## Install

```bash
npm install
```

## Run

```bash
# 1) Generate categories (writes categories.json by default)
npm run dev -- categories --product clickup --months-per-batch 3 --product-description "ClickUp is software-as-a-service offering various productivity tools."

# 2) Analyze comments with the saved categories
npm run dev -- analyze --product clickup --months-per-batch 3 --product-description "ClickUp is software-as-a-service offering various productivity tools." --categories-input categories.json
```

Constrain both steps to `r/productivity`:

```bash
npm run dev -- categories --product clickup --product-description "ClickUp is a SaaS unifying various work tools into one platform." --subreddits productivity

npm run dev -- analyze --product clickup --product-description "ClickUp is a SaaS unifying various work tools into one platform." --categories-input categories.json --subreddits productivity
```


## Notable Behaviors

- Reddit comments come from the public [pullpush.io](https://api.pullpush.io) search endpoint. Requests are chunked into `N`-month windows (default 6 months) from `start_epoch_s` (default 2017-01-01) through `end_epoch_s` (default now). Each request response is checksummed and cached to `.cache/<namespace>` so identical calls reuse disk hits.
- The same cache layer now wraps OpenAI Responses calls as well. Category discovery and per-comment classifications are keyed by the prompt payload, so re-running the CLI over the same data skips duplicate LLM calls entirely.
- The initial OpenAI call packs as many chronological comments as will fit beneath the configured context window minus an output reserve, then asks the model (default `gpt-5-nano-2025-08-07`) to emit a complete category list using structured outputs powered by the custom `jot` helper. When the source data includes unrelated chatter, the model is instructed to emit an `unrelated` category.
- Every comment is re-run through the same model (default concurrency 20) to label it with one of the discovered categories, a sentiment enum (mapped to numeric scores in the CSV), dense one-sentence summary, and confidence bands for both category & sentiment. Results always include two special categories: `unrelated` (off-topic) and `miscellaneous` (on-topic but not fitting other clusters). Classification responses are cached and the CLI streams each finalized row straight into the CSV once all earlier rows are ready, so memory usage stays low even for large exports.
- You can limit the analysis to a comma-separated list of subreddits via `--subreddits`. Each subreddit is fetched, analyzed, and written to its own CSV, so you can compare niche communities without mixing their feedback.
- Supplying `--product-description "..."` gives the LLM more context and reinforces that anything diverging from that description must be classified into the `unrelated` bucket.
- Categories are generated in a separate `categories` command and saved to JSON, giving you a chance to edit or curate them manually before running `analyze`.
- The final CSV lives in `output/<ISO_TIMESTAMP>_<product>.csv` with columns `timestamp,iso_date,iso_year,category,sentiment_keyword,sentiment_number,category_confidence,sentiment_confidence,summary,subreddit,comment_link` sorted by newest comment first and written incrementally.

## Key options

| Flag | Description | Default |
| --- | --- | --- |
| `--product` | Product to search for | `clickup` |
| `--start`, `--start-epoch` | Start of window (ISO or UNIX seconds) | `2017-01-01` |
| `--end`, `--end-epoch` | End of window (ISO/UNIX, defaults to now) | now |
| `--months-per-batch` | Number of months per pullpush batch | `6` |
| `--page-size` | Comments fetched per API call | `100` |
| `--model` | OpenAI model id | `gpt-5-nano-2025-08-07` |
| `--concurrency` | Parallel OpenAI classifications | `20` |
| `--context-tokens` | Context budget for category discovery | `400000` |
| `--output-reserve` | Tokens held back for structured output | `128000` |
| `--request-spacing` | Minimum milliseconds between pullpush requests (helps respect ~100 req/hr rate limits) | `0` |
| `--subreddits` | Comma-separated list of subreddit names to process individually (`r/` prefix optional) | *(all subreddits)* |
| `--product-description` | One-sentence product summary injected into prompts | `The product "<product>".` |

## CSV caching & rate limiting

Each pullpush request is hashed (method + URL) before execution. Success responses are written to `.cache/pullpush-comments/<checksum>.body` (raw body) plus a metadata file so retries immediately hit disk. On HTTP 429 the client waits for the `Retry-After` header (when present) or uses an exponential backoff multiplier. You can further slow requests via `--request-spacing` to better align with the service’s documented rate limits (100 req/hour in the current default tier).
