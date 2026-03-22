# rerank-lite

Lightweight retrieval reranker for JavaScript and TypeScript -- zero dependencies, three scoring modes, one API.

[![npm version](https://img.shields.io/npm/v/rerank-lite.svg)](https://www.npmjs.com/package/rerank-lite)
[![npm downloads](https://img.shields.io/npm/dt/rerank-lite.svg)](https://www.npmjs.com/package/rerank-lite)
[![license](https://img.shields.io/npm/l/rerank-lite.svg)](https://github.com/SiluPanda/rerank-lite/blob/master/LICENSE)
[![node](https://img.shields.io/node/v/rerank-lite.svg)](https://nodejs.org)

---

## Description

`rerank-lite` takes a query and a list of candidate documents from a first-stage retriever (vector search, BM25, or fusion) and reorders them by relevance. It is the second stage of a two-stage retrieval pipeline: stage one retrieves candidates quickly with approximate methods, stage two reranks them precisely for the final result set.

The package ships three scoring modes that share a single `rerank()` API and return the same `RerankResult[]` output:

- **Heuristic** -- BM25 + TF-IDF + position-based scoring with zero external dependencies. Works immediately after install.
- **LLM** -- Delegates scoring to a caller-supplied judge function. Bring any LLM provider (OpenAI, Anthropic, local Ollama, etc.).
- **Hybrid** -- A 50/50 blend of heuristic and LLM scores for balanced accuracy and cost.

All scores are normalized to the [0, 1] range. The output is ready to pass to a context packer, fusion ranker, or directly into an LLM prompt.

---

## Installation

```bash
npm install rerank-lite
```

Requires Node.js 18 or later.

---

## Quick Start

```typescript
import { rerank } from 'rerank-lite'

const documents = [
  { id: 'doc-0', text: 'Machine learning model training with neural networks' },
  { id: 'doc-1', text: 'Quick sort algorithm and data structures' },
  { id: 'doc-2', text: 'Deep learning transformer architecture for NLP' },
]

const results = await rerank('machine learning neural network', documents)

for (const result of results) {
  console.log(`[rank ${result.newRank}] ${result.document.id} -- score: ${result.score.toFixed(4)}`)
}
```

---

## Features

- **Three scoring modes** -- heuristic, LLM-as-judge, and hybrid, all behind one function signature.
- **Zero runtime dependencies** -- heuristic mode is pure TypeScript with no external packages.
- **BM25 scoring** -- corpus-aware term frequency / inverse document frequency with configurable k1 and b parameters, normalized via sigmoid.
- **TF-IDF cosine similarity** -- builds TF-IDF vectors for the query and each document, scores by cosine similarity against the corpus.
- **Position bias** -- original retrieval rank feeds into the composite score, preserving signal from the first-stage retriever.
- **Configurable weights** -- tune the relative contribution of BM25, TF-IDF, and position scoring.
- **Top-K and minimum score filtering** -- limit output to the top K results and filter below a score threshold.
- **Pluggable LLM judge** -- pass any async function that scores a (query, document) pair. No SDK lock-in.
- **Factory pattern** -- `createReranker()` returns a preconfigured instance for repeated use across many queries.
- **Full TypeScript support** -- ships with declaration files and source maps. All types are exported.

---

## API Reference

### `rerank(query, documents, options?)`

Reranks an array of documents against a query. Returns a `Promise<RerankResult[]>` sorted by relevance score descending.

```typescript
import { rerank } from 'rerank-lite'

const results = await rerank('machine learning', documents, {
  mode: 'heuristic',
  topK: 10,
  minScore: 0.3,
  weights: { bm25: 0.5, tfidf: 0.3, position: 0.2 },
})
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | `string` | Yes | The search query to rank documents against. |
| `documents` | `Document[]` | Yes | Array of candidate documents from a first-stage retriever. |
| `options` | `RerankOptions` | No | Configuration for scoring mode, filtering, and weights. |

**Returns:** `Promise<RerankResult[]>` -- results sorted by score descending, with `newRank` assigned as 0-based indices.

**Behavior with empty input:** Returns an empty array when `documents` is empty.

---

### `createReranker(config?)`

Factory that returns a configured `Reranker` instance. Preset options are applied to every `rerank()` call; per-call options override the preset.

```typescript
import { createReranker } from 'rerank-lite'

const reranker = createReranker({
  mode: 'heuristic',
  topK: 10,
  weights: { bm25: 0.6, tfidf: 0.3, position: 0.1 },
})

const results = await reranker.rerank('first query', docs)

// Override topK for this call only
const top3 = await reranker.rerank('second query', docs, { topK: 3 })
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config` | `RerankerConfig` | No | Default configuration applied to all calls. |

**Returns:** `Reranker` -- an object with a `rerank()` method and a read-only `config` property.

---

### Types

#### `Document`

Represents a candidate document to be scored.

```typescript
interface Document {
  id: string                        // Unique identifier for the document
  text: string                      // The text content to score against the query
  score?: number                    // Optional original retrieval score
  metadata?: Record<string, unknown> // Optional pass-through metadata
}
```

#### `RerankResult`

A scored and ranked document returned by `rerank()`.

```typescript
interface RerankResult {
  document: Document    // The original document object, including metadata
  score: number         // Relevance score in [0, 1]
  originalRank: number  // 0-based index in the input array
  newRank: number       // 0-based rank after reranking (0 = most relevant)
  explanation?: string  // Optional debug information
}
```

#### `RerankOptions`

Options for a single `rerank()` call.

```typescript
interface RerankOptions {
  topK?: number           // Return only the top K results (default: all)
  minScore?: number       // Filter out results with a score below this threshold
  mode?: RerankMode       // Scoring mode (default: 'heuristic')
  judgeFn?: JudgeFn       // Required when mode is 'llm' or 'hybrid'
  weights?: ScoringWeights // Custom weights for heuristic scoring components
}
```

#### `RerankMode`

```typescript
type RerankMode = 'heuristic' | 'llm' | 'hybrid'
```

| Mode | Description |
|------|-------------|
| `heuristic` | BM25 + TF-IDF + position weighted scoring. Zero dependencies. Default. |
| `llm` | Delegates scoring entirely to the caller-supplied `judgeFn`. |
| `hybrid` | 50/50 blend of heuristic scores and `judgeFn` scores. |

#### `JudgeFn`

The function signature for LLM-as-judge scoring.

```typescript
type JudgeFn = (query: string, document: string) => Promise<number>
```

Must return a numeric relevance score. Called once per document when mode is `llm` or `hybrid`.

#### `ScoringWeights`

Weights for the three heuristic scoring signals. All are optional and default to the values shown.

```typescript
interface ScoringWeights {
  bm25?: number      // Default: 0.5
  tfidf?: number     // Default: 0.3
  position?: number  // Default: 0.2
}
```

#### `RerankerConfig`

Configuration for the `createReranker()` factory. Same shape as `RerankOptions`.

```typescript
interface RerankerConfig {
  mode?: RerankMode
  topK?: number
  minScore?: number
  judgeFn?: JudgeFn
  weights?: ScoringWeights
}
```

#### `Reranker`

The instance returned by `createReranker()`.

```typescript
interface Reranker {
  rerank(query: string, documents: Document[], options?: RerankOptions): Promise<RerankResult[]>
  readonly config: RerankerConfig
}
```

---

## Configuration

### Scoring Weights

The heuristic composite score is computed as:

```
score = (bm25_weight * bm25_score) + (tfidf_weight * tfidf_score) + (position_weight * position_score)
```

Default weights:

| Signal | Default Weight | Description |
|--------|---------------|-------------|
| `bm25` | 0.5 | BM25 term frequency / inverse document frequency score, normalized to [0, 1] via sigmoid. |
| `tfidf` | 0.3 | TF-IDF cosine similarity between query and document vectors. |
| `position` | 0.2 | Position bias: `1 - (originalRank / totalDocuments)`. Preserves signal from the first-stage retriever. |

Override weights to tune for your retrieval domain:

```typescript
// Favor lexical match, ignore original ordering
const results = await rerank('exact keyword query', docs, {
  weights: { bm25: 0.7, tfidf: 0.3, position: 0.0 },
})

// Trust the first-stage retriever, use reranking as a tiebreaker
const results = await rerank('semantic query', docs, {
  weights: { bm25: 0.2, tfidf: 0.2, position: 0.6 },
})
```

### BM25 Parameters

The BM25 scorer accepts standard tuning parameters `k1` (term frequency saturation, default 1.5) and `b` (document length normalization, default 0.75). These are set internally and follow the Robertson/Zaragoza BM25 formulation.

---

## Error Handling

`rerank()` handles edge cases gracefully without throwing:

- **Empty documents array** -- returns an empty array `[]`.
- **No matching terms** -- all documents receive a baseline score (sigmoid of zero for BM25, zero for TF-IDF). Documents are still ranked by position bias.
- **Empty query or document text** -- TF-IDF returns 0 for empty strings. BM25 returns the sigmoid baseline (0.5).

When using `llm` or `hybrid` mode, errors from the `judgeFn` propagate as-is. Wrap your judge function with try/catch if you need graceful degradation:

```typescript
const safejudgeFn = async (query: string, doc: string): Promise<number> => {
  try {
    return await yourLLMScorer(query, doc)
  } catch {
    return 0 // Fallback score on failure
  }
}

const results = await rerank('query', docs, { mode: 'llm', judgeFn: safejudgeFn })
```

---

## Advanced Usage

### LLM-as-Judge Mode

Pass any async function that returns a relevance score. The function is called once per document.

```typescript
import { rerank } from 'rerank-lite'
import OpenAI from 'openai'

const openai = new OpenAI()

const judgeFn = async (query: string, document: string): Promise<number> => {
  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      {
        role: 'user',
        content: `Rate the relevance of this document to the query on a scale from 0 to 1.\nQuery: ${query}\nDocument: ${document}\nRespond with only a number between 0 and 1.`,
      },
    ],
  })
  return parseFloat(response.choices[0].message.content ?? '0')
}

const results = await rerank('machine learning', docs, { mode: 'llm', judgeFn })
```

### Hybrid Mode

Combines heuristic scoring with LLM judgment. The final score is a 50/50 blend:

```
hybrid_score = 0.5 * heuristic_score + 0.5 * llm_score
```

```typescript
const results = await rerank('machine learning', docs, {
  mode: 'hybrid',
  judgeFn,
  weights: { bm25: 0.5, tfidf: 0.3, position: 0.2 }, // Applies to heuristic half
})
```

### RAG Pipeline Integration

Use reranking as stage two after vector search or BM25 retrieval:

```typescript
// Stage 1: Retrieve candidates from your vector database
const candidates = await vectorDB.query(embedding, { topK: 50 })

// Stage 2: Rerank for precision
const reranked = await rerank(query, candidates, { topK: 10, minScore: 0.4 })

// Stage 3: Pass to LLM context
const context = reranked.map(r => r.document.text).join('\n\n')
```

### Reusable Reranker Instance

For applications that rerank many queries with the same configuration:

```typescript
import { createReranker } from 'rerank-lite'

const reranker = createReranker({
  mode: 'heuristic',
  topK: 10,
  minScore: 0.3,
  weights: { bm25: 0.6, tfidf: 0.25, position: 0.15 },
})

// Use across multiple queries
const results1 = await reranker.rerank('first query', docs1)
const results2 = await reranker.rerank('second query', docs2)

// Override per-call when needed
const results3 = await reranker.rerank('third query', docs3, { topK: 5 })
```

### Filtering with minScore and topK

`minScore` is applied first, then `topK` limits the output:

```typescript
// Return at most 5 results, all with score >= 0.5
const results = await rerank('query', docs, { topK: 5, minScore: 0.5 })
```

---

## TypeScript

`rerank-lite` is written in TypeScript and ships with declaration files (`.d.ts`) and source maps. All types are exported from the package entry point:

```typescript
import {
  rerank,
  createReranker,
  type Document,
  type RerankResult,
  type RerankOptions,
  type RerankMode,
  type JudgeFn,
  type ScoringWeights,
  type RerankerConfig,
  type Reranker,
} from 'rerank-lite'
```

The package targets ES2022 and emits CommonJS modules. It is compatible with both `require()` and bundlers that resolve the `exports` field.

---

## License

MIT
