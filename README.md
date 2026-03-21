# rerank-lite

Lightweight retrieval reranker for JavaScript/TypeScript. Takes a query and a list of candidate documents from a first-stage retriever and reorders them by relevance using heuristic BM25/TF-IDF scoring, a pluggable LLM-as-judge function, or a hybrid of both.

## Installation

```bash
npm install rerank-lite
```

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
  console.log(`[rank ${result.newRank}] ${result.document.id} — score: ${result.score.toFixed(4)}`)
}
```

## `rerank(query, documents, options?)`

Reranks an array of documents against a query. Returns a `RerankResult[]` sorted by relevance score descending.

```typescript
import { rerank, Document, RerankOptions } from 'rerank-lite'

const results = await rerank(query, documents, {
  mode: 'heuristic',   // 'heuristic' | 'llm' | 'hybrid'
  topK: 5,             // return only top 5 results
  minScore: 0.3,       // filter out results with score < 0.3
  weights: {
    bm25: 0.5,         // default 0.5
    tfidf: 0.3,        // default 0.3
    position: 0.2,     // default 0.2 (favour original top results)
  },
})
```

## `createReranker(config?)`

Factory that returns a configured `Reranker` instance for repeated use across many queries.

```typescript
import { createReranker } from 'rerank-lite'

const reranker = createReranker({
  mode: 'heuristic',
  topK: 10,
})

const results1 = await reranker.rerank('first query', docs1)
const results2 = await reranker.rerank('second query', docs2)
```

Options passed directly to `reranker.rerank()` override the config:

```typescript
const results = await reranker.rerank('query', docs, { topK: 3 })
```

## RerankMode

| Mode | Description |
|---|---|
| `heuristic` | Zero-dependency BM25 + TF-IDF + position weighted scoring. Default. |
| `llm` | Uses a caller-supplied `judgeFn` to score each document. |
| `hybrid` | 50/50 blend of heuristic and LLM scores. |

### LLM mode

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

## API

### `Document`

```typescript
interface Document {
  id: string
  text: string
  score?: number           // original retrieval score
  metadata?: Record<string, unknown>
}
```

### `RerankResult`

```typescript
interface RerankResult {
  document: Document
  score: number            // reranking score 0-1
  originalRank: number     // rank before reranking (0-based index in input array)
  newRank: number          // rank after reranking (0-based)
  explanation?: string     // optional debug info
}
```

### `RerankOptions`

```typescript
interface RerankOptions {
  topK?: number            // return only top K results (default: all)
  minScore?: number        // filter results below this score
  mode?: RerankMode        // default: 'heuristic'
  judgeFn?: JudgeFn        // required for 'llm' mode
  weights?: ScoringWeights // custom scoring weights
}
```

### `ScoringWeights`

```typescript
interface ScoringWeights {
  bm25?: number       // default 0.5
  tfidf?: number      // default 0.3
  position?: number   // default 0.2
}
```

## Integration

```typescript
// After vector search or BM25 retrieval, rerank the top candidates
const candidates = await vectorDB.query(embedding, { topK: 50 })
const reranked = await rerank(query, candidates, { topK: 10 })
// Pass top 10 to context packer or LLM
```

## License

MIT
