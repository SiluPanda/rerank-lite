import { describe, it, expect, vi } from 'vitest'
import { rerank } from '../rerank.js'
import { createReranker } from '../create-reranker.js'
import { Document } from '../types.js'

const docs: Document[] = [
  { id: 'doc-0', text: 'completely unrelated topic about cooking pasta and recipes' },
  { id: 'doc-1', text: 'machine learning neural network training data' },
  { id: 'doc-2', text: 'machine learning model deep neural network architecture' },
  { id: 'doc-3', text: 'quick sort algorithm and data structures' },
]

describe('rerank()', () => {
  it('returns results sorted by score descending', async () => {
    const results = await rerank('machine learning neural network', docs)
    expect(results.length).toBe(4)
    for (let i = 0; i < results.length - 1; i++) {
      expect(results[i].score).toBeGreaterThanOrEqual(results[i + 1].score)
    }
  })

  it('assigns correct newRank values (0-based ascending)', async () => {
    const results = await rerank('machine learning', docs)
    results.forEach((r, idx) => {
      expect(r.newRank).toBe(idx)
    })
  })

  it('assigns originalRank matching the input array index', async () => {
    const results = await rerank('machine learning', docs)
    const byOriginal = results.slice().sort((a, b) => a.originalRank - b.originalRank)
    byOriginal.forEach((r, idx) => {
      expect(r.originalRank).toBe(idx)
      expect(r.document.id).toBe(docs[idx].id)
    })
  })

  it('topK limits the number of returned results', async () => {
    const results = await rerank('machine learning', docs, { topK: 2 })
    expect(results.length).toBe(2)
  })

  it('topK returns only the highest scoring results', async () => {
    const allResults = await rerank('machine learning', docs)
    const top2 = await rerank('machine learning', docs, { topK: 2 })
    expect(top2[0].document.id).toBe(allResults[0].document.id)
    expect(top2[1].document.id).toBe(allResults[1].document.id)
  })

  it('minScore filters out low-scoring results', async () => {
    const results = await rerank('machine learning neural network', docs, { minScore: 0.6 })
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0.6)
    }
  })

  it('newRank is contiguous after minScore filtering', async () => {
    const results = await rerank('machine learning neural network', docs, { minScore: 0.3 })
    // Verify newRank is 0, 1, 2, ... with no gaps
    results.forEach((r, idx) => {
      expect(r.newRank).toBe(idx)
    })
  })

  it('newRank is contiguous after topK filtering', async () => {
    const results = await rerank('machine learning', docs, { topK: 2 })
    expect(results[0].newRank).toBe(0)
    expect(results[1].newRank).toBe(1)
  })

  it('returns empty array for empty documents', async () => {
    const results = await rerank('machine learning', [])
    expect(results).toEqual([])
  })

  it('mode heuristic is the default', async () => {
    const defaultResults = await rerank('machine learning', docs)
    const explicitResults = await rerank('machine learning', docs, { mode: 'heuristic' })
    expect(defaultResults.map(r => r.document.id)).toEqual(explicitResults.map(r => r.document.id))
  })

  it('mode llm calls judgeFn for each document', async () => {
    const judgeFn = vi.fn().mockResolvedValue(0.8)
    await rerank('query', docs, { mode: 'llm', judgeFn })
    expect(judgeFn).toHaveBeenCalledTimes(docs.length)
    expect(judgeFn).toHaveBeenCalledWith('query', docs[0].text)
  })

  it('mode llm uses judgeFn scores for ranking', async () => {
    // Return distinct scores so ranking is deterministic
    const scores = [0.1, 0.9, 0.7, 0.3]
    const judgeFn = vi.fn().mockImplementation((_q: string, docText: string) => {
      const idx = docs.findIndex(d => d.text === docText)
      return Promise.resolve(scores[idx])
    })
    const results = await rerank('query', docs, { mode: 'llm', judgeFn })
    // doc-1 should rank first (score 0.9)
    expect(results[0].document.id).toBe('doc-1')
    expect(results[0].score).toBe(0.9)
  })

  it('mode hybrid blends heuristic and llm scores', async () => {
    const judgeFn = vi.fn().mockResolvedValue(1.0)
    const hybridResults = await rerank('machine learning', docs, { mode: 'hybrid', judgeFn })
    const heuristicResults = await rerank('machine learning', docs, { mode: 'heuristic' })
    // Hybrid scores should differ from pure heuristic
    const hybridScore0 = hybridResults.find(r => r.document.id === docs[0].id)!.score
    const heuristicScore0 = heuristicResults.find(r => r.document.id === docs[0].id)!.score
    // With judgeFn always returning 1.0, hybrid = 0.5*heuristic + 0.5*1.0 > heuristic for scores < 1
    expect(hybridScore0).toBeGreaterThan(heuristicScore0)
  })

  it('passes metadata through unchanged', async () => {
    const docsWithMeta: Document[] = [
      { id: 'a', text: 'machine learning', metadata: { source: 'arxiv', page: 1 } },
      { id: 'b', text: 'cooking pasta', metadata: { source: 'blog' } },
    ]
    const results = await rerank('machine learning', docsWithMeta)
    const resultA = results.find(r => r.document.id === 'a')!
    expect(resultA.document.metadata).toEqual({ source: 'arxiv', page: 1 })
  })

  it('respects custom scoring weights', async () => {
    // With position weight = 1.0 and others 0, ranking should favor original top positions
    const results = await rerank('anything', docs, {
      weights: { bm25: 0, tfidf: 0, position: 1 },
    })
    // Original rank 0 should be first (position score = 1 - 0/4 = 1.0)
    expect(results[0].document.id).toBe('doc-0')
    expect(results[0].originalRank).toBe(0)
  })
})

describe('createReranker()', () => {
  it('returns a Reranker with config', () => {
    const r = createReranker({ topK: 3, mode: 'heuristic' })
    expect(r.config).toEqual({ topK: 3, mode: 'heuristic' })
  })

  it('rerank() method works like standalone rerank()', async () => {
    const reranker = createReranker({ topK: 2 })
    const results = await reranker.rerank('machine learning', docs)
    expect(results.length).toBe(2)
  })

  it('options passed to rerank() override config', async () => {
    const reranker = createReranker({ topK: 1 })
    const results = await reranker.rerank('machine learning', docs, { topK: 3 })
    expect(results.length).toBe(3)
  })

  it('empty config defaults work', () => {
    const r = createReranker()
    expect(r.config).toEqual({})
  })
})
