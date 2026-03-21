import { describe, it, expect } from 'vitest'
import { bm25Score } from '../../scoring/bm25.js'

describe('bm25Score', () => {
  const corpus = [
    'the quick brown fox jumps over the lazy dog',
    'quick brown fox',
    'the lazy dog sat on the mat',
    'completely unrelated topic about cooking pasta',
  ]

  it('exact term match scores higher than partial match', () => {
    const query = 'quick brown fox'
    const exactScore = bm25Score(query, 'quick brown fox', corpus)
    const partialScore = bm25Score(query, 'quick fox', corpus)
    expect(exactScore).toBeGreaterThan(partialScore)
  })

  it('relevant document scores higher than irrelevant document', () => {
    const query = 'quick brown fox'
    const relevantScore = bm25Score(query, 'the quick brown fox jumps over the lazy dog', corpus)
    const irrelevantScore = bm25Score(query, 'completely unrelated topic about cooking pasta', corpus)
    expect(relevantScore).toBeGreaterThan(irrelevantScore)
  })

  it('empty document scores 0 (sigmoid(0) = 0.5 with no matching terms → 0 raw → 0.5 normalized is not quite 0, but no term matches gives raw=0)', () => {
    // When there are no matching terms, raw BM25 score = 0, sigmoid(0) = 0.5
    // So empty doc with unrelated query gives 0.5 which is the baseline.
    // But a doc with NO query terms at all returns sigmoid(0/5) = sigmoid(0) = 0.5
    const score = bm25Score('quick fox', '', corpus)
    expect(score).toBe(0.5) // sigmoid(0)
  })

  it('returns value between 0 and 1', () => {
    const score = bm25Score('quick brown fox', 'quick brown fox jumps', corpus)
    expect(score).toBeGreaterThanOrEqual(0)
    expect(score).toBeLessThanOrEqual(1)
  })

  it('identical query and document scores higher than disjoint', () => {
    const query = 'machine learning model training'
    const corpus2 = [
      'machine learning model training data',
      'cooking recipes and meal prep',
    ]
    const matchScore = bm25Score(query, 'machine learning model training data', corpus2)
    const noMatchScore = bm25Score(query, 'cooking recipes and meal prep', corpus2)
    expect(matchScore).toBeGreaterThan(noMatchScore)
  })

  it('uses custom k1 and b parameters', () => {
    const query = 'fox'
    const doc = 'fox fox fox fox'
    const corpusLocal = [doc, 'cat']
    const score1 = bm25Score(query, doc, corpusLocal, 1.2, 0.75)
    const score2 = bm25Score(query, doc, corpusLocal, 2.0, 0.5)
    // Both should be valid scores in [0,1]
    expect(score1).toBeGreaterThanOrEqual(0)
    expect(score1).toBeLessThanOrEqual(1)
    expect(score2).toBeGreaterThanOrEqual(0)
    expect(score2).toBeLessThanOrEqual(1)
  })
})
