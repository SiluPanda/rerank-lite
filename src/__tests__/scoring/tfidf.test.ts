import { describe, it, expect } from 'vitest'
import { tfidfScore } from '../../scoring/tfidf.js'

describe('tfidfScore', () => {
  const corpus = [
    'the quick brown fox jumps over the lazy dog',
    'quick brown fox',
    'the lazy dog sat on the mat',
    'completely unrelated topic about cooking pasta',
  ]

  it('identical texts score close to 1.0', () => {
    const text = 'the quick brown fox jumps over the lazy dog'
    const score = tfidfScore(text, text, corpus)
    expect(score).toBeCloseTo(1.0, 5)
  })

  it('disjoint texts score 0', () => {
    const score = tfidfScore('quick brown fox', 'cooking pasta recipes', corpus)
    expect(score).toBe(0)
  })

  it('returns value between 0 and 1', () => {
    const score = tfidfScore('quick fox', 'the quick brown fox', corpus)
    expect(score).toBeGreaterThanOrEqual(0)
    expect(score).toBeLessThanOrEqual(1)
  })

  it('partial overlap scores between 0 and 1', () => {
    const score = tfidfScore('quick brown fox', 'quick brown cat', corpus)
    expect(score).toBeGreaterThan(0)
    expect(score).toBeLessThan(1)
  })

  it('empty query returns 0', () => {
    const score = tfidfScore('', 'the quick brown fox', corpus)
    expect(score).toBe(0)
  })

  it('empty document returns 0', () => {
    const score = tfidfScore('quick fox', '', corpus)
    expect(score).toBe(0)
  })

  it('uses corpus-wide vocabulary for IDF calculation', () => {
    // "cherry" only appears in corpus, not in query or document
    // With full corpus vocab, IDF of "cherry" affects the vector dimensions
    const corpus3 = ['apple banana', 'apple cherry', 'apple apple apple']
    const score1 = tfidfScore('apple', 'apple banana', corpus3)
    const score2 = tfidfScore('apple', 'apple cherry', corpus3)
    // Both should produce valid scores (not NaN or 0)
    expect(score1).toBeGreaterThan(0)
    expect(score2).toBeGreaterThan(0)
    // "banana" and "cherry" each appear in 1 doc so should have equal IDF
    // Both docs have same query overlap, so scores should be similar
    expect(Math.abs(score1 - score2)).toBeLessThan(0.1)
  })

  it('higher overlap documents score higher', () => {
    const query = 'machine learning neural network'
    const corpus2 = [
      'machine learning neural network deep learning',
      'machine learning basics',
      'cooking and baking recipes',
    ]
    const highScore = tfidfScore(query, 'machine learning neural network deep learning', corpus2)
    const midScore = tfidfScore(query, 'machine learning basics', corpus2)
    const lowScore = tfidfScore(query, 'cooking and baking recipes', corpus2)
    expect(highScore).toBeGreaterThan(midScore)
    expect(midScore).toBeGreaterThan(lowScore)
  })
})
