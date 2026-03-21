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
