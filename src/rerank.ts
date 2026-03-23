import { Document, RerankOptions, RerankResult, ScoringWeights } from './types.js'
import { bm25Score } from './scoring/bm25.js'
import { tfidfScore } from './scoring/tfidf.js'
import { positionScore } from './scoring/position.js'

const DEFAULT_WEIGHTS: Required<ScoringWeights> = {
  bm25: 0.5,
  tfidf: 0.3,
  position: 0.2,
}

export async function rerank(
  query: string,
  documents: Document[],
  options: RerankOptions = {}
): Promise<RerankResult[]> {
  if (!documents.length) return []

  const mode = options.mode ?? 'heuristic'
  const weights: Required<ScoringWeights> = {
    bm25: options.weights?.bm25 ?? DEFAULT_WEIGHTS.bm25,
    tfidf: options.weights?.tfidf ?? DEFAULT_WEIGHTS.tfidf,
    position: options.weights?.position ?? DEFAULT_WEIGHTS.position,
  }
  const allTexts = documents.map(d => d.text)

  // Compute heuristic composite score for each document
  const heuristicScores = documents.map((doc, idx) => {
    const bm25 = bm25Score(query, doc.text, allTexts)
    const tfidf = tfidfScore(query, doc.text, allTexts)
    const position = positionScore(idx, documents.length)
    return weights.bm25 * bm25 + weights.tfidf * tfidf + weights.position * position
  })

  // For llm and hybrid modes, call judgeFn if provided
  let finalScores = [...heuristicScores]

  if ((mode === 'llm' || mode === 'hybrid') && options.judgeFn) {
    const llmScores = await Promise.all(
      documents.map(doc => (options.judgeFn as NonNullable<typeof options.judgeFn>)(query, doc.text))
    )

    if (mode === 'llm') {
      finalScores = llmScores
    } else {
      // hybrid: blend 50/50 heuristic and llm
      finalScores = heuristicScores.map((h, i) => 0.5 * h + 0.5 * llmScores[i])
    }
  }

  // Build results with original ranks
  const results: RerankResult[] = documents.map((doc, idx) => ({
    document: doc,
    score: finalScores[idx],
    originalRank: idx,
    newRank: 0,
  }))

  // Sort descending by score
  results.sort((a, b) => b.score - a.score)

  // Assign new ranks (0-based)
  results.forEach((r, idx) => {
    r.newRank = idx
  })

  // Apply minScore filter
  let filtered = results
  if (options.minScore !== undefined) {
    filtered = results.filter(r => r.score >= (options.minScore as number))
  }

  // Apply topK
  if (options.topK !== undefined && options.topK > 0) {
    filtered = filtered.slice(0, options.topK)
  }

  // Recalculate newRank after filtering so values are contiguous 0-based
  filtered.forEach((r, idx) => {
    r.newRank = idx
  })

  return filtered
}
