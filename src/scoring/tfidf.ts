function tokenize(text: string): string[] {
  return text.toLowerCase().split(/\W+/).filter(t => t.length > 0)
}

function termFrequency(tokens: string[]): Map<string, number> {
  const freq = new Map<string, number>()
  for (const t of tokens) {
    freq.set(t, (freq.get(t) ?? 0) + 1)
  }
  return freq
}

function computeIdf(term: string, corpus: string[][]): number {
  const N = corpus.length
  const df = corpus.filter(tokens => tokens.includes(term)).length
  return Math.log((N + 1) / (df + 1)) + 1
}

function buildTfidfVector(
  tokens: string[],
  vocabulary: string[],
  corpus: string[][]
): number[] {
  const freq = termFrequency(tokens)
  const totalTerms = tokens.length || 1
  return vocabulary.map(term => {
    const tf = (freq.get(term) ?? 0) / totalTerms
    const idf = computeIdf(term, corpus)
    return tf * idf
  })
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB)
  if (denom === 0) return 0
  return dot / denom
}

export function tfidfScore(query: string, document: string, corpus: string[]): number {
  if (!query.trim() || !document.trim()) return 0

  const queryTokens = tokenize(query)
  const docTokens = tokenize(document)
  const corpusTokens = corpus.map(tokenize)

  // Build vocabulary from query + document + corpus for correct IDF weights
  const allTokens = new Set([...queryTokens, ...docTokens])
  for (const tokens of corpusTokens) {
    for (const t of tokens) allTokens.add(t)
  }
  const vocabulary = Array.from(allTokens)

  if (vocabulary.length === 0) return 0

  const queryVec = buildTfidfVector(queryTokens, vocabulary, corpusTokens)
  const docVec = buildTfidfVector(docTokens, vocabulary, corpusTokens)

  return cosineSimilarity(queryVec, docVec)
}
