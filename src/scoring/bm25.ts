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

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

export function bm25Score(
  query: string,
  document: string,
  corpus: string[],
  k1 = 1.5,
  b = 0.75
): number {
  const queryTerms = tokenize(query)
  const docTokens = tokenize(document)
  const N = corpus.length

  // Compute average document length
  const allCorpusTokens = corpus.map(d => tokenize(d))
  const avgdl = allCorpusTokens.reduce((sum, tokens) => sum + tokens.length, 0) / (N || 1)

  const dl = docTokens.length
  const docFreq = termFrequency(docTokens)

  // Document frequency per term across corpus
  const dfMap = new Map<string, number>()
  for (const tokens of allCorpusTokens) {
    const unique = new Set(tokens)
    for (const t of unique) {
      dfMap.set(t, (dfMap.get(t) ?? 0) + 1)
    }
  }

  let score = 0
  for (const term of new Set(queryTerms)) {
    const tf = docFreq.get(term) ?? 0
    if (tf === 0) continue
    const df = dfMap.get(term) ?? 0
    const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1)
    const tfNorm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / (avgdl || 1)))
    score += idf * tfNorm
  }

  // Normalize to 0-1 using sigmoid(score / 5)
  return sigmoid(score / 5)
}
