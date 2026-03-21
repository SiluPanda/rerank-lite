export interface Document {
  id: string
  text: string
  score?: number
  metadata?: Record<string, unknown>
}

export interface RerankResult {
  document: Document
  score: number
  originalRank: number
  newRank: number
  explanation?: string
}

export interface ScoringWeights {
  bm25?: number
  tfidf?: number
  position?: number
}

export type RerankMode = 'heuristic' | 'llm' | 'hybrid'

export type JudgeFn = (query: string, document: string) => Promise<number>

export interface RerankOptions {
  topK?: number
  minScore?: number
  mode?: RerankMode
  judgeFn?: JudgeFn
  weights?: ScoringWeights
}

export interface RerankerConfig {
  mode?: RerankMode
  topK?: number
  minScore?: number
  judgeFn?: JudgeFn
  weights?: ScoringWeights
}

export interface Reranker {
  rerank(query: string, documents: Document[], options?: RerankOptions): Promise<RerankResult[]>
  readonly config: RerankerConfig
}
