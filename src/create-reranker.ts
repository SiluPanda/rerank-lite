import { Reranker, RerankerConfig, Document, RerankOptions, RerankResult } from './types.js'
import { rerank } from './rerank.js'

export function createReranker(config: RerankerConfig = {}): Reranker {
  return {
    async rerank(query: string, documents: Document[], options?: RerankOptions): Promise<RerankResult[]> {
      return rerank(query, documents, { ...config, ...options })
    },
    get config(): RerankerConfig {
      return config
    },
  }
}
