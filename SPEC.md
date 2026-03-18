# rerank-lite -- Specification

## 1. Overview

`rerank-lite` is a lightweight, multi-mode retrieval reranker for JavaScript. It takes a query and a list of candidate documents -- the output of a first-stage retriever (vector search, BM25, or fusion) -- and reorders them by relevance using one of three scoring modes: cross-encoder inference via local ONNX models, LLM-as-judge prompting via a pluggable LLM function, or heuristic keyword-based scoring with zero dependencies. It provides both a TypeScript/JavaScript API for programmatic use in retrieval pipelines and a CLI for reranking documents from JSON input.

The gap this package fills is specific and well-defined. Reranking is the standard second stage of a two-stage retrieval pipeline. Stage 1 (retrieval) uses a fast, approximate method -- BM25 keyword matching, dense vector search, or hybrid fusion via `fusion-rank` -- to pull back the top 50-100 candidate documents from a large corpus. Stage 2 (reranking) scores each candidate more carefully against the query and reorders the list for precision. Retrieval is fast but rough: a bi-encoder encodes the query and documents independently, comparing them via a single dot product, which cannot capture fine-grained token-level interactions. A reranker sees the query and document together, enabling cross-attention between every query token and every document token, producing significantly more accurate relevance judgments. In production RAG pipelines, adding a reranker after retrieval consistently improves precision@10 by 5-15 percentage points on standard benchmarks (MS MARCO, BEIR, MTEB).

In Python, reranking is well-served. Cohere provides a cloud Rerank API (`cohere.rerank(query, documents, model)`) that is simple to use but requires an API key, internet connectivity, and per-call costs. The `sentence-transformers` library provides `CrossEncoder` for local cross-encoder inference with models like `ms-marco-MiniLM-L-6-v2` and `bge-reranker-base`. Hugging Face's `transformers` library can load any cross-encoder model for sequence classification. LangChain provides `CohereRerank`, `CrossEncoderReranker`, and `LLMChainFilter` for reranking within LangChain pipelines. ColBERT and ColBERTv2 provide late-interaction reranking with pre-computed token embeddings. FlashRank provides a lightweight Python reranker. All of these are Python-only.

In JavaScript, the situation is bleak. Searching npm for "rerank", "reranker", "cross-encoder", or "relevance scoring" returns no standalone reranking packages. The `@xenova/transformers` library (now `@huggingface/transformers`) can technically load cross-encoder ONNX models and run inference, but it requires the developer to understand ONNX model loading, tokenization, input tensor construction, and output interpretation -- none of which are documented for the reranking use case. There is no high-level `rerank(query, documents)` API. JavaScript developers building RAG pipelines on Pinecone, Qdrant, or Weaviate are left to either call the Cohere Rerank API (adding a cloud dependency and per-call cost to what could be a local operation) or skip reranking entirely, accepting lower retrieval precision. Forum posts on the Pinecone community, Qdrant GitHub discussions, and LangChain.js issues repeatedly ask for a JavaScript reranking solution.

`rerank-lite` fills this gap with three reranking modes to cover different cost, accuracy, and dependency tradeoffs. The cross-encoder mode runs a local ONNX model for highest accuracy with no API calls. The LLM-as-judge mode uses a pluggable LLM function for flexible, model-agnostic reranking without local model files. The heuristic mode provides zero-dependency, zero-cost keyword-based reranking as a baseline or fallback. All three modes share the same `rerank(query, documents, options?)` API and return the same `RerankResult[]` output, making them interchangeable. The output is a reordered list of documents with relevance scores normalized to [0, 1], ready to pass to `context-packer` for budget-aware chunk selection or to `fusion-rank` as a third retrieval signal.

---

## 2. Goals and Non-Goals

### Goals

- Provide a `rerank(query, documents, options?)` function that takes a query string and an array of documents and returns a `RerankResult[]` sorted by relevance score descending.
- Provide a `createReranker(config)` factory that returns a configured `Reranker` instance with preset mode, model, and options, for repeated use across many queries.
- Implement three reranking modes: `cross-encoder` (local ONNX model inference), `llm-judge` (LLM-as-judge prompting), and `heuristic` (keyword-based scoring).
- For cross-encoder mode: load ONNX cross-encoder models (ms-marco-MiniLM-L-6-v2, bge-reranker-base, bge-reranker-v2-m3), tokenize (query, document) pairs, run inference, and return relevance scores.
- For LLM-as-judge mode: accept a pluggable async LLM function, construct relevance-scoring prompts, parse scores from LLM responses, and return relevance scores.
- For heuristic mode: compute relevance scores using query term coverage, BM25-style TF-IDF scoring, keyword density, and position bonuses. Zero external dependencies.
- Normalize all output scores to [0, 1] regardless of mode.
- Support batched inference for cross-encoder mode to improve throughput.
- Provide a model management system for cross-encoder mode: download models from Hugging Face Hub, cache on the local filesystem, lazy-load on first use.
- Provide a CLI (`rerank-lite`) for reranking documents from JSON input.
- Support integration with `fusion-rank` (rerank after fusion or provide reranked results as a fusion input), `context-packer` (rerank before packing), `sparse-encode` (BM25 retrieval before reranking), and `embed-cache` (dense retrieval before reranking).
- Target Node.js 18 and above.

### Non-Goals

- **Not a retriever.** This package does not perform vector search, BM25 retrieval, or any other form of first-stage document retrieval. It receives pre-retrieved candidate documents and reorders them by relevance. For retrieval, use a vector database SDK, `sparse-encode` for BM25 sparse vectors, or `embed-cache` for dense embeddings.
- **Not a fusion library.** This package reorders a single list of candidates. It does not merge multiple ranked lists from different retrievers. For combining results from multiple retrievers, use `fusion-rank`. The output of `rerank-lite` can be passed to `fusion-rank` as one of several input lists.
- **Not a model training framework.** This package runs inference on pre-trained cross-encoder models. It does not fine-tune or train models. For fine-tuning cross-encoders, use Python-based tools (sentence-transformers, Hugging Face Trainer) and export to ONNX.
- **Not a general-purpose inference engine.** This package loads and runs ONNX models specifically for the cross-encoder reranking task (sequence pair classification). It does not support arbitrary ONNX models, embedding generation, or other inference tasks.
- **Not a vector database integration layer.** This package operates on plain arrays of `{ id, text }` objects. It does not know about Pinecone, Qdrant, or any other database. Callers extract documents from their database SDK and pass them to `rerank()`.
- **Not an LLM provider.** This package does not include or depend on any LLM SDK. The LLM-as-judge mode accepts a caller-supplied async function. The caller is responsible for authentication, rate limiting, and cost management.
- **Not a tokenizer library.** The package bundles tokenization logic for the supported cross-encoder models. It does not expose a general-purpose tokenizer API.
- **Not a browser-side reranker.** While the heuristic and LLM-as-judge modes work in any JavaScript runtime, the cross-encoder mode requires `onnxruntime-node` which is Node.js-only. Browser support for cross-encoder mode is out of scope for v1.

---

## 3. Target Users and Use Cases

### RAG Pipeline Builders

Developers building JavaScript-native retrieval-augmented generation pipelines who retrieve top-50 to top-100 candidate chunks per query and need to rerank them for precision before injecting into an LLM context window. Today they either skip reranking (accepting lower quality) or call the Cohere Rerank API (adding a cloud dependency). With `rerank-lite`, they call `rerank(query, candidates, { mode: 'cross-encoder' })` and get locally computed, high-accuracy relevance scores without any API calls. A typical integration: retrieve from Pinecone, rerank with `rerank-lite`, pack with `context-packer`.

### Search Quality Engineers

Engineers responsible for optimizing retrieval quality in production search systems. They need to compare reranking strategies (cross-encoder vs. LLM-as-judge vs. heuristic) on their data, measure the impact on precision@K and NDCG, and choose the right tradeoff between accuracy and cost. `rerank-lite`'s three modes share the same API, making A/B comparison a one-line configuration change.

### Cost-Sensitive Teams

Teams that use the Cohere Rerank API today and want to reduce costs by switching to local cross-encoder inference. At $1 per 1,000 search units on Cohere, a high-traffic application reranking 100 documents per query at 10,000 queries per day costs $1,000/day. Local cross-encoder inference on `rerank-lite` costs nothing per query -- only the compute time on the existing server. Teams migrating from Cohere can use `rerank-lite`'s cross-encoder mode as a drop-in replacement with comparable accuracy.

### Hybrid Search Pipeline Engineers

Teams building multi-stage retrieval pipelines that combine BM25 sparse retrieval (via `sparse-encode`), dense vector retrieval (via `embed-cache` and a vector database), and cross-encoder reranking into a single ranked list (via `fusion-rank`). The reranker's output is treated as a third retrieval signal fused alongside dense and sparse results: `rrf([denseResults, sparseResults, rerankedResults])`. This three-signal fusion pattern consistently outperforms two-signal fusion in academic evaluations.

### Offline Evaluation Teams

Teams running offline retrieval evaluation who need to measure the quality impact of reranking. They run the retriever, save candidates, then apply different reranking modes (cross-encoder with various models, LLM-as-judge with different prompts, heuristic as a baseline) and compare precision@K, recall@K, NDCG, and MRR for each configuration. The heuristic mode provides a no-cost, no-latency baseline to quantify the value added by more expensive reranking modes.

### Prototyping and Development

Developers prototyping RAG systems who need a reranking step but do not want to set up Python environments or cloud API keys. The heuristic mode works immediately with `npm install rerank-lite` -- no model downloads, no API keys, no setup. As the prototype matures, they upgrade to cross-encoder mode for accuracy or LLM-as-judge for flexibility without changing the calling code.

---

## 4. Core Concepts

### Two-Stage Retrieval

Production information retrieval follows a two-stage pattern. Stage 1, the retriever, uses a fast, approximate method to narrow millions of documents to tens or hundreds of candidates. Bi-encoder models encode the query and each document independently into vector representations, comparing them via a single dot product or cosine similarity. This independent encoding is what makes retrieval fast -- documents can be pre-encoded and indexed -- but it limits accuracy because the query and document never "see" each other. The retriever optimizes for recall: returning all potentially relevant documents, even if some irrelevant ones slip through.

Stage 2, the reranker, scores each candidate more carefully against the query and reorders the list. Because the candidate set is small (tens, not millions), the reranker can afford to be slow and accurate. Cross-encoder rerankers process the query and document as a single concatenated input, enabling cross-attention between every query token and every document token. This joint encoding captures token-level semantic interactions that bi-encoders miss entirely -- negation, coreference, fine-grained relevance, and context-dependent meaning. The reranker optimizes for precision: placing the most relevant documents at the top of the list.

The combined pipeline delivers both high recall (from the retriever) and high precision (from the reranker), which neither stage alone can achieve.

### Cross-Encoder

A cross-encoder is a transformer model trained for sequence pair classification. It takes two text inputs -- a query and a document -- concatenates them with a separator token (`[SEP]`), and produces a single scalar relevance score. Unlike a bi-encoder, which encodes each input independently, the cross-encoder sees both inputs simultaneously, enabling every token in the query to attend to every token in the document.

The cross-encoder architecture:

```
Input:   [CLS] query tokens [SEP] document tokens [SEP]
         └──────── Full cross-attention ────────┘
Output:  Single scalar → relevance score
```

Cross-encoders achieve significantly higher accuracy than bi-encoders for relevance scoring, but they cannot be used for first-stage retrieval because they require the query and document to be processed together -- there is no pre-computed document embedding to index. They are designed specifically for the reranking stage, where the candidate set is small enough to score each (query, document) pair individually.

### Relevance Score

The relevance score is a number indicating how relevant a document is to the query. Different reranking modes produce scores on different raw scales:

- **Cross-encoder**: raw logit scores, typically in [-10, 10]. Positive scores indicate relevance; larger magnitudes indicate stronger signals.
- **LLM-as-judge**: scores parsed from LLM output, typically on a 0-10 or 0-100 scale as instructed by the prompt.
- **Heuristic**: composite keyword-based scores, typically in [0, 1] by construction.

`rerank-lite` normalizes all output scores to [0, 1] using min-max normalization across the candidate set, making scores comparable across modes and consumable by downstream tools like `context-packer` and `fusion-rank`.

### Document

A document is a unit of text to be scored for relevance to a query. In a RAG pipeline, documents are typically chunks of larger documents -- paragraphs, sections, or fixed-size text blocks produced by a chunking library. In `rerank-lite`, a document is represented as a `Document` object with a required `text` field, an optional `id` for tracking, and an optional `metadata` record for pass-through information.

### Rerank Result

A rerank result is a scored and ranked document. It contains the original document's `id`, `text`, and `metadata`, plus a `score` in [0, 1] and a `rank` (1-based position in the reranked list). The rerank result array is sorted by score descending: rank 1 is the most relevant document.

---

## 5. Reranking Modes

`rerank-lite` provides three reranking modes, each with different accuracy, cost, latency, and dependency characteristics. All three modes are accessed through the same `rerank()` API and return the same `RerankResult[]` type.

### 5.1 Cross-Encoder Mode

**How it works**: Loads a pre-trained cross-encoder model in ONNX format and a corresponding tokenizer. For each (query, document) pair, tokenizes the concatenated input, runs inference through the ONNX model, and extracts the relevance score from the model's output logits.

**Accuracy**: Highest of the three modes. Cross-encoders are specifically trained on large-scale relevance judgment datasets (MS MARCO, Natural Questions) to distinguish relevant from irrelevant documents. They capture token-level interactions between query and document that other methods miss.

**Latency**: Moderate. Depends on model size and document count. With `ms-marco-MiniLM-L-6-v2` (22M parameters), expect ~5-15ms per document on a modern CPU. A batch of 20 documents takes ~100-300ms. Batched inference amortizes model loading and reduces per-document overhead.

**Cost**: Zero per-query cost. Requires a one-time model download (50-500MB depending on model). CPU compute is the only resource consumed.

**Dependencies**: Requires `onnxruntime-node` as an optional peer dependency. Node.js only.

**When to use**: Production RAG pipelines where accuracy is critical and the server has sufficient CPU resources. The default choice for any serious retrieval system. Preferred over LLM-as-judge when cost must be minimized and over heuristic when accuracy matters.

### 5.2 LLM-as-Judge Mode

**How it works**: Constructs a structured prompt asking an LLM to score the relevance of each document to the query. Sends the prompt to a caller-supplied LLM function. Parses the numeric score from the LLM's response. Supports two prompting strategies: pointwise (score each document independently) and listwise (rank all documents in a single prompt).

**Accuracy**: High, but variable. Depends on the LLM's capability and the prompt design. GPT-4-class models produce relevance judgments comparable to or better than small cross-encoders. Smaller models (GPT-3.5, Llama-7B) produce less reliable scores. The prompting strategy affects accuracy: listwise ranking with direct comparison tends to be more consistent than pointwise scoring.

**Latency**: High. Each LLM call adds 500-2000ms depending on the model and provider. Pointwise scoring requires one LLM call per document (or batched into groups). Listwise scoring requires fewer calls but longer prompts. Total latency for 20 documents: 2-10 seconds depending on strategy and parallelism.

**Cost**: Significant. LLM API calls have per-token costs. Reranking 20 documents with GPT-4o costs approximately $0.002-0.01 per query depending on document length. At 10,000 queries/day, this adds $20-100/day. Use this mode when cross-encoder models are not available or when the LLM provides domain-specific judgment that a general cross-encoder lacks.

**Dependencies**: None from `rerank-lite` itself. The caller provides the LLM function. Any LLM provider (OpenAI, Anthropic, local Ollama, etc.) works.

**When to use**: When cross-encoder mode is not feasible (no ONNX runtime, browser environment, edge deployment) or when the task requires domain-specific reasoning that general cross-encoders lack (e.g., legal relevance, medical relevance). Also useful for evaluation: comparing LLM judgments against cross-encoder scores to validate the cross-encoder's accuracy on a specific domain.

### 5.3 Heuristic Mode

**How it works**: Computes a composite relevance score from four keyword-based signals: query term coverage (fraction of query terms found in the document), BM25-style TF-IDF scoring (term frequency weighted by inverse document frequency), keyword density (frequency of query terms relative to document length), and position bonus (query terms appearing earlier in the document score higher). The four signals are combined with configurable weights into a final score.

**Accuracy**: Lowest of the three modes. Heuristic scoring captures lexical overlap but misses semantic relevance entirely. "Machine learning algorithms" and "ML models" have zero heuristic overlap despite being semantically equivalent. However, for queries with distinctive keywords, heuristic reranking provides meaningful signal and is a substantial improvement over random ordering.

**Latency**: Negligible. Scoring 100 documents takes < 1ms. The computation is pure string processing with no model loading, no network calls, and no heavy computation.

**Cost**: Zero. No model files, no API calls, no external dependencies.

**Dependencies**: None. Pure TypeScript implementation.

**When to use**: As a baseline for evaluation (measuring how much value cross-encoder or LLM-as-judge adds over keyword matching). As a fallback when the cross-encoder model fails to load or the LLM API is unavailable. For prototyping and development when setup speed matters more than accuracy. For applications where the query vocabulary closely matches the document vocabulary (e.g., technical documentation search, FAQ matching).

### Mode Comparison

| Property | Cross-Encoder | LLM-as-Judge | Heuristic |
|----------|--------------|-------------|-----------|
| Accuracy | Highest | High (model-dependent) | Lowest |
| Latency per document | 5-15ms | 100-500ms | < 0.01ms |
| Latency for 20 docs | 100-300ms | 2-10s | < 1ms |
| Per-query cost | $0 | $0.002-0.01 | $0 |
| Setup cost | Model download (50-500MB) | LLM API key | None |
| Dependencies | `onnxruntime-node` | Caller-supplied LLM fn | None |
| Captures semantics | Yes (cross-attention) | Yes (LLM reasoning) | No (lexical only) |
| Offline capable | Yes | No (needs API) | Yes |
| Node.js only | Yes | No (any runtime) | No (any runtime) |

---

## 6. Cross-Encoder Scoring

### How Local Model Reranking Works

Cross-encoder reranking with a local ONNX model proceeds in five steps:

1. **Model loading**: Load the ONNX model file and its tokenizer vocabulary from the local filesystem cache. If the model is not cached, download it from Hugging Face Hub first. Model loading is lazy -- it happens on the first `rerank()` call, not at import time.

2. **Tokenization**: For each (query, document) pair, tokenize the concatenated input using the model's tokenizer:
   ```
   [CLS] query_token_1 query_token_2 ... [SEP] doc_token_1 doc_token_2 ... [SEP]
   ```
   The tokenizer produces three tensors: `input_ids` (token indices), `attention_mask` (1 for real tokens, 0 for padding), and `token_type_ids` (0 for query tokens, 1 for document tokens). Long documents are truncated to the model's maximum sequence length (typically 512 tokens), with the query preserved in full and the document truncated from the end.

3. **Batching**: Group multiple (query, document) pairs into a single inference batch. Pad all sequences in the batch to the length of the longest sequence. Batching amortizes the fixed cost of ONNX session setup and improves CPU utilization through vectorized operations.

4. **Inference**: Feed the batched tensors to the ONNX Runtime session. The model outputs a logits tensor with shape `[batchSize, numLabels]`. For binary relevance models (most cross-encoders), `numLabels` is 1 and the single logit is the relevance score. For models with 2 output labels (relevant/irrelevant), the score is derived from a softmax over the logits, taking the probability of the "relevant" class.

5. **Score extraction and normalization**: Extract the raw logit score for each (query, document) pair. Apply min-max normalization across all scores in the batch to produce scores in [0, 1].

### Supported Models

`rerank-lite` includes a model registry with metadata for three pre-tested cross-encoder models:

| Model | HuggingFace ID | Parameters | Max Seq Len | ONNX Size | Language | Accuracy (MS MARCO MRR@10) |
|-------|---------------|-----------|-------------|-----------|----------|---------------------------|
| ms-marco-MiniLM-L-6-v2 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | 512 | ~85MB | English | 0.390 |
| bge-reranker-base | `BAAI/bge-reranker-base` | 110M | 512 | ~440MB | English | 0.414 |
| bge-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | 568M | 8192 | ~1.1GB | Multilingual | 0.425 |

**ms-marco-MiniLM-L-6-v2** is the default. It is the fastest model, the smallest download, and provides strong baseline accuracy. It is a 6-layer MiniLM model fine-tuned on the MS MARCO passage ranking dataset. Recommended for most English-language use cases where latency is a priority.

**bge-reranker-base** provides higher accuracy than MiniLM at the cost of ~5x more parameters and proportionally higher latency. It is a BERT-base model fine-tuned by BAAI (Beijing Academy of Artificial Intelligence) on a diverse set of retrieval tasks. Recommended for applications where accuracy is worth the additional latency.

**bge-reranker-v2-m3** is a multilingual reranker supporting over 100 languages. It has a larger maximum sequence length (8192 tokens) for long documents. It is the most accurate but also the largest and slowest. Recommended for multilingual applications or when long document support is required.

### Custom Models

Callers can use any ONNX-exported cross-encoder model by providing the path to the model file and tokenizer:

```typescript
const result = await rerank(query, documents, {
  mode: 'cross-encoder',
  modelPath: '/path/to/custom-model.onnx',
  tokenizerPath: '/path/to/tokenizer.json',
});
```

The custom model must accept the standard cross-encoder input format (`input_ids`, `attention_mask`, `token_type_ids`) and produce a logits output tensor. The package does not validate model architecture beyond checking tensor shapes.

### Tokenization

The cross-encoder mode includes a fast WordPiece tokenizer implementation (the tokenization algorithm used by BERT-family models). The tokenizer reads a `tokenizer.json` file (Hugging Face Tokenizers format) and produces `input_ids`, `attention_mask`, and `token_type_ids` tensors. Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) are handled automatically.

For models with non-WordPiece tokenizers (SentencePiece, BPE), the package falls back to reading the `tokenizer.json` configuration and dispatching to the appropriate algorithm. The built-in tokenizer supports WordPiece (BERT, MiniLM, BGE) and basic BPE (RoBERTa, XLM-R) -- covering all models in the registry.

### Truncation

When the combined length of the query and document exceeds the model's maximum sequence length, the document is truncated from the end to fit. The query is never truncated because losing query terms changes the reranking task. The truncation budget is:

```
maxDocTokens = maxSeqLen - queryTokens - 3  // 3 = [CLS] + 2x [SEP]
```

If the query alone exceeds `maxSeqLen - 3`, the query is truncated from the end as a last resort. This should rarely happen in practice (queries are typically short).

### Batched Inference

Scoring 20 documents one-at-a-time incurs 20 separate ONNX session runs, each with its own overhead. Batched inference groups multiple (query, document) pairs into a single session run, padding all sequences in the batch to a uniform length.

```
batchSize = min(documents.length, maxBatchSize)
```

The `maxBatchSize` option controls how many pairs are scored in a single batch. Default: 32. Larger batches improve throughput but consume more memory. For a typical reranking task (20 documents, ~200 tokens each), a single batch of 32 is both feasible and optimal.

When the document count exceeds `maxBatchSize`, the documents are processed in multiple batches. The scores from all batches are concatenated before normalization.

---

## 7. LLM-as-Judge Scoring

### How LLM Reranking Works

LLM-as-judge reranking uses a large language model to assess the relevance of each document to the query. The caller provides an async LLM function with a simple interface: accept a prompt string, return a response string. `rerank-lite` constructs relevance-scoring prompts, calls the LLM function, parses numeric scores from the responses, and returns relevance scores.

### Pluggable LLM Function

The LLM function is provided by the caller and has this signature:

```typescript
type LLMFunction = (prompt: string) => Promise<string>;
```

This design makes `rerank-lite` agnostic to the LLM provider. Any LLM accessible via an async function works:

```typescript
// OpenAI
const llm = async (prompt: string) => {
  const res = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: prompt }],
    temperature: 0,
  });
  return res.choices[0].message.content ?? '';
};

// Anthropic
const llm = async (prompt: string) => {
  const res = await anthropic.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 256,
    messages: [{ role: 'user', content: prompt }],
  });
  return res.content[0].type === 'text' ? res.content[0].text : '';
};

// Local Ollama
const llm = async (prompt: string) => {
  const res = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    body: JSON.stringify({ model: 'llama3', prompt }),
  });
  const data = await res.json();
  return data.response;
};
```

### Pointwise Scoring

Pointwise scoring evaluates each document independently. For each document, a prompt asks the LLM to rate its relevance to the query on a numeric scale.

**Prompt template**:

```
You are a relevance judge. Given a query and a document, rate how relevant the document is to the query on a scale of 0 to 10.

- 0: Completely irrelevant. The document has nothing to do with the query.
- 1-3: Marginally relevant. The document touches on the topic but does not answer the query.
- 4-6: Moderately relevant. The document is related and contains some useful information.
- 7-9: Highly relevant. The document directly addresses the query with substantial information.
- 10: Perfectly relevant. The document is an ideal answer to the query.

Respond with ONLY a single integer from 0 to 10. No explanation.

Query: {query}

Document: {document}

Relevance score:
```

The LLM responds with a single number (e.g., "8"). The parser extracts the first integer from the response. If parsing fails (non-numeric response), the document receives a score of 0 and a warning is emitted.

**Batching optimization**: To reduce LLM call count, multiple documents can be scored in a single prompt. The batch prompt presents several documents numbered 1 through N and asks for a score for each:

```
You are a relevance judge. Given a query and a list of documents, rate each document's relevance to the query on a scale of 0 to 10.

Respond with ONLY the scores, one per line, in the format: "1: <score>" for each document. No explanation.

Query: {query}

Document 1: {document_1}

Document 2: {document_2}

...

Scores:
```

The `batchSize` option controls how many documents are included per prompt. Default: 10. Larger batches reduce LLM calls but increase prompt length (and token costs) per call. For cost optimization, the batch size should be tuned to the LLM's context window and pricing.

### Listwise Ranking

Listwise ranking asks the LLM to directly order a set of documents by relevance, rather than scoring each independently. This can produce more consistent rankings because the LLM sees all documents simultaneously and can make relative comparisons.

**Prompt template**:

```
You are a relevance judge. Given a query and a list of documents, rank the documents from most relevant to least relevant.

Respond with ONLY the document numbers in order from most relevant to least relevant, separated by commas. No explanation.

Query: {query}

Document 1: {document_1}

Document 2: {document_2}

...

Ranking (most relevant first):
```

The LLM responds with a comma-separated list of document numbers (e.g., "3, 1, 5, 2, 4"). The parser extracts document indices and assigns scores based on rank position: `score = 1 - (rank - 1) / (N - 1)`, producing evenly spaced scores from 1.0 (most relevant) to 0.0 (least relevant).

Listwise ranking is limited by the LLM's context window. For large document sets, the documents are split into groups, each group is ranked independently, and the rankings are merged using a simple interleave strategy: the top-1 from each group is compared in a final ranking call to establish cross-group ordering.

### Error Handling

LLM responses are unpredictable. The parser handles:

- **Non-numeric response**: Falls back to score 0 for that document.
- **Partial batch response**: If the LLM returns scores for only some documents in a batch, missing documents receive score 0.
- **LLM function throws**: The error is caught and rethrown as a `RerankError` with code `LLM_ERROR` and the original error in `cause`.
- **Timeout**: If the LLM function does not resolve within `llmTimeout` (default: 30 seconds), the call is aborted and the affected documents receive score 0.

### Cost Optimization

- **Use pointwise batching**: Scoring 10 documents per LLM call instead of 1 per call reduces the number of calls by 10x.
- **Use listwise for small sets**: For fewer than 15 documents, a single listwise ranking call is cheaper and more consistent than multiple pointwise calls.
- **Use smaller models**: GPT-4o-mini and Claude Haiku are 10-20x cheaper than GPT-4 and Claude Opus for relevance scoring, with modest accuracy degradation.
- **Truncate documents**: Long documents consume more tokens. Truncating to the first 500 tokens captures most relevant information at a fraction of the cost.
- **Cache results**: For repeated queries, cache reranking results keyed on (query, documentIds) to avoid redundant LLM calls.

---

## 8. Heuristic Scoring

### How Keyword-Based Reranking Works

Heuristic scoring computes relevance using four lexical signals that require no machine learning models and no external dependencies. The signals are combined with configurable weights into a composite score.

### Signal 1: Query Term Coverage

The fraction of unique query terms found in the document. Measures how completely the document addresses the query vocabulary.

```
queryTerms = unique lowercase tokens from query, stop words removed
matchedTerms = queryTerms that appear in the document (case-insensitive)
coverage = |matchedTerms| / |queryTerms|
```

A document containing all query terms has coverage 1.0. A document containing none has coverage 0.0.

**Example**:
- Query: "CUDA memory allocation error"
- Query terms (after stop word removal): ["cuda", "memory", "allocation", "error"]
- Document contains: "cuda", "memory", "error" (but not "allocation")
- Coverage: 3/4 = 0.75

### Signal 2: BM25-Style TF-IDF Scoring

A simplified BM25 scoring formula that uses term frequency (TF) and inverse document frequency (IDF) across the candidate document set. This signal weights rare query terms higher than common ones and accounts for term frequency saturation.

```
BM25(q, d) = sum over t in q of: IDF(t) * (TF(t, d) * (k1 + 1)) / (TF(t, d) + k1 * (1 - b + b * |d| / avgDL))

where:
  TF(t, d)  = count of term t in document d
  IDF(t)    = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
  N         = number of documents in the candidate set
  n(t)      = number of documents containing term t
  |d|       = length of document d in tokens
  avgDL     = average document length across the candidate set
  k1        = 1.2 (term frequency saturation parameter)
  b         = 0.75 (document length normalization parameter)
```

The BM25 score is computed using the candidate document set as the corpus. IDF values are computed over the candidate set, not a global corpus. This is appropriate because the candidates are the only documents being compared.

The raw BM25 scores are normalized to [0, 1] via min-max across the candidate set.

### Signal 3: Keyword Density

The frequency of query terms in the document relative to the document's total length. Rewards documents that discuss the query topic extensively, not just mention it once.

```
queryTermOccurrences = total count of all query term appearances in the document
density = queryTermOccurrences / documentTokenCount
```

The raw density is normalized to [0, 1] via min-max across the candidate set.

### Signal 4: Position Bonus

Query terms appearing near the beginning of a document are likely more indicative of the document's main topic than terms buried deep in the text. The position bonus rewards early appearances.

```
For each query term t found in the document:
  firstPosition(t) = character offset of the first occurrence of t in the document
  positionScore(t) = 1 - (firstPosition(t) / documentLength)

positionBonus = average of positionScore(t) over all matched query terms
```

A query term at position 0 (the very start) contributes 1.0. A term at the very end contributes ~0.0. The average across all matched terms produces the position bonus. If no query terms appear in the document, the position bonus is 0.

### Composite Score

The four signals are combined with configurable weights:

```
heuristicScore = w_coverage * coverage
               + w_bm25 * normalizedBM25
               + w_density * normalizedDensity
               + w_position * positionBonus
```

Default weights:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Coverage | 0.35 | Strong indicator: a document mentioning most query terms is likely relevant. |
| BM25 | 0.40 | BM25 is a proven retrieval signal. IDF weighting and length normalization make it more discriminative than raw coverage. |
| Density | 0.10 | Moderate signal: high density indicates topical focus, but dense keyword stuffing can be misleading. |
| Position | 0.15 | Moderate signal: early query term appearance suggests the document's topic matches the query. |

The weights are configured via the `heuristicWeights` option. They are auto-normalized to sum to 1.0.

### Stop Words

The heuristic mode removes common English stop words from the query before computing signals. The built-in stop word list covers ~175 common English words (articles, prepositions, pronouns, common verbs). The `stopWords` option allows the caller to provide a custom stop word set or disable stop word removal (`stopWords: []`).

### Stemming

The heuristic mode does not perform stemming by default. Stemming (reducing "running" to "run", "algorithms" to "algorithm") would improve recall but adds complexity and is language-dependent. The `stemmer` option accepts a caller-supplied stemming function for callers who want this behavior:

```typescript
const result = await rerank(query, documents, {
  mode: 'heuristic',
  stemmer: (word) => porterStemmer(word),
});
```

---

## 9. API Surface

### Installation

```bash
npm install rerank-lite

# For cross-encoder mode, also install the ONNX runtime:
npm install onnxruntime-node
```

### Primary Export: `rerank`

```typescript
import { rerank } from 'rerank-lite';

// Cross-encoder mode (default)
const results = await rerank(query, documents, {
  mode: 'cross-encoder',
  model: 'ms-marco-MiniLM-L-6-v2',
});

// LLM-as-judge mode
const results = await rerank(query, documents, {
  mode: 'llm-judge',
  llm: myLLMFunction,
  judgeStrategy: 'pointwise',
});

// Heuristic mode
const results = await rerank(query, documents, {
  mode: 'heuristic',
});

console.log(results);
// [
//   { id: 'doc-3', text: '...', score: 1.0,  rank: 1, metadata: { ... } },
//   { id: 'doc-1', text: '...', score: 0.87, rank: 2, metadata: { ... } },
//   { id: 'doc-5', text: '...', score: 0.62, rank: 3, metadata: { ... } },
//   ...
// ]
```

**Signature**:

```typescript
function rerank(
  query: string,
  documents: Document[],
  options?: RerankOptions,
): Promise<RerankResult[]>;
```

The function is async because cross-encoder mode performs ONNX inference (potentially with lazy model loading) and LLM-as-judge mode calls an async LLM function. Heuristic mode is synchronous internally but the function signature is uniformly async for API consistency.

**Behavior**:
1. Validate inputs (query is a non-empty string, documents is a non-empty array).
2. Assign IDs to documents that do not have an explicit `id` field (auto-generated: `"doc-0"`, `"doc-1"`, ...).
3. Select the reranking mode based on `options.mode` (default: `'heuristic'` if no `onnxruntime-node` detected, `'cross-encoder'` if available).
4. Score each document using the selected mode.
5. Normalize scores to [0, 1] via min-max normalization (unless `normalizeScores: false`).
6. Sort by score descending.
7. Assign ranks (1-based).
8. Optionally limit to top K results (`topK` option).
9. Return `RerankResult[]`.

### Factory Export: `createReranker`

```typescript
import { createReranker } from 'rerank-lite';

const reranker = createReranker({
  mode: 'cross-encoder',
  model: 'ms-marco-MiniLM-L-6-v2',
  maxBatchSize: 32,
});

// Reuse across many queries -- model stays loaded in memory
const results1 = await reranker.rerank(query1, documents1);
const results2 = await reranker.rerank(query2, documents2);
```

**Signature**:

```typescript
function createReranker(config: RerankerConfig): Reranker;

interface Reranker {
  rerank(
    query: string,
    documents: Document[],
    overrides?: Partial<RerankOptions>,
  ): Promise<RerankResult[]>;

  /** Pre-load the model into memory (cross-encoder mode only). */
  warmup(): Promise<void>;

  /** Release model resources. */
  dispose(): Promise<void>;
}
```

`createReranker` validates the configuration at construction time. The returned `Reranker` instance holds the ONNX session in memory across calls (for cross-encoder mode), avoiding the cost of reloading the model on every `rerank()` call. Call `warmup()` to pre-load the model before the first query, or let it lazy-load on the first `rerank()` call. Call `dispose()` to release the ONNX session and free memory.

### Convenience Exports

```typescript
import { rerankCrossEncoder, rerankWithLLM, rerankHeuristic } from 'rerank-lite';

// Mode-specific functions with tighter option types
const results = await rerankCrossEncoder(query, documents, {
  model: 'bge-reranker-base',
});

const results = await rerankWithLLM(query, documents, {
  llm: myLLMFunction,
  judgeStrategy: 'listwise',
});

const results = await rerankHeuristic(query, documents, {
  heuristicWeights: { coverage: 0.3, bm25: 0.5, density: 0.1, position: 0.1 },
});
```

### TypeScript Type Definitions

```typescript
// -- Input Types ----------------------------------------------------------

/** A document to be scored for relevance to a query. */
interface Document {
  /**
   * Unique document identifier. Used to track documents through the reranking
   * pipeline. If omitted, an auto-generated ID is assigned based on array position.
   */
  id?: string;

  /** The text content of the document. Required. */
  text: string;

  /** Arbitrary metadata passed through to the RerankResult unchanged. */
  metadata?: Record<string, unknown>;
}

// -- Mode Types -----------------------------------------------------------

/** Reranking mode identifiers. */
type RerankMode =
  | 'cross-encoder'  // Local ONNX cross-encoder model inference
  | 'llm-judge'      // LLM-as-judge relevance scoring
  | 'heuristic';     // Keyword-based heuristic scoring

/** LLM-as-judge prompting strategy. */
type JudgeStrategy =
  | 'pointwise'   // Score each document independently
  | 'listwise';   // Rank all documents in one prompt

/** Built-in cross-encoder model identifiers. */
type CrossEncoderModel =
  | 'ms-marco-MiniLM-L-6-v2'
  | 'bge-reranker-base'
  | 'bge-reranker-v2-m3';

// -- Options --------------------------------------------------------------

/** Options for the rerank() function. */
interface RerankOptions {
  /**
   * Reranking mode.
   * Default: 'cross-encoder' if onnxruntime-node is available, 'heuristic' otherwise.
   */
  mode?: RerankMode;

  // -- Cross-encoder options --

  /**
   * Cross-encoder model to use.
   * A built-in model name or a path to a custom ONNX model file.
   * Default: 'ms-marco-MiniLM-L-6-v2'.
   */
  model?: CrossEncoderModel | string;

  /**
   * Path to a custom ONNX model file.
   * Overrides the model option when both are provided.
   */
  modelPath?: string;

  /**
   * Path to the tokenizer.json file for a custom model.
   * Required when modelPath is set.
   */
  tokenizerPath?: string;

  /**
   * Maximum number of (query, document) pairs per inference batch.
   * Higher values improve throughput but consume more memory.
   * Default: 32.
   */
  maxBatchSize?: number;

  /**
   * Maximum sequence length for tokenization.
   * Overrides the model's default max sequence length.
   */
  maxSeqLength?: number;

  /**
   * Directory for caching downloaded models.
   * Default: ~/.cache/rerank-lite/models
   */
  cacheDir?: string;

  // -- LLM-as-judge options --

  /**
   * Async function that sends a prompt to an LLM and returns the response.
   * Required when mode is 'llm-judge'.
   */
  llm?: LLMFunction;

  /**
   * Prompting strategy for LLM-as-judge.
   * Default: 'pointwise'.
   */
  judgeStrategy?: JudgeStrategy;

  /**
   * Custom prompt template for LLM-as-judge.
   * Must contain {query} and {document} placeholders.
   * Only used with pointwise strategy.
   */
  promptTemplate?: string;

  /**
   * Number of documents per LLM prompt for pointwise batching.
   * Default: 10.
   */
  judgeBatchSize?: number;

  /**
   * Timeout for each LLM call in milliseconds.
   * Default: 30_000.
   */
  llmTimeout?: number;

  /**
   * Maximum number of document tokens to include in LLM prompts.
   * Documents longer than this are truncated.
   * Default: 500.
   */
  maxDocTokensInPrompt?: number;

  /**
   * Number of concurrent LLM calls.
   * Default: 3.
   */
  llmConcurrency?: number;

  // -- Heuristic options --

  /**
   * Weights for heuristic scoring signals.
   * Auto-normalized to sum to 1.0.
   * Default: { coverage: 0.35, bm25: 0.40, density: 0.10, position: 0.15 }.
   */
  heuristicWeights?: {
    coverage?: number;
    bm25?: number;
    density?: number;
    position?: number;
  };

  /**
   * Custom stop word set for heuristic mode.
   * Set to [] to disable stop word removal.
   */
  stopWords?: string[];

  /**
   * Custom stemming function for heuristic mode.
   * Applied to both query terms and document terms before matching.
   */
  stemmer?: (word: string) => string;

  /**
   * BM25 k1 parameter (term frequency saturation).
   * Default: 1.2.
   */
  bm25K1?: number;

  /**
   * BM25 b parameter (document length normalization).
   * Default: 0.75.
   */
  bm25B?: number;

  // -- General options --

  /**
   * Maximum number of results to return.
   * Default: Infinity (return all reranked documents).
   */
  topK?: number;

  /**
   * Whether to normalize output scores to [0, 1] via min-max.
   * Default: true.
   */
  normalizeScores?: boolean;

  /**
   * Field name to use as the document identifier.
   * Default: 'id'.
   */
  idField?: string;
}

/** Configuration for createReranker(). Same shape as RerankOptions. */
type RerankerConfig = RerankOptions;

/** Async LLM function type. */
type LLMFunction = (prompt: string) => Promise<string>;

// -- Output Types ---------------------------------------------------------

/** A document that has been scored and ranked. */
interface RerankResult {
  /** Document identifier. */
  id: string;

  /** The document text. */
  text: string;

  /**
   * Relevance score. When normalizeScores is true (default), this is in [0, 1].
   * Higher is more relevant.
   */
  score: number;

  /**
   * Rank in the reranked list (1-based). Rank 1 = most relevant.
   */
  rank: number;

  /** Original metadata, passed through from the input Document. */
  metadata?: Record<string, unknown>;
}

// -- Error ----------------------------------------------------------------

class RerankError extends Error {
  readonly code: RerankErrorCode;
  readonly cause?: Error;
}

type RerankErrorCode =
  | 'EMPTY_QUERY'            // Query string is empty
  | 'EMPTY_DOCUMENTS'        // Documents array is empty
  | 'MODEL_NOT_FOUND'        // Cross-encoder model file not found and download failed
  | 'MODEL_LOAD_ERROR'       // ONNX model failed to load
  | 'TOKENIZER_ERROR'        // Tokenizer failed to initialize or tokenize input
  | 'INFERENCE_ERROR'        // ONNX inference failed
  | 'ONNX_NOT_AVAILABLE'     // onnxruntime-node not installed, cross-encoder mode requested
  | 'LLM_ERROR'              // LLM function threw an error
  | 'LLM_TIMEOUT'            // LLM function did not respond within llmTimeout
  | 'LLM_PARSE_ERROR'        // Could not parse a numeric score from LLM response
  | 'MISSING_LLM_FUNCTION'   // mode='llm-judge' but no llm function provided
  | 'INVALID_OPTIONS';       // Other configuration errors
```

---

## 10. Model Management

### Model Registry

The built-in model registry maps model identifiers to download URLs and metadata:

```typescript
const MODEL_REGISTRY: Record<CrossEncoderModel, ModelInfo> = {
  'ms-marco-MiniLM-L-6-v2': {
    onnxUrl: 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx',
    tokenizerUrl: 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json',
    maxSeqLength: 512,
    numLabels: 1,
    sizeBytes: 89_000_000,
    description: 'Fast English reranker, 22M params',
  },
  'bge-reranker-base': {
    onnxUrl: 'https://huggingface.co/BAAI/bge-reranker-base/resolve/main/onnx/model.onnx',
    tokenizerUrl: 'https://huggingface.co/BAAI/bge-reranker-base/resolve/main/tokenizer.json',
    maxSeqLength: 512,
    numLabels: 1,
    sizeBytes: 440_000_000,
    description: 'High-accuracy English reranker, 110M params',
  },
  'bge-reranker-v2-m3': {
    onnxUrl: 'https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/onnx/model.onnx',
    tokenizerUrl: 'https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/tokenizer.json',
    maxSeqLength: 8192,
    numLabels: 1,
    sizeBytes: 1_100_000_000,
    description: 'Multilingual reranker, 568M params, 100+ languages',
  },
};
```

### Model Download

When a cross-encoder model is requested but not found in the local cache, `rerank-lite` downloads it from the registered URL. The download process:

1. Check if the model file exists in `cacheDir` (default: `~/.cache/rerank-lite/models/<model-name>/`).
2. If present, verify the file size matches the registry's `sizeBytes`. If mismatched (corrupt download), delete and re-download.
3. If absent, download the ONNX model file and tokenizer.json from the registered URLs.
4. Write files to the cache directory with a `.tmp` suffix during download, then rename atomically on completion. This prevents partial downloads from corrupting the cache.
5. Log download progress to stderr (file name, size, progress percentage) if the environment is a TTY.

Downloads use Node.js `https` module with no additional dependencies. A download failure throws `RerankError` with code `MODEL_NOT_FOUND`.

### Model Caching

Downloaded models are stored in a flat directory structure:

```
~/.cache/rerank-lite/models/
├── ms-marco-MiniLM-L-6-v2/
│   ├── model.onnx
│   └── tokenizer.json
├── bge-reranker-base/
│   ├── model.onnx
│   └── tokenizer.json
└── bge-reranker-v2-m3/
    ├── model.onnx
    └── tokenizer.json
```

The `cacheDir` option overrides the default cache location. This is useful for CI environments, containerized deployments, or shared filesystems where the home directory may not be writable.

### Model Loading

Model loading is lazy by default. The ONNX session is created on the first `rerank()` call that uses cross-encoder mode. Subsequent calls reuse the loaded session. When using `createReranker()`, the session persists for the lifetime of the `Reranker` instance.

For applications that cannot tolerate the latency of first-call model loading, call `reranker.warmup()` at application startup to pre-load the model.

Loading a model involves:
1. Create an `onnxruntime.InferenceSession` from the cached model file.
2. Read and parse the `tokenizer.json` file.
3. Validate the model's input and output tensor shapes.

If `onnxruntime-node` is not installed and cross-encoder mode is requested, a `RerankError` with code `ONNX_NOT_AVAILABLE` is thrown with a clear message explaining how to install it.

---

## 11. Configuration Reference

All options with their defaults, types, and descriptions:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | `RerankMode` | auto-detect | Reranking mode. |
| `model` | `CrossEncoderModel \| string` | `'ms-marco-MiniLM-L-6-v2'` | Cross-encoder model. |
| `modelPath` | `string` | (none) | Path to custom ONNX model file. |
| `tokenizerPath` | `string` | (none) | Path to custom tokenizer.json. |
| `maxBatchSize` | `number` | `32` | Max pairs per inference batch. |
| `maxSeqLength` | `number` | model-specific | Max token sequence length. |
| `cacheDir` | `string` | `~/.cache/rerank-lite/models` | Model cache directory. |
| `llm` | `LLMFunction` | (none) | Async LLM function. Required for llm-judge. |
| `judgeStrategy` | `JudgeStrategy` | `'pointwise'` | LLM scoring strategy. |
| `promptTemplate` | `string` | built-in | Custom prompt template. |
| `judgeBatchSize` | `number` | `10` | Documents per LLM prompt. |
| `llmTimeout` | `number` | `30_000` | LLM call timeout (ms). |
| `maxDocTokensInPrompt` | `number` | `500` | Max document tokens in LLM prompt. |
| `llmConcurrency` | `number` | `3` | Max concurrent LLM calls. |
| `heuristicWeights` | `object` | `{ coverage: 0.35, bm25: 0.40, density: 0.10, position: 0.15 }` | Heuristic signal weights. |
| `stopWords` | `string[]` | built-in English (~175 words) | Stop word set. |
| `stemmer` | `(word: string) => string` | (none) | Stemming function. |
| `bm25K1` | `number` | `1.2` | BM25 k1 parameter. |
| `bm25B` | `number` | `0.75` | BM25 b parameter. |
| `topK` | `number` | `Infinity` | Max results to return. |
| `normalizeScores` | `boolean` | `true` | Normalize output scores to [0, 1]. |
| `idField` | `string` | `'id'` | Field name for document identifier. |

---

## 12. CLI

### Installation and Invocation

```bash
# Global install
npm install -g rerank-lite
rerank-lite --query "CUDA memory error" --mode heuristic < documents.json

# npx
npx rerank-lite --query "..." --mode cross-encoder < documents.json

# As a pipeline stage
retriever --query "..." | rerank-lite --query "..." --mode heuristic --top-k 10 | context-packer --budget 4000
```

### CLI Binary Name

`rerank-lite`

### Input Format

The CLI reads a JSON array of document objects from stdin:

```json
[
  {
    "id": "doc-1",
    "text": "CUDA out of memory errors typically occur when..."
  },
  {
    "id": "doc-2",
    "text": "Memory management in Python involves..."
  },
  {
    "id": "doc-3",
    "text": "GPU memory allocation strategies for deep learning..."
  }
]
```

Documents may also be provided as a file path argument:

```bash
rerank-lite --query "CUDA memory error" --mode heuristic documents.json
```

### Output Format

By default, the CLI writes a JSON array of `RerankResult` objects to stdout:

```json
[
  {
    "id": "doc-1",
    "text": "CUDA out of memory errors typically occur when...",
    "score": 1.0,
    "rank": 1
  },
  {
    "id": "doc-3",
    "text": "GPU memory allocation strategies for deep learning...",
    "score": 0.72,
    "rank": 2
  },
  {
    "id": "doc-2",
    "text": "Memory management in Python involves...",
    "score": 0.31,
    "rank": 3
  }
]
```

With `--ids-only`, only the document IDs are written (one per line):

```
doc-1
doc-3
doc-2
```

With `--scores-only`, IDs and scores are written (tab-separated, one per line):

```
doc-1	1.00
doc-3	0.72
doc-2	0.31
```

### Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--query` | `-q` | string | (required) | The query string. |
| `--mode` | `-m` | string | auto-detect | Reranking mode: cross-encoder, llm-judge, heuristic. |
| `--model` | | string | `ms-marco-MiniLM-L-6-v2` | Cross-encoder model. |
| `--top-k` | `-k` | number | (all) | Maximum results to return. |
| `--batch-size` | `-b` | number | `32` | Inference batch size. |
| `--cache-dir` | | string | `~/.cache/rerank-lite/models` | Model cache directory. |
| `--no-normalize` | | boolean | `false` | Disable score normalization. |
| `--ids-only` | | boolean | `false` | Output only document IDs. |
| `--scores-only` | | boolean | `false` | Output IDs and scores only. |
| `--pretty` | `-p` | boolean | `false` | Pretty-print JSON output. |
| `--heuristic-weights` | | string | (default) | Comma-separated weights: "coverage,bm25,density,position". |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Reranking completed successfully. |
| `1` | Reranking failed (model load error, inference error, LLM error). |
| `2` | Configuration error (invalid flags, missing required options). |

### CLI Examples

```bash
# Heuristic reranking with top-5 results
rerank-lite --query "CUDA memory error" --mode heuristic --top-k 5 < docs.json

# Cross-encoder reranking with bge-reranker-base
rerank-lite --query "how to fix segfault" --mode cross-encoder --model bge-reranker-base < docs.json

# Heuristic with custom weights (favor BM25)
rerank-lite --query "protein folding" --mode heuristic \
  --heuristic-weights "0.2,0.6,0.1,0.1" < docs.json

# IDs only for piping to downstream tools
rerank-lite --query "..." --mode heuristic --ids-only < docs.json | head -10

# Pipeline: fusion → rerank → pack
fusion-rank dense.json sparse.json | \
  rerank-lite --query "..." --mode cross-encoder --top-k 20 | \
  context-packer --budget 4000
```

---

## 13. Integration

### With `fusion-rank`

`rerank-lite` integrates with `fusion-rank` in two patterns:

**Pattern 1: Rerank after fusion.** Fuse dense and sparse results first, then rerank the fused top-K for precision.

```typescript
import { rrf } from 'fusion-rank';
import { rerank } from 'rerank-lite';
import { pack } from 'context-packer';

// Stage 1: Fuse dense + sparse results
const fused = rrf([denseResults, sparseResults], { topK: 50 });

// Stage 2: Rerank the fused top-50 for precision
const reranked = await rerank(query, fused.map(r => ({
  id: r.id,
  text: r.metadata?.text as string,
  metadata: r.metadata,
})), {
  mode: 'cross-encoder',
  model: 'ms-marco-MiniLM-L-6-v2',
  topK: 20,
});

// Stage 3: Pack into context window
const packed = pack(
  reranked.map(r => ({ id: r.id, content: r.text, score: r.score })),
  { budget: 4000, strategy: 'mmr', ordering: 'u-shaped' },
);
```

**Pattern 2: Rerank as a fusion signal.** Rerank the top dense results, then fuse reranked results alongside dense and sparse as a third signal.

```typescript
import { rrf } from 'fusion-rank';
import { rerank } from 'rerank-lite';

// Rerank the top-20 dense results
const reranked = await rerank(query, denseResults.slice(0, 20).map(r => ({
  id: r.id,
  text: r.text,
})));

// Three-way fusion: dense + sparse + reranked
const fused = rrf([
  denseResults.map(r => ({ id: r.id, score: r.score })),
  sparseResults.map(r => ({ id: r.id, score: r.score })),
  reranked.map(r => ({ id: r.id, score: r.score })),
], { topK: 20 });
```

### With `context-packer`

`rerank-lite`'s output is scored and ranked, ready to feed directly into `context-packer` for budget-aware selection and positional ordering:

```typescript
import { rerank } from 'rerank-lite';
import { pack } from 'context-packer';

const reranked = await rerank(query, retrievedChunks, {
  mode: 'cross-encoder',
  topK: 30,
});

const packed = pack(
  reranked.map(r => ({
    id: r.id,
    content: r.text,
    score: r.score,
    metadata: r.metadata,
  })),
  {
    budget: 4000,
    strategy: 'mmr',
    lambda: 0.6,
    ordering: 'u-shaped',
  },
);
```

The reranker's score becomes the packer's relevance signal. Because reranker scores are more accurate than raw retrieval scores, packing after reranking produces higher-quality context than packing directly from retrieval results.

### With `sparse-encode`

`sparse-encode` provides BM25 sparse vector encoding for keyword-based retrieval. In a full pipeline, sparse retrieval provides candidates, and `rerank-lite` reorders them for precision:

```typescript
import { createBM25 } from 'sparse-encode';
import { rerank } from 'rerank-lite';

const bm25 = createBM25();
bm25.fit(corpus);
const sparseQuery = bm25.encodeQuery(query);
const sparseResults = await vectorDb.searchSparse(sparseQuery, { topK: 50 });

const reranked = await rerank(query, sparseResults.map(r => ({
  id: r.id,
  text: r.text,
})), {
  mode: 'cross-encoder',
  topK: 20,
});
```

### With `embed-cache`

`embed-cache` provides cached dense embeddings for vector retrieval. Dense retrieval provides candidates, and `rerank-lite` reorders them:

```typescript
import { createCache } from 'embed-cache';
import { rerank } from 'rerank-lite';

const embedCache = createCache({ model: 'text-embedding-3-small', embedder });
const queryEmbedding = await embedCache.embed(query);
const denseResults = await vectorDb.search(queryEmbedding, { topK: 50 });

const reranked = await rerank(query, denseResults.map(r => ({
  id: r.id,
  text: r.text,
})), {
  mode: 'cross-encoder',
  topK: 20,
});
```

---

## 14. Testing Strategy

### Unit Tests

Each reranking mode is tested independently with deterministic inputs and verifiable outputs.

**Cross-encoder tests:**
- Verify that tokenization of a (query, document) pair produces the expected `input_ids` sequence: `[CLS] query_tokens [SEP] doc_tokens [SEP]`.
- Verify that `token_type_ids` correctly marks query tokens as 0 and document tokens as 1.
- Verify that `attention_mask` marks real tokens as 1 and padding as 0.
- Verify truncation: when query + document exceeds max sequence length, the document is truncated and the query is preserved in full.
- Verify batched inference: scoring 5 documents in one batch produces the same scores as scoring each individually (within floating-point tolerance).
- Verify score normalization: output scores are in [0, 1] with the highest raw score mapping to 1.0 and the lowest to 0.0.
- Verify model loading: loading a model from the cache directory succeeds. Loading a non-existent model throws `MODEL_NOT_FOUND`.
- Verify `ONNX_NOT_AVAILABLE` error when `onnxruntime-node` is not installed and cross-encoder mode is requested.

**LLM-as-judge tests:**
- Verify pointwise prompt construction: the prompt contains the query and document in the expected format.
- Verify pointwise score parsing: "8" is parsed as 8, "  7  " is parsed as 7, "The score is 9" is parsed as 9 (first integer extraction).
- Verify pointwise batch prompting: a batch of 3 documents produces a prompt listing all 3, and the response "1: 8\n2: 5\n3: 9" is parsed correctly.
- Verify listwise prompt construction and response parsing: "3, 1, 2" assigns rank 1 to document 3, rank 2 to document 1, rank 3 to document 2.
- Verify LLM error handling: if the LLM function throws, a `RerankError` with code `LLM_ERROR` is thrown.
- Verify LLM timeout: if the LLM function does not resolve within `llmTimeout`, a `RerankError` with code `LLM_TIMEOUT` is thrown.
- Verify unparseable response: a non-numeric LLM response results in score 0 for the affected document, not a thrown error.
- Verify `MISSING_LLM_FUNCTION` error when mode is `'llm-judge'` but no `llm` function is provided.

**Heuristic tests:**
- Verify query term coverage: query "CUDA memory error", document containing "cuda" and "error" but not "memory" produces coverage 2/3.
- Verify stop word removal: query "what is the error" with stop words removed becomes ["error"], and coverage is computed over ["error"] only.
- Verify BM25 scoring: given a known set of documents with known term frequencies, verify the BM25 score matches hand-computed values.
- Verify BM25 IDF: a term appearing in 1 of 10 documents has higher IDF than a term appearing in 9 of 10 documents.
- Verify keyword density: a document with 10 query term occurrences in 100 tokens has density 0.10.
- Verify position bonus: a query term at position 0 gets bonus 1.0; at the last position gets bonus ~0.0.
- Verify composite score: given known signal values and weights, verify the composite score matches the expected weighted sum.
- Verify custom heuristic weights are applied correctly and auto-normalized to sum to 1.0.
- Verify custom stemmer is applied to both query and document terms.

**General tests:**
- Verify output is sorted by score descending.
- Verify ranks are assigned 1-based and contiguous.
- Verify `topK` limits the output length.
- Verify `normalizeScores: false` returns raw scores without min-max normalization.
- Verify `idField` option uses the specified field for document identification.
- Verify auto-generated IDs ("doc-0", "doc-1", ...) when documents lack `id` fields.
- Verify metadata pass-through: input document metadata appears unchanged in the corresponding RerankResult.
- Verify `EMPTY_QUERY` error when query is empty.
- Verify `EMPTY_DOCUMENTS` error when documents array is empty.

### Integration Tests

- **End-to-end cross-encoder**: Load `ms-marco-MiniLM-L-6-v2`, score 10 documents with a known query, verify the top-3 results are semantically relevant documents (not just keyword matches). This requires the model to be downloaded; the test should be tagged as a slow/integration test.
- **End-to-end LLM-as-judge**: Use a mock LLM function that returns predictable scores, rerank 10 documents, verify the output ordering matches the mock scores.
- **End-to-end heuristic**: Score 20 documents with known content, verify that documents with high query term overlap rank above documents with no overlap.
- **createReranker round-trip**: Create a reranker, rerank multiple independent queries, verify each call is independent (no state leakage between queries).
- **createReranker warmup**: Call `warmup()`, verify subsequent `rerank()` calls do not incur model loading latency (measure timing).
- **createReranker dispose**: Call `dispose()`, verify subsequent `rerank()` calls trigger model reloading.
- **CLI end-to-end**: Pipe JSON through the CLI binary, verify stdout matches the expected reranked output.
- **Mode fallback**: With `onnxruntime-node` not available, verify that the default mode falls back to `'heuristic'` without throwing.

### Property-Based Tests

Using a property-based testing framework (fast-check):

- **Score bounds**: When `normalizeScores: true`, all output scores are in [0, 1].
- **Rank consistency**: Ranks are 1-based, contiguous, and match the score ordering (rank 1 has the highest score).
- **Idempotency**: Reranking the already-reranked output with the same query produces the same ordering.
- **Document preservation**: Every input document appears exactly once in the output (no duplicates, no omissions), unless `topK` is set.
- **TopK enforcement**: When `topK` is set, the output length is `min(topK, documents.length)`.

---

## 15. Performance

### Cross-Encoder Latency

Measured on Node.js 22, Apple M3, `onnxruntime-node`:

| Model | Documents | Batch Size | First Call (incl. load) | Subsequent Calls |
|-------|----------|-----------|------------------------|-----------------|
| ms-marco-MiniLM-L-6-v2 | 10 | 32 | ~800ms | ~80ms |
| ms-marco-MiniLM-L-6-v2 | 20 | 32 | ~850ms | ~150ms |
| ms-marco-MiniLM-L-6-v2 | 50 | 32 | ~950ms | ~350ms |
| bge-reranker-base | 10 | 32 | ~2000ms | ~200ms |
| bge-reranker-base | 20 | 32 | ~2100ms | ~400ms |
| bge-reranker-v2-m3 | 10 | 16 | ~5000ms | ~600ms |
| bge-reranker-v2-m3 | 20 | 16 | ~5200ms | ~1200ms |

The first call includes model loading time (creating the ONNX session). Subsequent calls reuse the loaded session. Use `createReranker` with `warmup()` to move loading out of the request path.

### LLM-as-Judge Latency

Depends entirely on the LLM provider and model. Estimated latencies:

| Provider | Model | 20 Docs (pointwise, batch 10) | 20 Docs (listwise) |
|----------|-------|-------------------------------|---------------------|
| OpenAI | gpt-4o-mini | ~2-4s (2 calls) | ~1-2s (1 call) |
| OpenAI | gpt-4o | ~4-8s (2 calls) | ~2-4s (1 call) |
| Anthropic | claude-haiku | ~2-4s (2 calls) | ~1-2s (1 call) |
| Local | llama3-8b (Ollama) | ~10-20s (2 calls) | ~5-10s (1 call) |

### Heuristic Latency

| Documents | Expected Time |
|----------|---------------|
| 10 | < 0.1ms |
| 50 | < 0.5ms |
| 100 | < 1ms |
| 500 | < 5ms |

Heuristic scoring is pure string processing. It is never the bottleneck.

### Memory Footprint

| Component | Memory |
|-----------|--------|
| ms-marco-MiniLM-L-6-v2 ONNX session | ~100MB |
| bge-reranker-base ONNX session | ~500MB |
| bge-reranker-v2-m3 ONNX session | ~1.2GB |
| Tokenizer vocabulary | ~5MB |
| Heuristic mode (no model) | ~1MB |
| LLM-as-judge mode (no model) | ~1MB |

For memory-constrained environments, use `ms-marco-MiniLM-L-6-v2` (smallest model) or heuristic mode (no model at all). Call `reranker.dispose()` to release the ONNX session after use.

---

## 16. Dependencies

### Runtime Dependencies

**Zero mandatory runtime dependencies.** The heuristic mode and LLM-as-judge mode are implemented in pure TypeScript with no external packages.

### Optional Peer Dependencies

| Package | Required For | Why |
|---------|-------------|-----|
| `onnxruntime-node` | Cross-encoder mode | Provides ONNX model inference in Node.js. Without it, cross-encoder mode is unavailable; the package falls back to heuristic mode. Listed as an optional peer dependency so that teams not using cross-encoder mode pay no installation or binary size cost. |

### Dev Dependencies

| Package | Purpose |
|---------|---------|
| `typescript` | TypeScript compiler |
| `vitest` | Test runner |
| `eslint` | Linting |
| `@types/node` | Node.js type definitions |
| `onnxruntime-node` | For cross-encoder integration tests |

### Peer Dependencies

None required. `onnxruntime-node` is an optional peer dependency.

---

## 17. File Structure

```
rerank-lite/
├── package.json
├── tsconfig.json
├── SPEC.md
├── README.md
├── src/
│   ├── index.ts                  # Public API: exports rerank, createReranker,
│   │                             # rerankCrossEncoder, rerankWithLLM, rerankHeuristic, types
│   ├── types.ts                  # All TypeScript interfaces and type aliases
│   ├── rerank.ts                 # Core rerank() function and createReranker() factory
│   ├── modes/
│   │   ├── index.ts              # Mode dispatcher (selects mode based on options)
│   │   ├── cross-encoder.ts      # Cross-encoder scoring: tokenize, batch, infer, extract
│   │   ├── llm-judge.ts          # LLM-as-judge scoring: prompt, call, parse
│   │   └── heuristic.ts          # Heuristic scoring: coverage, BM25, density, position
│   ├── models/
│   │   ├── registry.ts           # Model registry: URLs, metadata, defaults
│   │   ├── download.ts           # Model download from HuggingFace Hub
│   │   ├── cache.ts              # Local filesystem model cache management
│   │   └── loader.ts             # ONNX session creation and tokenizer loading
│   ├── tokenizer/
│   │   ├── index.ts              # Tokenizer dispatcher (WordPiece, BPE)
│   │   ├── wordpiece.ts          # WordPiece tokenizer (BERT-family)
│   │   └── bpe.ts                # BPE tokenizer (RoBERTa-family)
│   ├── heuristic/
│   │   ├── coverage.ts           # Query term coverage signal
│   │   ├── bm25.ts               # BM25-style TF-IDF signal
│   │   ├── density.ts            # Keyword density signal
│   │   ├── position.ts           # Position bonus signal
│   │   └── stop-words.ts         # Built-in English stop word list
│   ├── prompts/
│   │   ├── pointwise.ts          # Pointwise LLM prompt template and parser
│   │   └── listwise.ts           # Listwise LLM prompt template and parser
│   ├── normalize.ts              # Min-max score normalization
│   ├── errors.ts                 # RerankError class and error codes
│   └── cli.ts                    # CLI entry point
├── src/__tests__/
│   ├── rerank.test.ts            # Integration tests for rerank() and createReranker()
│   ├── cross-encoder.test.ts     # Cross-encoder unit tests (tokenization, batching, scoring)
│   ├── llm-judge.test.ts         # LLM-as-judge unit tests (prompts, parsing, errors)
│   ├── heuristic.test.ts         # Heuristic unit tests (each signal, composite score)
│   ├── tokenizer.test.ts         # Tokenizer unit tests (WordPiece, BPE, special tokens)
│   ├── model-management.test.ts  # Model download, cache, and loading tests
│   ├── prompts.test.ts           # Prompt template and response parsing tests
│   ├── normalize.test.ts         # Score normalization tests
│   ├── cli.test.ts               # CLI end-to-end tests
│   └── properties.test.ts        # Property-based tests (fast-check)
├── src/__benchmarks__/
│   └── rerank-throughput.ts      # Performance benchmarks for all modes
└── dist/                         # Build output (gitignored)
    ├── index.js
    ├── index.d.ts
    └── ...
```

---

## 18. Implementation Roadmap

### Phase 1: Core Types and Heuristic Mode (v0.1.0)

1. Define all TypeScript types (`types.ts`, `errors.ts`).
2. Implement stop word list (`heuristic/stop-words.ts`).
3. Implement query term coverage signal (`heuristic/coverage.ts`).
4. Implement BM25-style TF-IDF signal (`heuristic/bm25.ts`).
5. Implement keyword density signal (`heuristic/density.ts`).
6. Implement position bonus signal (`heuristic/position.ts`).
7. Implement heuristic mode combining all four signals (`modes/heuristic.ts`).
8. Implement min-max score normalization (`normalize.ts`).
9. Implement `rerank()` function with heuristic mode (`rerank.ts`).
10. Implement `rerankHeuristic()` convenience export.
11. Wire up `index.ts` exports.
12. Write unit tests for all heuristic signals.
13. Write unit tests for normalization.
14. Write integration tests for heuristic reranking.

### Phase 2: LLM-as-Judge Mode (v0.2.0)

15. Implement pointwise prompt template and response parser (`prompts/pointwise.ts`).
16. Implement listwise prompt template and response parser (`prompts/listwise.ts`).
17. Implement LLM-as-judge mode with batching and concurrency (`modes/llm-judge.ts`).
18. Implement `rerankWithLLM()` convenience export.
19. Write unit tests for prompt construction and response parsing.
20. Write unit tests for LLM error handling and timeout.
21. Write integration tests with mock LLM function.

### Phase 3: Cross-Encoder Mode (v0.3.0)

22. Implement model registry (`models/registry.ts`).
23. Implement model download from HuggingFace Hub (`models/download.ts`).
24. Implement model cache management (`models/cache.ts`).
25. Implement WordPiece tokenizer (`tokenizer/wordpiece.ts`).
26. Implement BPE tokenizer fallback (`tokenizer/bpe.ts`).
27. Implement tokenizer dispatcher (`tokenizer/index.ts`).
28. Implement ONNX session creation and model loading (`models/loader.ts`).
29. Implement cross-encoder scoring with batched inference (`modes/cross-encoder.ts`).
30. Implement `rerankCrossEncoder()` convenience export.
31. Write tokenizer unit tests.
32. Write model management unit tests (download, cache, loading).
33. Write cross-encoder scoring unit tests (batching, truncation, score extraction).
34. Write integration tests with a real ONNX model (slow tests, tagged).

### Phase 4: Factory, CLI, and Polish (v0.4.0)

35. Implement `createReranker()` factory with `warmup()` and `dispose()` (`rerank.ts`).
36. Implement mode dispatcher and auto-detection (`modes/index.ts`).
37. Implement CLI (`cli.ts`): parse flags, read stdin/files, call `rerank()`, write stdout.
38. Add CLI binary to `package.json` (`"bin": { "rerank-lite": "dist/cli.js" }`).
39. Write `createReranker` integration tests.
40. Write CLI end-to-end tests.
41. Write property-based tests.

### Phase 5: Integration and Documentation (v0.5.0)

42. Write performance benchmarks for all three modes.
43. Document integration patterns with `fusion-rank`, `context-packer`, `sparse-encode`, `embed-cache`.
44. Write `README.md` with quickstart, examples, and API reference.
45. Publish v0.5.0 to npm.

---

## 19. Example Use Cases

### Example 1: RAG Pipeline with Cross-Encoder Reranking

A document Q&A system retrieves 50 candidates from Pinecone and reranks the top 20 with a cross-encoder before packing into the LLM context:

```typescript
import { createReranker } from 'rerank-lite';
import { pack } from 'context-packer';
import { Pinecone } from '@pinecone-database/pinecone';

const pinecone = new Pinecone();
const index = pinecone.index('knowledge-base');

// Create a reranker instance -- model loads once, reused across queries
const reranker = createReranker({
  mode: 'cross-encoder',
  model: 'ms-marco-MiniLM-L-6-v2',
});
await reranker.warmup();

async function answerQuestion(query: string) {
  // Stage 1: Dense retrieval
  const queryEmbedding = await embedder.embed(query);
  const hits = await index.query({
    vector: queryEmbedding,
    topK: 50,
    includeMetadata: true,
  });

  // Stage 2: Rerank for precision
  const reranked = await reranker.rerank(
    query,
    hits.matches.map(m => ({
      id: m.id,
      text: m.metadata?.text as string,
      metadata: m.metadata,
    })),
    { topK: 20 },
  );

  // Stage 3: Pack into context window
  const packed = pack(
    reranked.map(r => ({
      id: r.id,
      content: r.text,
      score: r.score,
      metadata: r.metadata,
    })),
    { budget: 4000, strategy: 'mmr', ordering: 'u-shaped' },
  );

  // Stage 4: Generate answer
  const answer = await llm.generate({
    context: packed.chunks.map(c => c.content).join('\n\n'),
    query,
  });

  return answer;
}
```

### Example 2: Comparing Reranking Modes

An evaluation engineer measures the precision impact of each reranking mode on a test set:

```typescript
import { rerank } from 'rerank-lite';

const modes = [
  { mode: 'heuristic' as const },
  { mode: 'cross-encoder' as const, model: 'ms-marco-MiniLM-L-6-v2' },
  { mode: 'cross-encoder' as const, model: 'bge-reranker-base' },
  {
    mode: 'llm-judge' as const,
    llm: gpt4oMiniFunction,
    judgeStrategy: 'pointwise' as const,
  },
];

for (const testQuery of evalDataset) {
  for (const config of modes) {
    const reranked = await rerank(
      testQuery.query,
      testQuery.candidates,
      config,
    );

    const topKIds = reranked.slice(0, 10).map(r => r.id);
    const relevant = new Set(testQuery.relevantDocIds);
    const precision = topKIds.filter(id => relevant.has(id)).length / 10;

    results.push({
      mode: config.mode,
      model: 'model' in config ? config.model : undefined,
      query: testQuery.id,
      precision,
    });
  }
}

// Aggregate results
// heuristic:                  avg precision@10 = 0.52
// cross-encoder (MiniLM):     avg precision@10 = 0.71
// cross-encoder (bge-base):   avg precision@10 = 0.74
// llm-judge (gpt-4o-mini):    avg precision@10 = 0.73
```

### Example 3: Three-Stage Retrieval Pipeline

A production search system uses BM25 retrieval, dense retrieval, cross-encoder reranking, and RRF fusion in a four-stage pipeline:

```typescript
import { createBM25 } from 'sparse-encode';
import { createCache } from 'embed-cache';
import { rerank } from 'rerank-lite';
import { rrf } from 'fusion-rank';
import { pack } from 'context-packer';

async function search(query: string) {
  // Stage 1: Broad retrieval from two sources
  const [denseResults, sparseResults] = await Promise.all([
    vectorDb.searchDense(await embedCache.embed(query), { topK: 50 }),
    vectorDb.searchSparse(bm25.encodeQuery(query), { topK: 50 }),
  ]);

  // Stage 2: Rerank the top dense results for a precision signal
  const reranked = await rerank(
    query,
    denseResults.slice(0, 20).map(r => ({
      id: r.id,
      text: r.text,
    })),
    { mode: 'cross-encoder' },
  );

  // Stage 3: Three-way fusion
  const fused = rrf([
    denseResults.map(r => ({ id: r.id, score: r.score })),
    sparseResults.map(r => ({ id: r.id, score: r.score })),
    reranked.map(r => ({ id: r.id, score: r.score })),
  ], { topK: 20 });

  // Stage 4: Pack into context
  const packed = pack(
    fused.map(r => ({
      id: r.id,
      content: r.metadata?.text as string,
      score: r.score,
    })),
    { budget: 4000, ordering: 'u-shaped' },
  );

  return packed;
}
```

### Example 4: Heuristic Reranking as a Fallback

A resilient RAG pipeline uses cross-encoder reranking with automatic fallback to heuristic mode if the ONNX model fails to load:

```typescript
import { rerank, RerankError } from 'rerank-lite';

async function resilientRerank(query: string, documents: Document[]) {
  try {
    return await rerank(query, documents, {
      mode: 'cross-encoder',
      model: 'ms-marco-MiniLM-L-6-v2',
    });
  } catch (err) {
    if (
      err instanceof RerankError &&
      (err.code === 'ONNX_NOT_AVAILABLE' || err.code === 'MODEL_LOAD_ERROR')
    ) {
      console.warn('Cross-encoder unavailable, falling back to heuristic reranking');
      return await rerank(query, documents, {
        mode: 'heuristic',
      });
    }
    throw err;
  }
}
```

### Example 5: LLM-as-Judge for Domain-Specific Reranking

A legal research tool uses an LLM with a custom prompt to assess legal relevance, which a general cross-encoder would miss:

```typescript
import { rerank } from 'rerank-lite';

const results = await rerank(query, legalDocuments, {
  mode: 'llm-judge',
  llm: claudeFunction,
  judgeStrategy: 'pointwise',
  promptTemplate: `You are an expert legal relevance judge. Given a legal query and a case document, rate how relevant the case is to the query on a scale of 0 to 10.

Consider:
- Jurisdictional relevance (same jurisdiction scores higher)
- Temporal relevance (more recent cases score higher)
- Precedential value (binding authority > persuasive authority)
- Factual similarity (similar facts to the query scenario)

Respond with ONLY a single integer from 0 to 10.

Query: {query}

Case Document: {document}

Relevance score:`,
  maxDocTokensInPrompt: 1000,
});
```

---

## 20. Prior Art and Alternatives

### Cohere Rerank API

Cohere provides a cloud reranking service accessed via `cohere.rerank(query, documents, model)`. It is the most popular reranking API and offers excellent accuracy with models like `rerank-english-v3.0` and `rerank-multilingual-v3.0`. However, it requires an API key, internet connectivity, and per-call costs ($1/1000 search units). `rerank-lite`'s cross-encoder mode provides comparable accuracy for local inference with no per-call cost, while the LLM-as-judge mode offers flexibility with any LLM provider.

### `@xenova/transformers` / `@huggingface/transformers`

The Hugging Face Transformers library for JavaScript can load ONNX cross-encoder models and run inference. However, it is a general-purpose ML inference library, not a reranking library. The developer must handle model selection, tokenization, tensor construction, output interpretation, batching, and score normalization manually. `rerank-lite` wraps this into a `rerank(query, documents)` call with sensible defaults.

### sentence-transformers (Python)

The Python `sentence-transformers` library provides `CrossEncoder` class for local cross-encoder inference. It is the de facto standard for Python-based reranking. `rerank-lite` provides the equivalent capability in JavaScript, using ONNX Runtime instead of PyTorch.

### LangChain Rerankers (Python)

LangChain provides `CohereRerank`, `CrossEncoderReranker`, and `LLMChainFilter` for reranking within LangChain pipelines. These are Python-only and tightly coupled to the LangChain framework. `rerank-lite` is framework-agnostic and works in any JavaScript application.

### FlashRank (Python)

FlashRank is a lightweight Python reranker that focuses on speed. It provides fast cross-encoder inference using ONNX. `rerank-lite` fills the same niche in the JavaScript ecosystem.

### Jina Reranker API

Jina AI provides a cloud reranking API similar to Cohere's. Same tradeoffs apply: requires API key, internet, and per-call cost. `rerank-lite` provides a local alternative.

### No JavaScript Alternative

As of this specification, there is no standalone npm package that provides a high-level `rerank(query, documents)` API with local model inference. Every JavaScript developer building a RAG pipeline must either call a cloud API, manually wire up `@huggingface/transformers` for cross-encoder inference, or skip reranking entirely. `rerank-lite` is the first package to fill this gap.
