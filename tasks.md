# rerank-lite — Task Breakdown

This file tracks all tasks required to implement `rerank-lite` per the SPEC.md. Tasks are grouped into phases matching the implementation roadmap.

---

## Phase 1: Core Types, Errors, and Heuristic Mode (v0.1.0)

### 1.1 Project Scaffolding

- [ ] **Install dev dependencies** — Add `typescript`, `vitest`, `eslint`, and `@types/node` as dev dependencies in `package.json`. Run `npm install` to generate `node_modules` and `package-lock.json`. | Status: not_done
- [ ] **Configure ESLint** — Add an ESLint configuration file (`.eslintrc` or `eslint.config.js`) appropriate for a TypeScript project. Ensure `npm run lint` works against `src/`. | Status: not_done
- [ ] **Create directory structure** — Create the directories specified in the file structure: `src/modes/`, `src/models/`, `src/tokenizer/`, `src/heuristic/`, `src/prompts/`, and `src/__tests__/`. | Status: not_done
- [ ] **Add .gitignore entries** — Ensure `dist/`, `node_modules/`, and any build artifacts are in `.gitignore`. | Status: not_done

### 1.2 TypeScript Type Definitions (`src/types.ts`)

- [ ] **Define `Document` interface** — With optional `id` (string), required `text` (string), and optional `metadata` (Record<string, unknown>). | Status: not_done
- [ ] **Define `RerankResult` interface** — With `id` (string), `text` (string), `score` (number), `rank` (number), and optional `metadata` (Record<string, unknown>). | Status: not_done
- [ ] **Define `RerankMode` type** — Union of `'cross-encoder'`, `'llm-judge'`, `'heuristic'`. | Status: not_done
- [ ] **Define `JudgeStrategy` type** — Union of `'pointwise'`, `'listwise'`. | Status: not_done
- [ ] **Define `CrossEncoderModel` type** — Union of `'ms-marco-MiniLM-L-6-v2'`, `'bge-reranker-base'`, `'bge-reranker-v2-m3'`. | Status: not_done
- [ ] **Define `LLMFunction` type** — `(prompt: string) => Promise<string>`. | Status: not_done
- [ ] **Define `RerankOptions` interface** — All options as specified in Section 11 of the spec: `mode`, `model`, `modelPath`, `tokenizerPath`, `maxBatchSize` (default 32), `maxSeqLength`, `cacheDir`, `llm`, `judgeStrategy` (default 'pointwise'), `promptTemplate`, `judgeBatchSize` (default 10), `llmTimeout` (default 30000), `maxDocTokensInPrompt` (default 500), `llmConcurrency` (default 3), `heuristicWeights`, `stopWords`, `stemmer`, `bm25K1` (default 1.2), `bm25B` (default 0.75), `topK` (default Infinity), `normalizeScores` (default true), `idField` (default 'id'). | Status: not_done
- [ ] **Define `RerankerConfig` type** — Alias for `RerankOptions`. | Status: not_done
- [ ] **Define `Reranker` interface** — With `rerank(query, documents, overrides?)`, `warmup()`, and `dispose()` methods. | Status: not_done
- [ ] **Define `HeuristicWeights` interface** — With optional `coverage`, `bm25`, `density`, `position` number fields. | Status: not_done
- [ ] **Define `ModelInfo` interface** — With `onnxUrl`, `tokenizerUrl`, `maxSeqLength`, `numLabels`, `sizeBytes`, `description` fields for the model registry. | Status: not_done

### 1.3 Error Handling (`src/errors.ts`)

- [ ] **Define `RerankErrorCode` type** — Union of all error codes: `EMPTY_QUERY`, `EMPTY_DOCUMENTS`, `MODEL_NOT_FOUND`, `MODEL_LOAD_ERROR`, `TOKENIZER_ERROR`, `INFERENCE_ERROR`, `ONNX_NOT_AVAILABLE`, `LLM_ERROR`, `LLM_TIMEOUT`, `LLM_PARSE_ERROR`, `MISSING_LLM_FUNCTION`, `INVALID_OPTIONS`. | Status: not_done
- [ ] **Implement `RerankError` class** — Extends `Error` with a `readonly code: RerankErrorCode` property and an optional `readonly cause?: Error` property. Ensure proper prototype chain and `name` property. | Status: not_done

### 1.4 Score Normalization (`src/normalize.ts`)

- [ ] **Implement `normalizeScores` function** — Takes an array of raw scores and applies min-max normalization to produce scores in [0, 1]. The highest score maps to 1.0, the lowest to 0.0. Handle the edge case where all scores are equal (return 0.5 or uniform scores). Handle single-document case (return 1.0). | Status: not_done

### 1.5 Heuristic Scoring Signals

- [ ] **Implement stop word list (`src/heuristic/stop-words.ts`)** — Export a default set of ~175 common English stop words (articles, prepositions, pronouns, common verbs). Export it as a `Set<string>` for O(1) lookup. | Status: not_done
- [ ] **Implement query term coverage signal (`src/heuristic/coverage.ts`)** — Given a query (as an array of stemmed, stop-word-filtered terms) and a document text, compute the fraction of unique query terms found in the document (case-insensitive). Return a number in [0, 1]. | Status: not_done
- [ ] **Implement BM25-style TF-IDF signal (`src/heuristic/bm25.ts`)** — Compute BM25 scores for each document in the candidate set given query terms. Use the formula from Section 8 of the spec with configurable `k1` (default 1.2) and `b` (default 0.75). Compute IDF over the candidate document set. Compute `avgDL` over the candidate set. Return raw BM25 scores per document (normalization is done externally). | Status: not_done
- [ ] **Implement keyword density signal (`src/heuristic/density.ts`)** — Given query terms and a document, compute `totalQueryTermOccurrences / documentTokenCount`. Return a raw density value per document. | Status: not_done
- [ ] **Implement position bonus signal (`src/heuristic/position.ts`)** — For each matched query term, find the character offset of its first occurrence in the document and compute `1 - (firstPosition / documentLength)`. Return the average across all matched terms. Return 0 if no terms match. | Status: not_done

### 1.6 Heuristic Mode (`src/modes/heuristic.ts`)

- [ ] **Implement heuristic scoring orchestrator** — Accept a query string, an array of documents, and heuristic options (weights, stopWords, stemmer, bm25K1, bm25B). Tokenize the query, remove stop words, optionally apply stemmer. Compute all four signals for each document. Combine with weighted sum using configurable weights (defaults: coverage 0.35, bm25 0.40, density 0.10, position 0.15). Auto-normalize weights to sum to 1.0. Return raw composite scores per document. | Status: not_done
- [ ] **Implement query tokenization helper** — Split query into lowercase tokens, filter by stop words, optionally apply stemmer function. Reuse across signals. | Status: not_done

### 1.7 Core `rerank()` Function (`src/rerank.ts`)

- [ ] **Implement `rerank()` function** — Async function taking `query`, `documents`, and optional `options`. Validate inputs: throw `EMPTY_QUERY` if query is empty, throw `EMPTY_DOCUMENTS` if documents array is empty. Auto-assign IDs (`doc-0`, `doc-1`, ...) to documents without an `id` field. Respect `idField` option. Select mode (for Phase 1, only heuristic is available). Score documents. Apply normalization unless `normalizeScores: false`. Sort by score descending. Assign 1-based ranks. Apply `topK` if set. Return `RerankResult[]`. | Status: not_done
- [ ] **Implement input validation** — Validate that query is a non-empty string. Validate that documents is a non-empty array. Validate that each document has a `text` field. Throw appropriate `RerankError` for each validation failure. | Status: not_done
- [ ] **Implement auto-ID assignment** — For documents without an `id` field (or without the field named by `idField`), generate IDs as `"doc-0"`, `"doc-1"`, etc. based on array index. | Status: not_done
- [ ] **Implement `topK` limiting** — After sorting, slice the results to the first `topK` entries if the option is set. | Status: not_done
- [ ] **Implement `normalizeScores: false` passthrough** — When `normalizeScores` is `false`, skip min-max normalization and return raw mode scores. | Status: not_done
- [ ] **Implement metadata pass-through** — Ensure each `RerankResult` includes the original document's `metadata` field, passed through unchanged. | Status: not_done

### 1.8 Convenience Export (`rerankHeuristic`)

- [ ] **Implement `rerankHeuristic()` function** — Calls `rerank()` with `mode: 'heuristic'` forced. Accepts a narrower options type (only heuristic-relevant options). | Status: not_done

### 1.9 Public API (`src/index.ts`)

- [ ] **Wire up index.ts exports for Phase 1** — Export `rerank`, `rerankHeuristic`, `RerankError`, and all TypeScript types/interfaces. Ensure the public API surface matches Section 9 of the spec (for what is available in Phase 1). | Status: not_done

### 1.10 Phase 1 Tests

- [ ] **Unit tests: stop words (`src/__tests__/heuristic.test.ts`)** — Verify the stop word set contains expected words (the, is, at, etc.) and does not contain content words (algorithm, error, etc.). | Status: not_done
- [ ] **Unit tests: query term coverage** — Verify coverage for full match (1.0), partial match (e.g., 3/4 = 0.75), no match (0.0). Verify case insensitivity. Verify stop word removal from query before coverage computation. | Status: not_done
- [ ] **Unit tests: BM25 scoring** — Verify BM25 scores against hand-computed values for a known document set. Verify IDF: a term in 1/10 docs has higher IDF than a term in 9/10 docs. Verify term frequency saturation (k1 parameter). Verify document length normalization (b parameter). | Status: not_done
- [ ] **Unit tests: keyword density** — Verify density = queryTermOccurrences / documentTokenCount. Test document with 10 occurrences of query terms in 100 tokens yields 0.10. | Status: not_done
- [ ] **Unit tests: position bonus** — Verify term at position 0 contributes 1.0, term at last position contributes ~0.0. Verify average across multiple matched terms. Verify 0 when no terms match. | Status: not_done
- [ ] **Unit tests: composite heuristic score** — Verify weighted combination with default weights. Verify custom weights. Verify weight auto-normalization to sum to 1.0. | Status: not_done
- [ ] **Unit tests: custom stemmer** — Verify that a custom stemmer function is applied to both query and document terms before matching. | Status: not_done
- [ ] **Unit tests: score normalization (`src/__tests__/normalize.test.ts`)** — Verify min-max normalization maps highest score to 1.0, lowest to 0.0. Verify scores between min and max are correctly scaled. Verify single-document edge case. Verify all-equal-scores edge case. | Status: not_done
- [ ] **Unit tests: rerank() general behavior (`src/__tests__/rerank.test.ts`)** — Verify output is sorted by score descending. Verify ranks are 1-based and contiguous. Verify `topK` limits output length. Verify `normalizeScores: false` returns raw scores. Verify `idField` uses specified field. Verify auto-generated IDs. Verify metadata pass-through. | Status: not_done
- [ ] **Unit tests: input validation errors** — Verify `EMPTY_QUERY` error on empty string query. Verify `EMPTY_DOCUMENTS` error on empty array. | Status: not_done
- [ ] **Integration test: heuristic end-to-end** — Score 20 documents with known content, verify documents with high query term overlap rank above documents with no overlap. Verify top result is the most relevant by keyword match. | Status: not_done
- [ ] **Verify build** — Run `npm run build` and ensure TypeScript compiles cleanly with no errors. | Status: not_done
- [ ] **Verify lint** — Run `npm run lint` and ensure no lint errors. | Status: not_done
- [ ] **Verify tests** — Run `npm run test` and ensure all tests pass. | Status: not_done

---

## Phase 2: LLM-as-Judge Mode (v0.2.0)

### 2.1 Prompt Templates and Parsers

- [ ] **Implement pointwise prompt template (`src/prompts/pointwise.ts`)** — Build the pointwise prompt as specified in Section 7: include the relevance scale (0-10), instruction to respond with a single integer, query placeholder, and document placeholder. Support custom `promptTemplate` with `{query}` and `{document}` placeholders. | Status: not_done
- [ ] **Implement pointwise response parser** — Extract the first integer from the LLM response string. Handle edge cases: "8", "  7  ", "The score is 9", empty string, non-numeric text. Return parsed integer or 0 on parse failure. | Status: not_done
- [ ] **Implement pointwise batch prompt template** — Build the batched pointwise prompt that presents multiple documents numbered 1-N and asks for scores in the format "1: <score>". Support `judgeBatchSize` option (default 10). | Status: not_done
- [ ] **Implement pointwise batch response parser** — Parse multi-line responses in format "1: 8\n2: 5\n3: 9". Handle partial responses (missing documents get score 0). Handle malformed lines. | Status: not_done
- [ ] **Implement listwise prompt template (`src/prompts/listwise.ts`)** — Build the listwise prompt that presents all documents and asks for a comma-separated ranking of document numbers from most to least relevant. | Status: not_done
- [ ] **Implement listwise response parser** — Parse comma-separated document numbers (e.g., "3, 1, 5, 2, 4"). Convert to scores using formula: `score = 1 - (rank - 1) / (N - 1)`. Handle missing document numbers (assign lowest score). Handle duplicate numbers. | Status: not_done
- [ ] **Implement document truncation for prompts** — Truncate document text to `maxDocTokensInPrompt` tokens (default 500) before including in LLM prompts. Use a simple whitespace-based token count approximation. | Status: not_done

### 2.2 LLM-as-Judge Mode (`src/modes/llm-judge.ts`)

- [ ] **Implement LLM-as-judge scoring orchestrator** — Accept query, documents, and LLM-judge options. Validate that `llm` function is provided (throw `MISSING_LLM_FUNCTION` if not). Select strategy based on `judgeStrategy` (default: pointwise). Dispatch to pointwise or listwise implementation. Return raw scores per document. | Status: not_done
- [ ] **Implement pointwise scoring with batching** — Split documents into batches of `judgeBatchSize`. For each batch, construct the batch prompt, call the LLM function, parse scores. Respect `llmConcurrency` (default 3) for parallel LLM calls across batches. | Status: not_done
- [ ] **Implement listwise scoring** — For small document sets (fits in one prompt), construct a single listwise prompt. For large sets, split into groups, rank each group, then merge rankings using a final cross-group comparison call. | Status: not_done
- [ ] **Implement LLM call timeout** — Wrap each LLM function call with a timeout of `llmTimeout` ms (default 30000). If the call does not resolve in time, abort and throw `RerankError` with code `LLM_TIMEOUT`. | Status: not_done
- [ ] **Implement LLM concurrency control** — Limit concurrent LLM calls to `llmConcurrency` (default 3). Use a simple semaphore or queue pattern. | Status: not_done
- [ ] **Implement LLM error handling** — Catch errors thrown by the LLM function and rethrow as `RerankError` with code `LLM_ERROR` and the original error as `cause`. Handle non-numeric/unparseable responses by assigning score 0 and emitting a warning (not throwing). | Status: not_done

### 2.3 Convenience Export (`rerankWithLLM`)

- [ ] **Implement `rerankWithLLM()` function** — Calls `rerank()` with `mode: 'llm-judge'` forced. Accepts a narrower options type with LLM-relevant options. | Status: not_done

### 2.4 Update Public API

- [ ] **Update `src/index.ts` for Phase 2** — Add `rerankWithLLM` to exports. Ensure types for LLM options are exported. | Status: not_done

### 2.5 Mode Dispatcher (`src/modes/index.ts`)

- [ ] **Implement mode dispatcher** — Given `RerankOptions`, select and invoke the appropriate scoring function (heuristic or llm-judge for Phase 2). Handle mode auto-detection (default to heuristic if `onnxruntime-node` is not available). | Status: not_done

### 2.6 Phase 2 Tests

- [ ] **Unit tests: pointwise prompt construction (`src/__tests__/prompts.test.ts`)** — Verify single-document prompt contains query and document text in expected format. Verify batch prompt lists multiple documents numbered 1-N. | Status: not_done
- [ ] **Unit tests: pointwise response parsing** — Verify "8" parses to 8. Verify "  7  " parses to 7. Verify "The score is 9" parses to 9 (first integer). Verify empty/non-numeric returns 0. | Status: not_done
- [ ] **Unit tests: pointwise batch response parsing** — Verify "1: 8\n2: 5\n3: 9" parses correctly. Verify partial response (only 2 of 3 scores) assigns 0 to missing. | Status: not_done
- [ ] **Unit tests: listwise prompt construction** — Verify prompt lists all documents and asks for ranking. | Status: not_done
- [ ] **Unit tests: listwise response parsing** — Verify "3, 1, 2" assigns correct scores. Verify missing/duplicate numbers are handled. | Status: not_done
- [ ] **Unit tests: LLM error handling (`src/__tests__/llm-judge.test.ts`)** — Verify `LLM_ERROR` when LLM function throws. Verify `LLM_TIMEOUT` when LLM function exceeds timeout. Verify unparseable response results in score 0, not a thrown error. Verify `MISSING_LLM_FUNCTION` when no `llm` function provided. | Status: not_done
- [ ] **Unit tests: document truncation in prompts** — Verify long documents are truncated to `maxDocTokensInPrompt` tokens before inclusion in the prompt. | Status: not_done
- [ ] **Integration test: LLM-as-judge end-to-end** — Use a mock LLM function returning predictable scores. Rerank 10 documents. Verify output ordering matches mock scores. Test both pointwise and listwise strategies. | Status: not_done
- [ ] **Integration test: LLM concurrency** — Verify that no more than `llmConcurrency` LLM calls run simultaneously. Use timing or call counting in the mock. | Status: not_done
- [ ] **Verify build passes with Phase 2 code** — Run `npm run build`. | Status: not_done
- [ ] **Verify all tests pass after Phase 2** — Run `npm run test`. | Status: not_done

---

## Phase 3: Cross-Encoder Mode (v0.3.0)

### 3.1 Model Registry (`src/models/registry.ts`)

- [ ] **Implement model registry** — Define the `MODEL_REGISTRY` constant mapping `CrossEncoderModel` names to `ModelInfo` objects. Include all three models: `ms-marco-MiniLM-L-6-v2`, `bge-reranker-base`, `bge-reranker-v2-m3`. Each entry has `onnxUrl`, `tokenizerUrl`, `maxSeqLength`, `numLabels`, `sizeBytes`, and `description`. | Status: not_done
- [ ] **Implement registry lookup function** — Given a model name, return the `ModelInfo` or throw `MODEL_NOT_FOUND` if not a built-in model and not a custom path. | Status: not_done

### 3.2 Model Download (`src/models/download.ts`)

- [ ] **Implement model download function** — Download an ONNX model file and tokenizer.json from a URL (HuggingFace Hub) using Node.js `https` module. No additional dependencies. | Status: not_done
- [ ] **Implement atomic file writing** — Write downloaded files with a `.tmp` suffix during download, then rename atomically on completion. Prevents partial/corrupt downloads from being used. | Status: not_done
- [ ] **Implement download progress logging** — Log download progress (file name, size, percentage) to stderr when the environment is a TTY. | Status: not_done
- [ ] **Implement download error handling** — Throw `RerankError` with code `MODEL_NOT_FOUND` on download failure (network error, 404, etc.). | Status: not_done

### 3.3 Model Cache (`src/models/cache.ts`)

- [ ] **Implement cache directory management** — Determine the cache directory (default: `~/.cache/rerank-lite/models/`). Create it if it does not exist. Support `cacheDir` option override. | Status: not_done
- [ ] **Implement cache lookup** — Check if a model's ONNX file and tokenizer.json exist in the cache directory under `<model-name>/`. Return paths if found. | Status: not_done
- [ ] **Implement cache integrity check** — Verify the cached file size matches the registry's `sizeBytes`. If mismatched (corrupt download), delete the cached file and trigger re-download. | Status: not_done
- [ ] **Implement cache path resolution** — Given a model name, return the expected file paths: `<cacheDir>/<model-name>/model.onnx` and `<cacheDir>/<model-name>/tokenizer.json`. | Status: not_done

### 3.4 Tokenizer

- [ ] **Implement WordPiece tokenizer (`src/tokenizer/wordpiece.ts`)** — Read a `tokenizer.json` file (HuggingFace Tokenizers format). Implement WordPiece tokenization: split on whitespace and punctuation, look up each subword in the vocabulary, handle unknown tokens (`[UNK]`). Produce `input_ids` array. | Status: not_done
- [ ] **Implement special token handling** — Handle `[CLS]`, `[SEP]`, `[PAD]` tokens. For a (query, document) pair, produce the sequence: `[CLS] query_tokens [SEP] doc_tokens [SEP]`. | Status: not_done
- [ ] **Implement `token_type_ids` generation** — Produce `token_type_ids` array: 0 for query tokens (including `[CLS]` and first `[SEP]`), 1 for document tokens (including second `[SEP]`). | Status: not_done
- [ ] **Implement `attention_mask` generation** — Produce `attention_mask` array: 1 for real tokens, 0 for padding tokens. | Status: not_done
- [ ] **Implement truncation logic** — When combined query + document exceeds `maxSeqLength`, truncate the document from the end. Budget: `maxDocTokens = maxSeqLen - queryTokens - 3`. As a last resort, truncate the query if it alone exceeds `maxSeqLen - 3`. | Status: not_done
- [ ] **Implement padding logic** — Pad all sequences in a batch to the length of the longest sequence. Pad with `[PAD]` token ID for `input_ids`, 0 for `attention_mask`, 0 for `token_type_ids`. | Status: not_done
- [ ] **Implement BPE tokenizer (`src/tokenizer/bpe.ts`)** — Fallback for models using BPE (RoBERTa, XLM-R). Read the BPE merges and vocabulary from `tokenizer.json`. Implement byte-level BPE tokenization. | Status: not_done
- [ ] **Implement tokenizer dispatcher (`src/tokenizer/index.ts`)** — Read the `tokenizer.json` model type field. Dispatch to WordPiece or BPE implementation based on model type. | Status: not_done

### 3.5 Model Loader (`src/models/loader.ts`)

- [ ] **Implement ONNX session creation** — Load the ONNX model file using `onnxruntime-node`'s `InferenceSession.create()`. Handle `onnxruntime-node` not being installed: throw `RerankError` with code `ONNX_NOT_AVAILABLE` with a clear message about installing it. | Status: not_done
- [ ] **Implement tokenizer loading** — Read and parse the `tokenizer.json` file from the cache. Initialize the appropriate tokenizer (WordPiece or BPE). | Status: not_done
- [ ] **Implement model input/output validation** — Validate that the loaded model accepts `input_ids`, `attention_mask`, and `token_type_ids` inputs. Validate it produces a logits output tensor. Throw `MODEL_LOAD_ERROR` on shape mismatches. | Status: not_done
- [ ] **Implement lazy loading pattern** — Model is not loaded at import time. It is loaded on the first `rerank()` call. The loaded session is cached and reused on subsequent calls. | Status: not_done

### 3.6 Cross-Encoder Mode (`src/modes/cross-encoder.ts`)

- [ ] **Implement cross-encoder scoring orchestrator** — Accept query, documents, and cross-encoder options. Ensure model is loaded (trigger download + load if needed). Tokenize all (query, document) pairs. Batch pairs according to `maxBatchSize`. Run inference. Extract scores. Return raw scores per document. | Status: not_done
- [ ] **Implement batched inference** — Group tokenized pairs into batches of `maxBatchSize` (default 32). For each batch, construct input tensors (`input_ids`, `attention_mask`, `token_type_ids`) with proper padding. Run the ONNX session. Extract logit scores. | Status: not_done
- [ ] **Implement score extraction** — For single-label models (`numLabels = 1`), extract the single logit as the relevance score. For two-label models, apply softmax and take the probability of the "relevant" class. | Status: not_done
- [ ] **Implement custom model support** — When `modelPath` and `tokenizerPath` are provided, load the model from those paths instead of the registry/cache. Skip download. | Status: not_done
- [ ] **Implement multi-batch concatenation** — When documents exceed `maxBatchSize`, process in multiple batches. Concatenate scores from all batches before normalization. | Status: not_done

### 3.7 Convenience Export (`rerankCrossEncoder`)

- [ ] **Implement `rerankCrossEncoder()` function** — Calls `rerank()` with `mode: 'cross-encoder'` forced. Accepts a narrower options type with cross-encoder-relevant options. | Status: not_done

### 3.8 Update Mode Dispatcher

- [ ] **Update mode dispatcher for cross-encoder** — Add cross-encoder mode to the dispatcher. Implement auto-detection: default to `'cross-encoder'` if `onnxruntime-node` is available, `'heuristic'` otherwise. | Status: not_done

### 3.9 Update Public API

- [ ] **Update `src/index.ts` for Phase 3** — Add `rerankCrossEncoder` to exports. Ensure cross-encoder related types are exported. | Status: not_done

### 3.10 Phase 3 Tests

- [ ] **Unit tests: tokenizer WordPiece (`src/__tests__/tokenizer.test.ts`)** — Verify tokenization of a (query, document) pair produces `[CLS] query_tokens [SEP] doc_tokens [SEP]`. Verify `token_type_ids` marks query as 0, document as 1. Verify `attention_mask` marks real tokens as 1, padding as 0. | Status: not_done
- [ ] **Unit tests: truncation** — Verify that when query + document exceeds max seq length, the document is truncated and the query is preserved in full. Verify the truncation budget formula. | Status: not_done
- [ ] **Unit tests: padding** — Verify that sequences in a batch are padded to uniform length. Verify padding values: `[PAD]` for input_ids, 0 for attention_mask, 0 for token_type_ids. | Status: not_done
- [ ] **Unit tests: BPE tokenizer** — Verify basic BPE tokenization produces expected token IDs for known inputs. | Status: not_done
- [ ] **Unit tests: model registry** — Verify all three models are in the registry. Verify lookup returns correct metadata. Verify unknown model name throws. | Status: not_done
- [ ] **Unit tests: model cache (`src/__tests__/model-management.test.ts`)** — Verify cache path resolution. Verify cache lookup returns files when present. Verify cache integrity check detects size mismatch. | Status: not_done
- [ ] **Unit tests: model download** — Verify atomic write (tmp file then rename). Verify download failure throws `MODEL_NOT_FOUND`. (Use mocked HTTP.) | Status: not_done
- [ ] **Unit tests: ONNX not available** — Mock `onnxruntime-node` as unavailable. Verify requesting cross-encoder mode throws `ONNX_NOT_AVAILABLE`. | Status: not_done
- [ ] **Unit tests: cross-encoder scoring (`src/__tests__/cross-encoder.test.ts`)** — Verify batched inference: scoring 5 docs in one batch produces the same scores as individually (within float tolerance). Verify score extraction for single-label and two-label models. Verify multi-batch concatenation. | Status: not_done
- [ ] **Integration test: cross-encoder end-to-end (slow)** — Load `ms-marco-MiniLM-L-6-v2`, score 10 documents with a known query. Verify top-3 results are semantically relevant. Tag as slow/integration test requiring model download. | Status: not_done
- [ ] **Verify build passes with Phase 3 code** — Run `npm run build`. | Status: not_done
- [ ] **Verify all tests pass after Phase 3** — Run `npm run test`. | Status: not_done

---

## Phase 4: Factory, CLI, and Polish (v0.4.0)

### 4.1 `createReranker` Factory (`src/rerank.ts`)

- [ ] **Implement `createReranker()` factory function** — Accept `RerankerConfig`. Validate configuration at construction time. Return a `Reranker` instance. The instance holds the ONNX session in memory across calls (for cross-encoder mode). | Status: not_done
- [ ] **Implement `Reranker.rerank()` method** — Delegates to the core `rerank()` function with the pre-configured options merged with per-call overrides. Per-call overrides take precedence. | Status: not_done
- [ ] **Implement `Reranker.warmup()` method** — Pre-loads the ONNX model into memory (cross-encoder mode only). No-op for other modes. Returns a promise that resolves when the model is ready. | Status: not_done
- [ ] **Implement `Reranker.dispose()` method** — Releases the ONNX session and frees memory. Subsequent `rerank()` calls trigger model reloading. No-op for non-cross-encoder modes. | Status: not_done

### 4.2 CLI (`src/cli.ts`)

- [ ] **Implement CLI argument parsing** — Parse all flags specified in Section 12: `--query`/`-q`, `--mode`/`-m`, `--model`, `--top-k`/`-k`, `--batch-size`/`-b`, `--cache-dir`, `--no-normalize`, `--ids-only`, `--scores-only`, `--pretty`/`-p`, `--heuristic-weights`. Use manual arg parsing or a minimal parser (no heavy CLI framework dependency). | Status: not_done
- [ ] **Implement stdin JSON reading** — Read JSON array of document objects from stdin. Parse and validate the input. Handle parse errors gracefully. | Status: not_done
- [ ] **Implement file path input** — When a positional argument is provided (not a flag), read the JSON document array from that file path. | Status: not_done
- [ ] **Implement JSON output** — Write the `RerankResult[]` array to stdout as JSON. Support `--pretty` for formatted output. | Status: not_done
- [ ] **Implement `--ids-only` output** — Write only document IDs, one per line. | Status: not_done
- [ ] **Implement `--scores-only` output** — Write IDs and scores, tab-separated, one per line. | Status: not_done
- [ ] **Implement `--heuristic-weights` parsing** — Parse comma-separated weight string "coverage,bm25,density,position" into the `heuristicWeights` object. | Status: not_done
- [ ] **Implement exit codes** — Exit 0 on success. Exit 1 on reranking failure. Exit 2 on configuration error (invalid flags, missing required options). | Status: not_done
- [ ] **Implement CLI error messages** — Write user-friendly error messages to stderr for missing `--query`, invalid JSON input, invalid mode, etc. | Status: not_done
- [ ] **Add `bin` field to `package.json`** — Add `"bin": { "rerank-lite": "dist/cli.js" }` to `package.json`. Ensure `cli.ts` has a proper hashbang (`#!/usr/bin/env node`). | Status: not_done

### 4.3 Update Public API

- [ ] **Update `src/index.ts` for Phase 4** — Add `createReranker` to exports. Ensure all public API exports from Section 9 are available. | Status: not_done

### 4.4 Phase 4 Tests

- [ ] **Integration test: createReranker round-trip (`src/__tests__/rerank.test.ts`)** — Create a reranker, rerank multiple independent queries. Verify each call is independent (no state leakage). | Status: not_done
- [ ] **Integration test: createReranker warmup** — Call `warmup()`, verify subsequent `rerank()` calls work correctly. Optionally measure that first-call latency is reduced. | Status: not_done
- [ ] **Integration test: createReranker dispose** — Call `dispose()`, verify subsequent `rerank()` calls still work (trigger reloading). | Status: not_done
- [ ] **Integration test: mode fallback** — With `onnxruntime-node` not available, verify default mode falls back to `'heuristic'` without throwing. | Status: not_done
- [ ] **CLI end-to-end test: heuristic mode (`src/__tests__/cli.test.ts`)** — Pipe JSON through the CLI binary with `--mode heuristic --query "test query"`. Verify stdout is valid JSON with reranked results. | Status: not_done
- [ ] **CLI end-to-end test: stdin input** — Pipe documents via stdin and verify reranking works. | Status: not_done
- [ ] **CLI end-to-end test: file input** — Pass a file path argument and verify reranking works. | Status: not_done
- [ ] **CLI end-to-end test: `--ids-only` output** — Verify output is IDs only, one per line. | Status: not_done
- [ ] **CLI end-to-end test: `--scores-only` output** — Verify output is IDs and scores, tab-separated. | Status: not_done
- [ ] **CLI end-to-end test: `--pretty` output** — Verify output is pretty-printed JSON. | Status: not_done
- [ ] **CLI end-to-end test: `--top-k` flag** — Verify output is limited to top-k results. | Status: not_done
- [ ] **CLI end-to-end test: `--no-normalize` flag** — Verify output scores are not normalized. | Status: not_done
- [ ] **CLI end-to-end test: `--heuristic-weights` flag** — Verify custom weights are applied. | Status: not_done
- [ ] **CLI end-to-end test: exit code 2 on missing `--query`** — Verify exit code is 2 when `--query` is omitted. | Status: not_done
- [ ] **CLI end-to-end test: exit code 2 on invalid JSON** — Verify exit code is 2 when stdin is not valid JSON. | Status: not_done
- [ ] **CLI end-to-end test: exit code 1 on reranking failure** — Verify exit code is 1 when reranking fails (e.g., requesting cross-encoder without ONNX). | Status: not_done
- [ ] **Property-based tests (`src/__tests__/properties.test.ts`)** — Using `fast-check`: verify score bounds [0, 1] when normalized. Verify ranks are 1-based, contiguous, match score ordering. Verify idempotency. Verify document preservation (no duplicates, no omissions). Verify topK enforcement. | Status: not_done
- [ ] **Add `fast-check` dev dependency** — Install `fast-check` for property-based testing. | Status: not_done
- [ ] **Verify build passes with Phase 4 code** — Run `npm run build`. | Status: not_done
- [ ] **Verify all tests pass after Phase 4** — Run `npm run test`. | Status: not_done

---

## Phase 5: Integration, Documentation, and Publishing (v0.5.0)

### 5.1 Performance Benchmarks

- [ ] **Create benchmark file (`src/__benchmarks__/rerank-throughput.ts`)** — Benchmark heuristic mode: 10, 50, 100, 500 documents. Benchmark LLM-as-judge mode with a mock LLM. Benchmark cross-encoder mode with `ms-marco-MiniLM-L-6-v2` (if model available): 10, 20, 50 documents. Print results as a table. | Status: not_done
- [ ] **Add benchmark script to `package.json`** — Add a `"bench"` script that runs the benchmark file (e.g., `ts-node src/__benchmarks__/rerank-throughput.ts` or `vitest bench`). | Status: not_done

### 5.2 README and Documentation

- [ ] **Write README.md** — Include: package description, installation instructions (including optional `onnxruntime-node`), quickstart examples for all three modes, `rerank()` API reference, `createReranker()` API reference, convenience function reference, CLI usage and all flags, configuration reference table (all options with types and defaults), integration examples (fusion-rank, context-packer, sparse-encode, embed-cache), mode comparison table, performance characteristics. | Status: not_done
- [ ] **Add JSDoc comments to all public exports** — Ensure `rerank()`, `createReranker()`, `rerankCrossEncoder()`, `rerankWithLLM()`, `rerankHeuristic()`, `RerankError`, and all types have JSDoc comments that appear in IDE tooltips. | Status: not_done

### 5.3 Package Configuration

- [ ] **Update `package.json` metadata** — Set `description`, `keywords` (rerank, reranker, cross-encoder, relevance, scoring, retrieval, RAG, BM25, LLM), `author`, `license`, `repository`, `homepage`. | Status: not_done
- [ ] **Add `bin` field to `package.json`** — Ensure `"bin": { "rerank-lite": "dist/cli.js" }` is present. | Status: not_done
- [ ] **Add `peerDependencies` and `peerDependenciesMeta`** — Add `onnxruntime-node` as an optional peer dependency. Use `peerDependenciesMeta` to mark it optional. | Status: not_done
- [ ] **Verify `files` field** — Ensure `"files": ["dist"]` is correct and the published package includes all necessary build artifacts. | Status: not_done
- [ ] **Verify `engines` field** — Ensure `"engines": { "node": ">=18" }` is set. | Status: not_done

### 5.4 Version Bump

- [ ] **Bump version to 0.5.0** — Update `version` field in `package.json` to `0.5.0` (or appropriate version per semver). | Status: not_done

### 5.5 Final Verification

- [ ] **Run full test suite** — `npm run test` passes with all tests green. | Status: not_done
- [ ] **Run lint** — `npm run lint` passes with no errors. | Status: not_done
- [ ] **Run build** — `npm run build` succeeds. | Status: not_done
- [ ] **Verify CLI works** — Run `echo '[{"id":"d1","text":"hello world"}]' | node dist/cli.js --query "hello" --mode heuristic` and verify valid output. | Status: not_done
- [ ] **Verify published package contents** — Run `npm pack --dry-run` and verify the tarball includes only `dist/` contents and `package.json`. No source files, no test files, no SPEC.md. | Status: not_done
- [ ] **Test npm install from tarball** — Run `npm pack`, then install the tarball in a separate directory. Verify `require('rerank-lite')` works and exports all expected functions. | Status: not_done
