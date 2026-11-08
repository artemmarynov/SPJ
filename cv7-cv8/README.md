# Local RAG for Slovak Wikipedia

Compares whole-document TF-IDF retrieval, retrieval-augmented generation, and an LLM without access to the collection. Uses Ollama locally; no cloud API or model training is required.

## Setup

Requires Python 3.14 and a running Ollama instance. Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r cv7-cv8\requirements.txt
ollama pull qwen3:8b
ollama pull embeddinggemma
```

`qwen3:8b` has 8.19B parameters, below the assignment's 10B limit. `embeddinggemma` generates 768-dimensional embeddings.

## Run

```powershell
.\.venv\Scripts\python.exe -X utf8 cv7-cv8\rag.py check
.\.venv\Scripts\python.exe -X utf8 cv7-cv8\rag.py download
.\.venv\Scripts\python.exe -X utf8 cv7-cv8\rag.py prepare
.\.venv\Scripts\python.exe -X utf8 cv7-cv8\rag.py index
.\.venv\Scripts\python.exe -X utf8 cv7-cv8\rag.py ask
.\.venv\Scripts\python.exe -X utf8 cv7-cv8\rag.py evaluate
```

`download` automatically fetches pinned revisions of [Slovak Wikipedia](https://huggingface.co/datasets/PiotrSty/skwiki-slovak) and [SkQuAD](https://huggingface.co/datasets/TUKE-DeutscheTelekom/skquad), including article attribution. The initial dataset download is approximately 300 MB. Subsequent processing is local.

`prepare` selects 1,000 questions by default. Run it before indexing: the collection must include their source articles. `evaluate` processes all prepared questions and saves to `cv7-cv8/results/`. Resume an interrupted indexing run with `index`; use `index --rebuild` only when replacing an incompatible older index. Cached embeddings are reused.

Useful options:

```powershell
.\.venv\Scripts\python.exe -X utf8 cv7-cv8\rag.py ask --question "Čo je fotosyntéza?"
.\.venv\Scripts\python.exe -X utf8 cv7-cv8\rag.py evaluate --limit 3
.\.venv\Scripts\python.exe -X utf8 cv7-cv8\rag.py evaluate --retry-errors
```

## Method

- **Collection:** 5,000 articles, including source articles for 1,000 sampled validation questions; seed 42. Questions are not selected by model performance.
- **Chunks:** up to 250 whitespace-separated words, with 50-word overlap. Documents remain separate. Chunks exceeding the embedding token limit are subdivided without dropping text; each retains source IDs, URLs, and offsets.
- **Retrieval:** top-5 chunks by cosine similarity of normalized EmbeddingGemma vectors. The baseline searches whole documents using TF-IDF.
- **Generation:** the same `qwen3:8b` for RAG and standalone answers; temperature 0, seed 42, thinking disabled, 8,192-token context, and 256 output tokens.
- **Grounding:** RAG must use retrieved context, cite chunk IDs, provide an evidence quote, or abstain. QA reference answers and original QA contexts are never passed to the generator.

Downloads, embedding calculations, and evaluation results are cached. Rerunning a completed evaluation reuses saved answers. Changed data or settings require rebuilding the index when incompatible and clearing old evaluation results. See `rag.py <command> --help` for options.

The downloaded SkQuAD validation split contains 9,583 questions; 9,468 match articles in the Wikipedia dataset. The default sample contains 1,000 questions: 800 with reference answers and 200 without answers in their original contexts.

## Results

Completed run: 1,000 questions, 5,000 articles, and 14,647 chunks; seed 42. Of 2,000 generation attempts, 1,996 succeeded. Four RAG outputs contained invalid JSON; these remain recorded as errors.

| Method | Source Recall@5 | MRR@5 | Reference coverage@5 | Exact Match | Token F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole-document TF-IDF | 0.7110 | 0.5969 | 0.7025 | — | — |
| RAG | 0.8360 | 0.7829 | 0.7800 | 0.1972 | 0.4926 |
| LLM only | — | — | — | 0.0050 | 0.1277 |

Recall@5 and MRR@5 use all 1,000 questions and measure retrieval of the original source article. Reference coverage measures whether any reference answer occurs in the top-5 retrieved texts after normalization, over the 800 questions with references. EM/F1 use successful generations: 796 for RAG and 800 for LLM only. On the same 796 questions, LLM-only EM/F1 are 0.0050/0.1276. A dash means the metric does not apply: document retrieval produces no short answer, and LLM only retrieves no sources.

RAG improved source Recall@5 by 12.5 percentage points over document TF-IDF and achieved higher answer EM/F1 than LLM only. EM/F1 measure lexical agreement, not full semantic correctness. Reference answers were found in the current source articles for 750 of the 800 answerable questions; source revisions can differ from the original QA context.

| Additional check | RAG | LLM only |
| --- | ---: | ---: |
| Generation/format errors | 4/1,000 | 0/1,000 |
| Abstentions on the 200 originally unanswerable questions | 97/200 | 54/200 |
| Valid citation IDs among non-abstained successful RAG outputs | 736/828 (88.89%) | — |
| Evidence quote found in a cited chunk, among the same outputs | 610/828 (73.67%) | — |
| Mean generation time per successful output | 2.201 s | 0.591 s |

SkQuAD unanswerability applies to its original context, not the entire collection, so abstentions are a diagnostic rather than an accuracy score. Citation checks verify IDs and normalized quote presence; semantic support was not established by these checks. Timing excludes retrieval and failed calls.

`evaluate` automatically saves results under `cv7-cv8/results/`:

- `summary.csv` and `summary.json`: metrics for all three methods.
- `answers.csv`: questions, reference answers, generated answers, and scores.
- `evaluation.jsonl`: per-question results, retrieved text, citations, prompts, and raw outputs.
- `manifest.json`: dataset, model, and evaluation settings.

Results are saved after each question. Rerunning the command resumes from saved answers and refreshes the summary files. To generate all answers again, clear the generated files in `results/` first. Citation and quote checks run automatically; they do not establish semantic correctness.

Datasets, indexes, model files, caches, and generated outputs stay local and are excluded from Git. The commands above reproduce them.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s cv7-cv8 -p "test_*.py" -v
```

The 13 offline tests cover chunk boundaries, source provenance, overflow handling, cache reuse, interrupted evaluation, prompts, retrieval metrics, answer scoring, and citations.
