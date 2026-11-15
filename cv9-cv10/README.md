# Audio RAG with Whisper

Transcribes LibriSpeech test-clean and uses the recordings as a new collection for the local RAG from `cv7-cv8`. A spoken question is transcribed by the same Whisper model, then answered using retrieved audio transcripts. No model training or cloud API is required.

## Setup

Run from the repository root with Python 3.14 and Ollama running. The existing `qwen3:8b` and `embeddinggemma` models are reused.

```powershell
.\.venv\Scripts\python.exe -m pip install -r cv9-cv10\requirements.txt
.\.venv\Scripts\python.exe -X utf8 -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-base.en', revision='911407f4214e0e1d82085af863093ec0b66f9cd6', local_dir='cv9-cv10/models/whisper-base.en', allow_patterns=['*.json','*.txt','model.safetensors','README.md'])"
.\.venv\Scripts\python.exe -X utf8 -c "from huggingface_hub import hf_hub_download; [hf_hub_download(repo_id='openslr/librispeech_asr', repo_type='dataset', revision='71cacbfb7e2354c4226d01e70d77d5fca3d04ba1', filename=name, local_dir='cv9-cv10/data/raw') for name in ['clean/test/0000.parquet','README.md']]"
```

Downloads: [Whisper base.en](https://huggingface.co/openai/whisper-base.en) (~290 MB of weights) and [LibriSpeech clean/test](https://huggingface.co/datasets/openslr/librispeech_asr/tree/main/clean/test) (~350 MB). Training data is not needed. Whisper loads only from its local folder; it is separate from Ollama.

Record yourself asking:

> What did the English send to the French to greet the young princess?

Save a WAV recording in `input/question.wav` and its spoken text in `input/question.txt`. If `question.wav` is absent, the program accepts a single WAV in `input/` (the supplied recording is `output.wav`). Use `--audio path/to/recording.wav` to choose a file explicitly. Mono/stereo WAV at other sample rates is converted to mono 16 kHz automatically.

## Run

```powershell
.\.venv\Scripts\python.exe -X utf8 cv9-cv10\audio_rag.py check
.\.venv\Scripts\python.exe -X utf8 cv9-cv10\audio_rag.py prepare
.\.venv\Scripts\python.exe -X utf8 cv9-cv10\audio_rag.py transcribe
.\.venv\Scripts\python.exe -X utf8 cv9-cv10\audio_rag.py index
.\.venv\Scripts\python.exe -X utf8 cv9-cv10\audio_rag.py ask
.\.venv\Scripts\python.exe -X utf8 cv9-cv10\audio_rag.py evaluate
```

For a smoke run, use `transcribe --limit 3`, then `transcribe` to finish the collection. Successful transcriptions are committed to SQLite after each batch and reused on restart. `transcribe --retry-errors` retries failed recordings. Keep the default CPU thread count (6) and batch size (4) when resuming an experiment; model/settings changes are rejected. `index` requires all recordings to be successfully processed; `index --rebuild` replaces incompatible indexes while reusing embedding caches. `ask` and `evaluate` cache generation results; `--retry-errors` retries saved generation errors.

## Method and evaluation

- **Audio:** all 2,620 test-clean recordings (5.40 hours); local Whisper base.en, CPU float32, greedy decoding. Audio longer than 30 seconds uses Whisper long-form generation with timestamps and no input truncation. References are never passed to Whisper.
- **Collections:** each recording is a separate document with its recording ID, speaker/chapter IDs, local audio path, file hash, and duration. One index uses reference texts and one uses Whisper texts. Chunks have up to 250 whitespace-separated words and 50-word overlap; short recordings usually produce one chunk. Character offsets identify transcript spans; the displayed audio interval identifies the recording, not word-level alignment.
- **RAG:** the shared EmbeddingGemma implementation uses normalized vectors, cosine similarity, and top-5 search. Qwen3:8b answers in English with `temperature=0`, `seed=42`, `think=false`, an 8,192-token context and a 256-token output budget. It must cite supplied chunks and provide supporting evidence or abstain.
- **ASR metrics:** corpus WER/CER aggregate edit counts, not per-record error averages. Normalization is NFKC, casefold, punctuation replaced with spaces, and collapsed whitespace; CER includes spaces. Raw and normalized exact-match rates are both saved. Metrics cover successful recordings, with errors and pending recordings reported explicitly. The voice question is scored separately.
- **RAG comparison:** the same typed/recognized question is tested against both reference/Whisper collections. Accepted answer phrases are “Baskets of flowers” and “Flowers,” from recording `6930-75918-0001`. Answers are compared using EM/F1 and source retrieval checks. References are used for evaluation only. Identical prompts reuse cached responses.

The four cases isolate the effects of corpus and question transcription. This is a one-question demonstration, not a general RAG benchmark. The former Wikipedia scores are not comparable because its collection and questions differ. Citation/quote checks alone do not establish semantic support.

## Recorded results

Full test-clean run (2026-11-14): 2,620/2,620 successful transcriptions, no processing errors, and 2,620 chunks in each index. All nine recordings longer than 30 seconds were processed using long-form transcription.

| ASR metric | Result |
| --- | ---: |
| Corpus WER, normalized | 4.7609% |
| Corpus CER, normalized | 1.9401% |
| Exact match, raw | 0/2,620 (0.00%) |
| Exact match, normalized | 1,442/2,620 (55.04%) |
| Recorded question WER / CER | 0.00% / 0.00% |

Raw exact match is zero because the dataset uses uppercase text without normal sentence punctuation, while Whisper produces sentence case and punctuation. The normalized score removes these formatting differences using the policy above; it does not normalize number spellings or synonyms.

| Collection | Question | Source rank | Answer | EM | Token F1 |
| --- | --- | ---: | --- | ---: | ---: |
| Reference | Typed | 1 | flowers | 1.0 | 1.0 |
| Reference | Voice transcription | 1 | flowers | 1.0 | 1.0 |
| Whisper | Typed | 1 | baskets of flowers | 1.0 | 1.0 |
| Whisper | Voice transcription | 1 | baskets of flowers | 1.0 | 1.0 |

All four cases cite recording `6930-75918-0001` through its full chunk ID; citation IDs and evidence quotes pass the automatic checks. Inspection of the cited text confirms that the English sent baskets of flowers to greet the princess, supporting both accepted answers. The question was transcribed exactly, so typed/voice pairs have identical prompts and reuse the same response.

No answer deterioration was observed for this example: Whisper preserved the relevant fact and the question. Other retrieved results changed, and the source cosine score changed from 0.6474 to 0.6896; a larger similarity score does not establish better answer accuracy. These four cases do not demonstrate an improvement across the whole collection.

## Outputs

Files under `results/` include `transcriptions.csv`, `asr_summary.json`, `question.json`, `rag_comparison.csv`, `rag_comparison.json`, and `manifest.json`. `ask` additionally saves `last_question.json`. The manifest records input hashes, dependency versions, dataset/model revisions, index settings, and actual Ollama model digests. Downloads, extracted audio, weights, caches, indexes, results, and voice recordings are excluded from Git; include the recording and generated results separately when submitting the exercise.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s cv9-cv10 -p "test_*.py" -v
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s cv7-cv8 -p "test_*.py" -v
```

Offline tests cover resampling, long-form settings, provenance, weighted metrics, missing/error records, cache continuation, changed inputs/settings, English prompts, and shared retrieval compatibility.
