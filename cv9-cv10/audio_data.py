"""Local LibriSpeech preparation, resumable Whisper inference, and ASR metrics."""

import importlib.metadata
import json
import math
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cv7-cv8"))
from rag_data import file_hash, fingerprint, read_jsonl, write_json, write_jsonl

DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODEL = ROOT / "models/whisper-base.en"
PARQUET = DATA / "raw/clean/test/0000.parquet"
MODEL_REVISION = "911407f4214e0e1d82085af863093ec0b66f9cd6"
DATA_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
DEMO_SOURCE = "6930-75918-0001"
DEMO_ANSWERS = ["Baskets of flowers", "Flowers"]
ASR_OPTIONS = {"device": "cpu", "dtype": "float32", "do_sample": False,
               "num_beams": 1, "long_form": "timestamps", "sampling_rate": 16000}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "soundfile", "jiwer", "numpy", "scipy", "pyarrow", "huggingface-hub")}


def normalize_asr(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())


def load_audio(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    return np.ascontiguousarray(audio)


def question_path(explicit=None):
    if explicit:
        path = Path(explicit).resolve()
    else:
        path = ROOT / "input/question.wav"
        if not path.exists():
            candidates = sorted((ROOT / "input").glob("*.wav"))
            if len(candidates) != 1:
                raise ValueError("Provide input/question.wav or select a recording with --audio.")
            path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def model_signature(threads=6, batch_size=4):
    files = {path.name: file_hash(path) for path in sorted(MODEL.iterdir())
             if path.is_file() and path.suffix in {".json", ".txt", ".safetensors"}}
    if "model.safetensors" not in files:
        raise ValueError("Local Whisper weights are missing. Follow README setup instructions.")
    return {"model": "openai/whisper-base.en", "revision": MODEL_REVISION, "files": files,
            "options": ASR_OPTIONS, "threads": threads, "batch_size": batch_size, "versions": versions()}


def prepare():
    source_hash = file_hash(PARQUET)
    manifest_path = DATA / "prepared.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest["parquet_sha256"] != source_hash:
            raise ValueError("Dataset changed. Keep a separate workspace for a different dataset.")
    audio_dir = DATA / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    documents, identifiers = [], set()
    for batch in pq.ParquetFile(PARQUET).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            identifier = row["id"]
            if not re.fullmatch(r"\d+-\d+-\d+", identifier) or identifier in identifiers:
                raise ValueError(f"Invalid or duplicate recording ID: {identifier}")
            identifiers.add(identifier)
            audio = row["audio"]["bytes"]
            path = audio_dir / (identifier + ".flac")
            # Compare bytes so a damaged local extraction is repaired on preparation.
            if not path.exists() or path.read_bytes() != audio:
                path.write_bytes(audio)
            info = sf.info(path)
            documents.append({"doc_id": identifier, "title": "LibriSpeech " + identifier,
                              "url": path.resolve().as_uri(), "text": row["text"],
                              "audio_path": str(path.relative_to(ROOT)), "audio_sha256": file_hash(path),
                              "speaker_id": row["speaker_id"], "chapter_id": row["chapter_id"],
                              "duration_seconds": info.duration, "audio_start_seconds": 0.0,
                              "audio_end_seconds": info.duration})
    if len(documents) != 2620 or DEMO_SOURCE not in identifiers:
        raise ValueError("Expected all 2620 test-clean recordings, including the demonstration source.")
    write_jsonl(DATA / "reference_documents.jsonl", documents)
    manifest = {"dataset": "openslr/librispeech_asr", "config": "clean", "split": "test",
                "revision": DATA_REVISION, "parquet_sha256": source_hash, "recordings": len(documents),
                "audio_seconds": sum(row["duration_seconds"] for row in documents),
                "reference_documents_sha256": file_hash(DATA / "reference_documents.jsonl")}
    write_json(manifest_path, manifest)
    print(f"Prepared {len(documents)} separate audio documents ({manifest['audio_seconds']/3600:.2f} hours).")
    return documents


class WhisperASR:
    def __init__(self, threads=6):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        torch.set_num_threads(threads)
        self.torch = torch
        self.processor = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            MODEL, local_files_only=True, dtype=torch.float32).to("cpu").eval()

    def transcribe(self, audios):
        long_form = any(len(audio) > 30 * 16000 for audio in audios)
        inputs = self.processor(audios, sampling_rate=16000, return_tensors="pt",
                                return_attention_mask=True, truncation=False)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                         return_timestamps=long_form, return_dict_in_generate=True)
        tokens = output["sequences"]
        texts = self.processor.batch_decode(tokens, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
        if len(texts) != len(audios):
            raise ValueError("Whisper returned the wrong number of transcriptions.")
        eos = self.model.generation_config.eos_token_id
        sequences = ([segment["result"]["sequences"] for recording in output["segments"] for segment in recording]
                     if long_form else tokens)
        for sequence in sequences:
            if sequence.numel() >= self.model.config.max_target_positions and eos not in sequence.flatten().tolist():
                raise ValueError("Whisper reached the output limit without completing its transcription.")
        return [text.strip() for text in texts]


class TranscriptCache:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS transcripts (key TEXT PRIMARY KEY, payload TEXT)")

    def get(self, key):
        row = self.connection.execute("SELECT payload FROM transcripts WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, payload):
        self.connection.execute("INSERT OR REPLACE INTO transcripts VALUES (?,?)", (key, json.dumps(payload)))
        self.connection.commit()

    def close(self):
        self.connection.close()


def transcript_key(document, signature):
    return fingerprint([document["doc_id"], document["audio_sha256"], signature])


def transcribe(threads=6, batch_size=4, limit=None, retry_errors=False):
    documents = list(read_jsonl(DATA / "reference_documents.jsonl"))
    settings = model_signature(threads, batch_size)
    signature = fingerprint(settings)
    manifest_path = DATA / "asr_manifest.json"
    if manifest_path.exists() and load_json(manifest_path)["signature"] != signature:
        raise ValueError("ASR settings changed. Use the original threads/batch-size or a separate workspace.")
    write_json(manifest_path, {"signature": signature, **settings})
    cache = TranscriptCache(DATA / "transcripts.sqlite")
    selected = documents[:limit] if limit else documents
    for doc in selected:
        if file_hash(ROOT / doc["audio_path"]) != doc["audio_sha256"]:
            cache.close()
            raise ValueError(f"Extracted audio changed: {doc['audio_path']}; run prepare to repair it.")
    pending = [doc for doc in selected if (saved := cache.get(transcript_key(doc, signature))) is None
               or (retry_errors and saved["status"] == "error")]
    started = time.perf_counter()
    print(f"Whisper: {len(pending)} pending / {len(selected)} selected recordings; CPU threads={threads}.", flush=True)
    try:
        recognizer = WhisperASR(threads) if pending else None
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            before = time.perf_counter()
            try:
                audios = []
                for doc in batch:
                    path = ROOT / doc["audio_path"]
                    if file_hash(path) != doc["audio_sha256"]:
                        raise ValueError(f"Extracted audio changed: {path}; run prepare to repair it.")
                    audios.append(load_audio(path))
                texts = recognizer.transcribe(audios)
                responses = [{"status": "ok", "text": text} for text in texts]
            except (RuntimeError, ValueError, OSError) as error:
                # Isolate failures so a bad file cannot discard other recordings in its batch.
                responses = []
                for doc in batch:
                    try:
                        path = ROOT / doc["audio_path"]
                        if file_hash(path) != doc["audio_sha256"]:
                            raise ValueError("Extracted audio changed; run prepare to repair it.")
                        responses.append({"status": "ok", "text": recognizer.transcribe([load_audio(path)])[0]})
                    except (RuntimeError, ValueError, OSError) as individual:
                        responses.append({"status": "error", "error": str(individual), "batch_error": str(error)})
            seconds = (time.perf_counter() - before) / len(batch)
            for doc, response in zip(batch, responses):
                cache.put(transcript_key(doc, signature), {"doc_id": doc["doc_id"], **response,
                                                          "seconds": seconds})
            completed = start + len(batch)
            print(f"  {completed}/{len(pending)} newly processed; elapsed {time.perf_counter()-started:.1f}s; "
                  f"batch errors={sum(r['status']=='error' for r in responses)}", flush=True)
    finally:
        records = [saved for doc in documents if (saved := cache.get(transcript_key(doc, signature))) is not None]
        write_jsonl(DATA / "transcriptions.jsonl", records)
        cache.close()
    print(f"Saved {len(records)} records; errors={sum(r['status']=='error' for r in records)}.")
    return records


def paired_scores(reference, hypothesis):
    ref, hyp = normalize_asr(reference), normalize_asr(hypothesis)
    words = jiwer.process_words(ref, hyp)
    chars = jiwer.process_characters(ref, hyp)
    return {"normalized_reference": ref, "normalized_hypothesis": hyp,
            "substitutions": words.substitutions, "deletions": words.deletions, "insertions": words.insertions,
            "reference_words": len(ref.split()), "wer": words.wer,
            "char_substitutions": chars.substitutions, "char_deletions": chars.deletions,
            "char_insertions": chars.insertions, "reference_chars": len(ref), "cer": chars.cer,
            "raw_exact_match": reference == hypothesis, "normalized_exact_match": ref == hyp}


def evaluate_asr():
    from rag import save_csv
    RESULTS.mkdir(parents=True, exist_ok=True)
    documents = list(read_jsonl(DATA / "reference_documents.jsonl"))
    records = {row["doc_id"]: row for row in read_jsonl(DATA / "transcriptions.jsonl")}
    rows, successful = [], []
    score_keys = paired_scores("a", "a").keys()
    for doc in documents:
        record = records.get(doc["doc_id"], {"status": "pending"})
        scores = paired_scores(doc["text"], record["text"]) if record["status"] == "ok" else {}
        if scores:
            successful.append(scores)
        rows.append({"doc_id": doc["doc_id"], "audio_path": doc["audio_path"], "reference": doc["text"],
                     "hypothesis": record.get("text", ""), "status": record["status"],
                     "error": record.get("error", ""), "seconds": record.get("seconds"),
                     **{key: scores.get(key) for key in score_keys}})
    total_words = sum(row["reference_words"] for row in successful)
    total_chars = sum(row["reference_chars"] for row in successful)
    summary = {"total": len(documents), "successful": len(successful),
               "errors": sum(row["status"] == "error" for row in rows),
               "pending": sum(row["status"] == "pending" for row in rows),
               "coverage": len(successful) / len(documents),
               "wer": sum(row[k] for row in successful for k in ("substitutions", "deletions", "insertions")) / total_words if total_words else None,
               "cer": sum(row[k] for row in successful for k in ("char_substitutions", "char_deletions", "char_insertions")) / total_chars if total_chars else None,
               "raw_exact_match": sum(row["raw_exact_match"] for row in successful) / len(successful) if successful else None,
               "normalized_exact_match": sum(row["normalized_exact_match"] for row in successful) / len(successful) if successful else None,
               "reference_words": total_words, "reference_chars": total_chars,
               "normalization": "NFKC, casefold, punctuation to spaces, collapse whitespace; CER includes spaces",
               "metrics_population": "successfully transcribed recordings only"}
    save_csv(RESULTS / "transcriptions.csv", rows)
    write_json(RESULTS / "asr_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary
