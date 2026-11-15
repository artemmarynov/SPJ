"""Whisper transcription and a local audio RAG collection, using cv7-cv8 retrieval."""

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from audio_data import (
    DATA, DATA_REVISION, DEMO_ANSWERS, DEMO_SOURCE, MODEL, MODEL_REVISION,
    PARQUET, RESULTS, ROOT, TranscriptCache, WhisperASR, evaluate_asr, file_hash, fingerprint,
    load_audio, load_json, model_signature, normalize_asr, paired_scores, prepare,
    question_path, read_jsonl, transcribe, versions, write_json, write_jsonl,
)
from rag import generate, make_messages, save_csv
from rag_core import OPTIONS, KnowledgeBase, Ollama, answer_scores, build_index, grounding_checks, retrieval_scores

DEMO_QUESTION = "What did the English send to the French to greet the young princess?"


def check():
    required = [PARQUET, MODEL / "model.safetensors", MODEL / "config.json",
                MODEL / "preprocessor_config.json", MODEL / "tokenizer.json", ROOT / "input/question.txt"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Follow README setup instructions.")
    if pq.ParquetFile(PARQUET).metadata.num_rows != 2620:
        raise ValueError("Expected 2620 LibriSpeech test-clean rows.")
    audio = question_path()
    samples = load_audio(audio)
    print(f"Question recording: {audio}\nDuration: {len(samples)/16000:.3f}s; converted to mono 16000 Hz")
    print("Dependencies:", json.dumps(versions(), indent=2))
    print("Ollama models:", json.dumps(Ollama().models(), indent=2))
    print("All local prerequisites are available.")


def collection_documents(name):
    return DATA / ("reference_documents.jsonl" if name == "reference" else "whisper_documents.jsonl")


def publish_whisper_documents():
    documents = list(read_jsonl(collection_documents("reference")))
    records = {row["doc_id"]: row for row in read_jsonl(DATA / "transcriptions.jsonl")}
    missing = [doc["doc_id"] for doc in documents
               if doc["doc_id"] not in records or records[doc["doc_id"]]["status"] != "ok"]
    if missing:
        raise ValueError(f"Whisper collection is incomplete: {len(missing)} missing/failed recordings. "
                         "Run transcribe (or transcribe --retry-errors) before index.")
    write_jsonl(collection_documents("whisper"), ({**doc, "text": records[doc["doc_id"]]["text"]}
                                                  for doc in documents))


def index_collections(client, rebuild=False, size=250, overlap=50):
    publish_whisper_documents()
    for name in ("reference", "whisper"):
        build_index(client, ROOT / ("index-" + name), size=size, overlap=overlap,
                    rebuild=rebuild, documents_path=collection_documents(name))


def transcribe_question(audio=None, threads=6, batch_size=4):
    path = question_path(audio)
    settings = model_signature(threads, batch_size)
    signature = fingerprint(settings)
    audio_hash = file_hash(path)
    key = fingerprint(["question", audio_hash, signature])
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache = TranscriptCache(DATA / "transcripts.sqlite")
    try:
        saved = cache.get(key)
        if saved is None:
            started = time.perf_counter()
            text = WhisperASR(threads).transcribe([load_audio(path)])[0]
            saved = {"text": text, "seconds": time.perf_counter() - started}
            cache.put(key, saved)
    finally:
        cache.close()
    if not normalize_asr(saved["text"]):
        raise ValueError("The voice question has an empty transcription.")
    reference_path = ROOT / "input/question.txt"
    reference = reference_path.read_text(encoding="utf-8-sig").strip()
    question = {"audio_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "audio_sha256": audio_hash, "asr_signature": signature,
                "reference_text": reference, "transcription": saved["text"],
                "seconds": saved["seconds"], "scores": paired_scores(reference, saved["text"])}
    write_json(RESULTS / "question.json", question)
    print(f"VOICE QUESTION: {saved['text']}", flush=True)
    return question


def answer_case(client, kb, query, collection, query_type, retry_errors=False):
    chunks, _ = kb.retrieve(query, 5)
    settings = {"index_signature": kb.state["signature"], "models": kb.models,
                "options": OPTIONS, "think": False, "language": "en", "top_k": 5,
                "prompt_template_sha256": fingerprint(make_messages("", [], language="en")),
                "output_schema_version": 2}
    # The response cache is keyed by the actual prompt; answer references are not inputs.
    key = fingerprint([settings, make_messages(query, chunks, language="en")])
    cache_path = RESULTS / "rag_cache.jsonl"
    cached = {row["key"]: row for row in read_jsonl(cache_path)} if cache_path.exists() else {}
    previous = cached.get(key)
    if previous and (previous["response"]["status"] == "ok" or not retry_errors):
        response = previous["response"]
    else:
        response = generate(client, query, chunks, language="en")
        cached[key] = {"key": key, "response": response}
        write_jsonl(cache_path, cached.values())
    if response["status"] == "ok":
        response = {**response, "grounding": grounding_checks(response, chunks)}
    return {"collection": collection, "query_type": query_type, "question": query,
            "settings": settings, "chunks": chunks, "response": response}


def show_case(case):
    print(f"\n{case['collection'].upper()} collection / {case['query_type']} question")
    print("Question:", case["question"])
    for chunk in case["chunks"]:
        print(f"\n{chunk['score']:.4f} | {chunk['chunk_id']} | {chunk['audio_path']} "
              f"| recording interval 0–{chunk['duration_seconds']:.2f}s\n{chunk['text']}")
    response = case["response"]
    print("\nANSWER:", response.get("answer", "ERROR: " + response.get("error", "unknown")))
    print("Citations:", response.get("citations", []))
    print("Evidence:", response.get("evidence", ""))
    print("Automatic checks:", response.get("grounding", {}))


def manifest(question, client):
    return {"prepared": load_json(DATA / "prepared.json"), "asr": load_json(DATA / "asr_manifest.json"),
            "models": client.models(), "question_audio_sha256": question["audio_sha256"],
            "question_reference_sha256": file_hash(ROOT / "input/question.txt"),
            "question_asr_signature": question["asr_signature"],
            "index_signatures": {name: load_json(ROOT / ("index-" + name) / "state.json")["signature"]
                                 for name in ("reference", "whisper")},
            "options": OPTIONS, "think": False, "language": "en", "top_k": 5,
            "prompt_template_sha256": fingerprint(make_messages("", [], language="en")),
            "output_schema_version": 2,
            "dataset_revision": DATA_REVISION, "model_revision": MODEL_REVISION,
            "source_doc_id": DEMO_SOURCE, "expected_answers": DEMO_ANSWERS,
            "scope": "ASR on all test-clean recordings; RAG comparison on one predefined voice question"}


def evaluate(client, audio=None, threads=6, batch_size=4, retry_errors=False):
    evaluate_asr()
    question = transcribe_question(audio, threads, batch_size)
    if normalize_asr(question["reference_text"]) != normalize_asr(DEMO_QUESTION):
        raise ValueError("The predefined RAG comparison expects the demonstration question in input/question.txt.")
    cases = []
    for name in ("reference", "whisper"):
        kb = KnowledgeBase(client, ROOT / ("index-" + name), documents_path=collection_documents(name))
        for query_type, query in [("typed", question["reference_text"]), ("voice", question["transcription"])]:
            case = answer_case(client, kb, query, name, query_type, retry_errors)
            case["retrieval_scores"] = retrieval_scores(case["chunks"], DEMO_SOURCE, DEMO_ANSWERS)
            if case["response"]["status"] == "ok":
                case["answer_scores"] = answer_scores(case["response"]["answer"], DEMO_ANSWERS)
            cases.append(case)
            # Preserve each finished case, including generation errors.
            write_json(RESULTS / "rag_comparison.json", {"question": question, "cases": cases})
            show_case(case)
    flat = []
    for case in cases:
        response = case["response"]
        flat.append({"collection": case["collection"], "query_type": case["query_type"],
                     "question": case["question"], "status": response["status"],
                     "answer": response.get("answer", ""), "error": response.get("error", ""),
                     "expected_answer": DEMO_ANSWERS[0], **case["retrieval_scores"],
                     "exact_match": case.get("answer_scores", {}).get("exact_match"),
                     "token_f1": case.get("answer_scores", {}).get("token_f1"),
                     "citations": " | ".join(response.get("citations", [])),
                     "evidence": response.get("evidence", ""),
                     "citation_ids_valid": response.get("grounding", {}).get("citation_ids_valid"),
                     "evidence_in_cited_chunk": response.get("grounding", {}).get("evidence_in_cited_chunk")})
    save_csv(RESULTS / "rag_comparison.csv", flat)
    write_json(RESULTS / "manifest.json", manifest(question, client))
    print("\nSaved ASR metrics and four RAG cases to", RESULTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("prepare")
    for name in ("transcribe", "ask", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--threads", type=int, default=6)
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--retry-errors", action="store_true")
        if name == "transcribe":
            command.add_argument("--limit", type=int)
        else:
            command.add_argument("--audio", type=Path)
    command = commands.add_parser("index")
    command.add_argument("--rebuild", action="store_true")
    command.add_argument("--chunk-size", type=int, default=250)
    command.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()
    for name in ("threads", "batch_size", "limit"):
        if hasattr(args, name) and getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(name + " must be positive.")
    with threadpool_limits(limits=2):
        if args.command == "check":
            check()
        elif args.command == "prepare":
            prepare()
        elif args.command == "transcribe":
            transcribe(args.threads, args.batch_size, args.limit, args.retry_errors)
        elif args.command == "index":
            index_collections(Ollama(), args.rebuild, args.chunk_size, args.overlap)
        elif args.command == "evaluate":
            evaluate(Ollama(), args.audio, args.threads, args.batch_size, args.retry_errors)
        else:
            question = transcribe_question(args.audio, args.threads, args.batch_size)
            client = Ollama()
            kb = KnowledgeBase(client, ROOT / "index-whisper", documents_path=collection_documents("whisper"))
            case = answer_case(client, kb, question["transcription"], "whisper", "voice", args.retry_errors)
            show_case(case)
            write_json(RESULTS / "last_question.json", {"voice_question": question, **case})


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error
