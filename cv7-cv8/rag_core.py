"""Local Ollama calls, resumable embeddings, retrieval, and evaluation helpers."""

import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_data import DATA, ROOT, chunks_for_document, file_hash, fingerprint, normalize, read_jsonl, write_json, write_jsonl


GENERATOR = "qwen3:8b"
EMBEDDER = "embeddinggemma:latest"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 256}


class Ollama:
    def __init__(self, url="http://127.0.0.1:11434"):
        self.url = url.rstrip("/")

    def request(self, path, payload=None):
        request = urllib.request.Request(
            self.url + path, data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Ollama HTTP {error.code}: {error.read().decode('utf-8')[:1200]}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Cannot reach Ollama at {self.url}. Open the Ollama application. {error}") from error

    def models(self):
        tags = {row["name"]: row for row in self.request("/api/tags")["models"]}
        result = {}
        for name, capability in [(GENERATOR, "completion"), (EMBEDDER, "embedding")]:
            if name not in tags:
                raise ValueError(f"Missing model {name}. Run: ollama pull {name}")
            details = self.request("/api/show", {"model": name})
            if capability not in details.get("capabilities", []):
                raise ValueError(f"{name} does not support {capability}.")
            parameters = details.get("model_info", {}).get("general.parameter_count")
            if name == GENERATOR and (parameters is None or parameters >= 10_000_000_000):
                raise ValueError("The generator must have a verified parameter count below 10 billion.")
            result[name] = {"digest": tags[name]["digest"], "parameters": parameters,
                            "bytes": tags[name]["size"]}
        return result

    def embed(self, texts):
        result = self.request("/api/embed", {"model": EMBEDDER, "input": texts,
                                             "truncate": False, "keep_alive": "30m"})
        vectors = np.asarray(result["embeddings"], dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("Ollama returned an invalid embedding matrix.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Ollama returned a zero embedding.")
        return vectors / norms


def simple_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


def ranked_indices(scores, k):
    if k < 1:
        raise ValueError("top-k must be positive.")
    return np.argsort(-np.asarray(scores), kind="stable")[:k]


def split_overlong_chunk(chunk, overlap):
    """Subdivide an overflowing chunk without losing words or source offsets."""
    words = list(re.finditer(r"\S+", chunk["text"]))
    if len(words) < 2:
        raise ValueError(f"A single word exceeds the embedding context: {chunk['chunk_id']}")
    middle = len(words) // 2
    margin = min(overlap // 2, len(words) // 4)
    children = []
    for start, end in [(0, middle + margin), (middle - margin, len(words))]:
        left, right = words[start].start(), words[end - 1].end()
        word_start, word_end = chunk["word_start"] + start, chunk["word_start"] + end
        children.append({**chunk, "chunk_id": f"{chunk['doc_id']}:{word_start:06d}-{word_end:06d}",
                         "word_start": word_start, "word_end": word_end,
                         "char_start": chunk["char_start"] + left,
                         "char_end": chunk["char_start"] + right, "text": chunk["text"][left:right]})
    return children


class EmbeddingCache:
    """Cache successful vectors and explicit subdivisions in a local SQLite file."""

    def __init__(self, client, path, digest, overlap):
        self.client, self.digest, self.overlap = client, digest, overlap
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS splits (key TEXT PRIMARY KEY, children TEXT)")
        self.splits = 0

    def embed(self, chunks):
        results, missing = {}, []
        for index, chunk in enumerate(chunks):
            key = fingerprint([self.digest, chunk])
            cached = self.connection.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
            split = self.connection.execute("SELECT children FROM splits WHERE key=?", (key,)).fetchone()
            if cached:
                results[index] = [(chunk, np.frombuffer(cached[0], dtype=np.float32))]
            elif split:
                results[index] = self.embed(json.loads(split[0]))
            else:
                missing.append((index, key, chunk))
        if missing:
            try:
                vectors = self.client.embed([f"title: {chunk['title']} | text: {chunk['text']}"
                                             for _, _, chunk in missing])
            except RuntimeError as error:
                if "context length" not in str(error).lower():
                    raise
                if len(missing) > 1:
                    for index, _, chunk in missing:
                        results[index] = self.embed([chunk])
                else:
                    index, key, chunk = missing[0]
                    children = split_overlong_chunk(chunk, self.overlap)
                    self.connection.execute("INSERT OR REPLACE INTO splits VALUES (?,?)",
                                            (key, json.dumps(children, ensure_ascii=False)))
                    self.connection.commit()
                    self.splits += 1
                    print(f"  context overflow: splitting {chunk['chunk_id']} into smaller chunks", flush=True)
                    results[index] = self.embed(children)
            else:
                for (index, key, chunk), vector in zip(missing, vectors):
                    self.connection.execute("INSERT OR REPLACE INTO vectors VALUES (?,?)", (key, vector.tobytes()))
                    results[index] = [(chunk, vector)]
                self.connection.commit()
        return [item for index in range(len(chunks)) for item in results[index]]


def build_index(client, directory, size=250, overlap=50, batch_size=32, rebuild=False):
    directory = Path(directory)
    if batch_size < 1:
        raise ValueError("batch-size must be positive.")
    models = client.models()
    source = DATA / "documents.jsonl"
    if not source.exists():
        raise ValueError("Prepared documents are missing. Run: python rag.py prepare")
    documents = list(read_jsonl(source))
    configuration = {"documents_sha256": file_hash(source), "chunk_size_words": size,
                     "overlap_words": overlap, "embedding_model": EMBEDDER,
                     "embedding_digest": models[EMBEDDER]["digest"], "prompt_version": 1,
                     "adaptive_overflow_policy": 1}
    signature = fingerprint(configuration)
    state_path = directory / "state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    if previous and previous["signature"] != signature and not rebuild:
        raise ValueError("Index settings changed. Use a different --index-dir or index --rebuild.")
    directory.mkdir(parents=True, exist_ok=True)
    if previous and previous["complete"] and not rebuild:
        print(f"Compatible index already complete: {previous['total']} chunks. No embeddings recomputed.")
        return
    chunks = [chunk for document in documents for chunk in chunks_for_document(document, size, overlap)]
    if not chunks:
        raise ValueError("The document collection produced no chunks.")
    state = {"signature": signature, "configuration": configuration, "original_chunks": len(chunks),
             "total": len(chunks), "completed": 0, "dimensions": None, "complete": False}
    write_json(state_path, state)
    print(f"Original chunks: {len(chunks)}; maximum words/chunk: {size}; overlap: {overlap}", flush=True)
    cache = EmbeddingCache(client, directory / "embedding_cache.sqlite", models[EMBEDDER]["digest"], overlap)
    final_chunks, final_vectors, seen = [], [], set()
    started = time.perf_counter()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        for chunk, vector in cache.embed(batch):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                final_chunks.append(chunk)
                final_vectors.append(vector)
        state["completed"] = start + len(batch)
        write_json(state_path, state)
        if start == 0 or state["completed"] % (batch_size * 20) == 0 or state["completed"] == len(chunks):
            elapsed = time.perf_counter() - started
            print(f"  embedded {state['completed']}/{len(chunks)}; elapsed {elapsed:.1f}s", flush=True)

    cache.connection.close()
    vectors = np.stack(final_vectors)
    np.save(directory / "vectors.npy", vectors)
    write_jsonl(directory / "chunks.jsonl", final_chunks)
    state["dimensions"] = vectors.shape[1]
    state["total"] = len(final_chunks)
    state["adaptive_split_events_this_run"] = cache.splits

    baseline_path = directory / "document_tfidf.joblib"
    if not state["complete"] or not baseline_path.exists():
        print("Building original whole-document TF-IDF baseline...", flush=True)
        vectorizer = TfidfVectorizer(tokenizer=simple_tokenize, token_pattern=None,
                                     lowercase=True, smooth_idf=True, norm="l2")
        matrix = vectorizer.fit_transform([row["text"] for row in documents])
        joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, baseline_path)
    state["complete"] = True
    write_json(state_path, state)
    print(f"Index ready: {directory}", flush=True)


class KnowledgeBase:
    def __init__(self, client, directory):
        self.client = client
        self.directory = Path(directory)
        self.state = json.loads((self.directory / "state.json").read_text(encoding="utf-8"))
        if not self.state["complete"]:
            raise ValueError("The index is incomplete. Resume: python rag.py index")
        if file_hash(DATA / "documents.jsonl") != self.state["configuration"]["documents_sha256"]:
            raise ValueError("The prepared corpus changed. Rebuild the index.")
        self.models = client.models()
        if self.models[EMBEDDER]["digest"] != self.state["configuration"]["embedding_digest"]:
            raise ValueError("The embedding model changed. Rebuild the index.")
        self.documents = list(read_jsonl(DATA / "documents.jsonl"))
        self.chunks = list(read_jsonl(self.directory / "chunks.jsonl"))
        self.vectors = np.load(self.directory / "vectors.npy", mmap_mode="r")
        self.baseline = joblib.load(self.directory / "document_tfidf.joblib")
        self.query_cache_path = self.directory / "query_cache.jsonl"
        self.query_vectors = {}
        if self.query_cache_path.exists():
            self.query_vectors = {row["key"]: row["vector"] for row in read_jsonl(self.query_cache_path)}

    def query_key(self, question):
        return fingerprint([self.models[EMBEDDER]["digest"], "search result", question])

    def cache_queries(self, questions, batch_size=32):
        missing = list(dict.fromkeys(question for question in questions if self.query_key(question) not in self.query_vectors))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            vectors = self.client.embed(["task: search result | query: " + question for question in batch])
            for question, vector in zip(batch, vectors):
                self.query_vectors[self.query_key(question)] = vector.tolist()
            write_jsonl(self.query_cache_path, ({"key": key, "vector": vector}
                                              for key, vector in self.query_vectors.items()))

    def retrieve(self, question, k=5):
        self.cache_queries([question])
        vector = np.asarray(self.query_vectors[self.query_key(question)], dtype=np.float32)
        scores = self.vectors @ vector
        chunks = [{**self.chunks[index], "score": float(scores[index])} for index in ranked_indices(scores, k)]
        query = self.baseline["vectorizer"].transform([question])
        document_scores = (self.baseline["matrix"] @ query.T).toarray().ravel()
        documents = [{**self.documents[index], "score": float(document_scores[index])}
                     for index in ranked_indices(document_scores, k)]
        return chunks, documents


def answer_scores(answer, references):
    if not references:
        return None
    predicted = normalize(answer)
    tokens = Counter(predicted.split())
    exact, f1 = 0.0, 0.0
    for reference in references:
        expected = normalize(reference)
        exact = max(exact, float(predicted == expected))
        gold = Counter(expected.split())
        common = sum((tokens & gold).values())
        if common:
            precision = common / sum(tokens.values())
            recall = common / sum(gold.values())
            f1 = max(f1, 2 * precision * recall / (precision + recall))
    return {"exact_match": exact, "token_f1": f1}


def retrieval_scores(rows, source_id, references):
    ranks = [index + 1 for index, row in enumerate(rows) if row["doc_id"] == source_id]
    coverage = None if not references else any(
        normalize(answer) in normalize(row["text"]) for answer in references for row in rows)
    return {"source_recall_at_k": float(bool(ranks)), "reciprocal_rank_at_k": 1 / ranks[0] if ranks else 0.0,
            "reference_in_retrieved_text": coverage}


def grounding_checks(answer, chunks):
    by_id = {row["chunk_id"]: row for row in chunks}
    citations = answer["citations"]
    valid = bool(citations) and all(item in by_id for item in citations)
    evidence = normalize(answer["evidence"])
    quote_present = bool(evidence) and valid and any(evidence in normalize(by_id[item]["text"]) for item in citations)
    return {"citation_ids_valid": valid, "evidence_in_cited_chunk": bool(quote_present),
            "requires_semantic_review": not answer["abstained"]}
