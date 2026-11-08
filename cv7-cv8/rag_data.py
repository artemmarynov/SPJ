"""Pinned dataset downloads, deterministic preparation, and document-local chunks."""

import gzip
import hashlib
import html
import json
import random
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WIKI_REPO = "PiotrSty/skwiki-slovak"
WIKI_REV = "3eea6cba070a756e800b32bb5ed86da820b6f1cf"
QA_REPO = "TUKE-DeutscheTelekom/skquad"
QA_REV = "105ae10c96a7068e2e190c9a2d14458bc830dfd1"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(text):
    # Keep Slovak letters and accents; punctuation becomes word separators.
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(character if character.isalnum() else " " for character in text).split())


def clean_text(text):
    text = html.unescape(re.sub(r"</?[A-Za-z][^>]*>", " ", text))
    return " ".join(unicodedata.normalize("NFKC", text).split())


def download():
    from huggingface_hub import hf_hub_download

    files = [
        (WIKI_REPO, WIKI_REV, "data/train-00000-of-00001.parquet", "wiki"),
        (WIKI_REPO, WIKI_REV, "artifacts/attribution.jsonl", "wiki"),
        (WIKI_REPO, WIKI_REV, "README.md", "wiki"),
        (QA_REPO, QA_REV, "validation.json.gz", "skquad"),
        (QA_REPO, QA_REV, "README.md", "skquad"),
    ]
    manifest = []
    for repo, revision, filename, folder in files:
        print(f"Downloading {repo}: {filename}", flush=True)
        path = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision,
                               filename=filename, local_dir=DATA / "raw" / folder)
        manifest.append({"repo": repo, "revision": revision, "file": filename,
                         "bytes": Path(path).stat().st_size, "sha256": file_hash(path)})
    write_json(DATA / "downloads.json", manifest)
    print("Downloads verified and recorded in data/downloads.json.")


def prepare(max_documents=5000, questions=1000, seed=42, full=False):
    import pyarrow.parquet as pq

    parquet = DATA / "raw/wiki/data/train-00000-of-00001.parquet"
    attribution = DATA / "raw/wiki/artifacts/attribution.jsonl"
    qa_path = DATA / "raw/skquad/validation.json.gz"
    if not all(path.exists() for path in (parquet, attribution, qa_path)):
        raise ValueError("Dataset files are missing. Run: python rag.py download")
    if questions < 2 or (not full and max_documents < questions):
        raise ValueError("Use at least two questions and max-docs >= questions.")

    available = set(pq.read_table(parquet, columns=["id"]).column("id").to_pylist())
    metadata = {}
    titles = {}
    for row in read_jsonl(attribution):
        if row["id"] in available:
            metadata[row["id"]] = {"title": row["title"], "url": row["url"]}
            titles.setdefault(normalize(row["title"].replace("_", " ")), row["id"])
    with gzip.open(qa_path, "rt", encoding="utf-8") as file:
        qa = [json.loads(line) for line in file if line.strip()]
    groups = {True: [], False: []}
    for row in qa:
        doc_id = titles.get(normalize(row["title"].replace("_", " ")))
        if doc_id:
            answers = row["answers"]["text"]
            groups[bool(answers)].append({"id": row["id"], "question": row["question"],
                                         "answers": answers, "source_doc_id": doc_id,
                                         "title": row["title"], "original_context": row["context"]})
    rng = random.Random(seed)
    no_answer_count = max(1, questions // 5)
    answer_count = questions - no_answer_count
    if len(groups[True]) < answer_count or len(groups[False]) < no_answer_count:
        raise ValueError("Not enough questions can be matched to retained Wikipedia articles.")
    selected_qa = rng.sample(sorted(groups[True], key=lambda row: row["id"]), answer_count)
    selected_qa += rng.sample(sorted(groups[False], key=lambda row: row["id"]), no_answer_count)
    selected_qa.sort(key=lambda row: row["id"])
    required = {row["source_doc_id"] for row in selected_qa}
    if full:
        selected_ids = set(metadata)
    else:
        extra = sorted(set(metadata) - required)
        selected_ids = required | set(rng.sample(extra, min(max_documents - len(required), len(extra))))

    print(f"Preparing {len(selected_ids)} documents and {len(selected_qa)} questions...", flush=True)
    documents = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1024, columns=["id", "text"]):
        for row in batch.to_pylist():
            if row["id"] in selected_ids:
                documents.append({"doc_id": row["id"], **metadata[row["id"]], "text": clean_text(row["text"])})
    documents.sort(key=lambda row: row["doc_id"])
    by_id = {row["doc_id"]: row for row in documents}
    for row in selected_qa:
        source = normalize(by_id[row["source_doc_id"]]["text"])
        row["reference_in_source"] = any(normalize(answer) in source for answer in row["answers"])
        row["original_context_in_source"] = normalize(row["original_context"]) in source

    write_jsonl(DATA / "documents.jsonl", documents)
    write_jsonl(DATA / "questions.jsonl", selected_qa)
    settings = {"wiki_revision": WIKI_REV, "qa_revision": QA_REV, "seed": seed,
                "documents": len(documents), "questions": len(selected_qa),
                "answerable": answer_count, "unanswerable_diagnostic": no_answer_count,
                "matched_validation_questions": sum(map(len, groups.values())),
                "answerable_reference_in_source": sum(row["reference_in_source"] for row in selected_qa),
                "original_context_in_source": sum(row["original_context_in_source"] for row in selected_qa),
                "documents_sha256": file_hash(DATA / "documents.jsonl"),
                "questions_sha256": file_hash(DATA / "questions.jsonl")}
    write_json(DATA / "prepared.json", settings)
    print(json.dumps(settings, ensure_ascii=False, indent=2))


def chunks_for_document(document, size=250, overlap=50):
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Require chunk-size > 0 and 0 <= overlap < chunk-size.")
    words = list(re.finditer(r"\S+", document["text"]))
    for start in range(0, len(words), size - overlap):
        end = min(start + size, len(words))
        left, right = words[start].start(), words[end - 1].end()
        yield {"chunk_id": f"{document['doc_id']}:{start:06d}-{end:06d}", "doc_id": document["doc_id"],
               "title": document["title"], "url": document["url"], "word_start": start,
               "word_end": end, "char_start": left, "char_end": right,
               "text": document["text"][left:right]}
        if end == len(words):
            break


if __name__ == "__main__":
    download()
