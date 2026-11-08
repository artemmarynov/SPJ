"""Offline tests for chunk provenance, resumability, prompts, and QA metrics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rag import evaluate, generate, make_messages
from rag_core import (
    EMBEDDER, GENERATOR, EmbeddingCache, OPTIONS, answer_scores,
    build_index, grounding_checks, ranked_indices, retrieval_scores, split_overlong_chunk,
)
from rag_data import ROOT, chunks_for_document, normalize, write_json, write_jsonl


def document(identifier="doc1", count=12):
    return {"doc_id": identifier, "title": "Test", "url": "https://example.org/" + identifier,
            "text": " ".join(f"word{index}" for index in range(count))}


class FakeEmbeddings:
    def __init__(self, maximum_words=100):
        self.calls = 0
        self.maximum_words = maximum_words

    def models(self):
        return {EMBEDDER: {"digest": "embedding-v1"}, GENERATOR: {"digest": "generator-v1"}}

    def embed(self, texts):
        self.calls += 1
        if any(len(text.split(" | text: ")[-1].split()) > self.maximum_words for text in texts):
            raise RuntimeError("the input length exceeds the context length")
        return np.tile(np.array([0.6, 0.8, 0], dtype=np.float32), (len(texts), 1))


class TemporaryWorkspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rag-test-", dir=ROOT)
        self.folder = Path(self.temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to(ROOT.resolve()))

    def tearDown(self):
        self.assertTrue(self.folder.is_relative_to(ROOT.resolve()))
        self.temporary.cleanup()


class ChunkTests(unittest.TestCase):
    def test_overlap_and_character_provenance(self):
        source = document(count=10)
        chunks = list(chunks_for_document(source, size=4, overlap=1))
        self.assertEqual([(row["word_start"], row["word_end"]) for row in chunks], [(0, 4), (3, 7), (6, 10)])
        for row in chunks:
            self.assertEqual(row["text"], source["text"][row["char_start"]:row["char_end"]])
            self.assertEqual(row["doc_id"], "doc1")
        self.assertEqual(chunks[0]["text"].split()[-1], chunks[1]["text"].split()[0])

    def test_documents_are_separate_and_empty_text_has_no_chunks(self):
        chunks = [row for source in [document("a", 3), document("b", 3)]
                  for row in chunks_for_document(source)]
        self.assertEqual([row["doc_id"] for row in chunks], ["a", "b"])
        self.assertEqual(list(chunks_for_document(document(count=0))), [])

    def test_invalid_overlap_is_rejected(self):
        for size, overlap in [(0, 0), (4, 4), (4, -1)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                list(chunks_for_document(document(), size, overlap))

    def test_adaptive_split_preserves_all_words_and_offsets(self):
        source = document(count=20)
        parent = list(chunks_for_document(source, 12, 2))[1]
        children = split_overlong_chunk(parent, overlap=2)
        covered = set()
        for child in children:
            covered.update(range(child["word_start"], child["word_end"]))
            self.assertEqual(child["text"], source["text"][child["char_start"]:child["char_end"]])
        self.assertEqual(covered, set(range(parent["word_start"], parent["word_end"])))


class CacheTests(TemporaryWorkspace):
    def test_overflow_subdivision_is_cached_and_resumable(self):
        source = document()
        parent = list(chunks_for_document(source))[0]
        client = FakeEmbeddings(maximum_words=4)
        cache = EmbeddingCache(client, self.folder / "cache.sqlite", "v1", overlap=2)
        first = cache.embed([parent])
        calls = client.calls
        cache.connection.close()
        resumed = EmbeddingCache(client, self.folder / "cache.sqlite", "v1", overlap=2)
        second = resumed.embed([parent])
        self.assertEqual(client.calls, calls)
        self.assertEqual([row[0] for row in first], [row[0] for row in second])
        self.assertTrue(all(len(row["text"].split()) <= 4 for row, _ in second))
        covered = {position for row, _ in second for position in range(row["word_start"], row["word_end"])}
        self.assertEqual(covered, set(range(12)))
        resumed.connection.close()

    def test_complete_index_is_reused_and_incompatible_settings_fail(self):
        write_jsonl(self.folder / "documents.jsonl", [document("a"), document("b")])
        client = FakeEmbeddings()
        target = self.folder / "index"
        with patch("rag_core.DATA", self.folder):
            build_index(client, target, size=4, overlap=1)
            calls = client.calls
            build_index(client, target, size=4, overlap=1)
            self.assertEqual(client.calls, calls)
            with self.assertRaises(ValueError):
                build_index(client, target, size=4, overlap=0)
        matrix = np.load(target / "vectors.npy")
        self.assertEqual(matrix.shape, (8, 3))

    def test_interrupted_evaluation_resumes_without_regenerating_finished_mode(self):
        source = document(count=3)
        chunk = {**next(chunks_for_document(source)), "score": 1.0}
        question = {"id": "q1", "question": "Question?", "answers": ["word1"],
                    "source_doc_id": "doc1", "title": "Test", "reference_in_source": True,
                    "original_context_in_source": True}
        write_jsonl(self.folder / "questions.jsonl", [question])
        write_json(self.folder / "prepared.json", {})
        class KB:
            state = {"signature": "index-v1"}
            models = FakeEmbeddings().models()
            def cache_queries(self, questions):
                pass
            def retrieve(self, question, k):
                return [chunk], [source]
        response = {"status": "ok", "answer": "word1", "citations": [chunk["chunk_id"]],
                    "evidence": "word1", "abstained": False, "seconds": 0.1}
        target = self.folder / "results"
        with patch("rag.DATA", self.folder), patch("rag.generate", side_effect=[response, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                evaluate(None, KB(), target)
        with patch("rag.DATA", self.folder), patch("rag.generate", return_value=response) as generated:
            evaluate(None, KB(), target)
            self.assertEqual(generated.call_count, 1)
        result = json.loads((target / "evaluation.jsonl").read_text())
        self.assertEqual(result["rag"]["status"], "ok")
        self.assertEqual(result["llm_only"]["status"], "ok")
        for filename in ("answers.csv", "summary.csv", "summary.json", "manifest.json"):
            self.assertTrue((target / filename).is_file(), filename)
        self.assertFalse((target / "manual_review.csv").exists())
        self.assertFalse((target / "REPORT.md").exists())


class EvaluationTests(unittest.TestCase):
    def test_slovak_normalization_preserves_accents(self):
        self.assertEqual(normalize("  ČESKÁ, republika! "), "česká republika")
        self.assertNotEqual(normalize("sud"), normalize("súd"))

    def test_exact_match_best_reference_and_repeated_token_f1(self):
        self.assertEqual(answer_scores("Bratislava!", ["Košice", "bratislava"]), {"exact_match": 1, "token_f1": 1})
        self.assertAlmostEqual(answer_scores("a a b", ["a b b"])["token_f1"], 2 / 3)
        self.assertIsNone(answer_scores("anything", []))
        self.assertEqual(answer_scores("Neviem.", ["Košice"])["token_f1"], 0)

    def test_rankings_and_source_metrics(self):
        self.assertEqual(ranked_indices([0.2, 0.8, 0.8], 2).tolist(), [1, 2])
        rows = [{"doc_id": "a", "text": "other"}, {"doc_id": "b", "text": "Mesto Košice"}]
        scores = retrieval_scores(rows, "b", ["Košice"])
        self.assertEqual(scores, {"source_recall_at_k": 1, "reciprocal_rank_at_k": 0.5,
                                  "reference_in_retrieved_text": True})

    def test_quote_must_belong_to_the_cited_chunk(self):
        chunks = [{"chunk_id": "a", "text": "Mesto Košice"}, {"chunk_id": "b", "text": "Rieka Váh"}]
        answer = {"citations": ["a"], "evidence": "Rieka Váh", "abstained": False}
        self.assertFalse(grounding_checks(answer, chunks)["evidence_in_cited_chunk"])
        answer["citations"] = ["b"]
        self.assertTrue(grounding_checks(answer, chunks)["evidence_in_cited_chunk"])
        answer["citations"] = ["invented"]
        self.assertFalse(grounding_checks(answer, chunks)["citation_ids_valid"])

    def test_prompts_have_no_qa_reference_fields_and_plain_has_no_context(self):
        chunk = {"chunk_id": "a", "title": "Title", "url": "url", "text": "CORPUS_TEXT",
                 "answers": ["FORBIDDEN_REFERENCE"], "original_context": "FORBIDDEN_CONTEXT"}
        rag = json.dumps(make_messages("QUESTION", [chunk]))
        plain = json.dumps(make_messages("QUESTION"))
        self.assertNotIn("FORBIDDEN", rag)
        self.assertIn("CORPUS_TEXT", rag)
        self.assertNotIn("CORPUS_TEXT", plain)

    def test_same_model_settings_and_errors_are_not_fabricated_answers(self):
        class Client:
            def request(self, path, payload):
                self.payload = payload
                raise RuntimeError("server unavailable")
        client = Client()
        response = generate(client, "QUESTION", [])
        self.assertEqual(response["status"], "error")
        self.assertNotIn("answer", response)
        self.assertEqual(client.payload["model"], GENERATOR)
        self.assertEqual(client.payload["options"], OPTIONS)
        self.assertFalse(client.payload["think"])


if __name__ == "__main__":
    unittest.main()
