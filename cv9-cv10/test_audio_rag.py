"""Offline checks for audio conversion, ASR evaluation, provenance, and resumability."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import audio_data as data
import audio_rag as app
from rag import OUTPUT_SCHEMA, generate, make_messages
from rag_core import KnowledgeBase, build_index
from rag_data import chunks_for_document


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def document(self, identifier="a", text="one two three"):
        audio = self.folder / (identifier + ".wav")
        sf.write(audio, np.ones(800, dtype=np.float32) * 0.1, 16000)
        return {"doc_id": identifier, "title": identifier, "url": audio.as_uri(), "text": text,
                "audio_path": audio.name, "audio_sha256": data.file_hash(audio),
                "duration_seconds": 0.05}


class AudioTests(Workspace):
    def test_stereo_resampling_preserves_duration_and_averages_channels(self):
        path = self.folder / "stereo.wav"
        audio = np.column_stack([np.full(800, 0.2), np.full(800, 0.4)])
        sf.write(path, audio, 8000, subtype="FLOAT")
        converted = data.load_audio(path)
        self.assertEqual(converted.shape, (1600,))
        self.assertEqual(converted.dtype, np.float32)
        self.assertAlmostEqual(float(converted[100:-100].mean()), 0.3, places=3)

    def test_empty_audio_is_rejected(self):
        path = self.folder / "empty.wav"
        sf.write(path, np.array([], dtype=np.float32), 16000)
        with self.assertRaisesRegex(ValueError, "Empty"):
            data.load_audio(path)

    def test_question_uses_single_existing_recording_without_renaming(self):
        folder = self.folder / "input"
        folder.mkdir()
        (folder / "output.wav").touch()
        with patch.object(data, "ROOT", self.folder):
            self.assertEqual(data.question_path().name, "output.wav")
            (folder / "another.wav").touch()
            with self.assertRaisesRegex(ValueError, "select"):
                data.question_path()

    def test_long_audio_disables_truncation_and_uses_long_form(self):
        flags = {}
        class Processor:
            def __call__(self, audios, **kwargs):
                flags.update(kwargs)
                return {}
            def batch_decode(self, tokens, **kwargs):
                return ["The whole recording."]
        class Model:
            config = SimpleNamespace(max_target_positions=448)
            generation_config = SimpleNamespace(eos_token_id=9)
            def generate(self, **kwargs):
                flags.update(kwargs)
                return {"sequences": torch.tensor([[2, 3]]),
                        "segments": [[{"result": {"sequences": torch.tensor([2, 3, 9])}}]]}
        recognizer = data.WhisperASR.__new__(data.WhisperASR)
        recognizer.processor, recognizer.model, recognizer.torch = Processor(), Model(), torch
        result = recognizer.transcribe([np.zeros(31 * 16000, dtype=np.float32)])
        self.assertEqual(result, ["The whole recording."])
        self.assertFalse(flags["truncation"])
        self.assertTrue(flags["return_timestamps"])
        self.assertTrue(flags["return_attention_mask"])


class MetricsTests(Workspace):
    def test_normalization_and_edit_counts(self):
        self.assertEqual(data.normalize_asr("  ＨＥＬＬＯ, World! "), "hello world")
        scores = data.paired_scores("A b c", "a c")
        self.assertEqual(scores["deletions"], 1)
        self.assertAlmostEqual(scores["wer"], 1 / 3)
        self.assertFalse(scores["normalized_exact_match"])
        self.assertTrue(data.paired_scores("HELLO!", "Hello.")["normalized_exact_match"])
        self.assertFalse(data.paired_scores("HELLO!", "Hello.")["raw_exact_match"])

    def test_corpus_wer_is_word_weighted_and_pending_records_are_explicit(self):
        documents = [self.document("a", "a b c"), self.document("b", "d"), self.document("c", "pending")]
        data.write_jsonl(self.folder / "reference_documents.jsonl", documents)
        data.write_jsonl(self.folder / "transcriptions.jsonl", [
            {"doc_id": "a", "status": "ok", "text": "a b"},
            {"doc_id": "b", "status": "ok", "text": "x"}])
        with patch.object(data, "DATA", self.folder), patch.object(data, "RESULTS", self.folder / "results"):
            summary = data.evaluate_asr()
        self.assertEqual(summary["wer"], 0.5)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["coverage"], 2 / 3)
        self.assertEqual(summary["reference_words"], 4)


class CacheTests(Workspace):
    def test_transcription_resumes_and_rejects_changed_settings(self):
        documents = [self.document("a"), self.document("b")]
        data.write_jsonl(self.folder / "reference_documents.jsonl", documents)
        calls = []
        class Recognizer:
            def __init__(self, threads):
                pass
            def transcribe(self, audios):
                calls.append(len(audios))
                return ["recognized"] * len(audios)
        with patch.object(data, "DATA", self.folder), patch.object(data, "ROOT", self.folder), \
                patch.object(data, "model_signature", return_value={"version": 1}), \
                patch.object(data, "WhisperASR", Recognizer):
            data.transcribe(limit=1, batch_size=1)
            records = data.transcribe(batch_size=1)
            data.transcribe(batch_size=1)
            self.assertEqual(calls, [1, 1])
            self.assertEqual(len(records), 2)
            sf.write(self.folder / "a.wav", np.zeros(100), 16000)
            with self.assertRaisesRegex(ValueError, "audio changed"):
                data.transcribe()
            self.document("a")
            with patch.object(data, "model_signature", return_value={"version": 2}):
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    data.transcribe()

    def test_failed_transcripts_are_not_retried_without_flag(self):
        doc = self.document()
        data.write_jsonl(self.folder / "reference_documents.jsonl", [doc])
        class Recognizer:
            def __init__(self, threads):
                pass
            def transcribe(self, audios):
                raise RuntimeError("test failure")
        with patch.object(data, "DATA", self.folder), patch.object(data, "ROOT", self.folder), \
                patch.object(data, "model_signature", return_value={}), patch.object(data, "WhisperASR", Recognizer):
            self.assertEqual(data.transcribe()[0]["status"], "error")
            with patch.object(data, "WhisperASR") as recognizer:
                data.transcribe()
                recognizer.assert_not_called()
                recognizer.return_value.transcribe.return_value = ["repaired"]
                self.assertEqual(data.transcribe(retry_errors=True)[0]["text"], "repaired")


class RAGTests(Workspace):
    def test_chunk_preserves_audio_provenance(self):
        doc = self.document(text="a b c d e f")
        chunks = list(chunks_for_document(doc, size=4, overlap=1))
        for chunk in chunks:
            self.assertEqual(chunk["audio_path"], doc["audio_path"])
            self.assertEqual(chunk["text"], doc["text"][chunk["char_start"]:chunk["char_end"]])

    def test_cannot_index_incomplete_whisper_collection(self):
        data.write_jsonl(self.folder / "reference_documents.jsonl", [self.document()])
        data.write_jsonl(self.folder / "transcriptions.jsonl", [])
        with patch.object(app, "DATA", self.folder):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                app.publish_whisper_documents()

    def test_english_prompt_has_context_but_no_reference_metadata(self):
        chunk = {"chunk_id": "a:0-1", "title": "Test", "url": "audio", "text": "Source text",
                 "reference_answer": "FORBIDDEN"}
        messages = make_messages("Question?", [chunk], language="en")
        prompt = json.dumps(messages)
        self.assertIn("Answer in English", prompt)
        self.assertIn("Source text", prompt)
        self.assertNotIn("FORBIDDEN", prompt)
        self.assertIn("Odpovedaj po slovensky", make_messages("Q")[0]["content"])

    def test_english_generation_constrains_citations_to_supplied_chunk_ids(self):
        class Client:
            def request(self, path, payload):
                self.payload = payload
                raise RuntimeError("offline test")
        client = Client()
        chunk = {"chunk_id": "a:000000-000001", "title": "A", "url": "audio", "text": "Fact"}
        generate(client, "Q?", [chunk], language="en")
        allowed = client.payload["format"]["properties"]["citations"]["items"]["enum"]
        self.assertEqual(allowed, [chunk["chunk_id"]])
        self.assertNotIn("enum", OUTPUT_SCHEMA["properties"]["citations"]["items"])

    def test_custom_document_collection_builds_and_loads_without_global_data(self):
        from rag_core import EMBEDDER, GENERATOR
        class Client:
            def models(self):
                return {EMBEDDER: {"digest": "e1"}, GENERATOR: {"digest": "g1"}}
            def embed(self, texts):
                return np.tile(np.array([0.6, 0.8], dtype=np.float32), (len(texts), 1))
        source = self.folder / "custom.jsonl"
        data.write_jsonl(source, [self.document()])
        client, target = Client(), self.folder / "index"
        build_index(client, target, documents_path=source)
        kb = KnowledgeBase(client, target, documents_path=source)
        chunks, _ = kb.retrieve("one")
        self.assertEqual(chunks[0]["doc_id"], "a")
        self.assertEqual(chunks[0]["audio_path"], "a.wav")

    def test_generation_cache_resumes_and_retries_only_errors_when_requested(self):
        chunk = next(chunks_for_document(self.document()))
        class KB:
            state = {"signature": "test"}
            models = {"model": "digest"}
            def retrieve(self, query, k):
                return [chunk], []
        response = {"status": "ok", "answer": "one", "citations": [chunk["chunk_id"]],
                    "evidence": "one", "abstained": False}
        with patch.object(app, "RESULTS", self.folder), patch.object(app, "generate", return_value=response) as call:
            app.answer_case(None, KB(), "Q?", "whisper", "typed")
            app.answer_case(None, KB(), "Q?", "whisper", "voice")
            self.assertEqual(call.call_count, 1)
            call.return_value = {"status": "error", "error": "test"}
            app.answer_case(None, KB(), "Other?", "whisper", "voice")
            app.answer_case(None, KB(), "Other?", "whisper", "voice")
            self.assertEqual(call.call_count, 2)
            call.return_value = response
            result = app.answer_case(None, KB(), "Other?", "whisper", "voice", retry_errors=True)
            self.assertEqual(call.call_count, 3)
            self.assertTrue(result["response"]["grounding"]["evidence_in_cited_chunk"])


if __name__ == "__main__":
    unittest.main()
