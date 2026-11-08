"""Check context boundaries, aggregation, sequence order, and padding behavior."""

import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

from neural_models import (
    TextClassifier, Word2Vec, average_embeddings, encode_sequences, word2vec_examples,
)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.vocabulary = {"alpha": 2, "beta": 3, "gamma": 4, "delta": 5}
        self.vectors = torch.randn(6, 8)
        self.vectors[0].zero_()

    def test_skipgram_does_not_cross_document_boundaries(self):
        documents = [["alpha", "beta"], ["gamma", "delta"]]
        contexts, targets = word2vec_examples(documents, self.vocabulary, "skipgram")
        pairs = set(zip(contexts.flatten().tolist(), targets.tolist()))
        self.assertEqual(pairs, {(2, 3), (3, 2), (4, 5), (5, 4)})

    def test_cbow_predicts_center_from_context(self):
        contexts, targets = word2vec_examples([["alpha", "beta"]], self.vocabulary, "cbow")
        self.assertEqual(contexts.tolist(), [[3, 0, 0, 0], [2, 0, 0, 0]])
        self.assertEqual(targets.tolist(), [2, 3])

    def test_average_counts_repetitions_and_ignores_unknown_words(self):
        vectors = torch.tensor([[0., 0.], [9., 9.], [2., 0.], [0., 4.]])
        documents = [["alpha", "alpha", "beta", "unknown"], [], ["unknown"]]
        result = average_embeddings(documents, self.vocabulary, vectors)
        assert_allclose(result, [[4 / 3, 4 / 3], [0, 0], [0, 0]])

    def test_sequence_order_unknown_words_truncation_and_empty_input(self):
        sequences, lengths = encode_sequences(
            [["beta", "unknown", "alpha"], [], ["alpha"]], self.vocabulary, max_length=2
        )
        self.assertEqual(sequences.tolist(), [[3, 1], [1, 0], [2, 0]])
        self.assertEqual(lengths.tolist(), [2, 1, 1])

    def test_extra_padding_does_not_change_network_predictions(self):
        short, lengths = encode_sequences([["alpha", "beta"]], self.vocabulary, 3)
        long, _ = encode_sequences([["alpha", "beta"]], self.vocabulary, 8)
        for kind in ("cnn", "rnn", "lstm"):
            with self.subTest(model=kind):
                model = TextClassifier(kind, self.vectors).eval()
                with torch.no_grad():
                    assert_allclose(model(short, lengths).numpy(), model(long, lengths).numpy(), atol=1e-6)

    def test_network_fine_tuning_does_not_modify_source_embeddings(self):
        original = self.vectors.clone()
        sequences, lengths = encode_sequences([["alpha", "beta"]], self.vocabulary, 3)
        model = TextClassifier("cnn", self.vectors)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(sequences, lengths).sum().backward()
        optimizer.step()
        assert_allclose(self.vectors.numpy(), original.numpy())

    def test_accidental_true_targets_are_not_negative_examples(self):
        model = Word2Vec(vocabulary_size=6, dimension=8)
        # Output embeddings start at zero: only the positive loss should remain.
        loss = model(torch.tensor([[2]]), torch.tensor([3]), torch.tensor([[3, 3, 3]]))
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
