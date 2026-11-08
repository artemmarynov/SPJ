"""Check target-word alignment and cosine calculations without downloading a model."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from contextual_embeddings import cosine_matrix, find_word, target_positions


class ContextualEmbeddingTests(unittest.TestCase):
    def test_complete_word_and_case(self):
        self.assertEqual(find_word("The BANK approved it.", "bank"), (4, 8))
        self.assertEqual(find_word("A bankrupt company visited the bank.", "bank"), (31, 35))

    def test_missing_or_ambiguous_target_is_rejected(self):
        for sentence in ("The banker arrived.", "A bank near another bank."):
            with self.subTest(sentence=sentence), self.assertRaises(ValueError):
                find_word(sentence, "bank")

    def test_subtokens_at_sentence_start_exclude_special_tokens(self):
        offsets = [(0, 0), (0, 2), (2, 4), (5, 9), (0, 0)]
        self.assertEqual(target_positions(offsets, (0, 4)), [1, 2])

    def test_missing_or_partially_truncated_target_is_rejected(self):
        for offsets in [[(0, 0), (0, 2)], [(0, 0), (5, 9)]]:
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                target_positions(offsets, (0, 4))

    def test_cosine_values_and_zero_vector(self):
        vectors = np.array([[1., 0.], [0., 2.], [-3., 0.], [0., 0.]])
        assert_allclose(cosine_matrix(vectors),
                        [[1, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0]])

    def test_identical_lookup_vectors_have_identical_cosines(self):
        vectors = np.tile([1., 2., 3.], (6, 1))
        assert_allclose(cosine_matrix(vectors), np.ones((6, 6)))


if __name__ == "__main__":
    unittest.main()
