"""Check lecture formulas, absent terms, and query weighting."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from tfidf import TF_METHODS, IDF_METHODS, compute_tf, compute_idf, tokenize, weight


class WeightingTests(unittest.TestCase):
    def setUp(self):
        self.counts = csr_matrix([[2, 1, 0], [0, 1, 3], [0, 0, 0]])

    def test_tf_formulas_and_empty_document(self):
        expected = {
            "binary": [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
            "absolute": [[2, 1, 0], [0, 1, 3], [0, 0, 0]],
            "relative": [[2 / 3, 1 / 3, 0], [0, 1 / 4, 3 / 4], [0, 0, 0]],
            "logarithmic": [[1 + np.log(2), 1, 0], [0, 1, 1 + np.log(3)], [0, 0, 0]],
            "augmented": [[1, 0.75, 0], [0, 2 / 3, 1], [0, 0, 0]],
        }
        for method, values in expected.items():
            with self.subTest(tf=method):
                assert_allclose(compute_tf(self.counts, method).toarray(), values)
        assert_array_equal(self.counts.toarray(), [[2, 1, 0], [0, 1, 3], [0, 0, 0]])

    def test_idf_formulas(self):
        # N=4, df=[1, 2, 2], max(df)=2.
        counts = csr_matrix([[2, 1, 0], [0, 1, 3], [0, 0, 1], [0, 0, 0]])
        expected = {
            "standard": np.log([4, 2, 2]),
            "smoothed": np.log([5 / 2, 5 / 3, 5 / 3]) + 1,
            "probabilistic": np.log([3, 1, 1]),
            "maximum": np.log([2, 1, 1]),
            "unit": [1, 1, 1],
        }
        for method, values in expected.items():
            with self.subTest(idf=method):
                assert_allclose(compute_idf(counts, method), values)

    def test_probabilistic_idf_keeps_negative_finite_weights(self):
        counts = csr_matrix([[1, 1], [2, 0], [3, 1]])
        assert_allclose(compute_idf(counts, "probabilistic"), [0, np.log(1 / 2)])

    def test_smoothed_weights_match_sklearn_without_normalization(self):
        for method, sublinear in [("absolute", False), ("logarithmic", True)]:
            expected = TfidfTransformer(norm=None, sublinear_tf=sublinear).fit_transform(self.counts)
            actual = weight(self.counts, method, compute_idf(self.counts, "smoothed"))
            assert_allclose(actual.toarray(), expected.toarray())

    def test_query_uses_collection_idf_and_term_order(self):
        vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
        counts = vectorizer.fit_transform(["free free prize", "lunch", "prize"])
        query_counts = vectorizer.transform(["free free unknown"])
        idf = compute_idf(counts, "standard")
        query = weight(query_counts, "absolute", idf).toarray()[0]
        columns = vectorizer.vocabulary_
        self.assertAlmostEqual(query[columns["free"]], 2 * np.log(3))
        self.assertEqual(query[columns["prize"]], 0)
        self.assertNotIn("unknown", columns)

    def test_unknown_query_remains_zero_for_all_variants(self):
        vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
        counts = vectorizer.fit_transform(["free prize", "lunch prize"])
        query_counts = vectorizer.transform(["unknownword"])
        for tf_method in TF_METHODS:
            for idf_method in IDF_METHODS:
                idf = compute_idf(counts, idf_method)
                query = weight(query_counts, tf_method, idf)
                documents = weight(counts, tf_method, idf)
                self.assertEqual(query.nnz, 0)
                assert_allclose(cosine_similarity(query, documents), [[0, 0]])

    def test_cosine_similarity_and_descending_order(self):
        query = csr_matrix([[1, 0]])
        documents = csr_matrix([[0, 1], [1, 1], [1, 0]])
        similarities = cosine_similarity(query, documents).ravel()
        assert_allclose(similarities, [0, 1 / np.sqrt(2), 1])
        assert_array_equal(np.argsort(-similarities, kind="stable"), [2, 1, 0])


if __name__ == "__main__":
    unittest.main()
