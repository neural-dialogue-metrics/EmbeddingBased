import unittest
import numpy as np

from embedding_based.metrics import _cos_sim
from embedding_based.metrics import _map_to_embeddings

from embedding_based.tests import EMBEDDINGS
from embedding_based.utils import load_word2vec_binary


class TestMetricHelpers(unittest.TestCase):
    embeddings = load_word2vec_binary(EMBEDDINGS)

    def test_cosine_similarity(self):
        identity = np.array([1, 2, 3])
        self.assertAlmostEqual(
            _cos_sim(identity, identity), 1.0
        )
        self.assertAlmostEqual(
            _cos_sim(identity, -identity), -1.0
        )
        orthogonal = np.array([4, 1, -2])
        self.assertAlmostEqual(
            _cos_sim(identity, orthogonal), 0.0
        )

    def test_map_to_embeddings(self):
        self.assertTrue(len(_map_to_embeddings(['foo', 'bar'], self.embeddings)) == 0)
        self.assertTrue(len(_map_to_embeddings(['computer', 'trees', 'graph'], self.embeddings)) == 3)
