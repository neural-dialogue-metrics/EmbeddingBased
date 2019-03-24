import unittest
import numpy as np

from embedding_based.metrics import _get_extrema
from embedding_based.metrics import extrema_sentence_level
from embedding_based.tests import EMBEDDINGS
from embedding_based.utils import load_word2vec_binary


class TestExtrema(unittest.TestCase):
    embeddings = load_word2vec_binary(EMBEDDINGS)

    def test_get_extrema(self):
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [2.0, -3.0, -2.0],
        ])
        extrema = np.array([2.0, -3.0, 3.0])
        expected = np.abs(_get_extrema(vectors) - extrema) < 1e-5
        self.assertTrue(expected.all())

    def test_extrema_sentence_level(self):
        reference = 'graphs eps eps'.split()
        score_1 = extrema_sentence_level('computer eps eps'.split(), reference, self.embeddings)
        score_2 = extrema_sentence_level('eps eps eps'.split(), reference, self.embeddings)
        self.assertGreater(score_2, score_1)
