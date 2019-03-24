import unittest
import numpy as np

from embedding_based.metrics import _cosine_similarity
from embedding_based.metrics import _embedding_sum
from embedding_based.metrics import _get_average
from embedding_based.metrics import average_sentence_level

from embedding_based.tests import EMBEDDINGS
from embedding_based.utils import load_word2vec_binary


class TestMetrics(unittest.TestCase):
    embeddings = load_word2vec_binary(EMBEDDINGS)

    def test_cosine_similarity(self):
        identity = np.array([1, 2, 3])
        self.assertAlmostEqual(
            _cosine_similarity(identity, identity), 1.0
        )
        self.assertAlmostEqual(
            _cosine_similarity(identity, -identity), -1.0
        )
        orthogonal = np.array([4, 1, -2])
        self.assertAlmostEqual(
            _cosine_similarity(identity, orthogonal), 0.0
        )

    def test_embedding_sum(self):
        self.assertTrue(
            (_embedding_sum(['foo', 'bar'], self.embeddings) ==
             np.zeros((self.embeddings.vector_size,))).all(),
            msg='OOV vector gets all zero'
        )
        sentence = ['human', 'computer', 'trees']
        sum_ = _embedding_sum(sentence, self.embeddings)
        self.assertGreater(np.linalg.norm(sum_), 0.0, msg='in-vocab sentence gets non-zero norm')

    def test_get_average(self):
        sentence = ['human', 'computer', 'trees']
        average = _get_average(sentence, self.embeddings)
        self.assertAlmostEqual(
            np.linalg.norm(average), 1.0, msg='average is normalized to unit vector'
        )

    def test_average_sentence_level(self):
        """
        More similar sentence pair should have a higher score.
        :return:
        """
        score_1 = average_sentence_level(
            'computer computer trees graph'.split(),
            'system time survey eps'.split(),
            self.embeddings,
        )
        score_2 = average_sentence_level(
            'computer computer trees graph'.split(),
            'computer computer graph graph'.split(),
            self.embeddings,
        )
        self.assertGreaterEqual(score_2, score_1)
