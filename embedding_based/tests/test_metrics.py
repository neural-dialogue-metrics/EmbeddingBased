import unittest
import numpy as np

from embedding_based.metrics import _cosine_similarity, _map_to_embeddings
from embedding_based.metrics import _embedding_sum
from embedding_based.metrics import _get_average
from embedding_based.metrics import _get_extrema

from embedding_based.metrics import average_sentence_level
from embedding_based.metrics import average_corpus_level

from embedding_based.tests import EMBEDDINGS
from embedding_based.tests import GROUND_TRUTH
from embedding_based.tests import PREDICTED

from embedding_based.utils import load_word2vec_binary
from embedding_based.utils import load_corpus_from_file


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
            np.linalg.norm(average), 1.0, msg='average_score is normalized to unit vector'
        )

    def test_average_sentence_level(self):
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
        self.assertGreaterEqual(score_2, score_1,
                                msg='More similar sentence pair should have a higher score.')

    def test_average_corpus_level(self):
        hypothesis_corpus = load_corpus_from_file(PREDICTED)
        reference_corpus = load_corpus_from_file(GROUND_TRUTH)
        score = average_corpus_level(hypothesis_corpus, reference_corpus, self.embeddings)
        self.assertAlmostEqual(score[0], 1.0, msg="""
        since our predicted is a shuffle of ground truth and average_score ignores order, the average_score of them must equal.
        """)

    def test_get_extrema(self):
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [2.0, -3.0, -2.0],
        ])
        extrema = np.array([2.0, -3.0, 3.0])
        expected = np.abs(_get_extrema(vectors) - extrema) < 1e-5
        self.assertTrue(expected.all())

    def test_map_to_embeddings(self):
        self.assertTrue(len(_map_to_embeddings(['foo', 'bar'], self.embeddings)) == 0)
        self.assertTrue(len(_map_to_embeddings(['computer', 'trees', 'graph'], self.embeddings)) == 3)

    def test_extrema_sentence_level(self):
        pass
