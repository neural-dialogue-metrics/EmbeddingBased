import unittest
import numpy as np

from embedding_based.metrics import _cosine_similarity, _map_to_embeddings
from embedding_based.metrics import _embedding_sum
from embedding_based.metrics import _get_average
from embedding_based.metrics import _get_extrema

from embedding_based.metrics import extrema_sentence_level
from embedding_based.metrics import extrema_corpus_level

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

