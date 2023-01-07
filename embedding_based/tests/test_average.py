# MIT License
#
# Copyright (c) 2019 Cong Feng.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest
import numpy as np

from embedding_based.tests import EMBEDDINGS
from embedding_based.tests import GROUND_TRUTH
from embedding_based.tests import PREDICTED

from embedding_based.utils import load_word2vec_binary
from embedding_based.utils import load_corpus_from_file

from embedding_based.metrics import _embedding_sum
from embedding_based.metrics import _get_average

from embedding_based.metrics import average_sentence_level
from embedding_based.metrics import average_corpus_level


class TestAverage(unittest.TestCase):
    embeddings = load_word2vec_binary(EMBEDDINGS)

    def test_embedding_sum(self):
        self.assertTrue(
            (
                _embedding_sum(["foo", "bar"], self.embeddings)
                == np.zeros((self.embeddings.vector_size,))
            ).all(),
            msg="OOV vector gets all zero",
        )
        sentence = ["human", "computer", "trees"]
        sum_ = _embedding_sum(sentence, self.embeddings)
        self.assertGreater(
            np.linalg.norm(sum_), 0.0, msg="in-vocab sentence gets non-zero norm"
        )

    def test_get_average(self):
        sentence = ["human", "computer", "trees"]
        average = _get_average(sentence, self.embeddings)
        self.assertAlmostEqual(
            np.linalg.norm(average),
            1.0,
            msg="average_score is normalized to unit vector",
        )

    def test_average_sentence_level(self):
        score_1 = average_sentence_level(
            "computer computer trees graph".split(),
            "system time survey eps".split(),
            self.embeddings,
        )
        score_2 = average_sentence_level(
            "computer computer trees graph".split(),
            "computer computer graph graph".split(),
            self.embeddings,
        )
        self.assertGreaterEqual(
            score_2,
            score_1,
            msg="More similar sentence pair should have a higher score.",
        )

    def test_average_corpus_level(self):
        hypothesis_corpus = load_corpus_from_file(PREDICTED)
        reference_corpus = load_corpus_from_file(GROUND_TRUTH)
        score = average_corpus_level(
            hypothesis_corpus, reference_corpus, self.embeddings
        )
        # since our predicted is a shuffle of ground truth and average_score ignores order, the average_score of them
        # must equal.
        self.assertAlmostEqual(score[0], 1.0)
