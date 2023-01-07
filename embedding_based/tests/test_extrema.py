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

from embedding_based.metrics import _get_extrema
from embedding_based.metrics import extrema_sentence_level
from embedding_based.tests import EMBEDDINGS
from embedding_based.utils import load_word2vec_binary


class TestExtrema(unittest.TestCase):
    embeddings = load_word2vec_binary(EMBEDDINGS)

    def test_get_extrema(self):
        vectors = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, -3.0, -2.0],
            ]
        )
        extrema = np.array([2.0, -3.0, 3.0])
        expected = np.abs(_get_extrema(vectors) - extrema) < 1e-5
        self.assertTrue(expected.all())

    def test_extrema_sentence_level(self):
        reference = "graphs eps eps".split()
        score_1 = extrema_sentence_level(
            "computer eps eps".split(), reference, self.embeddings
        )
        score_2 = extrema_sentence_level(
            "eps eps eps".split(), reference, self.embeddings
        )
        self.assertGreater(score_2, score_1)
