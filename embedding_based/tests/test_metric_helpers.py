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

from embedding_based.metrics import _cos_sim
from embedding_based.metrics import _map_to_embeddings

from embedding_based.tests import EMBEDDINGS
from embedding_based.utils import load_word2vec_binary


class TestMetricHelpers(unittest.TestCase):
    embeddings = load_word2vec_binary(EMBEDDINGS)

    def test_cosine_similarity(self):
        identity = np.array([1, 2, 3])
        self.assertAlmostEqual(_cos_sim(identity, identity), 1.0)
        self.assertAlmostEqual(_cos_sim(identity, -identity), -1.0)
        orthogonal = np.array([4, 1, -2])
        self.assertAlmostEqual(_cos_sim(identity, orthogonal), 0.0)

    def test_map_to_embeddings(self):
        self.assertTrue(len(_map_to_embeddings(["foo", "bar"], self.embeddings)) == 0)
        self.assertTrue(
            len(_map_to_embeddings(["computer", "trees", "graph"], self.embeddings))
            == 3
        )
