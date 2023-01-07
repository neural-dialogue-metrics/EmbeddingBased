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

from embedding_based.utils import load_word2vec_binary
from embedding_based.utils import load_corpus_from_file

from embedding_based.tests import EMBEDDINGS
from embedding_based.tests import VOCAB
from embedding_based.tests import GROUND_TRUTH


class TestUtils(unittest.TestCase):
    def test_load_embeddings(self):
        embeddings = load_word2vec_binary(EMBEDDINGS)
        with open(VOCAB) as f:
            vocab_list = f.read().splitlines()
        for word in vocab_list:
            self.assertTrue(word in embeddings)

    def test_load_corpus(self):
        corpus = load_corpus_from_file(GROUND_TRUTH)
        self.assertTrue(isinstance(corpus, list))
        self.assertTrue(isinstance(corpus[0], list))
