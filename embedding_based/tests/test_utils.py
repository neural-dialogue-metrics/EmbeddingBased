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