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

# All baseline functions.
from embedding_based.origin import average_score
from embedding_based.origin import extrema_score
from embedding_based.origin import greedy_match_score

# All our functions.
from embedding_based.metrics import average_corpus_level
from embedding_based.metrics import extrema_corpus_level
from embedding_based.metrics import greedy_match_corpus_level

from embedding_based.tests import EMBEDDINGS, PREDICTED, GROUND_TRUTH
from embedding_based.utils import load_word2vec_binary as _load_word2vec_binary
from embedding_based.utils import load_corpus_from_file as _load_corpus_from_file

import collections

TestData = collections.namedtuple(
    "TestData", ["embeddings", "predicted", "ground_truth"]
)


def _load_all_data():
    """
    Load the test data for the tests against baseline.
    Both our and their functions take *corpus* as input, but in different
    format. Our version takes a nested list while theirs take a file.

    :return: TestData.
    """
    embeddings = _load_word2vec_binary(EMBEDDINGS)
    predicted = _load_corpus_from_file(PREDICTED)
    ground_truth = _load_corpus_from_file(GROUND_TRUTH)
    return TestData(
        embeddings=embeddings,
        predicted=(predicted, PREDICTED),
        ground_truth=(ground_truth, GROUND_TRUTH),
    )


TEST_DATA = _load_all_data()


class TestAgainstBaseline(unittest.TestCase):
    embeddings = TEST_DATA.embeddings

    def _test_one_metric(self, our_fn, their_fn):
        """
        Helper to run test on one metric.
        :param our_fn:
        :param their_fn:
        :return:
        """
        our_predicted, their_predicted = TEST_DATA.predicted
        our_ground_truth, their_ground_truth = TEST_DATA.ground_truth
        our_score = our_fn(our_predicted, our_ground_truth, self.embeddings)
        their_score = their_fn(their_predicted, their_ground_truth, self.embeddings)
        for our, their in zip(our_score, their_score):
            self.assertAlmostEqual(
                our,
                their,
                delta=1e15,
                msg="""
                our_fn %s
                their_fn %s
    
                our_score %r
                their_score %r
    
                predicted %r
                ground_truth %r
                """
                % (
                    our_fn,
                    their_fn,
                    our_score,
                    their_score,
                    our_predicted,
                    our_ground_truth,
                ),
            )

    def test_average(self):
        return self._test_one_metric(
            our_fn=average_corpus_level,
            their_fn=average_score,
        )

    def test_extrema(self):
        return self._test_one_metric(
            our_fn=extrema_corpus_level,
            their_fn=extrema_score,
        )

    def test_greedy_match(self):
        return self._test_one_metric(
            our_fn=greedy_match_corpus_level,
            their_fn=greedy_match_score,
        )
