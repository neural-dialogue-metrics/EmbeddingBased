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

from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

from gensim.models.keyedvectors import Word2VecKeyedVectors

__all__ = [
    "load_corpus_from_file",
    "apply_metric_on_files",
    "load_word2vec_binary",
]


def load_corpus_from_file(path):
    """
    Create a corpus from a file of sentence.
    The sentences in the file must be already tokenized.
    :param path: a path-like object.
    :return: a list of sentences.
    """
    with open(path) as f:
        return [line.split() for line in f.readlines()]


def apply_metric_on_files(metric, hypothesis_file, reference_file):
    """
    Apply a metric on two files -- the hypothesis corpus and the reference corpus.
    :param metric: a callable that accept two corpses.
    :param hypothesis_file:
    :param reference_file:
    :return: what the metric returns.
    """
    return metric(
        load_corpus_from_file(hypothesis_file),
        load_corpus_from_file(reference_file),
    )


def load_word2vec_binary(file):
    """
    Load a word2vec embeddings in binary format as in the origin C tool.
    :param file: a binary file.
    :return: KeyedVectors
    """
    return Word2VecKeyedVectors.load_word2vec_format(file, binary=True)
