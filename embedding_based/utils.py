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
