from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

from embedding_based import load_word2vec_binary
from embedding_based import load_corpus_from_file
from embedding_based import (average_corpus_level,
                             greedy_match_corpus_level,
                             extrema_corpus_level)
import argparse
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth', help="ground truth text file, one example per line")
    parser.add_argument('predicted', help="predicted text file, one example per line")
    parser.add_argument('embeddings', help="embeddings bin file")
    args = parser.parse_args()

    logging.info("loading embeddings file...")
    embeddings = load_word2vec_binary(args.embeddings)

    logging.info("loading predicted file...")
    predicted = load_corpus_from_file(args.predicted)

    logging.info("loading ground_truth file...")
    reference = load_corpus_from_file(args.ground_truth)

    r = average_corpus_level(predicted, reference, embeddings)
    print("Embedding Average Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))

    r = greedy_match_corpus_level(predicted, reference, embeddings)
    print("Greedy Matching Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))

    r = extrema_corpus_level(predicted, reference, embeddings)
    print("Extrema Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))
