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

import embedding_based as eb
from embedding_based import load_word2vec_binary
from embedding_based import load_corpus_from_file
from agenda.metric_helper import write_score

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


class MetricWrapper:

    def __init__(self, name, score_fn, corpus_score_fn):
        self.name = name
        self.score_fn = score_fn
        self.corpus_score_fn = corpus_score_fn

    def eval(self, hypo_corpus, ref_corpus, embeddings, embedding_file, output_dir):
        scores = [self.score_fn(h, r, embeddings) for h, r in zip(hypo_corpus, ref_corpus)]
        write_score(
            name=self.name,
            scores=scores,
            system=self.corpus_score_fn(hypo_corpus, ref_corpus, embeddings).mean,
            output=Path(output_dir).joinpath(self.name).with_suffix('.json'),
            params={
                'embedding': embedding_file,
            }
        )

    known_metrics = {
        'vector_average': (eb.average_sentence_level, eb.average_corpus_level),
        'vector_extrema': (eb.extrema_sentence_level, eb.extrema_corpus_level),
        'greedy_matching': (eb.greedy_match_sentence_level, eb.greedy_match_corpus_level)
    }

    @classmethod
    def factory(cls, name):
        args = cls.known_metrics[name]
        return cls(name, *args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-predicted', help="predicted text file, one example per line")
    parser.add_argument('-ground_truth', help="ground truth text file, one example per line")
    parser.add_argument('-e', '-embeddings', dest='embeddings', help="embeddings bin file")
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-A', action='store_true', help='compute embedding average')
    parser.add_argument('-X', action='store_true', help='compute vector extrema')
    parser.add_argument('-G', action='store_true', help='compute greedy matching')
    args = parser.parse_args()

    metrics = []
    if args.A:
        metrics.append(MetricWrapper.factory('vector_average'))
    if args.X:
        metrics.append(MetricWrapper.factory('vector_extrema'))
    if args.G:
        metrics.append(MetricWrapper.factory('greedy_matching'))

    if not metrics:
        parser.error('no metrics specified!')

    logging.info("loading embeddings file...")
    embeddings = load_word2vec_binary(args.embeddings)

    logging.info("loading predicted file...")
    predicted = load_corpus_from_file(args.predicted)

    logging.info("loading ground_truth file...")
    reference = load_corpus_from_file(args.ground_truth)

    for metric in metrics:
        metric.eval(
            hypo_corpus=predicted,
            ref_corpus=reference,
            embeddings=embeddings,
            embedding_file=args.embeddings,
            output_dir=args.prefix,
        )
