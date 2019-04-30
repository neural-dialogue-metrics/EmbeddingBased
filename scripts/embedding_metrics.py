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
        def wrapper(fn):
            def get_mean(*args, **kwargs):
                return fn(*args, **kwargs).mean

            return get_mean

        self.name = name
        self.score_fn = wrapper(score_fn)
        self.corpus_score_fn = wrapper(corpus_score_fn)

    def eval(self, hypo_corpus, ref_corpus, embeddings, embedding_file, output_dir):
        write_score(
            name=self.name,
            scores=[self.score_fn(h, r, embeddings) for h, r in zip(hypo_corpus, ref_corpus)],
            system=self.corpus_score_fn(hypo_corpus, ref_corpus, embeddings),
            output=Path(output_dir).joinpath(self.name).with_suffix('.json'),
            params={
                'embedding': embedding_file,
            }
        )

    @classmethod
    def vector_average(cls):
        return cls(
            name='vector_average',
            score_fn=eb.average_sentence_level,
            corpus_score_fn=eb.average_corpus_level
        )

    @classmethod
    def vector_extrema(cls):
        return cls(
            name='vector_extrema',
            score_fn=eb.extrema_sentence_level,
            corpus_score_fn=eb.extrema_corpus_level
        )

    @classmethod
    def greedy_matching(cls):
        return cls(
            name='greedy_matching',
            score_fn=eb.greedy_match_sentence_level,
            corpus_score_fn=eb.greedy_match_corpus_level,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-predicted', help="predicted text file, one example per line")
    parser.add_argument('-ground_truth', help="ground truth text file, one example per line")
    parser.add_argument('-embeddings', help="embeddings bin file")
    parser.add_argument('--prefix', '-p')
    parser.add_argument('-A', action='store_true', help='compute embedding average')
    parser.add_argument('-X', action='store_true', help='compute vector extrema')
    parser.add_argument('-G', action='store_true', help='compute greedy matching')
    args = parser.parse_args()

    logging.info("loading embeddings file...")
    embeddings = load_word2vec_binary(args.embeddings)

    logging.info("loading predicted file...")
    predicted = load_corpus_from_file(args.predicted)

    logging.info("loading ground_truth file...")
    reference = load_corpus_from_file(args.ground_truth)

    if args.A:
        metric = MetricWrapper.vector_average()
    elif args.X:
        metric = MetricWrapper.vector_extrema()
    else:
        metric = MetricWrapper.greedy_matching()

    metric.eval(
        hypo_corpus=predicted,
        ref_corpus=reference,
        embeddings=embeddings,
        embedding_file=args.embeddings,
        output_dir=args.prefix,
    )
