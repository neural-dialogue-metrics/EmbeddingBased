"""Script to generate test data in tiny scale so that we can test it locally."""
import argparse

from gensim.models import Word2Vec
from gensim.test.utils import common_texts, common_dictionary
import random
import tempfile
import os

_PREDICTED = 'predicted.txt'
_GROUND_TRUTH = 'ground_truth.txt'
_WORD_VECS = 'word_vecs.bin'
_VOCAB = 'vocab.txt'


def train_and_save_model(save_path):
    """
    Train the word2vec model on the tiny data and save it in binary format.
    :param save_path: where to save the model.
    :return:
    """
    model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format(save_path, binary=True)


def _make_lines(list_of_tokens):
    """
    Turn a list of tokens into a list of lines ending in a newline.
    :param list_of_tokens: List[List[string]]
    :return: List[string], a list of line suitable for writelines().
    """
    return ['%s\n' % ' '.join(line) for line in list_of_tokens]


def save_predicted_and_ground_truth(predicted_path, ground_truth_path):
    """
    Generate the save the predicted data and ground truth data.
    :param predicted_path:
    :param ground_truth_path:
    :return:
    """

    # predicted is just the common texts.
    predicted_data = _make_lines(common_texts)
    with open(predicted_path, 'w') as f:
        f.writelines(predicted_data)

    # ground truth is a shuffle of the common texts.
    ground_truth_data = [random.sample(line, k=len(line)) for line in common_texts]
    ground_truth_data = _make_lines(ground_truth_data)
    with open(ground_truth_path, 'w') as f:
        f.writelines(ground_truth_data)


def save_vocab(vocab_path):
    """
    Save vocab of our fake data to file, one word per line.
    :param vocab_path:
    :return:
    """
    # items() is sorted.
    vocab_data = [item[1] for item in common_dictionary.items()]
    vocab_data = ["%s\n" % word for word in vocab_data]
    with open(vocab_path, 'w') as f:
        f.writelines(vocab_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--out_dir', help='output directory (default is a temp dir)')
    args = parser.parse_args()

    out_dir = args.out_dir or tempfile.mkdtemp()

    ground_truth_path = os.path.join(out_dir, _GROUND_TRUTH)
    predicted_path = os.path.join(out_dir, _PREDICTED)
    word_vecs_path = os.path.join(out_dir, _WORD_VECS)
    vocab_path = os.path.join(out_dir, _VOCAB)

    print('Writing test data to %r' % out_dir)
    train_and_save_model(word_vecs_path)
    save_predicted_and_ground_truth(predicted_path, ground_truth_path)
    save_vocab(vocab_path)
