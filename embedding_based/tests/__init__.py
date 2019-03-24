import os

DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
GROUND_TRUTH = os.path.join(DATA_ROOT, 'ground_truth.txt')
PREDICTED = os.path.join(DATA_ROOT, 'predicted.txt')
EMBEDDINGS = os.path.join(DATA_ROOT, 'word_vecs.bin')
VOCAB = os.path.join(DATA_ROOT, 'vocab.txt')
