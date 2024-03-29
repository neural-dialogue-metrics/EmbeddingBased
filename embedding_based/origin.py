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

"""
Word embedding based evaluation metrics for dialogue.

This method implements three evaluation metrics based on Word2Vec word embeddings, which compare a target utterance
with a model utterance:

1) Computing cosine-similarity between the mean word embeddings of the target utterance and
of the model utterance
2) Computing greedy matching between word embeddings of target utterance and model utterance (
Rus et al., 2012)
3) Computing word embedding extrema scores (Forgues et al., 2014)

We believe that these metrics are suitable for evaluating dialogue systems.

Example run:

    python embedding_metrics.py path_to_ground_truth.txt path_to_predictions.txt path_to_embeddings.bin

The script assumes one example per line (e.g. one dialogue or one sentence per line), where line n in
'path_to_ground_truth.txt' matches that of line n in 'path_to_predictions.txt'.

NOTE: The metrics are not symmetric w.r.t. the input sequences.
      Therefore, DO NOT swap the ground truths with the predicted responses.

References:

A Comparison of Greedy and Optimal Assessment of Natural Language Student Input Word Similarity Metrics Using Word to
Word Similarity Metrics. Vasile Rus, Mihai Lintean. 2012. Proceedings of the Seventh Workshop on Building Educational
Applications Using NLP, NAACL 2012.

Bootstrapping Dialog Systems with Word Embeddings. G. Forgues, J. Pineau, J. Larcheveque, R. Tremblay. 2014. Workshop
on Modern Machine Learning and Natural Language Processing, NIPS 2014.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np

__docformat__ = "restructedtext en"
__authors__ = ("Chia-Wei Liu", "Iulian Vlad Serban")

__all__ = [
    "average_score",
    "extrema_score",
    "greedy_match_score",
]


def greedy_match_score(fileone, filetwo, w2v):
    res1 = _greedy_score(fileone, filetwo, w2v)
    res2 = _greedy_score(filetwo, fileone, w2v)
    res_sum = (res1 + res2) / 2.0

    return (
        np.mean(res_sum),
        1.96 * np.std(res_sum) / float(len(res_sum)),
        np.std(res_sum),
    )


def _greedy_score(fileone, filetwo, w2v):
    f1 = open(fileone, "r")
    f2 = open(filetwo, "r")
    r1 = f1.readlines()
    r2 = f2.readlines()
    dim = w2v.vector_size  # embedding dimensions

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        y_count = 0
        x_count = 0
        o = 0.0
        Y = np.zeros((dim, 1))
        for tok in tokens2:
            if tok in w2v:
                Y = np.hstack((Y, (w2v[tok].reshape((dim, 1)))))
                y_count += 1

        for tok in tokens1:
            if tok in w2v:
                tmp = w2v[tok].reshape((1, dim)).dot(Y)
                o += np.max(tmp)
                x_count += 1

        # if none of the words in response or ground truth have embeddings, count result as zero
        if x_count < 1 or y_count < 1:
            scores.append(0)
            continue

        o /= float(x_count)
        scores.append(o)
    f1.close()
    f2.close()
    return np.asarray(scores)


def extrema_score(fileone, filetwo, w2v):
    f1 = open(fileone, "r")
    f2 = open(filetwo, "r")
    r1 = f1.readlines()
    r2 = f2.readlines()

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        X = []
        for tok in tokens1:
            if tok in w2v:
                X.append(w2v[tok])
        Y = []
        for tok in tokens2:
            if tok in w2v:
                Y.append(w2v[tok])

        # if none of the words have embeddings in ground truth, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        xmax = np.max(X, 0)  # get positive max
        xmin = np.min(X, 0)  # get abs of min
        xtrema = []
        for i in range(len(xmax)):
            if np.abs(xmin[i]) > xmax[i]:
                xtrema.append(xmin[i])
            else:
                xtrema.append(xmax[i])
        X = np.array(xtrema)  # get extrema

        ymax = np.max(Y, 0)
        ymin = np.min(Y, 0)
        ytrema = []
        for i in range(len(ymax)):
            if np.abs(ymin[i]) > ymax[i]:
                ytrema.append(ymin[i])
            else:
                ytrema.append(ymax[i])
        Y = np.array(ytrema)

        o = np.dot(X, Y.T) / np.linalg.norm(X) / np.linalg.norm(Y)

        scores.append(o)

    scores = np.asarray(scores)
    f1.close()
    f2.close()
    return np.mean(scores), 1.96 * np.std(scores) / float(len(scores)), np.std(scores)


def average_score(fileone, filetwo, w2v):
    f1 = open(fileone, "r")
    f2 = open(filetwo, "r")
    r1 = f1.readlines()
    r2 = f2.readlines()
    dim = w2v.vector_size  # dimension of embeddings

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        X = np.zeros((dim,))
        for tok in tokens1:
            if tok in w2v:
                X += w2v[tok]
        Y = np.zeros((dim,))
        for tok in tokens2:
            if tok in w2v:
                Y += w2v[tok]

        # if none of the words in ground truth have embeddings, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        X = np.array(X) / np.linalg.norm(X)
        Y = np.array(Y) / np.linalg.norm(Y)
        o = np.dot(X, Y.T) / np.linalg.norm(X) / np.linalg.norm(Y)

        scores.append(o)

    scores = np.asarray(scores)
    f1.close()
    f2.close()
    return np.mean(scores), 1.96 * np.std(scores) / float(len(scores)), np.std(scores)
