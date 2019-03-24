"""
The origin implementation.
"""
import numpy as np

__all__ = [
    "average_score",
    "extrema_score",
    "greedy_match_score",
]


def greedy_match_score(fileone, filetwo, w2v):
    res1 = _greedy_score(fileone, filetwo, w2v)
    res2 = _greedy_score(filetwo, fileone, w2v)
    res_sum = (res1 + res2) / 2.0

    return np.mean(res_sum), 1.96 * np.std(res_sum) / float(len(res_sum)), np.std(res_sum)


def _greedy_score(fileone, filetwo, w2v):
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
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
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
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
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
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
