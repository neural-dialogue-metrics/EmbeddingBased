## Introduction

- Average Score, where the average_score of all word vectors composing a sentence is taken into account as a summary of a sentence, and cosine similarity of these averages is used as the final score.
```
Average(sent) = sum(sent) / norm(sent)
AverageScore(source, target) = cosine_similarity(Average(source), Average(target))
```

- Greedy Matching, as its name implies, it tries to find the maximum cosine similarity in a word-to-word basic, where each word of the source sentence is matched against all words of the target sentence to find the maximum cosine similarity. It then sums up these maximum cosine similarity scores for all words in a source sentence, normalized by the length of the source, following the pseudocode:
```
SumMaxCosine(source, target) = sum(max(cosine_similarity(s, t) for s in source) for t in target) / len(source)
```

The same procedure is performed on the (target, source) pair and the final result is the average_score of both, namely:
```
GreedyMacthing(source, target) = (SumMaxCosine(source, target) + SumMaxCosine(target, source)) / 2
```

- Vector Extrema, use a different way to generate a sentence representation from its constitute word embeddings. For each dimension of the word vectors, the extrema value is selected as below:
```
Min[i] = min(x for x in embeddings[i])  # Minimum of the i dimension.
Max[i] = max(x for x in embeddings[i])  # Maximum of the i dimension.
Extrema[i] = max(abs(Min[i]), Max[i])  # If the absolute of Min is large we take it. Otherwise take Max.
```

The sentence vector is then made up of these extrema values from all dimensions:
```
SentVec[i] = Extrema(i, embeddings)
```
    
Finally, the score is obtained by taking the cosine similarities of two `SentVec`.

## Related Works

These metrics are used as one of the metrics to evaluate the VHRED model proposed in Serban et al (2015 a), among others (the average length of response, the word entropy, and the utterance entropy w.r.t the unigram entropy of the training corpus and human evaluation). In *section 4.3 Results of Metric-based Evaluation*, settings of how the embedding-based metrics are used and their interpretation are detailed, along with evaluation results.

In the *How NOT to evaluate your dialogue system* paper, the authors discuss the embedding-based metrics in *section 3.2 Embedding-based Metrics*. They conclude that the embedding metrics, like other overlap-based metrics, do not correlate with human evaluation (not at all on Ubuntu Dialogue Corpus and weakly on Twitter Corpus). Despite this, Serban et al. interpret the metric as measuring *top similarity*. They then show that the HRED and VHRED models capture the topic in the context appropriately.
