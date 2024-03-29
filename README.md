# Embedding-based Metrics

Embedding-based metrics are methods to evaluate the semantic similarity between two sentences based on some pre-trained word vectors. These word vectors are trained to represent distributed meanings of words using models like *Skip-grams*, *CBOW (Continuous Bag of Words)*, or *Glove (Global Vector)*. They can also be applied to the evaluation of dialogue systems. This repository contains code to compute the embedding-based metrics as follows:

- Average
- Greedy matching
- Vector extrema

For more information about these embedding-based metrics, please read the [details](docs/details.md).

## Dependencies

- Python 3.6
- gensim 3.4.0 
    
## Usage

    python embedding_metrics.py path_to_ground_truth.txt path_to_predictions.txt path_to_embeddings.bin

The script assumes one example per line (e.g. one dialogue or one sentence per line). The embedding file should be in binary format as generated by the original word2vec tool from Google. If you find any problem loading your embedding, please refer to the [gensim document about the word2vec model](https://radimrehurek.com/gensim/models/word2vec.html).

## Recommended Word Embedding

The word embedding you are recommended to use is the *Word2Vec* vectors trained on the *Google News Corpus*.
This is also recommended by the original repository. To download this pre-trained embedding easily, here are some useful links:
- [Resource from Deeplearning4j](https://deeplearning4jblob.blob.core.windows.net/resources/wordvectors/GoogleNews-vectors-negative300.bin.gz)
- [Google Code Archive](https://code.google.com/archive/p/word2vec/)
- [Google Drive](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

## Acknowledgment

The main script `embedding_metrices.py` is adapted from [hed-dlg-truncated](https://github.com/julianser/hed-dlg-truncated). Thanks for their great script!

## Reference

[1] Section 3.1 of How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation

[2] A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues. Serban et al. (2015 a)
