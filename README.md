# Structured-Self-Attentive-Sentence-Embedding

A Structured Self-attentive Sentence Embedding
Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, Yoshua Bengio, ICLR 2017
https://arxiv.org/abs/1703.03130

<img src="https://user-images.githubusercontent.com/166852/33136258-ccc5bc08-cf72-11e7-8ddd-368e4a85a0a8.png" width="70%"/>

This repo contains the implementation of this paper in PyTorch.

## Requirements
- Python 3.x
- PyTorch 0.2.0.3
- nltk, for word tokenizer

## Dataset
Before you start to train the model, you need to download each dataset. Currently I only check Yelp dataset, no other two tasks in the paper.

- Yelp (https://www.yelp.com/dataset/download)
  - `./dataset/review.json`
- Glove 6B, 100 dim (https://nlp.stanford.edu/projects/glove/)
  - `./dataset/glove.6B.100d.txt`

## How to start

```
$ python main.py
```

## limitations

There may be stil some bugs, pull requests will be appreciated.
