import os
import pickle
import numpy as np
import json
from nltk.tokenize import word_tokenize
import torch
from torch.autograd import Variable


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f)


def load_pickle(path):
    print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)


def lower_list(word_list):
    return [w.lower() for w in word_list]


def load_task(dataset_path, N=50*1000):
    data = []
    with open(dataset_path) as f:
        for i, d in enumerate(f):
            if i >= N:
                break
            d = json.loads(d)
            text = lower_list(word_tokenize(d['text']))
            stars = d['stars'] - 1 # map to [0, 4] from [1, 5]
            data.append((text, stars))
    return data


def load_glove_weights(glove_dir, embd_dim, vocab_size, word_index):
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.' + str(embd_dim) + 'd.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((vocab_size, embd_dim))
    print('embed_matrix.shape', embedding_matrix.shape)
    count = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            count += 1
        # else:
        #     print('not found', word)
        #     if count > 1000: break
    print('Use pre-embedded weights:', count, '/', len(word_index.items()))

    return embedding_matrix


def add_padding(data, seq_len):
    pad_len = max(0, seq_len - len(data))
    data += [0] * pad_len
    data = data[:seq_len]
    return data


def make_word_vector(data, w2i, query_len):
    vec_data = []
    for sentence in data:
        index_vec = [w2i[w] for w in sentence]
        index_vec = add_padding(index_vec, query_len)
        vec_data.append(index_vec)

    return to_var(torch.LongTensor(vec_data))
