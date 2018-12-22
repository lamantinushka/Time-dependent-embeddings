import numpy as np
import pandas as pd
import os

from scipy import io as sio
from scipy import sparse as sp
from collections import Counter

def create_vocabulary(sentences, r=200):
    vocabulary = {}
    count = Counter()
    for sentence in sentences:
        count.update(sentence)
    ind = 0
    for word, c in count.most_common():
        if c >= r:
            vocabulary[word] = ind
            ind += 1
    return vocabulary
  
def create_corpus_matrix(sentences, vocabulary):
    row = []
    col = []
    data = []
    count = Counter()
    for sentence in sentences:
        sentence_t = [None]*2 + [vocab.setdefault(word, None) for word in sentence] + [None]*2
        for i, word in enumerate(sentence_t[2:]):
            if word != None:
                for pair in sentence_t[i : i + 2]:
                    if pair != None:
                        count.update([tuple(sorted((word, pair)))])
    for key in count:
        row.append(key[0])
        col.append(key[1])
        data.uppend(count[key])
        row.append(key[1])
        col.append(key[0])
        data.uppend(count[key])
    idx = np.argsort(row)
    corpus_matrix = sp.csr_matrix((np.array(data)[idx], (np.array(row)[idx], np.array(col)[idx])),
        shape = (len(vocabulary), len(vocabulary)))
    return corpus_matrix

def create_SPPMI(Times, k, path='data'):
    SPPMI = [sio.loadmat(os.path.join(path, 'pmi_{}.mat'.format(t)))['pmi'] for t in range(Times)]
    for t in range(Times):
        print(t)
        sppmi = SPPMI[t].toarray() - np.log(k)
        sppmi[sppmi < 0 ] = 0
        SPPMI[t] = sp.csc_matrix(sppmi)
    return SPPMI

def static_w2v(SPPMI, rank):
    U, S, V = sp.linalg.svds(SPPMI, k = rank)
    return U*np.sqrt(S)

class Corpus(object):
    def __init__(self, k = 5, path='data'):
        self.SPPMI = []
        self.k = 5
        self.times = 27
        for t in range(self.times):
            M = sio.loadmat(os.path.join(path, 'pmi_{}.mat'.format(t)))['pmi'] 
            M.data -= np.log(k)
            M.data = np.array([x if x>0 else 0. for x in M.data])
            M.eliminate_zeros()
            self.SPPMI.append(M)

        df = pd.read_csv(os.path.join(path, 'wordIDHash.csv'),
            names=['idx', 'word', 'freq'], index_col='word')
        self.vocab = df['idx'].to_dict()
        self.size = len(self.vocab)