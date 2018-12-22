import numpy as np
import pandas as pd
import os
from scipy import io as sio


def static_w2v(SPPMI, rank):
    U, S, V = sp.linalg.svds(SPPMI, k = rank)
    return U*np.sqrt(S)

class Corpus(object):
    def __init__(self, k = 5):
        self.SPPMI = []
        self.k = 5
        self.times = 27
        for t in range(self.times):
            M = sio.loadmat('data/pmi_{}.mat'.format(t))['pmi'] 
            M.data -= np.log(k)
            M.data = np.array([x if x>0 else 0. for x in M.data])
            M.eliminate_zeros()
            self.SPPMI.append(M)

        df = pd.read_csv('data/wordIDHash.csv', names=['idx', 'word', 'freq'], index_col='word')
        self.vocab = df['idx'].to_dict()
        self.size = len(self.vocab)
