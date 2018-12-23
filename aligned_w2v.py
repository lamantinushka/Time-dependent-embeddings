from scipy.spatial import procrustes
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import static_w2v

class AlignedW2V(object):
    def __init__(self, rank = 100):
        self.embeddings = []
        self.rank = rank
        self.vocab = None
        self.inv_vocab = None
    
    def fit(self, corpus):
        self.vocab = corpus.vocab
        self.inv_vocab = {val: key for key, val in self.vocab.items()}
        self.embeddings = [static_w2v(M, self.rank) for M in corpus.SPPMI]
        for t in range(1, corpus.times):
            self.embeddings[t-1], self.embeddings[t], disp = procrustes(self.embeddings[t-1], self.embeddings[t])
      
    def k_nearest(self, word, k = 5, T = None):
        idx = self.vocab[word]
        if T != None:
            v = self.embeddings[T][idx]
            cosin = cosine_similarity(v.reshape(1, -1), self.embeddings[T]).reshape(-1)
            arg = np.argsort(cosin)[-k - 1 : -1][::-1]
            neighbors = [(self.inv_vocab[ind], cosin[ind]) for ind in arg]
            return neighbors
        else:
            neighbors = [self.k_nearest(word, k, t) for t in range(len(self.embeddings))]
        return neighbors
  
    def embedding(self, word, T = None):
        idx = self.vocab[word]
        if T:
            return self.embeddings[T][idx]
        return [self.embeddings[t][idx] for t in range(len(self.embeddings))]
