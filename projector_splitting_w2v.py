from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import procrustes
from scipy.sparse.linalg import svds
import numpy as np

class IntegratedW2V(object):
    def __init__(self, rank = 100):
        self.embeddings = []
        self.rank = rank
        self.vocab = None
        self.inv_vocab = None
        self.symmetry_loss = [0]
    
    def step(self, dA, U, S, V):
        K_1 = U @ S + dA @ V
        U_1 , S_1_cap = np.linalg.qr(K_1)
        S_0_tilda = S_1_cap - U_1.T @ dA @ V
        L_1 = V @ S_0_tilda.T + dA.T @ U_1
        V_1, S_1_T = np.linalg.qr(L_1)
        return U_1, S_1_T.T, V_1
  
    def fit(self, corpus):
        self.vocab = corpus.vocab
        self.inv_vocab = {val: key for key, val in self.vocab.items()}
        N = corpus.times
        U, S, V = svds(corpus.SPPMI[0], k=self.rank) 
        V = V.T
        S = np.diag(S)
        self.embeddings.append(U.dot(np.sqrt(S)))
        for T in range(1, N):
            dA = (corpus.SPPMI[T] - corpus.SPPMI[T - 1])
            U, S, V = self.step(dA, U, S, V)
            u, s, v = np.linalg.svd(S)
            v = v.T
            U_, V_, l = procrustes(U.dot(u*np.sqrt(s)), V.dot(v*np.sqrt(s)))
            self.embeddings.append(U_)
            self.symmetry_loss.append(l)
      
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
