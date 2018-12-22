import numpy as np
from scipy import sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from scipy import io as sio

class DynamicW2V(object):
    def __init__(self, rank = 50, gamma = 50, llambda = 10, tau = 50, threshold=1e-3):
        self.embeddings = []
        self.gamma = gamma
        self.llambda = llambda
        self.tau = tau
        self.rank = rank
        self.threshold = threshold
    
    def _get_batches(self, n, b):
        batchinds = []
        current = 0
        while current < n:
            inds = range(current, min(current + b, n))
            current = min(current + b, n)
            batchinds.append(inds)
        return batchinds
  
    def _step(self, t, T, Yt, Ut, Wp, Wn, idx):
        # shape U: nxr
        # shape Y: nxb
        UtU = Ut.T @ Ut
        r = UtU.shape[0]
        if t > 0 and t < T-1:
            A = UtU + (self.gamma + self.llambda + 2 * self.tau) * sp.eye(r)
        else:
            A = UtU + (self.gamma + self.llambda + self.tau) * sp.eye(r)
        # B = Ut.T @ Yt + self.gamma * Ut[idx, :].T + self.tau * (Wp.T + Wn.T) # rxb
        B = Yt @ Ut + self.gamma * Ut[idx, :] + self.tau * (Wp + Wn) # bxr
        return np.linalg.lstsq(A, B.T, rcond=None)[0].T
    
    def step(self, corpus, indices):
        """
          Does an iteration of method. Runs a loop for all T.
        """
        for t in self.times:
            Yt = corpus.SPPMI[t]

            for j in range(len(indices)):
                idx = indices[j]
                Yb = Yt[idx, :].todense()

                if t > 0:
                    Wp = self.Ws[t-1][idx, :]
                    Up = self.embeddings[t-1][idx, :]
                else:
                    Wp = np.zeros((len(idx), self.rank))
                    Up = np.zeros((len(idx), self.rank))

                if t < len(self.times)-1:
                    Wn = self.Ws[t+1][idx, :]
                    Un = self.embeddings[t+1][idx, :]
                else:
                    Wn = np.zeros((len(idx), self.rank))
                    Un = np.zeros((len(idx), self.rank))

                self.Ws[t][idx, :] = self._step(t, len(self.times), Yb, self.embeddings[t], Wp, Wn, idx)
                self.embeddings[t][idx, :] = self._step(t, len(self.times), Yb, self.Ws[t], Up, Un, idx)
  
    def fit(self, corpus, init = None, iters = 5, block_size = None):
        self.vocab = corpus.vocab
        self.inv_vocab = {val: key for key, val in self.vocab.items()}
        self.times = range(corpus.times)
        if not init:
            self.embeddings = [sio.loadmat('/Volumes/M P/data/emb_static.mat')['emb'] for _ in range(corpus.times)]
        else:
            if isinstance(init, list):
                self.embeddings = init
            else:
                self.embeddings = [init for _ in range(corpus.times)]
        self.Ws = self.embeddings.copy()
        
        if not block_size:
            block_size = corpus.size
            
        if block_size < corpus.size:
            b_ind = self._get_batches(corpus.size, block_size)
        else:
            b_ind = [range(corpus.size)]
        for iter_n in range(iters):
            prev = self.embeddings.copy()
            self.step(corpus, b_ind)
            # if np.linalg.norm([x - y for x, y in zip(self.embeddings, prev)]) < self.threshold:
            #      break
            np.random.permutation(self.times)
      
    def k_nearest(self, word, k = 5, T = None):
        idx = self.vocab[word]
        v = self.embeddings[idx]
        if T:
            cosin = cosine_similarity(v.reshape(1, -1), self.embeddings[T]).reshape(-1)
            arg = np.argsort(cosin)[-top_n - 1 : -1][::-1]
            neighbors = [(self.inv_vocab[ind], cosin[ind]) for ind in arg]
            return neighbors
        neighbors = [self.k_nearest(word, k = k, T = t) for t in range(len(self.embeddings))]
        return neighbors
  
    def embedding(self, word, T = None):
        if T:
            idx = self.vocab[word]
            return self.embeddings[T][idx]
        return [self.embeddings[t][idx] for t in range(len(self.embeddings))]