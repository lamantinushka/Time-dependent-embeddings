# Time-dependent-embeddings
based on 
- Dynamic Word Embeddings for Evolving Semantic Discovery. Yao et al. [https://arxiv.org/pdf/1703.00607.pdf]
- A projector-splitting integrator for dynamical low-rank approximation. Ch. Lubich and I.V. Oseledets [https://arxiv.org/pdf/1301.1058.pdf]
## Data
train data can be obtained from https://www.dropbox.com/s/nifi5nj1oj0fu2i/data.zip?dl=0

data for k-means test https://www.kaggle.com/nzalake52/new-york-times-articles
## Models
### Aligned word2vec
Is a baseline model minimizing difference between Emb[t+1] and Emb[t]
### Dynamic word2vec
Is a model based on Ridge-regrission optimization problem
### Projector-splitting word2vec
Is a model based on the projector-splitting integrator
