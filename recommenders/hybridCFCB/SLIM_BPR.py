import time
import numpy as np
import scipy.sparse as sps
from recommenders.recommender import Recommender
from sklearn.preprocessing import normalize
import math


class SLIM_BPR(Recommender):

    NAME = 'SLIM_BPR'

    def __init__(self, urm ):
        super().__init__(urm = urm)

        urm_coo = self.urm.tocoo()

        self.interactions = []
        for u, i in zip(urm_coo.row, urm_coo.col):
          self.interactions.append((u, i))

        self.user_seen_items = {}
        for u in range(0, urm.shape[0]):
          self.user_seen_items[u] = urm.indices[urm.indptr[u]:urm.indptr[u+1]]
        self.user_seen_items[0]

    def fit(self, topK=300, epochs=150, lambda_i=0.1, lambda_j=0.01, lr=0.0005):
        """
        :param topK:
        :param epochs:
        :param lambda_i:
        :param lambda_j:
        :param lr:
        :return:
        """

        # Initialize similarity with zero values
        self.item_item_S = np.zeros((self.n_items, self.n_items), dtype = np.float)

        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.lr = lr

        stt = time.time()

        print('|{}| training |'.format(self.NAME))

        for n_epoch in range(epochs):
            ts = time.time()
            self._run_epoch()
            print("| epoch: {} | time: {:.2f} |".format(n_epoch+1, time.time() - ts))

        print("Train completed in {:.2f} minutes".format(time.time() - stt))

        self.W_sparse = self._similarity_matrix_topk(self.item_item_S, k=topK)
        m = sps.csr_matrix(self.W_sparse)
        self.sim_matrix = normalize(m, norm='l2', axis=1)
        self.r_hat = self.urm.dot(self.sim_matrix)       


    def _run_epoch(self):

        # Uniform user sampling without replacement
        for sample_num in range(self.n_users):

            u, i, j = self._sample_triplet()
            
            # user seen items
            usi = self.user_seen_items[u]
            
            # Compute positive and negative item predictions. Assuming implicit interactions.
            x_ui = self.item_item_S[i, usi].sum()
            x_uj = self.item_item_S[j, usi].sum()

            # Gradient
            x_uij = x_ui - x_uj
            sigmoid_gradient = 1 / (1 + np.exp(x_uij))

            # Update
            self.item_item_S[i, usi] += self.lr * (sigmoid_gradient - self.lambda_i * self.item_item_S[i, usi])
            self.item_item_S[i, i] = 0

            self.item_item_S[j, usi] -= self.lr * (sigmoid_gradient - self.lambda_j * self.item_item_S[j, usi])
            self.item_item_S[j, j] = 0


    def _sample_triplet(self):

        interaction = np.random.randint(0, self.urm.nnz)

        u, i = self.interactions[interaction]
        j = np.random.randint(0, self.n_items)
        while j in self.user_seen_items[u]:
            j = np.random.randint(0, self.n_items)

        return u, i, j
    
    
    def tuning(self):

        pass
 

