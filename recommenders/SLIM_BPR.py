import time
import numpy as np
import scipy.sparse as sps
from recommenders.recommender import Recommender


class SLIM_BPR(Recommender):

    NAME = 'SLIM_BPR'

    def __init__(self, urm ):
        super().__init__(urm = urm)


    def fit(self, topK=200, epochs=250, lambda_i=0.075, lambda_j=0.0075, lr=0.0005):
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

        for n_epoch in range(epochs):
            ts = time.time()
            self._run_epoch(n_epoch)
            print("| epoch: {} | time: {:.2f} |".format(n_epoch+1, time.time() - ts))

        print("Train completed in {:.2f} minutes".format(time.time() - stt))

        self.W_sparse = self._similarity_matrix_topk(self.item_item_S, k=topK)
        self.sim_matrix = sps.csr_matrix(self.W_sparse)
        self.r_hat = self.urm.dot(self.sim_matrix)
        #self.r_hat = self.r_hat.toarray()
        


    def _run_epoch(self, n_epoch):

        # Uniform user sampling without replacement
        for sample_num in range(self.n_users):

            user_id, pos_item_id, neg_item_id = self._sample_triplet()

            # Calculate current predicted score
            user_seen_items = self.urm.indices[self.urm.indptr[user_id]:self.urm.indptr[user_id+1]]

            # Compute positive and negative item predictions. Assuming implicit interactions.
            x_ui = self.item_item_S[pos_item_id, user_seen_items].sum()
            x_uj = self.item_item_S[neg_item_id, user_seen_items].sum()

            # Gradient
            x_uij = x_ui - x_uj
            sigmoid_gradient = 1 / (1 + np.exp(x_uij))

            # Update
            self.item_item_S[pos_item_id, user_seen_items] += self.lr * (sigmoid_gradient - self.lambda_i * self.item_item_S[pos_item_id, user_seen_items])
            self.item_item_S[pos_item_id, pos_item_id] = 0

            self.item_item_S[neg_item_id, user_seen_items] -= self.lr * (sigmoid_gradient - self.lambda_j * self.item_item_S[neg_item_id, user_seen_items])
            self.item_item_S[neg_item_id, neg_item_id] = 0


    def _sample_triplet(self):

        non_empty_user = False

        while not non_empty_user:
            user_id = np.random.choice(self.n_users)
            user_seen_items = self.urm.indices[self.urm.indptr[user_id]:self.urm.indptr[user_id + 1]]

            if len(user_seen_items) > 0:
                non_empty_user = True

        pos_item_id = np.random.choice(user_seen_items)

        neg_item_selected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not neg_item_selected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in user_seen_items):
                neg_item_selected = True

        return user_id, pos_item_id, neg_item_id 
    
    
    def tuning(self):

        pass
 

