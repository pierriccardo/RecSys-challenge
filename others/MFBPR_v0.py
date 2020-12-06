

import sys
sys.path.append('/home/riccardo/Documents/RecSys-challenge')
import numpy as np
import numba
import time
from tqdm import tqdm
import scipy.sparse as sps
from recommenders.recommender import Recommender

"""
Matrix Factorization using Bayesian Personalized Ranking

references:
- https://github.com/tmscarla/recsys-toolbox
- https://arxiv.org/pdf/1205.2618.pdf
"""
class MFBPR(Recommender):

    NAME = 'MFBPR'

    def __init__(self, urm, epochs=10, n_factors=700, lr=1e-4, reg=1e-5, user_reg=0.001, pos_item_reg=0.001, neg_item_reg=0.001):

        super(MFBPR, self).__init__(urm = urm)
        
        # hyperparams
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg 
        self.epochs = epochs

        self.user_reg = user_reg
        self.pos_item_reg = pos_item_reg
        self.neg_item_reg = neg_item_reg

        self.n_users, self.n_items = self.urm.shape
        self.user_factors = np.random.random((self.n_users, n_factors))
        self.item_factors = np.random.random((self.n_items, n_factors))

    def _sample_interaction(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

        userSeenItems = []

        while (len(userSeenItems) == 0):
            
            user_id = np.random.choice(self.n_users)

            # Get user seen items and choose one
            userSeenItems = self.urm[user_id, :].indices
            
        pos_item_id = np.random.choice(userSeenItems)
        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def _one_epoch_iteration(self):  

        n_interactions = int(self.urm.nnz)

        for it in tqdm(range(n_interactions)):

            u, i, j = self._sample_interaction()
            self._update_factors(u, i, j)

    def _update_factors(self, u, i, j, update_i=True, update_j=True, update_u=True):

        x = np.dot(self.user_factors[u, :], self.item_factors[i, :] - self.item_factors[j, :])

        z = 1.0 / (1.0 + np.exp(x))

        if update_u:
            d = (self.item_factors[i, :] - self.item_factors[j, :]) * z \
                - self.user_reg * self.user_factors[u, :]
            self.user_factors[u, :] += self.lr * d
        if update_i:
            d = self.user_factors[u, :] * z - self.pos_item_reg * self.item_factors[i, :]
            self.item_factors[i, :] += self.lr * d
        if update_j:
            d = -self.user_factors[u, :] * z - self.neg_item_reg * self.item_factors[j, :]
            self.item_factors[j, :] += self.lr * d

    def fit(self):
        print('Fitting MFBPR...')

        for epoch in range(self.epochs):
            print('Epoch:', epoch)
            self._one_epoch_iteration()
        print('computing R...')
        self.r_hat = self.user_factors.dot(self.item_factors.T)
     
        print('computing R... DONE')

    def tuning(self):
        pass


from dataset import Dataset
from evaluator import Evaluator

if __name__ == '__main__':

    d = Dataset(split=0.8)
    r = MFBPR(d.get_URM_train())
    r.fit()
    e = Evaluator(r, d.get_URM_valid())
    e.results()

