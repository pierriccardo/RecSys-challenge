from recommenders.recommender import Recommender
import numpy as np
from tqdm import tqdm
from Cython.Build import cythonize
from recommenders.cython_recommenders.FunkSVD_fast import fit 
from scipy import sparse


class FunkSVD(Recommender):

    NAME = 'FunkSVD'

    def __init__(self, urm):

        super().__init__(urm = urm)
        self.urm = urm
        
    def fit(self, epochs=100, sample_per_epoch=100000, n_factors=10, lr=1e-4, regularization=1e-5):

        self.user_factors = np.random.random((self.n_users, n_factors)) 
        self.item_factors = np.random.random((self.n_items, n_factors))  

        urm_coo = self.urm.tocoo()

        for epoch in range(epochs):

            loss = 0.0

            for sample_num in tqdm(range(sample_per_epoch)):
                
                # Randomly pick sample
                sample_index = np.random.randint(urm_coo.nnz)

                user_id = urm_coo.row[sample_index]
                item_id = urm_coo.col[sample_index]
                rating  = urm_coo.data[sample_index]

                # Compute prediction
                predicted_rating = np.dot(self.user_factors[user_id,:], self.item_factors[item_id,:])

                print('self.user_factors[user_id,:]' + str(self.user_factors[user_id,:])) 
                print('self.item_factors[item_id,:]' + str(self.item_factors[item_id,:]))
                print('predicted_rating' + str(predicted_rating))
                print(rating)
                    
                # Compute prediction error, or gradient
                prediction_error = rating - predicted_rating
                loss += prediction_error**2
                
                # Copy original value to avoid messing up the updates
                H_i = self.item_factors[item_id,:]
                W_u = self.user_factors[user_id,:]  
                
                user_factors_update = prediction_error * H_i - regularization * W_u
                item_factors_update = prediction_error * W_u - regularization * H_i
                
                self.user_factors[user_id,:] += lr * user_factors_update 
                self.item_factors[item_id,:] += lr * item_factors_update    
                
            print("|Epoch: {} | loss: {:.2f} |".format(epoch, loss/(sample_num+1)))

        self.r_hat = self.user_factors.dot(self.item_factors.T)

                    
    def fit_cython(self, epochs=100, sample_per_epoch=100000, n_factors=10, lr=1e-4, regularization=1e-5):

        self.user_factors, self.item_factors = fit(self.urm, epochs, sample_per_epoch, n_factors, lr, regularization)
        
        self.r_hat = self.user_factors.dot(self.item_factors.T)

    def tuning(self):

        pass