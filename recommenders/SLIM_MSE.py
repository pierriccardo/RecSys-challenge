import numpy as np
import time
from scipy import sparse 
from recommenders.recommender import Recommender
from tqdm import tqdm
from Cython.Build import cythonize
from recommenders.cython_recommenders.SLIM_MSE_fast import train 
import similaripy as sim 

"""
Best params:
lr 1e-4
epoch 65
samples = 135000
"""
class SLIM_MSE(Recommender):

    NAME = 'SLIM_MSE'
    
    def __init__(self, urm):

        super().__init__(urm = urm)
        self.urm = urm


    def fit(self, learning_rate=1e-4, epochs=65, samples=135000, use_cython=True):

        print('training SLIM_MSE...')

        if use_cython:
            try:
                self.sim_matrix = train(self.urm, learning_rate, epochs, samples)
                self.r_hat = self.urm.dot(self.sim_matrix)
            except: 'unable to train with cython'
        else:
            urm_coo = self.urm.tocoo()
            n_items = self.urm.shape[1]
            n_interactions = self.urm.nnz

            item_item_S = np.zeros((n_items, n_items), dtype = np.float16)

            for n_epoch in range(epochs):

                loss = 0.0
                start_time = time.time()

                for sample_num in tqdm(range(samples)):

                    # Randomly pick sample
                    sample_index = np.random.randint(urm_coo.nnz)

                    user_id = urm_coo.row[sample_index]
                    item_id = urm_coo.col[sample_index]
                    true_rating = urm_coo.data[sample_index]

                    # Compute prediction
                    predicted_rating = self.urm[user_id].dot(item_item_S[:,item_id])[0]
                        
                    # Compute prediction error, or gradient
                    prediction_error = true_rating - predicted_rating
                    loss += prediction_error**2
                    
                    # Update model, in this case the similarity
                    items_in_user_profile = self.urm.indices[self.urm.indptr[user_id]:self.urm.indptr[user_id+1]]
                    ratings_in_user_profile = self.urm.data[self.urm.indptr[user_id]:self.urm.indptr[user_id+1]]
                    item_item_S[items_in_user_profile,item_id] += learning_rate * prediction_error * ratings_in_user_profile

                    # Print some stats
                    if (sample_num +1)% 5000 == 0:
                        elapsed_time = time.time() - start_time
                        samples_per_second = (sample_num+1)/elapsed_time
                        print("Iteration {} in {:.2f} seconds, loss is {:.2f}. Samples per second {:.2f}".format(sample_num+1, elapsed_time, loss/(sample_num+1), samples_per_second))


                elapsed_time = time.time() - start_time
                samples_per_second = (sample_num+1)/elapsed_time

                print("Epoch {} complete in in {:.2f} seconds, loss is {:.3E}. Samples per second {:.2f}".format(n_epoch+1, time.time() - start_time, loss/(sample_num+1), samples_per_second))
        
            self.sim_matrix = sparse.csr_matrix(sparse.coo_matrix(item_item_S))
           
            self.r_hat = self.urm.dot(self.sim_matrix)
        

    def tuning(self, urm_valid):
        
        ts = time.time()

        BEST_MAP        = 0.0
        BEST_LR         = 0
        BEST_EPOCHS     = 0
        BEST_SAMPLES    = 0

        LR      = [1e-3, 1e-4, 1e-5]
        EPOCHS  = np.arange(40, 300, 10)
        SAMPLES = np.arange(100000, 500000, 50000)

        total = len(LR) * len(EPOCHS) * len(SAMPLES)

        i = 0
        for lr in LR:
            for e in EPOCHS:
                for s in SAMPLES:
                    self.fit(lr, e, s)
                    self._evaluate(urm_valid)

                    print('| iter: {}/{} | lr: {} | epochs: {} | samples: {} | MAP: {:.4f} | time: {} |'.format(
                        i, total, lr, e, s, self.MAP, time.time() - ts)
                        )

                    i+=1

                    if self.MAP > BEST_MAP:

                        BEST_MAP        = self.MAP
                        BEST_LR         = lr
                        BEST_EPOCHS     = e
                        BEST_SAMPLES    = s
                
        print('| best results | lr: {} | epochs: {} | samples: {} | MAP: {:.4f} |'.format(BEST_LR, BEST_EPOCHS, BEST_SAMPLES, BEST_MAP))


