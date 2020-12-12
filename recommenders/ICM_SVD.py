import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import svds
from recommenders.recommender import Recommender
from tqdm import tqdm

"""
Recommender with SVD: Singular Value Decomposition technique applied to 
the item content matrix. 
    * k: number of latent factors
    * knn: k-nearest-neighbours to evaluate similarity
If is_test is true, return a dataframe ready to be evaluated with the Evaluator class,
otherwise return a dataframe in the submission format.
"""


class ICM_SVD(Recommender):

    NAME = 'ICM_SVD'

    def __init__(self, URM, ICM):
        
        self.URM = URM
        self.ICM = ICM

    def fit(self, n_factors = 5, topK = 50):
        self.n_factors = n_factors
        self.topK = topK
        self.S_ICM_SVD = self.get_S_ICM_SVD(self.ICM, n_factors=self.n_factors, topK=self.topK)

        self.r_hat = self.URM.dot(self.S_ICM_SVD)

    def get_S_ICM_SVD(self, ICM, n_factors, topK):
        print('Computing S_ICM_SVD...')

        u, s, vt = svds(ICM, k=n_factors, which='LM')

        ut = u.T

        s_2_flatten = np.power(s, 2)
        s_2 = np.diagflat(s_2_flatten)
        s_2_csr = sps.csr_matrix(s_2)

        S = u.dot(s_2_csr.dot(ut))

        return S