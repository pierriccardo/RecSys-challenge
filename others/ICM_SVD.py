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

    def __init__(self, urm):
        
        super().__init__(urm = urm)
        
        #self.ICM = ICM

    def fit(self, n_factors = 30, topK = 150):
        self.n_factors = n_factors
        self.topK = topK
        self.S_ICM_SVD = self.get_S_urm_SVD(self.urm, n_factors=self.n_factors, topK=self.topK)

        self.r_hat = self.urm.dot(self.S_ICM_SVD)

    
    def get_S_urm_SVD(self, urm, n_factors, topK):
        print('Computing S _urm_SVD...')

        S_matrix_list = []

        urm = sps.csr_matrix(urm, dtype=float)

        u, s, vt = svds(urm, k=n_factors)
        v = vt.T

        for i in tqdm(range(0, v.shape[0])):
            S_row = v[i, :].dot(vt)
            r = S_row.argsort()[:-topK]
            S_row[r] = 0
            S_row_sparse = sps.csr_matrix(S_row)
            sps.csr_matrix.eliminate_zeros(S_row_sparse)
            S_matrix_list.append(S_row_sparse)

        S = sps.vstack(S_matrix_list)
        S.setdiag(0)

        return S