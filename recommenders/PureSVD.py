from recommenders.recommender import Recommender
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps
import numpy as np



class PureSVD(Recommender):
  
    NAME = "PureSVD"

    def __init__(self, urm):
        super(PureSVD, self).__init__(urm = urm)


    def fit(self, num_factors=10, random_seed = None):

        U, Sigma, QT = randomized_svd(self.urm,
                                      n_components=num_factors,
                                      n_iter=5,
                                      random_state = random_seed)
        U_s = U * sps.diags(Sigma)

        self.user_factors = U_s
        self.item_factors = QT.T

        self.r_hat = self.user_factors.dot(self.item_factors.T) 

        print('name: {}, r_hat => type {}  shape {} '.format(self.NAME, type(self.r_hat), self.r_hat.shape))
       
    def tuning(self):
        pass
    

