from recommenders.recommender import Recommender
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
import scipy.sparse as sps
import numpy as np
import configparser 

config = configparser.ConfigParser()
config.read('config.ini')


class PureSVD(Recommender):
  
    NAME = "PureSVD"

    def __init__(self, urm):
        super(PureSVD, self).__init__(urm = urm)
        self.SEED = int(config['DEFAULT']['SEED'])


    def fit(self, n_factors=10):

        U, Sigma, QT = randomized_svd(self.urm,
                                      n_components=n_factors,
                                      n_iter=5,
                                      random_state=self.SEED)
        U_s = U * sps.diags(Sigma)

        self.user_factors = U_s
        self.item_factors = QT.T

        self.r_hat = self.user_factors.dot(self.item_factors.T)

        print(type(self.r_hat))
        self.r_hat = normalize(self.r_hat, norm='l2', axis=1)
        print(self.r_hat)
       
    def tuning(self):
        pass
    

