from recommenders.recommender import Recommender
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
import scipy.sparse as sps
import numpy as np
import configparser 
import sys

config = configparser.ConfigParser()
config.read('config.ini')


class PureSVD(Recommender):
  
    NAME = "PureSVD"

    def __init__(self, urm):
        super(PureSVD, self).__init__(urm = urm)
        self.SEED = int(config['DEFAULT']['SEED'])


    def fit(self, n_factors=290, n_iter=5):

        U, Sigma, QT = randomized_svd(self.urm,
                                      n_components=n_factors,
                                      n_iter=n_iter,
                                      random_state=self.SEED)
        U_s = U * sps.diags(Sigma)

        self.user_factors = U_s
        self.item_factors = QT.T

        self.r_hat = self.user_factors.dot(self.item_factors.T)
  
    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_N_FACTORS = 0
        BEST_N_ITER = 0

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')
        
        t = cp.getlist('tuning.PureSVD', 'n_factors')
        iters = cp.getlist('tuning.PureSVD', 'n_iter')

        n_factors   = np.arange(int(t[0]), int(t[1]), int(t[2]))
        n_iters     = np.arange(int(iters[0]), int(iters[1]), int(iters[2]))

        total = len(n_factors) * len(n_iters)

        i = 0
        for nf in n_factors:
            for ni in n_iters:
                self.fit(n_factors=nf, n_iter=ni)

                self._evaluate(urm_valid)

                m = '|{}| iter: {:-5d}/{} | n factors: {:-3d} | n iter: {:-3d} | MAP: {:.4f} |'
                print(m.format(self.NAME, i, total, nf, ni, self.MAP))
                sys.stdout.flush()
                i+=1

                if self.MAP > BEST_MAP:

                    BEST_N_FACTORS = nf
                    BEST_N_ITER = ni
                    BEST_MAP = self.MAP
                
        m = '|{}| best results | n factors: {:-3d} | n iter: {:-3d} | MAP: {:.4f} |'
        print(m.format(self.NAME, BEST_N_FACTORS, BEST_N_ITER, BEST_MAP))
        


