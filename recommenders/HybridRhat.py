from recommenders.recommender import Recommender    
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import similaripy as sim
import sys
import configparser

class HybridRhat(Recommender):

    NAME = 'HRhat' 

    def __init__(self, urm, r1, r2):

        self.r1 = r1.r_hat
        self.r2 = r2.r_hat

        self.urm = urm       

        self.NAME = '{}({}, {})'.format(self.NAME, r1.NAME, r2.NAME) 

    def fit(self, alpha = 0.5, norm='l2'):
        
        if norm!='none':
            self.r1 = normalize(self.r1, norm=norm, axis=1)
            self.r2 = normalize(self.r2, norm=norm, axis=1)
        
        self.alpha = alpha
        self.r_hat = self.r1 * self.alpha + self.r2 * (1 - self.alpha)
    
    def _compute_items_scores(self, user):
        
        if isinstance(self.r_hat, sps.csc_matrix):
            scores = self.r_hat[user].toarray().ravel()
        else:
            scores = self.r_hat[user]
        return scores

    def tuning(self, urm_valid):
        
        BEST_MAP = 0.0
        BEST_ALPHA = 0
        BEST_NORM = ''

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')
        
        a = cp.getlist('tuning.HybridRhat', 'alphas')
        norms = cp.getlist('tuning.HybridRhat', 'norm')
        print(norms)

        alphas = np.arange(float(a[0]), float(a[1]), float(a[2]))

        alphas = np.arange(0.05, 0.95, 0.05)
        total = len(alphas)

        i = 0
        for a in alphas:
            for n in norms:
                self.fit(alpha=a, norm=n)
                self._evaluate(urm_valid)

                log = '|{}| iter: {:-5d}/{} | alpha: {:.3f} | norm: {} | MAP: {:.4f} |'
                print(log.format(self.NAME, i, total, a, n, self.MAP))
                sys.stdout.flush()

                i+=1
                if self.MAP > BEST_MAP:

                    BEST_ALPHA = a
                    BEST_NORM = n
                    BEST_MAP = self.MAP
            
        log = '|{}| best results | alpha: {:.3f} | norm: {} | MAP: {:.4f} |'
        print(log.format(self.NAME, BEST_ALPHA, BEST_NORM, BEST_MAP))
