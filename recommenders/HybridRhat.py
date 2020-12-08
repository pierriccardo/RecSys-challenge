from recommenders.recommender import Recommender    
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import similaripy as sim

#| best results CB + CF | topk: 65 | alpha: 0.09 | MAP: 0.0651 |

class HybridRhat(Recommender):

    NAME = 'HybridRhat' 

    def __init__(self, urm, r1, r2):

        self.r1 = normalize(r1.r_hat, norm='l2', axis=1)
        self.r2 = normalize(r2.r_hat, norm='l2', axis=1)

        self.urm = urm       

        self.NAME = '{}({}, {})'.format(self.NAME, r1.NAME, r2.NAME) 

    def fit(self, alpha = 0.5):
        
        self.alpha = alpha
        self.r_hat = self.r1 * self.alpha + self.r2 * (1 - self.alpha)

    def tuning(self, urm_valid):
        
        BEST_MAP = 0.0
        BEST_ALPHA = 0

        alphas = np.arange(0.05, 0.95, 0.05)
        total = len(alphas)

        i = 0
        for a in alphas:
            self.fit(a)
            self._evaluate(urm_valid)

            log = '|{}| iter: {:-5d}/{} | alpha: {:.3f} | MAP: {:.4f} |'
            print(log.format(self.NAME, i, total, a, self.MAP))

            i+=1
            if self.MAP > BEST_MAP:

                BEST_ALPHA = a
                BEST_MAP = self.MAP
            
        log = '|{}| best results | alpha: {:.3f} | MAP: {:.4f} |'
        print(log.format(self.NAME, BEST_ALPHA, BEST_MAP))
