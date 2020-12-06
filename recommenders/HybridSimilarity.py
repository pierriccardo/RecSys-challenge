from recommenders.recommender import Recommender
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import similaripy as sim

#| best results CB + CF | topk: 65 | alpha: 0.09 | MAP: 0.0651 |

class HybridSimilarity(Recommender):

    NAME = 'HybridSimilarity'

    def __init__(self, urm, sim1, sim2):

        super(HybridSimilarity, self).__init__(urm)

        self.sim1 = self._check_matrix(sim1.copy(), 'csr')
        self.sim2 = self._check_matrix(sim2.copy(), 'csr')

       

    def fit(self, topK=100, alpha = 0.5):

        # hyperparameters
        self.topK = topK
        self.alpha = alpha

        W = self.sim1*self.alpha + self.sim2*(1-self.alpha)
        
        self.sim_matrix = self._similarity_matrix_topk(W, k=self.topK).tocsr()
        self.r_hat = self.urm.dot(self.sim_matrix)
        

    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_TOPK = 0
        BEST_ALPHA = 0

        topKs = np.arange(20, 400, 10)
        alphas = np.arange(0.05, 0.96, 0.05)
        total = len(topKs) * len(alphas)

        i = 0
        for t in topKs:
            for a in alphas:
                self.fit(t, a)
                self._evaluate(urm_valid)

                print('| iter: {}/{} | topk: {} | alpha: {} | MAP: {:.4f} |'.format(
                    i, total, t, a, self.MAP)
                )

                i+=1
                if self.MAP > BEST_MAP:

                    BEST_TOPK = t
                    BEST_ALPHA = a
                    BEST_MAP = self.MAP
            
        print('| best results | topk: {} | alpha: {} | MAP: {:.4f} |'.format(
            BEST_TOPK, BEST_ALPHA, BEST_MAP
        ))


