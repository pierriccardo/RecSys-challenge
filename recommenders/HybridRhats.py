from recommenders.recommender import Recommender
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import similaripy as sim

#| best results CB + CF | topk: 65 | alpha: 0.09 | MAP: 0.0651 |

class HybridRhats(Recommender):

    NAME = 'HybridRhats'

    def __init__(self, urm, recs: list):

        self.recs = recs
        self.rnd_values = np.random.rand(len(self.recs))
        print(self.rnd_values)

        super(HybridRhat, self).__init__(urm)
       

    def fit(self, topK=100, alpha = 0.5):

        pass
    
    def _compute_items_scores(self, user):

        scores = None
        for r, v in zip(self.recs, self.rnd_values):
            if scores is None:
                scores = r.r_hat[user].toarray().ravel()
            else:
                scores = scores + r.r_hat[user].toarray().ravel() * v
    
        return scores


    def tuning(self, urm_valid):
        
        BEST_MAP = 0.0
        BEST_TOPK = 0
        BEST_ALPHA = 0

        topKs = np.arange(10, 510, 10)
        alphas = np.arange(0.05, 0.96, 0.05)
        total = len(topKs) * len(alphas)

        i = 0
        for t in topKs:
            for a in alphas:
                self.fit(t, a)
                self._evaluate(urm_valid)

                log = '| iter: {:-5d}/{} | topk: {:-3d} | alpha: {:.3f} | MAP: {:.4f} |'
                print(log.format(i, total, t, a, self.MAP))

                i+=1
                if self.MAP > BEST_MAP:

                    BEST_TOPK = t
                    BEST_ALPHA = a
                    BEST_MAP = self.MAP
            
        print('| best results | topk: {} | alpha: {} | MAP: {:.4f} |'.format(
            BEST_TOPK, BEST_ALPHA, BEST_MAP
        ))

        log = '| {} | topk: {:-3d} | alpha: {:.3f} | MAP: {:.4f} |'
        print(log.format(self.NAME, BEST_TOPK, BEST_ALPHA, BEST_MAP))
