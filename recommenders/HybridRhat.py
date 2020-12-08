import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import similaripy as sim

#| best results CB + CF | topk: 65 | alpha: 0.09 | MAP: 0.0651 |

class HybridRhat:

    NAME = 'HybridRhat'

    def __init__(self, urm, r1, r2):

        self.r1 = r1
        self.r2 = r2
        self.urm = urm
       

    def fit(self, alpha = 0.5):
        
        self.alpha = alpha
        
    
    def _compute_items_scores(self, user):

        s1 = self.r1[user].toarray().ravel() #* self.alpha
        #s2 = self.r2[user].toarray().ravel() #* (1 - self.alpha)

        return s1

    def recommend(self, user: int = None, cutoff: int = 10):
     
        scores = self._compute_items_scores(user)
        scores = self._remove_seen_items(user, scores)
        scores = scores.argsort()[::-1]

        return scores[:cutoff]

    def _remove_seen_items(self, user, scores):

        assert (
            self.urm.getformat() == 'csr'
        ), "_remove_seen_items: urm is not in csr format, actual format is {}".format(type(self.urm))
        
        s = self.urm.indptr[user]
        e = self.urm.indptr[user + 1]
        
        seen = self.urm.indices[s:e]
        scores[seen] = -np.inf

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
