from recommenders.recommender import Recommender
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import similaripy as sim
import sys
import configparser

#| best results CB + CF | topk: 65 | alpha: 0.09 | MAP: 0.0651 |

class HybridSimilarity(Recommender):

    NAME = 'Hsim'

    def __init__(self, urm, r1, r2):

        super(HybridSimilarity, self).__init__(urm)

        self.NAME = 'Hsim({}, {})'.format(r1.NAME, r2.NAME)
        self.r1 = r1
        self.r2 = r2
       
    def fit(self, topK=100, alpha=0.5, norm='none'):

        self.sim1 =  self.r1.sim_matrix if norm == 'none' else normalize(self.r1.sim_matrix, norm=norm, axis=1) 
        self.sim2 =  self.r2.sim_matrix if norm == 'none' else normalize(self.r2.sim_matrix, norm=norm, axis=1) 

        self.sim1 = self._check_matrix(self.sim1, 'csr')
        self.sim2 = self._check_matrix(self.sim2, 'csr')

        # hyperparameters
        self.topK = topK
        self.alpha = alpha

        W = self.sim1*self.alpha + self.sim2*(1-self.alpha)
        
        self.sim_matrix = self._similarity_matrix_topk(W, k=self.topK).tocsr()

        if self.sim_matrix.shape[0] == self.urm.shape[0]: # user-user similarity
            self.r_hat = self.sim_matrix.dot(self.urm)
        else:                                             # item-item similarity
            self.r_hat = self.urm.dot(self.sim_matrix)

    def tuning(self, urm_valid):
        
        BEST_MAP = 0.0
        BEST_TOPK = 0
        BEST_ALPHA = 0
        BEST_NORM = ''

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')
        
        t = cp.getlist('tuning.HybridSimilarity', 'topKs') 
        a = cp.getlist('tuning.HybridSimilarity', 'alphas')
        norms = cp.getlist('tuning.HybridSimilarity', 'norm')

        topKs  = np.arange(int(t[0]), int(t[1]), int(t[2]))
        alphas = np.arange(float(a[0]), float(a[1]), float(a[2]))

        total = len(topKs) * len(alphas) *len(norms)

        i = 0
        for t in topKs:
            for a in alphas:
                for n in norms:
                    self.fit(topK=t, alpha=a, norm=n)
                    self._evaluate(urm_valid)

                    log = '| iter: {:-5d}/{} | topk: {:-3d} | alpha: {:.3f} | norm: {:.3} | MAP: {:.4f} |'
                    print(log.format(i, total, t, a, n, self.MAP))
                    sys.stdout.flush()

                    i+=1
                    if self.MAP > BEST_MAP:

                        BEST_TOPK = t
                        BEST_ALPHA = a
                        BEST_NORM = n
                        BEST_MAP = self.MAP
        

        log = '| {} | topk: {:-3d} | alpha: {:.3f} | norm: {:.3} | MAP: {:.4f} |'
        print(log.format(self.NAME, BEST_TOPK, BEST_ALPHA, BEST_NORM, BEST_MAP))
