import numpy as np
import scipy.sparse as sps
from recommenders.recommender import Recommender
from sklearn.preprocessing import normalize
import configparser
import similaripy as sim
import sys
from tqdm import tqdm

# best params found
#| topK = 170 | alpha = 0.3  | beta  = 0.07 | MAP = 0.0518 | 

class RP3beta(Recommender):

    NAME = "RP3beta"

    def __init__(self, urm):
        
        super(RP3beta, self).__init__(urm = urm)

    def fit(self, alpha=0.35, beta=0.06, topK=110):

        self.alpha = alpha
        self.beta = beta
        self.topK = topK

        
        self.sim_matrix = sim.rp3beta(
            self.urm.T,
            alpha=alpha,
            beta=beta,
            k=topK)
        self.sim_matrix = normalize(self.sim_matrix, norm='l2', axis=1)
            
        self.r_hat = self.urm.dot(self.sim_matrix)

    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_TOPK = 0
        BEST_ALPHA = 0
        BEST_BETA = 0

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')
        
        t = cp.getlist('tuning.RP3beta', 'topKs') 
        a = cp.getlist('tuning.RP3beta', 'alphas')
        b = cp.getlist('tuning.RP3beta', 'betas')

        topKs   = np.arange(int(t[0]), int(t[1]), int(t[2]))
        alphas = np.arange(float(a[0]), float(a[1]), float(a[2]))
        betas = np.arange(float(b[0]), float(b[1]), float(b[2]))

        total = len(topKs) * len(alphas) * len(betas)

        i = 0
        for t in topKs:
            for a in alphas:
                for b in betas:
                    self.fit(alpha=a, beta=b, topK=t)

                    self._evaluate(urm_valid)

                    log = '| iter: {:-5d}/{} | topk: {:-3d} | alpha: {:.3f} | beta: {:.3f} | MAP: {:.4f} |'
                    print(log.format(i, total, t, a, b, self.MAP))
                    sys.stdout.flush()

                    i+=1

                    if self.MAP > BEST_MAP:

                        BEST_TOPK = t
                        BEST_ALPHA = a
                        BEST_BETA = b
                        BEST_MAP = self.MAP
        log = '| best results | topk: {:-3d} | alpha: {:.3f} | beta: {:.3f} | MAP: {:.4f} |'
        print(log.format(BEST_TOPK, BEST_ALPHA, BEST_BETA, BEST_MAP))
