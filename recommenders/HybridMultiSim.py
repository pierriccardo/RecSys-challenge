from recommenders.recommender import Recommender    
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import similaripy as sim
import sys
import configparser
from tqdm import tqdm

class HybridMultiSim(Recommender):

    NAME = 'HMS' 

    def __init__(self, urm, recs):

        self.recs = recs

        self.urm = urm
        
        names = ''
        first = True
        for r in recs:
            if first: 
                names = names + r.NAME
                first = False
            else: 
                names = names + ',' + r.NAME
        
        self.NAME = '{}({})'.format(self.NAME, names)


    def fit(self, vec, norm='l2'):
        
        if norm!='none':
            for r in self.recs:
                r.sim_matrix = normalize(r.sim_matrix, norm=norm, axis=1)

        first = True
        for alpha, rec in zip(vec, self.recs):
            if first:
                self.sim_matrix = rec.sim_matrix * alpha
                first = False
            else:
                self.sim_matrix = self.sim_matrix + alpha * rec.sim_matrix
        self.r_hat = self.urm.dot(self.sim_matrix)

    def _compute_items_scores(self, user):

        if isinstance(self.r_hat, sps.csc_matrix) or isinstance(self.r_hat, sps.csr_matrix):
            scores = self.r_hat[user].toarray().ravel()
        else:
            scores = np.array(self.r_hat[user]).flatten()
        return scores  
            
        

    def tuning(self, urm_valid):
        
        BEST_MAP = 0.0
        BEST_VEC = []
        BEST_NORM = ''

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')

        np.random.seed(int(cp['tuning.HybridMultiSim']['seed']))
        
        iterations  = int(cp.get('tuning.HybridMultiSim', 'iterations'))
        norms       = cp.getlist('tuning.HybridMultiSim', 'norm')

        total = iterations * len(norms)

        print(self.NAME)

        for n in norms:
            for i in range(iterations):

                vec = np.random.dirichlet(np.ones(len(self.recs)),size=1)[0]
                self.fit(vec, norm=n)
                self._evaluate(urm_valid)

                log = '|iter: {:-5d}/{} | vec: {} | norm: {} | MAP: {:.4f} |'
                print(log.format(i, total, vec, n, self.MAP))
                sys.stdout.flush()
                i+=1

                if self.MAP > BEST_MAP:

                    BEST_VEC = vec
                    BEST_NORM = n
                    BEST_MAP = self.MAP
            
        log = '|{}| best results | vec: {} | norm: {} | MAP: {:.4f} |'
        print(log.format(self.NAME, BEST_VEC, BEST_NORM, BEST_MAP))
