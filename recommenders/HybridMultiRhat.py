from recommenders.recommender import Recommender    
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import similaripy as sim
import sys
import configparser
from tqdm import tqdm

class HybridMultiRhat(Recommender):

    NAME = 'HMR' 

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


    def fit(self, vec, norm='none'):
        self.vec = vec
        self.norm = norm

        '''
        first = True
        for alpha, rec in zip(vec, self.recs):

            if norm != 'none':
                rec.r_hat = normalize(rec.r_hat, norm=norm, axis=1)
            
            if first:
                self.r_hat = rec.r_hat * alpha
                first = False
            else:
                self.r_hat = self.r_hat + alpha * rec.r_hat
        '''
    '''
    def _compute_items_scores(self, user):

        if isinstance(self.r_hat, sps.csc_matrix) or isinstance(self.r_hat, sps.csr_matrix):
            scores = self.r_hat[user].toarray().ravel()
        else:
            scores = np.array(self.r_hat[user]).flatten()
        return scores  
    '''

    def _compute_items_scores(self, user):

        scores_vec = []
        scores = np.zeros(25975)
        for alpha, rec in zip(self.vec, self.recs):

            v = rec._compute_items_scores(user)
            item_scores = normalize(v[:,np.newaxis], axis=0).ravel()
            #item_scores = normalize(rec._compute_items_scores(user), axis=0)
            scores = scores +  item_scores * alpha

        return scores
        

    def tuning(self, urm_valid):
        
        BEST_MAP = 0.0
        BEST_VEC = []
        BEST_NORM = ''

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')

        np.random.seed(int(cp['tuning.HybridMultiRhat']['seed']))
        
        iterations  = int(cp.get('tuning.HybridMultiRhat', 'iterations'))
        norms       = cp.getlist('tuning.HybridMultiRhat', 'norm')

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
