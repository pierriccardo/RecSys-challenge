from recommenders.recommender import Recommender
from similarity.similarity import similarity
from sklearn import feature_extraction
from sklearn.preprocessing import normalize
import numpy as np 
import similaripy as sim
import configparser
import sys

# | iter: 501/13600 | topk: 70 | shrink: 10 | sim type: cosine | MAP: 0.0319 |
class ItemKNNCB(Recommender):

    NAME = 'ItemKNNCB'

    def __init__(self, urm, icm):

        super().__init__(urm = urm)

        self.icm = icm

    def fit(self, topK=350, shrink=10, sim_type='splus'):

        m = similarity(self.icm.T, k=topK, sim_type=sim_type, shrink=shrink)
        m = self._check_matrix(m, format='csr')
        self.sim_matrix = normalize(m, norm='l2', axis=0)
        
        self.r_hat = self.urm.dot(self.sim_matrix)

    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_TOPK = 0
        BEST_SHRINK = 0
        BEST_SIM = ''

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')
        
        t = cp.getlist('tuning.ItemKNNCB', 'topKs') 
        s = cp.getlist('tuning.ItemKNNCB', 'shrinks')
        similarities = cp.getlist('tuning.ItemKNNCB', 'similarities')

        topKs   = np.arange(int(t[0]), int(t[1]), int(t[2]))
        shrinks = np.arange(int(s[0]), int(s[1]), int(s[2]))

        total = len(topKs) * len(shrinks) * len(similarities)

        i = 0
        for sim in similarities:
            for t in topKs:
                for s in shrinks:
                    self.fit(topK=t, shrink=s, sim_type=sim)

                    self._evaluate(urm_valid)

                    print('|{}| iter: {}/{} | topk: {} | shrink: {} | sim type: {} | MAP: {:.4f} |'.format(self.NAME, i, total, t, s, sim, self.MAP))
                    sys.stdout.flush()
                    i+=1

                    if self.MAP > BEST_MAP:

                        BEST_TOPK = t
                        BEST_SHRINK = s
                        BEST_MAP = self.MAP
                        BEST_SIM = sim
                
        print('|{}| topk: {} | shrink: {} | sim type: {} | MAP: {:.4f} |'.format(self.NAME, BEST_TOPK, BEST_SHRINK, BEST_SIM, BEST_MAP))


