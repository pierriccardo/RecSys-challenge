from recommenders.recommender import Recommender
from similarity.similarity import similarity
import numpy as np
import similaripy as sim
import scipy
import configparser
import sys
from sklearn.preprocessing import normalize


class UserKNNCB(Recommender):

    NAME = 'UserKNNCB'

    def __init__(self, urm, icm):

        super().__init__(urm = urm)

        self.urm = urm
        self.icm = icm       

    def fit(self, topK=80, shrink=40, sim_type='splus'):

        self.topK = topK
        self.shrink = shrink

        self.ucm = self.urm.dot(self.icm)
        m = similarity(self.ucm.T, k=topK, sim_type=sim_type, shrink=shrink)
        m = self._check_matrix(m, format='csr')

        self.sim_matrix = normalize(m, norm='l2', axis=0)
        
        self.r_hat = self.sim_matrix.dot(self.urm)

    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_TOPK = 0
        BEST_SHRINK = 0
        BEST_SIM = ''

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')
        
        t = cp.getlist('tuning.UserKNNCB', 'topKs') 
        s = cp.getlist('tuning.UserKNNCB', 'shrinks')
        similarities = cp.getlist('tuning.UserKNNCB', 'similarities')

        topKs   = np.arange(int(t[0]), int(t[1]), int(t[2]))
        shrinks = np.arange(int(s[0]), int(s[1]), int(s[2]))

        total = len(topKs) * len(shrinks) * len(similarities)

        i = 0
        for sim in similarities:
            for t in topKs:
                for s in shrinks:
                    self.fit(topK=t, shrink=s, sim_type=sim)

                    self._evaluate(urm_valid)

                    m = '|{}| iter: {:-5d}/{} | topk: {:-3d} | shrink: {:-3d} | sim type: {} | MAP: {:.4f} |'
                    print(m.format(self.NAME, i, total, t, s, sim, self.MAP))
                    sys.stdout.flush()
                    i+=1

                    if self.MAP > BEST_MAP:

                        BEST_TOPK = t
                        BEST_SHRINK = s
                        BEST_MAP = self.MAP
                        BEST_SIM = sim
                
        m = '|{}| best results | topk: {:-3d} | shrink: {:-3d} | sim type: {} | MAP: {:.4f} |'
        print(m.format(self.NAME, BEST_TOPK, BEST_SHRINK, BEST_SIM, BEST_MAP))


