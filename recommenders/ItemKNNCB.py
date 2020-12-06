from recommenders.recommender import Recommender
from similarity.similarity import similarity
from sklearn import feature_extraction
from sklearn.preprocessing import normalize
import numpy as np 
import similaripy as sim

# | iter: 501/13600 | topk: 70 | shrink: 10 | sim type: cosine | MAP: 0.0319 |
class ItemKNNCB(Recommender):

    NAME = 'ItemKNNCB'

    def __init__(self, urm, icm, saverhat=False):

        super().__init__(urm = urm)

        self.icm = icm
        self.saverhat = saverhat

    def fit(self, topK=50, shrink=100, sim_type='cosine'):

        # hyperparameters 
        self.topK = topK
        self.shrink = shrink

        
        self.sim_matrix = similarity(self.icm.T, k=topK, sim_type=sim_type, shrink=shrink)
        self.sim_matrix = self._check_matrix(self.sim_matrix, format='csr')

        # computing the scores matrix
        self.r_hat = self.urm.dot(self.sim_matrix)
        self.r_hat = self.r_hat.toarray()

        if self.saverhat:
            self.save_r_hat()

        
        

    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_TOPK = 0
        BEST_SHRINK = 0
        BEST_SIM = ''

        topKs = np.arange(20, 200, 10)
        shrinks = np.arange(0, 50, 5)
        similarities = ['jaccard', 'cosine']

        total = len(topKs) * len(shrinks) * len(similarities)

        i = 0
        for sim in similarities:
            for t in topKs:
                for s in shrinks:
                    self.fit(t, s)

                    self._evaluate(urm_valid)

                    print('| iter: {}/{} | topk: {} | shrink: {} | sim type: {} | MAP: {:.4f} |'.format(i, total, t, s, sim, self.MAP))

                    i+=1

                    if self.MAP > BEST_MAP:

                        BEST_TOPK = t
                        BEST_SHRINK = s
                        BEST_MAP = self.MAP
                        BEST_SIM = sim
                
        print('| best results | topk: {} | shrink: {} | sim type: {} | MAP: {:.4f} |'.format(BEST_TOPK, BEST_SHRINK, BEST_SIM, BEST_MAP))


