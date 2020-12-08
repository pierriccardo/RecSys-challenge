from recommenders.recommender import Recommender
from similarity.similarity import similarity
import numpy as np
import similaripy as sim
import scipy

# best params found
# | topK = 300 | shrink = 100 | MAP = 0.0487 |
#| best results | topk: 245 | shrink: 120 | sim type: cosine | MAP: 0.0471 |

class ItemKNNCF(Recommender):

    NAME = 'ItemKNNCF'

    def __init__(self, urm, saverhat=False):

        super().__init__(urm = urm)

        self.saverhat = saverhat

    
    def fit(self, topK=50, shrink=100, sim_type='cosine'):

        self.topK = topK
        self.shrink = shrink

        self.sim_matrix = similarity(self.urm, k=topK, sim_type=sim_type, shrink=shrink)
        #self.sim_matrix = sim.normalization.tfidf(self.sim_matrix)
        self.sim_matrix = self._check_matrix(self.sim_matrix, format='csr')

        self.r_hat = self.urm.dot(self.sim_matrix)
        #self.r_hat = sim.normalization.tfidf(self.r_hat)

        self.r_hat = self.r_hat.toarray()   

        if self.saverhat: 
            self.save_r_hat()
        print('name: {}, r_hat => type {}  shape {} '.format(self.NAME, type(self.r_hat), self.r_hat.shape))     
      

     

    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_TOPK = 0
        BEST_SHRINK = 0
        BEST_SIM = ''

        topKs = np.arange(20, 700, 15)
        shrinks = np.arange(0, 1000, 15)
        similarities = ['cosine', 'jaccard']

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


