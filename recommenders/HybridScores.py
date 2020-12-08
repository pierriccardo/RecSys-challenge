from recommenders.recommender import Recommender
import numpy as np


class HybridScores(Recommender):

    NAME = 'HybridScores'

    def __init__(self, urm, recommender1, recommender2):

        super(HybridScores, self).__init__(urm)

        self.recommender1 = recommender1
        self.recommender2 = recommender2
        
        try: n1 = self.recommender1.NAME
        except: n1 = 'unknown'

        try: n2 = self.recommender2.NAME
        except: n2 = 'unknown'

        self.NAME = '{}({},{})'.format(
            self.NAME, n1, n2 
        ) 

    def fit(self, alpha = 0.5):

        # hyperparameters
        self.alpha = alpha 
        #self.r_hat = self.r_hat.toarray()


    def _compute_items_scores(self, user):
        
        item_weights_1 = self.recommender1._compute_items_scores(user)
        item_weights_2 = self.recommender2._compute_items_scores(user)

        item_weights = item_weights_1*self.alpha + item_weights_2*(1-self.alpha)

        return item_weights

    def tuning(self, urm_valid):
        
        
        BEST_MAP = 0.0
        BEST_ALPHA = 0

        #topKs = np.arange(10, 510, 10)
        alphas = np.arange(0.01, 0.1, 0.001)
        total = len(alphas)

        i = 0
        
        for a in alphas:
            self.fit(a)
            #self._compute_items_scores()
            self._evaluate(urm_valid)

            print('| iter: {}/{} | alpha: {} | MAP: {:.4f} |'.format(
                i, total, a, self.MAP)
            )

            i+=1
            if self.MAP > BEST_MAP:

                BEST_ALPHA = a
                BEST_MAP = self.MAP
        
        print('| best results | alpha: {} | MAP: {:.4f} |'.format(
            BEST_ALPHA, BEST_MAP
        ))


