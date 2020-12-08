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


        pass

    def _compute_items_scores_round_robin(self, user):
        
        item_weights_1 = self.recommender1._compute_items_scores(user)
        item_weights_2 = self.recommender2._compute_items_scores(user)

        item_weights_1 = self._remove_seen_items(user, item_weights_1)
        item_weights_2 = self._remove_seen_items(user, item_weights_2)

        arr1 = np.full(shape=len(item_weights_1), fill_value=max(item_weights_1))
        arr2 = np.full(shape=len(item_weights_2), fill_value=max(item_weights_2))

        iw1 = np.divide(item_weights_1, arr1)
        iw2 = np.divide(item_weights_2, arr2)

        diw1 = []
        diw2 = []

        for idx, val in enumerate(iw1):
            diw1.append((idx, val))
        
        for idx, val in enumerate(iw2):
            diw2.append((idx, val))

        diws1 = sorted(diw1, key=lambda x: x[1], reverse=True)
        diws2 = sorted(diw2, key=lambda x: x[1], reverse=True)

        dtotal = []
        dtotal.append(diws1[:15])
        dtotal.append(diws2[:15])
        dtotals = sorted(dtotal, key=lambda x: x[1], reverse=True)

        items_to_recommend = []
        for e in dtotals[:10]:
            if e[0] not in items_to_recommend:
                items_to_recommend.append(e[0])
        
        
        return items_to_recommend
