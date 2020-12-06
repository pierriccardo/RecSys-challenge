from recommenders.recommender import Recommender


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


    def _compute_items_scores(self, user):
        
        item_weights_1 = self.recommender1._compute_items_scores(user)
        item_weights_2 = self.recommender2._compute_items_scores(user)

        item_weights = item_weights_1*self.alpha + item_weights_2*(1-self.alpha)

        return item_weights

    def tuning(self):

        pass