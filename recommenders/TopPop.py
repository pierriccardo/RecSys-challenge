from recommenders.recommender import Recommender
from similarity.similarity import similarity
from sklearn.preprocessing import normalize
import numpy as np
import similaripy as sim
import scipy
import configparser
import sys

#| best results | topk: 245 | shrink: 120 | sim type: cosine | MAP: 0.0471 |

class TopPop(Recommender):

    NAME = 'TopPop'

    def __init__(self, urm):

        super().__init__(urm = urm)  

    def fit(self):

        item_popularity = np.ediff1d(self.urm.tocsc().indptr)

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis = 0)
    
    
    def recommend(self, user_id, at=10):
    
        recommended_items = self.popular_items[0:at]

        s = self.urm.indptr[user_id]
        e = self.urm.indptr[user_id + 1]
        
        seen = self.urm.indices[s:e]

        return recommended_items, len(seen)

