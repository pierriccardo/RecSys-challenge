import numpy as np
from tqdm import tqdm

class Evaluator:

    def __init__(self, recommender, urm_test, cutoff=10):

        self.recommender = recommender
        self.urm_test = urm_test
        self.cutoff = cutoff

        self.cumulative_precision = 0.0
        self.cumulative_recall = 0.0
        self.cumulative_MAP = 0.0
        self.num_eval = 0

        self.clusters = []

        self.clusters_range = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30, 50, 100, 1600]
        for x in range(0, len(self.clusters_range) - 1):
            cluster = {
                'start': self.clusters_range[x],
                'end': self.clusters_range[x+1],
                'pred': [],
                'users': 0
            }
            self.clusters.append(cluster)
        

        self._evaluate_algorithm()

    def get_MAP(self):
        return self.cumulative_MAP

    def precision(self, recommended_items, relevant_items):
        
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        
        precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
        
        return precision_score

    def recall(self, recommended_items, relevant_items):
    
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        
        recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
        
        return recall_score

    def MAP(self, recommended_items, relevant_items):
    
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        
        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        
        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score
    '''
    def _evaluate_algorithm(self):

        pbar = tqdm(range(self.urm_test.shape[0]))
        for user_id in pbar:
            
            pbar.set_description('|{}| evaluating'.format(self.recommender.NAME))
            relevant_items = self.urm_test.indices[self.urm_test.indptr[user_id]:self.urm_test.indptr[user_id+1]]
            
            if len(relevant_items)>0:
                
                recommended_items = self.recommender.recommend(user_id, self.cutoff)
                self.num_eval+=1

                self.cumulative_precision += self.precision(recommended_items, relevant_items)
                self.cumulative_recall += self.recall(recommended_items, relevant_items)
                self.cumulative_MAP += self.MAP(recommended_items, relevant_items)
                
        self.cumulative_precision /= self.num_eval
        self.cumulative_recall /= self.num_eval
        self.cumulative_MAP /= self.num_eval
    '''
    

    def _evaluate_algorithm(self):

        pbar = tqdm(range(self.urm_test.shape[0]))
        for user_id in pbar:
            
            pbar.set_description('|{}| evaluating'.format(self.recommender.NAME))
            relevant_items = self.urm_test.indices[self.urm_test.indptr[user_id]:self.urm_test.indptr[user_id+1]]
            
            if len(relevant_items)>0:
                
                recommended_items, seen = self.recommender.recommend(user_id, self.cutoff)
                self.num_eval+=1

                P = self.precision(recommended_items, relevant_items)
                R = self.recall(recommended_items, relevant_items)
                M = self.MAP(recommended_items, relevant_items)
                
                for c in self.clusters:
                    if seen >= c['start'] and seen < c['end']:
                        c['pred'].append(M)
                        c['users'] += 1

                self.cumulative_precision += P 
                self.cumulative_recall += R
                self.cumulative_MAP += M

        self.cumulative_precision /= self.num_eval
        self.cumulative_recall /= self.num_eval
        self.cumulative_MAP /= self.num_eval  
        

    def results(self, cluster=True):

        try: name = self.recommender.NAME
        except: name = 'Recommender'
        
        print("|{}| precision = {:.4f} | Recall = {:.4f} | MAP = {:.4f} |".format(
            name,
            self.cumulative_precision, 
            self.cumulative_recall, 
            self.cumulative_MAP
            )
        )
        if cluster:
            m = "|range(-----,-----) | users ----- |{:8s}|".format(self.recommender.NAME)
            print(m)
            for c in self.clusters:
                den = 1
                if len(c['pred']) > 0:
                    den = len(c['pred'])
                p = sum(c['pred']) / den

                m = "|range({:5},{:5}) | users {:5} |  {:.4f}  |"
                print(m.format(c['start'], c['end'], c['users'], p))
        
        return self.cumulative_MAP
