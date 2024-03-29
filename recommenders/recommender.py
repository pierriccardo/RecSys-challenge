import abc
import scipy.sparse as sps
import numpy as np
import time
from tqdm import tqdm
import similaripy as sim
from time import strftime, gmtime
from sklearn.preprocessing import normalize
import os

class Recommender(abc.ABC):

    NAME = 'Recommender'

    def __init__(self, urm, norm=True):
 
        self.urm = sim.normalization.bm25(urm) if norm else urm 
        #self.urm = sim.normalization.tfidf(urm) if norm else urm 
        self.n_users, self.n_items = self.urm.shape
        self.r_hat = None # R_HAT is a matrix n° user x n° item 

        self.popitems = self._compute_toppop_items()

    #@abc.abstractmethod
    def fit(self):
        
        pass
    
    #@abc.abstractmethod
    def tuning(self):

        pass

    def _evaluate(self, urm_test, cutoff=10):

        cumulative_MAP = 0
        num_eval = 0
        
        for user_id in tqdm(range(urm_test.shape[0])):
            
            relevant_items = urm_test.indices[urm_test.indptr[user_id]:urm_test.indptr[user_id+1]]
            
            if len(relevant_items)>0:
                
                recommended_items, k = self.recommend(user_id, cutoff)
                num_eval+=1

                cumulative_MAP += self._MAP(recommended_items, relevant_items)
                
        cumulative_MAP /= num_eval
        self.MAP = cumulative_MAP
        #print('|MAP: {}|'.format(cumulative_MAP))
        

    def _MAP(self, recommended_items, relevant_items):
    
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        
        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        
        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score
        
    def _compute_items_scores(self, user):
        
        #if self.r_hat is None:
        #    print('Please fit the recommender first')

        #scores = self.r_hat[user].toarray().ravel()

        if isinstance(self.r_hat, sps.csc_matrix) or isinstance(self.r_hat, sps.csr_matrix):
            scores = self.r_hat[user].toarray().ravel()
        else:
            scores = self.r_hat[user]
        return scores

        #try: 
        #    scores = self.r_hat[user].toarray().ravel()
        # 
        #except: 
        #    scores = np.dot(self.user_factors[user], self.item_factors.T)
            
        return scores

    
    def recommend(self, user: int = None, cutoff: int = 10, remove_toppop=False):
    
        scores = self._compute_items_scores(user)
        scores, seen = self._remove_seen_items(user, scores)
        if remove_toppop:
            scores[self.popitems] = -np.inf
        scores = scores.argsort()[::-1]

        return scores[:cutoff], seen

    def save_r_hat(self, folder='raw_data', test=False):

        self.r_hat = self.r_hat[:7947]

        if not os.path.exists(folder):
            os.mkdir(folder)

        if test:
            PATH = os.path.join(folder, self.NAME + '-r-hat-test')
        else:
            PATH = os.path.join(folder, self.NAME + '-r-hat-valid')
        
        if self.r_hat is None:
            msg = '|{}| can not save r_hat train the model first!'
            print(msg.format(self.NAME))

        else:
            if isinstance(self.r_hat, np.ndarray):
                np.save(PATH, self.r_hat)
            else:
                np.savez(
                    PATH, 
                    data=self.r_hat.data, 
                    indices=self.r_hat.indices, 
                    indptr=self.r_hat.indptr, 
                    shape=self.r_hat.shape
                )
            print('|{}| r hat saved in:{}'.format(self.NAME, PATH))

    def load_r_hat(self, path):
        if path[-1] == 'y':
            self.r_hat = np.load(path)
        else:
            loader = np.load(path)
            self.r_hat = sps.csr_matrix(
                (loader['data'], loader['indices'], loader['indptr']),
                shape=loader['shape']
            )
        print('|{}| r hat loaded:{}'.format(self.NAME, path))
    
    def save_sim_matrix(self, folder='raw_data', test=False):

        if not os.path.exists(folder):
            os.mkdir(folder)

        if test:
            PATH = os.path.join(folder, self.NAME + '-sim-matrix-test')
        else:
            PATH = os.path.join(folder, self.NAME + '-sim-matrix-valid')  

        self.sim_matrix = self._check_matrix(self.sim_matrix, format='csr')

        if self.sim_matrix is None:
            msg = '|{}| can not save sim_matrix train the model first!'
            print(msg.format(self.NAME))

        else:
            np.savez(
                PATH, 
                data=self.sim_matrix.data, 
                indices=self.sim_matrix.indices, 
                indptr=self.sim_matrix.indptr, 
                shape=self.sim_matrix.shape
            )
            print('|{}| sim matrix saved in: {}'.format(self.NAME, PATH))

    def load_sim_matrix(self, path):
        loader = np.load(path)
        self.sim_matrix = sps.csr_matrix(
            (loader['data'], loader['indices'], loader['indptr']),
            shape=loader['shape']
        )

    def _remove_seen_items(self, user, scores):
        
        s = self.urm.indptr[user]
        e = self.urm.indptr[user + 1]
        
        seen = self.urm.indices[s:e]
        
        scores[seen] = -np.inf
        return scores, len(seen)

    def _remove_toppop_items(self, user, scores, at=2):

        item_popularity = np.ediff1d(self.urm.tocsc().indptr)
        popular_items = np.argsort(item_popularity)
        popular_items = np.flip(popular_items, axis = 0)
        toremove = popular_items[0:at]

        scores[toremove] = -np.inf
        return scores
    
    def _compute_toppop_items(self, at=10):
        item_popularity = np.ediff1d(self.urm.tocsc().indptr)
        popular_items = np.argsort(item_popularity)
        popular_items = np.flip(popular_items, axis = 0)
        popitems = popular_items[0:at]
        return popitems


    def _check_matrix(self, X, format='csc', dtype=np.float32):
        """
        This function takes a matrix as input and transforms it into the specified format.
        The matrix in input can be either sparse or ndarray.
        If the matrix in input has already the desired format, it is returned as-is
        the dtype parameter is always applied and the default is np.float32
        :param X:
        :param format:
        :param dtype:
        :return:
        """


        if format == 'csc' and not isinstance(X, sps.csc_matrix):
            return X.tocsc().astype(dtype)
        elif format == 'csr' and not isinstance(X, sps.csr_matrix):
            return X.tocsr().astype(dtype)
        elif format == 'coo' and not isinstance(X, sps.coo_matrix):
            return X.tocoo().astype(dtype)
        elif format == 'dok' and not isinstance(X, sps.dok_matrix):
            return X.todok().astype(dtype)
        elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
            return X.tobsr().astype(dtype)
        elif format == 'dia' and not isinstance(X, sps.dia_matrix):
            return X.todia().astype(dtype)
        elif format == 'lil' and not isinstance(X, sps.lil_matrix):
            return X.tolil().astype(dtype)

        elif format == 'npy':
            if sps.issparse(X):
                return X.toarray().astype(dtype)
            else:
                return np.array(X)

        elif isinstance(X, np.ndarray):
            X = sps.csr_matrix(X, dtype=dtype)
            X.eliminate_zeros()
            return self._check_matrix(X, format=format, dtype=dtype)
        else:
            return X.astype(dtype)

    def _similarity_matrix_topk(self, item_weights, k=100):
        """
        The function selects the TopK most similar elements, column-wise

        :param item_weights:
        :param forceSparseOutput:
        :param k:
        :param verbose:
        :param inplace: Default True, WARNING matrix will be modified
        :return:
        """

        assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

        nitems = item_weights.shape[1]
        k = min(k, nitems)

        # for each column, keep only the top-k scored items
        sparse_weights = not isinstance(item_weights, np.ndarray)

        # iterate over each column and keep only the top-k similar items
        data, rows_indices, cols_indptr = [], [], []

        if sparse_weights:
            item_weights = self._check_matrix(item_weights, format='csc', dtype=np.float32)
        else:
            column_row_index = np.arange(nitems, dtype=np.int32)

        for item_idx in range(nitems):

            cols_indptr.append(len(data))

            if sparse_weights:
                start_position = item_weights.indptr[item_idx]
                end_position = item_weights.indptr[item_idx+1]

                column_data = item_weights.data[start_position:end_position]
                column_row_index = item_weights.indices[start_position:end_position]

            else:
                column_data = item_weights[:,item_idx]


            non_zero_data = column_data!=0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-k:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])


        cols_indptr.append(len(data))

        # During testing CSR is faster
        W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)

        return W_sparse