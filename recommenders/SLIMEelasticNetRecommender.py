from base.similarity_matrix_recommender import ItemSimilarityMatrixRecommender

from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning

import scipy.sparse as sps
import numpy as np
from tqdm import tqdm
import warnings


class SLIMElasticNetRecommender(ItemSimilarityMatrixRecommender):

    NAME = "SLIMElasticNet"

    def __init__(self, urm):

        super().__init__(urm=urm)

        self.urm = urm

    #@ignore_warnings(category=ConvergenceWarning)
    def fit(self, l1_ratio=0.1, alpha = 1.0, positive_only=True, topK = 100):

        assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        #warnings.simplefilter('ignore', category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
       
        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        urm = self._check_matrix(self.urm, 'csc', dtype=np.float32)

        n_items = urm.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows    = np.zeros(dataBlock, dtype=np.int32)
        cols    = np.zeros(dataBlock, dtype=np.int32)
        values  = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        # fit each item's factors sequentially (not in parallel)
        for currentItem in tqdm(range(n_items)):

            # get the target column
            y = urm[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = urm.indptr[currentItem]
            end_pos = urm.indptr[currentItem + 1]

            current_item_data_backup = urm.data[start_pos: end_pos].copy()
            urm.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(urm, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value)-1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            urm.data[start_pos:end_pos] = current_item_data_backup

        # generate the sparse weight matrix
        self.sim_matrix = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                    shape=(n_items, n_items), dtype=np.float32)

