from recommenders.recommender import Recommender
from sklearn.preprocessing import normalize
import numpy as np
import time
import scipy.sparse as sps
from similarity.similarity import similarity 



class EASE_R(Recommender):
    """ EASE_R_Recommender
        https://arxiv.org/pdf/1905.03375.pdf
     @article{steck2019embarrassingly,
      title={Embarrassingly Shallow Autoencoders for Sparse Data},
      author={Steck, Harald},
      journal={arXiv preprint arXiv:1905.03375},
      year={2019}
    }
    """

    NAME = "EASE_R"


    def __init__(self, URM_train, sparse_threshold_quota = None):
        super(EASE_R, self).__init__(URM_train)
        self.URM_train = URM_train
        self.sparse_threshold_quota = sparse_threshold_quota

    def fit(self, topK=None, l2_norm = 1e3, normalize_matrix = False):

        start_time = time.time()
        print("|{}| Fitting model... |".format(self.NAME))

        if normalize_matrix:
            # Normalize rows and then columns
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)
            self.URM_train = sps.csr_matrix(self.URM_train)


        # Grahm matrix is X X^t, compute dot product
        grahm_matrix =  similarity(self.URM_train, shrink=0, k=self.URM_train.shape[1], sim_type = "cosine").toarray()

        diag_indices = np.diag_indices(grahm_matrix.shape[0])

        grahm_matrix[diag_indices] += l2_norm

        P = np.linalg.inv(grahm_matrix)

        B = P / (-np.diag(P))

        B[diag_indices] = 0.0

        print("Fitting model... done in {:.2f}".format(time.time() - start_time))

        # Check if the matrix should be saved in a sparse or dense format
        # The matrix is sparse, regardless of the presence of the topK, if nonzero cells are less than sparse_threshold_quota %
        if topK is not None:
            B = self._similarity_matrix_topk(B, k = topK, verbose = False)


        if self._is_content_sparse_check(B):
            self._print("Detected model matrix to be sparse, changing format.")
            self.W_sparse = self._check_matrix(B, format='csr', dtype=np.float32)

        else:
            self.W_sparse = self._check_matrix(B, format='npy', dtype=np.float32)
            self._W_sparse_format_checked = True
        
        self.r_hat = self.URM_train.dot(self.W_sparse)


    def _is_content_sparse_check(self, matrix):

        if self.sparse_threshold_quota is None:
            return False

        if sps.issparse(matrix):
            nonzero = matrix.nnz
        else:
            nonzero = np.count_nonzero(matrix)

        return nonzero / (matrix.shape[0]**2) <= self.sparse_threshold_quota


    def _compute_items_score(self, user_id_array):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        item_scores = user_profile_array.dot(self.W_sparse)#.toarray()

        return item_scores


