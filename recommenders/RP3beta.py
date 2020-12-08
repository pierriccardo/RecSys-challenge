import numpy as np
import scipy.sparse as sps
from recommenders.recommender import Recommender
from sklearn.preprocessing import normalize

from tqdm import tqdm

# best params found
#| topK = 170 | alpha = 0.3  | beta  = 0.07 | MAP = 0.0518 | 

class RP3beta(Recommender):

    NAME = "RP3beta"

    def __init__(self, urm, norm=False):
        
        super(RP3beta, self).__init__(urm = urm, norm=norm)

    def fit(self, alpha=1., beta=0.6, min_rating=0, topK=100, implicit=True, normalize_similarity=True):

        self.alpha = alpha
        self.beta = beta
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        
        # if X.dtype != np.float32:
        #     print("RP3beta fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.urm.data[self.urm.data < self.min_rating] = 0
            self.urm.eliminate_zeros()
            if self.implicit:
                self.urm.data = np.ones(self.urm.data.size, dtype=np.float32)

        #Pui is the row-normalized urm
        Pui = normalize(self.urm, norm='l1', axis=1)

        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.urm.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(self.urm.shape[1])

        nonZeroMask = X_bool_sum!=0.0

        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu


        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        for current_block_start_row in tqdm(range(0, Pui.shape[1], block_dim)):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)


        if self.topK != False:
            self.W_sparse = self._similarity_matrix_topk(self.W_sparse, k=self.topK)


        self.sim_matrix = self._check_matrix(self.W_sparse, format='csr')
        self.r_hat = self.urm.dot(self.sim_matrix)
        self.r_hat = self.r_hat.toarray()

    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_TOPK = 0
        BEST_ALPHA = 0
        BEST_BETA = 0

        topKs = np.arange(20, 300, 15)
        alphas = np.arange(0.25, 0.35, 0.3)
        betas = np.arange(0.05, 0.15, 0.03)

        total = len(topKs) * len(alphas) * len(betas)

        i = 0
        for t in topKs:
            for a in alphas:
                for b in betas:
                    self.fit(alpha=a, beta=b, topK=t)

                    self._evaluate(urm_valid)

                    log = '| iter: {:-5d}/{} | topk: {:-3d} | alpha: {:.3f} | beta: {:.3f} | MAP: {:.4f} |'
                    print(log.format(i, total, t, a, b, self.MAP))

                    i+=1

                    if self.MAP > BEST_MAP:

                        BEST_TOPK = t
                        BEST_ALPHA = a
                        BEST_BETA = b
                        BEST_MAP = self.MAP
        log = '| best results | topk: {:-3d} | alpha: {:.3f} | beta: {:.3f} | MAP: {:.4f} |'
        print(log.format(BEST_TOPK, BEST_ALPHA, BEST_BETA, BEST_MAP))