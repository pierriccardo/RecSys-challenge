import numpy as np
import scipy.sparse as sps
import similaripy as sim
from sklearn.preprocessing import normalize
from recommenders.recommender import Recommender
import time, sys
import configparser




class P3alpha(Recommender):

    NAME = "P3alpha"

    def __init__(self, urm):
        super(P3alpha, self).__init__(urm = urm)

    def fit(self, topK=210, alpha=0.3):

        self.topK = topK
        self.alpha = alpha

        self.sim_matrix = sim.p3alpha(
            self.urm.T,
            alpha=alpha,
            k=topK)
        #self.sim_matrix = normalize(self.sim_matrix, norm='l2', axis=1)

        
        self.r_hat = self.urm.dot(self.sim_matrix)


    def tuning(self, urm_valid):
        
        BEST_MAP = 0.0
        BEST_TOPK = 0
        BEST_ALPHA = 0

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')
        
        t = cp.getlist('tuning.P3alpha', 'topKs') 
        a = cp.getlist('tuning.P3alpha', 'alphas')

        topKs   = np.arange(int(t[0]), int(t[1]), int(t[2]))
        alphas = np.arange(float(a[0]), float(a[1]), float(a[2]))

        total = len(topKs) * len(alphas)

        i = 0
        for t in topKs:
            for a in alphas:
                self.fit(alpha=a, topK=t)

                self._evaluate(urm_valid)

                log = '| iter: {:-5d}/{} | topk: {:-3d} | alpha: {:.3f} | MAP: {:.4f} |'
                print(log.format(i, total, t, a, self.MAP))
                sys.stdout.flush()

                i+=1

                if self.MAP > BEST_MAP:

                    BEST_TOPK = t
                    BEST_ALPHA = a
                    BEST_MAP = self.MAP
                    
        log = '| best results | topk: {:-3d} | alpha: {:.3f} | MAP: {:.4f} |'
        print(log.format(BEST_TOPK, BEST_ALPHA, BEST_MAP))


'''
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

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
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


        if norm!='none':
            self.W_sparse = normalize(self.W_sparse, norm=norm, axis=1)


        if self.topK != False:
            self.W_sparse = self._similarity_matrix_topk(self.W_sparse, k=self.topK)

        self.sim_matrix = self._check_matrix(self.W_sparse, format='csr')
'''