from recommenders.recommender import Recommender
from sklearn.preprocessing import normalize
from tqdm import tqdm
import numpy as np
from scipy import sparse
from similarity.similarity import similarity

class SLIMBPR(Recommender):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self                The object
    ## @param      user_rating_matrix  (numpy array) The user rating matrix
    ## @param      learning_rate       (Float) The learning rate
    ## @param      epochs              (Integer) The number of epochs of training
    ## @param      pir                 (Float) The positive item regularization
    ## @param      nir                 (Float) The negative item regularization
    ##
    def __init__(self, urm, learning_rate=0.0005, epochs=5, pir=0.1, nir=0.01):
        super().__init__(urm = urm)
        self.epochs = epochs
        self.n_users = self.urm.shape[0]
        self.n_items = self.urm.shape[1]
        self.learning_rate = learning_rate
        self.positive_item_regularization = pir
        self.negative_item_regularization = nir
        self.sm = np.zeros((self.n_items, self.n_items))

    ##
    ## @brief      Samples a random triplet from the user rating matrix
    ##
    ## @param      self  The object
    ##
    ## @return     The index of the sampled user, the index of a positive interaction, the index of a negative interaction
    ##
    def sample(self):
        user_index = np.random.choice(self.n_users)
        interactions = self.urm[user_index].indices
        interaction_index = np.random.choice(interactions)
        selected = False
        while not selected:
            negative_interaction_index = np.random.randint(0, self.n_items)
            if negative_interaction_index not in interactions: selected = True
        return user_index, interaction_index, negative_interaction_index

    ##
    ## @brief      updates the similarity matrix once for each positive interaction
    ##
    ## @param      self  The object
    ##
    ## @return     None
    ##
    def iteration(self):
        num_positive_iteractions = int(self.urm.nnz)
        for _ in tqdm(range(num_positive_iteractions)):
            try:
                user_index, positive_item_id, negative_item_id = self.sample()
            except:
                continue
            user_interactions = self.urm[user_index, :].indices
            x_i = self.sm[positive_item_id, user_interactions].sum()
            x_j = self.sm[negative_item_id, user_interactions].sum()
            z = 1. / (1. + np.exp(x_i - x_j))
            for v in user_interactions:
                d = z - self.positive_item_regularization * x_i
                self.sm[positive_item_id, v] += self.learning_rate * d
                d = z - self.negative_item_regularization * x_j
                self.sm[negative_item_id, v] -= self.learning_rate * d
                self.sm[positive_item_id, positive_item_id] = 0
                self.sm[negative_item_id, negative_item_id] = 0

    ##
    ## @brief      Fits the model computing the similarity between items that maximises
    ##
    ## @param      self                  The object
    ## @param      k_nearest_neighbours  (Integer) The number of nearest neighbours
    ##
    ## @return     None
    ##
    def fit(self, k_nearest_neighbours=200):
        self._compute_similarity_matrix(k_nearest_neighbours)

    ##
    ## @brief      Calculates the similarity matrix.
    ##
    ## @param      self  The object
    ## @param      knn   The knn
    ##
    ## @return     None
    ##
    def _compute_similarity_matrix(self, knn):
        for e in range(self.epochs):
            self.iteration()
        s_tmp = []
        for i in tqdm(range(self.n_items)):
            mat = self.sm[i, :]
            s_tmp.append(self._knn(mat, knn))
        s = sparse.vstack(s_tmp, format='csr')
        s.setdiag(0)
        self.sm = s
        self.r_hat = self.urm.dot(self.sm)
        self.r_hat = normalize(self.r_hat)
    
    def _knn(self, mat, k):
        """Given a similarity matrix removes all but the k most similar elements from each row
            Args:
                mat: similarity matrix
                k: number of neighbours
            Returns:
                mat: similarity matrix with k most similar
        """
        mat = sparse.csr_matrix(mat)
        i = mat.data.argsort()[:-k]
        mat.data[i] = 0
        sparse.csr_matrix.eliminate_zeros(mat)
        return mat
