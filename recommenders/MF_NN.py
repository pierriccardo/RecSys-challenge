
from recommenders.recommender import Recommender
from sklearn.decomposition import NMF
import scipy.sparse as sps


class MF_NN(Recommender):
    """ Non Negative Matrix Factorization Recommender
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    """

    NAME = "MF_NN"

    SOLVER_VALUES = {"coordinate_descent": "cd",
                     "multiplicative_update": "mu"}

    INIT_VALUES = ["random", "nndsvda"]
    # random: non-negative random matrices, scaled with sqrt(X.mean() / n_components)
    # nndsvda: Nonnegative Double Singular Value Decomposition with zeros filled with the average of X

    BETA_LOSS_VALUES = ["frobenius", "kullback-leibler"]

    def __init__(self, URM_train):
        super(MF_NN, self).__init__(URM_train)

        self.URM_train = URM_train

    def fit(self, num_factors=10,
            l1_ratio = 0.5,
            solver = "multiplicative_update",
            init_type = "random",
            beta_loss = "frobenius"):

        print('|{}| training |'.format(self.NAME))
        assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

        if solver not in self.SOLVER_VALUES:
           raise ValueError("Value for 'solver' not recognized. Acceptable values are {}, provided was '{}'".format(self.SOLVER_VALUES.keys(), solver))

        if init_type not in self.INIT_VALUES:
           raise ValueError("Value for 'init_type' not recognized. Acceptable values are {}, provided was '{}'".format(self.INIT_VALUES, init_type))

        if beta_loss not in self.BETA_LOSS_VALUES:
           raise ValueError("Value for 'beta_loss' not recognized. Acceptable values are {}, provided was '{}'".format(self.BETA_LOSS_VALUES, beta_loss))

        print("|{}| Computing NMF decomposition ... |".format(self.NAME))

        nmf_solver = NMF(n_components  = num_factors,
                         init = init_type,
                         solver = self.SOLVER_VALUES[solver],
                         beta_loss = beta_loss,
                         random_state = None,
                         l1_ratio = l1_ratio,
                         shuffle = True,
                         verbose=True,
                         max_iter = 500)

        nmf_solver.fit(self.URM_train)

        self.ITEM_factors = nmf_solver.components_.copy().T
        self.USER_factors = nmf_solver.transform(self.URM_train)

        self.r_hat = self.USER_factors.dot(self.ITEM_factors)

        print("|{}| Done |".format(self.NAME))