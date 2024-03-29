from recommenders.recommender import Recommender
import sys
from sklearn.preprocessing import normalize
import numpy as np
import time
import os
import configparser
from tqdm import tqdm

class MF_IALS(Recommender):
    """
    Binary/Implicit Alternating Least Squares (IALS)
    See:
    Y. Hu, Y. Koren and C. Volinsky, Collaborative filtering for implicit feedback datasets, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf
    R. Pan et al., One-class collaborative filtering, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf
    Factorization model for binary feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \lambda(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})}
    """

    NAME = "MF_IALS"

    AVAILABLE_CONFIDENCE_SCALING = ["linear", "log"]

    def __init__(self, urm):
        
        super().__init__(urm = urm)
        self.URM_train = urm

    def save_r_hat(self, folder='raw_data'):

        if not os.path.exists(folder):
            os.mkdir(folder)

        if self.r_hat is None:
            msg = '|{}| can not save r_hat train the model first!'
            print(msg.format(self.NAME))

        else:
            PATH = os.path.join(folder, self.NAME + '-r-hat') 
            np.save(PATH, self.r_hat)
            print('|{}| r hat has been saved'.format(self.NAME))

    def load_r_hat(self, path):
        self.r_hat = np.load(path)

    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_N_FACTORS = 0
        BEST_ALPHA = 0
        BEST_EPSILON = 0
        BEST_EPOCHS = 0
        BEST_CONFIDENCE_SCALING = ''
        BEST_REG = 0
        BEST_INIT_MEAN=0.0
        BEST_INIT_STD=0

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')

        confidence_scaling = ['linear'] #['linear', 'log']
        epsilon = 1.0
        init_mean=0.0
        init_std=0.1
        
        epochs = int(cp.get('tuning.MF_IALS', 'epochs'))
        t = cp.getlist('tuning.MF_IALS', 'n_factors')
        a = cp.getlist('tuning.MF_IALS', 'alphas')

        n_factors = np.arange(int(t[0]), int(t[1]), int(t[2]))
        alpha     = np.arange(float(a[0]), float(a[1]), float(a[2]))
        reg       = [1e-1, 1e-2, 1e-3] 

        total = len(n_factors) * len(alpha) * len(reg) * len(confidence_scaling)

        i = 0
        for nf in n_factors:
            for a in alpha:
                for r in reg:
                    for cs in confidence_scaling:
                        self.fit(
                            epochs=epochs,
                            num_factors=nf,
                            confidence_scaling=cs,
                            epsilon=epsilon,
                            reg=r,
                            alpha=a,
                            init_mean=init_mean,
                            init_std=init_std
                            )

                        self._evaluate(urm_valid)

                        m = '|{}|iter:{:-4d}/{}|nfact:{:-3d}|alpha:{:.3f}|epsilon:{}|epochs:{:-3d}|confscal:{}|reg:{:.6f}|imean:{}|istd:{}| MAP: {:.4f} |'
                        print(m.format(self.NAME, i, total, nf, a, epsilon, epochs, cs[:2], r, init_mean, init_std ,self.MAP))
                        sys.stdout.flush()
                        i+=1

                        if self.MAP > BEST_MAP:

                            BEST_N_FACTORS = nf
                            BEST_ALPHA = a
                            BEST_EPSILON = epsilon
                            BEST_EPOCHS = epochs
                            BEST_CONFIDENCE_SCALING = cs
                            BEST_REG = r
                            BEST_INIT_MEAN=init_mean
                            BEST_INIT_STD=init_std
                            BEST_MAP = self.MAP
        m = '|{}|best|nfact:{:-3d}|alpha:{:.3f}|epsilon:{}|epochs:{:-3d}|confscal:{}|reg:{:.6f}|imean:{}|istd:{}| MAP: {:.4f} |'
        print(m.format(self.NAME, BEST_N_FACTORS, BEST_ALPHA, BEST_EPSILON, BEST_EPOCHS, BEST_CONFIDENCE_SCALING, BEST_REG, BEST_INIT_MEAN, BEST_INIT_STD ,self.MAP))       


    def fit(self, 
            epochs = 300,
            num_factors = 100,
            confidence_scaling = "linear",
            alpha = 1.0,
            epsilon = 1.0,
            reg = 1e-3,
            init_mean=0.0,
            init_std=0.1):
        """
        :param epochs:
        :param num_factors:
        :param confidence_scaling: supported scaling modes for the observed values: 'linear' or 'log'
        :param alpha: Confidence weight, confidence c = 1 + alpha*r where r is the observed "rating".
        :param reg: Regularization constant.
        :param epsilon: epsilon used in log scaling only
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :return:
        """

        if confidence_scaling not in self.AVAILABLE_CONFIDENCE_SCALING:
           raise ValueError("Value for 'confidence_scaling' not recognized. Acceptable values are {}, provided was '{}'".format(self.AVAILABLE_CONFIDENCE_SCALING, confidence_scaling))


        self.num_factors = num_factors
        self.alpha = alpha
        self.epsilon = epsilon
        self.reg = reg

        self.USER_factors = self._init_factors(self.n_users, False)  # don't need values, will compute them
        self.ITEM_factors = self._init_factors(self.n_items)


        self._build_confidence_matrix(confidence_scaling)

        warm_user_mask = np.ediff1d(self.URM_train.indptr) > 0
        warm_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) > 0

        self.warm_users = np.arange(0, self.n_users, dtype=np.int32)[warm_user_mask]
        self.warm_items = np.arange(0, self.n_items, dtype=np.int32)[warm_item_mask]

        self.regularization_diagonal = np.diag(self.reg * np.ones(self.num_factors))

        
        for epochs_current in range(epochs):
            
            self._run_epoch(epochs_current)

        self.r_hat = self.USER_factors.dot(self.ITEM_factors.T)     

    def _build_confidence_matrix(self, confidence_scaling):

        if confidence_scaling == 'linear':
            self.C = self._linear_scaling_confidence()
        else:
            self.C = self._log_scaling_confidence()

        self.C_csc= self._check_matrix(self.C.copy(), format="csc", dtype = np.float32)


    def _linear_scaling_confidence(self):

        C = self._check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + self.alpha*C.data

        return C

    def _log_scaling_confidence(self):

        C = self._check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / self.epsilon)

        return C


    def _run_epoch(self, num_epoch):

        # fit user factors
        # VV = n_factors x n_factors
        VV = self.ITEM_factors.T.dot(self.ITEM_factors)

        for user_id in tqdm(self.warm_users):
            # get (positive i.e. non-zero scored) items for user

            start_pos = self.C.indptr[user_id]
            end_pos = self.C.indptr[user_id + 1]

            user_profile = self.C.indices[start_pos:end_pos]
            user_confidence = self.C.data[start_pos:end_pos]

            self.USER_factors[user_id, :] = self._update_row(user_profile, user_confidence, self.ITEM_factors, VV)

        # fit item factors
        # UU = n_factors x n_factors
        UU = self.USER_factors.T.dot(self.USER_factors)

        for item_id in self.warm_items:

            start_pos = self.C_csc.indptr[item_id]
            end_pos = self.C_csc.indptr[item_id + 1]

            item_profile = self.C_csc.indices[start_pos:end_pos]
            item_confidence = self.C_csc.data[start_pos:end_pos]

            self.ITEM_factors[item_id, :] = self._update_row(item_profile, item_confidence, self.USER_factors, UU)



    def _update_row(self, interaction_profile, interaction_confidence, Y, YtY):
        """
        Update latent factors for a single user or item.
        Y = |n_interactions|x|n_factors|
        YtY =   |n_factors|x|n_factors|
        """

        # Latent factors ony of item/users for which an interaction exists in the interaction profile
        Y_interactions = Y[interaction_profile, :]

        # Following the notation of the original paper we report the update rule for the Item factors (User factors are identical):
        # Y are the item factors |n_items|x|n_factors|
        # Cu is a diagonal matrix |n_interactions|x|n_interactions| with the user confidence for the observed items
        # p(u) is a boolean vectors indexing only observed items. Here it will disappear as we already extract only the observed latent factors
        #       however, it will have an impact in the dimensions of the matrix, since it transforms Cu from a diagonal matrix to a row vector of 1 row and |n_interactions| columns
        # (Yt*Cu*Y + reg*I)^-1 * Yt*Cu*profile
        # which can be decomposed as
        # (YtY + Yt*(Cu-I)*Y + reg*I)^-1 * Yt*Cu*p(u)

        # A = (|n_interactions|x|n_factors|) dot (|n_interactions|x|n_interactions| ) dot (|n_interactions|x|n_factors| )
        #   = |n_factors|x|n_factors|
        # A_slow = Y_interactions.T.dot(np.diag(interaction_confidence - 1)).dot(Y_interactions)

        # if v = diag(|n_interactions|) and k = |n_interactions|x|n_factors|
        # computing np.diag(v).dot(k) will be SLOW
        # we use an equivalent formulation (v * k.T).T which is much faster
        A = Y_interactions.T.dot(((interaction_confidence - 1) * Y_interactions.T).T)

        B = YtY + A + self.regularization_diagonal

        return np.dot(np.linalg.inv(B), Y_interactions.T.dot(interaction_confidence))


    def _init_factors(self, num_factors, assign_values=True):

        if assign_values:
            return self.num_factors**-0.5*np.random.random_sample((num_factors, self.num_factors))

        else:
            return np.empty((num_factors, self.num_factors))