from recommenders.recommender import Recommender
import implicit
import sys
import configparser
import numpy as np

class IALS(Recommender):

    NAME = 'IALS'

    def __init__(self, urm):

        super().__init__(urm = urm)

        self.n_users, self.n_items = self.urm.shape
        self.urmt = self.urm.T

    def fit(self, n_factors=100, reg=0.02, iterations=90, alpha=15):

        model = implicit.als.AlternatingLeastSquares(factors=n_factors, regularization=reg, iterations=iterations)
        data_conf = (self.urmt * alpha).astype('double')
        model.fit(data_conf)

        # save user and item factors

        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

        self.r_hat = model.user_factors.dot(model.item_factors.T)
    
    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_N_FACTORS = 0
        BEST_ALPHA = 0
        BEST_EPOCHS = 0
        BEST_REG = 0

        cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        cp.read('config.ini')
        
        e = cp.getlist('tuning.IALS', 'epochs')
        f = cp.getlist('tuning.IALS', 'n_factors')
        a = cp.getlist('tuning.IALS', 'alphas')
        r = cp.getlist('tuning.IALS', 'reg')

        #r = float(cp.get('tuning.IALS', 'reg'))

        n_factors = np.arange(int(f[0]), int(f[1]), int(f[2]))
        alphas    = np.arange(int(a[0]), int(a[1]), int(a[2]))
        reg       = np.arange(float(r[0]), float(r[1]), float(r[2]))
        epochs    = np.arange(int(e[0]), int(e[1]), int(e[2]))
        
        total = len(n_factors) * len(alphas) * len(reg) * len(epochs)

        i = 0
        for nf in n_factors:
            for a in alphas:
                for r in reg:
                    for e in epochs:
                        self.fit(
                            iterations=e,
                            n_factors=nf,
                            reg=r,
                            alpha=a
                        )

                        self._evaluate(urm_valid)

                        m = '|{}|iter:{:-4d}/{}|nfact:{:-3d}|alpha:{:.3f}|epochs:{:-3d}|reg:{:.6f}|MAP: {:.4f} |'
                        print(m.format(self.NAME, i, total, nf, a, e, r ,self.MAP))
                        sys.stdout.flush()
                        i+=1

                        if self.MAP > BEST_MAP:

                            BEST_N_FACTORS = nf
                            BEST_ALPHA = a
                            BEST_EPOCHS = e
                            BEST_REG = r
                            BEST_MAP = self.MAP

        m = '|{}|best|nfact:{:-3d}|alpha:{:.3f}|epochs:{:-3d}|reg:{:.6f}| MAP: {:.4f} |'
        print(m.format(self.NAME, BEST_N_FACTORS, BEST_ALPHA, BEST_EPOCHS, BEST_REG, BEST_MAP))       


