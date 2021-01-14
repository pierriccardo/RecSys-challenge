from recommenders.recommender import Recommender    
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import similaripy as sim
import sys
import configparser
from tqdm import tqdm
from evaluator import Evaluator

from recommenders.recommender       import Recommender
from recommenders.ItemKNNCF         import ItemKNNCF
from recommenders.ItemKNNCB         import ItemKNNCB
from recommenders.SLIM_MSE          import SLIM_MSE
from recommenders.HybridSimilarity  import HybridSimilarity
from recommenders.HybridScores      import HybridScores
from recommenders.HybridRhat        import HybridRhat
from recommenders.HybridMultiRhat   import HybridMultiRhat
from recommenders.P3alpha           import P3alpha
from recommenders.RP3beta           import RP3beta
from recommenders.PureSVD           import PureSVD
from recommenders.SLIM_BPR          import SLIM_BPR
from recommenders.HybridRhat        import HybridRhat
from recommenders.UserKNNCF         import UserKNNCF
from recommenders.UserKNNCB         import UserKNNCB
from recommenders.IALS              import IALS
from recommenders.HybridMultiSim    import HybridMultiSim
from recommenders.SLIM_ELN          import SLIM_ELN
from recommenders.MF_BPR            import MF_BPR
from recommenders.TopPop            import TopPop

class HybridCluster(Recommender):

    NAME = 'HMC' 

    def __init__(self, urm, urmicm, icm, urm_valid, is_test=False):

        self.urm = urm
        self.urmicm = urmicm
        self.icm = icm
        self.urm_valid = urm_valid

        TYPE = 'test' if is_test else 'valid'

        # --- PATHS ---
        ICF_path = 'raw_data/ItemKNNCF-r-hat-{}.npz'.format(TYPE)
        ICB_path = 'raw_data/ItemKNNCB-r-hat-{}.npz'.format(TYPE)
        RP3_path = 'raw_data/RP3beta-r-hat-{}.npz'.format(TYPE)
        RP3ICM_path = 'raw_data/RP3betaICM-r-hat-{}.npz'.format(TYPE)
        P3AICM_path = 'raw_data/P3alpha-r-hat-{}.npz'.format(TYPE)
        UCF_path = 'raw_data/UserKNNCF-r-hat-{}.npz'.format(TYPE)
        UCB_path = 'raw_data/UserKNNCB-r-hat-{}.npz'.format(TYPE)
        MSE_path = 'raw_data/SLIM_MSE-r-hat-{}.npz'.format(TYPE)
        ALS_path = 'raw_data/IALS-r-hat-{}.npy'.format(TYPE)
        ELN_path = 'raw_data/SLIM_ELN-r-hat-{}.npz'.format(TYPE)

        
        # --- RECOMMENDERS ---
        self.ICF = ItemKNNCF(self.urm)
        self.ICF.load_r_hat(ICF_path)

        self.ICB = ItemKNNCB(self.urm, self.icm)
        self.ICB.load_r_hat(ICB_path)

        self.RP3 = RP3beta(self.urm)
        self.RP3.load_r_hat(RP3_path)

        self.RP3ICM = RP3beta(self.urm)
        self.RP3ICM.load_r_hat(RP3ICM_path)

        self.P3AICM = P3alpha(self.urm)
        self.P3AICM.load_r_hat(P3AICM_path)

        self.UCF = UserKNNCF(self.urm)
        self.UCF.load_r_hat(UCF_path)

        self.UCB = UserKNNCB(self.urm, self.icm)
        self.UCB.load_r_hat(UCB_path)

        self.MSE = SLIM_MSE(self.urm)
        self.MSE.load_r_hat(MSE_path)

        self.ALS = IALS(self.urm)
        self.ALS.load_r_hat(ALS_path)

        self.ELN = SLIM_ELN(self.urm)
        self.ELN.load_r_hat(ELN_path)

        # --- HYBRIDS ---
        self.HMR_ICF_ICB = HybridMultiRhat(self.urm, [self.ICF, self.ICB]) 
        self.HMR_ICF_ICB.fit([0.61860621, 0.38139379])

        self.HMR_ICB_RP3 = HybridMultiRhat(self.urm, [self.ICB, self.RP3]) 
        self.HMR_ICB_RP3.fit([0.71494442, 0.28505558])

        self.HMR_P3AICM_UCF_UCB = HybridMultiRhat(self.urm, [self.P3AICM, self.UCF, self.UCB])
        self.HMR_P3AICM_UCF_UCB.fit([0.11178127, 0.03897336, 0.84924537])

        self.HMR_MSE_ALS_ELN = HybridMultiRhat(self.urm, [self.MSE, self.ALS, self.ELN])
        self.HMR_MSE_ALS_ELN.fit([0.414438, 0.5002536, 0.0853084])

        self.HMR_RP3ICM_UCF_UCB = HybridMultiRhat(self.urm, [self.RP3ICM, self.UCF, self.UCB])
        self.HMR_RP3ICM_UCF_UCB.fit([0.67314843, 0.24931299, 0.07753858])

    def fit(self):
        pass   
        

    def _compute_items_scores(self, user):
        pass
        
    
    def recommend(self, user, cutoff=10):
       
        s = self.urm.indptr[user]
        e = self.urm.indptr[user + 1]
        
        sn = self.urm.indices[s:e]
        seen = len(sn)


        '''
        if seen >= 0 and seen < 2:
            scores, k = self.HMR_ICF_ICB.recommend(user, cutoff=10)
        elif seen >= 2 and seen < 5:
            scores, k = self.HMR_P3AICM_UCF_UCB.recommend(user, cutoff=10)
        elif seen >= 6 and seen < 7:
            scores, k = self.HMR_RP3ICM_UCF_UCB.recommend(user, cutoff=10)
        elif seen >= 7 and seen < 8:
            scores, k = self.RP3ICM.recommend(user, cutoff=10)
        elif seen >= 8 and seen < 9:
            scores, k = self.HMR_RP3ICM_UCF_UCB.recommend(user, cutoff=10)
        elif seen >= 9 and seen < 10:
            scores, k = self.RP3ICM.recommend(user, cutoff=10)
        elif seen >= 10 and seen < 11:
            scores, k = self.HMR_RP3ICM_UCF_UCB.recommend(user, cutoff=10)
        elif seen >= 11 and seen < 12:
            scores, k = self.HMR_RP3ICM_UCF_UCB.recommend(user, cutoff=10)
        elif seen >= 12 and seen < 13:
            scores, k = self.RP3ICM.recommend(user, cutoff=10)
        elif seen >= 13 and seen < 14:
            scores, k = self.HMR_P3AICM_UCF_UCB.recommend(user, cutoff=10)
        elif seen >= 14 and seen < 15:
            scores, k = self.HMR_ICB_RP3.recommend(user, cutoff=10)
        elif seen >= 15 and seen < 30:
            scores, k = self.HMR_RP3ICM_UCF_UCB.recommend(user, cutoff=10)
        elif seen >= 30 and seen < 1600:
            scores, k = self.HMR_MSE_ALS_ELN.recommend(user, cutoff=10)
        else:
            scores, k = self.HMR_ICB_RP3.recommend(user, cutoff=10)
        '''

        if seen >= 0 and seen < 30:
            scores, k = self.HMR_ICB_RP3.recommend(user, cutoff=10)
        else:
            scores, k = self.HMR_MSE_ALS_ELN.recommend(user, cutoff=10)

        return scores, seen
        