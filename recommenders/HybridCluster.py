from recommenders.recommender import Recommender    
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import similaripy as sim
import sys
import configparser
from tqdm import tqdm

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

    def __init__(self, urm, urmicm, icm):

        self.urm = urm
        self.urmicm = urmicm
        self.icm = icm

        TYPE = 'test'

        # --- paths ---
        ICF_path = 'raw_data/ItemKNNCF-r-hat-{}.npz'.format(TYPE)
        ICB_path = 'raw_data/ItemKNNCB-r-hat-{}.npz'.format(TYPE)
        RP3_path = 'raw_data/RP3beta-r-hat-{}.npz'.format(TYPE)
        P3A_path = 'raw_data/P3alpha-r-hat-{}.npz'.format(TYPE)
        UCF_path = 'raw_data/UserKNNCF-r-hat-{}.npz'.format(TYPE)
        UCB_path = 'raw_data/UserKNNCB-r-hat-{}.npz'.format(TYPE)
        MSE_path = 'raw_data/SLIM_MSE-r-hat-{}.npz'.format(TYPE)

        
        # --- recommenders ---
        self.ICF = ItemKNNCF(self.urm)
        self.ICF.load_r_hat(ICF_path)

        self.ICB = ItemKNNCB(self.urm, self.icm)
        self.ICB.load_r_hat(ICB_path)

        self.RP3 = RP3beta(self.urm)
        self.RP3.load_r_hat(RP3_path)

        self.P3A = P3alpha(self.urm)
        self.P3A.load_r_hat(P3A_path)

        self.UCF = UserKNNCF(self.urm)
        self.UCF.load_r_hat(UCF_path)

        self.UCB = UserKNNCB(self.urm, self.icm)
        self.UCB.load_r_hat(UCB_path)

        self.MSE = SLIM_MSE(self.urm)
        self.MSE.load_r_hat(MSE_path)

        # --- HYBRIDS ---
        self.HMR_ICF_ICB = HybridMultiRhat(self.urm, [self.ICF, self.ICB]) 
        self.HMR_ICF_ICB.fit([0.41466565, 0.58533435]) 

        self.HMR_ICB_RP3 = HybridMultiRhat(self.urm, [self.ICB, self.RP3]) 
        self.HMR_ICB_RP3.fit([0.71494442, 0.28505558]) 

       

    def fit(self):
        pass   
        

    def _compute_items_scores(self, user):
        pass
        
    
    def recommend(self, user, cutoff=10):
        '''
        |range(    0,    2) |ItemCF
        |range(    2,    3) |P3alpha
        |range(    3,    4) |RP3beta
        |range(    4,    5) |RP3beta
        |range(    5,    6) |RP3beta
        |range(    6,    7) |RP3beta
        |range(    7,    8) |RP3beta
        |range(    8,    9) |RP3beta
        |range(    9,   10) |RP3beta
        |range(   10,   11) |RP3beta
        |range(   11,   12) |RP3beta
        |range(   12,   13) |RP3beta
        |range(   13,   14) |RP3beta
        |range(   14,   15) |RP3beta
        |range(   15,   30) |RP3beta
        |range(   30,   50) |SLIM_MSE
        |range(   50,  100) |UserKNNCF
        |range(  100, 1600) |SLIM_MSE
        '''
        s = self.urm.indptr[user]
        e = self.urm.indptr[user + 1]
        
        sn = self.urm.indices[s:e]
        seen = len(sn)

        

        if seen >= 0 and seen < 2:
            scores, k = self.HMR_ICF_ICB.recommend(user, cutoff=10)
        if seen >= 2 and seen < 3:
            scores, k = self.P3A.recommend(user, cutoff=10)
        if seen >= 3 and seen < 30:
            scores, k = self.UCF.recommend(user, cutoff=10)
        if seen >= 30 and seen < 50:
            scores, k = self.MSE.recommend(user, cutoff=10)
        if seen >= 50 and seen < 100:
            scores, k = self.UCF.recommend(user, cutoff=10)
        if seen >= 100 and seen < 1600:
            scores, k = self.MSE.recommend(user, cutoff=10)
        #if seen >= 500 and seen < 600:
        #    scores, k = self.itemKNNCF.recommend(user, cutoff=10)
        #if seen >= 600 and seen < 1000:
        #    scores, k = self.rp3beta.recommend(user, cutoff=10)
        #if seen >= 1000 and seen < 1600:
        #    scores, k = self.userKNNCF.recommend(user, cutoff=10)
        else:
            scores, k = self.HMR_ICB_RP3.recommend(user, cutoff=10)

        return scores, seen
        