import numpy as np
import pandas as pd
import os
import scipy.sparse as sps
from matplotlib import pyplot
import configparser
from tqdm import tqdm
#------------------------------
#       MISC
#------------------------------

config = configparser.ConfigParser()
config.read('config.ini')

np.random.seed(int(config['DEFAULT']['SEED']))
#------------------------------
#       DATASET
#------------------------------
from dataset import Dataset
d = Dataset(split=0.8)

URM         = d.URM
URM_train   = d.get_URM_train()
URM_valid   = d.get_URM_valid()
ICM         = d.get_ICM()
test_set    = d.get_test()
train_df = d.urm_train_df
#------------------------------
#       RECOMMENDER
#------------------------------
from recommenders.recommender       import Recommender
from recommenders.ItemKNNCF         import ItemKNNCF
from recommenders.ItemKNNCB         import ItemKNNCB
from recommenders.SLIM_MSE          import SLIM_MSE
from recommenders.SLIM_BPR          import SLIM_BPR
from recommenders.HybridSimilarity  import HybridSimilarity
from recommenders.HybridScores      import HybridScores
from recommenders.P3alpha           import P3alpha
from recommenders.RP3beta           import RP3beta
from recommenders.UserKNNCB         import UserKNNCB 
from recommenders.MF_BPR            import MF_BPR
from recommenders.UserKNNCB         import UserKNNCB
from recommenders.UserKNNCF         import UserKNNCF


icf = ItemKNNCF(URM_train)
icf.fit()
icb = ItemKNNCB(URM_train, ICM)
icb.fit()

ucf= UserKNNCF(URM_train)
ucf.fit()
ucb = UserKNNCB(URM_train, ICM)
ucb.fit()

slim_mse = SLIM_MSE(URM_train)
slim_bpr = SLIM_BPR(URM_train)
slim_mse.load_r_hat('raw_data/SLIM_MSE-r-hat.npz') 
slim_mse.load_sim_matrix('raw_data/SLIM_MSE-sim-matrix.npz') 
slim_bpr.load_r_hat('raw_data/SLIM_BPR-r-hat.npz')
slim_bpr.load_sim_matrix('raw_data/SLIM_BPR-sim-matrix.npz')

rp3b    = RP3beta(URM_train)
rp3b.fit()
p3a     = P3alpha(URM_train)
p3a.fit()

ar1 = HybridSimilarity(URM_train, ucf, ucb)
ar1.fit(topK=310, alpha=0.96)

br1 = HybridSimilarity(URM_train, icf, icb)
br1.fit(topK=70, alpha=0.8)

cr1 = HybridSimilarity(URM_train, slim_mse, slim_bpr)
cr1.fit(topK=380, alpha=0.52, norm='none')

dr1 = HybridSimilarity(URM_train, rp3b, p3a)
dr1.fit(topK=490, alpha=.016)

H = Recommender(URM_train)
ar = ar1.r_hat
br = br1.r_hat
cr = cr1.r_hat
dr = dr1.r_hat
H.r_hat = 0.25*ar + 0.5*br + 0.2*cr + 0.05*dr


recs = [rp3b, dr1, H]
#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator

for r in recs:

    evaluator = Evaluator(r, URM_valid)
    evaluator.results()
   

            