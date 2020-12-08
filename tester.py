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

URM = d.URM
URM_train = d.get_URM_train()
URM_valid = d.get_URM_valid()
ICM = d.get_ICM()
test_set = d.get_test()

urm_train_df = d.urm_train_df

#------------------------------
#       MODEL
#------------------------------
from recommenders.recommender       import Recommender
from recommenders.ItemKNNCF         import ItemKNNCF
from recommenders.ItemKNNCB         import ItemKNNCB
from recommenders.SLIM_MSE          import SLIM_MSE
from recommenders.HybridSimilarity  import HybridSimilarity
from recommenders.HybridScores      import HybridScores
from recommenders.P3alpha           import P3alpha
from recommenders.RP3beta           import RP3beta
from recommenders.FunkSVD           import FunkSVD
from recommenders.PureSVD           import PureSVD
from recommenders.SLIM_BPR          import SLIM_BPR
from recommenders.MF_BPR            import MF_BPR
from recommenders.HybridRhat        import HybridRhat
from recommenders.UserKNNCF         import UserKNNCF
from recommenders.UserKNNCB         import UserKNNCB

from evaluator                      import Evaluator


#r1 = ItemKNNCF(URM_train)
#r1.load_r_hat('raw_data/RHAT-ItemKNNCF-08-12-2020-11:57:54.npz')
#
#r2 = ItemKNNCB(URM_train, ICM)
#r2.load_r_hat('raw_data/RHAT-ItemKNNCB-08-12-2020-11:57:55.npz')
#
#r3 = SLIM_MSE(URM_train)
#r3.load_r_hat('raw_data/RHAT-SLIM_MSE-08-12-2020-11:58:52.npz')
#  
#r4 = RP3beta(URM_train)
#r4.load_r_hat('raw_data/RHAT-RP3beta-08-12-2020-15:22:19.npz')
#
#r5 = P3alpha(URM_train)
#r5.load_r_hat('raw_data/RHAT-P3alpha-08-12-2020-15:22:35.npz')
#
#
#
#CBCF = HybridRhat(URM_train, r1, r2)
#CBCF.fit(alpha=0.61)
#
#SMCBCF = HybridRhat(URM_train, r3, CBCF)
#SMCBCF.fit(alpha=0.35)
#
#r6 = HybridRhat(URM_train, SMCBCF, r4)
#r6.fit(alpha=0.8)
#
#r7 = HybridRhat(URM_train, r6, r5)
#r7.fit(0.55)
#
#r8 = HybridRhat(URM_train, CBCF, r4)
#r8.tuning(URM_valid)

r1 = UserKNNCB(URM_train, ICM)
r1.fit()



recs = [r1]





#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator
for r in recs:

    evaluator = Evaluator(r, URM_valid)
    evaluator.results()
