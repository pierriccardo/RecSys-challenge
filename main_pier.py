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
#------------------------------
#       RECOMMENDER
#------------------------------
from recommenders.ItemKNNCF import ItemKNNCF
from recommenders.ItemKNNCB import ItemKNNCB
from recommenders.SLIM_MSE import SLIM_MSE
from recommenders.HybridSimilarity import HybridSimilarity
from recommenders.HybridScores import HybridScores
from recommenders.P3alpha import P3alpha
from recommenders.RP3beta import RP3beta 
from recommenders.SLIM_BPR import SLIM_BPR



CB = ItemKNNCB(URM_train, ICM)
CB.fit(topK=70, shrink=10)

CF = ItemKNNCF(URM_train)
CF.fit(topK=245, shrink=120)

P3A = P3alpha(URM_train)
P3A.fit(topK=500, alpha=0.549)

RP3B = RP3beta(URM_train)
RP3B.fit(topK=170, alpha=0.3, beta=0.070)

SM = SLIM_MSE(URM_train)
SM.fit(samples=200000, learning_rate=1e-4, epochs=50)

CBCF = HybridSimilarity(URM_train, CB.sim_matrix, CF.sim_matrix)
CBCF.fit(topK=65, alpha=0.09)

CBCF_P3A = HybridSimilarity(URM_train, CBCF.sim_matrix, P3A.sim_matrix)
CBCF_P3A.fit(topK=490, alpha=0.9)

print('tune CBCFP3A_SM')
CBCFP3A_SM = HybridSimilarity(URM_train, CBCF_P3A.sim_matrix, SM.sim_matrix)
CBCFP3A_SM.tuning(URM_valid)

print('tune SM_RP3B')
SM_RP3B = HybridSimilarity(URM_train, SM.sim_matrix, RP3B.sim_matrix)
SM_RP3B.tuning(URM_valid)

SB = SLIM_BPR(URM_train)
SB.fit()

print('tune CBCFP3A_RP3B')
CBCFP3A_SB = HybridSimilarity(URM_train, CBCF_P3A.sim_matrix, SB.sim_matrix)
CBCFP3A_SB.tuning(URM_valid)




recs = [CBCFP3A_SM]
#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator

for r in recs:

    evaluator = Evaluator(r, URM_valid)
    evaluator.results()
   
#------------------------------
#      CSV RESULTS CREATION
#------------------------------
from utils import create_submission_csv

create_submission_csv(H2, test_set, config['paths']['results'])

            