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


CB = ItemKNNCB(URM_train, ICM)
CB.fit(topK=70, shrink=10)

CF = ItemKNNCF(URM_train)
CF.fit(topK=245, shrink=120)

#P3A = P3alpha(URM_train)
#P3A.fit(topK=500, alpha=0.549)
#
#SM = SLIM_MSE(URM_train)
#SM.fit(learning_rate=1e-4, epochs=50, samples=200000)

#CBCF = HybridScores(URM_train, CB, CF)
#CBCF.fit(alpha=0.09)

CBCF = RP3beta(URM_train)
CBCF.fit(topK=170, alpha=0.3, beta=0.070)

#CBCF_P3A = HybridScores(URM_train, CBCF, P3A)
#CBCF_P3A.fit(alpha=0.9)
#
#CBCFP3A_SM = HybridScores(URM_train, CBCF_P3A, SM)
#CBCFP3A_SM.fit(alpha=0.7)

recs = [CBCF]
#recs = [CB, CF, P3A, SM, CBCF, CBCF_P3A, CBCFP3A_SM]
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

#create_submission_csv(H2, test_set, config['paths']['results'])

            