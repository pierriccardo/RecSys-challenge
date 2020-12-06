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
from recommenders.P3alphaRecommender import P3alphaRecommender
from recommenders.MF_MSE import MF_MSE


CB = ItemKNNCB(URM_train, ICM)
CB.fit(topK=70, shrink=10)

CF = ItemKNNCF(URM_train)
CF.fit(topK=245, shrink=120)

CBCF = HybridSimilarity(URM_train, CB.sim_matrix, CF.sim_matrix)
CBCF.fit(topK=65, alpha=0.09)

P3A = P3alphaRecommender(URM_train)
P3A.fit(topK=500, alpha=0.549)

H2 = HybridSimilarity(URM_train, CBCF.sim_matrix, P3A.sim_matrix)
H2.fit(topK=490, alpha=0.9)

H3 = HybridSimilarity(URM_train, SM.sim_matrix, H2.sim_matrix)
H3.tuning(URM_valid)


recs = [CB, CF, CBCF, P3A, H2]
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

            