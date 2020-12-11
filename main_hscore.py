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
from recommenders.ItemKNNCF import ItemKNNCF
from recommenders.ItemKNNCB import ItemKNNCB
from recommenders.SLIM_MSE import SLIM_MSE
from recommenders.HybridSimilarity import HybridSimilarity
from recommenders.HybridScores import HybridScores
from recommenders.P3alpha import P3alpha
from recommenders.RP3beta import RP3beta
from recommenders.UserKNNCB import UserKNNCB 
from recommenders.MF_BPR import MF_BPR


r = MF_BPR(URM_train,train_df)
r.fit(URM_valid)

recs = [r]
#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator

for r in recs:

    evaluator = Evaluator(r, URM_valid)
    evaluator.results()
   

            