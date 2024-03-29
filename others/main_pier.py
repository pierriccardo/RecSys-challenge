import numpy as np
import pandas as pd
import os
import scipy.sparse as sps
from matplotlib import pyplot
import configparser
from tqdm import tqdm
import similaripy as sim
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
from sklearn.preprocessing import normalize
from recommenders.recommender import Recommender
from recommenders.ItemKNNCF import ItemKNNCF
from recommenders.ItemKNNCB import ItemKNNCB
from recommenders.SLIM_MSE import SLIM_MSE
from recommenders.HybridSimilarity import HybridSimilarity
from recommenders.HybridScores import HybridScores
from recommenders.P3alpha import P3alpha
from recommenders.RP3beta import RP3beta 
from recommenders.SLIM_BPR import SLIM_BPR
from recommenders.MF_IALS import MF_IALS
from recommenders.MF_NN import MF_NN
from recommenders.EASE_R import EASE_R 
from recommenders.ICM_SVD import ICM_SVD
from recommenders.IALS import IALS
from recommenders.UserKNNCB import UserKNNCB
from recommenders.UserKNNCF import UserKNNCF

#r = SLIM_BPR(URM_train)
#r = SLIM_MSE(URM_train)
r = UserKNNCB(URM_train, ICM)
#r.fit()
#r.save_r_hat()
r.load_r_hat('raw_data/UserKNNCB-r-hat-valid.npz')
recs = [r]

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

            
