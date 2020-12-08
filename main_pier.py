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
from recommenders.recommender import Recommender
from recommenders.ItemKNNCF import ItemKNNCF
from recommenders.ItemKNNCB import ItemKNNCB
from recommenders.SLIM_MSE import SLIM_MSE
from recommenders.HybridSimilarity import HybridSimilarity
from recommenders.HybridScores import HybridScores
from recommenders.P3alpha import P3alphaRecommender
from recommenders.RP3beta import RP3beta
from recommenders.SLIM_BPR import SLIM_BPR
#from recommenders.MF_MSE import MF_MSE

SM = SLIM_MSE(URM_train)
SM.fit()

BPR = SLIM_BPR(URM_train)
BPR.fit(topK=200, epochs=250, lambda_i=0.075, lambda_j=0.0075, lr=0.0005)


P3A = P3alphaRecommender(URM_train)
P3A.fit(topK=500, alpha=0.549)

#H2 = HybridSimilarity(URM_train, CBCF.sim_matrix, P3A.sim_matrix)
#H2.fit(topK=490, alpha=0.9)

RP3 = RP3beta(URM_train)
RP3.fit(topK=50, alpha=0.25, beta=0.080)

#H3 = HybridSimilarity(URM_train, RP3.sim_matrix, P3A.sim_matrix)
#H3.fit(topK=490, alpha=0.1)

CB = ItemKNNCB(URM_train, ICM)
CB.fit(topK=70, shrink=10)

CF = ItemKNNCF(URM_train)
CF.fit(topK=245, shrink=120)

CBCF = HybridSimilarity(URM_train, CB.sim_matrix, CF.sim_matrix)
CBCF.fit(topK=65, alpha=0.09)

#H4 = HybridSimilarity(URM_train, H3.sim_matrix, CBCF.sim_matrix)
#H4.tuning(URM_valid)
from evaluator import Evaluator

#best param so far a=0.3, b=0.1, c=0, d=0.4 | MAP = 0.06937

H = Recommender(URM_train)
values = np.arange(0.0, 1.1, 0.1)
bestMAP = 0.0000
bestalfa = 0.0
bestbeta = 0.0
bestc = 0.0
bestd = 0.0
for a in values:
    for b in values:
        for c in values:
            for d in values:
                H.r_hat = a*CBCF.r_hat + b*BPR.r_hat + c*P3A.r_hat + d*RP3.r_hat + (1-a-b-c-d)*SM.r_hat
                if (a + b + c + d <= 1):
                    evaluator = Evaluator(H, URM_valid)
                    print("alfa=" + str(a) + "beta=" + str(b) + "c=" + str(c) + "d=" + str(d) + " | MAP = " + str(evaluator.cumulative_MAP))
                    if evaluator.cumulative_MAP > bestMAP:
                        bestMAP = evaluator.cumulative_MAP
                        bestalfa = a
                        bestbeta = b
                        bestc = c
                        bestd = d

print("best alfa=" + str(bestalfa) + "beta=" + str(bestbeta) + "c=" + str(bestc) + "d=" + str(bestd) + " | MAP = " + str(bestMAP))

#recs = [H]
#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator
'''
for r in recs:

    evaluator = Evaluator(r, URM_valid)
    evaluator.results()
'''
#------------------------------
#      CSV RESULTS CREATION
#------------------------------
from utils import create_submission_csv

#create_submission_csv(H2, test_set, config['paths']['results'])

            
