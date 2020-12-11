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
URM_train   = d.URM #d.get_URM_train()
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
from recommenders.P3alpha import P3alpha
from recommenders.RP3beta import RP3beta 
from recommenders.SLIM_BPR import SLIM_BPR

CB = ItemKNNCB(URM_train, ICM)
CB.fit()

#CF = ItemKNNCF(URM_train)
#CF.fit()

H = HybridSimilarity(URM_train, CB, CB)
H.fit()

P3A = P3alpha(URM_train)
P3A.fit()

H2 = HybridSimilarity(URM_train, H, P3A)
H2.fit(topK=350, alpha=0.5)
'''
RP3 = RP3beta(URM_train)
RP3.fit(topK=50, alpha=0.25, beta=0.080)

H3 = HybridSimilarity(URM_train, RP3.sim_matrix, P3A.sim_matrix)
H3.fit(topK=490, alpha=0.1)



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


recs = [CBCFP3A_SM]
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

'''

recs = [H2]

#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator

#for r in recs:
#
#    evaluator = Evaluator(r, URM_valid)
#    evaluator.results()

#------------------------------
#      CSV RESULTS CREATION
#------------------------------
from utils import create_submission_csv

create_submission_csv(H2, test_set, config['paths']['results'])

            
