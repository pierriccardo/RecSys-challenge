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
from sklearn.preprocessing import normalize
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

# ItemKNNCF
BEST_TOPK   = int(config['tuning.ItemKNNCF']['BEST_TOPK']) 
BEST_SHRINK = int(config['tuning.ItemKNNCF']['BEST_SHRINK'])
BEST_SIM    = config['tuning.ItemKNNCF']['BEST_SIM']
itemKNNCF = ItemKNNCF(URM_train)
itemKNNCF.fit(topK=BEST_TOPK, shrink=BEST_SHRINK)

BEST_TOPK   = int(config['tuning.ItemKNNCB']['BEST_TOPK']) 
BEST_SHRINK = int(config['tuning.ItemKNNCB']['BEST_SHRINK'])
BEST_SIM    = config['tuning.ItemKNNCB']['BEST_SIM']
itemKNNCB = ItemKNNCB(URM_train, ICM)
itemKNNCB.fit(topK=BEST_TOPK, shrink=BEST_SHRINK)

BEST_TOPK   = int(config['tuning.UserKNNCF']['BEST_TOPK']) 
BEST_SHRINK = int(config['tuning.UserKNNCF']['BEST_SHRINK'])
BEST_SIM    = config['tuning.UserKNNCF']['BEST_SIM']
userKNNCF = UserKNNCF(URM_train)
userKNNCF.fit(topK=BEST_TOPK, shrink=BEST_SHRINK)

BEST_TOPK   = int(config['tuning.UserKNNCB']['BEST_TOPK']) 
BEST_SHRINK = int(config['tuning.UserKNNCB']['BEST_SHRINK'])
BEST_SIM    = config['tuning.UserKNNCB']['BEST_SIM']
userKNNCB = UserKNNCB(URM_train, ICM)
userKNNCB.fit(topK=BEST_TOPK, shrink=BEST_SHRINK)

BEST_TOPK   = int(config['tuning.RP3beta']['BEST_TOPK'])
BEST_ALPHA  = float(config['tuning.RP3beta']['BEST_ALPHA'])
BEST_BETA   = float(config['tuning.RP3beta']['BEST_BETA'])
rp3beta = RP3beta(URM_train)
rp3beta.fit(alpha=BEST_BETA, beta=BEST_BETA, topK=BEST_TOPK)

BEST_TOPK   = int(config['tuning.P3alpha']['BEST_TOPK'])
BEST_ALPHA  = float(config['tuning.P3alpha']['BEST_ALPHA'])
p3alpha = P3alpha(URM_train)
p3alpha.fit(alpha=BEST_BETA, topK=BEST_TOPK)

single_recs = [itemKNNCB, itemKNNCF, userKNNCB, userKNNCF, rp3beta, p3alpha]


values = np.arange(0.0, 1.1, 0.05)
bestMAP = 0.0000
besta = 0.0
bestb = 0.0
bestc = 0.0
bestd = 0.0
beste = 0.0

total = 5 ** 5

H = Recommender(URM_train)

ar = normalize(itemKNNCF.r_hat, norm='l2', axis=1)
br = normalize(itemKNNCB.r_hat, norm='l2', axis=1)
cr = normalize(userKNNCF.r_hat, norm='l2', axis=1)
dr = normalize(userKNNCB.r_hat, norm='l2', axis=1)
er = normalize(p3alpha.r_hat, norm='l2', axis=1)
fr = normalize(rp3beta.r_hat, norm='l2', axis=1)
i = 0
for a in values:
    for b in values:
        for c in values:
            for d in values:
                for e in values:
                    if (a + b + c + d + e <= 1):
                        H.r_hat = a*ar + b*br + c*cr + d*dr + e*er + (1-a-b-c-d-e)*fr
                        evaluator = Evaluator(H, URM_valid)
                        log="|iter {:-5d}/{}|a: {:.4f} |b: {:.4f} |c: {:.4f}|d: {:.4f}|e: {:.4f}|MAP: {:.4f}|"
                        print(log.format(i, total,a, b, c, d, e, evaluator.cumulative_MAP))
                        if evaluator.cumulative_MAP > bestMAP:
                            bestMAP = evaluator.cumulative_MAP
                            besta = a
                            bestb = b
                            bestc = c
                            bestd = d
                            beste = e
                    i+=1

log="|best result |a: {:.4f} |b: {:.4f} |c: {:.4f}|d: {:.4f}|e: {:.4f}|MAP: {:.4f}|"
print(log.format(besta, bestb, bestc, bestd, beste, bestMAP))


'''
r1 = UserKNNCB(URM_train, ICM)
r1.fit(topK=35, shrink=0)

r2 = UserKNNCF(URM_train)
r2.fit(topK=65, shrink=45)

h1 = HybridSimilarity(URM_train, r1, r2)
h1.fit(topK=60, alpha=0.6)

h2 = HybridSimilarity(URM_train, r1, r2)
h2.fit(topK=60, alpha=0.2)

h3 = HybridSimilarity(URM_train, r1, r2)
h3.fit(topK=60, alpha=0.9)

recs = [h1, h2, h3]
'''
recs = []
for r in single_recs:
    recs.append(r)
#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator
for r in recs:

    evaluator = Evaluator(r, URM_valid)
    evaluator.results()
