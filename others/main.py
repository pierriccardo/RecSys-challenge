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
from recommenders.IALS              import IALS
from recommenders.HybridMultiRhat   import HybridMultiRhat

# |iter:   474/2500 | vec: [0.02107824 0.37487624 0.21933112 0.05357673 0.03866059 0.0762722
# 0.21620488] | norm: l2 | MAP: 0.0750 |

recs = []

rSLIM_BPR = SLIM_BPR(URM_train)
rSLIM_BPR.fit()
rSLIM_BPR.save_r_hat()

rItemKNNCF = ItemKNNCF(URM_train)
rItemKNNCF.fit()
recs.append(rItemKNNCF)
rItemKNNCB = ItemKNNCB(URM_train, ICM)
rItemKNNCB.fit()
recs.append(rItemKNNCB)
rRP3beta = RP3beta(URM_train)
rRP3beta.fit()
recs.append(rRP3beta)
rP3alpha = P3alpha(URM_train)
rP3alpha.fit()
recs.append(rP3alpha)
rUserKNNCF = UserKNNCF(URM_train)
rUserKNNCF.fit()
recs.append(rUserKNNCF)
rUserKNNCB = UserKNNCB(URM_train, ICM)
rUserKNNCB.fit()
recs.append(rUserKNNCB)
rSLIM_MSE = SLIM_MSE(URM_train)
rSLIM_MSE.load_r_hat('raw_data/SLIM_MSE-r-hat.npz')
recs.append(rSLIM_MSE)
rSLIM_BPR = SLIM_BPR(URM_train)
rSLIM_BPR.load_r_hat('raw_data/SLIM_BPR-r-hat.npz')
recs.append(rSLIM_BPR)
rIALS = IALS(URM_train)
rIALS.fit()
recs.append(rIALS)

vec = [0.02146381, 0.16695917, 0.01549932, 0.01086414, 0.11211956, 0.16555224, 0.16055148, 0.06453604, 0.28245425]
r = HybridMultiRhat(URM_train, recs)
r.fit(vec)

recomm = [rSLIM_MSE, ]
'''
HMR(ItemKNNCF,ItemKNNCB,RP3beta,P3alpha,UserKNNCF,UserKNNCB,SLIM_MSE,SLIM_BPR,IALS)
 |iter:    43/2500 | vec: [0.03518162 0.36148473 0.0773163  0.0546652  0.09869039 0.04335913
 0.09798169 0.02526836 0.20605259] | norm: none | MAP: 0.0761 |

 |iter:    71/2500 | vec: [0.0068366  0.15018189 0.00792804 0.00724071 0.32836327 0.19991625
 0.03564293 0.05093109 0.21295923] | norm: none | MAP: 0.0765 |

 |iter:   115/2500 | vec: [0.02146381 0.16695917 0.01549932 0.01086414 0.11211956 0.16555224
 0.16055148 0.06453604 0.28245425] | norm: none | MAP: 0.0777 |
 '''
#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator

for r in recomm:

    evaluator = Evaluator(r, URM_valid)
    evaluator.results()

#------------------------------
#      CSV RESULTS CREATION
#------------------------------
from utils import create_submission_csv

create_submission_csv(r, test_set, config['paths']['results'])





            