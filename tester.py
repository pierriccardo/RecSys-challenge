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
from recommenders.P3alpha           import P3alphaRecommender
from recommenders.RP3beta           import RP3beta
from recommenders.FunkSVD           import FunkSVD
from recommenders.PureSVD           import PureSVD
from recommenders.SLIM_BPR          import SLIM_BPR
from recommenders.MF_BPR            import *
from recommenders.MF_BPR            import MF_BPR

from evaluator                      import Evaluator


#r2 = MF_BPR(URM_train, urm_train_df, n_factors=30, loadmodel=False, savemodel=False)
#r2.fit(epochs=20, lr=1e-4)
#
#r1 = PureSVD(URM_train)
#r1.fit()

#r3 = ItemKNNCF(URM_train)
#r3.fit(topK=245, shrink=120)

recs = []

rs = next(os.walk('raw_data'))[2]
for filename in rs:
    r = Recommender(URM_train)
    r.load_r_hat('raw_data/'+ filename)
    recs.append(r)

hybrid = []


new_r = Recommender(URM_train)
new_r.r_hat = recs[0].r_hat * 0.5 + recs[1].r_hat * 0.5

new_r2 = Recommender(URM_train)
new_r2.r_hat = recs[0].r_hat * 0.95 + recs[1].r_hat * 0.05

from sklearn.preprocessing import normalize
import similaripy as sim
new_r3 = Recommender(URM_train)
new_r3.r_hat = normalize(recs[0].r_hat) * 0.95 + normalize(recs[1].r_hat) * 0.05

recs.append(new_r)
recs.append(new_r2)
recs.append(new_r3)

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

#create_submission_csv(recommender, test_set, config['paths']['results'])

            

if __name__ == "__main__":

    
    pass
