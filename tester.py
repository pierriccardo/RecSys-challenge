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
from recommenders.P3alpha           import P3alpha
from recommenders.RP3beta           import RP3beta
from recommenders.FunkSVD           import FunkSVD
from recommenders.PureSVD           import PureSVD
from recommenders.SLIM_BPR          import SLIM_BPR
from recommenders.MF_BPR            import MF_BPR
from recommenders.HybridRhat        import HybridRhat

from evaluator                      import Evaluator


r1 = ItemKNNCF(URM_train, saverhat=True)
r1.load_r_hat('raw_data/RHAT-ItemKNNCF-08-12-2020-11:57:54.npz')

r2 = ItemKNNCB(URM_train, ICM, saverhat=True)
r2.load_r_hat('raw_data/RHAT-ItemKNNCB-08-12-2020-11:57:55.npz')

r3 = SLIM_MSE(URM_train, saverhat=True)
r3.load_r_hat('raw_data/RHAT-SLIM_MSE-08-12-2020-11:58:52.npz')

CBCF = HybridRhat(URM, r1.r_hat, r2.r_hat)
CBCF.fit(alpha=0.09)

recs = [CBCF]



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
