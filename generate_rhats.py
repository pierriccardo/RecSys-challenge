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
from evaluator                      import Evaluator

'''
| KNNCF     | topk: 245 | shrink: 120 | sim type: cosine  | MAP: 0.0471 |
| KNNCB     | topk: 70  | shrink: 10  | sim type: cosine  | MAP: 0.0319 |
| RP3B      | topk: 50  | alpha: 0.25 | beta: 0.080       | MAP: 0.0517 |
| P3A       | topk: 500 | alpha: 0.55 |                   | MAP: 0.0571 |
| SLIMBPR   | topk: 200 | epochs: 250 | lr = 0.0005       | MAP: 0.0514 | lambda_i = 0.0075 | lambda_j = 0.00075 |
| SLIMMSE   | lr = 1e-4 | epochs: 50  | samples: 200000   | MAP: 0.0520 |
'''

folder = input('insert folder path to store r_hats (or press enter to store in raw_data): ')

import os
if folder == '':
    folder = 'raw_data'
elif not os.path.exists(folder):
    os.makedirs(folder)
    print('folder created ...')


r1 = ItemKNNCF(URM_train, saverhat=True)
r1.r_hat_folder = folder
r1.fit(topK=245, shrink=120)

r2 = ItemKNNCB(URM_train, ICM, saverhat=True)
r2.r_hat_folder = folder
r2.fit(topK=70, shrink=10)

r3 = SLIM_MSE(URM_train, saverhat=True)
r3.r_hat_folder = folder
r3.fit(learning_rate=1e-4, epochs=50, samples=200000)


recs = [r1, r2, r3]

#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator
for r in recs:

    evaluator = Evaluator(r, URM_valid)
    evaluator.results()

