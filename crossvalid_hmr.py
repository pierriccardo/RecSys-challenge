import numpy as np
import pandas as pd
import os
import scipy.sparse as sps
from matplotlib import pyplot
import configparser
from utils import *
from contextlib import redirect_stdout

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

#URM = d.URM
URM_train = d.get_URM_train()
URM_valid = d.get_URM_valid()
ICM = d.get_ICM()

#------------------------------
#       MODEL
#------------------------------
from sklearn.preprocessing import normalize
from recommenders.recommender       import Recommender
from recommenders.ItemKNNCF         import ItemKNNCF
from recommenders.ItemKNNCB         import ItemKNNCB
from recommenders.SLIM_MSE          import SLIM_MSE
from recommenders.HybridSimilarity  import HybridSimilarity
from recommenders.HybridMultiRhat   import HybridMultiRhat
from recommenders.HybridScores      import HybridScores
from recommenders.P3alpha           import P3alpha
from recommenders.RP3beta           import RP3beta
from recommenders.PureSVD           import PureSVD
from recommenders.SLIM_BPR          import SLIM_BPR
from recommenders.HybridRhat        import HybridRhat
from recommenders.UserKNNCF         import UserKNNCF
from recommenders.UserKNNCB         import UserKNNCB
from recommenders.IALS              import IALS

from evaluator                      import Evaluator
import similaripy as sim
from utils import cross_validate
import sys

datasets = d.k_fold_icm()

#------------------------------
# PARAMS
#------------------------------

BEST_MAP = 0.0
BEST_TOPK = 0
BEST_SHRINK = 0
BEST_SIM = ''

total = 100
message = '| vec: {} | avgMAP: {:.4f} |'

i = 0

filename = os.path.join('tuning', 'HMR-ICB-RP3-CROSSVALID.txt')

with open(filename, 'w') as f:
    with redirect_stdout(f):
        timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())
        print(timestamp)
        
        for i in range(total):

            list_precision = []
            list_recall = []
            list_MAP = []    

            np.random.seed(1)
            vec = np.random.dirichlet(np.ones(2),size=1)[0]

            for ds in datasets:
                cumulative_MAP = 0.0
                num_eval = 0

                train_ds = ds[0]
                valid_ds = ds[1]
                
                #------------------------------
                # RECOMMENDER
                #------------------------------
                
                r1 = RP3beta(train_ds)
                r1.fit()
                r2 = ItemKNNCB(train_ds, ICM)
                r2.fit()

                rec = HybridMultiRhat(train_ds, [r2, r1])
                rec.fit(vec, norm='l2')

                pbar = tqdm(range(valid_ds.shape[0]))
                for user_id in pbar:
                    
                    pbar.set_description('|{}| evaluating'.format(rec.NAME))
                    relevant_items = valid_ds.indices[valid_ds.indptr[user_id]:valid_ds.indptr[user_id+1]]
                    
                    if len(relevant_items)>0:
                        
                        recommended_items, k = rec.recommend(user_id, 10)
                        num_eval+=1
                        cumulative_MAP += MAP(recommended_items, relevant_items)
                
                cumulative_MAP /= num_eval  
                list_MAP.append(cumulative_MAP)

            avgmap = sum(list_MAP) / len(list_MAP)

            m = '|{}| iter: {:-5d}/{} '+ message
            print(m.format(rec.NAME, i, total, vec, avgmap))
            sys.stdout.flush()
            i+=1

            if avgmap > BEST_MAP:

                BEST_MAP = avgmap
                BEST_VEC = sim
                
        m = '|{}| best results ' + message
        print(m.format(rec.NAME, BEST_VEC, BEST_MAP))