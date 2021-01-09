import numpy as np
import pandas as pd
import os
import scipy.sparse as sps
from matplotlib import pyplot
import configparser
from utils import *

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

similarities = np.arange(0.0, 0.1, 0.02)
topKs   = np.arange(250, 400, 20)
shrinks = np.arange(0.3, 0.49, 0.05) 

total = len(topKs) * len(shrinks) * len(similarities)
message = '| topk: {} | alpha: {:.3f} | beta: {:.3f} | avgMAP: {:.4f} |'

i = 0
for sim in similarities:
    for t in topKs:
        for s in shrinks:

            list_precision = []
            list_recall = []
            list_MAP = []    

            for ds in datasets:
                cumulative_MAP = 0.0
                num_eval = 0

                train_ds = ds[0]
                valid_ds = ds[1]
                
                #------------------------------
                # RECOMMENDER
                #------------------------------
                
                rec = RP3beta(train_ds)
                rec.fit(topK=t, alpha=s, beta=sim)

                pbar = tqdm(range(valid_ds.shape[0]))
                for user_id in pbar:
                    
                    pbar.set_description('|{}| evaluating'.format(rec.NAME))
                    relevant_items = valid_ds.indices[valid_ds.indptr[user_id]:valid_ds.indptr[user_id+1]]
                    
                    if len(relevant_items)>0:
                        
                        recommended_items = rec.recommend(user_id, 10)
                        num_eval+=1
                        cumulative_MAP += MAP(recommended_items, relevant_items)
                
                cumulative_MAP /= num_eval  
                list_MAP.append(cumulative_MAP)

            avgmap = sum(list_MAP) / len(list_MAP)

            m = '|{}| iter: {:-5d}/{} '+ message
            print(m.format(rec.NAME, i, total, t, s, sim, avgmap))
            sys.stdout.flush()
            i+=1

            if avgmap > BEST_MAP:

                BEST_TOPK = t
                BEST_SHRINK = s
                BEST_MAP = avgmap
                BEST_SIM = sim
        
m = '|{}| best results ' + message
print(m.format(rec.NAME, BEST_TOPK, BEST_SHRINK, BEST_SIM, BEST_MAP))