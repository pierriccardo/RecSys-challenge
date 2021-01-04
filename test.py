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

datasets = d.k_fold()

BEST_MAP = 0.0
BEST_TOPK = 0
BEST_SHRINK = 0
BEST_SIM = ''

cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
cp.read('config.ini')

t = cp.getlist('tuning.ItemKNNCF', 'topKs') 
s = cp.getlist('tuning.ItemKNNCF', 'shrinks')
similarities = cp.getlist('tuning.ItemKNNCF', 'similarities')

topKs   = np.arange(int(t[0]), int(t[1]), int(t[2]))
shrinks = np.arange(int(s[0]), int(s[1]), int(s[2]))

total = len(topKs) * len(shrinks) * len(similarities)

i = 0
for sim in similarities:
    for t in topKs:
        for s in shrinks:

            list_precision = []
            list_recall = []
            list_MAP = []    

            for ds in datasets:
                cumulative_precision = 0.0
                cumulative_recall = 0.0
                cumulative_MAP = 0.0
                num_eval = 0

                train_ds = ds[0]
                valid_ds = ds[1]

                rec = UserKNNCF(train_ds)
                rec.fit(topK=t, sim_type=sim, shrink=s)

                pbar = tqdm(range(valid_ds.shape[0]))
                for user_id in pbar:
                    
                    pbar.set_description('|{}| evaluating'.format(rec.NAME))
                    relevant_items = valid_ds.indices[valid_ds.indptr[user_id]:valid_ds.indptr[user_id+1]]
                    
                    if len(relevant_items)>0:
                        
                        recommended_items = rec.recommend(user_id, 10)
                        num_eval+=1

                        cumulative_precision += precision(recommended_items, relevant_items)
                        cumulative_recall += recall(recommended_items, relevant_items)
                        cumulative_MAP += MAP(recommended_items, relevant_items)
                
                cumulative_precision /= num_eval
                cumulative_recall /= num_eval
                cumulative_MAP /= num_eval  
                
                list_precision.append(cumulative_precision)
                list_recall.append(cumulative_recall)
                list_MAP.append(cumulative_MAP)

            #print('[avg precision:  {:.4f}]'.format(sum(list_precision) / len(list_precision)))
            #print('[avg recall:     {:.4f}]'.format(sum(list_recall) / len(list_recall)))
            avgmap = sum(list_MAP) / len(list_MAP)

            m = '|{}| iter: {:-5d}/{} | topk: {:-3d} | shrink: {:-3d} | sim type: {} | avgMAP: {:.4f} |'
            print(m.format(rec.NAME, i, total, t, s, sim, avgmap))
            sys.stdout.flush()
            i+=1

            if avgmap > BEST_MAP:

                BEST_TOPK = t
                BEST_SHRINK = s
                BEST_MAP = avgmap
                BEST_SIM = sim
        
m = '|{}| best results | topk: {:-3d} | shrink: {:-3d} | sim type: {} | MAP: {:.4f} |'
print(m.format(rec.NAME, BEST_TOPK, BEST_SHRINK, BEST_SIM, BEST_MAP))



#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator
for r in recs:

    evaluator = Evaluator(r, URM_valid)
    evaluator.results()
