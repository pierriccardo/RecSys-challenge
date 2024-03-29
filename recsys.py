import numpy as np
import pandas as pd
import os
import scipy.sparse as sps
from matplotlib import pyplot
import configparser
from sklearn.preprocessing import normalize
from tqdm import tqdm
import argparse
import os
from contextlib import redirect_stdout
from time import strftime, gmtime
from utils import *
from similarity.similarity import similarity



#------------------------------
#       PARSER
#------------------------------
parser = argparse.ArgumentParser(description='Recsys main.')
parser.add_argument('-f', '--folder', type=str, default='tuning')
parser.add_argument('-t', '--test', action='store_true', default=False)
parser.add_argument('-lr', '--loadrhat', action='store_true', default=False)

args = parser.parse_args()
if not os.path.exists(args.folder):
    os.mkdir(args.folder)

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

if args.test:
    URM_train = d.get_URM()
    URMICM_train = d.get_URMICM()
    warning('NB: Complete URM loaded')

else:
    URM_train = d.get_URM_train()
    URMICM_train = d.get_URMICM_train()
    warning('NB: Train URM loaded')
    
URM_valid    = d.get_URM_valid()
ICM          = d.get_ICM()
ICM_selected = d.get_ICM_selected()
test_set     = d.get_test()

#------------------------------
#       RECOMMENDERS
#------------------------------
from recommenders.recommender       import Recommender
from recommenders.TopPop            import TopPop
from recommenders.ItemKNNCF         import ItemKNNCF
from recommenders.ItemKNNCB         import ItemKNNCB
from recommenders.P3alpha           import P3alpha
from recommenders.RP3beta           import RP3beta
from recommenders.UserKNNCF         import UserKNNCF
from recommenders.UserKNNCB         import UserKNNCB
from recommenders.SLIM_MSE          import SLIM_MSE
from recommenders.SLIM_BPR          import SLIM_BPR
from recommenders.SLIM_ELN          import SLIM_ELN
from recommenders.PureSVD           import PureSVD
from recommenders.IALS              import IALS
from recommenders.MF_BPR            import MF_BPR

#------------------------------
#       HYBRIDS
#------------------------------
from recommenders.HybridCluster     import HybridCluster
from recommenders.HybridSimilarity  import HybridSimilarity
from recommenders.HybridScores      import HybridScores
from recommenders.HybridRhat        import HybridRhat
from recommenders.HybridMultiRhat   import HybridMultiRhat
from recommenders.HybridMultiSim    import HybridMultiSim
from recommenders.HybridRhat        import HybridRhat

from evaluator                      import Evaluator


def tune_and_log(h, filename):
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())
            print(timestamp)
            h.tuning(URM_valid)

def fit_and_save(r):
    r.fit()
    try: 
        r.save_r_hat(test=args.test)
    except:
        msg = '|{}|  failed to save r-hat'.format(r.NAME)
        error(msg)
    try: 
        r.save_sim_matrix(test=args.test)
    except:
        msg = '|{}| rec with no sim-matrix or failed to save'.format(r.NAME)
        error(msg)
    return r

def fit_or_load(r, matrix='r-hat'):
    """
    try to load a given matrix for the recommender, if
    the matrix is not present in the folder raw data then
    will fit the recommender and save the matrix

    Args:
        r (Recommender): recommender
        matrix (str, optional): define the matrix type: 'r-hat' or 'sim-matrix'.
        type (str, optional): type of matrix to save 'valid' or 'test'.
    """
    matrix_type = 'test' if args.test else 'valid'
    filename = 'raw_data/{}-{}-{}'.format(r.NAME, matrix, matrix_type)
    if args.loadrhat:
        try:
            if matrix == 'r-hat':
                try: r.load_r_hat(filename + '.npy')
                except: r.load_r_hat(filename + '.npz')
                msg = '|{}|  rhat loaded '.format(r.NAME)
                success(msg)
            elif matrix == 'sim-matrix':
                try: r.load_sim_matrix(filename + '.npy')
                except: r.load_sim_matrix(filename + '.npz')
                msg = '|{}|  sim-matrix loaded '.format(r.NAME)
                success(msg)
        except:
            msg = '|{}|  no rhat file found, proceeding with fit '.format(r.NAME)
            warning(msg)
            r = fit_and_save(r)
    else:
        r = fit_and_save(r)
        
        
    return r

#------------------------------
# CONSOLE
#------------------------------

print('')
print('   ██████╗ ███████╗ ██████╗███████╗██╗   ██╗███████╗')
print('   ██╔══██╗██╔════╝██╔════╝██╔════╝╚██╗ ██╔╝██╔════╝')
print('   ██████╔╝█████╗  ██║     ███████╗ ╚████╔╝ ███████╗')
print('   ██╔══██╗██╔══╝  ██║     ╚════██║  ╚██╔╝  ╚════██║')
print('   ██║  ██║███████╗╚██████╗███████║   ██║   ███████║')
print('   ╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝   ╚═╝   ╚══════╝')
print(Fore.BLACK + Back.GREEN)
print('   Choose a list of algorithms:                     ')
print(Style.RESET_ALL + Fore.GREEN)    
print('  press:                                            ')
print('   [0]  --> TopPop')
print('   [1]  --> ItemKNNCF')
print('   [2]  --> ItemKNNCB')
print('   [3]  --> RP3beta')
print('   [4]  --> P3alpha')
print('   [5]  --> UserKNNCF')
print('   [6]  --> UserKNNCB')
print('   [7]  --> SLIM_MSE')
print('   [8]  --> SLIM_BPR')
print('   [9]  --> PureSVD')
print('   [10] --> IALS')
print('   [11] --> SLIM_ELN')
print('   [12] --> MF_BPR')
print('   [c]  --> Hybrid Cluster')
print('')
choice = input(Fore.BLUE + Back.WHITE + ' -> ' + Style.RESET_ALL)
list = choice.split()

msg = '   Choose an action:                                                     '
info(msg)

print('   tunehs        --> tune a hybrid with 2 algorithms with sim matrix          ')
print('   evalhs        --> eval a hybrid with 2 algorithms with sim matrix          ')
print('   hms           --> tune a hybrid with n algorithms by sim matrix, random val')
print('')       
print('   hmr           --> tune a hybrid with n algorithms by r hat, random val     ')
print('   subhmr        --> submit a hybrid multi rhat algorithm                     ')
print('   evalhmr       --> evaluate a hybrid multi rhat recommender                 ')
print('')       
print('   tune          --> tune the choosen algorithms                              ')
print('   eval          --> evaluate the selected algorithms                         ')
print('')
c = input(Fore.BLUE + Back.WHITE + ' -> ' + Style.RESET_ALL)

#------------------------------
# ALGORITHMS LIST CREATION
#------------------------------

recs = []

for e in list:

    if e == '0':
        r = TopPop(URM_train)
        recs.append(r)

    if e == '1':
        r = ItemKNNCF(URMICM_train)
        recs.append(r)

    elif e == '2':
        r = ItemKNNCB(URM_train, ICM)
        recs.append(r)

    elif e == '3':
        r = RP3beta(URMICM_train)
        recs.append(r)

    elif e == '4':
        r = P3alpha(URMICM_train)
        recs.append(r)

    elif e == '5':
        r = UserKNNCF(URMICM_train)
        recs.append(r)

    elif e == '6':
        r = UserKNNCB(URM_train, ICM)
        recs.append(r)
    
    elif e == '7':
        r = SLIM_MSE(URM_train)
        recs.append(r)
    
    elif e == '8':
        r = SLIM_BPR(URM_train)
        recs.append(r)
    
    elif e == '9':
        r = PureSVD(URM_train)
        recs.append(r)

    elif e == '10':
        r = IALS(URMICM_train)
        recs.append(r)
    
    elif e == '11':
        r = SLIM_ELN(URM_train)
        recs.append(r)
    
    elif e == '12':
        r = MF_BPR(URMICM_train)
        recs.append(r)

    elif e == 'c':
        r = HybridCluster(URM_train, URMICM_train, ICM, URM_valid, is_test=args.test)
        recs.append(r)

    else:
        print("wrong insertion, skipped")

#------------------------------
# TRAIN OR LOAD SELECTED RECS
#------------------------------

for r in recs:
    
    if c == 'hms':
        r = fit_or_load(r, matrix='sim-matrix')
    else:
        r = fit_or_load(r)

if c == 'tunehs':
    h = HybridSimilarity(URM_train, recs[0], recs[1])
    filename = os.path.join(args.folder, '{}-TUNING.txt'.format(h.NAME))
    tune_and_log(h, filename)

if c == 'evalhs':
    print('insert alpha:')
    a = input(Fore.BLUE + Back.WHITE + ' -> ' + Style.RESET_ALL)

    h = HybridSimilarity(URM_train, recs[0], recs[1])
    h.fit(alpha=float(a))
    evaluator = Evaluator(h, URM_valid)
    evaluator.results()

elif c == 'hms':
    h = HybridMultiSim(URM_train, recs)
    filename = os.path.join(args.folder, '{}-TUNING.txt'.format(h.NAME))
    tune_and_log(h, filename)

elif c == 'hmr':
    h = HybridMultiRhat(URM_train, recs)
    filename = os.path.join(args.folder, '{}-TUNING.txt'.format(h.NAME))
    tune_and_log(h, filename)
    
elif c == 'tune':
    for r in recs:
        filename = os.path.join(args.folder, '{}-TUNING.txt'.format(r.NAME))
        tune_and_log(r, filename)

elif c == 'eval':
    for r in recs:
        evaluator = Evaluator(r, URM_valid)
        evaluator.results()

        #create_submission_csv(r, test_set, config['paths']['results'])
    
elif c == 'subhmr':
    vec_input = input('Insert the value vector:')
    list = vec_input.split()
    vec = [float(ele) for ele in list] 

    h = HybridMultiRhat(URM_train, recs)
    h.fit(vec)
    create_submission_csv(h, test_set, config['paths']['results'])

elif c == 'evalhmr':
    vec_input = input('Insert the value vector:')
    list = vec_input.split()
    vec = [float(ele) for ele in list] 

    h = HybridMultiRhat(URM_train, recs)
    h.fit(vec)
    evaluator = Evaluator(h, URM_valid)
    evaluator.results()

elif c == 'sub':
    for r in recs:
        create_submission_csv(r, test_set, config['paths']['results'])

else:
    print('wrong selection')

