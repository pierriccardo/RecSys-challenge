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
from colorama import Fore, Back, Style
from utils import create_submission_csv
from utils import *

parser = argparse.ArgumentParser(description='Recsys main.')
parser.add_argument('-f', '--folder', type=str, default='tuning')
parser.add_argument('-t', '--test', action='store_true', default=False)
parser.add_argument('-lr', '--loadrhat', action='store_true', default=False)

def warning(msg):
    print(Back.YELLOW + Fore.BLACK + msg + Style.RESET_ALL)

def success(msg):
    print(Back.GREEN + Fore.BLACK + msg + Style.RESET_ALL)

def error(msg):
    print(Back.RED + Fore.BLACK + msg + Style.RESET_ALL)

def info(msg):
    print(Back.BLUE + Fore.BLACK + msg + Style.RESET_ALL)



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
    URM_train = d.URM
    msg = '   NB: Complete URM loaded                          '
    warning(msg)

else:
    URM_train = d.URMICM.tocsr()#d.get_URM_train()
    msg = '   NB: Train URM loaded                             '
    warning(msg)
    
URM_valid   = d.get_URM_valid()
ICM         = d.get_ICM()
test_set    = d.get_test()

#------------------------------
#       MODEL
#------------------------------

from recommenders.recommender       import Recommender
from recommenders.ItemKNNCF         import ItemKNNCF
from recommenders.ItemKNNCB         import ItemKNNCB
from recommenders.SLIM_MSE          import SLIM_MSE
from recommenders.HybridSimilarity  import HybridSimilarity
from recommenders.HybridScores      import HybridScores
from recommenders.HybridRhat        import HybridRhat
from recommenders.HybridMultiRhat   import HybridMultiRhat
from recommenders.P3alpha           import P3alpha
from recommenders.RP3beta           import RP3beta
from recommenders.PureSVD           import PureSVD
from recommenders.SLIM_BPR          import SLIM_BPR
from recommenders.HybridRhat        import HybridRhat
from recommenders.UserKNNCF         import UserKNNCF
from recommenders.UserKNNCB         import UserKNNCB
from recommenders.IALS              import IALS
from recommenders.HybridMultiSim    import HybridMultiSim
from recommenders.SLIM_ELN          import SLIM_ELN
from recommenders.MF_BPR            import MF_BPR

from evaluator                      import Evaluator

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
# BASIC RECOMMENDERS
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
print('   press 1  --> ItemKNNCF')
print('   press 2  --> ItemKNNCB')
print('   press 3  --> RP3beta')
print('   press 4  --> P3alpha')
print('   press 5  --> UserKNNCF')
print('   press 6  --> UserKNNCB')
print('   press 7  --> SLIM_MSE')
print('   press 8  --> SLIM_BPR')
print('   press 9  --> PureSVD')
print('   press 10 --> IALS')
print('   press 11 --> SLIM_ELN')
print('   press 12 --> MF_BPR')
print('')
print('   press h1 --> Hsim(ItemKNNCF, ItemKNNCB)')
print('   press h2 --> Hsim(UserKNNCF, UserKNNCB)')
print('   press h3 --> Hsim(RP3beta, ItemKNNCB)')
print('   press h4 --> Hsim(P3alpha, ItemKNNCB)')
print('   press h5 --> Hsim(ItemKNNCB, SLIM_MSE)')
print('   press h6 --> Hsim(ItemKNNCB, SLIM_BPR)')
print('   press h7 --> Hsim(ItemKNNCB, SLIM_ELN)')
print('   press h8 --> HMR(ItemKNNCB,UserKNNCB)')
print('')
choice = input(Fore.BLUE + Back.WHITE + ' -> ' + Style.RESET_ALL)
list = choice.split()

msg = '   Choose an action:                                                     '
info(msg)

print('   tunehs        --> tune a hybrid with 2 algorithms with sim matrix          ')
print('   evalhs        --> eval a hybrid with 2 algorithms with sim matrix          ')
print('')
print('   hrhat         --> tune a hybrid with 2 algorithms with r hat               ')
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
hybrids = []

for e in list:

    #------------------------------
    # RECOMMENDERS
    #------------------------------

    if e == '1':
        r = ItemKNNCF(URM_train)
        recs.append(r)

    elif e == '2':
        r = ItemKNNCB(URM_train, ICM)
        recs.append(r)

    elif e == '3':
        r = RP3beta(URM_train)
        recs.append(r)

    elif e == '4':
        r = P3alpha(URM_train)
        recs.append(r)

    elif e == '5':
        r = UserKNNCF(URM_train)
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
        r = IALS(URM_train)
        recs.append(r)
    
    elif e == '11':
        r = SLIM_ELN(URM_train)
        recs.append(r)
    
    elif e == '12':
        r = MF_BPR(URM_train)
        recs.append(r)

    #------------------------------
    # HYBRIDS
    #------------------------------
    elif e == 'h1':
        
        r1 = ItemKNNCF(URM_train)
        r2 = ItemKNNCB(URM_train, ICM)
        fit_or_load(r1, matrix='sim-matrix')
        fit_or_load(r2, matrix='sim-matrix')

        a = 0.45
        h1 = HybridSimilarity(URM_train, r1, r2)
        h1.fit(alpha=a)
        
        hybrids.append(h1)

    elif e == 'h2':
        
        r1 = UserKNNCF(URM_train)
        r2 = UserKNNCB(URM_train, ICM)
        fit_or_load(r1, matrix='sim-matrix')
        fit_or_load(r2, matrix='sim-matrix')

        a = 0.7
        h2 = HybridSimilarity(URM_train, r1, r2)
        h2.fit(alpha=a)
        
        hybrids.append(h2)
    
    elif e == 'h3':
        
        r1 = RP3beta(URM_train)
        r2 = ItemKNNCB(URM_train, ICM)
        fit_or_load(r1, matrix='sim-matrix')
        fit_or_load(r2, matrix='sim-matrix')

        a = 0.5
        h3 = HybridSimilarity(URM_train, r1, r2)
        h3.fit(alpha=a)
        
        hybrids.append(h3)

    elif e == 'h4':
        
        r1 = P3alpha(URM_train)
        r2 = ItemKNNCB(URM_train, ICM)
        fit_or_load(r1, matrix='sim-matrix')
        fit_or_load(r2, matrix='sim-matrix')

        a = 0.5
        h4 = HybridSimilarity(URM_train, r1, r2)
        h4.fit(alpha=a)
        
        hybrids.append(h4)

    elif e == 'h5':
        
        r1 = ItemKNNCB(URM_train, ICM)
        r2 = SLIM_MSE(URM_train)
        fit_or_load(r1, matrix='sim-matrix')
        fit_or_load(r2, matrix='sim-matrix')

        a = 0.4
        h5 = HybridSimilarity(URM_train, r1, r2)
        h5.fit(alpha=a)
        
        hybrids.append(h5)

    elif e == 'h6':
        
        r1 = ItemKNNCB(URM_train, ICM)
        r2 = SLIM_BPR(URM_train)
        fit_or_load(r1, matrix='sim-matrix')
        fit_or_load(r2, matrix='sim-matrix')

        a = 0.6
        h6 = HybridSimilarity(URM_train, r1, r2)
        h6.fit(alpha=a)
        
        hybrids.append(h6)

    elif e == 'h7':
        
        r1 = ItemKNNCB(URM_train, ICM)
        r2 = SLIM_ELN(URM_train)
        fit_or_load(r1, matrix='sim-matrix')
        fit_or_load(r2, matrix='sim-matrix')

        a = 0.5
        h7 = HybridSimilarity(URM_train, r1, r2)
        h7.fit(alpha=a)
        
        hybrids.append(h7)
    
    elif e == 'h8':
        vec = [0.187771, 0.812229]
        r1 = ItemKNNCB(URM_train, ICM)
        r2 = UserKNNCB(URM_train, ICM)
        fit_or_load(r1)
        fit_or_load(r2)

        h = HybridMultiRhat(URM_train, [r1, r2])
        h.fit(vec)
        
        hybrids.append(h)

    else:
        print("wrong insertion, skipped")


for r in recs:
    
    if c == 'hms':
        r = fit_or_load(r, matrix='sim-matrix')
    else:
        r = fit_or_load(r)

recs = recs + hybrids
 

def tune_and_log(h, filename):
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())
            print(timestamp)
            h.tuning(URM_valid)

if c == 'tunehs':
    for r in recs:
        fit_or_load(r, matrix='sim-matrix')
    h = HybridSimilarity(URM_train, recs[0], recs[1])
    h.tuning(URM_valid)

if c == 'evalhs':
    for r in recs:
        r.fit()
    h = HybridSimilarity(URM_train, recs[0], recs[1])
    h.fit()
    evaluator = Evaluator(h, URM_valid)
    evaluator.results()


elif c == 'hrhat':
    h = HybridRhat(URM_train, recs[0], recs[1])
    h.tuning(URM_valid)

elif c == 'hms':
    h = HybridMultiSim(URM_train, recs)
    filename = os.path.join(args.folder, '{}-TUNING.txt'.format(h.NAME))
    tune_and_log(h, filename)

elif c == 'hmr':
    h = HybridMultiRhat(URM_train, recs)
    filename = os.path.join(args.folder, '{}-TUNING.txt'.format(h.NAME))

    with open(filename, 'w') as f:
        with redirect_stdout(f):
            timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())
            print(timestamp)
            h.tuning(URM_valid)
    
elif c == 'tune':
    for r in recs:

        filename = os.path.join(args.folder, '{}-TUNING.txt'.format(r.NAME))

        with open(filename, 'w') as f:
            with redirect_stdout(f):
                timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())
                print(timestamp)
                r.tuning(URM_valid)

elif c == 'eval':
    for r in recs:
        evaluator = Evaluator(r, URM_valid)
        evaluator.results()
    
elif c == 'subhmr':
    vec_input = input('Insert the value vector:')
    list = vec_input.split()
    vec = [float(ele) for ele in list] 

    assert len(vec) == len(recs), "number of recommenders different from number of values"
    for r in recs:
        try:
            try:
                filename = 'raw_data/' + r.NAME + '-r-hat-test.npy'
                r.load_r_hat(filename)
            except:
                filename = 'raw_data/' + r.NAME + '-r-hat-test.npz'
                r.load_r_hat(filename)
            
            msg = '|{}|  rhat loaded '.format(r.NAME)
            success(msg)
        except:
            msg = '|{}|  no rhat file found, proceeding with fit '.format(r.NAME)
            warning(msg)
            r.fit()
            r.save_r_hat(test=True)
            
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

else:
    print('wrong selection')

