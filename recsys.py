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
    URM_train = d.get_URM_train()
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

from recommenders.hybridCFCB.UserKNNCFCB import UserKNNCFCB

from evaluator                      import Evaluator

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
print('   press h1 --> UserKNNCFCB')
print('')
choice = input(Fore.BLUE + Back.WHITE + ' -> ' + Style.RESET_ALL)
list = choice.split()

recs = []

for e in list:

    if e == '1':
        itemKNNCF = ItemKNNCF(URM_train)
        recs.append(itemKNNCF)

    elif e == '2':
        itemKNNCB = ItemKNNCB(URM_train, ICM)
        recs.append(itemKNNCB)

    elif e == '3':
        rp3beta = RP3beta(URM_train)
        recs.append(rp3beta)

    elif e == '4':
        p3alpha = P3alpha(URM_train)
        recs.append(p3alpha)

    elif e == '5':
        userKNNCF = UserKNNCF(URM_train)
        recs.append(userKNNCF)

    elif e == '6':
        userKNNCB = UserKNNCB(URM_train, ICM)
        recs.append(userKNNCB)
    
    elif e == '7':
        slim_mse = SLIM_MSE(URM_train)
        recs.append(slim_mse)
    
    elif e == '8':
        slim_bpr = SLIM_BPR(URM_train)
        recs.append(slim_bpr)
    
    elif e == '9':
        pureSVD = PureSVD(URM_train)
        recs.append(pureSVD)

    elif e == '10':
        ials = IALS(URM_train)
        recs.append(ials)

    elif e == 'h1':
        r = UserKNNCFCB(URM_train, ICM)
        recs.append(r)
    else:
        print("wrong insertion, skipped")

msg = '   Choose an action:                                                     '
info(msg)

print('   hsim          --> tune a hybrid with 2 algorithms with sim matrix          ')
print('   hrhat         --> tune a hybrid with 2 algorithms with r hat               ')
print('   hms           --> tune a hybrid with n algorithms by sim matrix, random val')
print('')       
print('   hmr           --> tune a hybrid with n algorithms by r hat, random val     ')
print('   subhmr        --> submit a hybrid multi rhat algorithm                     ')
print('   evalhmr       --> evaluate a hybrid multi rhat recommender                 ')
print('')       
print('   tune          --> tune the choosen algorithms                              ')
print('   saverhat      --> save r hats of the selected algorithms                   ')
print('   savesim       --> save r hats of the selected algorithms                   ')
print('   eval          --> evaluate the selected algorithms                         ')
print('   crossvalid    --> evaluate with cross validation the selected algorithms   ')
print('')
c = input(Fore.BLUE + Back.WHITE + ' -> ' + Style.RESET_ALL)


def fit_or_load(rec, matrix='r-hat', type='valid'):
    """
    try to load a given matrix for the recommender, if
    the matrix is not present in the folder raw data then
    will fit the recommender and save the matrix

    Args:
        rec ([type]): [description]
        matrix (str, optional): define the matrix type: 'r-hat' or 'sim-matrix'.
        type (str, optional): type of matrix to save 'valid' or 'test'.

    Returns:
        [type]: [description]
    """
    filename = 'raw_data/{}-{}-{}'.format(r.NAME, matrix, type)
    if args.loadrhat:
        try:
            try: r.load_r_hat(filename + '.npy')
            except: r.load_r_hat(filename + '.npz')
            msg = '|{}|  rhat loaded '.format(r.NAME)
            success(msg)
        except:
            msg = '|{}|  no rhat file found, proceeding with fit '.format(r.NAME)
            warning(msg)
            r.fit()
            if matrix=='r-hat':
                r.save_r_hat(test=args.test)
            if matrix=='sim-matrix':
                r.save_sim_matrix(test=args.test)
    else:
        r.fit()
    
    return rec

 

def tune_and_log(h, filename):
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())
            print(timestamp)
            h.tuning(URM_valid)

if c == 'hsim':
    h = HybridSimilarity(URM_train, recs[0], recs[1])
    h.tuning(URM_valid)

elif c == 'hrhat':
    h = HybridRhat(URM_train, recs[0], recs[1])
    h.tuning(URM_valid)

elif c == 'hms':
    for r in recs:
        if args.loadrhat:
            try:
                try:
                    filename = 'raw_data/' + r.NAME + '-sim-matrix-valid.npy'
                    r.load_sim_matrix(filename)
                except:
                    filename = 'raw_data/' + r.NAME + '-sim-matrix-valid.npz'
                    r.load_sim_matrix(filename)
                
                msg = '|{}| sim matrix loaded '.format(r.NAME)
                success(msg)
            except:
                msg = '|{}|  no sim matrix file found, proceeding with fit '.format(r.NAME)
                warning(msg)
                r.fit()
                r.save_sim_matrix(test=args.test)
        else:
            r.fit()
    recs = fit_or_load(recs)
    h = HybridMultiSim(URM_train, recs)
    filename = os.path.join(args.folder, '{}-TUNING.txt'.format(h.NAME))
    tune_and_log(h, filename)

elif c == 'hmr':
    for r in recs:
        if args.loadrhat:
            try:
                try:
                    filename = 'raw_data/' + r.NAME + '-r-hat-valid.npy'
                    r.load_r_hat(filename)
                except:
                    filename = 'raw_data/' + r.NAME + '-r-hat-valid.npz'
                    r.load_r_hat(filename)
                
                msg = '|{}|  rhat loaded '.format(r.NAME)
                success(msg)
            except:
                msg = '|{}|  no rhat file found, proceeding with fit '.format(r.NAME)
                warning(msg)
                r.fit()
                r.save_r_hat(test=args.test)
        else:
            r.fit()

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

elif c == 'saverhat':
    for r in recs:
        msg = '|{}|  fitting... '.format(r.NAME)
        success(msg)
        r.fit()
        r.save_r_hat(test=args.test)
        evaluator = Evaluator(r, URM_valid)
        evaluator.results()

elif c == 'savesim':
    for r in recs:
        msg = '|{}|  fitting... '.format(r.NAME)
        success(msg)
        r.fit()
        r.save_sim_matrix(test=args.test)
        evaluator = Evaluator(r, URM_valid)
        evaluator.results()

elif c == 'save':
    for r in recs:
        msg = '|{}|  fitting... '.format(r.NAME)
        success(msg)
        r.fit()
        r.save_r_hat(test=args.test)
        msg = '|{}|  saved r hat '.format(r.NAME)
        success(msg)
        try:
            r.save_sim_matrix(test=arg.test)
            msg = '|{}|  saved sim matrix '.format(r.NAME)
            success(msg)
        except:
            msg = '|{}|  failed or no sim matrix to save '.format(r.NAME)
            warning(msg)

        evaluator = Evaluator(r, URM_valid)
        evaluator.results()

elif c == 'eval':
    for r in recs:
        if args.loadrhat:
            try:
                try:
                    filename = 'raw_data/' + r.NAME + '-r-hat-valid.npy'
                    r.load_r_hat(filename)
                except:
                    filename = 'raw_data/' + r.NAME + '-r-hat-valid.npz'
                    r.load_r_hat(filename)
                
                msg = '|{}|  rhat loaded '.format(r.NAME)
                success(msg)
            except:
                msg = '|{}|  no rhat file found, proceeding with fit '.format(r.NAME)
                warning(msg)
                r.fit()
                r.save_r_hat(test=args.test)
        else:
            r.fit()
            
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

    assert len(vec) == len(recs), "number of recommenders different from number of values"
    for r in recs:
        try:
            try:
                filename = 'raw_data/' + r.NAME + '-r-hat-valid.npy'
                r.load_r_hat(filename)
            except:
                filename = 'raw_data/' + r.NAME + '-r-hat-valid.npz'
                r.load_r_hat(filename)
            
            msg = '|{}|  rhat loaded '.format(r.NAME)
            success(msg)
        except:
            msg = '|{}|  no rhat file found, proceeding with fit '.format(r.NAME)
            warning(msg)
            r.fit()
            if not args.test:
                r.save_r_hat(test=False)
            else:
                msg = 'Error: you select the complete URM option saving r-hat is skipped'
                error(msg)

            
    h = HybridMultiRhat(URM_train, recs)
    h.fit(vec)
    evaluator = Evaluator(h, URM_valid)
    evaluator.results()

elif c == 'crossvalid':

    datasets = d.k_fold()

    for r in recs:
        cross_validate(r, datasets)


else:
    print('wrong selection')

