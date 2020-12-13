import numpy as np
import pandas as pd
import os
import scipy.sparse as sps
from matplotlib import pyplot
import configparser

from tqdm import tqdm
import argparse

SAVE_RHAT = False


parser = argparse.ArgumentParser(description='Recsys main.')

parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('-f', '--folder', type=str, default='tuning')
parser.add_argument('--urm', action='store_true')



args = parser.parse_args()
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
from sklearn.preprocessing import normalize
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
from recommenders.FunkSVD           import FunkSVD
from recommenders.PureSVD           import PureSVD
from recommenders.SLIM_BPR          import SLIM_BPR
from recommenders.MF_BPR            import MF_BPR
from recommenders.HybridRhat        import HybridRhat
from recommenders.UserKNNCF         import UserKNNCF
from recommenders.UserKNNCB         import UserKNNCB
from recommenders.IALS              import IALS
from recommenders.HybridMultiSim    import HybridMultiSim

from evaluator                      import Evaluator

#------------------------------
# BASIC RECOMMENDERS
#------------------------------

from colorama import Fore, Back, Style
print('                                                            ')
print('██╗  ██╗██╗   ██╗██████╗ ██████╗ ██╗██████╗ ███████╗██████╗ ')
print('██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██║██╔══██╗██╔════╝██╔══██╗')
print('███████║ ╚████╔╝ ██████╔╝██████╔╝██║██║  ██║█████╗  ██████╔╝')
print('██╔══██║  ╚██╔╝  ██╔══██╗██╔══██╗██║██║  ██║██╔══╝  ██╔══██╗')
print('██║  ██║   ██║   ██████╔╝██║  ██║██║██████╔╝███████╗██║  ██║')
print('╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝')
print('                                                            ')
print(Fore.BLACK + Back.GREEN + '   Choose 2 algorithms to Hybrid                           ' + Style.RESET_ALL)
print(Fore.BLACK + Back.GREEN + '   put the prefix hsim, hrhat, hmr, hms                    ' + Style.RESET_ALL)
print(Fore.GREEN)                                                                           
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
print('')
print('')
print('   press h1 --> Hsim(ItemKNNCF, ItemKNNCB)')
print('   press h2 --> Hsim(UserKNNCF, UserKNNCB)')
print('   press h3 --> Hrhat(itemKNNCB, UserKNNCB)')
print('   press h4 --> Hsim(P3alpha, RP3beta)')
print('   press h5 --> Hsim(SLIM_MSE, SLIM_BPR)')
print(Fore.BLACK + Back.GREEN + '   Enter a list with elems separated by space:' + Style.RESET_ALL)


itemKNNCF   = ItemKNNCF(URM_train)
itemKNNCB   = ItemKNNCB(URM_train, ICM)
rp3beta     = RP3beta(URM_train)
p3alpha     = P3alpha(URM_train)
userKNNCF   = UserKNNCF(URM_train)
userKNNCB   = UserKNNCB(URM_train, ICM)
slim_mse    = SLIM_MSE(URM_train)
slim_bpr    = SLIM_BPR(URM_train)
pureSVD     = PureSVD(URM_train)
ials        = IALS(URM_train)

choice = input(Fore.BLUE + Back.WHITE + ' -> ' + Style.RESET_ALL)
list = choice.split()

recs = []

for e in list[1:]:

    if e == '1':
        itemKNNCF.fit()
        itemKNNCF.save_r_hat()
        #itemKNNCF.load_r_hat('raw_data/ItemKNNCF-r-hat.npz')
        recs.append(itemKNNCF)


    elif e == '2':
        itemKNNCB.fit()
        itemKNNCB.save_r_hat()
        #itemKNNCB.load_r_hat('raw_data/ItemKNNCB-r-hat.npz')
        recs.append(itemKNNCB)

    elif e == '3':
        #rp3beta.load_r_hat('raw_data/RP3beta-r-hat.npz')
        rp3beta.fit()
        rp3beta.save_r_hat()
        recs.append(rp3beta)

    elif e == '4':
        p3alpha.fit()
        p3alpha.save_r_hat()
        #p3alpha.load_r_hat('raw_data/P3alpha-r-hat.npz')
        recs.append(p3alpha)

    elif e == '5':
        userKNNCF.fit()
        userKNNCF.save_r_hat()
        #userKNNCF.load_r_hat('raw_data/UserKNNCF-r-hat.npz')
        recs.append(userKNNCF)

    elif e == '6':
        userKNNCB.fit()
        userKNNCB.save_r_hat()
        #userKNNCB.load_r_hat('raw_data/UserKNNCB-r-hat.npz')
        recs.append(userKNNCB)
    
    elif e == '7':
        if SAVE_RHAT:
            slim_mse.fit()
            slim_mse.save_r_hat()
            slim_mse.save_sim_matrix()
        else:
            slim_mse.load_r_hat('raw_data/SLIM_MSE-r-hat.npz')
            slim_mse.load_sim_matrix('raw_data/SLIM_MSE-sim-matrix.npz')
        recs.append(slim_mse)
    
    elif e == '8':
        if SAVE_RHAT:
            slim_bpr.fit()
            slim_bpr.save_r_hat()
            slim_bpr.save_sim_matrix()
        else:
            slim_bpr.load_r_hat('raw_data/SLIM_BPR-r-hat.npz')
            slim_bpr.load_sim_matrix('raw_data/SLIM_BPR-sim-matrix.npz')
        recs.append(slim_bpr)
    
    elif e == '9':
        #pureSVD.fit()
        #pureSVD.save_r_hat()
        pureSVD.load_r_hat('raw_data/PureSVD-r-hat.npy')
        recs.append(pureSVD)

    elif e == '10':
        #ials.fit()
        #ials.save_r_hat()
        ials.load_r_hat('raw_data/IALS-r-hat.npy')
        recs.append(ials)


    elif e == 'h1':
        itemKNNCB.fit()
        itemKNNCF.fit()
        h = HybridSimilarity(URM_train, itemKNNCF, itemKNNCB)
        h.fit(topK=70, alpha=0.8)
        recs.append(h)

    elif e == 'h2':
        userKNNCB.fit()
        userKNNCF.fit()
        h = HybridSimilarity(URM_train, userKNNCF, userKNNCB)
        h.fit(topK=350, alpha=0.96)
        recs.append(h)

    elif e == 'h3':
        itemKNNCB.fit()
        userKNNCB.fit()
        h = HybridRhat(URM_train, itemKNNCB, userKNNCB)
        h.fit(alpha=0.55, norm='l1')
        recs.append(h)
    
    elif e == 'h4':
        p3alpha.fit()
        rp3beta.fit()
        h = HybridSimilarity(URM_train, rp3beta, p3alpha)
        h.fit(topk=400, alpha=0.160 )
        recs.append(h)
    
    elif e == 'h5':
        slim_mse.load_sim_matrix('raw_data/SLIM_MSE-sim-matrix.npz')
        slim_bpr.load_sim_matrix('raw_data/SLIM_BPR-sim-matrix.npz')
        h = HybridSimilarity(URM_train, slim_mse, slim_bpr)
        h.fit()
        recs.append(h)

    else:
        print("wrong insertion, skipped")


totune = []

if list[0] == 'hsim':
    h = HybridSimilarity(URM_train, recs[0], recs[1])
    totune.append(h)

elif list[0] == 'hrhat':
    h = HybridRhat(URM_train, recs[0], recs[1])
    totune.append(h)

elif list[0] == 'hmr':
    h = HybridMultiRhat(URM_train, recs)
    totune.append(h)
elif list[0] == 'hms':
    h = HybridMultiSim(URM_train, recs)
    totune.append(h)
else:
    print('wrong hybrid type')


import os
from contextlib import redirect_stdout


if not os.path.exists(args.folder):
    os.mkdir(args.folder)
    
from time import strftime, gmtime

for r in totune:

    filename = os.path.join(args.folder, '{}-TUNING.txt'.format(r.NAME))

    with open(filename, 'w') as f:
        with redirect_stdout(f):
            timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())
            print(timestamp)
            r.tuning(URM_valid)

