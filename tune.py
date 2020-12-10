import numpy as np
import pandas as pd
import os
import scipy.sparse as sps
from matplotlib import pyplot
import configparser
import sys
from time import strftime, gmtime
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

URM = d.URM
URM_train = d.get_URM_train()
URM_valid = d.get_URM_valid()
ICM = d.get_ICM()
test_set = d.get_test()

urm_train_df = d.urm_train_df

def hsim():
    print(Fore.BLACK + Back.GREEN + '   choose two algorithm to hybrid:   ' + Style.RESET_ALL)
    print(Fore.GREEN)
    print('   press 1 --> ItemKNNCF')
    print('   press 2 --> ItemKNNCB')
    print('   press 3 --> RP3beta')
    print('   press 4 --> P3alpha')
    print('   press 5 --> UserKNNCF')
    print('   press 6 --> UserKNNCB')
    print(Style.RESET_ALL)

    print(Fore.BLACK + Back.GREEN + '   Enter the two separated by space: ' + Style.RESET_ALL)

    hrecs = []

    choice = input(Fore.BLUE + Back.MAGENTA + ' -> ' + Style.RESET_ALL)
    list = choice.split()
    for e in list:
        if e == '1':
            hrecs.append(r1)
        elif e == '2':
            hrecs.append(r2)
        elif e == '3':
            hrecs.append(r3)
        elif e == '4':
            hrecs.append(r4)
        elif e == '5':
            hrecs.append(r5)
        elif e == '6':
            hrecs.append(r6)
    hrecs[0].fit()
    hrecs[1].fit()
    h = HybridSimilarity(URM_train, hrecs[0], hrecs[1])
    recs.append(h)

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
from recommenders.UserKNNCF         import UserKNNCF
from recommenders.UserKNNCB         import UserKNNCB

from evaluator                      import Evaluator

print('    ________                                      ')
print('   |        \                                     ')
print('    \▓▓▓▓▓▓▓▓__    __ _______   ______   ______   ')
print('      | ▓▓  |  \  |  \       \ /      \ /      \  ')
print('      | ▓▓  | ▓▓  | ▓▓ ▓▓▓▓▓▓▓\  ▓▓▓▓▓▓\  ▓▓▓▓▓▓\ ')
print('      | ▓▓  | ▓▓  | ▓▓ ▓▓  | ▓▓ ▓▓    ▓▓ ▓▓   \▓▓ ')
print('      | ▓▓  | ▓▓__/ ▓▓ ▓▓  | ▓▓ ▓▓▓▓▓▓▓▓ ▓▓       ')
print('      | ▓▓   \▓▓    ▓▓ ▓▓  | ▓▓\▓▓     \ ▓▓       ')
print('       \▓▓    \▓▓▓▓▓▓ \▓▓   \▓▓ \▓▓▓▓▓▓▓\▓▓       ')                         
print('    ______ ______ ______ ______ ______ ______ ______  ')
print('   |      \      \      \      \      \      \      \ ')
print('    \▓▓▓▓▓▓\▓▓▓▓▓▓\▓▓▓▓▓▓\▓▓▓▓▓▓\▓▓▓▓▓▓\▓▓▓▓▓▓\▓▓▓▓▓▓ ')
                                           
                                           

from colorama import Fore, Back, Style


# folder input
print(Fore.BLACK + Back.GREEN + '   Choose the folder (press enter for /tuning):          ' + Style.RESET_ALL)

choice = input(Fore.BLUE + Back.MAGENTA + ' -> ' + Style.RESET_ALL)

folder = '/tuning' if (choice == '' or choice is None) else choice + '/'
if not os.path.exists(folder):
    os.mkdir(folder)

# algorithms choice
print(Fore.BLACK + Back.GREEN + '   Which algorithm you want to tune?          ' + Style.RESET_ALL)
print(Fore.GREEN)
print('   press 1 --> ItemKNNCF')
print('   press 2 --> ItemKNNCB')
print('   press 3 --> RP3beta')
print('   press 4 --> P3alpha')
print('   press 5 --> UserKNNCF')
print('   press 6 --> UserKNNCB')
print('')
print('   press hsim --> Hybrid Similarity')
print('')
print('')

print(Fore.BLACK + Back.GREEN + '   Enter a list with elems separated by space:' + Style.RESET_ALL)

choice = input(Fore.BLUE + Back.MAGENTA + ' -> ' + Style.RESET_ALL)
list = choice.split()


r1 = ItemKNNCF(URM_train)
r2 = ItemKNNCB(URM_train, ICM)
r3 = RP3beta(URM_train)
r4 = P3alpha(URM_train)
r5 = UserKNNCF(URM_train)
r6 = UserKNNCB(URM_train, ICM)

recs = []

for e in list:

    if e == '1':
        recs.append(r1)
    elif e == '2':
        recs.append(r2)
    elif e == '3':
        recs.append(r3)
    elif e == '4':
        recs.append(r4)
    elif e == '5':
        recs.append(r5)
    elif e == '6':
        recs.append(r6)
    elif e == 'hsim':
        hsim()

    else:
        print("wrong insertion, skipped")

for r in recs:

    timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())
    #folder = 'tuning/'
    #folder = 'tuning2/'
    filename = '{}{}-{}-TUNING.txt'.format(folder, r.NAME, timestamp)

    with open(filename, 'w') as f:
        with redirect_stdout(f):
            r.tuning(URM_valid)


    
    

    
