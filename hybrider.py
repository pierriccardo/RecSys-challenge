import numpy as np
import pandas as pd
import os
import scipy.sparse as sps
from matplotlib import pyplot
import configparser

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Recsys main.')

parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('-f', '--folder', type=str, default='tuning')


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
print(Fore.BLACK + Back.GREEN + '   Choose 2 algorithms to Hybrid            ' + Style.RESET_ALL)
print(Fore.BLACK + Back.GREEN + '   put the prefix hsim, hscore, hrhat       ' + Style.RESET_ALL)
print(Fore.GREEN)
print('   press 1 --> ItemKNNCF')
print('   press 2 --> ItemKNNCB')
print('   press 3 --> RP3beta')
print('   press 4 --> P3alpha')
print('   press 5 --> UserKNNCF')
print('   press 6 --> UserKNNCB')

print('   press 7 --> Hsim(ItemKNNCF, ItemKNNCB)')
print('   press 8 --> Hsim(UserKNNCF, UserKNNCB)')
print('   press 9 --> Hrhat(itemKNNCB, UserKNNCB)')
print('   press 10 --> Hsim(itemKNNCB, itemKNNCB)')
print(Fore.BLACK + Back.GREEN + '   Enter a list with elems separated by space:' + Style.RESET_ALL)


itemKNNCF   = ItemKNNCF(URM_train)
itemKNNCB   = ItemKNNCB(URM_train, ICM)
rp3beta     = RP3beta(URM_train)
p3alpha     = P3alpha(URM_train)
userKNNCF   = UserKNNCF(URM_train)
userKNNCB   = UserKNNCB(URM_train, ICM)

choice = input(Fore.BLUE + Back.WHITE + ' -> ' + Style.RESET_ALL)
list = choice.split()

recs = []

for e in list[1:]:

    if e == '1':
        itemKNNCF.fit()
        recs.append(itemKNNCF)

    elif e == '2':
        itemKNNCB.fit()
        recs.append(itemKNNCB)

    elif e == '3':
        rp3beta.fit()
        recs.append(rp3beta)

    elif e == '4':
        p3alpha.fit()
        recs.append(p3alpha)

    elif e == '5':
        userKNNCF.fit()
        recs.append(userKNNCF)

    elif e == '6':
        userKNNCB.fit()
        recs.append(userKNNCB)

    elif e == '7':
        itemKNNCB.fit()
        itemKNNCF.fit()
        h = HybridSimilarity(URM_train, itemKNNCF, itemKNNCB)
        h.fit(topK=70, alpha=0.8)
        recs.append(h)

    elif e == '8':
        userKNNCB.fit()
        userKNNCF.fit()
        h = HybridSimilarity(URM_train, userKNNCF, userKNNCB)
        h.fit(topK=350, alpha=0.96)
        recs.append(h)

    elif e == '9':
        itemKNNCB.fit()
        userKNNCB.fit()
        h = HybridRhat(URM_train, itemKNNCB, userKNNCB)
        h.fit(alpha=0.55, norm='l1')
        recs.append(h)
    
    elif e == '10':
        itemKNNCB.fit()
        h = HybridSimilarity(URM_train, itemKNNCB, itemKNNCB)
        h.fit(topK=110 , alpha=0, norm='l1')
        recs.append(h)

    else:
        print("wrong insertion, skipped")


totune = []

if list[0] == 'hsim':
    h = HybridSimilarity(URM_train, recs[0], recs[1])
    totune.append(h)

elif list[0] == 'hscore':
    h = HybridScores(URM_train, recs[0], recs[1])
    totune.append(h)

elif list[0] == 'hrhat':
    h = HybridRhat(URM_train, recs[0], recs[1])
    totune.append(h)
else:
    print('wrong hybrid type')


import os
from contextlib import redirect_stdout


if not os.path.exists(args.folder):
    os.mkdir(args.folder)
    


for r in totune:

    filename = os.path.join(args.folder, '{}-TUNING.txt'.format(r.NAME))

    with open(filename, 'w') as f:
        with redirect_stdout(f):
            r.tuning(URM_valid)














single_recs = [itemKNNCB, itemKNNCF, userKNNCB, userKNNCF, rp3beta, p3alpha]

#------------------------------
# HYBRIDS
#------------------------------

hsim_itemKNNCBCF = HybridSimilarity(URM_train, itemKNNCB, itemKNNCF)
hsim_userKNNCBCF = HybridSimilarity(URM_train, userKNNCB, userKNNCF)


recs = []
for r in single_recs:
    recs.append(r)
#------------------------------
#       EVALUATION
#------------------------------
from evaluator import Evaluator
for r in recs:

    evaluator = Evaluator(r, URM_valid)
    evaluator.results()
