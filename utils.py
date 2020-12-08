import csv 
from datetime import datetime
import scipy.sparse as sps
import numpy as np
from time import strftime, gmtime
from tqdm import tqdm
import os


def create_submission_csv(recommender, users_list, save_path='./'):

    timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())

    try:
        rec_name = recommender.name + '-'
    
    except AttributeError:

        rec_name = ''
        print("Recommender has no name")

    filename = os.path.join(save_path, rec_name + 'results-' + timestamp + '.csv')

    with open(filename, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['user_id', 'item_list'])

        pbar = tqdm(users_list)
        for user_id in pbar:
            
            pbar.set_description("|{}| creating submission csv".format(recommender.NAME))

            rec_list = recommender.recommend(user_id, 10).tolist()
            item_list = ''
            for e in rec_list:
                item_list += str(e) + ' '
            filewriter.writerow([user_id, item_list])

def check_matrix(X, format='csc', dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    :param X:
    :param format:
    :param dtype:
    :return:
    """


    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)

    elif format == 'npy':
        if sps.issparse(X):
            return X.toarray().astype(dtype)
        else:
            return np.array(X)

    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)

