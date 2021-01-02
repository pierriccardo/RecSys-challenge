import csv 
from datetime import datetime
import scipy.sparse as sps
import numpy as np
from time import strftime, gmtime
from tqdm import tqdm
import os
from recommenders.ItemKNNCF import ItemKNNCF


def create_submission_csv(recommender, users_list, save_path='./'):

    timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())

    filename = os.path.join(save_path, recommender.NAME + 'results-' + timestamp + '.csv')

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

def precision(recommended_items, relevant_items):
        
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score

def recall(recommended_items, relevant_items):

    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    
    return recall_score

def MAP(recommended_items, relevant_items):

    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score

from recommenders.P3alpha import P3alpha

def cross_validate(rec, datasets, cutoff=10):
    
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

        rec.urm = train_ds
        rec.fit()

        pbar = tqdm(range(valid_ds.shape[0]))
        for user_id in pbar:
            
            pbar.set_description('|{}| evaluating'.format(rec.NAME))
            relevant_items = valid_ds.indices[valid_ds.indptr[user_id]:valid_ds.indptr[user_id+1]]
            
            if len(relevant_items)>0:
                
                recommended_items = rec.recommend(user_id, cutoff)
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

    print('[avg precision:  {:.4f}]'.format(sum(list_precision) / len(list_precision)))
    print('[avg recall:     {:.4f}]'.format(sum(list_recall) / len(list_recall)))
    print('[avg MAP:        {:.4f}]'.format(sum(list_MAP) / len(list_MAP)))


