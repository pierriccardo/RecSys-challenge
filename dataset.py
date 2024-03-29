import configparser
import pandas as pd
import scipy.sparse as sps
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.feature_selection import VarianceThreshold
from similarity.similarity import similarity
import similaripy as sim
from matplotlib import pyplot as plt
from tqdm import tqdm

config = configparser.ConfigParser()
config.read('config.ini')
seed = int(config['DEFAULT']['SEED'])

class Dataset:

    def __init__(self, split=0.8):

        assert (
            split > 0 and split < 1
        ), 'Dataset: split value must be between 0 and 1, you set: {}'.format(split)
        
        # pandas dataframe objects containing the dataset
        self.URM_dataframe = pd.read_csv(config['paths']['URM'])
        self.ICM_dataframe = pd.read_csv(config['paths']['ICM'])
        self.test_dataframe = pd.read_csv(config['paths']['test'])

        # constants
        self.NUM_USERS = max(self.URM_dataframe["row"]) + 1
        self.NUM_ITEMS = max(self.URM_dataframe["col"]) + 1
        self.NUM_INTERACTIONS = len(self.URM_dataframe)
        self.split = split

        # matrices
        self.URM = self._create_URM()
        self.ICM = self._create_ICM()
        self.test = self._create_test()
        self.URM_train, self.URM_valid = self._split_train_validation()

        self.urm_train_df.columns = ['user_id', 'item_id', 'data']
        self.urm_valid_df.columns = ['user_id', 'item_id', 'data']

        selector = VarianceThreshold(0.00002) 
        self.ICM_selected = selector.fit_transform(self.ICM.copy())

    def get_URM(self):
        return self.URM
        
    def get_URM_train(self):
        return self.URM_train

    def get_URM_valid(self):
        return self.URM_valid
    
    def get_test(self):
        return self.test

    def get_ICM(self):
        icm_norm = sim.normalization.bm25(self.ICM.copy())
        return icm_norm
    
    def get_URMICM(self):        
        self.URMICM = sps.vstack((self.URM, self.ICM_selected.T))
        return self.URMICM.tocsr()
    
    def get_URMICM_train(self):
        self.URMICM_train = sps.vstack((self.URM_train, self.ICM_selected.T))
        icm_norm = sim.normalization.bm25(self.URMICM_train.tocsr())
        return icm_norm
        #return self.URMICM_train.tocsr()
    
    def get_ICM_selected(self):
        return self.ICM_selected

    def _split_train_validation(self):

        # make results reproducible
        seed = int(config['DEFAULT']['SEED'])
        np.random.seed(seed)

        train_mask = np.random.choice([True,False], 
                                      self.NUM_INTERACTIONS, 
                                      p=[self.split,1-self.split]) 

        self.urm_train_df = self.URM_dataframe.iloc[train_mask]
        self.urm_train_df.to_csv('train.csv', index = True)   
          
        train_d = self.URM_dataframe.data[train_mask]  
        train_r = self.URM_dataframe.row[train_mask]
        train_c = self.URM_dataframe.col[train_mask]
        URM_train = sps.csr_matrix((train_d,(train_r, train_c)), shape=(7947, 25975))

        valid_mask = np.logical_not(train_mask)

        self.urm_valid_df = self.URM_dataframe.iloc[valid_mask]  
        self.urm_valid_df.to_csv('valid.csv', index = True)     

        valid_d = self.URM_dataframe.data[valid_mask]
        valid_r = self.URM_dataframe.row[valid_mask]
        valid_c = self.URM_dataframe.col[valid_mask]

        URM_valid = sps.csr_matrix((valid_d,(valid_r, valid_c)), shape=(7947, 25975))
        
        #assert URM_train.shape == URM_valid.shape, "shapes aren't equal"

        # TODO: FIX shapes not equal

        return URM_train, URM_valid
        

    def _create_URM(self):
        URM_user_list = list(self.URM_dataframe.row)  # users ids
        URM_item_list = list(self.URM_dataframe.col)  # items ids
        URM_data_list = list(self.URM_dataframe.data) # data (1 or 0)

        URM_all = sps.coo_matrix((URM_data_list, (URM_user_list, URM_item_list)))
        URM_all = URM_all.tocsr()

        return URM_all 

    def _create_ICM(self):
        ICM_item_list = list(self.ICM_dataframe.row)  # item ids
        ICM_feat_list = list(self.ICM_dataframe.col)  # feature ids
        ICM_data_list = list(self.ICM_dataframe.data) # value

        ICM_all = sps.coo_matrix((ICM_data_list, (ICM_item_list, ICM_feat_list)))
        ICM_all = ICM_all.tocsr()

        return ICM_all

    def _create_test(self):
        users_to_recommend = list(self.test_dataframe.user_id)

        return users_to_recommend

    def k_fold_icm(self, splits=5, shuff=True, seed=5):
        ds = np.arange(0, self.NUM_INTERACTIONS, 1)
        datasets = []
        
        kf = KFold(n_splits=splits, shuffle=shuff, random_state=seed)
        for train_index, test_index in kf.split(ds):

            train_mask = np.zeros(self.NUM_INTERACTIONS, dtype=bool)
            for i in train_index:
                train_mask[i] = True

            d = self.URM_dataframe.data[train_mask]
            r = self.URM_dataframe.row[train_mask]
            c = self.URM_dataframe.col[train_mask]
            URM_train = sps.csr_matrix((d,(r, c)), shape=(7947, 25975))

            icm = self.ICM.copy()
            selector = VarianceThreshold(0.00002) 
            NewICM = selector.fit_transform(icm)

            URMICM_train = sps.vstack((URM_train,NewICM.T))

            valid_mask = np.logical_not(train_mask)

            d = self.URM_dataframe.data[valid_mask]
            r = self.URM_dataframe.row[valid_mask]
            c = self.URM_dataframe.col[valid_mask]
            URM_valid = sps.csr_matrix((d,(r, c)), shape=(7947, 25975))

            tmp_ds = (URMICM_train.tocsr(), URM_valid)
            datasets.append(tmp_ds)

        return datasets

    def k_fold(self, splits=5, shuff=True, seed=5):
        ds = np.arange(0, self.NUM_INTERACTIONS, 1)
        datasets = []
        
        kf = KFold(n_splits=splits, shuffle=shuff, random_state=seed)
        for train_index, test_index in kf.split(ds):

            train_mask = np.zeros(self.NUM_INTERACTIONS, dtype=bool)
            for i in train_index:
                train_mask[i] = True

            d = self.URM_dataframe.data[train_mask]
            r = self.URM_dataframe.row[train_mask]
            c = self.URM_dataframe.col[train_mask]
            URM_train = sps.csr_matrix((d,(r, c)), shape=(7947, 25975))

            valid_mask = np.logical_not(train_mask)

            d = self.URM_dataframe.data[valid_mask]
            r = self.URM_dataframe.row[valid_mask]
            c = self.URM_dataframe.col[valid_mask]
            URM_valid = sps.csr_matrix((d,(r, c)), shape=(7947, 25975))

            tmp_ds = (URM_train, URM_valid)
            datasets.append(tmp_ds)

        return datasets
   
    def urm_statistics(self):

        n_user, n_item = self.URM.shape
        n_interactions = self.URM.nnz
        
        user_profile_length = np.ediff1d(self.URM.indptr)
      
        min_interaction = user_profile_length.min()
        max_interaction = user_profile_length.max()

        print('n_user: {}'.format(n_user))
        print('n_item: {}'.format(n_item))
        print('n_interaction: {}'.format(n_interactions))
        print('min_interaction: {}'.format(min_interaction))
        print('max_interaction: {}'.format(max_interaction))

        data = []
        for x in tqdm(range(min_interaction, max_interaction)):
            user_with_x_interactions = 0
            for user in range(0,7947):
                s = self.URM.indptr[user]
                e = self.URM.indptr[user + 1]
        
                seen = self.URM.indices[s:e]
                if len(seen) == x:
                    user_with_x_interactions += 1
            
            data.append([x, user_with_x_interactions])
        df = pd.DataFrame(data, columns = ['interactions', 'num_users']) 
        df = df[:30]

        df.plot(kind='bar',x='interactions',y='num_users',color='blue')
        plt.show()

    def icm_statistics(self):

        item_profile_length = np.ediff1d(self.ICM.indptr)
      
        min_interaction = item_profile_length.min()
        max_interaction = item_profile_length.max()

        data = []
        for x in tqdm(range(min_interaction, max_interaction)):
            item_with_x_interactions = 0
            for item in range(0,25975):
                s = self.ICM.indptr[item]
                e = self.ICM.indptr[item + 1]
        
                n_features = self.ICM.indices[s:e]
                if len(n_features) == x:
                    item_with_x_interactions += 1
            
            data.append([x, item_with_x_interactions])
        df = pd.DataFrame(data, columns = ['n_items', 'n_features']) 
        #df = df[:30]
        print(df)

        df.plot(kind='line',x='n_items',y='n_features',color='blue')
        plt.show()

        print(min_interaction)
        print(max_interaction)


        self.ICM = sps.csr_matrix(self.ICM)
        features_per_item = np.ediff1d(self.ICM.indptr)

        self.ICM = sps.csc_matrix(self.ICM)
        items_per_feature = np.ediff1d(self.ICM.indptr)

        self.ICM = sps.csr_matrix(self.ICM)


        plt.plot(features_per_item, 'ro')
        plt.ylabel('Num features ')
        plt.xlabel('Sorted items')
        plt.show()


def main():

    d = Dataset()

    #d.icm_stats()

main()