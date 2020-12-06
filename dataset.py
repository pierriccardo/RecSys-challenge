import configparser
import pandas as pd
import scipy.sparse as sps
import numpy as np
from sklearn.model_selection import train_test_split

config = configparser.ConfigParser()
config.read('config.ini')

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
        self.NUM_USERS = self.URM_dataframe["row"].unique()
        self.NUM_ITEMS = self.URM_dataframe["col"].unique()
        self.NUM_INTERACTIONS = len(self.URM_dataframe)
        self.split = split

        # matrices
        self.URM = self._create_URM()
        self.ICM = self._create_ICM()
        self.test = self._create_test()
        self.URM_train, self.URM_valid = self._split_train_validation()

        
        self.urm_train_df.columns = ['user_id', 'item_id', 'data']
        self.urm_valid_df.columns = ['user_id', 'item_id', 'data']

    def get_URM_train(self):
        return self.URM_train

    def get_URM_valid(self):
        return self.URM_valid
    
    def get_test(self):
        return self.test

    def get_ICM(self):
        return self.ICM

    def _split_train_validation(self):

        # make results reproducible
        seed = int(config['DEFAULT']['SEED'])
        np.random.seed(seed)

        train_mask = np.random.choice([True,False], 
                                      self.NUM_INTERACTIONS, 
                                      p=[self.split,1-self.split]) 

        self.urm_train_df = self.URM_dataframe.iloc[train_mask]       

        URM_train = sps.csr_matrix((self.URM_dataframe.data[train_mask],
                            (self.URM_dataframe.row[train_mask], self.URM_dataframe.col[train_mask])))

        valid_mask = np.logical_not(train_mask)

        self.urm_valid_df = self.URM_dataframe.iloc[valid_mask]       

        URM_valid = sps.csr_matrix((self.URM_dataframe.data[valid_mask],
                            (self.URM_dataframe.row[valid_mask], self.URM_dataframe.col[valid_mask])))
        
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

    def statistics(self):

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








        