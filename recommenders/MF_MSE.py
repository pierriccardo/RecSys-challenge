import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import time
from recommenders.recommender import Recommender
import numpy as np
import similaripy as sim
from scipy import sparse 
from tqdm import tqdm




loss_function = nn.MSELoss(reduction='sum')


class MF(torch.nn.Module, Recommender):

    def __init__(self, urm, n_factors=10):

        Recommender.__init__(self, urm = urm)
        nn.Module.__init__(self)

        self.n_factors = n_factors
        n_users, n_items = urm.shape

        self.users_emb = nn.Embedding( 
            num_embeddings=n_users,
            embedding_dim=n_factors
        )
        self.items_emb = nn.Embedding(
            num_embeddings=n_items, 
            embedding_dim=n_factors
        )

    def forward(self, u, i):
        """
        Args:
            u: coordinate of the user
            i: coordinate of the item

        Returns:
            p: (float) prediction 
        """
        user_factor = self.users_emb(u)
        item_factor = self.items_emb(i)

        p = torch.mul(user_factor, item_factor)
        p = nn.Linear(
            in_features=self.n_factors, 
            out_features = 1
        )(p)

        return F.relu(p)

    def compute_factors(self):

        self.user_factors = self.users_emb.weight.detach().cpu().numpy()
        self.item_factors = self.items_emb.weight.detach().cpu().numpy()

    def fit(self): pass
    def tuning(self): pass


class BPRDataset(torch.Dataset):

    def __init__(self, urm):
                
        urm = urm.tocoo()

        self.urm_coo = urm

        self.n_interactions = urm.nnz

        self.user_item_coordinates = np.empty((self.n_interactions, 2))

        self.user_item_coordinates[:,0] = urm.row.copy()
        self.user_item_coordinates[:,1] = urm.col.copy()

        self.rating = urm.data.copy().astype(np.float)

        self.user_item_coordinates = torch.Tensor(self.user_item_coordinates).type(torch.LongTensor)
        self.rating = torch.Tensor(self.rating)

    def __getitem__(self, index):
        """
        Format is (row, col, data)
        :param index:
        :return:
        """
        return self.user_item_coordinates[index, :], self.rating[index]

    def __len__(self):

        return self.n_interactions





'''

class MF_MSE(Recommender):

    NAME = 'MF_MSE'

    def __init__(self, urm):

        

        super().__init__(urm = urm)

    def fit(self, epochs=5, batch_size=128, n_factors=100, lr=1e-3, use_cuda=True):

        self.n_factors = n_factors
        self.batch_size = batch_size
        self.lr = lr

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        n_users, n_items = self.urm.shape
        self.model = MF_MSE_Model(n_users, n_items, self.n_factors).to(self.device)

        
        self.loss_function = torch.nn.MSELoss(size_average=False)
        
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr = self.lr)

        dataset_iterator = DatasetIterator(self.urm)

        self.train_data_loader = DataLoader(dataset=dataset_iterator,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            )

        self._train(epochs)

        self.item_factors = self.model.get_item_factors()
        self.user_factors = self.model.get_user_factors()

    
        #self.r_hat = self.user_factors.dot(self.item_factors.T)
        
    
    def _train(self, epochs):
        current_epoch = 0

        while current_epoch < epochs:
            
            ts = time.time()
            self._run_epoch(current_epoch)  
            print("| epoch {}/{} | time: {:.2f} |".format(
                current_epoch+1, epochs, time.time() - ts
                )
            )

            current_epoch += 1
    
    def _run_epoch(self, num_epoch):

        start_time = time.time()

        for num_batch, (input_data, label) in enumerate(self.train_data_loader, 0):
               
            input_data_tensor = Variable(input_data).to(self.device)

            label_tensor = Variable(label).to(self.device)
            
            user_coordinates = input_data_tensor[:,0]
            item_coordinates = input_data_tensor[:,1]

            # forward pass
            prediction = self.model(user_coordinates, item_coordinates)

            # Pass prediction and label removing last empty dimension of prediction
            loss = self.loss_function(prediction.view(-1), label_tensor)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def tuning(self):

        pass

class MF_MSE_Model(torch.nn.Module):

    NAME = 'MF_MSE_Model'

    def __init__(self, n_users, n_items, n_factors):

        super(MF_MSE_Model, self).__init__()

        self.print_pred = 0

        self.n_factors = n_factors
        self.n_users = n_users  
        self.n_items = n_items  

        self.user_factors = torch.nn.Embedding(num_embeddings = n_users, embedding_dim = n_factors)
        self.item_factors = torch.nn.Embedding(num_embeddings = n_items, embedding_dim = n_factors)

        self.layer_1 = torch.nn.Linear(in_features = self.n_factors, out_features = 1)

        self.activation_function = torch.nn.ReLU()

    def forward(self, user_coordinates, item_coordinates):

        current_user_factors = self.user_factors(user_coordinates)
        current_item_factors = self.item_factors(item_coordinates)

        prediction = torch.mul(current_user_factors, current_item_factors)

        prediction = self.layer_1(prediction)
        prediction = self.activation_function(prediction)

        return prediction
    
    def get_user_factors(self):
        return self.user_factors.weight.detach().cpu().numpy()


    def get_item_factors(self):
        return self.item_factors.weight.detach().cpu().numpy()


class DatasetIterator:

    def __init__(self, URM):
                
        URM = URM.tocoo()

        self.n_interactions = URM.nnz

        self.user_item_coordinates = np.empty((self.n_interactions, 2))

        self.user_item_coordinates[:,0] = URM.row.copy()
        self.user_item_coordinates[:,1] = URM.col.copy()

        self.rating = URM.data.copy().astype(np.float)

        self.user_item_coordinates = torch.Tensor(self.user_item_coordinates).type(torch.LongTensor)
        self.rating = torch.Tensor(self.rating)

    def __getitem__(self, index):
        """
        Format is (row, col, data)
        :param index:
        :return:
        """
        return self.user_item_coordinates[index, :], self.rating[index]

    def __len__(self):

        return self.n_interactions


'''