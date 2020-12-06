import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

import time
from recommenders.recommender import Recommender
import numpy as np
import similaripy as sim
from scipy import sparse 
from tqdm import tqdm
import pandas as pd

from time import strftime, gmtime



class MF_BPR(Recommender):

    NAME = 'MF_BPR'

    def __init__(self, urm, urm_df, n_factors=10, batch_size=32, loadmodel=False, savemodel=False):

        super().__init__(urm = urm)
        self.urm = urm
        self.urm_df = urm_df
        self.loadmodel = loadmodel
        self.savemodel = savemodel

        train_dataset = TripletsBPRDataset(self.urm_df)

        
        rnd_sampler = RandomSampler(
            train_dataset, 
            replacement=True, 
            num_samples=len(train_dataset)
        )
        
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=rnd_sampler, 
            num_workers=8
        )

        self.n_factors = n_factors
        self.n_users, self.n_items = self.urm.shape

        self.model = MF_BP_Model(
            self.n_users, 
            self.n_items, 
            self.n_factors,
            self.load_model
        )
        if self.loadmodel:
            self.load_model()


    
    def fit(self, epochs=5, lr=1e-4):

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=lr
        )

        device = torch.device('cpu')
        self.model.to(device)

        for epoch in range(epochs):
            cum_loss = 0
            t1 = time.time()
            for batch in tqdm(self.train_dataloader):
                # move the batch to the correct device
                batch = [b.to(device) for b in batch]
                #print(batch)
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.model(batch)
                loss.backward()
                self.optimizer.step()
                cum_loss += loss
            cum_loss /= len(self.train_dataloader)

            m = "[ Epoch: {} | Loss: {:.4f} | Time: {:.4f}s ]"
            print(m.format(epoch, cum_loss, time.time() - t1))

        user_emb = self.model.user_embeddings.detach().cpu().numpy()
        item_emb = self.model.item_embeddings.detach().cpu().numpy()

        self.r_hat = user_emb.dot(item_emb.T)

        print('name: {}, r_hat => type {}  shape {} '.format(self.NAME, type(self.r_hat), self.r_hat.shape))

        self.user_factors = user_emb
        self.item_factors = item_emb

        if self.savemodel:
            self.save_model()

    def tuning(self, urm_valid):

        BEST_MAP = 0.0
        BEST_EPOCHS = 0
        BEST_FACTORS = 0
        BEST_LR = 0

        epochs      = [10, 15, 20]
        n_factors   = [30, 70, 50, 100, 300]
        lrs         = [1e-4]

        total = len(epochs) * len(n_factors) * len(lrs)

        i = 0
        for t in epochs:
            for a in n_factors:
                for b in lrs:
                    self.fit(epochs=t, n_factors=a, lr=b)

                    self._evaluate(urm_valid)

                    m = '| iter: {:-5d}/{} | epochs: {:-3d} | factors: {:-3d} | lr: {} | MAP: {:.4f} |'
                    print(m.format(i, total, t, a, b, self.MAP))

                    i+=1

                    if self.MAP > BEST_MAP:

                        BEST_EPOCHS = t
                        BEST_FACTORS = a
                        BEST_LR = b
                        BEST_MAP = self.MAP
        log = '| best results | epochs: {:-3d} | factors: {:-3d} | lr: {} | MAP: {:.4f} |'
        print(log.format(BEST_EPOCHS, BEST_FACTORS, BEST_LR, BEST_MAP))

    
    def save_model(self):
        timestamp = strftime("%d-%m-%Y-%H:%M:%S", gmtime())
        PATH = 'recommenders/models/MF-BPR-{}-NF-{}.pt'
        PATH.format(timestamp, self.n_factors)
        torch.save(self.model.state_dict(), PATH)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        
        


class MF_BP_Model(nn.Module):

    def __init__(self, n_users, n_items, n_factors=10, loadmodel=False):

        nn.Module.__init__(self)

        self.n_factors = n_factors                
        self.n_users = n_users 
        self.n_items = n_items
        
        # initialize user embeddings
        self.user_embeddings = nn.Parameter(
            torch.empty( # a matrix n° users x n° factors
                self.n_users,
                self.n_factors
            )
        )
        # apply xavier distr 
        nn.init.xavier_normal_(self.user_embeddings)

        # initialize user embeddings
        self.item_embeddings = nn.Parameter(
            torch.empty( # a matrix n° items x n° factors
                self.n_items,
                self.n_factors
            )
        )
        # apply xavier distr 
        nn.init.xavier_normal_(self.item_embeddings)

    def forward(self, x):

        x_u = torch.index_select(self.user_embeddings, 0, x[0])
        x_i = torch.index_select(self.item_embeddings, 0, x[1])
        x_j = torch.index_select(self.item_embeddings, 0, x[2])

        x_ui = torch.einsum("nf,nf->n", x_u, x_i)
        x_uj = torch.einsum("nf,nf->n", x_u, x_j)

        x_uij = x_ui - x_uj

        loss = -F.logsigmoid(x_uij).sum()
        return loss


class TripletsBPRDataset(Dataset):

    def __init__(self, urm_df):

        self.urm_df = urm_df
        
        self.interactions_list = [
            (u, i) for u, i in zip(urm_df['user_id'], urm_df['item_id'])
        ]

        self.n_users = 7947
        self.n_items = 25975
        self.items_of_user = {}
        for x in range(0, self.n_users):
            self.items_of_user[x] = self._items_of_user(x)
    
    def __getitem__(self, item):

        u, i = self.interactions_list[item]

        j = np.random.randint(0, self.n_items)
        
        while j in self.items_of_user[u]:
            j = np.random.randint(0, self.n_items)

        return u, i, j


    def _items_of_user(self, user):
        """ 
        Create a list of items visualized by a user
        
        Args:
            user: (int) represent user id

        Returns:
            items: (list) of int, items the user interacted with 
        """
        items = []

        for i in self.interactions_list:
            if i[0] == user:
                items.append(i[1])
        return items


    def __len__(self):
        return len(self.interactions_list)












