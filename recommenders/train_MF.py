import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd.variable
from torch.utils.data import Dataset, DataLoader
from recommenders.MF_MSE import MF
from dataset import Dataset
from recommenders.MF_MSE import DatasetIterator
from tqdm import tqdm
import time


# params
BATCH_SIZE = 128
EPOCHS = 10


# dataset
d = Dataset(split=0.8)
urm_train = d.get_URM_train()
urm_valid = d.get_URM_valid()

dataset_iter = DatasetIterator(urm_train)

train_data_loader = DataLoader(
    dataset=dataset_iter,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# model
model = MF(urm_train, 20)


# optimizer
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=1e-4
    # add weight decay
)

device = torch.device('cpu')
model.to(device)

def train_routine():
    model.train()
    optimizer.zero_grad()

    # pick user u item i

    pred = model()

for epoch in tqdm(EPOCHS):

    cumulative_loss = 0 
    ts = time.time()
    for (input_data, rating) in enumerate(train_data_loader, 0):
                
        coordinates = Variable(input_data).to(device)
        rating_tensor = Variable(rating).to(device)
                
        u = coordinates[:,0]
        i = coordinates[:,1]

        # forward pass
        prediction = model(u, i)

        # Pass prediction and label removing last empty dimension of prediction
        loss = nn.MSELoss(size_average=False)(prediction.view(-1), rating_tensor)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cumulative_loss /= len(train_data_loader)
    log = "[epoch: {:03d} | loss: {:.4f} | Time: {:.4f}s]"
    print(log.format(epoch, cumulative_loss, time.time() - ts))

from evaluator import Evaluator

e = Evaluator()