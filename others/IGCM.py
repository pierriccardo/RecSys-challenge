
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:02:15 2019
@author: YuxuanLong
"""

import torch
import torch.nn as nn
import torch.sparse as sp
import numpy as np
import utils


def sparse_drop(feature, drop_out):
    tem = torch.rand((feature._nnz()))
    feature._values()[tem < drop_out] = 0
    return feature

class GCMC(nn.Module):

    
    def __init__(self, feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side, out_dim, drop_out = 0.0):
        super(GCMC, self).__init__()
        ###To Do:
        #### regularization on Q
        
        self.drop_out = drop_out
        
        side_feature_u_dim = side_feature_u.shape[1]
        side_feature_v_dim = side_feature_v.shape[1]
        self.use_side = use_side

        self.feature_u = feature_u
        self.feature_v = feature_v
        self.rate_num = rate_num
        
        self.num_user = feature_u.shape[0]
        self.num_item = feature_v.shape[1]
        
        self.side_feature_u = side_feature_u
        self.side_feature_v = side_feature_v
        
        self.W = nn.Parameter(torch.randn(rate_num, feature_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W, mode = 'fan_out', nonlinearity = 'relu')
        
        self.all_M_u = all_M_u
        self.all_M_v = all_M_v
        
        self.reLU = nn.ReLU()
        
        if use_side:
            self.linear_layer_side_u = nn.Sequential(*[nn.Linear(side_feature_u_dim, side_hidden_dim, bias = True), 
                                                       nn.BatchNorm1d(side_hidden_dim), nn.ReLU()])
            self.linear_layer_side_v = nn.Sequential(*[nn.Linear(side_feature_v_dim, side_hidden_dim, bias = True), 
                                                       nn.BatchNorm1d(side_hidden_dim), nn.ReLU()])
    
            self.linear_cat_u = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2 + side_hidden_dim, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])
            self.linear_cat_v = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2 + side_hidden_dim, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])    
        else:
            
            self.linear_cat_u = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])
            self.linear_cat_v = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])

#        self.linear_cat_u2 = nn.Sequential(*[nn.Linear(out_dim, out_dim, bias = True), 
#                                            nn.BatchNorm1d(out_dim), nn.ReLU()])
#        self.linear_cat_v2 = nn.Sequential(*[nn.Linear(out_dim, out_dim, bias = True), 
#                                            nn.BatchNorm1d(out_dim), nn.ReLU()])
    
        self.Q = nn.Parameter(torch.randn(rate_num, out_dim, out_dim))
        nn.init.orthogonal_(self.Q)
        
        
    def forward(self):
        
        feature_u_drop = sparse_drop(self.feature_u, self.drop_out) / (1.0 - self.drop_out)
        feature_v_drop = sparse_drop(self.feature_v, self.drop_out) / (1.0 - self.drop_out)
        
        hidden_feature_u = []
        hidden_feature_v = []
        
        W_list = torch.split(self.W, self.rate_num)
        W_flat = []
        for i in range(self.rate_num):
            Wr = W_list[0][i]
            M_u = self.all_M_u[i]
            M_v = self.all_M_v[i]
            hidden_u = sp.mm(feature_v_drop, Wr)
            hidden_u = self.reLU(sp.mm(M_u, hidden_u))
            
            ### need to further process M, normalization
            hidden_v = sp.mm(feature_u_drop, Wr)
            hidden_v = self.reLU(sp.mm(M_v, hidden_v))

            
            hidden_feature_u.append(hidden_u)
            hidden_feature_v.append(hidden_v)
            
            W_flat.append(Wr)
            
        hidden_feature_u = torch.cat(hidden_feature_u, dim = 1)
        hidden_feature_v = torch.cat(hidden_feature_v, dim = 1)
        W_flat = torch.cat(W_flat, dim = 1)


        cat_u = torch.cat((hidden_feature_u, torch.mm(self.feature_u, W_flat)), dim = 1)
        cat_v = torch.cat((hidden_feature_v, torch.mm(self.feature_v, W_flat)), dim = 1)
        
        if self.use_side:
            side_hidden_feature_u = self.linear_layer_side_u(self.side_feature_u)
            side_hidden_feature_v = self.linear_layer_side_v(self.side_feature_v)    
            
            cat_u = torch.cat((cat_u, side_hidden_feature_u), dim = 1)
            cat_v = torch.cat((cat_v, side_hidden_feature_v), dim = 1)
        
        
        embed_u = self.linear_cat_u(cat_u)
        embed_v = self.linear_cat_v(cat_v)
        
        score = []
        Q_list = torch.split(self.Q, self.rate_num)
        for i in range(self.rate_num):
            Qr = Q_list[0][i]
            
            tem = torch.mm(torch.mm(embed_u, Qr), torch.t(embed_v))
            
            score.append(tem)
        return torch.stack(score)


class Loss(nn.Module):
    def __init__(self, all_M, mask, user_item_matrix, laplacian_loss_weight):
            
        super(Loss, self).__init__()
            
        self.all_M = all_M
        self.mask = mask
        self.user_item_matrix = user_item_matrix
        
        self.rate_num = all_M.shape[0]
        self.num = float(mask.sum())
        
        self.logsm = nn.LogSoftmax(dim = 0)
        self.sm = nn.Softmax(dim = 0)
        self.laplacian_loss_weight = laplacian_loss_weight
        
    def cross_entropy(self, score):
        l = torch.sum(-self.all_M * self.logsm(score))
        return l / self.num
    
    def rmse(self, score):
        score_list = torch.split(self.sm(score), self.rate_num)
        total_score = 0
        for i in range(self.rate_num):
            total_score += (i + 1) * score_list[0][i]
        
        square_err = torch.pow(total_score * self.mask - self.user_item_matrix, 2)
        mse = torch.sum(square_err) / self.num
        return torch.sqrt(mse)
        
    def loss(self, score):
        return self.cross_entropy(score) + self.rmse(score)
    
    def laplacian_loss(self, score, laplacian_u, laplacian_v):
        
        score_list = torch.split(self.sm(score), self.rate_num)
        total_score = 0
        for i in range(self.rate_num):
            total_score += (i + 1) * score_list[0][i]
        
        dirichlet_r = torch.mm(torch.mm(torch.transpose(total_score,0,1), laplacian_u.to(torch.float)), total_score.to(torch.float))
        dirichlet_c = torch.mm(torch.mm(total_score.to(torch.float),laplacian_v.to(torch.float)),torch.transpose(total_score.to(torch.float),0,1))

        dirichlet_norm_r = torch.trace(dirichlet_r)/(laplacian_u.shape[0] * laplacian_u.shape[1])
        dirichlet_norm_c = torch.trace(dirichlet_c)/(laplacian_v.shape[0] * laplacian_v.shape[1])
        
        
        return self.laplacian_loss_weight*(dirichlet_norm_r + dirichlet_norm_c) + (1-self.laplacian_loss_weight)*self.loss(score)

import numpy as np
import torch
from torch.autograd import Variable



# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# Set random seeds
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)


def np_to_var(x):
    """
    Convert numpy array to Torch variable.
    """
    x = torch.from_numpy(x)
    if RUN_ON_GPU:
        x = x.cuda()
    return Variable(x)


def var_to_np(x):
    """
    Convert Torch variable to numpy array.
    """
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def normalize(M):
    s = np.sum(M, axis = 1)
    s[s == 0] = 1
    return (M.T / s).T


def create_models(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side, out_dim, drop_out = 0.0):
    """
    Choose one model from our implementations
    """
    side_feature_u = np_to_var(side_feature_u.astype(np.float32))
    side_feature_v = np_to_var(side_feature_v.astype(np.float32))
    
    for i in range(rate_num):
        all_M_u[i] = to_sparse(np_to_var(all_M_u[i].astype(np.float32)))
        all_M_v[i] = to_sparse(np_to_var(all_M_v[i].astype(np.float32)))   
    
    feature_u = to_sparse(np_to_var(feature_u.astype(np.float32)))
    feature_v = to_sparse(np_to_var(feature_v.astype(np.float32)))

    net = model.GCMC(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side, out_dim, drop_out)

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        net.cuda()
    else:
        print('Keeping models on CPU.')

    return net


def epsilon_similarity_graph(X: np.ndarray, sigma=None, epsilon=0):
    """ X (n x d): coordinates of the n data points in R^d.
        sigma (float): width of the kernel
        epsilon (float): threshold
        Return:
        adjacency (n x n ndarray): adjacency matrix of the graph.
    """
    # Your code here
    W = np.array([np.sum((X[i] - X)**2, axis = 1) for i in range(X.shape[0])])
    typical_dist = np.mean(np.sqrt(W))
    # print(np.mean(W))
    c = 0.35
    if sigma == None:
        sigma = typical_dist * c
    
    mask = W >= epsilon
    
    adjacency = np.exp(- W / 2.0 / (sigma ** 2))
    adjacency[mask] = 0.0
    adjacency -= np.diag(np.diag(adjacency))
    return adjacency

def compute_laplacian(adjacency: np.ndarray, normalize: bool):
    """ Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    # Your code here
    d = np.sum(adjacency, axis = 1)
    d_sqrt = np.sqrt(d)  
    D = np.diag(1 / d_sqrt)
    if normalize:
        L = np.eye(adjacency.shape[0]) - (adjacency.T / d_sqrt).T / d_sqrt
        # L = np.dot(np.dot(D, np.diag(d) - adjacency), D)
    else:
        L = np.diag(d) - adjacency
    return L


def loss(all_M, mask, user_item_matrix, laplacian_loss_weight):
    all_M = np_to_var(all_M.astype(np.float32))
    mask = np_to_var(mask.astype(np.float32))
    user_item_matrix = np_to_var(user_item_matrix.astype(np.float32))
    
    return Loss(all_M, mask, user_item_matrix, laplacian_loss_weight)