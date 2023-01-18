from __future__ import division
from __future__ import print_function

import scipy.sparse as sp
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from expanded_gcn.models import GCNModified, GCN
from expanded_gcn.utils import load_data, accuracy, get_split, sparse_mx_to_torch_sparse_tensor, normalize


no_cuda = False
seed=42

cuda = not no_cuda and torch.cuda.is_available()

def train(epoch, model, optim, idx_train, idx_val):
    model.train()
    optim.zero_grad()
    output = model(features, adjs)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optim.step()
    
    # to clamp parameters to a certain edge. originally is disabled
    # for child in list(model.children())[0]:
    #     if hasattr(child, 'gamma'):
    #         child.gamma.data.clamp_(0, 1.0)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    res = {'epoch':epoch+1,
          'loss_train':loss_train.item(),
          'acc_train':acc_train.item(),
          'loss_val':loss_val.item(),
          'acc_val':acc_val.item()}
    
    return res


def test(model, idx_test):
    model.eval()
    output = model(features, adjs)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test.item()


dataset = 'cora'

epochs=100
lr=0.01
weight_decay=5e-4
hidden=16
dropout=0.5
n_epochs_early_stop = 20


adj_gcn, adj, features, labels = load_data(path=f"../data/{dataset}/")

adj0 = sp.eye(adj.shape[0], dtype=np.float32)
adj0 = sparse_mx_to_torch_sparse_tensor(adj0)

row = []
col = []
data = []

for i in range(adj.shape[0]):
    n_i = adj[i].indices
    for j in n_i:
        n_j = adj[j].indices
        for n in n_j:
            if n!=i and n not in n_i:
                row.append(i)
                col.append(n)
                data.append(1)

adj2 = sp.csr_matrix((data, (row, col)), shape=(adj.shape[0], adj.shape[1]), dtype=np.float32)
adj2 = normalize(adj2)

row = []
col = []
data = []

for i in range(adj2.shape[0]):
    n_2_i = adj2[i].indices
    n_i = adj[i].indices
    for j in n_2_i:
        n_j = adj[j].indices
        for n in n_j:
            if n!=i and n not in n_2_i and n not in n_i:
                row.append(i)
                col.append(n)
                data.append(1)

adj3 = sp.csr_matrix((data, (row, col)), shape=(adj2.shape[0], adj2.shape[1]), dtype=np.float32)
adj3 = normalize(adj3)

adj = sparse_mx_to_torch_sparse_tensor(adj)
adj2 = sparse_mx_to_torch_sparse_tensor(adj2)
adj3 = sparse_mx_to_torch_sparse_tensor(adj3)

if cuda:
    features = features.cuda()
    adj_gcn = adj_gcn.cuda()
    adj0 = adj0.cuda()
    adj = adj.cuda()
    adj2 = adj2.cuda()
    adj3 = adj3.cuda()
    labels = labels.cuda()

adjs = [adj_gcn, adj0, adj, adj2, adj3]

models = {
    'GCN':GCN(nfeat=features.shape[1],
            nlayers = 2,
            nhid=hidden,
            nclass=labels.max().item() + 1,
            dropout=dropout),
    'GCN-2':GCNModified(nfeat=features.shape[1],
            nlayers = 2,
            nhid=hidden,
            nclass=labels.max().item() + 1,
            nneighbors=2,
            dropout=dropout),
    'GCN-3':GCNModified(nfeat=features.shape[1],
            nlayers = 2,
            nhid=hidden,
            nclass=labels.max().item() + 1,
            nneighbors=3,
            dropout=dropout)
}

for name, model in models.items():
    print('Training model {} on {}...'.format(name, dataset[:-1]))
    for n_train in ([1,2,5,10,15,20]):
        
        idx_train, idx_val, idx_test = get_split(labels, n_train=n_train, n_val=30)
        
        if cuda:
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
            model.cuda()

        for layer in list(model.children())[0]:
            layer.reset_parameters()

        optimizer = optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)

        epoch_results = []
        count_early_stop = 0
        best_val = 0
        best_model = model
        for epoch in range(epochs):
            epoch_results = train(epoch, model, optimizer, idx_train, idx_val)
            if epoch_results['acc_val'] > best_val:
                count_early_stop = 0
                best_val = epoch_results['acc_val']
                best_model = model
            else:
                count_early_stop += 1
                if count_early_stop >= n_epochs_early_stop:
                    break
    
        # Testing
        acc = test(best_model, idx_test)

        print(f"Test Accuracy = {acc}")