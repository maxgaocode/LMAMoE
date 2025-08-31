#coding=utf-8
import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from torch_geometric.utils import dropout_adj

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from models import Newencoders
import scipy.sparse  as sp
from scipy.sparse import coo_matrix

import numpy as np
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

#################################################

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, y, t=2):
    uniloss1 =torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    uniloss2 = torch.pdist(y, p=2).pow(2).mul(-t).exp().mean().log()
    return (uniloss1+uniloss2)*0.5


##################################
def cooadj2Sparsetentor(coo_adj):
    ' there is a transformation '
    values = coo_adj.data
    indices = np.vstack((coo_adj.row, coo_adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_adj.shape
    # torch_sparse_adj is torch.sparse matrix
    adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    print(adj.shape)
    adj=adj.to_dense()
    return adj
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))*1.0

   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def train(model, data, adj):
    model.train()
    optimizer.zero_grad()
    x=data.x
    x1=drop_feature(x, args.dropX1)
    batch_size=1
    nb_nodes=x.shape[0]

    z1, z2, list_out1,list_out2 = model(x1, x, data.edge_index)

  


    loss = model.neighbor_gcl(z1, z2,list_out1,list_out2,1,adj)
    # #loss2 = model.local_global(list_out1, list_out2,lbl)
    # loss= loss1#+args.alpha*loss2
    loss.backward()
    optimizer.step()

    return z1,z2, loss.item()

###################################################
from pGRACE.eval import log_regression, MulticlassEvaluator

def test(model, data, branch, final=False):
    model.eval()
    x, edge_index, y =data.x, data.edge_index, data.y,
    z,_,list_out1,list_out2 = model(x,x, edge_index)

 
    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        accs_1 = []
        accs_2 = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        if args.dataset == 'cora' or args.dataset == 'citeseer' or  args.dataset == 'pubmed':
            #acc = log_regression(z, dataset, evaluator, split='preloaded', num_epochs=3000, preload_split=0)['acc']
            acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=300, preload_split=0)['acc']
        else :
            acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=300, preload_split=0)['acc']


    return acc
#################################
def get_adj(I, adj,order):
    adj = adj.to_dense()
    if order==0:
       adj =I
    elif order==2:
        adj=torch.mm(adj, adj)
    elif order==3:
        adj2=torch.mm(adj, adj)
        adj = torch.mm(adj2, adj)
    elif order == 4:
        adj2 = torch.mm(adj, adj)
        adj = torch.mm(adj2, adj2)
    return adj.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--gpu_id', type=int, default=0)  
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--K1', type=int, default=3)
    parser.add_argument('--K2', type=int, default=3)
    parser.add_argument('--dropX1', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--tau', type=float, default=0.6)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--order', type=int, default=1)

    parser.add_argument('--net', type=str, default='LMAMoE')#LightGCL
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

   
    from datasets import *
    dataset = get_data(args)


    #np.save('label_{}.npy'.format(args.dataset),data.y)

    data =dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=Newencoders(data.x.shape[1], args.hidden, args.hidden, args.tau, args.K2, activation="relu").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    coo_adj = coo_matrix((np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
                         shape=(data.num_nodes, data.num_nodes))
    I = coo_matrix((np.ones(data.num_nodes), (np.arange(0, data.num_nodes, 1), np.arange(0, data.num_nodes, 1))),
                   shape=(data.num_nodes, data.num_nodes))
    I =sparse_mx_to_torch_sparse_tensor(coo_adj)
    adj = sparse_mx_to_torch_sparse_tensor(sys_normalized_adjacency(coo_adj))
    # adj=cooadj2Sparsetentor(coo_adj)
    adj=get_adj(I,adj,args.order)
    data = data.to(device)

    print(args.dataset)
    print(data)
    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        now = t()
        z1, z2, loss = train(model, data, adj)
        # np.save("./savefigures/{}_z1.npy".format(args.dataset), z1.detach().cpu().numpy())
        # np.save("./savefigures/{}_z2.npy".format(args.dataset), z2.detach().cpu().numpy())

        if epoch % 5 == 0:
             print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, ')
         
        prev = now
    acc = []
    for i in range(10):
        testacc= test(model, data, 1, final=True)
        print(testacc)
        acc.append(testacc)
    print(acc)
    meanmicro = sum(acc) / 10
    m1 = np.std(acc)

    print("=== Final ===")
    print(args.dataset)
    print(meanmicro)
    filename = f'{args.net}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write("net:{}, dataset:{}, hidden:{}, lr:{}, K1:{},order:{}, dropX1:{}, tau:{}, epoch:{}, meanmicro:{}, std:{} "
                        .format(args.net, args.dataset, args.hidden, args.lr, args.K1, args.order, args.dropX1, args.tau, args.num_epochs,
                                meanmicro, m1))
        write_obj.write("\n")

