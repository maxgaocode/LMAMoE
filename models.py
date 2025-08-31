import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, GCNConv,SGConv
import numpy as np
from torch.nn import Parameter

from time import perf_counter as t
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret




##################################

from torch_geometric.utils import get_laplacian
from torch_geometric.utils import remove_self_loops, add_self_loops


class List_prop(MessagePassing):
   
    def __init__(self, K, bias=True, **kwargs):
        super(List_prop, self).__init__(aggr='add', **kwargs)
        self.K = K

    def forward(self, x1, high, edge_index, edge_weight):
        list_mat1 = []
        list_mat2 = []
        list_mat1.append(x1)
        list_mat2.append(high)

        # D^(-0.5)AD^(-0.5)
        #edge_index1, norm1 = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x1.dtype,
                                           num_nodes=x1.size(0))
        # 2I-L
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x1.size(0))

        for i in range(self.K):
            x1 = self.propagate(edge_index2, x=x1, norm=norm2, size=None)
            list_mat1.append(x1)
            high = self.propagate(edge_index1, x=high, norm=norm1, size=None)
            list_mat2.append(high)

        return list_mat1,list_mat2


#######################################

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class Newencoders(nn.Module):
    def __init__(self, nfeat, numhidden, num_projection, tau, K2, activation="relu",**kwargs):
        super(Newencoders, self).__init__()

        self.numhidden = numhidden
        self.tau = tau
        self.layer_num = K2
        self.lin1 = nn.ModuleList([nn.Linear(nfeat, self.numhidden) for _ in range(self.layer_num+1)])#there需要一个
        assert activation in ["relu", "leaky_relu", "elu"]
        self.activation = getattr(F, activation)

        self.edge_weight = None
        self.prop1 =List_prop(self.layer_num)
        self.layers = nn.ModuleList()
        self.fc1 = nn.Linear((self.layer_num+1)*self.numhidden, num_projection)
        self.fc2 = nn.Linear(num_projection, num_projection)

        self.readout = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(self.numhidden)

    def forward(self, x1, x2,  edge_index):
        list_mat1, list_mat2 = self.prop1(x1, x2,edge_index,None)

        list_out1 = list()
        for ind, mat in enumerate(list_mat1):
            tmp_out = self.lin1[ind](mat)
            tmp_out = F.normalize(tmp_out, p=2, dim=1)
            list_out1.append(tmp_out)
        final_mat1 = torch.cat(list_out1, dim=1)
        list_out2= list()
        for ind, mat in enumerate(list_mat2):
            tmp_out = self.lin1[ind](mat)
            tmp_out = F.normalize(tmp_out, p=2, dim=1)
            list_out2.append(tmp_out)
        final_mat2 = torch.cat(list_out2, dim=1)

      
        return  final_mat1,final_mat2, list_out1, list_out2

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    def semi_loss(self, z1):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        return refl_sim.sum(1)

    def adj_loss(self, z1, z2, diverNegative, adj):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        positive=torch.mul(refl_sim, adj)
       
        return -torch.log(positive.sum(1)
            / (diverNegative+ between_sim.sum(1)))

    
    def Diver_loss(self, z1, z2, diverNegative):
        f = lambda x: torch.exp(x / self.tau)        
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.sum(1)
                          / (diverNegative+ between_sim.sum(1)))

    def neighbor_gcl(self, z1,z2, list_out1, list_out2,neighOrder,adj):
        l, l1, l2, l3 = list_out1[0], list_out1[1], list_out1[2], list_out1[3]
        a, a1, a2, a3 = list_out2[0], list_out2[1], list_out2[2], list_out2[3]
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        f = lambda x: torch.exp(x / self.tau)
        diverN0=f(self.sim(l, a))
        diverN1=f(self.sim(l1, a1))
        diverN2=f(self.sim(l2, a2))
        diverN3=f(self.sim(l3, a3))
        diverNegative=diverN0.diag()+diverN1.diag()+diverN2.diag()+diverN3.diag()        

        # print("diverNegative.shape")
        # print(diverNegative.shape)

        if neighOrder == 0:
            l1 = self.Diver_loss(h1, h2, diverNegative)
        else:
            l1 = self.adj_loss(h1, h2, diverNegative,adj)

       
        ret = l1
        ret = ret.mean()


        return ret
    def InfoNCE_loss(self, z1, z2, diverNegative):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.sum(1)
                              /  (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    def loss(self, z1, z2, adj, batch_size: int = 0, order: int=1):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.I_loss(h1, h2, adj)

        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)

        ret = l1
        ret = ret.mean()

        return ret
