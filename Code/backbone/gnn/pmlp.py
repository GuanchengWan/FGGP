from re import S
from xml.dom import xmlbuilder

import scipy as sp
import torch
import torch.nn as nn
import pyro
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import num_nodes, to_dense_adj
import numpy as np
import math
from backbone.ETF_classifier import ETF_Classifier
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from backbone.gnn.mlp import Linear
# Implementation of PMLP_GCN, which can become MLP or GCN depending on whether using message passing
class PMLP_GCN(nn.Module): 
    def __init__(self,in_channels,
                 out_channels,
                 hidden_channels=64,
                 max_depth=3,
                 dropout=0.5):
        super(PMLP_GCN, self).__init__()
        self.args = ''
        self.dropout = dropout
        self.num_layers = max_depth
        self.ff_bias = True
        self.hidden_channels  = hidden_channels
        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.proj_head = Linear(hidden_channels, hidden_channels, dropout, bias=True)
        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias))
        # self.etf = ETF_Classifier(hidden_channels,out_channels)
        self.reset_parameters()
        self.aug = Augmentor(in_channels, hidden_channels)
    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, data, use_conv=True, adj=False):
        x = data.x
        edge_index = data.edge_index
        if adj:  adj_ma = data.adj
        for i in range(self.num_layers - 1):
            x = x @ self.fcs[i].weight.t() 
            if adj:
                x = gcn_conv_adj(x, adj_ma) # Optionally replace 'gcn_conv' with other conv functions in conv.py
            else:
                x = gcn_conv(x, edge_index)
            if self.ff_bias: x = x + self.fcs[i].bias
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.classifier(x,edge_index)
        return x

    def features(self,data,use_conv=True,adj=False):
        x = data.x
        edge_index = data.edge_index
        if adj:      adj_ma = data.adj
        for i in range(self.num_layers - 1):
            x = x @ self.fcs[i].weight.t()
            if adj:
                x = gcn_conv_adj(x, adj_ma)  # Optionally replace 'gcn_conv' with other conv functions in conv.py
            else:
                x = gcn_conv(x, edge_index)
            if self.ff_bias: x = x + self.fcs[i].bias
            if i != self.num_layers - 2:
                x = self.activation(self.bns(x))
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.activation(x)
        return x

    def classifier(self,x,edge_index,use_conv=True):
        x = x @ self.fcs[-1].weight.t()
        if use_conv: x = gcn_conv(x, edge_index)
        if self.ff_bias: x = x + self.fcs[-1].bias
        x = torch.log_softmax(x, dim=1)
        return x

    def augment(self,x):
        return self.aug(x)



# Implementation of PMLP_SGC, which can become MLP or SGC depending on whether using message passing
class PMLP_SGC(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 hidden_channels=64,
                 max_depth=2,
                 dropout=0.5):
        super(PMLP_SGC, self).__init__()
        # self.args = args
        self.dropout = dropout
        self.num_mps = 5
        # self.num_node = num_node
        self.num_layers = max_depth
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.ff_bias = True

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, data, use_conv=True):
        x = data.x
        edge_index = data.edge_index
        adj = data.adj
        for i in range(self.num_mps):
            if use_conv: x = gcn_conv_adj(x, edge_index)

        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fcs[-1](x) 
        return x


# Implementation of PMLP_APP, which can become MLP or SGC depending on whether using message passing
class PMLP_APPNP(nn.Module): #residual connection
    def __init__(self, in_channels,
                 out_channels,
                 hidden_channels=64,
                 max_depth=2,
                 dropout=0.5):
        super(PMLP_APPNP, self).__init__()
        # self.args = args
        self.dropout = dropout
        # self.num_node = num_node
        self.num_layers = max_depth
        self.ff_bias = True
        self.num_mps = 5
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, data,use_conv=True):
        x = data.x
        edge_index = data.edge_index
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.fcs[-1](x) 
        for i in range(self.num_mps):
            if use_conv: x = gcn_conv(x, edge_index)    
        return x
    


# The rest models are used for additional experiments in the paper

# Implementation of PMLP_GCNII, which can become ResNet (MLP with residual connections) or GCNII depending on whether using message passing
class PMLP_GCNII(nn.Module): #GCNII
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_GCNII, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        x = x @ self.fcs[0].weight.t()
        x = self.activation(self.bns(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_ = x.clone()

        for i in range(1, self.num_layers - 1):
            x = x * (1. - 0.5 / i) + x @ self.fcs[i].weight.t() * (0.5 / i) 
            if use_conv: x = conv_resi(x, edge_index, x_)
            else: x = 0.9 * x + 0.1 * x_
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x =  x @ self.fcs[-1].weight.t() 
        if use_conv: x = gcn_conv(x, edge_index)
        return x


class PMLP_JKNet(nn.Module): #JKNET(concatation pooling)
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_JKNet, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels * (self.num_layers - 1), out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        xs = []
        for i in range(0, self.num_layers - 1):
            x = x @ self.fcs[i].weight.t() 
            if use_conv: x = gcn_conv(x, edge_index)
            x = self.activation(self.bns(x))
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.cat(xs, dim=-1)
        x =  x @ self.fcs[-1].weight.t() 
        return x

class PMLP_SGCres(nn.Module): #SGC with residual connections
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_SGCres, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.ff_bias = True

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)

        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fcs[-1](x) 
        return x



class PMLP_SGCresinf(nn.Module): #SGC with residual connections (in test but not in train)
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_SGCresinf, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.ff_bias = True

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)
            else: x =  gcn_conv(x, edge_index)

        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fcs[-1](x) 
        return x
    
    
class PMLP_APPNPres(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_APPNPres, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.fcs[-1](x) 
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)
        return x


class PMLP_APPNPresinf(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_APPNPresinf, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.fcs[-1](x) 
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)
            else: x =  gcn_conv(x, edge_index)
        return x


def gcn_conv(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)

    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = a @ h
    return h_prime

def gcn_conv_adj(h, adj):
    h_prime = torch.sparse.mm(adj, h)
    return h_prime


def conv_noloop(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)

    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5)  # 1/d^0.5(v_i)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5)  # 1/d^0.5(v_j)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = a @ h
    return h_prime


def conv_rw(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)

    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-1)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = a @ h
    return h_prime


def conv_diff(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)

    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-1)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = heat_kernel(a, h, 10)
    return h_prime


def heat_kernel(a, h, k):
    h_prime = h, h_temp = h
    for i in range(1, k + 1):
        h_temp = a @ h_temp / i
        h_prime += h_temp
    return h_prime / math.e


def conv_resi(h, edge_index, h_ori, alpha):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)

    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5)  # 1/d^0.5(v_i)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5)  # 1/d^0.5(v_j)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = (1. - alpha) * a @ h + alpha * h_ori
    return h_prime


class Augmentor(nn.Module):
    def __init__(self, in_dim, hid_dim, alpha=0.5, temp=1.0):
        super(Augmentor, self).__init__()

        self.gcn1 = GCNLayer(in_dim, hid_dim, None, 0, bias=False)
        self.gcn2 = GCNLayer(hid_dim, hid_dim, F.relu, 0, bias=False)

        self.alpha = alpha
        self.temp = temp

    def forward(self, data):
        x = data.x
        adj_orig = data.adj_orig
        adj_norm = data.adj_norm

        # Parameterized Augmentation Distribution
        h = self.gcn1(adj_norm, x)
        h = self.gcn2(adj_norm, h)
        adj_logits = h @ h.T

        edge_probs = adj_logits / (torch.max(adj_logits) + 1e-6 )
        edge_probs = self.alpha * edge_probs + (1-self.alpha) * adj_orig

        # Gumbel-Softmax Sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temp, probs=edge_probs).rsample()
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T

        adj_sampled.fill_diagonal_(1)
        D_norm = torch.diag(torch.pow(adj_sampled.sum(1), -0.5))
        adj_sampled = D_norm @ adj_sampled @ D_norm

        return adj_sampled, adj_logits

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, dropout=0.3, bias=True):
        super(GCNLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = dropout

    def forward(self, adj, x):
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.linear(x)
        h = adj @ h

        if self.activation:
            h = self.activation(h)

        return h

def scipysp_to_pytorchsp(sp_mx):

    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()

    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape

    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T), torch.FloatTensor(values), torch.Size(shape))

    return pyt_sp_mx