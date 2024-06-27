from re import S
from xml.dom import xmlbuilder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import num_nodes, to_dense_adj
import numpy as np
import math
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from backbone.gnn.mlp import Linear

class GDG(nn.Module):  # residual connection
    def __init__(self, in_channels,
                 out_channels,
                 hidden_channels=64,
                 max_depth=2,
                 dropout=0.5):
        super(GDG, self).__init__()

        self.conv = GDG_conv(in_channels,
                 hidden_channels,
                 hidden_channels=hidden_channels,
                 max_depth=max_depth,
                 dropout=dropout)

        self.mlp = GDG_conv(in_channels,
                 hidden_channels,
                 hidden_channels=hidden_channels,
                 max_depth=max_depth,
                 dropout=dropout)

        # self.att_model = Attention(hidden_channels)
        #
        # self.classifier = MLP_classifier(nfeat=hidden_channels,
        #                                  nclass=out_channels,
        #                                  dropout=dropout)
        self.proj_head = Linear(hidden_channels, hidden_channels, dropout, bias=True)

    def forward(self,data):
        out = self.conv(data)
        return out
        # return out1

    def features(self,data):
        out = self.conv.features(data)
        return out

    def project(self,features):
        out = self.proj_head(features)
        return out

class GDG_conv(nn.Module):  # residual connection
    def __init__(self, in_channels,
                 out_channels,
                 hidden_channels=64,
                 max_depth=2,
                 dropout=0.5):
        super(GDG_conv, self).__init__()
        # self.args = args

        self.num_mps = 5

        self.dropout = dropout
        # self.num_node = num_node
        self.num_layers = max_depth
        self.ff_bias = True
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(
            nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias))  # 1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias))  # 1
        self.reset_parameters()
    def reset_parameters(self):
        for mlp in self.fcs:
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def reset_parameters(self):
        for mlp in self.fcs:
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, data, use_conv=True):
        x = data.x
        edge_index = data.edge_index
        for i in range(self.num_layers - 1):
            x = x @ self.fcs[i].weight.t()
            if use_conv: x = gcn_conv(x, edge_index)  # Optionally replace 'gcn_conv' with other conv functions in conv.py
            if self.ff_bias: x = x + self.fcs[i].bias
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x @ self.fcs[-1].weight.t()
        if use_conv: x = gcn_conv(x, edge_index)
        if self.ff_bias: x = x + self.fcs[-1].bias
        x = torch.log_softmax(x, dim=1)
        return x

    def features(self,data, use_conv=True):
        x = data.x
        edge_index = data.edge_index
        x = x @ self.fcs[0].weight.t()
        if use_conv: x = gcn_conv(x, edge_index)  # Optionally replace 'gcn_conv' with other conv functions in conv.py
        if self.ff_bias: x = x + self.fcs[0].bias
        x = self.activation(x)
        return x

class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs

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


class MLP_classifier(nn.Module):#
    def __init__(self, nfeat, nclass, dropout):
        super(MLP_classifier, self).__init__()
        self.Linear1 = Linear(nfeat, nclass, dropout, bias=True)

    def forward(self, x):
        out = self.Linear1(x)
        return torch.log_softmax(out, dim=1), out


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output