import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch.nn import BatchNorm1d, Identity

import torch.nn as nn
import math
import torch
import torch.nn.functional as F
class MLP(torch.nn.Module):
    """
    Multilayer Perceptron
    """
    def __init__(self,
                 channel_list,
                 dropout=0.,
                 batch_norm=True,
                 relu_first=False):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first

        self.linears = ModuleList()
        self.norms = ModuleList()
        for in_channel, out_channel in zip(channel_list[:-1],
                                           channel_list[1:]):
            self.linears.append(Linear(in_channel, out_channel))
            self.norms.append(
                BatchNorm1d(out_channel) if batch_norm else Identity())

    def forward(self, x):
        x = self.linears[0](x)
        for layer, norm in zip(self.linears[1:], self.norms[:-1]):
            if self.relu_first:
                x = F.relu(x)
            x = norm(x)
            if not self.relu_first:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer.forward(x)
        return x


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

class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return x