import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import torch.nn.functional as F

# https://github.com/yuetan031/fedproto

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedProtog.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


class FedProtog(FederatedModel):
    NAME = 'fedprotog'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProtog, self).__init__(nets_list, args, transform)
        self.mu = args.mu
        self.global_protos = []
        self.local_protos = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def proto_aggregation(self,local_protos_list):
        agg_protos_label = dict()
        for idx in self.online_clients:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label


    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.global_protos=self.proto_aggregation(self.local_protos)
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = F.nll_loss
        train_loader = train_loader.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:
            agg_protos_label = {}
            optimizer.zero_grad()
            outputs = net(train_loader)
            lossCE = criterion(outputs[train_loader.train_mask], train_loader.y[train_loader.train_mask])
            f = net.features(train_loader)
            loss_mse = nn.MSELoss()
            if len(self.global_protos) == 0:
                lossProto = 0*lossCE
            else:
                f_new = copy.deepcopy(f.data)
                i = 0
                for label in train_loader.y[train_loader.train_mask]:
                    if label.item() in self.global_protos.keys():
                        f_new[i, :] = self.global_protos[label.item()][0].data
                    i += 1
                lossProto = loss_mse(f_new, f)

            lossProto = lossProto * self.mu

            loss = lossCE + lossProto
            loss.backward()
            iterator.desc = "Local Pariticipant %d CE = %0.3f,Proto = %0.3f" % (index, lossCE, lossProto)
            optimizer.step()

            if iter == self.local_epoch-1:
                for i in range(len(train_loader.y[train_loader.train_mask])):
                    if train_loader.y[train_loader.train_mask][i].item() in agg_protos_label:
                        agg_protos_label[train_loader.y[train_loader.train_mask][i].item()].append(f[i,:])
                    else:
                        agg_protos_label[train_loader.y[train_loader.train_mask][i].item()] = [f[i,:]]

        agg_protos = agg_func(agg_protos_label)
        self.local_protos[index] = agg_protos
