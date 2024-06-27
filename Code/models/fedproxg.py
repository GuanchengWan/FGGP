import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from models.utils.federated_model import FederatedModel
import torch
import torch.nn.functional as F

class FedProxg(FederatedModel):
    NAME = 'fedproxg'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProxg, self).__init__(nets_list, args, transform)
        self.mu = args.mu

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = F.nll_loss
        iterator = tqdm(range(self.local_epoch))
        train_loader = train_loader.to(self.device)
        self.global_net = self.global_net.to(self.device)
        global_weight_collector = list(self.global_net.parameters())
        for _ in iterator:
            out = net(train_loader)
            loss = criterion(out[train_loader.train_mask], train_loader.y[train_loader.train_mask])
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((0.01 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += self.mu * fed_prox_reg
            optimizer.zero_grad()
            loss.backward()
            iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
            optimizer.step()
