import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from models.utils.federated_model import FederatedModel
import torch
import torch.nn.functional as F

class MOONg(FederatedModel):
    NAME = 'moong'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(MOONg, self).__init__(nets_list, args, transform)
        self.prev_nets_list = []
        self.temperature_moon = args.temperature_moon
        self.mu_moon = args.mu_moon

    def ini(self):
        for j in range(self.args.parti_num):
            self.prev_nets_list.append(copy.deepcopy(self.nets_list[j]))
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], self.prev_nets_list[i], priloader_list[i])
        self.copy_nets2_prevnets()
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, prev_net, train_loader):
        net = net.to(self.device)
        prev_net = prev_net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        train_loader = train_loader.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        self.global_net = self.global_net.to(self.device)
        cos = torch.nn.CosineSimilarity(dim=-1)
        for _ in iterator:
            f = net.features(train_loader)
            pre_f = prev_net.features(train_loader)
            g_f = self.global_net.features(train_loader)
            posi = cos(f, g_f)
            temp = posi.reshape(-1, 1)
            nega = cos(f, pre_f)
            temp = torch.cat((temp, nega.reshape(-1, 1)), dim=1)
            temp /= self.temperature_moon
            temp = temp[train_loader.train_mask].to(self.device)
            targets = torch.zeros(train_loader.y[train_loader.train_mask].size(0)).to(self.device).long()
            lossCON = self.mu_moon * criterion(temp, targets)
            out = net(train_loader)
            lossCE = criterion(out[train_loader.train_mask], train_loader.y[train_loader.train_mask])
            loss = lossCE + lossCON
            optimizer.zero_grad()
            loss.backward()
            iterator.desc = "Local Pariticipant %d CE = %0.3f,CON = %0.3f" % (index, lossCE, lossCON)
            optimizer.step()
