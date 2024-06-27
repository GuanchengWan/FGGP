import os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

from torch_geometric.io import read_txt_array
from utils.conf import data_path
from torch_geometric.datasets import Twitch
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data, DataLoader
from datasets.transforms.denormalization import DeNormalize
import torchvision.transforms as transforms
from datasets.utils.federated_dataset import FederatedDataset, partition_office_domain_skew_loaders_new
from backbone.ResNet import resnet10, resnet12, resnet18, resnet34
from backbone.gnn.gcn import GCN_Net
from backbone.efficientnet import EfficientNetB0
from backbone.googlenet import GoogLeNet
from backbone.mobilnet_v2 import MobileNetV2
from torchvision.datasets import ImageFolder, DatasetFolder
from datasets.utils.splitter import RandomSplitter,split_train_test


class Twitch1(Twitch):
    def __init__(self, root: str, name: str,
                 transform = None,
                 pre_transform= None):
        super().__init__(root, name, transform, pre_transform)
        self.data_name = name
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = torch.from_numpy(data['features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)

        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.6)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        train_masks[train_node_indices] = 1
        train_masks = train_masks.bool()
        val_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        val_masks[val_node_indices] = 1
        val_masks = val_masks.bool()
        test_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        test_masks[test_node_indices] = 1
        test_masks = test_masks.bool()


        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class FedTwitch(FederatedDataset):
    NAME = 'fl_twitch'
    SETTING = 'domain_skew'
    DOMAINS_LIST =  ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
    domain_dict =  { 'EN':2, 'ES':2, 'FR':2, 'PT':2, 'RU': 2,'DE':20}
    N_CLASS = 2

    def get_data_loaders(self, selected_domain_list={}):

        # using_list = self.DOMAINS_LIST if selected_domain_list == [] else selected_domain_list
        using_list = selected_domain_list
        data_list = []


        train_dataset_list = []
        test_dataset_list = []

        for domain in selected_domain_list.keys():
            num = selected_domain_list[domain]
            splitter = RandomSplitter(num)
            train_dataset = Twitch1(root=os.path.join(data_path(),"Twitch"), name=domain)
            global_data = Twitch1(root=os.path.join(data_path(),"Twitch"), name=domain)
            global_data.data.data_name = global_data.data_name
            graphs, test_global_graph = splitter(train_dataset[0], global_data[0], percent=30)
            train_dataset_list += graphs
            test_dataset_list.append(test_global_graph)

        return train_dataset_list,test_dataset_list



    def get_backbone(self,parti_num, names_list):
        nets_dict = self.get_gnn_backbone_dict()
        nets_list = []
        if names_list == None:
            for j in range(parti_num):
                nets_list.append(GCN_Net(128,FedTwitch.N_CLASS))
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name](128,FedTwitch.N_CLASS,self.args.hidden))
        return nets_list



if __name__ == '__main__':
    test = FedTwitch()