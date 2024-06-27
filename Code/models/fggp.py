import itertools

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from scipy.sparse import coo_matrix

from backbone.gnn.mlp import Linear
from backbone.knn import MomentumQueue
from utils.args import *
import numpy as np
import torch.nn.functional as F
from utils.finch import FINCH
import torch_geometric
from models.utils.federated_model import FederatedModel
from utils.utils_acc import get_scores
from sklearn.neighbors import kneighbors_graph
from utils.util import diff_loss, proto_align_loss, get_stable_node, dict_to_tensor, soft_predict, \
    edge_index_to_adj_matrix, get_norm_and_orig


def get_proto_norm_weighted(num_classes, embedding, class_label, weight,unique_labels):
    m1= F.one_hot(class_label, num_classes=num_classes)
    m2 = (m1 * weight[:, None]).t()
    m = m2 / (m2.sum(dim=1, keepdim=True)+ 1e-6)
    m = m[unique_labels]
    return torch.mm(m, embedding)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def com_distillation_loss(t_logits, s_logits, adj_orig, adj_sampled, temp, loss_mode):

    s_dist = F.log_softmax(s_logits / temp, dim=-1)
    t_dist = F.softmax(t_logits / temp, dim=-1)
    if loss_mode == 0:
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist)
    elif loss_mode == 1:
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())

    adj = torch.triu(adj_orig * adj_sampled).detach()
    edge_list = (adj + adj.T).nonzero().t()

    s_dist_neigh = F.log_softmax(s_logits[edge_list[0]] / temp, dim=-1)
    t_dist_neigh = F.softmax(t_logits[edge_list[1]] / temp, dim=-1)
    if loss_mode == 0:
        kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh)
    elif loss_mode == 1:
        kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh.detach())

    return kd_loss

class fggp(FederatedModel):
    NAME = 'fggp'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(fggp, self).__init__(nets_list,args,transform)
        self.global_centroids = []
        self.local_centroids = {}
        self.global_protos = []
        self.local_protos = {}
        self.local_protos_ema = {}
        self.infoNCET = args.infoNCET
        self.eval = args.size

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        self.personal_project = [Linear(self.global_net.hidden_channels, self.global_net.hidden_channels, 0.5, bias=True) for _ in range(len(self.nets_list))]
        self.eval_knn = MomentumQueue(self.global_net.hidden_channels,
                                           self.N_CLASS * self.eval, 0.1, self.args.knn,
                                           self.N_CLASS).to(self.device)
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)


    def proto_aggregation(self, local_protos_list):
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
                proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)

                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                    agg_selected_proto.append(torch.tensor(proto))
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = [proto_list[0].data]

        for num, each_class_proto in agg_protos_label.items():
            if len(each_class_proto) == 1:
                proto = each_class_proto[0].to(self.device)
            else:
                proto = torch.cat(each_class_proto, dim=0).to(self.device)
            y_hat = torch.ones(proto.shape[0]).to(self.device) * num
            self.eval_knn.update_queue(proto, y_hat)
        return agg_protos_label



    def aggregate_nets(self, freq=None, personal='lkl'):
        global_net = self.global_net
        nets_list = self.nets_list

        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[online_clients_index] for online_clients_index in online_clients]
            online_clients_len = []
            for dl in online_clients_dl:
                if isinstance(dl, torch_geometric.data.Data):
                    # 判断是否是图数据集
                    online_clients_len.append(dl.num_nodes)
                else:
                    online_clients_len.append(dl.sampler.indices.size)
                    # online_clients_len = [dl.sampler.indices.size if isinstance(dl, torch_geometric.data.Data) eles '2' for dl in online_clients_dl]
            online_clients_all = np.sum(online_clients_len)
            freq = online_clients_len / online_clients_all
        else:
            # if freq == None:
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]

        first = True
        for index, net_id in enumerate(online_clients):
            net = nets_list[net_id]
            net_para = net.state_dict()
            # if net_id == 0:
            if first:
                first = False
                for key in net_para:
                    # 检查网络层的名称是否包含特定名称
                    if personal not in key:
                        global_w[key] = net_para[key] * freq[index]
            else:
                for key in net_para:
                    # 检查网络层的名称是否包含特定名称
                    if personal not in key:
                        global_w[key] += net_para[key] * freq[index]

        global_net.load_state_dict(global_w)

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

    def loc_update(self,priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        # self.global_centroids = self.proto_aggregation(self.local_centroids)
        self.global_protos = self.proto_aggregation(self.local_protos)
        self.aggregate_nets(None,self.args.personal)

        return  None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        train_loader = train_loader.to(self.device)
        net.train()
        proj_head = self.personal_project[index].to(self.device)
        params = itertools.chain(*[net.parameters()])
        optimizer = optim.SGD(params, lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = F.nll_loss
        self.global_net = self.global_net.to(self.device)
        # us = get_stable_node(train_loader,self.global_net)
        train_loader2 = self.other_view[index]
        if self.epoch_index % self.args.knn_frequence == 0:
            with torch.no_grad():
                global_feature = self.global_net.features(train_loader)
                adj = kneighbors_graph(global_feature.cpu(), self.args.neibor, metric='cosine')
                del global_feature
                adj.setdiag(1)
                coo = adj.tocoo()
                train_loader.global_edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long).to(self.device)
                del coo
                del adj
                combined_edge_index = torch.cat([train_loader.edge_index, train_loader.global_edge_index], dim=1)
                # 将 edge_index 转换为元组集合，删除重复的边
                # combined_edge_index = torch.cat([train_loader.edge_index, train_loader.edge_index], dim=1)
                edge_set = set(zip(combined_edge_index[0].tolist(), combined_edge_index[1].tolist()))
                # 将结果转换回 edge_index 的格式
                union_edge_index = torch.tensor([[i[0] for i in edge_set], [i[1] for i in edge_set]], dtype=torch.long)
                train_loader2.edge_index = union_edge_index
                train_loader2 = get_norm_and_orig(train_loader2)
                adj_orig = train_loader2.adj_orig
                norm_w = adj_orig.shape[0] ** 2 / float((adj_orig.shape[0] ** 2 - adj_orig.sum()) * 2)
                pos_weight = torch.FloatTensor([float(adj_orig.shape[0] ** 2 - adj_orig.sum()) / adj_orig.sum()])
                train_loader2.norm_w = norm_w
                train_loader2.pos_weight = pos_weight
                train_loader2 = train_loader2.to(self.device)

        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = self.global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())
                mean_f.append(torch.mean(temp_f, dim=0).cpu())
            all_f = [item.detach() for item in all_f]
            mean_f = [item.detach() for item in mean_f]

        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:
            # out = net.(train_loader)
            # out1 = net.conv(train_loader)
            # loss1 = criterion(out1[train_loader.train_mask], train_loader.y[train_loader.train_mask])
            #
            # out2 = net.mlp(train_loader)
            # loss2 = criterion(out2[train_loader.train_mask], train_loader.y[train_loader.train_mask])

            # out3 = net.att_model([out1,out2])
            # loss3 = criterion(out3[train_loader.train_mask], train_loader.y[train_loader.train_mask])

            out4 = net(train_loader)
            # out5 = net(train_loader2)
            adj_sampled, adj_logits = net.aug(train_loader2)
            train_loader2.adj = adj_sampled
            out5 = net(train_loader2, adj=True)
            ga_loss = train_loader2.norm_w * F.binary_cross_entropy_with_logits(adj_logits, train_loader2.adj_orig, pos_weight=train_loader2.pos_weight)
            kd_loss = com_distillation_loss(out4, out5, train_loader2.adj_orig, adj_sampled, 0.1, 0)
            lossCE = criterion(out4[train_loader.train_mask], train_loader.y[train_loader.train_mask])
            lossCE2 = criterion(out5[train_loader2.train_mask], train_loader2.y[train_loader2.train_mask])

            feat = net.features(train_loader)
            feat_global = net.features(train_loader2,adj=True)

            output_exp = torch.exp(out4)
            confidences = output_exp.max(1)[0]
            pseudo_labels = output_exp.max(1)[1].type_as(train_loader.y)
            pseudo_labels[train_loader.train_mask] = train_loader.y[train_loader.train_mask]
            confidences[train_loader.train_mask] = 1.0
            unique_labels = torch.unique(pseudo_labels)
            proto = get_proto_norm_weighted(self.N_CLASS, feat, pseudo_labels, confidences,unique_labels)
            proto_global = get_proto_norm_weighted(self.N_CLASS, feat_global, pseudo_labels, confidences,unique_labels)

            loss_pa = proto_align_loss(proto_global, proto,temperature=0.5)

            if len(self.global_protos) == 0:
                loss_InfoNCE = 0 * lossCE
                loss_proto = 0 * lossCE
            else:
                pass

            loss =  lossCE + lossCE2 + loss_pa + ga_loss

            optimizer.zero_grad()
            loss.backward()
            iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
            optimizer.step()


            if iter == self.local_epoch - 1:
                output_exp = torch.exp(out4)
                feat = proj_head(net.features(train_loader))
                feat = net.features(train_loader)
                confidences = output_exp.max(1)[0]
                pseudo_labels = output_exp.max(1)[1].type_as(train_loader.y)
                pseudo_labels[train_loader.train_mask] = train_loader.y[train_loader.train_mask]
                confidences[train_loader.train_mask] = 1.0
                unique_labels = torch.unique(pseudo_labels)
                proto = get_proto_norm_weighted(self.N_CLASS, feat, pseudo_labels, confidences,unique_labels)
                tensor_dict = {i: proto[i].data for i in range(proto.shape[0])}

        self.local_protos[index] = tensor_dict

