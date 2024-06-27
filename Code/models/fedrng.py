import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from scipy.sparse import coo_matrix
from utils.args import *
import numpy as np
import torch.nn.functional as F
from utils.finch import FINCH
from models.utils.federated_model import FederatedModel
from utils.utils_acc import get_scores
import copy
from sklearn.neighbors import kneighbors_graph
from utils.util import diff_loss,centroids_loss
def get_proto_norm_weighted(num_classes, embedding, class_label, weight):
    m = F.one_hot(class_label, num_classes=num_classes)
    m = (m * weight[:, None]).t()
    m = m / m.sum(dim=1, keepdim=True)
    return torch.mm(m, embedding)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class Fedrng(FederatedModel):
    NAME = 'fedrng'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(Fedrng, self).__init__(nets_list,args,transform)
        self.global_centroids = []
        self.local_centroids = {}
        self.global_protos = []
        self.local_protos = {}
        self.local_protos_ema = {}
        self.infoNCET = args.infoNCET

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    # def proto_aggregation(self, local_protos_list):
    #     agg_protos_label = dict()
    #     for idx in self.online_clients:
    #         local_protos = local_protos_list[idx]
    #         for label in local_protos.keys():
    #             if label in agg_protos_label:
    #                 agg_protos_label[label].append(local_protos[label])
    #             else:
    #                 agg_protos_label[label] = [local_protos[label]]
    #     for [label, proto_list] in agg_protos_label.items():
    #         if len(proto_list) > 1:
    #             proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
    #             proto_list = np.array(proto_list)
    #
    #             c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
    #                                         ensure_early_exit=False, verbose=True)
    #
    #             m, n = c.shape
    #             class_cluster_list = []
    #             for index in range(m):
    #                 class_cluster_list.append(c[index, -1])
    #
    #             class_cluster_array = np.array(class_cluster_list)
    #             uniqure_cluster = np.unique(class_cluster_array).tolist()
    #             agg_selected_proto = []
    #
    #             for _, cluster_index in enumerate(uniqure_cluster):
    #                 selected_array = np.where(class_cluster_array == cluster_index)
    #                 selected_proto_list = proto_list[selected_array]
    #                 proto = np.mean(selected_proto_list, axis=0, keepdims=True)
    #
    #                 agg_selected_proto.append(torch.tensor(proto))
    #             agg_protos_label[label] = agg_selected_proto
    #         else:
    #             agg_protos_label[label] = [proto_list[0].data]
    #
    #     return agg_protos_label

    def con_info_loss(self, f, label, all_f, all_global_protos_keys):
        f_pos = np.array(all_f, dtype=object)[all_global_protos_keys == label.item()][0].to(self.device)

        f_neg = torch.cat(list(np.array(all_f, dtype=object)[all_global_protos_keys != label.item()])).to(
            self.device)

        f_now = f

        embedding_len = f_pos.shape
        f_neg = f_neg.unsqueeze(1).view(-1, embedding_len[0])
        f_pos = f_pos.view(-1, embedding_len[0])
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        loss_instance = -torch.log(sum_pos_l / sum_exp_l)

        return loss_instance
    def proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for idx in range(self.args.parti_num):
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

    def hierarchical_info_loss(self, f_now, label, all_f, mean_f, all_global_protos_keys):
        f_pos = np.array(all_f,dtype=object)[all_global_protos_keys == label.item()][0].to(self.device)
        f_neg = torch.cat(list(np.array(all_f,dtype=object)[all_global_protos_keys != label.item()])).to(self.device)
        xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)

        mean_f_pos = np.array(mean_f,dtype=object)[all_global_protos_keys == label.item()][0].to(self.device)
        mean_f_pos = mean_f_pos.view(1, -1)
        loss_mse = nn.MSELoss()
        cu_info_loss = loss_mse(f_now, mean_f_pos)

        hierar_info_loss = xi_info_loss + cu_info_loss
        return hierar_info_loss

    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.infoNCET

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss


    def loc_update(self,priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        # self.global_centroids = self.proto_aggregation(self.local_centroids)
        self.global_protos = self.proto_aggregation(self.local_protos)
        self.aggregate_nets(None)

        return  None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        train_loader = train_loader.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        Epoch = self.args.communication_epoch - 1
        alpha = 1 - self.epoch_index / Epoch
        self.global_net = self.global_net.to(self.device)
        # with torch.no_grad():
        #     global_feature = 0
        #     if self.epoch_index == 0:
        #         global_feature = self.global_net.features(train_loader)
        #     else:
        #         global_feature = global_feature * self.args.ema + (1-self.args.ema) * self.global_net.features(train_loader)
        # adj = kneighbors_graph(global_feature.cpu(), 5, metric='cosine')
        # adj.setdiag(1)
        # coo = adj.tocoo()
        # # 生成 edge_index
        # global_edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
        # # global_edge_index = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
        # train_loader2 = copy.deepcopy(train_loader)
        # train_loader2.edge_index = global_edge_index
        # train_loader2 = train_loader2.to(self.device)
        out_global= self.global_net(train_loader)
        output_exp = torch.exp(out_global)
        confidences = output_exp.max(1)[0]
        pseudo_labels = output_exp.max(1)[1].type_as(train_loader.y)
        pseudo_labels[train_loader.train_mask] = train_loader.y[train_loader.train_mask]
        confidences[train_loader.train_mask] = 1.0

        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            all_f = []
            for protos_key in all_global_protos_keys:
                temp_f = self.global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())


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
            # out5 = net(train_loader2)
            # loss5 = criterion(out5[train_loader2.train_mask], train_loader2.y[train_loader2.train_mask])

            out4 = net(train_loader)
            lossCE = criterion(out4[train_loader.train_mask], train_loader.y[train_loader.train_mask])
            feat = net.features(train_loader)
            centroid = torch.mean(net.project(feat),dim=0)

            if len(self.global_protos) == 0:
                loss_InfoNCE = 0 * lossCE
                loss_centroid = 0 * lossCE
            else:
                proto = get_proto_norm_weighted(self.N_CLASS, feat, pseudo_labels, confidences)

                # loss_centroid = centroids_loss(proto, self.local_centroids[index], self.local_centroids)
                i = 0
                loss_InfoNCE = None
                for label in train_loader.y[train_loader.train_mask]:
                    if label.item() in self.global_protos.keys():
                        f_now = feat[train_loader.train_mask][i].unsqueeze(0)
                        loss_instance = self.con_info_loss(f_now, label, all_f, all_global_protos_keys)
                        if loss_InfoNCE is None:
                            loss_InfoNCE = loss_instance
                        else:
                            loss_InfoNCE += loss_instance
                    i += 1
                loss_InfoNCE = loss_InfoNCE / i

            # loss =  alpha * loss_InfoNCE + (1-alpha)* lossCE
            loss = lossCE + loss_centroid


            # loss = loss2
            optimizer.zero_grad()
            loss.backward()
            iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
            optimizer.step()

            if iter == self.local_epoch - 1:
                output_exp = torch.exp(out_global)
                confidences = output_exp.max(1)[0]
                pseudo_labels = output_exp.max(1)[1].type_as(train_loader.y)
                pseudo_labels[train_loader.train_mask] = train_loader.y[train_loader.train_mask]
                confidences[train_loader.train_mask] = 1.0
                proto = get_proto_norm_weighted(self.N_CLASS, feat, pseudo_labels, confidences)
                if index in self.local_protos_ema.keys():
                    proto_old = self.local_protos_ema[index] * self.args.ema + (1-self.args.ema) * proto
                else:
                    proto_old = proto
                self.local_protos_ema[index] = proto_old
                tensor_dict = {i: proto_old[i] for i in range(proto_old.shape[0])}

        self.local_protos[index] = tensor_dict
        self.local_centroids[index] = copy.deepcopy(centroid.detach().data)

