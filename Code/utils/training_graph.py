import copy

import scipy as sp
import torch
from argparse import Namespace

from scipy.sparse import csc_matrix

from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from utils.logger import CsvWriter
from collections import Counter
import scipy.sparse as sp
from utils.utils_acc import get_scores
from utils.utils_acc import plot_predict_heatmap
from utils.util import softmax_entropy,diffusion_adj,normalize_adj
# from utils.vis import visual_tsne
def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy

def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str,epoch_index) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
        graph = dl
        graph = graph.to(model.device)
        if model.NAME != 'fggp':
            if 'pmlp' in model.args.backbone:
                logits = net(graph,True)
            else:
                logits = net(graph)
            # out1 = model.nets_list[0](graph)
        else:
            if model.args.use_knn:
                feat = net.features(graph)
                model.eval_knn.memory = model.eval_knn.memory[:model.eval_knn.queue_size, :]  # drop last
                model.eval_knn.memory_label  = model.eval_knn.memory_label[:model.eval_knn.queue_size]
                logits = model.eval_knn(feat)
            else:
                logits = net(graph)
        preds = logits[graph.test_mask + graph.val_mask]
        labels = graph.y[graph.test_mask + graph.val_mask]
        acc, auc = get_scores(preds, labels, dl.data_name)
        accs.append(acc)
    net.train(status)
    return accs


def get_norm_and_orig(graph):
    num_nodes = graph.x.shape[0]
    edge_index = graph.edge_index
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    edge_index_t = edge_index.t()
    adj_matrix[edge_index_t[:, 0], edge_index_t[:, 1]] = 1
    adj_orig = adj_matrix.to_dense()


    edge_weight = np.array([1] * len(edge_index[0]))
    adj = csc_matrix((edge_weight, (edge_index[0], edge_index[1])),
                     shape=(num_nodes, num_nodes)).tolil()
    degrees = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
    adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt

    adj_norm = scipysp_to_pytorchsp(adj_norm).to_dense()

    graph.adj_orig = adj_orig
    graph.adj_norm = adj_norm
    return graph


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    if args.rand_dataset:
        max_num = 10
        is_ok = False

        while not is_ok:
            if model.args.dataset == 'fl_officecaltech':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_digits':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        selected_domain_dict = private_dataset.domain_dict
        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)

    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(result)
    # print("diffusing....")
    # for graph in pri_train_loaders:
    #     print("训练比例：")
    #     print(round((sum(graph.train_mask) / graph.x.shape[0]).item(), 2))
    #     print("最小标签")
    #     unique_elements, counts = np.unique(graph.y[graph.train_mask].cpu(), return_counts=True)
    #     min = np.argmin(counts)
    #     # 元素及其出现次数
    #     print(len(unique_elements))
    #     print("节点数量：" + str(graph.x.shape[0]))
    # print("diffused")
    model.trainloaders = pri_train_loaders
    model.other_view = [copy.deepcopy(m) for m in pri_train_loaders]
    # m = diffusion_adj(pri_train_loaders[0].edge_index, pri_train_loaders[0].num_nodes, mode="ppr", transport_rate=0.2)
    model.testlodaers = test_loaders
    if hasattr(model, 'ini'):
        model.ini()

    accs_dict = {}
    mean_accs_list = []

    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(pri_train_loaders)

        accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME, epoch_index)
        accs = [acc.item() for acc in accs]
        mean_acc = round(np.mean(accs,axis=0),4)
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]
            if epoch_index >= Epoch - 5:
                print('-' * 80)
                print('The ' + str(epoch_index) + ' Communication Accuracy:', str(mean_acc), 'Method:',
                      model.args.model)
                print('-' * 80)


    final_mean_acc = round(np.mean(mean_accs_list[-5:]), 4)
    print('-' * 80)
    print('Final Communication Accuracy (last 5 epochs):', str(final_mean_acc), 'Method:', model.args.model)
    print('-' * 80)

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)


def scipysp_to_pytorchsp(sp_mx):

    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()

    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape

    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T), torch.FloatTensor(values), torch.Size(shape))

    return pyt_sp_mx