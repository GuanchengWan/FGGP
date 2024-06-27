import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.utils import add_self_loops, remove_self_loops, \
    to_undirected
from torch_geometric.utils import dropout_adj, index_to_mask
import numpy as np
import networkx as nx

import community as community_louvain

EPSILON = 1e-5

def split_train_test(data, train_rate, test_rate):
    random_node_indices = np.random.permutation(data.y.shape[0])
    training_size = int(len(random_node_indices) * 0.1)
    val_size = int(len(random_node_indices) * 0.1)
    train_node_indices = random_node_indices[:training_size]
    val_node_indices = random_node_indices[training_size:training_size + val_size]
    test_node_indices = random_node_indices[training_size + val_size:]

    train_masks = torch.zeros([data.y.shape[0]], dtype=torch.uint8)
    train_masks[train_node_indices] = 1
    train_masks = train_masks.bool()
    val_masks = torch.zeros([data.y.shape[0]], dtype=torch.uint8)
    val_masks[val_node_indices] = 1
    val_masks = val_masks.bool()
    test_masks = torch.zeros([data.y.shape[0]], dtype=torch.uint8)
    test_masks[test_node_indices] = 1
    test_masks = test_masks.bool()

    data.train_mask = train_masks
    data.val_mask = val_masks
    data.test_mask = test_masks
    return data
class RandomSplitter(BaseTransform):
    """
    Split Data into small data via random sampling.

    Args:
        client_num (int): Split data into client_num of pieces.
        sampling_rate (str): Samples of the unique nodes for each client, \
            eg. ``'0.2,0.2,0.2'``
        overlapping_rate(float): Additional samples of overlapping data, \
            eg. ``'0.4'``
        drop_edge(float): Drop edges (drop_edge / client_num) for each \
            client within overlapping part.
    """
    def __init__(self,
                 client_num,
                 sampling_rate=None,
                 overlapping_rate=0,
                 drop_edge=0,missing_link = 0,percent=30):
        self.client_num = percent
        self.client_num_need = client_num
        self.ovlap = overlapping_rate
        if sampling_rate is not None:
            self.sampling_rate = np.array(
                [float(val) for val in sampling_rate.split(',')])
        else:
            # Default: Average
            self.sampling_rate = (np.ones(self.client_num) -
                                  self.ovlap) / self.client_num

        if len(self.sampling_rate) != self.client_num:
            raise ValueError(
                f'The client_num ({self.client_num}) should be equal to the '
                f'lenghth of sampling_rate and overlapping_rate.')

        if abs((sum(self.sampling_rate) + self.ovlap) - 1) > EPSILON:
            raise ValueError(
                f'The sum of sampling_rate:{self.sampling_rate} and '
                f'overlapping_rate({self.ovlap}) should be 1.')

        self.drop_edge = drop_edge
        self.missing_link = missing_link

    def __call__(self, data, global_dataset, **kwargs):
        if self.missing_link > 0:
            print(data.num_edges)
            data.edge_index, _ = dropout_adj(data.edge_index, p=self.missing_link, force_undirected=True,
                                                   num_nodes=data.num_nodes)
            print(data.num_edges)
        data.index_orig = torch.arange(data.num_nodes)
        G = to_networkx(
            data,
            node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
            to_undirected=True)
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")

        client_node_idx = {idx: [] for idx in range(self.client_num)}

        indices = data.index_orig.cpu().numpy()
        # indices = np.random.permutation(data.num_nodes)
        sum_rate = 0
        for idx, rate in enumerate(self.sampling_rate):
            client_node_idx[idx] = indices[round(sum_rate *
                                                 data.num_nodes):round(
                                                     (sum_rate + rate) *
                                                     data.num_nodes)]
            sum_rate += rate

        if self.ovlap:
            ovlap_nodes = indices[round(sum_rate * data.num_nodes):]
            for idx in client_node_idx:
                client_node_idx[idx] = np.concatenate(
                    (client_node_idx[idx], ovlap_nodes))

        # Drop_edge index for each client
        if self.drop_edge:
            ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
            ovlap_edge_ind = np.random.permutation(
                ovlap_graph.number_of_edges())
            drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                             self.drop_edge)]
            drop_client = [
                drop_all[s:s + round(len(drop_all) / self.client_num)]
                for s in range(0, len(drop_all),
                               round(len(drop_all) / self.client_num))
            ]

        graphs = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            sub_g = nx.DiGraph(nx.subgraph(G, nodes))
            if self.drop_edge:
                sub_g.remove_edges_from(
                    np.array(ovlap_graph.edges)[drop_client[owner]])
            graphs.append(from_networkx(sub_g))

        dataset = [ds for ds in graphs]
        client_num = min(len(dataset), self.client_num
                         ) if self.client_num > 0 else len(dataset)


        graphs = []
        for client_idx in range(1, len(dataset) + 1):
            local_data = dataset[client_idx - 1]
            # To undirected and add self-loop
            local_data.edge_index = add_self_loops(
                to_undirected(remove_self_loops(local_data.edge_index)[0]),
                num_nodes=local_data.x.shape[0])[0]
            graphs.append(local_data)
        if global_dataset is not None:
            global_graph = global_dataset
            train_mask = torch.zeros_like(global_graph.train_mask)
            val_mask = torch.zeros_like(global_graph.val_mask)
            test_mask = torch.zeros_like(global_graph.test_mask)

            for client_subgraph in graphs:
                train_mask[client_subgraph.index_orig[
                    client_subgraph.train_mask]] = True
                val_mask[client_subgraph.index_orig[
                    client_subgraph.val_mask]] = True
                test_mask[client_subgraph.index_orig[
                    client_subgraph.test_mask]] = True
            global_graph.train_mask = train_mask
            global_graph.val_mask = val_mask
            global_graph.test_mask = test_mask
        global_dataset = global_graph
        graphs = graphs[:self.client_num_need]
        return graphs,global_dataset



class LouvainSplitter():
    """
    Split Data into small data via louvain algorithm.

    Args:
        client_num (int): Split data into ``client_num`` of pieces.
        delta (int): The gap between the number of nodes on each client.
    """
    def __init__(self, client_num, delta=20):
        self.delta = delta
        self.client_num = client_num

    def __call__(self, data, **kwargs):
        data.index_orig = torch.arange(data.num_nodes)
        G = to_networkx(
            data,
            node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
            to_undirected=True)
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")
        partition = community_louvain.best_partition(G)

        cluster2node = {}
        for node in partition:
            cluster = partition[node]
            if cluster not in cluster2node:
                cluster2node[cluster] = [node]
            else:
                cluster2node[cluster].append(node)

        max_len = len(G) // self.client_num - self.delta
        max_len_client = len(G) // self.client_num

        tmp_cluster2node = {}
        for cluster in cluster2node:
            while len(cluster2node[cluster]) > max_len:
                tmp_cluster = cluster2node[cluster][:max_len]
                tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
                                 1] = tmp_cluster
                cluster2node[cluster] = cluster2node[cluster][max_len:]
        cluster2node.update(tmp_cluster2node)

        orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
        orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

        client_node_idx = {idx: [] for idx in range(self.client_num)}
        idx = 0
        for (cluster, node_list) in orderedc2n:
            while len(node_list) + len(
                    client_node_idx[idx]) > max_len_client + self.delta:
                idx = (idx + 1) % self.client_num
            client_node_idx[idx] += node_list
            idx = (idx + 1) % self.client_num

        graphs = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            graphs.append(from_networkx(nx.subgraph(G, nodes)))

        graphs = graphs[:self.client_num_need]
        return graphs