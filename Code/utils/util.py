import os
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch
from scipy.sparse import csc_matrix
from scipy.stats import entropy
import copy
def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def save_networks(model, communication_idx):
    nets_list = model.nets_list
    model_name = model.NAME

    checkpoint_path = model.checkpoint_path
    model_path = os.path.join(checkpoint_path, model_name)
    model_para_path = os.path.join(model_path, 'para')
    create_if_not_exists(model_para_path)
    for net_idx, network in enumerate(nets_list):
        each_network_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(net_idx) + '.ckpt')
        torch.save(network.state_dict(), each_network_path)


def save_protos(model, communication_idx):
    model_name = model.NAME

    checkpoint_path = model.checkpoint_path
    model_path = os.path.join(checkpoint_path, model_name)
    model_para_path = os.path.join(model_path, 'protos')
    create_if_not_exists(model_para_path)

    for i in range(len(model.global_protos_all)):
        label = i
        protos = torch.cat(model.global_protos_all[i], dim=0).cpu().numpy()
        save_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(label) + '.npy')
        np.save(save_path, protos)

def edge_distribution_high(edge_idx, feats, tau):

    src = edge_idx[0]
    dst = edge_idx[1]

    feats_abs = torch.abs(feats[src] - feats[dst])
    e_softmax = F.log_softmax(feats_abs / tau, dim=-1)

    return e_softmax

def diff_loss(edge_idx,logits,out_t_all,tau):

    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    loss_s = criterion_t(edge_distribution_high(edge_idx, logits, tau), edge_distribution_high(edge_idx, out_t_all, tau))

    return loss_s

def centroids_loss(feat, centroids, global_centroids, temperature=0.2):
    global_centroids = torch.stack(list(global_centroids.values()), dim=0)
    cl_dim = feat.shape[0]
    feat_norm = torch.norm(feat, dim=-1)
    feat = torch.div(feat, feat_norm.unsqueeze(1))

    centroids_norm = torch.norm(centroids, dim=-1)
    centroids = torch.div(centroids, centroids_norm)

    global_centroids_norm = torch.norm(global_centroids, dim=-1)
    global_centroids = torch.div(global_centroids, global_centroids_norm.unsqueeze(1))
    logits_pos = torch.mm(feat, centroids.unsqueeze(1))
    logits_neg = torch.mm(feat, global_centroids.t())

    logits = torch.cat([logits_pos, logits_neg], dim=1)
    instance_labels = torch.zeros(cl_dim).long().to(feat.device)

    loss = F.cross_entropy(logits / temperature, instance_labels)

    return loss


def proto_align_loss(feat, feat_aug, temperature=0.3):
    cl_dim = feat.shape[0]

    feat_norm = torch.norm(feat, dim=-1)
    feat = torch.div(feat, feat_norm.unsqueeze(1))

    feat_aug_norm = torch.norm(feat_aug, dim=-1)
    feat_aug = torch.div(feat_aug, feat_aug_norm.unsqueeze(1))

    sim_clean = torch.mm(feat, feat.t())
    mask = (torch.ones_like(sim_clean) - torch.eye(cl_dim, device=sim_clean.device)).bool()
    sim_clean = sim_clean.masked_select(mask).view(cl_dim, -1)

    sim_aug = torch.mm(feat, feat_aug.t())
    sim_aug = sim_aug.masked_select(mask).view(cl_dim, -1)

    logits_pos = torch.bmm(feat.view(cl_dim, 1, -1), feat_aug.view(cl_dim, -1, 1)).squeeze(-1)
    logits_neg = torch.cat([sim_clean, sim_aug], dim=1)

    logits = torch.cat([logits_pos, logits_neg], dim=1)
    instance_labels = torch.zeros(cl_dim).long().to(sim_clean.device)

    loss = F.cross_entropy(logits / temperature, instance_labels)

    return loss


def get_stable_node(data, model):
    feats = data.x

    model.eval()
    data_teacher = model(data).detach().cpu().numpy()

    weight_t = []
    for i in range(data_teacher.shape[0]):
        weight_t.append(entropy(data_teacher[i]))
    weight_t = np.array(weight_t)

    feats_noise = copy.deepcopy(feats)
    feats_noise += torch.randn(feats.shape[0], feats.shape[1]).to(feats.device) * 1
    data2 = copy.deepcopy(data)
    data2.x = feats_noise
    out_noise = model(data2).detach().cpu().numpy()

    weight_s = np.zeros(feats.shape[0])
    for i in range(feats.shape[0]):
        weight_s[i] = np.abs(entropy(out_noise[i]) - weight_t[i])
    delta_entropy = weight_s / np.max(weight_s)
    KD_prob = 1 - delta_entropy

    mu = torch.bernoulli(torch.tensor(KD_prob)).bool()

    return mu

def dict_to_tensor(dic):
    lit = []
    for key,tensor in dic.items():
        lit.append(tensor[0])
    lit = torch.stack(lit)
    return lit


def soft_predict(Z,temp=0.3):
    m,n = Z.shape
    Q = torch.zeros(m,n)
    Z_sum = torch.sum(torch.exp(Z/temp),dim=1)
    for i in range(n):
        Q[:,i] = torch.exp(Z[:,i]/temp)/Z_sum
    return Q




def barlow_twins_loss(z1, z2, lambda_param=5e-3):
    # Normalize the representations along the batch dimension
    z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0) # BxD
    z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0) # BxD

    # Compute the correlation matrix for each representation
    corr_z1 = torch.mm(z1_norm.T, z1_norm) / z1_norm.size(0) # DxD
    corr_z2 = torch.mm(z2_norm.T, z2_norm) / z2_norm.size(0) # DxD

    # Compute the off-diagonal loss
    off_diag_loss = (corr_z1 * (1 - torch.eye(corr_z1.size(0), device=corr_z1.device))).pow(2).sum()
    off_diag_loss += (corr_z2 * (1 - torch.eye(corr_z2.size(0), device=corr_z2.device))).pow(2).sum()

    # Compute the on-diagonal loss
    on_diag_loss = (torch.diagonal(corr_z1) - torch.diagonal(corr_z2)).pow(2).sum()

    # Combine the two loss terms
    loss = on_diag_loss + lambda_param * off_diag_loss

    return loss
def softmax_entropy(x: torch.Tensor):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def remove_edge(A, similarity, remove_rate=0.1):
    """
    remove edge based on embedding similarity
    Args:
        A: the origin adjacency matrix
        similarity: cosine similarity matrix of embedding
        remove_rate: the rate of removing linkage relation
    Returns:
        Am: edge-masked adjacency matrix
    """
    # remove edges based on cosine similarity of embedding
    n_node = A.shape[0]
    for i in range(n_node):
        A[i, torch.argsort(similarity[i].cpu())[:int(round(remove_rate * n_node))]] = 0


def edge_index_to_adj_matrix(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    edge_index_t = edge_index.t()
    adj_matrix[edge_index_t[:, 0], edge_index_t[:, 1]] = 1
    return adj_matrix


def normalize_adj(edge_index, num_nodes, self_loop=True, symmetry=True):
    """
    normalize the adj matrix
    :param edge_index: input edge_index
    :param num_nodes: number of nodes in the graph
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # Convert edge_index to adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    edge_index = edge_index.t()
    adj_matrix[edge_index[0], edge_index[1]] = 1

    # add the self_loop
    if self_loop:
        adj_tmp = adj_matrix + torch.eye(num_nodes)
    else:
        adj_tmp = adj_matrix

    # calculate degree matrix and it's inverse matrix
    d = torch.diag(adj_tmp.sum(0))
    d_inv = torch.inverse(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = torch.sqrt(d_inv)
        norm_adj = torch.matmul(torch.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = torch.matmul(d_inv, adj_tmp)

    return norm_adj.to_sparse()


def diffusion_adj(edge_index, num_nodes, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param edge_index: input edge_index
    :param num_nodes: number of nodes in the graph
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # Convert edge_index to adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    edge_index = edge_index
    adj_matrix[edge_index[0], edge_index[1]] = 1

    # add the self_loop
    adj_tmp = adj_matrix + torch.eye(num_nodes)

    # calculate degree matrix and it's inverse matrix
    d = torch.diag(adj_tmp.sum(0))
    d_inv = torch.inverse(d)
    sqrt_d_inv = torch.sqrt(d_inv)

    # calculate norm adj
    norm_adj = torch.matmul(torch.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # calculate graph diffusion
    if mode == "ppr":
        diff_adj = transport_rate * torch.inverse((torch.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))


    return diff_adj.to_sparse()


def remove_edge(A, similarity, remove_rate=0.1):
    """
    remove edge based on embedding similarity
    Args:
        A: the origin adjacency matrix
        similarity: cosine similarity matrix of embedding
        remove_rate: the rate of removing linkage relation
    Returns:
        Am: edge-masked adjacency matrix
    """
    # remove edges based on cosine similarity of embedding
    n_node = A.shape[0]
    for i in range(n_node):
        A[i, torch.argsort(similarity[i].cpu())[:int(round(remove_rate * n_node))]] = 0

    # normalize adj
    Am = normalize_adj(A, self_loop=True, symmetry=True)
    Am = numpy_to_torch(Am)
    return Am


def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a

def get_subgraph(train_dataset):
    from torch_geometric.utils import subgraph
    import networkx as nx
    from torch_geometric.utils import to_networkx, from_networkx

    train_indices = np.array(torch.where(train_dataset[0].train_mask)[0])

    G = to_networkx(
        train_dataset[0],
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True)
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")
    sub_g =  nx.DiGraph(nx.subgraph(G, train_indices))
    mm = from_networkx(sub_g)
    return mm

def assign_mask(data,train_rate):
    y = data.y
    random_node_indices = np.random.permutation(y.shape[0])
    training_size = int(len(random_node_indices) * train_rate)
    val_size = int(len(random_node_indices) * 0.2)
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

    return data

def get_subgraph(data,index):
    from torch_geometric.utils import to_networkx, from_networkx
    import networkx as nx
    data.index_orig = torch.arange(data.num_nodes)
    G = to_networkx(
        data,
        node_attrs=['x', 'y'],
        to_undirected=True)
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")

    sub_g = nx.DiGraph(nx.subgraph(G, index))
    return from_networkx(sub_g)


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


def scipysp_to_pytorchsp(sp_mx):

    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()

    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape

    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T), torch.FloatTensor(values), torch.Size(shape))

    return pyt_sp_mx
