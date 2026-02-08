import os

import os.path as osp
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch
import torch.nn as nn
from torch_geometric.datasets import CoraFull
# import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import NormalizeFeatures as T
from collections import Counter, defaultdict
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict


def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))
       
    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

import numpy as np
import scipy.sparse as sp
import torch


def load_data(dataset_str,n_tasks):
    """Load data."""
    
    if dataset_str in ['Arxiv','Products','CoraFull']:

        if dataset_str == 'Arxiv':
            dataset = PygNodePropPredDataset(root='data/', name='ogbn-arxiv', transform=T()).shuffle()
        elif dataset_str == 'Products':
            dataset = PygNodePropPredDataset(root='data/', name='ogbn-products', transform=T()).shuffle()
        elif dataset_str == 'CoraFull':
            path = 'data/CoraFull'
            dataset = CoraFull(path, transform=T()).shuffle()
        data = dataset[0]

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        adj = adj.tocsr()  # Convert to CSR format
        features = data.x.numpy()
        labels = data.y.squeeze().numpy()  # 确保标签是一维数组


        # Filter out classes with fewer than 150 instances
        class_counts = np.bincount(labels)
        if dataset_str == 'Arxiv':
            valid_classes = np.argsort(-class_counts)[:38]  # 取基数最多的前45个类别
            valid_classes = valid_classes[28:36]  # 取其中的第30到44个类别
        elif dataset_str == 'Products':
            valid_classes = np.argsort(-class_counts)[:46]  # 取基数最多的前45个类别
            valid_classes = valid_classes[28:44]  # 取其中的第30到44个类别
        elif dataset_str == 'CoraFull':
            valid_classes = np.argsort(-class_counts)[:43]  # 取基数最多的前45个类别
            valid_classes = valid_classes[35:43]  # 取其中的第30到44个类别


        label_map = defaultdict(lambda: -1)  # 默认值为 -1
        for new_label, old_label in enumerate(valid_classes):
            label_map[old_label] = new_label

        labels = np.array([label_map[label] for label in labels])

        # valid_indices = np.in1d(labels, np.arange(len(valid_classes)))
        valid_indices = np.where(labels != -1)[0]  # 过滤无效标签
        adj = adj[valid_indices, :][:, valid_indices]
        features = features[valid_indices]
        labels = labels[valid_indices]
        num_classes = len(valid_classes)
        print("num_classes", num_classes)
        # 获取前n_tasks个任务的所有训练数据索引

        return adj, features, num_classes,  None, None, None,None
    
    else:


        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_str)
        dataset = Planetoid(path, dataset_str, transform=T()).shuffle()
        data = dataset[0]

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        adj = adj.tocsr()  # Convert to CSR format
        features = data.x.numpy()
        labels = data.y.squeeze().numpy()  # 确保标签是一维数组
        num_classes = dataset.num_classes
        class_per_task = num_classes // n_tasks
        leftover_classes = num_classes % n_tasks

        return adj, features, num_classes, None, None, None,None
        
def load_data_class(dataset_str, task_no, n_tasks, Joint_Training=0):
    """Load data with an option for class incremental setting."""

    if dataset_str in ['Arxiv','Products','CoraFull']:

        if dataset_str == 'Arxiv':
            dataset = PygNodePropPredDataset(root='data/', name='ogbn-arxiv', transform=T()).shuffle()
        elif dataset_str == 'Products':
            dataset = PygNodePropPredDataset(root='data/', name='ogbn-products', transform=T()).shuffle()
        elif dataset_str == 'CoraFull':
            path = 'data/CoraFull'
            dataset = CoraFull(path, transform=T()).shuffle()

        data = dataset[0]

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        adj = adj.tocsr()  # Convert to CSR format
        features = data.x.numpy()
        labels = data.y.squeeze().numpy()  # 确保标签是一维数组

        # Filter out classes with fewer than 150 instances
        class_counts = np.bincount(labels)
        if dataset_str == 'Arxiv':
            valid_classes = np.argsort(-class_counts)[:38]  # 取基数最多的前45个类别
            valid_classes = valid_classes[26:36]  # 取其中的第30到44个类别
        elif dataset_str == 'Products':
            valid_classes = np.argsort(-class_counts)[:46]  # 取基数最多的前45个类别
            valid_classes = valid_classes[28:44]  # 取其中的第30到44个类别
        elif dataset_str == 'CoraFull':
            valid_classes = np.argsort(-class_counts)[:43]  # 取基数最多的前45个类别
            valid_classes = valid_classes[35:43]  # 取其中的第30到44个类别

        # 重新编码标签,使其从0开始
        label_map = defaultdict(lambda: -1)  # 默认值为 -1
        for new_label, old_label in enumerate(valid_classes):
            label_map[old_label] = new_label
        labels = np.array([label_map[label] for label in labels])

        # valid_indices = np.in1d(labels, np.arange(len(valid_classes)))
        valid_indices = np.where(labels != -1)[0]  # 过滤无效标签

        adj = adj[valid_indices, :][:, valid_indices]
        features = features[valid_indices]
        labels = labels[valid_indices]
        num_classes = len(valid_classes)
        print("num_classes", num_classes)

        if Joint_Training==1:
            train_indices_all = []
            for i in range(1, task_no):
                start_class = (i - 1) * (num_classes // n_tasks)
                end_class = start_class + (num_classes // n_tasks)
                class_indices = np.where((labels >= start_class) & (labels < end_class))[0]
                # 分割数据为训练集、验证集和测试集
                idx_train, idx_remaining, label_train, label_remaining = train_test_split(
                    class_indices,
                    labels[class_indices],
                    test_size=0.2,
                    stratify=labels[class_indices],
                    random_state=39
                )
                train_indices_all.extend(idx_train)

            # 创建训练数据集，包含前n_tasks个任务的所有训练数据
            idx_train_all = np.array(train_indices_all).astype(int)  # 将索引数组转换为整数类型

        # 计算当前任务的类别范围
        start_class = (task_no - 1) * (num_classes // n_tasks)
        end_class = start_class + (num_classes // n_tasks)
        class_indices = np.where((labels >= start_class) & (labels < end_class))[0]
        
        if len(class_indices) == 0:
            print(f"Task {task_no}: No samples found. Skipping task.")
        
        # 分割数据为训练集、验证集和测试集
        idx_train, idx_remaining, label_train, label_remaining = train_test_split(
            class_indices,
            labels[class_indices],
            test_size=0.5,
            stratify=labels[class_indices],
            random_state=39
        )

        idx_valid, idx_test, label_valid, label_test = train_test_split(
            idx_remaining,
            label_remaining,
            test_size=0.6,
            stratify=label_remaining,
            random_state=39
        )

        if Joint_Training == 1:
            idx_train = np.concatenate((idx_train, idx_train_all))
            label_train = np.concatenate((label_train, labels[idx_train_all]))  # 使用原始标签数组


        idx_train = torch.LongTensor(idx_train)
        idx_valid = torch.LongTensor(idx_valid)
        idx_test = torch.LongTensor(idx_test)
        label_train = torch.LongTensor(label_train)
        label_valid = torch.LongTensor(label_valid)
        label_test = torch.LongTensor(label_test)

        # return features,adj, idx_train, idx_valid, idx_test, label_train, label_valid, label_test
        return idx_train, idx_valid, idx_test, label_train, label_valid, label_test
    else:

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_str)
        dataset = Planetoid(path, dataset_str, transform=T()).shuffle()
        data = dataset[0]

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        adj = adj.tocsr()  # Convert to CSR format
        features = data.x.numpy()
        labels = data.y.squeeze().numpy()  # 确保标签是一维数组
        num_classes = dataset.num_classes
        class_per_task = num_classes // n_tasks
        leftover_classes = num_classes % n_tasks

        if Joint_Training==1:
            # 获取前n_tasks个任务的所有训练数据索引
            train_indices_all = []
            for i in range(1,task_no):
                start_class = i * class_per_task + min(i, leftover_classes)
                end_class = start_class + class_per_task + (1 if i <= leftover_classes else 0)
                class_indices = np.where((labels >= start_class) & (labels < end_class))[0]
                # class_indices_train = np.intersect1d(class_indices_train, np.where(np.argmax(labels, axis=1) < end_class)[0])
                idx_train, idx_remaining, label_train, label_remaining = train_test_split(
                    class_indices,
                    labels[class_indices],
                    test_size=0.2,
                    stratify=labels[class_indices],
                    random_state=39
                )
                train_indices_all.extend(idx_train)
            # 创建训练数据集，包含前n_tasks个任务的所有训练数据
            idx_train_all = np.array(train_indices_all).astype(int)  # 将索引数组转换为整数类型

        # Handle the case where classes cannot be evenly divided by tasks
        start_class = (task_no - 1) * class_per_task + min((task_no - 1), leftover_classes)
        end_class = start_class + class_per_task + (1 if task_no <= leftover_classes else 0)
        class_indices = np.where((labels >= start_class) & (labels < end_class))[0]
        # class_indices = np.intersect1d(class_indices, np.where(labels < end_class)[0])

        # Split data into train, validation, and test sets
        idx_train, idx_remaining, label_train, label_remaining = train_test_split(
            class_indices,
            labels[class_indices],
            test_size=0.9,
            stratify=labels[class_indices],
            random_state=39
        )

        idx_valid, idx_test, label_valid, label_test = train_test_split(
            idx_remaining,
            label_remaining,
            test_size=0.66,
            stratify=label_remaining,
            random_state=39
        )
        if Joint_Training == 1:
            idx_train = np.concatenate((idx_train, idx_train_all))
            label_train = np.concatenate((label_train, labels[idx_train_all]))  # 使用原始标签数组

        # # 将前n_tasks个任务的测试数据添加到当前任务的测试数据中
        # idx_valid = np.concatenate((idx_valid, idx_val_all))
        # label_valid = np.concatenate((label_valid, labels[idx_val_all]))  # 使用原始标签数组
        # idx_test = np.concatenate((idx_test, idx_test_all))
        # label_test = np.concatenate((label_test, labels[idx_test_all]))  # 使用原始标签数组

        idx_train = torch.LongTensor(idx_train)
        idx_valid = torch.LongTensor(idx_valid)
        idx_test = torch.LongTensor(idx_test)
        label_train = torch.LongTensor(label_train)
        label_valid = torch.LongTensor(label_valid)
        label_test = torch.LongTensor(label_test)

        # return features,adj,idx_train, idx_valid, idx_test, label_train, label_valid, label_test
        return idx_train, idx_valid, idx_test, label_train, label_valid, label_test    


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

# def preprocess_features(features,flag=True):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     if flag:
#         rowsum = np.array(features.sum(1))
#         # r_inv = np.power(rowsum, -1).flatten()
#         epsilon = 1e-10
#         r_inv = np.power(rowsum + epsilon, -1).flatten()

#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = sp.diags(r_inv)
#         features = r_mat_inv.dot(features)
#         return features.todense(), sparse_to_tuple(features)
#     else:
#         rowsum = np.sum(features, axis=1)
#         epsilon = 1e-10
#         r_inv = np.power(rowsum + epsilon, -1).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = np.diag(r_inv)
#         features = np.matmul(r_mat_inv, features)
#         return features, None

def preprocess_features(features, use_sparse=True):
    """
    行归一化特征矩阵，避免显存/内存爆炸。

    Args:
        features (np.ndarray or sp.spmatrix): 输入的特征矩阵。
        use_sparse (bool): 是否使用稀疏矩阵进行处理。

    Returns:
        归一化后的特征矩阵。
    """
    epsilon = 1e-10  # 避免除以零

    if use_sparse:
        # --- 稀疏矩阵处理（适用于大规模数据，节省内存） ---
        # 确保输入是稀疏格式
        if not sp.issparse(features):
            features = sp.csr_matrix(features)
        
        # 计算每行的和
        rowsum = np.array(features.sum(1)).flatten()
        
        # 计算行和的倒数 (D^-1)
        r_inv = np.power(rowsum + epsilon, -1)
        r_inv[np.isinf(r_inv)] = 0.
        
        # 创建一个稀疏对角矩阵，内存效率极高
        r_mat_inv = sp.diags(r_inv)
        
        # 左乘对角矩阵，完成行归一化
        features = r_mat_inv.dot(features)
        return features, None
    
    else:
        # --- 密集矩阵处理（已优化，避免内存爆炸） ---
        # 计算每行的和
        rowsum = np.sum(features, axis=1, keepdims=True) # 使用 keepdims=True 方便广播
        
        # 计算行和的倒数 (D^-1)
        r_inv = np.power(rowsum + epsilon, -1)
        r_inv[np.isinf(r_inv)] = 0.
        
        # 【关键优化点】
        # 不再创建巨大的 np.diag(r_inv) 矩阵
        # 而是利用 NumPy 的广播机制，将每行特征直接乘以其对应的 r_inv 值
        # features 的形状是 (N, F), r_inv 的形状是 (N, 1)
        # NumPy 会自动将 r_inv 扩展为 (N, F) 进行逐元素相乘
        features = features * r_inv
        
        return features, None

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """支持输入 scipy.sparse 矩阵或 COO 格式的元组 (edge_index, edge_weight, shape)"""
#     if isinstance(sparse_mx, tuple):
#         # 已经是 COO 格式的元组
#         edge_index, edge_weight, shape = sparse_mx
#         return torch.sparse_coo_tensor(edge_index, edge_weight, shape)
#     else:
#         # 原始逻辑：处理 scipy.sparse 矩阵
#         sparse_mx = sparse_mx.tocoo().astype(np.float32)
#         indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#         values = torch.from_numpy(sparse_mx.data)
#         shape = torch.Size(sparse_mx.shape)
#         return torch.sparse.FloatTensor(indices, values, shape)