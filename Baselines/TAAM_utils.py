# Standard library imports
import os
import copy
import pickle
from functools import partial
from copy import deepcopy

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.manifold import TSNE
import random

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Graph-related imports
import dgl
import dgl.function as fn
from torch_geometric.data import Data, NeighborSampler
from torch_geometric.datasets import Planetoid, Amazon, CoraFull
from torch_geometric.utils import (
    to_dense_adj,
    to_scipy_sparse_matrix,
)
from torch_geometric.loader import ClusterData, ClusterLoader


from torch_geometric.nn.inits import glorot


def addedges(subgraph):
    subgraph = copy.deepcopy(subgraph)
    nodedegree = subgraph.in_degrees().cpu()
    isolated_nodes = torch.where(nodedegree==1)[0]
    connected_nodes = torch.where(nodedegree!=1)[0]
    isolated_nodes = isolated_nodes.numpy()
    connected_nodes = connected_nodes.numpy()
    randomnode = np.random.choice(connected_nodes, isolated_nodes.shape[0])
    srcs = np.concatenate([isolated_nodes, randomnode])
    dsts = np.concatenate([randomnode, isolated_nodes])
    subgraph.add_edges(srcs, dsts)
    return subgraph


class random_select(nn.Module):
    """
    随机选择策略：为每个类别随机选择固定数量的样本。
    """
    def __init__(self, args=None):
        super(random_select, self).__init__()
    
    def forward(self, ids_per_cls_train, budget):
        store_ids = []
        for i, ids in enumerate(ids_per_cls_train):
            # 确定每个类别的预算
            budget_ = min(budget, len(ids))
            store_ids.extend(random.sample(ids, budget_))
        return store_ids
    
    
class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x):
        return x + self.global_emb


class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p

def accuracy(logits, labels, ids_per_cls=None, offset1=0, offset2=None):

    if offset2 is None:
        offset2 = logits.shape[1]
    
    # 关键：只关注当前任务对应的输出神经元
    # 使用 offset1 和 offset2 对 logits 进行切片
    task_logits = logits[:, offset1:offset2]
    
    # 在当前任务的 logits 范围内找最大值
    _, indices = torch.max(task_logits.detach(), dim=1)
    
    # labels 是本地标签，无需再调整
    
    # 过滤掉没有样本的类别，防止除以零
    if ids_per_cls is None or not any(ids_per_cls):
        correct = torch.sum(indices == labels)
        return correct.item() / len(labels) if len(labels) > 0 else 0.0

    valid_ids_per_cls = [ids for ids in ids_per_cls if ids and len(ids) > 0]
    if not valid_ids_per_cls:
        return 0.0

    acc_per_cls = [torch.sum((indices == labels)[ids]).item() / len(ids) for ids in valid_ids_per_cls]
    return sum(acc_per_cls) / len(acc_per_cls)

def evaluatewp(output, labels, mask=None, ids_per_cls=None, offset1=None, offset2=None):

    return accuracy(output, labels, ids_per_cls=ids_per_cls, offset1=offset1, offset2=offset2)


# 在类定义外添加可视化函数
def plot_tsne(data, labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(data)
    
    plt.figure(figsize=(20, 16))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab20', alpha=0.6)
    plt.title(title)
    plt.colorbar(scatter)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_history_prompt(history_prompt, title, save_path):
    """
    可视化不同任务的prompts在特征空间中的分布 (t-SNE降维)
    
    参数:
        history_prompt: 存储了每个task的prompts的列表
        title: 图像标题
        save_path: 图像保存路径
    """
    # 准备数据
    all_prompts = []
    task_labels = []
    
    for record in history_prompt:
        # prompts形状假设为 (top_k, embed_dim)
        prompts = record['prompts']
        task_idx = record['task']
        
        # 展平所有prompts (如果是top_k > 1的情况)
        if len(prompts.shape) > 2:
            prompts = prompts.reshape(-1, prompts.shape[-1])
        
        all_prompts.append(prompts)
        task_labels.extend([task_idx] * len(prompts))
    
    # 合并所有prompts
    all_prompts = np.concatenate(all_prompts, axis=0)
    task_labels = np.array(task_labels)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    prompts_tsne = tsne.fit_transform(all_prompts)
    
    # 绘制图像
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("hsv", len(np.unique(task_labels)))
    
    # 为每个task绘制不同颜色
    for task_idx in np.unique(task_labels):
        mask = task_labels == task_idx
        plt.scatter(prompts_tsne[mask, 0], prompts_tsne[mask, 1], 
                    color=palette[task_idx], 
                    label=f'Task {task_idx}', 
                    alpha=0.7)
    
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Prompt visualization saved to {save_path}")

def plot_history_means(history_means, title, save_path):
    """
    可视化不同类别的class means在特征空间中的分布 (t-SNE降维)
    
    参数:
        history_means: 存储了每个class的means的列表
        title: 图像标题
        save_path: 图像保存路径
    """
    # 准备数据
    all_means = []
    class_labels = []
    
    for record in history_means:
        class_mean = record['class_means']
        label = record['class_labels']
        
        all_means.append(class_mean)
        class_labels.append(label)
    
    # 合并数据
    all_means = np.stack(all_means)
    class_labels = np.array(class_labels)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    means_tsne = tsne.fit_transform(all_means)
    
    # 绘制图像
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("hsv", len(np.unique(class_labels)))
    
    # 为每个class绘制不同颜色
    for label in np.unique(class_labels):
        mask = class_labels == label
        plt.scatter(means_tsne[mask, 0], means_tsne[mask, 1], 
                    color=palette[label], 
                    label=f'Class {label}', 
                    alpha=0.7)
    
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Class means visualization saved to {save_path}")

def dgl_to_pyg(dgl_graph):
    # 获取边的源节点和目标节点
    src_nodes, dst_nodes = dgl_graph.edges()
    # 获取边类型
    edge_types = dgl_graph.edata['etype'] if 'etype' in dgl_graph.edata else None
    # 获取节点特征
    node_features = dgl_graph.ndata['feat'] if 'feat' in dgl_graph.ndata else None
    # 获取节点标签
    node_labels = dgl_graph.ndata['label'] if 'label' in dgl_graph.ndata else None
    
    # 创建 PyTorch Geometric 的 Data 对象
    pyg_data = Data(
        x=node_features,
        edge_index=torch.stack((src_nodes, dst_nodes), dim=0),
        y=node_labels,
        edge_type=edge_types
    )
    return pyg_data


    




