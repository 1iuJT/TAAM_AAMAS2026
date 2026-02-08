import random
import pickle
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.data import CoraGraphDataset, CoraFullDataset, RedditDataset, CiteseerGraphDataset,WikiCSDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,PPIDataset
from ogb.graphproppred import  Evaluator
import copy
import os
from dgl.data.utils import download
from dgl.data import PubmedGraphDataset

class Linear_IL(nn.Linear):
    def forward(self, input: Tensor, n_cls=10000, normalize = True) -> Tensor:
        if normalize:
            return F.linear(F.normalize(input,dim=-1), F.normalize(self.weight[0:n_cls],dim=-1), bias=None)
        else:
            return F.linear(input, self.weight[0:n_cls], bias=None)
        
def accuracy(logits, labels, cls_balance=True, ids_per_cls=None, offset1=0, offset2=None):
    if cls_balance:
        if offset2 is None:
            offset2 = logits.shape[1]
        
        task_logits = logits[:, offset1:offset2]
        
        # 在当前任务的 logits 范围内找最大值
        _, indices = torch.max(task_logits.detach(), dim=1)
        
        if ids_per_cls is None or not any(ids_per_cls):
            correct = torch.sum(indices == labels)
            return correct.item() / len(labels) if len(labels) > 0 else 0.0

        valid_ids_per_cls = [ids for ids in ids_per_cls if ids and len(ids) > 0]
        if not valid_ids_per_cls:
            return 0.0

        acc_per_cls = [torch.sum((indices == labels)[ids]).item() / len(ids) for ids in valid_ids_per_cls]
        return sum(acc_per_cls) / len(acc_per_cls)
    else:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def mean_AP(args,logits, labels, cls_balance=True, ids_per_cls=None):
    eval_ogb = Evaluator(args.dataset)
    pos = (F.sigmoid(logits)>0.5)
    APs = 0
    if cls_balance:
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().numpy()
        acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        input_dict = {"y_true": labels, "y_pred": logits}

        eval_result_ogb = eval_ogb.eval(input_dict)
        for c,ids in enumerate(ids_per_cls):
            TP_ = (pos[ids,c]*labels[ids,c]).sum()
            FP_ = (pos[ids,c]*(labels[ids, c]==False)).sum()
            med0 = TP_ + FP_ + 0.0001
            med1 = TP_ / med0
            APs += med1
        med2 = APs/labels.shape[1]

            #mAP_per_cls.append((TP / (TP+FP)).mean().item())
        #return (TP / (TP+FP)).mean().item()

        return med2.item()

def evaluate_batch(args,model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None):
    model.eval()
    with torch.no_grad():
        dataloader = dgl.dataloading.NodeDataLoader(g.cpu(), list(range(labels.shape[0])), args.nb_sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
        output = torch.tensor([]).cuda(args.gpu)
        output_l = torch.tensor([]).cuda(args.gpu)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_predictions, _ = model.forward_batch(blocks, input_features)
            output = torch.cat((output,output_predictions),dim=0)
            output_l = torch.cat((output_l, output_labels), dim=0)

        #output, _ = model(g, features)
        #judget = (labels==output_l).sum()
        logits = output[:, label_offset1:label_offset2]
        if cls_balance:
            return accuracy(logits, labels.cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask].cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def evaluate(model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, save_logits_name=None):
    model.eval()
    with torch.no_grad():
        output, _ = model(g, features)
        logits = output[:, label_offset1:label_offset2]
        if save_logits_name is not None:
            with open(
                    '/store/continual_graph_learning/baselines_by_TWP/NCGL/results/logits_for_tsne/{}.pkl'.format(
                        save_logits_name), 'wb') as f:
                pickle.dump({'logits':logits,'ids_per_cls':ids_per_cls}, f)

        if cls_balance:
            return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        
def evaluatewp(output, labels, mask=None, ids_per_cls=None, offset1=None, offset2=None,cls_balance=True):

    return accuracy(output, labels, ids_per_cls=ids_per_cls, offset1=offset1, offset2=offset2)


class incremental_graph_trans_(nn.Module):
    def __init__(self, dataset, n_cls):
        super().__init__()
        # 接收数据集信息
        self.graph, self.labels = dataset[0]
        self.tr_va_te_split = dataset[1]
        
        # 初始化图的基本属性
        self.graph.ndata['label'] = self.labels
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_cls = n_cls
        self.n_nodes = self.labels.shape[0]

    def get_graph(self, tasks_to_retain=None, node_ids=None, remove_edges=False):
        """
        全归纳（Fully Inductive）设置下的 get_graph 方法。
        训练、验证、测试子图完全独立。
        """
        if tasks_to_retain is not None:

            orig_ids_train, orig_ids_valid, orig_ids_test = [], [], []

            # 1. 收集所有待处理任务的原始节点ID
            for t_cls in tasks_to_retain:
                orig_ids_train.extend(self.tr_va_te_split[t_cls][0])
                orig_ids_valid.extend(self.tr_va_te_split[t_cls][1])
                orig_ids_test.extend(self.tr_va_te_split[t_cls][2])

            orig_ids_train = sorted(list(set(orig_ids_train)))
            orig_ids_valid = sorted(list(set(orig_ids_valid)))
            orig_ids_test = sorted(list(set(orig_ids_test)))

            train_g = dgl.node_subgraph(self.graph, orig_ids_train, store_ids=True)
            
            valid_g = dgl.node_subgraph(self.graph, orig_ids_valid, store_ids=True)
        
            test_g = dgl.node_subgraph(self.graph, orig_ids_test, store_ids=True)
            
            train_g = dgl.add_self_loop(train_g)
            valid_g = dgl.add_self_loop(valid_g)
            test_g = dgl.add_self_loop(test_g)

            node_ids_per_task_reordered = []
            for c in tasks_to_retain:
                nodes_in_class_c = self.tr_va_te_split[c][0] + self.tr_va_te_split[c][1] + self.tr_va_te_split[c][2]
                node_ids_per_task_reordered.append(sorted(list(set(nodes_in_class_c))))

            new_ids_train = list(range(train_g.num_nodes()))

            new_ids_val = list(range(valid_g.num_nodes()))
            new_ids_test = list(range(test_g.num_nodes()))

            return train_g, valid_g, test_g, node_ids_per_task_reordered, [new_ids_train, new_ids_val, new_ids_test]
        # --- 模式二：为经验回放等，根据给定的node_ids构建子图 ---
        elif node_ids is not None:
            subgraph = dgl.node_subgraph(self.graph, node_ids, store_ids=True)
            
            if remove_edges:
                # 移除所有边，只保留节点
                eids = subgraph.edges(form='eid')
                subgraph.remove_edges(eids)

            # 为了匹配一些需要解包3个值的调用，我们返回None作为占位符
            # 例如 aux_g, aux_ids_per_cls, _ = dataset.get_graph(...)
            return subgraph, None, None
        else:
            raise ValueError("Either 'tasks_to_retain' or 'node_ids' must be provided.")
        
def train_valid_test_split(ids, ratio_valid_test):
    ids = np.array(ids)
    
    if len(ids) < 3:

        train_ids = ids[:1]
        valid_ids = ids[1:2]
        test_ids = ids[2:] # May be empty
        return [train_ids.tolist(), valid_ids.tolist(), test_ids.tolist()]
    va_te_ratio = sum(ratio_valid_test)
    train_size = max(1, int(len(ids) * (1 - va_te_ratio)))
    
    # Perform the initial split
    train_ids, va_te_ids = train_test_split(ids, train_size=train_size, random_state=42)

    # Ensure at least one sample for validation and test sets if possible
    if len(va_te_ids) < 2:
        return [train_ids.tolist(), va_te_ids.tolist(), []]
    
    test_ratio_in_remainder = ratio_valid_test[1] / va_te_ratio
    va_ids, te_ids = train_test_split(va_te_ids, test_size=test_ratio_in_remainder, random_state=42)

    return [train_ids.tolist(), va_ids.tolist(), te_ids.tolist()]

class NodeLevelDataset(incremental_graph_trans_):
    def __init__(self,name='ogbn-arxiv',IL='class',default_split=False,ratio_valid_test=None,args=None):
        r""""
        name: name of the dataset
        IL: use task- or class-incremental setting
        default_split: if True, each class is split according to the splitting of the original dataset, which may cause the train-val-test ratio of different classes greatly different
        ratio_valid_test: in form of [r_val,r_test] ratio of validation and test set, train set ratio is directly calculated by 1-r_val-r_test
        """

        # return an incremental graph instance that can return required subgraph upon request
        if name[0:4] == 'ogbn':
            data = DglNodePropPredDataset(name, root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name in ['CoraFullDataset', 'CoraFull','corafull', 'CoraFull-CL','Corafull-CL']:
            data = CoraFullDataset()
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
        elif name in ['reddit','Reddit','Reddit-CL']:
            data = RedditDataset(self_loop=False)
            graph, label = data.graph, data.labels.view(-1, 1)
        elif name == 'Arxiv-CL':
            data = DglNodePropPredDataset('ogbn-arxiv', './ogb_downloaded')
            graph, label = data[0]
        elif name == 'Products-CL':
            data = DglNodePropPredDataset('ogbn-products', root='./ogb_downloaded')
            graph, label = data[0]

        elif name in ['WikiCS', 'WikiCS-CL', 'wikics']:
            print("Loading WikiCS dataset...")
            # WikiCS 数据集下载和加载由DGL自动处理
            data = WikiCSDataset()
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)

        elif name == 'Cora-CL':
            custom_download_path = f'{args.ori_data_path}'
            download_path = f'{args.ori_data_path}/cora.zip'
            if not os.path.exists(download_path):
                download('https://data.dgl.ai/dataset/cora.zip', path=download_path)
            data = CoraGraphDataset(custom_download_path)
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)

        elif name == 'Citeseer-CL':
            custom_download_path = f'{args.ori_data_path}'
            download_path = f'{args.ori_data_path}/citeseer.zip'
            if not os.path.exists(download_path):
                download('https://data.dgl.ai/dataset/citeseer.zip', path=download_path)
            data = CiteseerGraphDataset(custom_download_path)
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)

        elif name in ['Pubmed', 'Pubmed-CL']:
            data = PubmedGraphDataset()
            graph, label = data[0], data[0].ndata['label'].view(-1, 1)

        elif name in ['AmazonCoBuyComputer', 'Computer-CL']:
            data = AmazonCoBuyComputerDataset()
            graph, label = data[0], data[0].ndata['label'].view(-1, 1)

        elif name in ['AmazonCoBuyPhoto', 'Photo-CL']:
            data = AmazonCoBuyPhotoDataset()
            graph, label = data[0], data[0].ndata['label'].view(-1, 1)

        else:
            print('invalid data name')

        n_cls = data.num_classes
        cls = [i for i in range(n_cls)]
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        cls_sizes = {c: len(cls_id_map[c]) for c in cls_id_map}
        for c in cls_sizes:
            if cls_sizes[c] < 2:
                cls.remove(c) # remove classes with less than 2 examples, which cannot be split into train, val, test sets
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        n_cls = len(cls)
        if default_split:
            split_idx = data.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"].tolist(), split_idx["valid"].tolist(), split_idx[
                "test"].tolist()
            tr_va_te_split = {c: [list(set(cls_id_map[c]).intersection(set(train_idx))),
                                  list(set(cls_id_map[c]).intersection(set(valid_idx))),
                                  list(set(cls_id_map[c]).intersection(set(test_idx)))] for c in cls}

        elif not default_split:
            split_name = f'{args.data_path}/tr{round(1-ratio_valid_test[0]-ratio_valid_test[1],2)}_va{ratio_valid_test[0]}_te{ratio_valid_test[1]}_split_{name}.pkl'
            try:
                tr_va_te_split = pickle.load(open(split_name, 'rb')) # could use same split across different experiments for consistency
            except:
                if ratio_valid_test[1] > 0:
                    tr_va_te_split = {c: train_valid_test_split(cls_id_map[c], ratio_valid_test=ratio_valid_test)
                                      for c in
                                      cls}
                    print(f'splitting is {ratio_valid_test}')
                elif ratio_valid_test[1] == 0:
                    tr_va_te_split = {c: [cls_id_map[c], [], []] for c in
                                      cls}
                with open(split_name, 'wb') as f:

                    pickle.dump(tr_va_te_split, f)
        super().__init__([[graph, label], tr_va_te_split], n_cls)