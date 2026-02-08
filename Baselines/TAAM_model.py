
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .TAAM_utils import *
from .TAAM_utils import evaluatewp
from Baselines.grace import ModelGrace, traingrace, LogReg
import dgl.function as fn


# class AttentionalFiLMGenerator(nn.Module):

#     def __init__(self, task_emb_dim: int, node_feature_dim: int, output_dim: int, num_heads: int = 3):

#         super().__init__()
#         self.num_heads = num_heads
#         self.output_dim = output_dim

#         self.base_generator = nn.Linear(task_emb_dim, num_heads * output_dim * 2)
#         self.attention_net = nn.Sequential(
#             nn.Linear(node_feature_dim, num_heads)
#         )
#         self.task_embedding = nn.Parameter(torch.randn(1, task_emb_dim))

#     def forward(self, node_features):

#         base_modulations = self.base_generator(self.task_embedding).view(self.num_heads, self.output_dim * 2)
#         attention_scores = self.attention_net(node_features)
#         attention_weights = F.softmax(attention_scores, dim=1)
#         final_modulations = torch.matmul(attention_weights, base_modulations)
#         gamma, beta = torch.split(final_modulations, self.output_dim, dim=1)

#         return gamma, beta


def get_appnp_profile(g, features, hops=[0,1,2], alpha=0.1, k_iterations=10):
    """
    采用APPNP的逻辑进行图扩散，并仿照SIGN拼接多尺度特征。
    
    Args:
        g (DGLGraph): 输入图。
        features (Tensor): 原始节点特征。
        hops (list): 要拼接的APPNP迭代次数。e.g., [0, 1, 2] 表示拼接0次、1次、2次迭代后的结果。
        alpha (float): 传送概率，回到初始特征的概率。
        k_iterations (int): APPNP的总迭代次数，用于计算最终的平滑结果。
                           注意：这个k_iterations与hops是不同的概念。
                           hops指定了要从迭代过程中取出哪些中间结果进行拼接。
    """
    device = features.device
    smoothed_features = []

    g_for_smooth = g
    degs = g_for_smooth.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5).to(device).unsqueeze(1)

    # 0阶特征就是原始特征
    h_initial = features
    if 0 in hops:
        smoothed_features.append(h_initial)

    h_current = h_initial

    for i in range(1, k_iterations + 1):
        # 计算 (D^-1/2 * A * D^-1/2) * H_prev
        h_prev_prop = h_current * norm
        g_for_smooth.ndata['h'] = h_prev_prop
        g_for_smooth.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h_prev_prop = g_for_smooth.ndata.pop('h')
        h_prev_prop = h_prev_prop * norm
    
        h_current = (1 - alpha) * h_prev_prop + alpha * h_initial
        
        if i in hops:
            smoothed_features.append(h_current)
            
    final_features_to_cat = []
    seen_hops = set()
    for h in hops:
        if h not in seen_hops:
            # 找到对应的特征（注意索引偏移）
            if h == 0:
                final_features_to_cat.append(smoothed_features[0])
            else:
                idx = hops.index(h)
                final_features_to_cat.append(smoothed_features[idx]) # 根据hops列表的顺序来取
            seen_hops.add(h)

    return torch.cat(final_features_to_cat, dim=1)

# 
class AttentionalFiLMGenerator(nn.Module):
    def __init__(self, task_emb_dim: int, node_feature_dim: int, output_dim: int, num_heads: int = 3, rank: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.rank = rank # 引入低秩维度

        self.generator_A = nn.Linear(task_emb_dim, num_heads * self.rank)
        self.generator_B = nn.Parameter(torch.Tensor(self.rank, self.output_dim * 2))

        self.attention_net = nn.Linear(node_feature_dim, num_heads)
        self.task_embedding = nn.Parameter(torch.randn(1, task_emb_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # 对A和B进行初始化
        nn.init.kaiming_uniform_(self.generator_A.weight, a=math.sqrt(3))
        nn.init.kaiming_uniform_(self.generator_B, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.attention_net.weight, a=math.sqrt(5))
        
    def forward(self, node_features):
        # 1. 生成低秩矩阵 A
        delta_A = self.generator_A(self.task_embedding).view(self.num_heads, self.rank)
        
        # 2. 通过矩阵乘法重构出 base_modulations
        # (num_heads, rank) @ (rank, output_dim * 2) -> (num_heads, output_dim * 2)
        base_modulations = torch.matmul(delta_A, self.generator_B)

        # attention部分保持不变
        attention_scores = self.attention_net(node_features)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # (num_nodes, num_heads) @ (num_heads, output_dim * 2) -> (num_nodes, output_dim * 2)
        final_modulations = torch.matmul(attention_weights, base_modulations)
        gamma, beta = torch.split(final_modulations, self.output_dim, dim=1)

        return gamma, beta 



class TAAM(nn.Module):

    def __init__(self, sgc_backbone, args):
        super(TAAM, self).__init__()
        self.sgc_backbone = sgc_backbone
        self.args = args

        self.task_emb_dim = args.TAAM_args['task_emb_dim']
        
        self.film_generators = nn.ModuleList()
        self.classifiers = nn.ModuleList()

        self.task_embeddings = nn.ParameterList()
        h_dims = args.TAAM_args['h_dims']
        self.hid_units = h_dims[-1] if h_dims else args.d_data

        self.layer_norm = nn.LayerNorm(args.d_data)
        
        self.task_prototypes =[]
        self.tasks =0
        self.n_classes = args.n_cls

        # 分类器逻辑保持不变
        self.classifier = LogReg(self.hid_units, args.n_cls_per_task)

    def _create_new_adapter(self, device):
        feature_dim = self.args.d_data
        
        generator = AttentionalFiLMGenerator(
            task_emb_dim=self.task_emb_dim,
            node_feature_dim=feature_dim, 
            output_dim=feature_dim,       
            num_heads=self.args.TAAM_args['num_heads']
        ).to(device)
        
        return generator
    
    def prepare_for_new_task(self, g, features, train_ids):
        device = f"cuda:{self.args.gpu}"
        trainable_params = []

        new_adapter = self._create_new_adapter(device)
        self.film_generators.append(new_adapter) # 直接append
        current_prototype = self._get_prototype(g, features, train_ids)
        trainable_params.extend(list(self.film_generators[-1].parameters()))
        if self.tasks > 0:
            prev_n_cls = self.classifier.fc.out_features
            new_n_cls = prev_n_cls + self.args.n_cls_per_task
            new_classifier = LogReg(self.hid_units, new_n_cls).to(device)
            
            with torch.no_grad():
                new_classifier.fc.weight.data[:prev_n_cls, :] = self.classifier.fc.weight.data
                new_classifier.fc.bias.data[:prev_n_cls] = self.classifier.fc.bias.data
            
            self.classifier = new_classifier

        self.task_prototypes.append(current_prototype.cpu())
        self.tasks += 1
        
        return trainable_params
        
    def _get_prototype(self, g, features, node_ids):
        with torch.no_grad():
            enhanced_features = get_appnp_profile(g, features, hops=[0,2], alpha=0.1, k_iterations=10)
            prototype = torch.mean(enhanced_features[node_ids], dim=0)
            return prototype

    def infer_task_id(self, g, features, test_ids):

        test_prototype = self._get_prototype(g, features, test_ids)
        prototypes_tensor = torch.stack(self.task_prototypes).to(test_prototype.device)
        similarity_scores = F.cosine_similarity(test_prototype, prototypes_tensor, dim=1)
        inferred_id = torch.argmax(similarity_scores)
        return inferred_id.item()

    def forward(self, graph, x, task_id):
        x = self.sgc_backbone.neighbor_agg(graph, x)
        adapter = self.film_generators[task_id]

        gamma, beta = adapter(x)
        x_modulated = gamma * self.layer_norm(x) + beta
        x_modulated = x_modulated + x
        x_processed = self.sgc_backbone.feat_trans(x_modulated)
        logits = self.classifier(x_processed)
        
        return logits

class NET(torch.nn.Module):
    def __init__(self, model, task_manager, args):
        super(NET, self).__init__() 
        self.model = model
        self.args = args
        self.device = f"cuda:{args.gpu}"
        self.task_manager = task_manager
        self.n_tasks = args.n_tasks

        self.ft_size = args.d_data
        self.nb_classes = args.n_cls
        self.hid_units = 256
        self.ce = torch.nn.functional.cross_entropy
        self.drop_edge = 0.2
        self.drop_feature = 0.3
        self.best_acc = 0
        self.patience_counter = 0

        self.net = TAAM(
            sgc_backbone=self.model,
            args=self.args
        ).to(self.device)


    def evaluate(self, g, features, taskid):

        self.net.eval()
        with torch.no_grad():
            logits = self.net(g, features, taskid)
        return logits

    def prepare(self, subgraph, features, train_ids):

        trainable_params = self.net.prepare_for_new_task(subgraph, features, train_ids)
        self.opt = torch.optim.Adam(trainable_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
    
        
    def observe(self, subgraph, features, labels, task, train_ids, ids_per_cls, offset1, offset2):
        self.net.train() 
        self.net.sgc_backbone.eval()
        self.opt.zero_grad()
        logits_student= self.net(subgraph,features,task_id=task) 

        n_per_cls = [(labels[train_ids] == j).sum() for j in range(self.nb_classes)]
        loss_w_ = [1. / max(i, 1) for i in n_per_cls]
        loss_w_ = torch.tensor(loss_w_).to(self.device)
        
        total_loss = self.ce(logits_student[train_ids, offset1:offset2], labels[train_ids], weight=loss_w_[offset1:offset2]) 
        total_loss.backward()
        self.opt.step()
    

