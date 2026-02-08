import os
import pickle
import numpy as np
import torch
from Backbones.model_factory import get_model
from Backbones.utils import evaluate, evaluatewp, NodeLevelDataset, evaluate_batch
from training.utils import mkdir_if_missing
from dataset.utils import semi_task_manager
import importlib
import copy
import dgl
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import time
import copy
from argparse import Namespace
import tempfile
import json

def visualize_resource_usage(tasks, gpu_usage, storage_usage, save_path='resource_usage.png'):
    """
    一个独立的函数，用于绘制资源使用情况图表。
    """
    num_tasks = len(tasks)
    tasks_x_axis = list(range(1, num_tasks + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Resource Usage Profile per Task', fontsize=16)

    # 图1: 峰值显存占用
    ax1.plot(tasks_x_axis, gpu_usage, marker='o', linestyle='-', color='deepskyblue', label='Peak GPU Memory')
    ax1.set_title('Peak GPU Memory Usage per Task')
    ax1.set_xlabel('Task Number')
    ax1.set_ylabel('Memory (MB)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xticks(tasks_x_axis)
    ax1.legend()

    # 图2: 模型存储占用
    ax2.plot(tasks_x_axis, storage_usage, marker='s', linestyle='--', color='salmon', label='Model Storage Size')
    ax2.set_title('Model Storage Size after Each Task')
    ax2.set_xlabel('Task Number')
    ax2.set_ylabel('Storage (MB)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xticks(tasks_x_axis)
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(save_path)
    print(f"\n📊 Resource usage visualization saved to '{save_path}'")
    plt.show()

joint_alias = ['joint', 'Joint', 'joint_replay_all', 'jointtrain']
def get_pipeline(args):
    if args.minibatch:
        if args.ILmode == 'classIL':
            if args.inter_task_edges:
                if args.method in joint_alias:
                    return pipeline_class_IL_inter_edge_minibatch_joint
                else:
                    return pipeline_class_IL_inter_edge_minibatch
            else:
                if args.method in joint_alias:
                    return pipeline_class_IL_no_inter_edge_minibatch_joint
                else:
                    return pipeline_class_IL_no_inter_edge_minibatch
        elif args.ILmode == 'taskIL':
            if args.inter_task_edges:
                if args.method in joint_alias:
                    return pipeline_task_IL_inter_edge_minibatch_joint
                else:
                    return pipeline_task_IL_inter_edge_minibatch
            else:
                if args.method in joint_alias:
                    return pipeline_task_IL_no_inter_edge_minibatch_joint
                else:
                    return pipeline_task_IL_no_inter_edge_minibatch
    else:
        if args.ILmode == 'classIL':
            if args.inter_task_edges:
                if args.method in joint_alias:
                    return pipeline_class_IL_inter_edge_joint
                else:
                    return pipeline_class_IL_inter_edge
            else:
                if args.method in joint_alias:
                    return pipeline_class_IL_no_inter_edge_joint
                else:
                    return pipeline_class_IL_no_inter_edge
        
        elif args.ILmode == 'taskIL':

            if args.method in joint_alias:
                return pipeline_task_IL_no_inter_edge_joint

    
def data_prepare(args, dataset):
    torch.cuda.set_device(args.gpu)
    str_int_tsk = 'inter_tsk_edge' if args.inter_task_edges else 'no_inter_tsk_edge'
    args.n_tasks = len(args.task_seq)

    for task, task_cls in enumerate(args.task_seq):
        save_dir = f'{args.data_path}/{str_int_tsk}'
        file_path = f'{save_dir}/{args.dataset}_{task_cls}.pkl'

        if args.load_check and os.path.exists(file_path):
            print(f"Data for task {task} ({task_cls}) already exists. Skipping generation.")
            continue
        
        if os.path.exists(file_path) and not hasattr(args, 'force_regenerate'):
            continue

        print(f'Preparing data for task {task} ({task_cls})')
        mkdir_if_missing(save_dir)

        if args.inter_task_edges:
            cls_retain = []
            for clss in args.task_seq[0:task + 1]:
                cls_retain.extend(clss)
            
            train_g, valid_g, test_g, ids_per_cls, indices = dataset.get_graph(tasks_to_retain=cls_retain)
            
            data_to_save = {
                'train_g': train_g, 'valid_g': valid_g, 'test_g': test_g,
                'ids_per_cls': ids_per_cls, 'indices': indices
            }
            with open(file_path, 'wb') as f:
                pickle.dump(data_to_save, f)

        else: # if not inter_task_edges
            # --- 以下是需要缩进的正确逻辑 ---
            train_g, valid_g, test_g, ids_per_cls, indices = dataset.get_graph(tasks_to_retain=task_cls)

            data_to_save = {
                'train_g': train_g, 'valid_g': valid_g, 'test_g': test_g,
                'ids_per_cls': ids_per_cls, 'indices': indices
            }
            with open(file_path, 'wb') as f:
                pickle.dump(data_to_save, f)
        
        print(f"Successfully generated and saved data for task {task} to {file_path}")


def pipeline_class_IL_no_inter_edge(args, valid=False):

    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]

    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    data_prepare(args, dataset)

    model = get_model(dataset, args).cuda(args.gpu) if valid else None
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    if args.method in ['taam']:
        life_model_ins = life_model.NET(model, task_manager, args)
        acc_mean, mean_backward, acc_matrix,acc_means = life_model_ins.observe(args)
        tp =100
        return acc_mean, mean_backward, acc_matrix,acc_means,tp
    
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    if args.method == 'tpp':
        prototypes = torch.zeros(args.n_tasks, args.d_data)

    gpu_memory_per_task = []
    storage_usage_per_task = []

    name, ite = args.current_model_save_path
    config_name = name.split('/')[-1]
    subfolder_c = name.split(config_name)[-2]
    save_model_name = f'{config_name}_{ite}'
    save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'

    if args.method == 'tpp':
        save_proto_name = save_model_name + '_prototypes'
        save_proto_path = f'{args.result_path}/{subfolder_c}val_models/{save_proto_name}.pkl'
    
    if not valid:
        life_model_ins = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        if args.method == 'tpp':
            prototypes = pickle.load(open(save_proto_path,'rb'))

    meanas = []
    prev_model = None
    n_cls_so_far = 0

    all_inference_id_times_ms = []
    all_inference_cls_times_ms = []
    total_training_time_seconds= []
    TPs = []
    for task, task_cls in enumerate(args.task_seq):

        train_time = time.time()
        n_cls_so_far+=len(task_cls)

        file_path = f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl'
        data_dict = pickle.load(open(file_path, 'rb'))
    
        train_g = data_dict['train_g']
        ids_per_cls = data_dict['ids_per_cls']
        train_ids, valid_ids, _ = data_dict['indices'] # 训练阶段只需要 train_ids
        
        subgraph = train_g.to(device='cuda:{}'.format(args.gpu))


        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        if args.method == 'tpp':
            label_offset1 = task_manager.get_label_offset(task - 1)[1]
        else:
            label_offset1, label_offset2 = task_manager.get_label_offset(task)

        # if args.method == 'tpp' and task ==0:
        #     life_model_ins.pretrain(args, subgraph, features)

        if args.method == 'TAAM' and valid:
            life_model_ins.prepare(subgraph, features, train_ids)
        ids_per_cls = []
        for cls in task_cls:
            nodes_for_cls = (subgraph.ndata['label'] == cls).nonzero(as_tuple=True)[0].tolist()
            ids_per_cls.append(nodes_for_cls)

        for epoch in range(epochs):

            if args.method == 'lwf':
                life_model_ins.observe(args, subgraph, features, labels, task, prev_model, train_ids, ids_per_cls, dataset)
            elif args.method == 'tpp':
                life_model_ins.observe_il(subgraph, features, labels, task, train_ids, ids_per_cls, label_offset1, dataset)
            
            elif args.method == 'TAAM':
                life_model_ins.observe(subgraph, features, labels, task, train_ids, ids_per_cls, label_offset1, label_offset2)
                torch.cuda.empty_cache()
            elif args.method == 'DeLoMe':
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, valid_ids, ids_per_cls, dataset)
            else:
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)

                torch.cuda.empty_cache()

        if valid and args.method == 'tpp':
            prototypes[task] = life_model_ins.getprototype(subgraph, features, train_ids)
        
        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))

        e_time = time.time() 
        total_training_time_seconds.append((e_time - train_time) * 1000)

        peak_gpu_mem = torch.cuda.max_memory_allocated(args.gpu) / (1024 ** 2) # 转换为 MB
        gpu_memory_per_task.append(peak_gpu_mem)
        print(f"📈 Peak GPU Memory for Task {task+1}: {peak_gpu_mem:.2f} MB")

        with tempfile.NamedTemporaryFile(delete=True) as temp_f:
            # 将模型移动到CPU以测量纯粹的参数存储，避免GPU元数据影响
            life_model_ins.cpu()
            pickle.dump(life_model_ins, temp_f)
            storage_size_mb = os.path.getsize(temp_f.name) / (1024 ** 2) # 转换为 MB
            life_model_ins.cuda(args.gpu) # 移回GPU
        
        storage_usage_per_task.append(storage_size_mb)
        acc_mean = []
        current_stage_predictions = []
        for t in range(task+1):
            eval_task_cls = args.task_seq[t]
            eval_file_path = f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{eval_task_cls}.pkl'
            eval_data_dict = pickle.load(open(eval_file_path, 'rb'))

            if valid:
                subgraph = eval_data_dict['valid_g'].to(device='cuda:{}'.format(args.gpu))
                test_ids = eval_data_dict['indices'][1] 
            else:
                subgraph = eval_data_dict['test_g'].to(device='cuda:{}'.format(args.gpu))
                test_ids = eval_data_dict['indices'][2] 
            
            ids_per_cls = eval_data_dict['ids_per_cls']

            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

            ids_per_cls_test = []
            for cls in eval_task_cls:
                nodes_for_cls = (subgraph.ndata['label'] == cls).nonzero(as_tuple=True)[0].tolist()
                ids_per_cls_test.append(nodes_for_cls)

            torch.cuda.synchronize()
            start_time = time.time()

            if args.method == 'tpp':

                taskid = life_model_ins.gettaskid(prototypes, subgraph, features, task + 1, test_ids)

                if taskid != t:
                    print("="*20 + " Task ID Mismatch! " + "="*20)
                    print(f"Data from actual Task {t} was INCORRECTLY predicted as Task {taskid}.")
                    print("="*62)
                torch.cuda.synchronize()
                id_time = time.time()

                label_offset1, label_offset2 = task_manager.get_label_offset(int(taskid) - 1)[1], task_manager.get_label_offset(int(taskid))[1]
                labels = labels - label_offset1
                output = life_model_ins.getpred(subgraph, features,taskid)

                torch.cuda.synchronize()
                cls_time = time.time()

                all_inference_id_times_ms.append((id_time - start_time) * 1000)
                all_inference_cls_times_ms.append((cls_time - id_time) * 1000)

                acc = evaluatewp(output, labels, test_ids, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
                
                current_stage_predictions.append({
                    'ground_truth': t,
                    'predicted': int(taskid) if taskid != -1 else -1
                })

            elif args.method == 'TAAM':

                taskid = life_model_ins.net.infer_task_id(subgraph, features, test_ids)

                torch.cuda.synchronize()
                id_time = time.time()

                if taskid != t:
                    print("="*20 + " Task ID Mismatch! " + "="*20)
                    print(f"Data from actual Task {t} was INCORRECTLY predicted as Task {taskid}.")
                    print("="*62)
                t_offset1, t_offset2 = task_manager.get_label_offset(taskid)

                output = life_model_ins.evaluate(subgraph, features,taskid)

                torch.cuda.synchronize()
                cls_time = time.time()

                # Append to the global lists
                all_inference_id_times_ms.append((id_time - start_time) * 1000)
                all_inference_cls_times_ms.append((cls_time - id_time) * 1000)

                acc = evaluatewp(output, labels, test_ids, ids_per_cls=ids_per_cls_test, offset1=label_offset1, offset2=label_offset2)
                
                current_stage_predictions.append({
                    'ground_truth': t,
                    'predicted': int(taskid) if taskid != -1 else -1
                    })
            else: # 其他方法的逻辑保持不变
                if args.classifier_increase:
                    acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
                else:
                    acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, args.n_cls, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)

            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc*100:.2f}|", end="")
            
        if args.method == 'tpp':
            acc_mean = round(np.mean(acc_mean)*100,2)
            print(f"acc_mean(ID acc): {acc_mean})", end="")
            meanas.append(acc_mean) 
        else:
            accs = acc_mean[:task+1]
            meana = round(np.mean(accs)*100,2)
            meanas.append(meana)
            acc_mean = round(np.mean(acc_mean)*100,2)
            print(f"acc_mean: {acc_mean}", end="")

        if current_stage_predictions: # 确保列表不为空
            correct_id_preds = sum(1 for pred in current_stage_predictions if pred['ground_truth'] == pred['predicted'])
            task_id_accuracy = (correct_id_preds / len(current_stage_predictions)) * 100
            TPs.append(task_id_accuracy)
        else:
            # 如果没有评估，可以追加一个标记值，如 -1 或 0
            task_id_accuracy = 0
            TPs.append(0) 

        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            try:
                if args.method == 'tpp':
                    with open(save_model_path, 'wb') as f:
                        pickle.dump(life_model_ins, f) # save the best model for each hyperparameter composition
                    with open(save_proto_path, 'wb') as f:
                        pickle.dump(prototypes, f)
                elif args.method == 'TAAM':
                    with open(save_model_path, 'wb') as f:
                        pickle.dump(life_model_ins, f) # save the best model for each hyperparameter composition
                else:
                    with open(save_model_path, 'wb') as f:
                        pickle.dump(model, f) # save the best model for each hyperparameter composition
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))

        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks-1):
        b = acc_matrix[args.n_tasks-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)

    print('AF: ', mean_backward)
    print('\n')
    print(" " * 15 + "FINAL RUN SUMMARY")
    print("="*50)
    print(f" TP | Task ID Acc: {task_id_accuracy:.2f}%", end="")
    total_training_time= np.mean(total_training_time_seconds)
    print(f"⌛ Total Training Time (All Tasks): {total_training_time:.2f} ms")

# ... (in the pipeline function, before the final return statement) ...

    # --- [START] Final Analysis Block (v5 - with TPP/TAAM specifics) ---
    print("\n" + "="*70)
    print(" " * 10 + "EFFICIENCY ANALYSIS (Unit: M, measured in float32 equivalents)")
    print("="*70)

    # --- 1. Model Parameter Analysis (Unit: M, 1 param ≈ 1 float32) ---
    print("--- [Model Parameters (M)] ---")
    
    total_params = sum(p.numel() for p in life_model_ins.parameters())

    # --- Dynamically locate the backbone based on the method ---
    if args.method == 'TAAM':
        print("INFO: Applying TAAM-specific structure (backbone at: .net.sgc_backbone).")
        backbone_params = sum(p.numel() for p in life_model_ins.net.sgc_backbone.parameters())
    elif args.method == 'tpp':
        print("INFO: Applying TPP-specific structure (backbone at: .model).")
        backbone_params = sum(p.numel() for p in life_model_ins.model.parameters())
    else:
        # Fallback for generic or other methods
        print("INFO: Applying generic structure check.")
        if hasattr(life_model_ins, 'net'):
            backbone_params = sum(p.numel() for p in life_model_ins.net.parameters())
        else: # Assumes the instance itself is the backbone
            backbone_params = total_params
    
    extra_params = total_params - backbone_params
        
    print(f"Σ Total Parameters: {total_params / 1e6:.4f} M")
    print(f"🧠 Backbone Parameters: {backbone_params / 1e6:.4f} M")
    print(f"🔌 Extra Parameters: {extra_params / 1e6:.4f} M")

    # --- 2. Additional Data Storage Analysis (Unit: M, float32 equivalents) ---
    print("\n--- [Additional Data Storage (M)] ---")
    
    additional_data_m = 0.0
    
    # This section remains unchanged as it correctly checks for various buffer types
    if hasattr(life_model_ins, 'TEM_vecs'):
        print("INFO: Detected direct tensor replay buffer (e.g., TEM).")
        replay_vecs = life_model_ins.TEM_vecs
        replay_labels = life_model_ins.TEM_labels
        if replay_vecs.numel() > 0:
            vecs_f32_equiv = replay_vecs.numel() * (replay_vecs.element_size() / 4)
            labels_f32_equiv = replay_labels.numel() * (replay_labels.element_size() / 4)
            additional_data_m = (vecs_f32_equiv + labels_f32_equiv) / 1e6
        print(f"📦 Data Storage: {additional_data_m:.4f} M")

    elif hasattr(life_model_ins, 'buffer_all_nodes') or hasattr(life_model_ins, 'buffer_node_ids'):
        node_ids = []
        if hasattr(life_model_ins, 'buffer_all_nodes'):
             print("INFO: Detected node ID replay buffer (e.g., SSM). Estimating size.")
             node_ids = life_model_ins.buffer_all_nodes
        else:
             print("INFO: Detected node ID replay buffer (e.g., ER-GNN). Estimating size.")
             node_ids = life_model_ins.buffer_node_ids
        
        num_nodes = len(node_ids)
        if num_nodes > 0:
            feature_f32_equiv = num_nodes * args.d_data
            label_f32_equiv = num_nodes * 2
            additional_data_m = (feature_f32_equiv + label_f32_equiv) / 1e6
        print(f"📦 Estimated Data Storage: {additional_data_m:.4f} M ({num_nodes} nodes)")
        
    elif hasattr(life_model_ins, 'aux_g') and isinstance(life_model_ins.aux_g, list):
        print("INFO: Detected condensed graph buffer (e.g., DeLoMe).")
        condensed_graphs = life_model_ins.aux_g
        total_f32_equiv = 0
        total_nodes = 0
        if condensed_graphs:
            for g in condensed_graphs:
                if 'feat' in g.ndata:
                    total_f32_equiv += g.ndata['feat'].numel() * (g.ndata['feat'].element_size() / 4)
                if 'label' in g.ndata:
                    total_f32_equiv += g.ndata['label'].numel() * (g.ndata['label'].element_size() / 4)
                total_nodes += g.num_nodes()
            additional_data_m = total_f32_equiv / 1e6
        print(f"📦 Data Storage: {additional_data_m:.4f} M ({len(condensed_graphs)} graphs, {total_nodes} total nodes)")

    else:
        print("INFO: No additional data storage (e.g., replay buffer) detected for this method.")

    # --- 3. Total Additional Cost Summary (Unit: M, float32 equivalents) ---
    print("\n--- [Total Additional Cost (M)] ---")
    
    # Direct sum of extra parameter count and data storage's float32 equivalent count
    total_additional_cost_m = (extra_params / 1e6) + additional_data_m
    
    print(f"📈 Total Additional Cost (Extra Params + Data): {total_additional_cost_m:.4f} M")
    
    print("="*70)
    # --- [END] Final Analysis Block ---

    return acc_mean, mean_backward, acc_matrix, meanas, task_id_accuracy

def pipeline_class_IL_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):

        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)

        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
            f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        
        test_ids = valid_ids_ if valid else test_ids_
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        cls_ids_new = [cls_retain.index(i) for i in task_cls]
        ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
        train_ids_current_task = []
        for ids in ids_per_cls_train:
            train_ids_current_task.extend(ids)

        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe(args, subgraph, features, labels, task, prev_model,
                                               train_ids_current_task, ids_per_cls_current_task, dataset)
            else:
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids_current_task,
                                               ids_per_cls_current_task, dataset)
        # test
        label_offset1, label_offset2 = task_manager.get_label_offset(task)

        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []
        for t in range(task + 1):
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            try:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(model, f) # save the best model for each hyperparameter composition
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    # 如果是TAAM、TEM或TPP方法且在验证阶段，绘制训练曲线
    if (args.method == 'TAAM' or args.method == 'TEM' or args.method == 'tpp') and valid and life_model_ins is not None:
        try:
            life_model_ins.plot_training_curves()
        except Exception as e:
            print(f"Error plotting training curves: {e}")
    
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args, dataset)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far+=len(task_cls)

        file_path = f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl'
        data_dict = pickle.load(open(file_path, 'rb'))
    
        train_g = data_dict['train_g']
        ids_per_cls = data_dict['ids_per_cls']
        train_ids, valid_ids, _ = data_dict['indices'] # 训练阶段只需要 train_ids
        
        subgraph = train_g.to(device='cuda:{}'.format(args.gpu))

        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            file_path = f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl'
            data_dict = pickle.load(open(file_path, 'rb'))
        
            train_g = data_dict['train_g']
            ids_per_cls = data_dict['ids_per_cls']
            train_ids, valid_ids, _ = data_dict['indices'] # 训练阶段只需要 train_ids
            
            subgraph = train_g.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)

        for epoch in range(epochs):
            life_model_ins.observe(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss, dataset)

        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []
        for t in range(task + 1):
            eval_task_cls = args.task_seq[t]
            eval_file_path = f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{eval_task_cls}.pkl'
            eval_data_dict = pickle.load(open(eval_file_path, 'rb'))

            if valid:
                subgraph = eval_data_dict['valid_g'].to(device='cuda:{}'.format(args.gpu))
                test_ids = eval_data_dict['indices'][1] 
            else:
                subgraph = eval_data_dict['test_g'].to(device='cuda:{}'.format(args.gpu))
                test_ids = eval_data_dict['indices'][2] 
            
            ids_per_cls = eval_data_dict['ids_per_cls']
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            ids_per_cls_test = []

            for cls in eval_task_cls:
                nodes_for_cls = (subgraph.ndata['label'] == cls).nonzero(as_tuple=True)[0].tolist()
                ids_per_cls_test.append(nodes_for_cls)

            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            try:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(model, f) # save the best model for each hyperparameter composition
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')

    tp = 0
    return acc_mean, mean_backward, acc_matrix, meanas,tp


def pipeline_class_IL_no_inter_edge_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    
    # 实现不等划分：根据数据集类型设置不同的任务划分
    if 'Arxiv' in args.dataset:
        # Arxiv: 第一个task为10个类，后面task按5个类划分
        cls = [[i for i in range(10)]]  # 第一个任务：0-9类
        for i in range(10, args.n_cls, 5):
            cls.append(list(range(i, min(i + 5, args.n_cls))))
        print(f"Applying unequal class split for Arxiv: 10 base classes, then 5 classes per task.")
    elif 'CoraFull' in args.dataset:
        # CoraFull: 第一个task为20个类，后面task按10个类划分
        cls = [[i for i in range(20)]]  # 第一个任务：0-19类
        for i in range(20, args.n_cls, 10):
            cls.append(list(range(i, min(i + 10, args.n_cls))))
        print(f"Applying unequal class split for CoraFull: 20 base classes, then 10 classes per task.")
    else:
        # 其他数据集使用默认等划分
        cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args,dataset)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far+=len(task_cls)

        file_path = f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl'
        data_dict = pickle.load(open(file_path, 'rb'))
    
        train_g = data_dict['train_g']
        ids_per_cls = data_dict['ids_per_cls']
        train_ids, valid_ids, _ = data_dict['indices'] # 训练阶段只需要 train_ids
        
        subgraph = train_g.to(device='cuda:{}'.format(args.gpu))

        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        # 1. 定义你的目标设备 (通常是从 args 中获取)
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

        # 2. 确保 train_ids 是一个张量
        # 如果 train_ids 本身就是一个 Python 列表，需要先转换成张量
        if isinstance(train_ids, list):
            train_ids = torch.tensor(train_ids, dtype=torch.long)

        # 3. 将图和节点ID都移动到目标设备
        subgraph = subgraph.to(device)
        train_ids = train_ids.to(device)
        # build the dataloader for mini batch training
        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)
        dataloader.enable_cpu_affinity()
        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model, train_ids, ids_per_cls,
                                       dataset)
            else:
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls, dataset)
                torch.cuda.empty_cache()  # tracemalloc.stop()
        if valid:
            para_save_path = f'./para/model_classIL_batch_para_{task}.pth'
            
            # 确保 ./para/ 目录存在 (需要提前 import os)
            os.makedirs(os.path.dirname(para_save_path), exist_ok=True)
            
            # 保存用于正则化的 state_dict
            # 注意：需要保存的模型是 life_model_ins 内部的 self.net
            torch.save(life_model_ins.net.state_dict(), para_save_path)
            
            print(f"Task {task} training finished. Saved regularization parameters to {para_save_path}")

        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        # test
        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []
        for t in range(task + 1):
            eval_task_cls = args.task_seq[t]
            eval_file_path = f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{eval_task_cls}.pkl'
            eval_data_dict = pickle.load(open(eval_file_path, 'rb'))

            if valid:
                subgraph = eval_data_dict['valid_g'].to(device='cuda:{}'.format(args.gpu))
                test_ids = eval_data_dict['indices'][1] 
            else:
                subgraph = eval_data_dict['test_g'].to(device='cuda:{}'.format(args.gpu))
                test_ids = eval_data_dict['indices'][2] 
            
            ids_per_cls = eval_data_dict['ids_per_cls']
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            try:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(model, f) # save the best model for each hyperparameter composition
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    tp=0
    return acc_mean, mean_backward, acc_matrix,meanas,tp


def pipeline_task_IL_inter_edge_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
            f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        test_ids = valid_ids_ if valid else test_ids_
        cls_ids_new = [cls_retain.index(i) for i in task_cls]
        ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]

        train_ids_current_task = []
        for ids in ids_per_cls_train:
            train_ids_current_task.extend(ids)
        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids_current_task, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model,
                                               train_ids_current_task, ids_per_cls_current_task, dataset)
            else:
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids_current_task,
                                               ids_per_cls_current_task, dataset)
                torch.cuda.empty_cache()

        # test
        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []
        for t in range(task + 1):
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            try:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(model, f) # save the best model for each hyperparameter composition
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_inter_edge_minibatch_joint(args, valid=False):
    epochs = args.epochs if valid else 0
    args.method = 'joint_replay_all'
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(
            open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(epochs):
            life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls_all, dataset)

        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        test_ids = valid_ids_ if valid else test_ids_ # whether use validation or test set
        for t, cls_ids_new in enumerate(args.task_seq[0:task+1]):
            # cls_ids_new = args.task_seq[t]
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            if cls_ids_new != args.task_seq[t]:
                print(
                    '-------------------------------sequence is not as default--------------------------------------------------------')
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            try:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(model, f) # save the best model for each hyperparameter composition
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix


def pipeline_task_IL_no_inter_edge_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args, dataset)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []

        for t in range(task + 1):

            file_path = f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl'
            data_dict = pickle.load(open(file_path, 'rb'))
        
            train_g = data_dict['train_g']
            ids_per_cls = data_dict['ids_per_cls']
            train_ids, _, _ = data_dict['indices'] # 训练阶段只需要 train_ids
            
            subgraph = train_g.to(device='cuda:{}'.format(args.gpu))
            ids_per_cls_train_local = []
            for cls in task_cls:
                nodes_for_cls = (subgraph.ndata['label'] == cls).nonzero(as_tuple=True)[0].tolist()
                ids_per_cls_train_local.append(nodes_for_cls)
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls_train_local)

        for epoch in range(epochs):
            life_model_ins.observe_task_IL(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss, dataset)

        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t in range(task + 1):
            eval_task_cls = args.task_seq[t]
            eval_file_path = f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{eval_task_cls}.pkl'
            eval_data_dict = pickle.load(open(eval_file_path, 'rb'))

            if valid:
                subgraph = eval_data_dict['valid_g'].to(device='cuda:{}'.format(args.gpu))
                test_ids = eval_data_dict['indices'][1] 
            else:
                subgraph = eval_data_dict['test_g'].to(device='cuda:{}'.format(args.gpu))
                test_ids = eval_data_dict['indices'][2] 
            
            ids_per_cls = eval_data_dict['ids_per_cls']
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

            ids_per_cls_test = []
            for cls in eval_task_cls:
                nodes_for_cls = (subgraph.ndata['label'] == cls).nonzero(as_tuple=True)[0].tolist()
                ids_per_cls_test.append(nodes_for_cls)

            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')

    print("\n" + "="*70)
    print(" " * 10 + "EFFICIENCY ANALYSIS (Unit: M, measured in float32 equivalents)")
    print("="*70)

    # --- 1. Model Parameter Analysis (Unit: M, 1 param ≈ 1 float32) ---
    print("--- [Model Parameters (M)] ---")
    
    # For joint training, the model is retrained, so we analyze the final instance
    total_params = sum(p.numel() for p in life_model_ins.parameters())
    
    # The generic structure check works for the 'joint' NET wrapper
    if hasattr(life_model_ins, 'net'):
        backbone_params = sum(p.numel() for p in life_model_ins.net.parameters())
        extra_params = total_params - backbone_params
    else:
        backbone_params = total_params
        extra_params = 0
        
    print(f"Σ Total Parameters: {total_params / 1e6:.4f} M")
    print(f"🧠 Backbone Parameters: {backbone_params / 1e6:.4f} M")
    print(f"🔌 Extra Parameters: {extra_params / 1e6:.4f} M")

    # --- 2. Additional Data Storage Analysis (Unit: M, float32 equivalents) ---
    print("\n--- [Additional Data Storage (M)] ---")
    
    additional_data_m = 0.0
    
    # Case for Joint Training: The "buffer" is the entire dataset replayed
    if args.method == 'joint_replay_all':
        print("INFO: Detected Joint Training. Calculating total dataset size as replay cost.")
        total_f32_equiv = 0
        for t in range(args.n_tasks):
            # Reload each data file to measure its size
            file_path = f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl'
            data_dict = pickle.load(open(file_path, 'rb'))
            train_g = data_dict['train_g']
            features = train_g.srcdata['feat']
            labels = train_g.dstdata['label']
            
            # Calculate float32 equivalent size and add to total
            total_f32_equiv += features.numel() * (features.element_size() / 4)
            total_f32_equiv += labels.numel() * (labels.element_size() / 4)
            
        additional_data_m = total_f32_equiv / 1e6
        print(f"📦 Total Replayed Data Storage: {additional_data_m:.4f} M")

    # The block can be extended with elif for other replay methods if needed
    else:
        print("INFO: No additional data storage calculation defined for this method.")

    # --- 3. Total Additional Cost Summary (Unit: M, float32 equivalents) ---
    print("\n--- [Total Additional Cost (M)] ---")
    
    # Direct sum of extra parameter count and data storage's float32 equivalent count
    total_additional_cost_m = (extra_params / 1e6) + additional_data_m
    
    print(f"📈 Total Additional Cost (Extra Params + Data): {total_additional_cost_m:.4f} M")
    
    print("="*70)
    # --- [END] Final Analysis Block ---

    TP = 0
    return acc_mean, mean_backward, acc_matrix, meanas, TP

