import pickle
import matplotlib.pyplot as plt
import numpy as np


def AP_err(performance_matrices):
    # given a list of performence matrices, return the APs the errors (range), and std
    n_tasks = performance_matrices[0].shape[0]
    performance_means_all_repeats = np.stack([[np.mean(m[i,0:i+1]) for i in range(n_tasks)] for m in performance_matrices])
    performance_means = np.mean(performance_means_all_repeats, axis=0)
    err_all = performance_means_all_repeats - performance_means
    std = performance_means_all_repeats.std(0)
    err_plus = err_all.max(axis=0)
    err_minus = err_all.min(axis=0).__abs__()
    err = np.stack([err_minus, err_plus])
    return performance_means, err, std

def AF(acc_matrix):
    # given a acc matrix, return AF
    n_tasks = acc_matrix.shape[0]
    backward = []
    for t in range(n_tasks - 1):
        b = acc_matrix[n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(b)
    return np.mean(backward)

def AF_err(performance_matrices):
    # given a list of acc matrices, return the AMs and the errors
    AF_all_repeats = []
    for m in performance_matrices:
        AF_all_repeats.append(AF(m))
    AF_mean = np.mean(AF_all_repeats)
    AF_all_repeats = np.stack(AF_all_repeats)
    err_all = AF_all_repeats - AF_mean
    std = AF_all_repeats.std(0)
    err_plus = err_all.max(axis=0)
    err_minus = err_all.min(axis=0).__abs__()
    err = np.stack([err_minus, err_plus])
    return AF_mean, err, std

def show_performance_matrices(result_path, save_fig_name=None, multiplier=1.0):
    """
    The function to visualize the performance matrix.

    :param result_path: The path to the experimental result
    :param save_fig_name: If specified, the generated visualization will be stored with the specified name under the directory "./results/figures"
    """
    # visualize the acc matrices
    print(result_path)
    fig, ax = plt.subplots()
    performance_matrices = pickle.load(open(result_path, 'rb'))
    acc_matrix_mean = np.mean(performance_matrices, axis=0)
    mask = np.tri(acc_matrix_mean.shape[0], k=-1).T
    acc_matrix_mean = np.ma.array(acc_matrix_mean, mask=mask) * multiplier
    im = plt.imshow(acc_matrix_mean)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel('$\mathrm{Tasks}$')
    plt.ylabel('$\mathrm{Tasks}$')
    plt.clim(vmin=0, vmax=100)
    cbar = fig.colorbar(im, ticks=[0, 50, 100])  # , fontsize = 15)
    cbar.ax.tick_params()

    if save_fig_name is not None:
        plt.savefig(f'./results/figures/{save_fig_name}_performance_matrix', bbox_inches='tight')
    plt.show()

def show_learning_curve(result_path, save_fig_name=None):
    """
        The function to visualize the dynamics of AP.

        :param result_path: The path to the experimental result
        :param save_fig_name: If specified, the generated visualization will be stored with the specified name under the directory "./results/figures"
        """
    #to draw AP against buffer task with different methods
    print(result_path)
    performance_matrices = pickle.load(open(result_path, 'rb'))
    performance_mean, err, _ = AP_err(performance_matrices)
    x = list(range(len(performance_mean)))
    plt.errorbar(x, performance_mean)
    if save_fig_name is not None:
        plt.savefig(
            f'./results/figures/{save_fig_name}_learning_curve', bbox_inches='tight')
    plt.show()

def show_final_APAF(result_path, GCGL=False):
    """
        The function to show the final AP and AF. Output are orgnized in a LaTex firendly way.

        :param result_path: The path to the experimental result
        """
    #show the final AP and AF
    performance_matrices = pickle.load(open(result_path, 'rb'))
    performance_mean, err, std_am = AP_err(performance_matrices)

    # AF
    AF_mean, err_AF, std_AF = AF_err(performance_matrices)

    if not GCGL:
        output_str=r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1], std_am[-1], AF_mean, std_AF)
        print(r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1], std_am[-1], AF_mean, std_AF))
    else:
        output_str=r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1]*100, std_am[-1]*100, AF_mean*100, std_AF*100)
        print(r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1]*100, std_am[-1]*100, AF_mean*100, std_AF*100)) # convert GCGL results to percentages
    return output_str


def show_final_APAF_f1(result_path):
    #show the final AP and AF for results in the form of f1 score
    performance_matrices = pickle.load(open(result_path, 'rb'))
    performance_mean, err, std_am = AP_err(performance_matrices)

    # AF
    AF_mean, err_AF, std_AF = AF_err(performance_matrices)

    output_str = r'{:.3f}$\pm${:.3f}&{:.3f}$\pm${:.3f}'.format(performance_mean[-1], std_am[-1], AF_mean, std_AF)
    print(r'{:.3f}$\pm${:.3f}&{:.3f}$\pm${:.3f}'.format(performance_mean[-1], std_am[-1], AF_mean, std_AF))
    return output_str

def get_result_file_name(args):
    if args.method == "ergnn":
        result_name = f"_MFPlus"
    elif args.method == "ssm":
        result_name = f"_random"
    elif "cat" in args.method:
        cat_args = eval(args.cat_args)
        result_name = f"_{cat_args['feat_init']}_feat_{cat_args['feat_lr']}_{cat_args['n_encoders']}_{cat_args['n_layers']}_layer_{cat_args['hid_dim']}_GCN_hop_{cat_args['hop']}"
        if cat_args['activation'] == False:
            result_name += "_nonact"
    else:
        result_name = f""
    return f"{args.dataset}_{args.budget}_{args.method}" + result_name

def print_performance_matrix(performace_matrix, m_update):
    for k in range(performace_matrix.shape[0]):
        accs = []
        for k_ in range(k + 1):
            acc = performace_matrix[k, k_]
            accs.append(acc)
            if m_update == "all":
                print(f"T{k_} {acc:.2f}",end="|")
            elif m_update == "onlyCurrent":
                if k == k_:
                    print(f"T{k_} {acc:.2f}",end="|")
        print(f"AP: {sum(accs) / len(accs):.2f}")


import os
import csv
from datetime import datetime
import json # 用于序列化矩阵

def log_experiment_results(args, run_time, max_memory,  ap, af, acc_matrix,running_avg_accs,hyp_params_str):
    """
    将单次实验的结果记录到 CSV 文件中。

    Args:
        args: 包含数据集和方法等信息的参数对象。
        run_time (float): 实验运行的总时间（秒）。
        max_memory (float): 峰值 GPU 内存占用（MB）。
        trainable_params (int): 模型的可训练参数数量。
        ap (float): 平均性能 (Average Performance)。
        af (float): 平均遗忘 (Average Forgetting)。
        acc_matrix (np.ndarray): 性能矩阵。
        hyp_params_str (str): 当前使用的超参数字符串。
    """
    # 1. 定义日志文件名和路径
    log_filename = f"{args.dataset}_{args.method}.csv"
    log_path = os.path.join(args.result_path, log_filename)
    
    # 2. 准备要写入的数据行
    # 将矩阵序列化为 JSON 字符串以便存储在单个单元格中
    acc_matrix_str = json.dumps(acc_matrix.tolist())
    running_avg_accs_str = json.dumps(running_avg_accs)
    data_row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': args.dataset,
        'method': args.method,
        'backbone': args.backbone,
        'hyperparameters': hyp_params_str,
        'run_time_s': f"{run_time:.2f}",
        'max_memory_mb': f"{max_memory:.2f}",
        'AP': f"{ap:.4f}",
        'AF': f"{af:.4f}",
        'acc_matrix': acc_matrix_str,
        'running_avg_accs': running_avg_accs_str
    }
    
    # 3. 检查文件是否存在，如果不存在则写入表头
    file_exists = os.path.isfile(log_path)
    header = data_row.keys()
    
    # 4. 以追加模式写入数据
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()  # 写入表头
        writer.writerow(data_row) # 写入数据