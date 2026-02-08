for dataset in Reddit-CL #'Products-CL, Reddit-CL, Arxiv-CL, CoraFull-CL, Cora-CL, Citeseer-CL'
do # sgreplay 
python train.py --dataset $dataset --method TAAM --backbone SGC --gpu 1 --ILmode classIL --inter-task-edges False --minibatch False
done


# method choices=["bare", 'lwf', 'gem',  'tpp', 'ewc', 'mas', 'twp', 'cat', 'jointtrain','sgreplay','taam(backbone : SGC)', 'tpp(backbone : SGC)','ergnn', 'joint','Joint','TEM(backbone : CustomDecoupledSGC)'] 