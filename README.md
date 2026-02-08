
# Experiment environment
Our experiments are run on the enviroment based on `Python 3.8` with the following packages:

```
pytorch==1.13.0
torch-geometric==2.2.0  # for deploying GNNs.#torch_geometric-2.0.3
ogb==1.3.6  # for obtaining arxiv and prodcuts datasets.
progressbar2==4.2.0  # for visulasing the process of the condensation.
```

# Usage
To reproduce the results of Table 1 (classIL setting), please run the `table2.sh` in the `srcripts` folder:
```
bash run.sh
```
