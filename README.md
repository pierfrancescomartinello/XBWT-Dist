# mp3treesim_supp

Supplementary repository for: https://github.com/AlgoLab/mp3treesim

## Setup
```bash
pip3 install mp3treesim

git clone --recursive https://github.com/AlgoLab/mp3treesim_supp.git
cd mp3treesim_supp
cd measures/MLTED
g++ -std=c++11 main.cpp -o MLTED
cd ../..
```

## Requirements
- Python 3
- snakemake
- numpy
- pandas
- networkx
- matplotlib
- seaborn

## Run node ascension experiment
```bash
cd simulated/node_ascend
chmod +x run_all.sh
./run_all.sh
```

This will produce the file `simulated/node_ascend/plot.pdf`.

## Run label-duplication experiment
```bash
snakemake -s run_duplication.smk run_all
```
This will produce CSV of the heatmaps for every tool in the format of `simulated/duplication/[TOOL]_pbp[1-5]/results.csv`

## Run experimental configurations 1, 2 and clustering
```bash
snakemake -s run_configs.smk --configfile config1.yaml
snakemake -s run_configs.smk --configfile config2.yaml
snakemake -s run_configs.smk --configfile cluster.yaml
```

This will produce CSV of the heatmaps for every tool in the format of `simulated/config[N]/[TOOL]/results.csv` and `simulated/cluster/[TOOL]/results.csv`.

## Run on real data
```bash
cd real
chomd +x run_exps.sh
./run_exps.sh
```

## Simulation creation and tree perturbation

The perturbation script is available in `script/generate_tree.py`.

```
usage: generate_tree.py [-h] [-n N] [-l L] [-s S] [-f] [-o OUT]

MP3-treesim tree generation tool

optional arguments:
  -h, --help         show this help message and exit
  -n N, --nodes N    Total number of nodes (internal nodes and leafs)
                     [Default: 10]
  -l L, --labels L   Total number of labels [Default: 10]
  -s S, --sons S     Maximum number of sons for node [Default: 3]
  -f, --full         Generate a complete tree [Default: false]
  -o OUT, --out OUT  Output prefix [Default: out]

```

The perturbation script is available in `script/perturbation.py`.

```
usage: perturbation.py [-h] -t TREE [--labelswap LABELSWAP]
                       [--noderemove NODEREMOVE] [--labelremove LABELREMOVE]
                       [--labelduplication LABELDUPLICATION]
                       [--nodeswap NODESWAP] --out OUT
                       [--totoperations TOTOPERATIONS]

MP3-treesim tree perturbation tool

optional arguments:
  -h, --help            show this help message and exit
  -t TREE, --tree TREE  Path to the tree
  --labelswap LABELSWAP
                        Number/probability of label swaps to produce
  --noderemove NODEREMOVE
                        Number/probability of nodes to remove
  --labelremove LABELREMOVE
                        Number/probability of labels to remove
  --labelduplication LABELDUPLICATION
                        Number/probability of labels to duplicate
  --nodeswap NODESWAP   Number/probability of nodes to swap
  --out OUT             Path to output file
  --totoperations TOTOPERATIONS
                        Number of total operations. If set to -1 (default) all
                        operations will be executed.

```