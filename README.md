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