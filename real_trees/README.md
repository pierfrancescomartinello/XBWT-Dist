# Experiments on real trees

This folder contains the instruction to reproduce the experiments on real trees.

### Input data
##### Gerlinger et al. Nature Genetics (2014)
* we built the tree for patient RMH002 from [1] (Figure 3 and Supplementary Table 6)
* we built the LICHeE tree from the [.txt file](https://github.com/zhero9/MIPUP/blob/master/data/results/ccRCC/lichee/RMH002.txt.trees.txt) via the `lichee2gv.py`
* we built the MIPUP tree from the [.csv](https://github.com/zhero9/MIPUP/blob/master/data/results/ccRCC/ipd/RMH002_filtered_ipd_columns.csv) and [.pdf](https://github.com/zhero9/MIPUP/blob/master/data/results/ccRCC/ipd/RMH002_filtered_ipd_tree.dot.pdf) files

##### Eirew et al. Nature (2015)
* we built the tree for case SA501 from [2] (Supplementary Figure 12) as done in [3]
* we built the LICHeE tree from the [.txt file](https://github.com/zhero9/MIPUP/blob/master/data/results/xenoengrafment/lichee/SA501-X1X2X4-noLCU.txt.trees.txt) via the `lichee2gv.py`
* we built the MIPUP tree from the [.csv](https://github.com/zhero9/MIPUP/blob/master/data/results/xenoengrafment/ipd/SA501-X1X2X4-noLCU_filtered_ipd_columns.csv) and [.pdf](https://github.com/zhero9/MIPUP/blob/master/data/results/xenoengrafment/ipd/SA501-X1X2X4-noLCU_filtered_ipd_tree.dot.pdf) files
* we built the edge case tree by collapsing all nodes from MIPUP tree in a single node

### How to
Convert the LICHeE txt to a .gv file (this requires [graphviz](https://pypi.org/project/graphviz/)):
```bash
python3 lichee2gv.py /path/to/txt > /path/to/gv
```
Run the experiments:
```bash
bash run_exps.sh
```
Use the proxy distance between two trees:
```bash
python3 check_proxy.py /path/to/tree1 /path/to/tree2
```

### Output data
##### Gerlinger et al. Nature Genetics (2014)
The pairwise similarity between each pair of trees is stored as a set of `.csv` files in the folder `./output/gerlinger` (one file for each considered measure).

##### Eirew et al. Nature (2015)
The pairwise similarity between each pair of trees is stored as a set of `.csv` files in the folder `./output/eirew` (one file for each considered measure).

The pairwise similarity with the edge tree can be found in the folder `./output/eirew_edge`.

###### References
[1] Gerlinger et al. Genomic architecture and evolution of clear cell renal cell carcinomas defined by multiregion sequencing. Nature Genetics (2014)
[2] Eirew et al. Dynamics of genomic clones in breast cancer patient xenografts at single-cell resolution. Nature (2015)
[3] DiNardo et al. Distance measures for tumor evolutionary trees. Bioinformatics (2019)
