import sys

import graphviz
from graphviz import Digraph

def main():
    fpath = sys.argv[1]

    nodes = {}
    edges = set()
    snv2label = {}
    # "Nodes:" -> nodes -> "\n" -> skip -> "****Tree" -> edges -> "Error" -> skip -> SNV info -> labels -> EOF
    mode = "skip"
    for line in open(fpath):
        if line.startswith("Nodes:"):
            mode = "nodes"
            continue
        elif line.startswith("****Tree"):
            mode = "edges"
            continue
        elif line.startswith("SNV info"):
            mode = "labels"
            continue
        elif line == "\n" or line.startswith("Error"):
            mode = "skip"
            continue

        if mode == "nodes":
            idx, _, _, *snvs = line.strip().split('\t')
            nodes[idx] = snvs
        elif mode == "edges":
            idx1, _, idx2 = line.strip().split(' ')
            if idx1 != "0":
                edges.add((idx1, idx2))
        elif mode == "labels":
            snv, chrom, pos, _ = line.strip().split(' ')
            snv = snv[:-1]
            snv2label[snv] = chrom + ':' + pos

    g = Digraph('G')
    for node,snvs in nodes.items():
        label = ""
        for snv in snvs:
            label+= snv2label[snv] + ","
        label = label[:-1]
        g.node(node, label)

    for (idx1, idx2) in edges:
        g.edge(idx1, idx2)

    print(g)
if __name__ == "__main__":
    main()
