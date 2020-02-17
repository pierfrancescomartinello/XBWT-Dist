import sys, math, itertools

def parse(tree_path):
    all_labels, labels_by_node = {}, []
    i = 0
    for line in open(tree_path):
        line = line.strip('\n')
        if "[label=" in line:
            labels = line.split('"')[1].split(',')
            labels_by_node.append(labels)
            for label in labels:
                all_labels[label] = all_labels[label]+1 if label in all_labels else 1
            i+=1
    return all_labels, labels_by_node

def intersection(dict_1, dict_2):
    hits = 0
    nohits = 0
    for l1,w1 in dict_1.items():
        if l1 in dict_2:
            hits += min(w1, dict_2[l1])
            pass
        else:
            nohits += 1
    return hits, nohits

def allpairs(labels_by_node_1, labels_by_node_2):
    hits, nohits = 0, 0
    for node_labels_1 in labels_by_node_1:
        for l1,l2 in itertools.combinations(node_labels_1, 2):
            hit_flag = False
            for node_labels_2 in labels_by_node_2:
                if l1 in node_labels_2 and l2 in node_labels_2:
                    hit_flag = True
                    break
            if hit_flag:
                hits += 1
            else:
                nohits += 1
    return hits, nohits

def main():
    tree1_path = sys.argv[1]
    tree2_path = sys.argv[2]

    all_labels_1, labels_by_node_1 = parse(tree1_path)
    all_labels_2, labels_by_node_2 = parse(tree2_path)

    both_1, only_1 = intersection(all_labels_1, all_labels_2)
    both_2, only_2 = intersection(all_labels_2, all_labels_1)
    print("Only in 1:", only_1, sep='\t')
    print("Intersection:", both_1, sep='\t')
    print("Only in 2:", only_2, sep='\t')
    print("")

    h1,nh1 = allpairs(labels_by_node_1, labels_by_node_2)
    h2,nh2 = allpairs(labels_by_node_2, labels_by_node_1)

    # print([len(L) for L in labels_by_node_1])
    combs_1 = [int(math.factorial(len(L))/(2*(math.factorial(len(L)-2)))) for L in labels_by_node_1]
    # print(combs_1, " = ", sum(combs_1))
    print("Pairs in tree1:", h1+nh1, sep='\t')
    
    # print("")
    # print([len(L) for L in labels_by_node_2])
    combs_2 = [int(math.factorial(len(L))/(2*(math.factorial(len(L)-2)))) for L in labels_by_node_2]
    # print(combs_2, " = ", sum(combs_2))
    print("Pairs in tree2:", h2+nh2, sep='\t')
    print("Shared pairs:", h1, sep='\t')

if __name__ == "__main__":
    main()
