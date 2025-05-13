# -*- coding: utf-8 -*-
"""
An implementation of the XBW-Trasform 

@author: Danilo Dolce
"""

import time
import math
import numpy as np
import copy
import random
from natsort import natsorted
import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv
import matplotlib
from matplotlib.pyplot import figure
from tqdm import tqdm
import os
import svgling
import nltk
if os.name == 'nt':
    os.environ["PATH"] = os.path.join("C:\\Program Files\\Graphviz\\bin", os.getenv("PATH"))

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
            "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

#alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

#alphabet = ["A", "B", "C", "D", "E"]


def editDistDP(str1, str2, m, n):
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):

            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j

            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i

            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace

    return dp[m][n]

def jaccardDist(str1, str2):
    labs1 = sorted(str1)
    labs2 = sorted(str2)
    labsintersect, labsunion = 0, 0
    idx1, idx2 = 0, 0
    while idx1 < len(str1) and idx2 < len(str2):
        if labs1[idx1] == labs2[idx2]:
            labsintersect+=1
            labsunion+=1
            idx1+=1
            idx2+=1
        elif labs1[idx1] < labs2[idx2]:
            labsunion +=1
            idx1+=1
        else:
            labsunion +=1
            idx2+=1
    labsunion += len(str1)+len(str2)-idx1-idx2
    return 1 - labsintersect/labsunion if labsunion > 0 else 0


def Compute_Partition(m_lcp):
    partitions = []  # Partizioni che hanno sia nodi di T1 che di T2
    other_partitions = []  # Partizioni che hanno solo nodi di T1 o di T2

    # La prima partizione corrisponde alle radici dei due alberi
    partitions.append([0, 1])

    # Setto come lcp minimo quello presente in seconda posizione
    min_lcp = m_lcp[2][4]

    # Posizione del minimo lcp
    pos_min_lcp = 2

    # Indice di partenza
    i = 2

    # Dizionario che traccia l'albero opposto
    trees_opp = {}
    trees_opp["T1"] = "T2"
    trees_opp["T2"] = "T1"

    while i < len(m_lcp):
        start = i
        # Albero apposto
        temp = trees_opp[m_lcp[i][0]]
        tree_type = []
        #print("Ricomincio", i)
        flag = True
        i = i+1
        while flag:
            # Se sono giunto alla fine
            if i >= len(m_lcp):
                # Controllo se l'ultimo nodo ha LCP > di 0 e se l'albero di appartenenza è opposto a quello di partenza
                if m_lcp[i-1][4] > 0 and m_lcp[i-1][0] != trees_opp[temp]:
                    partitions.append([start, i-1])
                else:
                    other_partitions.append([start, i-1])
                flag = False
            # Ho trovato il minimo
            # print(i)
            elif m_lcp[i][4] <= min_lcp:
                #print("->: ", m_lcp[i][4], i)
                min_lcp = m_lcp[i][4]
                pos_min_lcp = i
                # Ho trovato la partizione (T1 e T2)
                if m_lcp[i-1][0] == trees_opp[temp]:
                    other_partitions.append([start, i-1])
                    flag = False
                else:
                    partitions.append([start, i-1])
                    flag = False
            else:
                # Fin quando non trovo il minimo incremento l'indice
                i += 1
    return partitions, other_partitions


def Compute_Partition2(m_lcp):
    partitions = []  # Partizioni che hanno sia nodi di T1 che di T2
    other_partitions = []  # Partizioni che hanno solo nodi di T1 o di T2

    # La prima partizione corrisponde alle radici dei due alberi
    partitions.append([0, 1])

    # Setto come lcp minimo quello presente in seconda posizione
    min_lcp = m_lcp[2][4]

    # Posizione del minimo lcp
    pos_min_lcp = 2

    # Indice di partenza
    i = 2

    # Dizionario che traccia l'albero opposto
    trees_opp = {}
    trees_opp["T1"] = "T2"
    trees_opp["T2"] = "T1"

    while i < len(m_lcp):
        start = i
        # Albero apposto
        temp = trees_opp[m_lcp[i][0]]
        if i+1 == len(m_lcp):
            #print("Sono qui in mlcp")
            other_partitions.append([start, i])
            break
        i = i+1
        if m_lcp[i][4] > min_lcp:
            max_lcp = m_lcp[i][4]
            flag = True
            i += 1
            while flag:
                # Se sono giunto alla fine
                if i >= len(m_lcp):
                    # Controllo se l'ultimo nodo ha LCP > di 0 e se l'albero di appartenenza è opposto a quello di partenza
                    if m_lcp[i-1][0] == temp:
                        partitions.append([start, i-1])
                    else:
                        #print("Ho add partizione indipendente: ", [start, i-1], temp)
                        other_partitions.append([start, i-1])
                    flag = False
                elif m_lcp[i][4] < max_lcp:
                    min_lcp = m_lcp[i][4]
                    if m_lcp[i-1][0] == trees_opp[temp]:
                        #print("Ho aggiunto la partizione indipendente; ", [start, i-1])
                        other_partitions.append([start, i-1])
                        flag = False
                    else:
                        partitions.append([start, i-1])
                        flag = False
                else:
                    # Fin quando non trovo il minimo incremento l'indice
                    i += 1
        else:
            #print("Ho aggiunto un'altra partizione indipendente")
            other_partitions.append([start, start])
            min_lcp = m_lcp[i][4]

    return partitions, other_partitions


def merge_lcp(m_array, lcp):
    for i in range(len(m_array)):
        m_array[i].append(lcp[i])
    return m_array


def compute_lcp_array(m_array):
    lcp = np.zeros(len(m_array), dtype="int")
    x = 1
    while x != len(m_array):
        count = 0
        temp = m_array[x][3]
        j = 0
        for i in range(len(m_array[x-1][3])):
            if j != len(temp) and temp[i] == m_array[x-1][3][i]:
                count += 1
                j += 1
            else:
                break
        lcp[x] = count
        x += 1
    return lcp


def Xbwt_Edit_Distance(tree1, tree2):
    xbwt1 = XBWT(tree1)
    IntNodes1, IntNodes_Pos_Sort1 = xbwt1.pathSort(xbwt1.getTree())

    #print(IntNodes1, end="\n\n")
    #print(IntNodes_Pos_Sort1, end="\n\n")

    xbw1 = xbwt1.Compute_XBWT(IntNodes1, IntNodes_Pos_Sort1)

    # print(xbw1)

    S_pi1 = xbwt1.Compute_Spi_Sort(IntNodes1, IntNodes_Pos_Sort1)

    ai1 = []
    for i in range(0, len(IntNodes1)):
        ai1.append(["T1", xbw1[i][0], xbw1[i][1], S_pi1[i]])

    xbwt2 = XBWT(tree2)
    IntNodes2, IntNodes_Pos_Sort2 = xbwt2.pathSort(xbwt2.getTree())
    xbw2 = xbwt2.Compute_XBWT(IntNodes2, IntNodes_Pos_Sort2)
    S_pi2 = xbwt2.Compute_Spi_Sort(IntNodes2, IntNodes_Pos_Sort2)

    ai2 = []
    for i in range(0, len(IntNodes2)):
        ai2.append(["T2", xbw2[i][0], xbw2[i][1], S_pi2[i]])

    merged = sorted(ai1+ai2, key=lambda elem: elem[3])

    lcp = compute_lcp_array(merged)

    m_lcp = merge_lcp(merged, lcp)

    """
    print("TREE, S_LAST, S_ALPHA, S_PI, LCP", end="\n\n")
    j = 0
    for i in m_lcp:
        print(j, i)
        j+=1
    """

    partitions, other_partitions = Compute_Partition(m_lcp)

    partitions_strings = {}
    for t in partitions:
        partitions_strings[str(t)] = {}
        partitions_strings[str(t)]["T1"] = []
        partitions_strings[str(t)]["T2"] = []
        for i in range(t[0], t[1]+1):
            partitions_strings[str(t)][m_lcp[i][0]].append(m_lcp[i][2][0])
    dists = {}
    total = 0
    for t in partitions:
        str1 = "".join(partitions_strings[str(t)]["T1"])
        str2 = "".join(partitions_strings[str(t)]["T2"])
        dist = jaccardDist(str1, str2)
        total = total+dist
        dists[str(t)] = dist

    for p in other_partitions:
        dists[str(p)] = p[1]+1-p[0]
        total += p[1]+1-p[0]

    #print(dists, end="\n\n")
    return total


def Xbwt_Edit_Distance2(tree1, tree2):
    xbwt1 = XBWT(tree1)
    IntNodes1, IntNodes_Pos_Sort1 = xbwt1.pathSort(xbwt1.getTree())

    #print(IntNodes1, end="\n\n")
    #print(IntNodes_Pos_Sort1, end="\n\n")

    xbw1 = xbwt1.Compute_XBWT(IntNodes1, IntNodes_Pos_Sort1)

    # print(xbw1)

    S_pi1 = xbwt1.Compute_Spi_Sort(IntNodes1, IntNodes_Pos_Sort1)

    ai1 = []
    for i in range(0, len(IntNodes1)):
        ai1.append(["T1", xbw1[i][0], xbw1[i][1], S_pi1[i]])

    print(ai1, end="\n\n")

    xbwt2 = XBWT(tree2)
    IntNodes2, IntNodes_Pos_Sort2 = xbwt2.pathSort(xbwt2.getTree())
    xbw2 = xbwt2.Compute_XBWT(IntNodes2, IntNodes_Pos_Sort2)
    S_pi2 = xbwt2.Compute_Spi_Sort(IntNodes2, IntNodes_Pos_Sort2)

    ai2 = []
    for i in range(0, len(IntNodes2)):
        ai2.append(["T2", xbw2[i][0], xbw2[i][1], S_pi2[i]])

    print(ai2)

    merged = sorted(ai1+ai2, key=lambda elem: elem[3])

    lcp = compute_lcp_array(merged)

    m_lcp = merge_lcp(merged, lcp)

    print("\nTREE, S_LAST, S_ALPHA, S_PI, LCP", end="\n\n")
    j = 0
    for i in m_lcp:
        print(j, i)
        j += 1

    print("\n")
    partitions, other_partitions = Compute_Partition2(m_lcp)
    print("\n", other_partitions)

    partitions_strings = {}
    for t in partitions:
        partitions_strings[str(t)] = {}
        partitions_strings[str(t)]["T1"] = []
        partitions_strings[str(t)]["T2"] = []
        for i in range(t[0], t[1]+1):
            partitions_strings[str(t)][m_lcp[i][0]].append(m_lcp[i][2][0])
    dists = {}
    total = 0
    for t in partitions:
        str1 = "".join(partitions_strings[str(t)]["T1"])
        str2 = "".join(partitions_strings[str(t)]["T2"])
        dist = jaccardDist(str1, str2)
        total = total+dist
        dists[str(t)] = dist
    print(partitions_strings)

    for p in other_partitions:
        dists[str(p)] = p[1]+1-p[0]
        total += p[1]+1-p[0]

    #print(dists, end="\n\n")
    return total


def mergeXBWT(tree1, tree2, path):
    xbwt1 = XBWT(tree1)
    IntNodes1, IntNodes_Pos_Sort1 = xbwt1.pathSort(xbwt1.getTree())

    #print(IntNodes1, end="\n\n")
    #print(IntNodes_Pos_Sort1, end="\n\n")

    xbw1 = xbwt1.Compute_XBWT(IntNodes1, IntNodes_Pos_Sort1)

    # print(xbw1)

    S_pi1 = xbwt1.Compute_Spi_Sort(IntNodes1, IntNodes_Pos_Sort1)

    ai1 = []
    for i in range(0, len(IntNodes1)):
        ai1.append(["T1", xbw1[i][0], xbw1[i][1], S_pi1[i]])

    xbwt2 = XBWT(tree2)
    IntNodes2, IntNodes_Pos_Sort2 = xbwt2.pathSort(xbwt2.getTree())
    xbw2 = xbwt2.Compute_XBWT(IntNodes2, IntNodes_Pos_Sort2)
    S_pi2 = xbwt2.Compute_Spi_Sort(IntNodes2, IntNodes_Pos_Sort2)

    ai2 = []
    for i in range(0, len(IntNodes2)):
        ai2.append(["T2", xbw2[i][0], xbw2[i][1], S_pi2[i]])

    merged = sorted(ai1+ai2, key=lambda elem: elem[3])

    lcp = compute_lcp_array(merged)

    m_lcp = merge_lcp(merged, lcp)

    f = open(os.path.join(path,"Merge.txt"), "w+")
    f.write("INDEX, TREE, S_LAST, S_ALPHA, S_PI, LCP\n\n")
    j = 0
    for i in m_lcp:
        f.write(str(j)+" "+str(i)+"\n")
        j += 1
    f.close()


def get_all_subtree(T):
    nodes = T.getNodes()
    subtrees = []
    for node in nodes:
        subtree = get_subtree(node)
        subtrees.append(subtree)
    return subtrees


def get_subtree(start_node):
    Q = []
    Q.append(start_node)
    subtree = []
    while len(Q) > 0:
        for n in Q[0].getChildren():
            subtree.append((Q[0], n))
            Q.append(n)
        Q.pop(0)
    return subtree


def Generate_Random_Tree(alphabet):
    nodes = copy.deepcopy(alphabet)
    random.shuffle(nodes)
    print(nodes)
    leafs = np.ones(len(alphabet))
    #print("Nodi: ", nodes)
    i = 0
    for node in nodes:
        nodes[i] = Node(nodes[i])
        i += 1
    n = len(nodes)
    tree = Tree()
    tree.insert(nodes[0], None)
    #print("Radice: ", nodes[0].getLabel())
    n -= 1
    j = 1
    v = 0
    while j < len(nodes):
        nchild = random.randint(1, 3)
        #print("Numero figli: ", nchild)
        if nchild > n:
            nchild = n
        for i in range(j, j+nchild):
            #print("Sto inserendo: ", nodes[i].getLabel(), nodes[v].getLabel())
            tree.insert(nodes[i], nodes[v])
        leafs[v] = 0
        v += 1
        j = j+nchild
        n -= nchild
    # Aggiungo i dollari per le foglie
    # print(leafs)
    firstLeafIndex = list(leafs).index(1)
    for i in range(firstLeafIndex, len(nodes)):
        tree.insert(Node("$"), nodes[i])
    return tree


def Remove_Subtrees(T, maxRem, path):
    T0 = copy.deepcopy(T)
    # La ricostruzione funziona ma non funziona il preordine
    distances = []
    removals_array = []
    nremovals = random.randint(1, maxRem)
    nodeTemp = []
    size_sub_rem = []
    #print("Numero random estratto: ", nremovals)
    subtrees = get_all_subtree(T)
    sbt_dim_real = []
    for st in subtrees:
        dimTmp = 0
        if len(st) >= 1:
            for e in st:
                #print((e[0].getLabel(), e[1].getLabel()))
                #print((e[0].getLabel(), e[1].getLabel()))
                """
                if e[1].getLabel() != "$":
                    dimTmp+=1
                """
                dimTmp += 1
            # print("\n\n")
            sbt_dim_real.append(dimTmp+1)
        else:
            sbt_dim_real.append(0)
    # Non considerare il primo sottoalbero (cioè quello della radice)
    subtrees_dim = [len(e) for e in subtrees]

    #print("Dimensioni sottoalberi 1:", subtrees_dim)
    firstEmptySubtreeIndex = list(subtrees_dim).index(0)
    # print(firstEmptySubtreeIndex)
    posSubtree = random.randint(1, firstEmptySubtreeIndex-1)
    f = open(os.path.join(path,"SUBTREES_REMOVED.txt"), "w+")
    f.write("SOTTOALBERI RIMOSSI\n\n")
    f.write(str(subtrees[posSubtree][0][0].getLabel())+"\n")
    for e in subtrees[posSubtree]:
        print(e[0].getLabel(), e[1].getLabel())
    print("Nodi rimossi")
    #print("Subtrees_dim 1: ", subtrees_dim)
    #print("Subtrees_dim 2: ", sbt_dim_real)
    #print("Possubtree: ", posSubtree)
    size_sub_rem.append(sbt_dim_real[posSubtree])
    r = 0
    # print(edges)
    removals = 0
    newTree = None
    while r < nremovals:
        edgesToRemove = []
        edges = T.getEdges()
        #print("Rimozione: ", r+1)
        edgesToRemove.append(
            (subtrees[posSubtree][0][0].getParent(), subtrees[posSubtree][0][0]))
        #print("Eccolo: ", edgesToRemove)
        for st in subtrees[posSubtree]:
            edgesToRemove.append(st)
        newEdges = [(x[0].getLabel(), x[1].getLabel())
                    for x in edges if x not in edgesToRemove]
        #print("Archi da eliminare: ")
        # for i in edgesToRemove:
        #print("Arco:", i[0].getLabel(), i[1].getLabel())
        # Inserisco la coppia genitore-radice sottoalbero
        dictNodes = {}
        newEdges2 = []
        for e in newEdges:
            temp = None
            if e[0] in dictNodes.keys():
                temp = dictNodes[e[0]]
            else:
                dictNodes[e[0]] = Node(e[0])
                temp = dictNodes[e[0]]
            if e[1] == "$":
                newEdges2.append((temp, Node(e[1])))
            elif e[1] in dictNodes.keys():
                newEdges2.append((temp, dictNodes[e[1]]))
            else:
                dictNodes[e[1]] = Node(e[1])
                newEdges2.append((temp, dictNodes[e[1]]))
        newTree = Tree()
        root = T.getRoot().getLabel()
        if root in dictNodes.keys():
            newTree.insert(dictNodes[root], None)
        else:
            newTree.insert(Node(root), None)
        # Inserisco i nuovi archi all'albero
        for e in newEdges2:
            #print(e, e[1].getLabel(), e[0].getLabel())
            newTree.insert(e[1], e[0])
        r += 1
        removals += 1
        # Aggiungo i dollari alle foglie
        for node in newTree.getNodes():
            #print("Nodo: ", node.getLabel(), [n.getLabel() for n in node.getChildren()])
            if len(node.getChildren()) == 0 and node.getLabel() != "$":
                nodeTemp.append(node.getLabel())
                newTree.insert(Node('$'), node)
        removals_array.append(removals)
        distances.append(Xbwt_Edit_Distance(T0, newTree))
        if r < nremovals:
            T = copy.deepcopy(newTree)
            #print("Preordine 1:", T.preorder(T.getRoot()))
            subtrees = get_all_subtree(T)
            sbt_dim_real = []
            for st in subtrees:
                dimTmp = 0
                if len(st) >= 1:
                    for e in st:
                        #print((e[0].getLabel(), e[1].getLabel()))
                        """
                        if e[1].getLabel() != "$":
                            dimTmp+=1
                        """
                        dimTmp += 1
                    # print("\n")
                    sbt_dim_real.append(dimTmp+1)
                else:
                    sbt_dim_real.append(0)
            # Non considerare il primo sottoalbero (cioè quello della radice)
            subtrees_dim = [len(e) for e in subtrees]
            #print("Dimensioni sottoalberi: ", subtrees_dim)
            # print(subtrees_dim)
            if len(subtrees_dim) != 2:
                firstEmptySubtreeIndex = list(subtrees_dim).index(0)
                # print(firstEmptySubtreeIndex)
                posSubtree = random.randint(1, firstEmptySubtreeIndex-1)
                dec = 0
                f.write(str(subtrees[posSubtree][0][0].getLabel())+"\n")
                for e in subtrees[posSubtree]:
                    if e[1].getLabel() == '$' and e[1].getParent().getLabel() in nodeTemp:
                        #print("Sono nell'if atteso")
                        dec += 1
                    print(e[0].getLabel(), e[1].getLabel())
                size_sub_rem.append(sbt_dim_real[posSubtree]-dec)
                #print("Subtrees_dim 1: ", subtrees_dim)
                #print("Subtrees_dim 2: ", sbt_dim_real)
            else:
                #print("Fine rimozioni")
                r = nremovals
        #print("Numero di rimozioni effettuate: ", removals)
        #print("Dimensioni sottoalberi rimossi: ", size_sub_rem)
        # print(newTree.preorder(newTree.getRoot()))
    return newTree, removals, size_sub_rem, removals_array, distances


def Swap_Subtrees(T):
    #nswaps = random.randint(1, maxSwaps)
    distances = []
    swaps_array = []
    newTree = copy.deepcopy(T)
    subtrees = get_all_subtree(newTree)
    dictSubtrees = {}
    dictParents = {}
    # Per ogni radice di sottoalbero allego il rispettivo sottoalbero
    for st in subtrees[1:len(subtrees)]:
        if len(st) > 0:
            dictSubtrees[st[0][0]] = st
            parents = []
            parent = st[0][0]
            while parent.getParent() is not None:
                parents.append(parent.getParent())
                parent = parent.getParent()
            dictParents[st[0][0]] = parents
    exchangeable = []
    keys = list(dictSubtrees.keys())
    i = 1
    # Ricavo i sottoalberi scambiabili
    for key in keys:
        if i <= len(keys)-2:
            for k in keys[i:]:
                if key not in dictParents[k]:
                    exchangeable.append([key, k])
        i += 1
    dictExchanged = {}
    for k in keys:
        dictExchanged[k] = False
    random.shuffle(exchangeable)
    swaps = 0
    i = 0
    nswaps = len(exchangeable)
    while swaps < nswaps and i < len(exchangeable):
        pair = exchangeable[i]
        if not dictExchanged[pair[0]] and not dictExchanged[pair[1]]:
            if pair[1] not in dictParents[pair[0]] and pair[0] not in dictParents[pair[1]]:
                print("Sto scambiando: ",
                      pair[0].getLabel(), pair[1].getLabel())
                swaps += 1
                dictExchanged[pair[0]] = True
                dictExchanged[pair[1]] = True
                p1 = pair[0].getParent()
                tmp = p1.getChildren()
                for j in range(0, len(tmp)):
                    if tmp[j] == pair[0]:
                        tmp[j] = pair[1]
                        break
                p1.setChildren(tmp)
                p2 = pair[1].getParent()
                tmp = p2.getChildren()
                if p2 == pair[0].getParent():
                    # Nel caso in cui sto scambiando fratelli
                    flag = False
                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[1]:
                            if flag:
                                tmp[j] = pair[0]
                                break
                            flag = True
                else:
                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[1]:
                            tmp[j] = pair[0]
                            break
                p2.setChildren(tmp)
                pair[0].setParent(p2)
                pair[1].setParent(p1)

                # Aggiorno gli antenati per ogni nodo
                for node in newTree.getNodes():
                    parents = []
                    parent = node
                    while parent.getParent() is not None:
                        parents.append(parent.getParent())
                        parent = parent.getParent()
                    dictParents[node] = parents

                    newEdges = []
                    preorder = newTree.preorder(newTree.getRoot())
                    for n in preorder:
                        for c in n.getChildren():
                            newEdges.append((n, c))
                    newTree.setEdges(newEdges)

                swaps_array.append(swaps)
                distances.append(Xbwt_Edit_Distance(T, newTree))

        i += 1
    #print("Numero di scambi effettuati: ", swaps)
    # print(newTree.preorder(newTree.getRoot()))

    """
    for node in preorder:
        if node.getParent()is not None:
            print(node.getLabel(), node.getParent().getLabel())
        else:
            print(node.getLabel())
    """

    return newTree, swaps, swaps_array, distances


def Swap_Subtrees2(T):
    #nswaps = random.randint(1, maxSwaps)
    distances = []
    swaps_array = []
    newTree = copy.deepcopy(T)
    subtrees = get_all_subtree(newTree)
    dictSubtrees = {}
    dictParents = {}
    # Per ogni radice di sottoalbero allego il rispettivo sottoalbero
    for st in subtrees[1:len(subtrees)]:
        if len(st) > 0:
            dictSubtrees[st[0][0]] = st
            parents = []
            parent = st[0][0]
            while parent.getParent() is not None:
                parents.append(parent.getParent())
                parent = parent.getParent()
            dictParents[st[0][0]] = parents
    exchangeable = []
    keys = list(dictSubtrees.keys())
    i = 1
    # Ricavo i sottoalberi scambiabili
    for key in keys:
        if i <= len(keys)-2:
            for k in keys[i:]:
                if key not in dictParents[k]:
                    exchangeable.append([key, k])
        i += 1
    dictExchanged = {}
    for k in keys:
        dictExchanged[k] = False
    random.shuffle(exchangeable)
    swaps = 0
    i = 0
    nswaps = 10  # Numero di scambi da effettuare
    while swaps < nswaps and i < len(exchangeable):
        pair = exchangeable[i]
        # if not dictExchanged[pair[0]] and not dictExchanged[pair[1]]:
        if pair[1] not in dictParents[pair[0]] and pair[0] not in dictParents[pair[1]]:
            print("Sto scambiando: ", pair[0].getLabel(), pair[1].getLabel())
            swaps += 1
            dictExchanged[pair[0]] = True
            dictExchanged[pair[1]] = True
            p1 = pair[0].getParent()
            tmp = p1.getChildren()
            for j in range(0, len(tmp)):
                if tmp[j] == pair[0]:
                    tmp[j] = pair[1]
                    break
            p1.setChildren(tmp)
            p2 = pair[1].getParent()
            tmp = p2.getChildren()
            if p2 == pair[0].getParent():
                # Nel caso in cui sto scambiando fratelli
                flag = False
                for j in range(0, len(tmp)):
                    if tmp[j] == pair[1]:
                        if flag:
                            tmp[j] = pair[0]
                            break
                        flag = True
            else:
                for j in range(0, len(tmp)):
                    if tmp[j] == pair[1]:
                        tmp[j] = pair[0]
                        break
            p2.setChildren(tmp)
            pair[0].setParent(p2)
            pair[1].setParent(p1)

            # Aggiorno gli antenati per ogni nodo
            for node in newTree.getNodes():
                parents = []
                parent = node
                while parent.getParent() is not None:
                    parents.append(parent.getParent())
                    parent = parent.getParent()
                dictParents[node] = parents

            newEdges = []
            preorder = newTree.preorder(newTree.getRoot())
            for n in preorder:
                for c in n.getChildren():
                    newEdges.append((n, c))
            newTree.setEdges(newEdges)

            swaps_array.append(swaps)
            distances.append(Xbwt_Edit_Distance(T, newTree))

        i += 1
    #print("Numero di scambi effettuati: ", swaps)
    # print(newTree.preorder(newTree.getRoot()))

    """
    for node in preorder:
        if node.getParent()is not None:
            print(node.getLabel(), node.getParent().getLabel())
        else:
            print(node.getLabel())
    """

    return newTree, swaps, swaps_array, distances


def Swap_Subtrees3(T):
    #nswaps = random.randint(1, maxSwaps)
    distances = []
    swaps_array = []
    newTree = copy.deepcopy(T)
    subtrees = get_all_subtree(newTree)
    dictSubtrees = {}
    dictParents = {}
    # Per ogni radice di sottoalbero allego il rispettivo sottoalbero
    for st in subtrees[1:len(subtrees)]:
        if len(st) > 0:
            dictSubtrees[st[0][0]] = st
            parents = []
            parent = st[0][0]
            while parent.getParent() is not None:
                parents.append(parent.getParent())
                parent = parent.getParent()
            dictParents[st[0][0]] = parents
    exchangeable = []
    keys = list(dictSubtrees.keys())
    i = 1
    # Ricavo i sottoalberi scambiabili
    for key in keys:
        if i <= len(keys)-2:
            for k in keys[i:]:
                if key not in dictParents[k]:
                    exchangeable.append([key, k])
        i += 1
    dictExchanged = {}
    for k in keys:
        dictExchanged[k] = False
    random.shuffle(exchangeable)
    swaps = 0
    # i = 0
    nswaps = 50  # Numero di scambi da effettuare
    while swaps < nswaps:
        pair = random.choice(exchangeable)
        # if not dictExchanged[pair[0]] and not dictExchanged[pair[1]]:
        if pair[1] not in dictParents[pair[0]] and pair[0] not in dictParents[pair[1]]:
            print("Sto scambiando: ", pair[0].getLabel(), pair[1].getLabel())
            swaps += 1
            dictExchanged[pair[0]] = True
            dictExchanged[pair[1]] = True
            p1 = pair[0].getParent()
            tmp = p1.getChildren()
            for j in range(0, len(tmp)):
                if tmp[j] == pair[0]:
                    tmp[j] = pair[1]
                    break
            p1.setChildren(tmp)
            p2 = pair[1].getParent()
            tmp = p2.getChildren()
            if p2 == pair[0].getParent():
                # Nel caso in cui sto scambiando fratelli
                flag = False
                for j in range(0, len(tmp)):
                    if tmp[j] == pair[1]:
                        if flag:
                            tmp[j] = pair[0]
                            break
                        flag = True
            else:
                for j in range(0, len(tmp)):
                    if tmp[j] == pair[1]:
                        tmp[j] = pair[0]
                        break
            p2.setChildren(tmp)
            pair[0].setParent(p2)
            pair[1].setParent(p1)

            # Aggiorno gli antenati per ogni nodo
            for node in newTree.getNodes():
                parents = []
                parent = node
                while parent.getParent() is not None:
                    parents.append(parent.getParent())
                    parent = parent.getParent()
                dictParents[node] = parents

            newEdges = []
            preorder = newTree.preorder(newTree.getRoot())
            for n in preorder:
                for c in n.getChildren():
                    newEdges.append((n, c))
            newTree.setEdges(newEdges)

            subtrees = get_all_subtree(newTree)
            dictSubtrees = {}
            dictParents = {}
            # Per ogni radice di sottoalbero allego il rispettivo sottoalbero
            for st in subtrees[1:len(subtrees)]:
                if len(st) > 0:
                    dictSubtrees[st[0][0]] = st
                    parents = []
                    parent = st[0][0]
                    while parent.getParent() is not None:
                        parents.append(parent.getParent())
                        parent = parent.getParent()
                    dictParents[st[0][0]] = parents
            exchangeable = []
            keys = list(dictSubtrees.keys())
            j = 1
            # Ricavo i sottoalberi scambiabili
            for key in keys:
                if j <= len(keys)-2:
                    for k in keys[j:]:
                        if key not in dictParents[k]:
                            exchangeable.append([key, k])
                j += 1

            random.shuffle(exchangeable)

            swaps_array.append(swaps)
            distances.append(Xbwt_Edit_Distance(T, newTree))

        # i+=1
    #print("Numero di scambi effettuati: ", swaps)
    # print(newTree.preorder(newTree.getRoot()))

    """
    for node in preorder:
        if node.getParent()is not None:
            print(node.getLabel(), node.getParent().getLabel())
        else:
            print(node.getLabel())
    """

    return newTree, swaps, swaps_array, distances


def Swap_Symbols(T, maxSwaps):
    maxDegFor2 = []
    nodeT = T.preorder(T.getRoot())
    nswaps = random.randint(1, maxSwaps)
    newTree = copy.deepcopy(T)
    nodeNT = newTree.preorder(newTree.getRoot())

    dictNodes1 = {}
    for n in nodeT:
        if n.getLabel() != '$':
            children = []
            if n.getChildren()[0].getLabel() != '$':
                for child in n.getChildren():
                    children.append(child.getLabel())
                dictNodes1[n.getLabel()] = children

    """
    dictNodes = {}
    i = 0
    for node in nodeNT:
        dictNodes[node] = nodeT[i] 
        i+=1
    """

    # Coppie padre figlio che uguali alla configurazione di partenza dopo uno scambio
    rem = 0

    nodes = []
    for n in newTree.getNodes():
        if n.getLabel() != '$':
            nodes.append(n)
    exchangeable = []
    i = 1
    for node in nodes:
        if i < len(nodes)-2:
            for n in nodes[i:]:
                exchangeable.append([node, n])
        i += 1
    #print([(e[0].getLabel(), e[1].getLabel()) for e in exchangeable])
    dictExchanged = {}
    for node in nodes:
        dictExchanged[node] = False
    random.shuffle(exchangeable)
    swaps = 0
    i = 0
    while swaps < nswaps and i < len(exchangeable):
        pair = exchangeable[i]
        if not dictExchanged[pair[0]] and not dictExchanged[pair[1]]:
            print("Sto scambiando: ", pair[0].getLabel(), pair[1].getLabel())
            swaps += 1
            dictExchanged[pair[0]] = True
            dictExchanged[pair[1]] = True
            if len(pair[0].getChildren()) == 1 and pair[0].getChildren()[0].getLabel() == '$':
                maxDegFor2.append(0)
            else:
                maxDegFor2.append(
                    max(len(pair[0].getChildren()), len(pair[1].getChildren()))*2)
            # Caso in cui scambio padre-figlio
            if pair[1] in pair[0].getChildren():
                print("Caso 1")
                # Caso in cui scambio la radice
                tmpParent = pair[0].getParent()
                tmp = pair[0].getChildren()
                #print("TEMP 1: ", [n.getLabel() for n in tmp])
                for j in range(0, len(tmp)):
                    if tmp[j] == pair[1]:
                        tmp[j] = pair[0]
                        break
                #print("TEMP 2: ", [n.getLabel() for n in tmp])
                tmpChildren = pair[1].getChildren()
                #print("TEMP CHILDREN: ", tmpChildren)
                pair[1].setChildren(tmp)
                pair[0].setChildren(tmpChildren)
                # Assegno i nuovi genitori
                for n in tmp:
                    n.setParent(pair[1])
                for n in tmpChildren:
                    n.setParent(pair[0])
                #print("Parent for children 1: ", [x.getParent().getLabel() for x in tmp])
                if pair[0].getLabel() == T.getRoot().getLabel():
                    print("Caso 1 - radice")
                    newTree.setRoot(pair[1])
                    pair[0].setParent(pair[1])
                else:
                    parent = tmpParent
                    #print("Parent 0: ", parent.getLabel())
                    tmp = parent.getChildren()
                    #print("TEMP 3: ", [n.getLabel() for n in tmp])
                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[0]:
                            tmp[j] = pair[1]
                            tmp[j].setParent(parent)
                            break
                    #print("TEMP 4: ", [n.getLabel() for n in tmp])
                    parent.setChildren(tmp)
            # Caso in cui scambio due nodi fratelli
            elif pair[0] != T.getRoot() and pair[0].getParent() == pair[1].getParent():
                #print("Caso 2")
                parent = pair[0].getParent()
                #print("Parent : ", parent.getLabel())
                tmp = parent.getChildren()
                #print("TEMP 1: ", [n.getLabel() for n in tmp])
                flag = -1
                for j in range(0, len(tmp)):
                    if flag < 2:
                        if flag == -1 and tmp[j] == pair[0]:
                            tmp[j] = pair[1]
                            flag += 1
                        if flag > 0 and tmp[j] == pair[1]:
                            tmp[j] = pair[0]
                            flag += 1
                        if flag == 0:
                            flag += 1
                    if flag == 2:
                        #print("TEMP 2: ", [n.getLabel() for n in tmp])
                        # Aggiorno i figli del padre
                        parent.setChildren(tmp)
                        break
                # Aggiornamento dei genitori dei due figli scambiati
                for c in pair[0].getChildren():
                    c.setParent(pair[1])
                #print("Pair for children 1: ", [x.getParent().getLabel() for x in pair[0].getChildren()])
                for c in pair[1].getChildren():
                    c.setParent(pair[0])
                #print("Pair for children 2: ", [x.getParent().getLabel() for x in pair[1].getChildren()])
                tmpChildren = pair[1].getChildren()
                #print("TEMP CHILDREN: ", [c.getLabel() for c in tmpChildren])
                #print("PAIR[0].GETCHILDREN(): ", [c.getLabel() for c in pair[0].getChildren()])
                pair[1].setChildren(pair[0].getChildren())
                pair[0].setChildren(tmpChildren)
            # Tutti gli altri casi
            else:
                #print("Caso 3")
                if pair[0].getLabel() == T.getRoot().getLabel():
                    #print("Caso 3 - radice")
                    # Ricavo i figli della radice
                    tmp = pair[0].getChildren()
                    #print("TEMP: ", [n.getLabel() for n in tmp])
                    # Setto la nuova radice
                    newTree.setRoot(pair[1])
                    # Aggiorno il genitore dei figli della radice
                    for n in tmp:
                        n.setParent(pair[1])
                    #print("Parent for children 1: ", [x.getParent().getLabel() for x in tmp])
                    parent = pair[1].getParent()
                    pair[1].setParent(None)  # NUOVA STRINGA
                    #print("Parent 0: ", parent.getLabel())
                    tmp = parent.getChildren()
                    #print("TEMP1: ", [n.getLabel() for n in tmp])
                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[1]:
                            tmp[j] = pair[0]
                            tmp[j].setParent(parent)
                            break
                    parent.setChildren(tmp)
                    tmpChildren = pair[0].getChildren()
                    pair[0].setChildren(pair[1].getChildren())
                    pair[1].setChildren(tmpChildren)
                    # Aggiorno il genitore dei figli della nuova posizione della radice
                    for c in pair[0].getChildren():
                        c.setParent(pair[0])
                    #print("TEMP2: ", [n.getLabel() for n in tmp])

                # Caso in cui i due nodi scambiati non sono due fratelli, non riguardano la radice e non sono padre-figlio
                else:
                    #print("Else del caso 3")
                    # Cambio il primo nodo con il secondo
                    parent1 = pair[0].getParent()
                    #print("Parent 1: ", parent1.getLabel())
                    children1 = pair[0].getChildren()
                    tmp = parent1.getChildren()
                    #print("TEMP1: ", [n.getLabel() for n in tmp])

                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[0]:
                            tmp[j] = pair[1]
                            # tmp[j].setParent(parent)
                            break
                    #print("TEMP2: ", [n.getLabel() for n in tmp])
                    parent1.setChildren(tmp)
                    for c in children1:
                        c.setParent(pair[1])
                    #print("Parent for children 1: ", [x.getParent().getLabel() for x in children1])
                    # Cambio il secondo nodo con il primo
                    parent2 = pair[1].getParent()
                    #print("Parent 2: ", parent2.getLabel())
                    children2 = pair[1].getChildren()
                    pair[1].setChildren(children1)
                    pair[0].setChildren(children2)
                    tmp = parent2.getChildren()
                    #print("TEMP3: ", [n.getLabel() for n in tmp])
                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[1]:
                            tmp[j] = pair[0]
                            tmp[j].setParent(parent2)
                            break
                    # Aggiornamento padre di pair[1]
                    pair[1].setParent(parent1)
                    #print("TEMP4: ", [n.getLabel() for n in tmp])
                    parent2.setChildren(tmp)
                    for c in children2:
                        c.setParent(pair[0])
                    #print("Parent for children 2: ", [x.getParent().getLabel() for x in children2])
                    #print("Children del genitore di pair[0]: ", [c.getLabel() for c in parent1.getChildren()])

            """
            # Conteggio coppie genitore-figlio con la stessa configurazione originale dopo lo scambio
            if not pair[0].isRoot():
                if not dictNodes[pair[0]].isRoot():
                    if pair[0].getParent().getLabel() == dictNodes[pair[0]].getParent().getLabel() and pair[0].getParent() in dictExchanged:
                        posP0 = 0
                        ind = 0
                        for n in pair[0].getParent().getChildren():
                            if n.getLabel() == pair[0].getLabel():
                                posP0 = ind
                                break
                            ind+=1
                        posP0originale = 0
                        ind = 0
                        for n in dictNodes[pair[0]].getParent().getChildren():
                            if n.getLabel() == dictNodes[pair[0]].getLabel():
                                posP0originale = ind
                                break
                            ind+=1
                        if posP0 == posP0originale:
                            rem+=1
                        print("Pair 0: ", pair[0].getParent().getLabel())
                        rem+=1
            else:
                for child in pair[0].getChildren():
                    if child.getLabel() in [c.getLabel() for c in dictNodes[pair[0]].getChildren()]:
                        print("Sono nei children 0 con: ", child.getLabel())
                        rem+=1
            if not pair[1].isRoot():
                if not dictNodes[pair[1]].isRoot():
                     if pair[1].getParent().getLabel() == dictNodes[pair[1]].getParent().getLabel() and pair[1].getParent() in dictExchanged:
                        posP1 = 0
                        ind = 0
                        for n in pair[1].getParent().getChildren():
                            if n.getLabel() == pair[1].getLabel():
                                posP1 = ind
                                break
                            ind+=1
                        posP1originale = 0
                        ind = 0
                        for n in dictNodes[pair[1]].getParent().getChildren():
                            if n.getLabel() == dictNodes[pair[1]].getLabel():
                                posP1originale = ind
                                break
                            ind+=1
                        if posP1 == posP1originale:
                            rem+=1
                            print("Pair 1: ", pair[1].getParent().getLabel())
            else:
                for child in pair[1].getChildren():
                    if child.getLabel() in [c.getLabel() for c in dictNodes[pair[1]].getChildren()]:
                        print("Sono nei children 1 con: ", child.getLabel())
                        rem+=1
            """
        i += 1
    #print("Numero di scambi effettuati: ", swaps)
    preorder = newTree.preorder(newTree.getRoot())
    #print("Finale: ", [n.getLabel() for n in preorder])
    newEdges = []
    for n in preorder:
        #print("Nodo: ", n.getLabel())
        for c in n.getChildren():
            #print((n.getLabel(), c.getLabel()))
            newEdges.append((n, c))
    newTree.setEdges(newEdges)
    #print("Radice attuale: ", newTree.getRoot().getLabel())
    #print("Archi finali: ")
    # for e in newEdges:
    #print((e[0].getLabel(), e[1].getLabel()))

    labelExchanged = []
    for k in dictExchanged.keys():
        if dictExchanged[k]:
            labelExchanged.append(k.getLabel())

    # Determino il numero di coppie padre-figlio scambiate
    nodeNT = newTree.preorder(newTree.getRoot())
    dictNodes2 = {}
    for n in nodeNT:
        if n.getLabel() != '$':
            children = []
            if n.getChildren()[0].getLabel() != '$':
                for child in n.getChildren():
                    children.append(child.getLabel())
                dictNodes2[n.getLabel()] = children

    for k in dictNodes1.keys():
        c1 = dictNodes1[k]
        c2 = []
        if k in dictNodes2.keys():
            c2 = dictNodes2[k]
        if len(c2) > 0:
            if k in labelExchanged:
                for l in c1:
                    if l in c2 and l in labelExchanged:
                        rem += 1

    pcSwaps = 0
    for n in preorder:
        if n.getLabel() != '$' and dictExchanged[n]:
            if len(n.getChildren()) >= 1:
                for c in n.getChildren():
                    if c.getLabel() != '$' and dictExchanged[c]:
                        pcSwaps += 1
    """
    for edge in newEdges:
        print((e[0].getLabel(), e[1].getLabel()))
    """
    return newTree, swaps, maxDegFor2, pcSwaps, rem


class Node(object):
    """ Node of a Tree """

    def __init__(self, label='root', children=None, parent=None):
        self.label = label
        self.parent = parent
        self.children = []
        if children is not None:
            for child in children:
                self.addChild(child)

    def representation(self):
        label = self.getLabel()
        children = [child.representation() for child in self.getChildren()]
        # return a tuple of the label and the children
        return (label, *children)

    def getLabel(self):
        """ Return the label of a node """
        return self.label

    def setLabel(self, label):
        self.label = label

    def getParent(self):
        return self.parent

    def setParent(self, parent):
        self.parent = parent

    def isRoot(self):
        """ Check if the node is the root """
        if self.parent is None:
            return True
        else:
            return False

    def isLeaf(self):
        """ Check if the node is a leaf """
        if len(self.getChildren()) == 0:
            return True
        else:
            return False

    def level(self):
        """ Return the level of a node """
        if self.isRoot():
            return 0
        else:
            return (1 + self.getParent().level()) if self.getParent() is not None else 1
            
    def isRightmost(self):
        """ 
        Return 1 if node is the rightmost children of the parent, 0
        otherwise
        """
        if (par:=self.getParent() )is not None:
            length_parent = len(par.getChildren())
            if length_parent != 0:
                if (par.getChildren()[length_parent-1] == self):
                    return 1
            return 0
        else:
            return 0

    def addChild(self, node):
        """ Add a child at node """
        node.parent = self
        assert isinstance(node, Node)
        self.getChildren().append(node)

    def getChildren(self):
        """ Return the children's array of a node"""
        return self.children

    def setChildren(self, children):
        self.children = children
        
    
    
    def __repr__(self) -> str:
        return self.label


class Tree(object):
    """ A Generic Tree """

    def __init__(self):
        self.root = None
        self.height = 0
        self.nodes = []
        self.edges = []

    def insert(self, node, parent):
        """ Insert a node into tree """
        if parent is not None:
            parent.addChild(node)
            self.edges.append((parent, node))
            if parent.level == self.height:
                self.height +=1
                node.level = parent.level +1
            
        else:
            if self.root is None:
                self.root = node
                self.height = 1
                node.level = 1
        self.nodes.append(node)
    
    def addDollarsToLeaves(self):
        for r in [i for i in self.nodes if len(i.getChildren()) == 0]:
            doll = Node("$")
            self.insert(doll, r)
        
    def getRoot(self):
        """ Return the root of tree"""
        return self.root

    def setRoot(self, root):
        self.root = root

    def getNodes(self):
        """ Return the nodes of tree"""
        return self.nodes

    def getEdges(self):
        """ Return the edges of tree"""
        return self.edges

    def setEdges(self, edges):
        self.edges = edges

    def printAllNodes(self):
        """ TODO """
        print("Nodes: ")
        for n in self.nodes:
            print(n.getLabel())

    def preorder(self, root):
        """ TODO """
        if not root:
            return []
        result = []
        if root.children:
            for node in root.children:
                result.extend(self.preorder(node))
        return [root] + result

    def getHeight(self):
        return self.height

class XBWT(object):

    def __init__(self, T):
        self.T = T

    def getTree(self):
        return self.T

    def preorderTraversal(self, root):
        """ Visita in preordine di un albero k-ario """

        S_last = []  # 1 o 0 se il nodo è o non è il figlio più a destra del suo genitore
        S_alpha = []  # Etichette dei nodi
        S_pi = []  # Cammini verso l'alto dei vari nodi
        # Array di triple (etichetta nodo, livello, posizione del padre in IntNodes del nodo)
        IntNodes = []
        level = 0  # Tiene traccia del livello corrente di un nodo

        Child = [] # Keep track of the number of child of each node

        index = 0  # Indice corrente dell'array IntNodes
        pos_sub = 0  # Posizioni da togliere a curr_index per trovare il padre di un nodo
        curr_index = 0  # Tiene traccia della posizione del padre di un nodo nell'array IntNodes
        currentPath = ""  # Memorizza il path corrente (radice-nodo) di un nodo
        Stack = []  # Coda per analizzare in preordine i vari nodi

        # Inizializzazione

        Preorder = []  # Tiene traccia dei nodi visitati
        Preorder.append(root)
        Stack.append(root)

        S_last.append(0)
        S_alpha.append(root.getLabel())
        S_pi.append("")
        IntNodes.append((root.getLabel(), level, curr_index))
        currentPath += root.getLabel()
        curr_index += 1
        index += 1

        while len(Stack) > 0:
            # Uso il flag per verificare se tutti i nodi sono stati visitati
            flag = 0
            # Caso 1: se l'elemento iniziale della pila è una foglia rimuovo
            # questo elemento dalla pila
            if len((Stack[len(Stack)-1]).getChildren()) == 0:
                Stack.pop()
                currentPath = currentPath[:len(currentPath)-1]
                level -= 1
                pos_sub += 1
                curr_index = index-pos_sub
                # Caso 2: Se l'elemento iniziale della pila è un nodo con dei
                # figli
            else:
                Par = Stack[len(Stack)-1]
            # Quando viene trovato un nodo figlio non visitato (nella sequenza
            # da sinistra a destra), inseriscilo nella pila e nel
            # vettore Preorder (nodi visitati). Quindi, ricomincia dal caso 1
            # per esplorare il nodo visitato.
            for i in Par.getChildren():
                if i not in Preorder:
                    flag = 1
                    S_last.append(i.isRightmost())
                    # Appendo il cammino verso l'alto
                    S_pi.append(currentPath[::-1])
                    currentPath += i.getLabel()
                    level += 1
                    S_alpha.append(i.getLabel())
                    IntNodes.append((i.getLabel(), level, curr_index))
                    index += 1
                    curr_index = index
                    Stack.append(i)
                    Preorder.append(i)
                    break
                    # Se tutti i nodi figli da sinistra a destra di un
                    # genitore sono stati visitati, rimuovi il genitore dalla
                    # pila
            if flag == 0:
                Stack.pop()
                currentPath = currentPath[:len(currentPath)-1]
                pos_sub += 1
                curr_index = index-pos_sub
                level -= 1
                # Nodo figlio della radice esaminato
                if len(Stack) == 1:
                    curr_index = 1  # Il padre del prossimo nodo si troverà in posizione 1
                    pos_sub = 0

        return S_last, S_alpha, S_pi, IntNodes

    def computeIntNodesArray(self, root):
        """ 
            Visita in preordine di un albero k-ario e determina l'array
            IntNodes
        """

        # Array di triple (etichetta nodo, livello, posizione del padre in IntNodes del nodo)
        IntNodes = []
        level = 0  # Tiene traccia del livello corrente di un nodo

        index = 0  # Indice corrente dell'array IntNodes
        # pos_sub = 0 # Posizioni da togliere a curr_index per trovare il padre di un nodo
        # curr_index = 0 # Tiene traccia della posizione del padre di un nodo nell'array IntNodes
        currentPath = ""  # Memorizza il path corrente (radice-nodo) di un nodo
        Stack = []  # Coda per analizzare in preordine i vari nodi

        # Inizializzazione

        Preorder = []  # Tiene traccia dei nodi visitati
        Preorder.append(root)
        Stack.append(root)

        IntNodes.append([root.getLabel(), level, 0])
        currentPath += root.getLabel()
        # curr_index+=1
        index += 1

        posParent = {}
        posParent[root] = 1

        while len(Stack) > 0:
            # Uso il flag per verificare se tutti i nodi sono stati visitati
            flag = 0
            # Caso 1: se l'elemento iniziale della pila è una foglia rimuovo
            # questo elemento dalla pila
            if len((Stack[len(Stack)-1]).getChildren()) == 0:
                Stack.pop()
                level -= 1
                # pos_sub+=1
                # Caso 2: Se l'elemento iniziale della pila è un nodo con dei
                # figli
            else:
                Par = Stack[len(Stack)-1]
            # Quando viene trovato un nodo figlio non visitato (nella sequenza
            # da sinistra a destra), inseriscilo nella pila e nel
            # vettore Preorder (nodi visitati). Quindi, ricomincia dal caso 1
            # per esplorare il nodo visitato.
            for i in Par.getChildren():
                if i not in Preorder:
                    flag = 1
                    level += 1
                    IntNodes.append(
                        [i.getLabel(), level, posParent[i.getParent()]])
                    index += 1
                    if i.getLabel() != '$':
                        posParent[i] = index
                    #curr_index = index
                    Stack.append(i)
                    Preorder.append(i)
                    break
                    # Se tutti i nodi figli da sinistra a destra di un
                    # genitore sono stati visitati, rimuovi il genitore dalla
                    # pila
            if flag == 0:
                Stack.pop()
                level -= 1

        return IntNodes

    """
    def radixSortLSDInteger(self, array, w, maxName):
        #TODO
        a = array.copy()
        print("Array: ", a, end="\n\n")
        n = len(a)
        print("Lunghezza array: ", n)
        R = maxName
        aux = ["" for i in range(0, n)]
        
        for d in range(w-1, -1, -1):
            # sort by key-indexed counting on dth character
            
            count = np.zeros(R, dtype="int")
            print("Count: ", count)
            # Count frequencies
            for i in range(0, n):
                count[int(a[i][1][d])]+=1
            
            print("Count pre-cumulate: ", count)
            
            # Compute cumulates
            for r in range(1, R):
                count[r]+=count[r-1]
                
            # Move data
            for i in range(0, n):
                print("Aux: ", aux)
                print("Count: ", count)
                print("Count index: ", count[int(a[i][1][d])])
                print("Dato: ", int(a[i][1][d])-1)
                aux[count[int(a[i][1][d])]] = a[i]
                count[int(a[i][1][d])]+=1
                
            # Copy back
            for i in range(0, n):
                a[i] = aux[i]
            print("---------")
        return a
    """

    def radixSortInteger(self, array, radix=10):
        if len(array) == 0:
            return array

        # Determine minimum and maximum values
        minValue = array[0][1]
        maxValue = array[0][1]
        for i in range(1, len(array)):
            if array[i][1] < minValue:
                minValue = array[i][1]
            elif array[i][1] > maxValue:
                maxValue = array[i][1]

        # Perform counting sort on each exponent/digit, starting at the least
        # significant digit
        exponent = 1
        while (maxValue - minValue) / exponent >= 1:
            array = self.countingSortByDigit(array, radix, exponent, minValue)
            exponent *= radix

        return array

    def countingSortByDigit(self, array, radix, exponent, minValue):
        #print("Array: ", array)
        bucketIndex = -1
        buckets = [0] * radix
        output = [None] * len(array)

        # Count frequencies
        for i in range(0, len(array)):
            bucketIndex = math.floor(
                ((array[i][1] - minValue) / exponent) % radix)
            buckets[bucketIndex] += 1

        # Compute cumulates
        for i in range(1, radix):
            buckets[i] += buckets[i - 1]

        # Move records
        for i in range(len(array) - 1, -1, -1):
            bucketIndex = math.floor(
                ((array[i][1] - minValue) / exponent) % radix)
            buckets[bucketIndex] -= 1
            output[buckets[bucketIndex]] = array[i]
            #print("Output: ", output)
        return output

    def radixSortLSD(self, array, w):
        """ TODO """
        a = array.copy()
        n = len(a)
        R = 256  # extend ASCII alphabet size
        aux = ["" for i in range(0, n)]

        for d in range(w-1, -1, -1):
            # sort by key-indexed counting on dth character

            count = np.zeros(R+1)
            # Count frequencies
            for i in range(0, n):
                count[ord(a[i][1][d])+1] += 1

            # Compute cumulates
            for r in range(0, R):
                count[r+1] += count[r]

            # Move data
            for i in range(0, n):
                #print("Count index: ", count[int(a[i][1][d])])
                aux[int(count[ord(a[i][1][d])])] = a[i]
                count[ord(a[i][1][d])] += 1

            # Copy back
            for i in range(0, n):
                a[i] = aux[i]
        return a

    def nameTriplets(self, sortedTriplets):
        """ TODO """
        notUnique = False
        lexName = []
        lexName.append([2, sortedTriplets[0][0]])
        #count = 1
        for i in range(1, len(sortedTriplets)):
            if sortedTriplets[i-1][1] == sortedTriplets[i][1]:
                notUnique = True
                # lexName.append(count)
                lexName.append([lexName[i-1][0], sortedTriplets[i][0]])
            else:
                # count+=1
                lexName.append([lexName[i-1][0]+1, sortedTriplets[i][0]])
        return lexName, notUnique

    def contractedTree(self, IntNodes, j, lexName, first_iteration):

        #print("J contracted tree: ", j)
        """ 
            Return the contracted tree for PathSort function 
        """
        IntNodes_temp = []
        for i in range(len(IntNodes)):
            #tmp = IntNodes[i]
            # tmp[0]=Node(tmp[0])
            IntNodes_temp.append(IntNodes[i])
        if first_iteration:
            for i in range(len(lexName)):
                IntNodes_temp[lexName[i][1]][0] = str(lexName[i][0])
        if j == 0:
            IntNodes_temp[0][0] = '1'
        for i in range(len(IntNodes)):
            IntNodes_temp[i][0] = Node(IntNodes_temp[i][0])
        #print("\n\nIntNodesTempCT: ", IntNodes_temp, end="\n\n")
        #print("\n\nIntNodesSEG: ", IntNodes, end="\n\n")

        dictNext = {}
        dictNext[0] = 1
        dictNext[1] = 2
        dictNext[2] = 0
        links = []
        # Ricostruisco i vari link partendo dall'ultimo elemento di IntNodes
        if j == 0:
            for i in range(len(IntNodes_temp)-1, -1, -1):
                if IntNodes_temp[i][1] % 3 != j:
                    if IntNodes_temp[i][1] % 3 == dictNext[j]:
                        # Se il genitore è la radice
                        if IntNodes_temp[i][2] == 1:
                            links.append(
                                [IntNodes_temp[IntNodes_temp[i][2]-1][0], IntNodes_temp[i][0]])
                        else:
                            aux = IntNodes_temp[IntNodes_temp[IntNodes_temp[i][2]-1][2]-1][0]
                            links.append([aux, IntNodes_temp[i][0]])
                    else:
                        links.append(
                            [IntNodes_temp[IntNodes_temp[i][2]-1][0], IntNodes_temp[i][0]])
        else:
            for i in range(len(IntNodes_temp)-1, 0, -1):
                if IntNodes_temp[i][1] % 3 != j:
                    if IntNodes_temp[i][1] % 3 == dictNext[j]:
                        aux = IntNodes_temp[IntNodes_temp[IntNodes_temp[i][2]-1][2]-1][0]
                        links.append([aux, IntNodes_temp[i][0]])
                    else:
                        links.append(
                            [IntNodes_temp[IntNodes_temp[i][2]-1][0], IntNodes_temp[i][0]])

        tree = Tree()
        tree.insert(IntNodes_temp[0][0], None)
        for i in range(len(links)-1, -1, -1):
            tree.insert(links[i][1], links[i][0])

        return tree

    def merge(self, sV, SA_first, SA_second, IntNodes, j_v, dummy_root=False, first_iteration=False):
        #print("\n******* Procedura di Merge (sV, j_v, dummy_root) *******: ", sV, j_v, dummy_root)
        #print("\nSA_First: ", SA_first)
        #print("\nSA_Second: ", SA_second, end="\n")
        # sV sta per start Value
        dictCond = {}
        dictCond[0] = 1
        dictCond[1] = 2
        dictCond[2] = 0

        IN = IntNodes.copy()

        SA_merge = []
        i = 0  # index SA_first
        j = 0  # index SA_second

        flag = True
        while flag:
            #print("Prima condizione: ", IntNodes[SA_first[i]+sV], i, j)
            # In quanto stiamo considerando il livello
            if IntNodes[SA_first[i]+sV][1] % 3 == dictCond[j_v]:  # secondo caso
                #print("Sono nel primo if di tutti")
                # etichetta genitore, etichetta nonno
                pair1 = [IN[IN[SA_first[i]+sV][2]][0],
                         IN[IN[IN[SA_first[i]+sV][2]][2]][0]]
                pair2 = [IN[IN[SA_second[j]+sV][2]][0],
                         IN[IN[IN[SA_second[j]+sV][2]][2]][0]]
                #print(pair1, pair2)
                if not first_iteration:
                    #print("Pair1: ", IN[SA_first[i]+sV][0], IN[IN[SA_first[i]+sV][2]][0])
                    pair1 = [int(IN[SA_first[i]+sV][0])]
                    pair2 = [int(IN[SA_second[j]+sV][0])]
                #print("pair1 e pair2: ", pair1, pair2)
                if pair2[0] == pair1[0]:
                    if len(pair1) == 1 or pair2[1] == pair1[1]:
                        if SA_first.index(IN[SA_second[j]+sV][2]-sV) > SA_first.index(SA_first[i]):
                            SA_merge.append(SA_first[i])
                            i += 1
                        else:
                            SA_merge.append(SA_second[j])
                            j += 1
                    elif pair2[1] > pair1[1]:
                        SA_merge.append(SA_first[i])
                        i += 1
                    else:
                        SA_merge.append(SA_second[j])
                        j += 1
                elif pair2[0] > pair1[0]:
                    SA_merge.append(SA_first[i])
                    i += 1
                else:
                    SA_merge.append(SA_second[j])
                    j += 1

            elif IntNodes[SA_first[i]+sV][1] % 3 == dictCond[dictCond[j_v]]:
                #print("1 ELSEIF con: ", IntNodes[SA_first[i]+sV][1]%3, dictCond[dictCond[j_v]])
                # [etichetta dek genitore, indice del nonno - sv]
                # print(IN[SA_first[i]+sV])
                val1 = IN[IN[SA_first[i]+sV][2]][0]
                #print("val1: ", val1, type(val1))
                val2 = IN[IN[SA_second[j]+sV][2]][0]
                #print("val2: ", val2, type(val2))
                if not first_iteration:
                    #print("Non sono nella prima iterazioe")
                    val1 = int(IN[SA_first[i]+sV][0])
                    val2 = int(IN[SA_second[j]+sV][0])
                #print("(val1, val2): ", val1, val2)
                if val2 == val1:
                    """ 
                    Se i genitori hanno la stessa etichetta, confronto l'indice
                    delle posizioni dei genitori. Entrambe si troveranno in
                    SA_First
                    """
                    if SA_first.index(IN[SA_second[j]+sV][2]-sV) > SA_first.index(IN[SA_first[i]+sV][2]-sV):
                        SA_merge.append(SA_first[i])
                        i += 1
                    else:
                        SA_merge.append(SA_second[j])
                        j += 1
                elif val2 > val1:
                    SA_merge.append(SA_first[i])
                    i += 1
                else:
                    SA_merge.append(SA_second[j])
                    j += 1

            if j >= len(SA_second):
                for index in range(i, len(SA_first)):
                    SA_merge.append(SA_first[index])
                flag = False

            if i >= len(SA_first):
                for index in range(j, len(SA_second)):
                    SA_merge.append(SA_second[index])
                flag = False
            #print("\n\nSA_Merge attuale, dummy_root, first_iteration: ", SA_merge, dummy_root, first_iteration)

        if dummy_root:
            pos = 1
            if first_iteration:
                pos = 0
            return [SA_merge[i] for i in range(pos, len(SA_merge))]
        else:
            return SA_merge

    def pathSort(self, T, dummy_root=False, first_iteration=True, rem=0, maxName=0):

        #print("Radice fittizia: ", dummy_root)

        #global IntNodes
        IntNodes = self.computeIntNodesArray(T.getRoot())

        #print("\nIntNodes: \n", IntNodes,end="\n\n")

        # Numero di nodi a livello j = 0, 1, 2 mod 3
        nnl = np.zeros(3, dtype="int")
        for i in IntNodes:
            nnl[i[1] % 3] += 1
        nnl[0] -= rem
        #print("Numero di nodi a livello j = 0, 1, 2 mod 3:\n", nnl, end="\n\n")

        # t/3
        #print("Rem: ", rem)
        if rem > 0:
            x = math.ceil((len(IntNodes)-rem)/3)
            #print("x: ", x)
        else:
            x = math.ceil(len(IntNodes)/3)
        # print(len(IntNodes))
        #print("t/3: ", x, end="\n\n")

        j = None
        for i in range(len(nnl)):
            if nnl[i] >= x:
                j = i
                break
        #print("j: ", j, end="\n\n")

        Pos_first = []
        Pos_second = []
        for i in range(len(IntNodes)):
            if IntNodes[i][1] % 3 != j:
                Pos_first.append(i)
            else:
                Pos_second.append(i)
        print("Pos_first: \n", Pos_first, end="\n\n")
        #print("Pos_second: \n", Pos_second, end="\n\n")

        # Inserisco i caratteri speciali in un array temporaneo
        IntNodes_temp = []
        inc = 0
        if j != 0:
            inc = 3
            IntNodes_temp.append(['0', -3, -1])
            IntNodes_temp.append(['0', -2, 0])
            IntNodes_temp.append(['0', -1, 1])
            for i in range(len(IntNodes)):
                temp = list(IntNodes[i])
                temp[2] += 2
                IntNodes_temp.append(temp)
        else:
            # dummy_root=True (scegliere se ripristinare o meno)
            inc = 2
            IntNodes_temp.append(['0', -2, -1])
            IntNodes_temp.append(['0', -1, 0])
            for i in range(len(IntNodes)):
                temp = list(IntNodes[i])
                temp[2] += 1
                IntNodes_temp.append(temp)
        #print("IntNodes_temp: \n", IntNodes_temp, end="\n\n")

        # --> Possibile errore nella generazione
        triplets = []
        for i in Pos_first:
            triplet_container = []
            triple = []
            index = i+inc
            triplet_container.append(i)
            for c in range(3):
                triple.append(IntNodes_temp[IntNodes_temp[index][2]][0])
                index = IntNodes_temp[index][2]
            triplet_container.append(triple)
            triplets.append(triplet_container)
        #print("Triplette:\n", triplets, end="\n\n")

        """ valutare se rimuovere """
        arr = []
        for i in range(len(triplets)):
            arr.append(list((i, triplets[i])))
        # print(arr)

        if first_iteration:
            sortedTriplets = self.radixSortLSD(triplets, 3)
            #print("Triplette ordinate: ", sortedTriplets)
        else:
            sortedTriplets = self.radixSortInteger(
                [[e[0], int(''.join(e[1]))] for e in triplets])
            #print("Triplette ordinate (ri): ", sortedTriplets)
        print("Triplette ordinate:\n", sortedTriplets, end="\n\n")

        lexName, notUnique = self.nameTriplets(sortedTriplets)
        #print("Ranking:\n", lexName, end="\n\n")

        print("\nLexName: ", lexName)

        maxName = lexName[len(lexName)-1][0]

        SA = np.zeros(len(triplets), dtype="int")
        if notUnique:
            contract_tree = self.contractedTree(
                copy.deepcopy(IntNodes), j, lexName, first_iteration)
            if j != 0:
                SA = self.pathSort(contract_tree, False, False,
                                   rem, maxName)  # prima era true
            else:
                rem += 1
                SA = self.pathSort(contract_tree, True, False, rem, maxName)
        else:
            for i in range(len(sortedTriplets)):
                # rivedere visto la modifica precedente
                SA[i] = Pos_first.index(sortedTriplets[i][0])
                # SA[i] = i #nuova modifica da testare
        #print("j, dummy_root e first_iteration precedente (ricorsione ripresa): ", j, dummy_root, first_iteration)
        #print("SA 1:\n", SA, end="\n\n")

        #print("Pos_first (2) e dummy_root: ", Pos_first, dummy_root)
        
        if j == 0 and rem > 0:
            for i in range(len(SA)):
                SA[i] -= 1
            #print("Ho decrentato gli elementi di SA: ", SA)

        #print("Pos_first: ", Pos_first)
        print("SA 2: ", SA)

        # Determino SA_first
        SA_first = []
        for i in range(len(SA)):
            SA_first.append(Pos_first[SA[i]])
        #print("SA_first:\n", SA_first, end="\n\n")

        # Compute SA_second
        SA_second = np.zeros(len(Pos_second), dtype="int")
        # Coppie: (etichetta genitore, indice in SA_first del genitore)
        pairs = []
        c = 0

        if not first_iteration:
            if j == 0:
                pairs.append(
                    (IntNodes_temp[Pos_second[0]+inc][0], 0, Pos_second[0]))
                c = 1
            for i in range(c, len(Pos_second)):
                #print("C: ", c)
                #print("Dati essenziali (IntNodesTemp, i, inc): ", IntNodes_temp, i, inc)
                # print(IntNodes_temp[Pos_second[i]+inc][2]-inc)
                #print("Nuovamente SA_first: ", SA_first)
                pairs.append((IntNodes_temp[Pos_second[i]+inc][0],
                             SA_first.index(IntNodes_temp[Pos_second[i]+inc][2]-inc), Pos_second[i]))
        else:
            if j == 0:
                # Perché altrimenti non posso calcolare SA_first.index
                pairs.append((IntNodes_temp[IntNodes_temp[Pos_second[0]+inc][2]][0],
                              0, Pos_second[0]))
                c = 1
            for i in range(c, len(Pos_second)):
                pairs.append((IntNodes_temp[IntNodes_temp[Pos_second[i]+inc][2]][0],
                              SA_first.index(IntNodes_temp[Pos_second[i]+inc][2]-inc), Pos_second[i]))
        #print("Triple per SA_second:\n", pairs, end="\n\n")

        sortedPairs = copy.deepcopy(pairs)
        # Ordino in base al primo elemento, altrimenti in base al secondo
        if first_iteration:
            sortedPairs.sort(key=lambda x: (x[0], int(x[1])))
        else:
            sortedPairs.sort(key=lambda x: (int(x[0]), int(x[1])))
        #sortedPairs = natsorted(sortedPairs, key=lambda x:(x[0].isdigit(), x))
        #sortedPairs = sorted(sortedPairs, key=lambda x:(type(x[0]), int(x[1])))

        #print("SORTED PAIRS ORDINATE: ", sortedPairs)

        #print("Valori per SA_second ordinati:\n", sortedPairs, end="\n\n")
        for i in range(len(sortedPairs)):
            SA_second[i] = sortedPairs[i][2]
        #print("SA_second:\n", SA_second, end="\n\n")

        if not first_iteration:
            return self.merge(inc, SA_first, SA_second, IntNodes_temp, j, dummy_root, first_iteration)
        else:
            #print("SA_Merge: ", self.merge(inc, SA_first, SA_second, IntNodes_temp, j, dummy_root, first_iteration))
            return IntNodes, self.merge(inc, SA_first, SA_second, IntNodes_temp, j, dummy_root, first_iteration)

    def Compute_Spi_Sort(self, IntNodes, IntNodes_Pos_Sort):
        S_pi = []
        S_pi.append('')  # root
        label_root = IntNodes[0][0]
        for i in range(1, len(IntNodes)):
            spi = ""
            j = IntNodes_Pos_Sort[i]
            while IntNodes[IntNodes[j][2]-1][0] != label_root:
                spi += IntNodes[IntNodes[j][2]-1][0]
                j = IntNodes[j][2]-1
            spi += label_root
            S_pi.append(spi)
        return S_pi

    def Compute_XBWT(self, IntNodes, IntNodes_Pos_Sort):
        # S_last
        pos_last = np.zeros(len(IntNodes), dtype="int")
        for i in range(1, len(IntNodes)):
            pos_last[IntNodes[i][2]-1] = i
        pos_last = list(set(pos_last))[1:]
        S_last = np.zeros(len(IntNodes), dtype='int')
        for i in pos_last:
            S_last[i] = 1

        # S_alpha (bit)
        node_flag = np.ones(len(IntNodes), dtype="int")
        for i in range(1, len(IntNodes)):
            node_flag[IntNodes[i][2]-1] = 0

        S = []
        for i in range(len(IntNodes)):
            S.append(list((S_last[IntNodes_Pos_Sort[i]], [
                     IntNodes[IntNodes_Pos_Sort[i]][0], node_flag[IntNodes_Pos_Sort[i]]])))
        return S
    
    
    def get_SimplerXBWT(self):
        '''
            Get method to recall from comp_layer.py in order to fetch the necessary information for the computation of the Distance
            
            Args:
                self: The reference to the object
            
            Returns:
                SA: The array representing the labels of the XBWT
                Child: The array representing the amount of children each node has
                tree_height: The height of the tree
        '''
        
        SA = self.preorderTraversal(self.getTree().getRoot())
        
        # Child è un po' più complicato
        Child:list[int] = []
        tree_height:int = self.getTree().getHeight()
        
        
        
        return SA, Child, tree_height
        
        ...

# Commentato da Pierfrancesco
'''# Creazione dei nodi dell'albero  
tree = Tree()
root = Node('A')
node1 = Node('B')
node2 = Node('C')
node3 = Node('D')
node4 = Node('E')

tree.insert(root, None)
tree.insert(node1, root)
tree.insert(node2, node1)
tree.insert(node3, node2)
tree.insert(node4, node3)

xbwt = XBWT(tree)
IntNodes, IntNodes_Pos_Sort = xbwt.pathSort(xbwt.getTree())
print(IntNodes_Pos_Sort)'''

"""
# Creazione dei nodi dell'albero  
tree = Tree()
root = Node('A')
node1 = Node('C')
node2 = Node('D')
node3 = Node('D')
node4 = Node('C')
node5 = Node('a')
node6 = Node('B')
node7 = Node('E')
node8 = Node('B')
node9 = Node('a')
node10 = Node('B')
node11 = Node('b')
node12 = Node('a')
node13 = Node('c')
node14 = Node('a')
node15 = Node('c')

tree.insert(root, None)
tree.insert(node1, root)
tree.insert(node2, root)
tree.insert(node3, root)
tree.insert(node4, node1)
tree.insert(node5, node1)
tree.insert(node6, node1)
tree.insert(node7, node2)
tree.insert(node8, node3)
tree.insert(node9, node3)
tree.insert(node10, node3)
tree.insert(node11, node4)
tree.insert(node12, node6)
tree.insert(node13, node7)
tree.insert(node14, node8)
tree.insert(node15, node10)

# Creazione dei nodi dell'albero  
tree2 = Tree()
root = Node('A')
node1 = Node('C')
node2 = Node('E')
node3 = Node('E')
node4 = Node('C')
node5 = Node('a')
node6 = Node('B')
node7 = Node('E')
node8 = Node('B')
node9 = Node('a')
node10 = Node('B')
node11 = Node('a')
node12 = Node('c')
node13 = Node('d')
node14 = Node('a')
node15 = Node('c')

tree2.insert(root, None)
tree2.insert(node1, root)
tree2.insert(node2, root)
tree2.insert(node3, root)
tree2.insert(node4, node1)
tree2.insert(node5, node1)
tree2.insert(node6, node1)
tree2.insert(node7, node2)
tree2.insert(node8, node3)
tree2.insert(node9, node3)
tree2.insert(node10, node3)
tree2.insert(node11, node4)
tree2.insert(node12, node6)
tree2.insert(node13, node7)
tree2.insert(node14, node8)
tree2.insert(node15, node10)

# preorder = tree.preorder(tree.getRoot())
# print(preorder)

# tree.printAllNodes()
"""

"""
xbwt = XBWT(tree)
IntNodes, IntNodes_Pos_Sort = xbwt.pathSort(xbwt.getTree())
xbw = xbwt.Compute_XBWT(IntNodes, IntNodes_Pos_Sort)
S_pi = xbwt.Compute_Spi_Sort(IntNodes, IntNodes_Pos_Sort)
#Xbwt_Edit_Distance(tree, tree)
"""

#Commentato da Pierfrancesco
'''#S_last, S_alpha, S_pi, IntNodes = xbwt.preorderTraversal(xbwt.getTree().getRoot())
#IntNodes = xbwt.computeIntNodesArray(xbwt.getTree().getRoot())


def Export_Tree(tree, path, label=""):
    f = open(os.path.join(path,"t"+label+".txt"), "w+")

    nodes1 = tree.preorder(tree.getRoot())
    f.write("# Dichiarazione nodi \n\n")
    arrayDollars = []
    dictNodes = {}
    i = 0
    f.write("root=Node('"+str(nodes1[i].getLabel())+"')\n")
    dictNodes[nodes1[i].getLabel()] = "root"
    for i in range(1, len(nodes1)):
        f.write("n"+str(i)+"=Node('"+str(nodes1[i].getLabel())+"')\n")
        if nodes1[i].getLabel() == "$":
            arrayDollars.append("n"+str(i))
        else:
            dictNodes[nodes1[i].getLabel()] = "n"+str(i)
    f.write("\n\n")
    f.write("# Inserimento nodi dell'albero\n\n")
    f.write("tree"+label+"=Tree()\n")
    f.write("tree"+label+".insert(root, None)\n")
    j = 0
    for e in tree.getEdges():
        if e[1].getLabel() != "$":
            f.write("tree"+label+".insert(" +
                    dictNodes[e[1].getLabel()]+", "+dictNodes[e[0].getLabel()]+")\n")
        else:
            f.write("tree"+label+".insert(" +
                    arrayDollars[j]+", "+dictNodes[e[0].getLabel()]+")\n")
            j += 1
    f.close()'''


def Export_Tree2(tree, path, label=""):
    newTree = copy.deepcopy(tree)
    f = open(os.path.join(path,"t"+label+".txt"), "w+")
    dictNodes = {}
    nodes = newTree.getNodes()
    dictNodes["root"] = nodes[0].getLabel()
    nodes[0].setLabel("root")
    for i in range(1, len(nodes)):
        dictNodes["n"+str(i)] = nodes[i].getLabel()
        nodes[i].setLabel("n"+str(i))
    f.write("root=Node('"+str(dictNodes["root"])+"')\n")
    for i in range(1, len(nodes)):
        f.write("n"+str(i)+"=Node('"+str(dictNodes["n"+str(i)])+"')\n")
    f.write("\n\n")
    f.write("# Inserimento nodi dell'albero\n\n")
    f.write("tree"+label+"=Tree()\n")
    f.write("tree"+label+".insert(root, None)\n")
    for node in nodes:
        for child in node.getChildren():
            f.write("tree"+label+".insert("+child.getLabel() +
                    ", "+node.getLabel()+")\n")
    f.close()


def Export_Tree3(tree, path, label=""):
    f = open(path+".txt", "w+")
    root = tree.getRoot()
    representation = str(root.representation())
    representation = representation.replace(",", "")
    f.write(representation)
    f.close()
    return representation


def Plot_Exp(xdata, ydata, title, xlabel, ylabel, numexp, prelabel, savepath):

    # print(xdata)
    # print(ydata)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.vlines(xdata, 0, ydata, color="black", linestyle = "dashed", linewidth=0.5)
    #plt.hlines(ydata, 0, xdata, color="black", linestyle = "dashed", linewidth=0.5)

    plt.yticks(np.arange(min(ydata), max(ydata)+2, 2))

    plt.scatter(xdata, ydata, color='r', zorder=2)
    plt.plot(xdata, ydata, color='c', zorder=1)
    path = os.path.join(savepath,prelabel+"_"+str(numexp)+"_PLOT.png")
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.switch_backend('agg')
    plt.show(block=False)
    plt.close()


def Plot_Exp2(xdata, ydata, title, xlabel, ylabel, numexp, prelabel, savepath):

    # print(xdata)
    # print(ydata)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.vlines(xdata, 0, ydata, color="black", linestyle = "dashed", linewidth=0.5)
    #plt.hlines(ydata, 0, xdata, color="black", linestyle = "dashed", linewidth=0.5)

    plt.yticks(np.arange(min(ydata), max(ydata)+2, 2))

    plt.scatter(xdata, ydata, color='r', zorder=2, s=2)
    plt.plot(xdata, ydata, color='c', zorder=1)
    path = os.path.join(savepath,prelabel+"_"+str(numexp)+"_PLOT.png")
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.switch_backend('agg')
    plt.show(block=False)
    plt.close()


def Draw_Tree(tree, num_exp, label_exp, label_tree, path):
    G = nx.DiGraph()
    i = 0
    for e in tree.getEdges():
        if e[1].getLabel() == "$":
            _str = str(i)
            _str = _str.translate(sub)
            G.add_edge(e[0].getLabel(), "$"+_str)
            i += 1
        else:
            G.add_edge(e[0].getLabel(), e[1].getLabel())
    pos = graphviz_layout(G, prog='dot')
    figure(figsize=(13, 6))
    nx.draw(G, pos, node_size=220, with_labels=True,
            arrows=True, node_color='w', font_size=14)
    plt.title("ALBERO "+label_tree+" - ESPERIMENTO " +
              label_exp+" "+str(num_exp))
    path = os.path.join(path, "tree"+label_tree+"_plot.png")
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.switch_backend('agg')
    plt.show(block=False)
    plt.close()


def Draw_Tree2(tree, num_exp, label_tree, path):
    t1 = svgling.draw_tree(nltk.Tree.fromstring(tree))
    t1.get_svg().saveas(os.path.join(path,"tree"+label_tree+"_plot.svg"))

# Commentato da Pierfrancesco
'''numero_esperimenti = 50'''

"""
path = os.path.join(os.getcwd(),"Esperimenti","Rimozioni")
print(path)

# Rimozioni
print("ESPERIMENTI - RIMOZIONI SOTTOALBERI")
for e in tqdm(range(1, numero_esperimenti+1)):
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(alphabet)
    tree2, removals, size_sub_rem, removals_array, distances = Remove_Subtrees(tree, 10, newpath)
    Draw_Tree(tree, e, "RIMOZIONI SOTTOALBERI", "1", newpath)
    Draw_Tree(tree2, e, "RIMOZIONI SOTTOALBERI", "2", newpath)
    Export_Tree(tree, newpath)
    Export_Tree(tree2, newpath, "2")
    asp_distance = sum(size_sub_rem)
    real_distance = Xbwt_Edit_Distance(tree, tree2)
    f= open(os.path.join(newpath,"EXP_REM_"+str(e)+"_DETAILS.txt"),"w+")
    f2 = open(os.path.join(newpath,"EXP_REM_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - RIMOZIONI SOTTOALBERI *****\n\n")
    f.write("Dimensione albero 1: "+str(len(tree.getNodes()))+"\n")
    f.write("Numero sottoalberi rimossi: "+str(removals)+"\n")
    f.write("Dimensioni sottoalberi rimossi: "+str(size_sub_rem)+"\n")
    f.write("Dimensione albero 2: "+str(len(tree2.getNodes()))+"\n")
    f.write("Misura aspettata: "+str(asp_distance)+"\n")
    #f.write("-------------------------------------------------------")
    f.write("d_xbw: "+str(real_distance))
    f.close()
    f2.write("***** ESPERIMENTO "+str(e)+" - RIMOZIONE SOTTOALBERI - DATI DA PLOTTARE *****\n\n")
    f2.write("Numero di rimozioni | Valore misura\n")
    for i in range(len(distances)):
        f2.write(str(removals_array[i])+" "+str(distances[i])+"\n")
    f2.close()
    Plot_Exp(removals_array, distances, "ESPERIMENTO "+str(e)+" - RIMOZIONE SOTTOALBERI", "NUMERO RIMOZIONI", "VALORE MISURA", e, "EXP_REM", newpath)
"""

"""
# Dichiarazione nodi 

root=Node('J')
n1=Node('C')
n2=Node('H')
n3=Node('A')
n4=Node('E')
n5=Node('F')
n6=Node('B')
n7=Node('I')
n8=Node('D')
n9=Node('G')
n10=Node('$')
n11=Node('$')
n12=Node('$')
n13=Node('$')


# Inserimento nodi dell'albero

tree=Tree()
tree.insert(root, None)
tree.insert(n1, root)
tree.insert(n2, root)
tree.insert(n3, n1)
tree.insert(n4, n2)
tree.insert(n5, n2)
tree.insert(n6, n2)
tree.insert(n7, n3)
tree.insert(n8, n4)
tree.insert(n9, n5)
tree.insert(n10, n6)
tree.insert(n11, n7)
tree.insert(n12, n8)
tree.insert(n13, n9)
"""
"""
# Dichiarazione nodi 

root=Node('J')
n1=Node('C')
n2=Node('H')
n3=Node('A')
n4=Node('E')
n5=Node('F')
n6=Node('B')
n7=Node('I')
n8=Node('D')
n9=Node('G')
n10=Node('$')
n11=Node('$')
n12=Node('$')
n13=Node('$')


# Inserimento nodi dell'albero

tree2=Tree()
tree2.insert(root, None)
tree2.insert(n8, root)
tree2.insert(n2, root)
tree2.insert(n10, n8)
tree2.insert(n9, n2)
tree2.insert(n7, n2)
tree2.insert(n3, n2)
tree2.insert(n11, n9)
tree2.insert(n12, n7)
tree2.insert(n5, n3)
tree2.insert(n4, n5)
tree2.insert(n1, n4)
tree2.insert(n6, n1)
tree2.insert(n13, n6)
"""

"""
xbwt1 = XBWT(tree)
IntNodes1, IntNodes_Pos_Sort1 = xbwt1.pathSort(xbwt1.getTree())

#print(IntNodes1, end="\n\n")
#print(IntNodes_Pos_Sort1, end="\n\n")

xbw1 = xbwt1.Compute_XBWT(IntNodes1, IntNodes_Pos_Sort1)
"""
"""
xbwt2 = XBWT(tree2)
IntNodes2, IntNodes_Pos_Sort2 = xbwt2.pathSort(xbwt2.getTree())

#print(IntNodes1, end="\n\n")
#print(IntNodes_Pos_Sort1, end="\n\n")

xbw2 = xbwt2.Compute_XBWT(IntNodes2, IntNodes_Pos_Sort2)

#print(xbw1, end="\n\n")
print(xbw2)

"""
#distance = Xbwt_Edit_Distance(tree, tree2)
#print("Distanza: ", distance)

"""
path = os.path.join(os.getcwd(),"Esperimenti","Scambi sottoalberi")
print(path)

# Scambi di sottoalberi
print("ESPERIMENTI - SCAMBI SOTTOALBERI 1")
dictAverage={}
dictSwaps= {}
for e in tqdm(range(1, numero_esperimenti+1)):
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(alphabet)           
    tree2, swaps, swaps_array, distances = Swap_Subtrees(tree)
    Draw_Tree(tree, e, "SCAMBI SOTTOALBERI", "1", newpath)
    Draw_Tree(tree2, e, "SCAMBI SOTTOALBERI", "2", newpath)
    Export_Tree(tree, newpath)
    Export_Tree(tree2, newpath, "2")
    asp_distance = 2*swaps
    real_distance = Xbwt_Edit_Distance(tree, tree2)
    dimTree2 = 0
    for n in tree2.getNodes():
        if n.getLabel() != "$":
            dimTree2+=1
    f= open(os.path.join(newpath,"EXP_SST_"+str(e)+"_DETAILS.txt"),"w+")
    f2 = open(os.path.join(newpath,"EXP_SST_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1*****\n\n")
    f.write("Numero sottoalberi scambiati: "+str(swaps)+"\n")
    f.write("Misura aspettata: "+str(asp_distance)+"\n")
    f.write("d_xbw: "+str(real_distance))
    f2.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1 - DATI DA PLOTTARE *****\n\n")
    f2.write("Numero di scambi | Valore misura\n")
    for s in range(1, swaps+1):
        if s in dictSwaps.keys():
            dictSwaps[s]+=1
        else:
            dictSwaps[s]=1
    for i in range(len(distances)):
        if i+1 in dictAverage.keys():
            dictAverage[i+1] = dictAverage[i+1] + distances[i]
        else:
            dictAverage[i+1] = distances[i]
        f2.write(str(swaps_array[i])+" "+str(distances[i])+"\n")
    print("Distanze: ", distances, )
    Plot_Exp(swaps_array, distances, "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT", newpath)
    f.close()
    f2.close()
f3 = open(os.path.join(path,"EXP_SST_DATA_AVG_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
f3.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1 - DATI DA PLOTTARE (AVG) *****\n\n")
f3.write("Numero di esperimenti: "+str(numero_esperimenti)+"\n\n")
f3.write("Numero di scambi | Somma misure | Media | Media 2\n")
final_swaps_array = []
avg_distances = []
avg_distances2 = []
for k in dictAverage.keys():
    final_swaps_array.append(k)
    avg_distances.append(dictAverage[k]/numero_esperimenti)
    avg_distances2.append(dictAverage[k]/dictSwaps[k])
    f3.write(str(k)+" "+str(dictAverage[k])+" "+str(dictAverage[k]/numero_esperimenti)+" "+str(dictAverage[k]/dictSwaps[k])+"\n")
Plot_Exp(final_swaps_array, avg_distances, "ESPERIMENTO - SCAMBI SOTTOALBERI 1 (AVG)", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG", path)
Plot_Exp(final_swaps_array, avg_distances2, "ESPERIMENTO - SCAMBI SOTTOALBERI 1 (AVG) - 2", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG_2", path)
f3.close()
"""

"""
path = os.path.join(os.getcwd(),"Esperimenti","Scambi sottoalberi 2")
print(path)

# Scambi di sottoalberi
print("ESPERIMENTI - SCAMBI SOTTOALBERI")
dictAverage={}
dictSwaps= {}
for e in tqdm(range(1, numero_esperimenti+1)):
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(alphabet)           
    tree2, swaps, swaps_array, distances = Swap_Subtrees2(tree)
    Draw_Tree(tree, e, "SCAMBI SOTTOALBERI", "1", newpath)
    Draw_Tree(tree2, e, "SCAMBI SOTTOALBERI", "2", newpath)
    Export_Tree(tree, newpath)
    Export_Tree(tree2, newpath, "2")
    asp_distance = 2*swaps
    real_distance = Xbwt_Edit_Distance(tree, tree2)
    dimTree2 = 0
    for n in tree2.getNodes():
        if n.getLabel() != "$":
            dimTree2+=1
    f= open(os.path.join(newpath,"EXP_SST_"+str(e)+"_DETAILS.txt"),"w+")
    f2 = open(os.path.join(newpath,"EXP_SST_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 2*****\n\n")
    f.write("Numero sottoalberi scambiati: "+str(swaps)+"\n")
    f.write("Misura aspettata: "+str(asp_distance)+"\n")
    f.write("d_xbw: "+str(real_distance))
    f2.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 2 - DATI DA PLOTTARE *****\n\n")
    f2.write("Numero di scambi | Valore misura\n")
    for s in range(1, swaps+1):
        if s in dictSwaps.keys():
            dictSwaps[s]+=1
        else:
            dictSwaps[s]=1
    for i in range(len(distances)):
        if i+1 in dictAverage.keys():
            dictAverage[i+1] = dictAverage[i+1] + distances[i]
        else:
            dictAverage[i+1] = distances[i]
        f2.write(str(swaps_array[i])+" "+str(distances[i])+"\n")
    Plot_Exp(swaps_array, distances, "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 2", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT", newpath)
    #f.write("-------------------------------------------------------")
    f.close()
    f2.close()
f3 = open(os.path.join(path,"EXP_SST_DATA_AVG_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
f3.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 2 - DATI DA PLOTTARE (AVG) *****\n\n")
f3.write("Numero di esperimenti: "+str(numero_esperimenti)+"\n\n")
f3.write("Numero di scambi | Somma misure | Media | Media 2\n")
final_swaps_array = []
avg_distances = []
avg_distances2 = []
for k in dictAverage.keys():
    final_swaps_array.append(k)
    avg_distances.append(dictAverage[k]/numero_esperimenti)
    avg_distances2.append(dictAverage[k]/dictSwaps[k])
    f3.write(str(k)+" "+str(dictAverage[k])+" "+str(dictAverage[k]/numero_esperimenti)+" "+str(dictAverage[k]/dictSwaps[k])+"\n")
Plot_Exp(final_swaps_array, avg_distances, "ESPERIMENTO - SCAMBI SOTTOALBERI 2 (AVG)", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG", path)
Plot_Exp(final_swaps_array, avg_distances2, "ESPERIMENTO - SCAMBI SOTTOALBERI 2 (AVG) - 2", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG_2", path)
f3.close()
"""

"""
path = os.path.join(os.getcwd(),"Esperimenti","Scambi sottoalberi 3")
print(path)

# Scambi di sottoalberi
print("ESPERIMENTI - SCAMBI SOTTOALBERI")
dictAverage={}
dictSwaps= {}
for e in tqdm(range(1, numero_esperimenti+1)):
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(alphabet)           
    tree2, swaps, swaps_array, distances = Swap_Subtrees3(tree)
    Draw_Tree(tree, e, "SCAMBI SOTTOALBERI", "1", newpath)
    Draw_Tree(tree2, e, "SCAMBI SOTTOALBERI", "2", newpath)
    Export_Tree(tree, newpath)
    Export_Tree(tree2, newpath, "2")
    asp_distance = 2*swaps
    real_distance = Xbwt_Edit_Distance(tree, tree2)
    dimTree2 = 0
    for n in tree2.getNodes():
        if n.getLabel() != "$":
            dimTree2+=1
    f= open(os.path.join(newpath,"EXP_SST_"+str(e)+"_DETAILS.txt"),"w+")
    f2 = open(os.path.join(newpath,"EXP_SST_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 3*****\n\n")
    f.write("Numero sottoalberi scambiati: "+str(swaps)+"\n")
    f.write("Misura aspettata: "+str(asp_distance)+"\n")
    f.write("d_xbw: "+str(real_distance))
    f2.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 3 - DATI DA PLOTTARE *****\n\n")
    f2.write("Numero di scambi | Valore misura\n")
    for s in range(1, swaps+1):
        if s in dictSwaps.keys():
            dictSwaps[s]+=1
        else:
            dictSwaps[s]=1
    for i in range(len(distances)):
        if i+1 in dictAverage.keys():
            dictAverage[i+1] = dictAverage[i+1] + distances[i]
        else:
            dictAverage[i+1] = distances[i]
        f2.write(str(swaps_array[i])+" "+str(distances[i])+"\n")
    Plot_Exp2(swaps_array, distances, "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 3", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT", newpath)
    #f.write("-------------------------------------------------------")
    f.close()
    f2.close()
f3 = open(os.path.join(newpath,"EXP_SST_DATA_AVG_TO_PLOT_DETAILS.txt"),"w+")
f3.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 3 - DATI DA PLOTTARE (AVG) *****\n\n")
f3.write("Numero di esperimenti: "+str(numero_esperimenti)+"\n\n")
f3.write("Numero di scambi | Somma misure | Media | Media 2\n")
final_swaps_array = []
avg_distances = []
avg_distances2 = []
for k in dictAverage.keys():
    final_swaps_array.append(k)
    avg_distances.append(dictAverage[k]/numero_esperimenti)
    avg_distances2.append(dictAverage[k]/dictSwaps[k])
    f3.write(str(k)+" "+str(dictAverage[k])+" "+str(dictAverage[k]/numero_esperimenti)+" "+str(dictAverage[k]/dictSwaps[k])+"\n")
Plot_Exp2(final_swaps_array, avg_distances, "ESPERIMENTO - SCAMBI SOTTOALBERI 3 (AVG)", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG", path)
Plot_Exp2(final_swaps_array, avg_distances2, "ESPERIMENTO - SCAMBI SOTTOALBERI 3 (AVG) - 2", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG_2", path)
f3.close()
"""

# Commentato da Pierfrancesco
'''path = os.path.join(os.getcwd(),"Esperimenti","Scambi sottoalberi 4")
print(path)

# Scambi di sottoalberi
print("ESPERIMENTI - SCAMBI SOTTOALBERI")
dictAverage={}
dictSwaps= {}
for e in tqdm(range(1, numero_esperimenti+1)):
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(alphabet)           
    tree2, swaps, swaps_array, distances = Swap_Subtrees3(tree)
    Draw_Tree(tree, e, "SCAMBI SOTTOALBERI", "1", newpath)
    Draw_Tree(tree2, e, "SCAMBI SOTTOALBERI", "2", newpath)
    Export_Tree(tree, newpath)
    Export_Tree(tree2, newpath, "2")
    asp_distance = 2*swaps
    real_distance = Xbwt_Edit_Distance(tree, tree2)
    dimTree2 = 0
    for n in tree2.getNodes():
        if n.getLabel() != "$":
            dimTree2+=1
    f= open(os.path.join(newpath,"EXP_SST_"+str(e)+"_DETAILS.txt"),"w+")
    f2 = open(os.path.join(newpath,"EXP_SST_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 4*****\n\n")
    f.write("Numero sottoalberi scambiati: "+str(swaps)+"\n")
    f.write("Misura aspettata: "+str(asp_distance)+"\n")
    f.write("d_xbw: "+str(real_distance))
    f2.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 4 - DATI DA PLOTTARE *****\n\n")
    f2.write("Numero di scambi | Valore misura\n")
    for s in range(1, swaps+1):
        if s in dictSwaps.keys():
            dictSwaps[s]+=1
        else:
            dictSwaps[s]=1
    for i in range(len(distances)):
        if i+1 in dictAverage.keys():
            dictAverage[i+1] = dictAverage[i+1] + distances[i]
        else:
            dictAverage[i+1] = distances[i]
        f2.write(str(swaps_array[i])+" "+str(distances[i])+"\n")
    Plot_Exp2(swaps_array, distances, "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 4", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT", newpath)
    #f.write("-------------------------------------------------------")
    f.close()
    f2.close()
f3 = open(os.path.join(path,"EXP_SST_DATA_AVG_TO_PLOT_DETAILS.txt"),"w+")
f3.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 4 - DATI DA PLOTTARE (AVG) *****\n\n")
f3.write("Numero di esperimenti: "+str(numero_esperimenti)+"\n\n")
f3.write("Numero di scambi | Somma misure | Media | Media 2\n")
final_swaps_array = []
avg_distances = []
avg_distances2 = []
for k in dictAverage.keys():
    final_swaps_array.append(k)
    avg_distances.append(dictAverage[k]/numero_esperimenti)
    avg_distances2.append(dictAverage[k]/dictSwaps[k])
    f3.write(str(k)+" "+str(dictAverage[k])+" "+str(dictAverage[k]/numero_esperimenti)+" "+str(dictAverage[k]/dictSwaps[k])+"\n")
Plot_Exp2(final_swaps_array, avg_distances, "ESPERIMENTO - SCAMBI SOTTOALBERI 4 (AVG)", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG", path)
Plot_Exp2(final_swaps_array, avg_distances2, "ESPERIMENTO - SCAMBI SOTTOALBERI 4 (AVG) - 2", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG_2", path)
f3.close()'''

"""
path = os.getcwd()+"\\Esperimenti\\Scambi simboli"
print(path)

# Scambi di simboli
print("ESPERIMENTI - SCAMBI SIMBOLI")
for e in tqdm(range(1, numero_esperimenti+1)):
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(alphabet)           
    tree2, swaps, maxDegFor2, pcSwaps, rem = Swap_Symbols(tree, 15)
    real_distance = Xbwt_Edit_Distance(tree, tree2)
    f= open(os.path.join(newpath,"EXP_SS_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SIMBOLI *****\n\n")
    f.write("Numero simboli scambiati: "+str(swaps*2)+"\n")
    f.write("Max degree per 2 per ogni coppia di simboli scambiati: "+str(maxDegFor2)+"\n")
    f.write("Numero coppie genitore-figlio scambiate: "+str(pcSwaps)+"\n")
    f.write("Numero coppie genitore-figlio con la stessa configurazione di partenza dopo lo scambio: "+str(rem)+"\n")
    if (sum(maxDegFor2)-pcSwaps-rem) < 0:
        f.write("Quantità da togliere alla distanza: "+str(0)+"\n")
        f.write("Quantità da aggiungere alla distanza: "+str(int(sum(maxDegFor2)-pcSwaps-rem))+"\n")
    else:
        f.write("Quantità da togliere alla distanza: "+str(sum(maxDegFor2)-pcSwaps-rem)+"\n")
        f.write("Quantità da aggiungere alla distanza: "+str(0)+"\n")
    f.write("xbwt-edit-distance: "+str(real_distance))
    #f.write("-------------------------------------------------------")
    f.close()
    Draw_Tree(tree, e, "SCAMBI SIMBOLI", "1", newpath)
    Draw_Tree(tree2, e, "SCAMBI SIMBOLI", "2", newpath)
    Export_Tree(tree, newpath)
    Export_Tree(tree2, newpath, "2")
"""

"""
numero_esperimenti = 2
path = os.getcwd()+"\\Esperimenti 2\\Etichette multiple"

for e in tqdm(range(1, numero_esperimenti+1)):
    num_labels = random.randint(2, 9)
    new_alphabet = copy.deepcopy(alphabet)
    random.shuffle(new_alphabet)
    
    for i in range(num_labels):
        for j in range(4):
            new_alphabet.append(new_alphabet[i])
    random.shuffle(new_alphabet)
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(new_alphabet)
    tree2, swaps, swaps_array, distances = Swap_Subtrees(tree)  
    t1 = Export_Tree3(tree, newpath)
    t2 = Export_Tree3(tree2, newpath, "2")
    Draw_Tree2(t1, e, "", newpath)
    Draw_Tree2(t2, e, "2", newpath)
    asp_distance = 2*swaps
    f= open(os.path.join(newpath,"EXP_SST_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1 (ETICHETTE MULTIPLE) *****\n\n")
    f.write("Numero sottoalberi scambiati: "+str(swaps)+"\n")
    f.write("Distanza aspettata: "+str(asp_distance)+"\n")
    #f.write("xbwt-edit-distance: "+str(real_distance))
    f.close()
    mergeXBWT(tree, tree2, newpath)
"""

"""
numero_esperimenti = 200
path = os.path.join(os.getcwd(),"Esperimenti 3","Confronto alberi casuali")
f1 = open(os.path.join(path,"Riepilogo.txt"), "w+")
for e in tqdm(range(1, numero_esperimenti+1)):
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(alphabet)
    tree2 = Generate_Random_Tree(alphabet)
    t1 = Export_Tree3(tree, newpath)
    t2 = Export_Tree3(tree2, newpath, "2")
    Draw_Tree2(t1, e, "", newpath)
    Draw_Tree2(t2, e, "2", newpath)
    f= open(os.path.join(newpath,"EXP_RTC_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - CONFRONTO ALBERI CASUALI CON LE STESSE ETICHETTE *****\n\n")
    dist = Xbwt_Edit_Distance(tree, tree2)
    f.write("Distanza ottenuta: "+str(dist)+"\n")
    f1.write("EXP "+str(e)+": "+str(dist)+"\n")
    f.close()
f1.close()
"""

"""
numero_esperimenti = 200
path = os.path.join(os.getcwd(),"Esperimenti 3","Confronto alberi casuali con dimensione inferiore")
f1 = open(os.path.join(path,"Riepilogo.txt"), "w+")
for e in tqdm(range(1, numero_esperimenti+1)):
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    tree = Generate_Random_Tree(alphabet)
    alphabet = ["A", "B", "C", "D", "E"]
    tree2 = Generate_Random_Tree(alphabet)
    t1 = Export_Tree3(tree, newpath)
    t2 = Export_Tree3(tree2, newpath, "2")
    Draw_Tree2(t1, e, "", newpath)
    Draw_Tree2(t2, e, "2", newpath)
    f= open(os.path.join(newpath,"EXP_RTC_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - CONFRONTO ALBERI CASUALI CON LE STESSE ETICHETTE *****\n\n")
    dist = Xbwt_Edit_Distance(tree, tree2)
    f.write("Distanza ottenuta: "+str(dist)+"\n")
    f1.write("EXP "+str(e)+": "+str(dist)+"\n")
    f.close()
f1.close()
"""

"""
numero_esperimenti = 200
path = os.path.join(os.getcwd(),"Esperimenti 3","Confronto alberi casuali con dimensione superiore")
f1 = open(os.path.join(path,"Riepilogo.txt"), "w+")
for e in tqdm(range(1, numero_esperimenti+1)):
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    alphabet = ["A", "B", "C", "D", "E"]
    tree = Generate_Random_Tree(alphabet)
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    tree2 = Generate_Random_Tree(alphabet)
    t1 = Export_Tree3(tree, newpath)
    t2 = Export_Tree3(tree2, newpath, "2")
    Draw_Tree2(t1, e, "", newpath)
    Draw_Tree2(t2, e, "2", newpath)
    f= open(os.path.join(newpath,"EXP_RTC_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - CONFRONTO ALBERI CASUALI CON LE STESSE ETICHETTE *****\n\n")
    dist = Xbwt_Edit_Distance(tree, tree2)
    f.write("Distanza ottenuta: "+str(dist)+"\n")
    f1.write("EXP "+str(e)+": "+str(dist)+"\n")
    f.close()
f1.close()
"""

""" ESPERIMENTI CON ETICHETTE MULTIPLE """


def Remove_Subtrees2(T, maxRem, path):
    T0 = copy.deepcopy(T)
    # La ricostruzione funziona ma non funziona il preordine
    distances = []
    removals_array = []
    nremovals = random.randint(1, maxRem)
    nodeTemp = []
    size_sub_rem = []
    #print("Numero random estratto: ", nremovals)
    subtrees = get_all_subtree(T)
    sbt_dim_real = []
    for st in subtrees:
        dimTmp = 0
        if len(st) >= 1:
            for e in st:
                dimTmp += 1
            # print("\n\n")
            sbt_dim_real.append(dimTmp+1)
        else:
            sbt_dim_real.append(0)
    # Non considerare il primo sottoalbero (cioè quello della radice)
    subtrees_dim = [len(e) for e in subtrees]

    #print("Dimensioni sottoalberi 1:", subtrees_dim)
    firstEmptySubtreeIndex = list(subtrees_dim).index(0)
    # print(firstEmptySubtreeIndex)
    posSubtree = random.randint(1, firstEmptySubtreeIndex-1)
    f = open(os.path.join(path,"SUBTREES_REMOVED.txt"), "w+")
    f.write("SOTTOALBERI RIMOSSI\n\n")
    f.write(str(subtrees[posSubtree][0][0].getLabel())+"\n")
    for e in subtrees[posSubtree]:
        print(e[0].getLabel(), e[1].getLabel())
    print("Nodi rimossi")
    #print("Subtrees_dim 1: ", subtrees_dim)
    #print("Subtrees_dim 2: ", sbt_dim_real)
    #print("Possubtree: ", posSubtree)
    size_sub_rem.append(sbt_dim_real[posSubtree])
    r = 0
    # print(edges)
    removals = 0
    newTree = None
    while r < nremovals:
        edgesToRemove = []
        edges = T.getEdges()

        #print("Rimozione: ", r+1)
        edgesToRemove.append(
            (subtrees[posSubtree][0][0].getParent(), subtrees[posSubtree][0][0]))
        print("EdgesToRemove: ", edgesToRemove)
        print("EdgesToRemove 2: ", [
              (g[0].getLabel(), g[1].getLabel()) for g in edgesToRemove])
        print("Edges: ", edges)
        #print("Eccolo: ", edgesToRemove)
        for st in subtrees[posSubtree]:
            edgesToRemove.append(st)
        newEdges = [(x[0], x[1]) for x in edges if x not in edgesToRemove]
        print("\nNewEdges: ", [
              (g[0].getLabel(), g[1].getLabel()) for g in newEdges])
        print("\n NewEdges 2: ", newEdges)
        print("\nRoot: ", T.getRoot())

        newNodes = {}
        newNodes[T.getRoot()] = Node(T.getRoot().getLabel())
        for e in newEdges:
            if e[0] not in newNodes.keys():
                newNodes[e[0]] = Node(e[0].getLabel())
            if e[1] not in newNodes.keys():
                newNodes[e[1]] = Node(e[1].getLabel())

        newTree = Tree()
        # print(dictNodes)
        # Inserimento della radice
        newTree.insert(newNodes[list(newNodes.keys())[0]], None)
        # Inserisco i nuovi archi all'albero
        print("\n")
        for e in newEdges:
            print(e, e[1].getLabel(), e[0].getLabel())
            newTree.insert(newNodes[e[1]], newNodes[e[0]])
        r += 1
        removals += 1
        # Aggiungo i dollari alle foglie
        for node in newTree.getNodes():
            #print("Nodo: ", node.getLabel(), [n.getLabel() for n in node.getChildren()])
            if len(node.getChildren()) == 0 and node.getLabel() != "$":
                nodeTemp.append(node.getLabel())
                newTree.insert(Node('$'), node)
        removals_array.append(removals)
        print("Numero nodi nuovo albero: ", len(newTree.getNodes()))
        print("Preordine: ", [g.getLabel()
              for g in newTree.preorder(newTree.getRoot())])
        distances.append(Xbwt_Edit_Distance2(T0, newTree))
        print("Distanza temp: ", distances[len(distances)-1])
        if r < nremovals:
            T = copy.deepcopy(newTree)
            #print("Preordine 1:", T.preorder(T.getRoot()))
            subtrees = get_all_subtree(T)
            sbt_dim_real = []
            for st in subtrees:
                dimTmp = 0
                if len(st) >= 1:
                    for e in st:
                        #print((e[0].getLabel(), e[1].getLabel()))
                        """
                        if e[1].getLabel() != "$":
                            dimTmp+=1
                        """
                        dimTmp += 1
                    # print("\n")
                    sbt_dim_real.append(dimTmp+1)
                else:
                    sbt_dim_real.append(0)
            # Non considerare il primo sottoalbero (cioè quello della radice)
            subtrees_dim = [len(e) for e in subtrees]
            #print("Dimensioni sottoalberi: ", subtrees_dim)
            print(subtrees_dim)
            if list(subtrees_dim).index(0) > 1:
                firstEmptySubtreeIndex = list(subtrees_dim).index(0)
                # print(firstEmptySubtreeIndex)
                posSubtree = random.randint(1, firstEmptySubtreeIndex-1)
                dec = 0
                f.write(str(subtrees[posSubtree][0][0].getLabel())+"\n")
                for e in subtrees[posSubtree]:
                    if e[1].getLabel() == '$' and e[1].getParent().getLabel() in nodeTemp:
                        #print("Sono nell'if atteso")
                        dec += 1  # conteggio dollari
                size_sub_rem.append(sbt_dim_real[posSubtree]-dec)
                #print("Subtrees_dim 1: ", subtrees_dim)
                #print("Subtrees_dim 2: ", sbt_dim_real)
                for e in subtrees[posSubtree]:
                    print(e[0].getLabel(), e[1].getLabel())
                print("Nodi rimossi")
            else:
                #print("Fine rimozioni")
                r = nremovals
        #print("Numero di rimozioni effettuate: ", removals)
        #print("Dimensioni sottoalberi rimossi: ", size_sub_rem)
        # print(newTree.preorder(newTree.getRoot()))
    return newTree, removals, size_sub_rem, removals_array, distances

#Should you need this function, just change the if condition to True
if False:
    def Swap_Subtrees2(T):
        #nswaps = random.randint(1, maxSwaps)
        distances = []
        swaps_array = []
        newTree = copy.deepcopy(T)
        subtrees = get_all_subtree(newTree)
        dictSubtrees = {}
        dictParents = {}
        # Per ogni radice di sottoalbero allego il rispettivo sottoalbero
        for st in subtrees[1:len(subtrees)]:
            if len(st) > 0:
                dictSubtrees[st[0][0]] = st
                parents = []
                parent = st[0][0]
                while parent.getParent() is not None:
                    parents.append(parent.getParent())
                    parent = parent.getParent()
                dictParents[st[0][0]] = parents
        exchangeable = []
        keys = list(dictSubtrees.keys())
        i = 1
        # Ricavo i sottoalberi scambiabili
        for key in keys:
            if i <= len(keys)-2:
                for k in keys[i:]:
                    if key not in dictParents[k]:
                        exchangeable.append([key, k])
            i += 1
        dictExchanged = {}
        for k in keys:
            dictExchanged[k] = False
        random.shuffle(exchangeable)
        swaps = 0
        i = 0
        nswaps = len(exchangeable)
        while swaps < nswaps and i < len(exchangeable):
            pair = exchangeable[i]
            if not dictExchanged[pair[0]] and not dictExchanged[pair[1]]:
                if pair[1] not in dictParents[pair[0]] and pair[0] not in dictParents[pair[1]]:
                    print("Sto scambiando: ",
                        pair[0].getLabel(), pair[1].getLabel())
                    swaps += 1
                    dictExchanged[pair[0]] = True
                    dictExchanged[pair[1]] = True
                    p1 = pair[0].getParent()
                    tmp = p1.getChildren()
                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[0]:
                            tmp[j] = pair[1]
                            break
                    p1.setChildren(tmp)
                    p2 = pair[1].getParent()
                    tmp = p2.getChildren()
                    if p2 == pair[0].getParent():
                        # Nel caso in cui sto scambiando fratelli
                        flag = False
                        for j in range(0, len(tmp)):
                            if tmp[j] == pair[1]:
                                if flag:
                                    tmp[j] = pair[0]
                                    break
                                flag = True
                    else:
                        for j in range(0, len(tmp)):
                            if tmp[j] == pair[1]:
                                tmp[j] = pair[0]
                                break
                    p2.setChildren(tmp)
                    pair[0].setParent(p2)
                    pair[1].setParent(p1)
    
                    # Aggiorno gli antenati per ogni nodo
                    for node in newTree.getNodes():
                        parents = []
                        parent = node
                        while parent.getParent() is not None:
                            parents.append(parent.getParent())
                            parent = parent.getParent()
                        dictParents[node] = parents
    
                        newEdges = []
                        preorder = newTree.preorder(newTree.getRoot())
                        for n in preorder:
                            for c in n.getChildren():
                                newEdges.append((n, c))
                        newTree.setEdges(newEdges)
    
                    swaps_array.append(swaps)
                    distances.append(Xbwt_Edit_Distance2(T, newTree))
    
            i += 1
        #print("Numero di scambi effettuati: ", swaps)
        # print(newTree.preorder(newTree.getRoot()))
    
        """
        for node in preorder:
            if node.getParent()is not None:
                print(node.getLabel(), node.getParent().getLabel())
            else:
                print(node.getLabel())
        """
    
        return newTree, swaps, swaps_array, distances
    
    
    def Swap_Subtrees2_2(T):
        #nswaps = random.randint(1, maxSwaps)
        distances = []
        swaps_array = []
        newTree = copy.deepcopy(T)
        subtrees = get_all_subtree(newTree)
        dictSubtrees = {}
        dictParents = {}
        # Per ogni radice di sottoalbero allego il rispettivo sottoalbero
        for st in subtrees[1:len(subtrees)]:
            if len(st) > 0:
                dictSubtrees[st[0][0]] = st
                parents = []
                parent = st[0][0]
                while parent.getParent() is not None:
                    parents.append(parent.getParent())
                    parent = parent.getParent()
                dictParents[st[0][0]] = parents
        exchangeable = []
        keys = list(dictSubtrees.keys())
        i = 1
        # Ricavo i sottoalberi scambiabili
        for key in keys:
            if i <= len(keys)-2:
                for k in keys[i:]:
                    if key not in dictParents[k]:
                        exchangeable.append([key, k])
            i += 1
        dictExchanged = {}
        for k in keys:
            dictExchanged[k] = False
        random.shuffle(exchangeable)
        swaps = 0
        i = 0
        nswaps = 10  # Numero di scambi da effettuare
        while swaps < nswaps and i < len(exchangeable):
            pair = exchangeable[i]
            # if not dictExchanged[pair[0]] and not dictExchanged[pair[1]]:
            if pair[1] not in dictParents[pair[0]] and pair[0] not in dictParents[pair[1]]:
                print("Sto scambiando: ", pair[0].getLabel(), pair[1].getLabel())
                swaps += 1
                dictExchanged[pair[0]] = True
                dictExchanged[pair[1]] = True
                p1 = pair[0].getParent()
                tmp = p1.getChildren()
                for j in range(0, len(tmp)):
                    if tmp[j] == pair[0]:
                        tmp[j] = pair[1]
                        break
                p1.setChildren(tmp)
                p2 = pair[1].getParent()
                tmp = p2.getChildren()
                if p2 == pair[0].getParent():
                    # Nel caso in cui sto scambiando fratelli
                    flag = False
                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[1]:
                            if flag:
                                tmp[j] = pair[0]
                                break
                            flag = True
                else:
                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[1]:
                            tmp[j] = pair[0]
                            break
                p2.setChildren(tmp)
                pair[0].setParent(p2)
                pair[1].setParent(p1)
    
                # Aggiorno gli antenati per ogni nodo
                for node in newTree.getNodes():
                    parents = []
                    parent = node
                    while parent.getParent() is not None:
                        parents.append(parent.getParent())
                        parent = parent.getParent()
                    dictParents[node] = parents
    
                newEdges = []
                preorder = newTree.preorder(newTree.getRoot())
                for n in preorder:
                    for c in n.getChildren():
                        newEdges.append((n, c))
                newTree.setEdges(newEdges)
    
                swaps_array.append(swaps)
                distances.append(Xbwt_Edit_Distance2(T, newTree))
    
            i += 1
        #print("Numero di scambi effettuati: ", swaps)
        # print(newTree.preorder(newTree.getRoot()))
    
        """
        for node in preorder:
            if node.getParent()is not None:
                print(node.getLabel(), node.getParent().getLabel())
            else:
                print(node.getLabel())
        """
    
        return newTree, swaps, swaps_array, distances
    
    
    def Swap_Subtrees3_2(T):
        #nswaps = random.randint(1, maxSwaps)
        distances = []
        swaps_array = []
        newTree = copy.deepcopy(T)
        subtrees = get_all_subtree(newTree)
        dictSubtrees = {}
        dictParents = {}
        # Per ogni radice di sottoalbero allego il rispettivo sottoalbero
        for st in subtrees[1:len(subtrees)]:
            if len(st) > 0:
                dictSubtrees[st[0][0]] = st
                parents = []
                parent = st[0][0]
                while parent.getParent() is not None:
                    parents.append(parent.getParent())
                    parent = parent.getParent()
                dictParents[st[0][0]] = parents
        exchangeable = []
        keys = list(dictSubtrees.keys())
        i = 1
        # Ricavo i sottoalberi scambiabili
        for key in keys:
            if i <= len(keys)-2:
                for k in keys[i:]:
                    if key not in dictParents[k]:
                        exchangeable.append([key, k])
            i += 1
        dictExchanged = {}
        for k in keys:
            dictExchanged[k] = False
        random.shuffle(exchangeable)
        swaps = 0
        # i = 0
        nswaps = 50  # Numero di scambi da effettuare
        while swaps < nswaps:
            pair = random.choice(exchangeable)
            # if not dictExchanged[pair[0]] and not dictExchanged[pair[1]]:
            if pair[1] not in dictParents[pair[0]] and pair[0] not in dictParents[pair[1]]:
                print("Sto scambiando: ", pair[0].getLabel(), pair[1].getLabel())
                swaps += 1
                dictExchanged[pair[0]] = True
                dictExchanged[pair[1]] = True
                p1 = pair[0].getParent()
                tmp = p1.getChildren()
                for j in range(0, len(tmp)):
                    if tmp[j] == pair[0]:
                        tmp[j] = pair[1]
                        break
                p1.setChildren(tmp)
                p2 = pair[1].getParent()
                tmp = p2.getChildren()
                if p2 == pair[0].getParent():
                    # Nel caso in cui sto scambiando fratelli
                    flag = False
                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[1]:
                            if flag:
                                tmp[j] = pair[0]
                                break
                            flag = True
                else:
                    for j in range(0, len(tmp)):
                        if tmp[j] == pair[1]:
                            tmp[j] = pair[0]
                            break
                p2.setChildren(tmp)
                pair[0].setParent(p2)
                pair[1].setParent(p1)
    
                # Aggiorno gli antenati per ogni nodo
                for node in newTree.getNodes():
                    parents = []
                    parent = node
                    while parent.getParent() is not None:
                        parents.append(parent.getParent())
                        parent = parent.getParent()
                    dictParents[node] = parents
    
                newEdges = []
                preorder = newTree.preorder(newTree.getRoot())
                for n in preorder:
                    for c in n.getChildren():
                        newEdges.append((n, c))
                newTree.setEdges(newEdges)
    
                subtrees = get_all_subtree(newTree)
                dictSubtrees = {}
                dictParents = {}
                # Per ogni radice di sottoalbero allego il rispettivo sottoalbero
                for st in subtrees[1:len(subtrees)]:
                    if len(st) > 0:
                        dictSubtrees[st[0][0]] = st
                        parents = []
                        parent = st[0][0]
                        while parent.getParent() is not None:
                            parents.append(parent.getParent())
                            parent = parent.getParent()
                        dictParents[st[0][0]] = parents
                exchangeable = []
                keys = list(dictSubtrees.keys())
                j = 1
                # Ricavo i sottoalberi scambiabili
                for key in keys:
                    if j <= len(keys)-2:
                        for k in keys[j:]:
                            if key not in dictParents[k]:
                                exchangeable.append([key, k])
                    j += 1
    
                random.shuffle(exchangeable)
    
                swaps_array.append(swaps)
                distances.append(Xbwt_Edit_Distance(T, newTree))
    
            # i+=1
        #print("Numero di scambi effettuati: ", swaps)
        # print(newTree.preorder(newTree.getRoot()))
    
    
    
        """
        for node in preorder:
            if node.getParent()is not None:
                print(node.getLabel(), node.getParent().getLabel())
            else:
                print(node.getLabel())
        """
    
        return newTree, swaps, swaps_array, distances


# Commentato da Pierfrancesco
'''numero_esperimenti = 50'''

"""
path = os.path.join(os.getcwd(),"Esperimenti etichette multiple","Rimozioni")
print(path)

# Rimozioni
print("ESPERIMENTI - RIMOZIONI SOTTOALBERI")
for e in tqdm(range(1, numero_esperimenti+1)):
    num_labels = random.randint(2, 9)
    new_alphabet = copy.deepcopy(alphabet)
    random.shuffle(new_alphabet)
    for i in range(num_labels):
        for j in range(2):
            new_alphabet.append(new_alphabet[i])
    random.shuffle(new_alphabet)
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(new_alphabet)
    tree2, removals, size_sub_rem, removals_array, distances = Remove_Subtrees2(tree, 10, newpath)
    t1 = Export_Tree3(tree, newpath)
    t2 = Export_Tree3(tree2, newpath, "2")
    Draw_Tree2(t1, e, "", newpath)
    Draw_Tree2(t2, e, "2", newpath)
    asp_distance = sum(size_sub_rem)
    real_distance = Xbwt_Edit_Distance2(tree, tree2)
    f= open(os.path.join(newpath,"EXP_REM_"+str(e)+"_DETAILS.txt"),"w+")
    f2 = open(os.path.join(newpath,"EXP_REM_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - RIMOZIONI SOTTOALBERI CON ETICHETTE MULTIPLE *****\n\n")
    f.write("Dimensione albero 1: "+str(len(tree.getNodes()))+"\n")
    f.write("Numero sottoalberi rimossi: "+str(removals)+"\n")
    f.write("Dimensioni sottoalberi rimossi: "+str(size_sub_rem)+"\n")
    f.write("Dimensione albero 2: "+str(len(tree2.getNodes()))+"\n")
    f.write("Misura aspettata: "+str(asp_distance)+"\n")
    #f.write("-------------------------------------------------------")
    f.write("d_xbw: "+str(real_distance))
    f.close()
    f2.write("***** ESPERIMENTO "+str(e)+" - RIMOZIONE SOTTOALBERI - DATI DA PLOTTARE *****\n\n")
    f2.write("Numero di rimozioni | Valore misura\n")
    for i in range(len(distances)):
        f2.write(str(removals_array[i])+" "+str(distances[i])+"\n")
    f2.close()
    Plot_Exp(removals_array, distances, "ESPERIMENTO "+str(e)+" - RIMOZIONE SOTTOALBERI", "NUMERO RIMOZIONI", "VALORE MISURA", e, "EXP_REM", newpath)
"""

"""
path = os.path.join(os.getcwd(),"Esperimenti etichette multiple","Scambi sottoalberi")
print(path)

# Scambi di sottoalberi
print("ESPERIMENTI - SCAMBI SOTTOALBERI 1")
dictAverage={}
dictSwaps= {}
for e in tqdm(range(1, numero_esperimenti+1)):
    num_labels = 4
    new_alphabet = copy.deepcopy(alphabet)
    random.shuffle(new_alphabet)
    for i in range(num_labels):
        for j in range(2): # 3 ripetizioni
            new_alphabet.append(new_alphabet[i])
    random.shuffle(new_alphabet)
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(new_alphabet)           
    tree2, swaps, swaps_array, distances = Swap_Subtrees2(tree)
    t1 = Export_Tree3(tree, newpath)
    t2 = Export_Tree3(tree2, newpath, "2")
    Draw_Tree2(t1, e, "", newpath)
    Draw_Tree2(t2, e, "2", newpath)
    asp_distance = 2*swaps
    real_distance = Xbwt_Edit_Distance(tree, tree2)
    dimTree2 = 0
    for n in tree2.getNodes():
        if n.getLabel() != "$":
            dimTree2+=1
    f= open(os.path.join(newpath,"EXP_SST_"+str(e)+"_DETAILS.txt"),"w+")
    f2 = open(os.path.join(newpath,"EXP_SST_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1*****\n\n")
    f.write("Numero sottoalberi scambiati: "+str(swaps)+"\n")
    f.write("Misura aspettata: "+str(asp_distance)+"\n")
    f.write("d_xbw: "+str(real_distance))
    f2.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1 - DATI DA PLOTTARE *****\n\n")
    f2.write("Numero di scambi | Valore misura\n")
    for s in range(1, swaps+1):
        if s in dictSwaps.keys():
            dictSwaps[s]+=1
        else:
            dictSwaps[s]=1
    for i in range(len(distances)):
        if i+1 in dictAverage.keys():
            dictAverage[i+1] = dictAverage[i+1] + distances[i]
        else:
            dictAverage[i+1] = distances[i]
        f2.write(str(swaps_array[i])+" "+str(distances[i])+"\n")
    Plot_Exp(swaps_array, distances, "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT", newpath)
    f.close()
    f2.close()
f3 = open(os.path.join(path,"EXP_SST_DATA_AVG_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
f3.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1 - DATI DA PLOTTARE (AVG) *****\n\n")
f3.write("Numero di esperimenti: "+str(numero_esperimenti)+"\n\n")
f3.write("Numero di scambi | Somma misure | Media | Media 2\n")
final_swaps_array = []
avg_distances = []
avg_distances2 = []
for k in dictAverage.keys():
    final_swaps_array.append(k)
    avg_distances.append(dictAverage[k]/numero_esperimenti)
    avg_distances2.append(dictAverage[k]/dictSwaps[k])
    f3.write(str(k)+" "+str(dictAverage[k])+" "+str(dictAverage[k]/numero_esperimenti)+" "+str(dictAverage[k]/dictSwaps[k])+"\n")
Plot_Exp(final_swaps_array, avg_distances, "ESPERIMENTO - SCAMBI SOTTOALBERI 1 (AVG)", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG", path)
Plot_Exp(final_swaps_array, avg_distances2, "ESPERIMENTO - SCAMBI SOTTOALBERI 1 (AVG) - 2", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG_2", path)
f3.close()
"""

"""
path = os.path.join(os.getcwd(),"Esperimenti etichette multiple","Scambi sottoalberi 2")
print(path)

# Scambi di sottoalberi
print("ESPERIMENTI - SCAMBI SOTTOALBERI 2")
dictAverage={}
dictSwaps= {}
for e in tqdm(range(1, numero_esperimenti+1)):
    num_labels = 4
    new_alphabet = copy.deepcopy(alphabet)
    random.shuffle(new_alphabet)
    for i in range(num_labels):
        for j in range(2):
            new_alphabet.append(new_alphabet[i])
    random.shuffle(new_alphabet)
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(new_alphabet)           
    tree2, swaps, swaps_array, distances = Swap_Subtrees2_2(tree)
    t1 = Export_Tree3(tree, newpath)
    t2 = Export_Tree3(tree2, newpath, "2")
    Draw_Tree2(t1, e, "", newpath)
    Draw_Tree2(t2, e, "2", newpath)
    asp_distance = 2*swaps
    real_distance = Xbwt_Edit_Distance(tree, tree2)
    dimTree2 = 0
    for n in tree2.getNodes():
        if n.getLabel() != "$":
            dimTree2+=1
    f= open(os.path.join(newpath,"EXP_SST_"+str(e)+"_DETAILS.txt"),"w+")
    f2 = open(os.path.join(newpath,"EXP_SST_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 2*****\n\n")
    f.write("Numero sottoalberi scambiati: "+str(swaps)+"\n")
    f.write("Misura aspettata: "+str(asp_distance)+"\n")
    f.write("d_xbw: "+str(real_distance))
    f2.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 2 - DATI DA PLOTTARE *****\n\n")
    f2.write("Numero di scambi | Valore misura\n")
    for s in range(1, swaps+1):
        if s in dictSwaps.keys():
            dictSwaps[s]+=1
        else:
            dictSwaps[s]=1
    for i in range(len(distances)):
        if i+1 in dictAverage.keys():
            dictAverage[i+1] = dictAverage[i+1] + distances[i]
        else:
            dictAverage[i+1] = distances[i]
        f2.write(str(swaps_array[i])+" "+str(distances[i])+"\n")
    Plot_Exp(swaps_array, distances, "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 2", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT", newpath)
    f.close()
    f2.close()
f3 = open(os.path.join(path,"EXP_SST_DATA_AVG_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
f3.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 2 - DATI DA PLOTTARE (AVG) *****\n\n")
f3.write("Numero di esperimenti: "+str(numero_esperimenti)+"\n\n")
f3.write("Numero di scambi | Somma misure | Media | Media 2\n")
final_swaps_array = []
avg_distances = []
avg_distances2 = []
for k in dictAverage.keys():
    final_swaps_array.append(k)
    avg_distances.append(dictAverage[k]/numero_esperimenti)
    avg_distances2.append(dictAverage[k]/dictSwaps[k])
    f3.write(str(k)+" "+str(dictAverage[k])+" "+str(dictAverage[k]/numero_esperimenti)+" "+str(dictAverage[k]/dictSwaps[k])+"\n")
Plot_Exp(final_swaps_array, avg_distances, "ESPERIMENTO - SCAMBI SOTTOALBERI 2 (AVG)", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG", path)
Plot_Exp(final_swaps_array, avg_distances2, "ESPERIMENTO - SCAMBI SOTTOALBERI 2 (AVG) - 2", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG_2", path)
f3.close()
"""

"""
path = os.path.join(os.getcwd(),"Esperimenti etichette multiple","Scambi sottoalberi 3")
print(path)

# Scambi di sottoalberi
print("ESPERIMENTI - SCAMBI SOTTOALBERI 3")
dictAverage={}
dictSwaps= {}
for e in tqdm(range(1, numero_esperimenti+1)):
    num_labels = random.randint(2, 9)
    new_alphabet = copy.deepcopy(alphabet)
    random.shuffle(new_alphabet)
    for i in range(num_labels):
        for j in range(2):
            new_alphabet.append(new_alphabet[i])
    random.shuffle(new_alphabet)
    newpath = os.path.join(path,str(e))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    tree = Generate_Random_Tree(new_alphabet)           
    tree2, swaps, swaps_array, distances = Swap_Subtrees2_2(tree)
    t1 = Export_Tree3(tree, newpath)
    t2 = Export_Tree3(tree2, newpath, "2")
    Draw_Tree2(t1, e, "", newpath)
    Draw_Tree2(t2, e, "2", newpath)
    asp_distance = 2*swaps
    real_distance = Xbwt_Edit_Distance(tree, tree2)
    dimTree2 = 0
    for n in tree2.getNodes():
        if n.getLabel() != "$":
            dimTree2+=1
    f= open(os.path.join(newpath,"EXP_SST_"+str(e)+"_DETAILS.txt"),"w+")
    f2 = open(os.path.join(newpath,"EXP_SST_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
    f.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 3*****\n\n")
    f.write("Numero sottoalberi scambiati: "+str(swaps)+"\n")
    f.write("Misura aspettata: "+str(asp_distance)+"\n")
    f.write("d_xbw: "+str(real_distance))
    f2.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 3 - DATI DA PLOTTARE *****\n\n")
    f2.write("Numero di scambi | Distanza\n")
    for s in range(1, swaps+1):
        if s in dictSwaps.keys():
            dictSwaps[s]+=1
        else:
            dictSwaps[s]=1
    for i in range(len(distances)):
        if i+1 in dictAverage.keys():
            dictAverage[i+1] = dictAverage[i+1] + distances[i]
        else:
            dictAverage[i+1] = distances[i]
        f2.write(str(swaps_array[i])+" "+str(distances[i])+"\n")
    Plot_Exp(swaps_array, distances, "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 3", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT", newpath)
    f.close()
    f2.close()
f3 = open(os.path.join(path,"EXP_SST_DATA_AVG_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
f3.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 3 - DATI DA PLOTTARE (AVG) *****\n\n")
f3.write("Numero di esperimenti: "+str(numero_esperimenti)+"\n\n")
f3.write("Numero di scambi | Somma misure | Media | Media 2\n")
final_swaps_array = []
avg_distances = []
avg_distances2 = []
for k in dictAverage.keys():
    final_swaps_array.append(k)
    avg_distances.append(dictAverage[k]/numero_esperimenti)
    avg_distances2.append(dictAverage[k]/dictSwaps[k])
    f3.write(str(k)+" "+str(dictAverage[k])+" "+str(dictAverage[k]/numero_esperimenti)+" "+str(dictAverage[k]/dictSwaps[k])+"\n")
Plot_Exp(final_swaps_array, avg_distances, "ESPERIMENTO - SCAMBI SOTTOALBERI 3 (AVG)", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG", path)
Plot_Exp(final_swaps_array, avg_distances2, "ESPERIMENTO - SCAMBI SOTTOALBERI 3 (AVG) - 2", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_AVG_2", path)
f3.close()
"""

"""
# Dichiarazione nodi

root = Node('B')
n1 = Node('A')
n2 = Node('A')
n3 = Node('C')
n4 = Node('E')
n5 = Node('F')
n6 = Node('G')
n7 = Node('H')
n8 = Node('I')
n9 = Node('$')
n10 = Node('$')
n11 = Node('$')
n12 = Node('$')
n13 = Node('$')


# Inserimento nodi dell'albero

tree = Tree()
tree.insert(root, None)
tree.insert(n1, root)
tree.insert(n2, root)
tree.insert(n3, root)
tree.insert(n4, n1)
tree.insert(n5, n1)
tree.insert(n6, n2)
tree.insert(n7, n3)
tree.insert(n8, n3)
tree.insert(n9, n4)
tree.insert(n10, n5)
tree.insert(n11, n6)
tree.insert(n12, n7)
tree.insert(n13, n8)

# Dichiarazione nodi

root = Node('A')
n1 = Node('B')
n2 = Node('D')
n3 = Node('D')
n4 = Node('C')
n5 = Node('C')
n6 = Node('F')
n7 = Node('E')
n8 = Node('S')
n9 = Node('$')
n10 = Node('$')
n11 = Node('$')
n12 = Node('$')
n13 = Node('$')
n14 = Node('$')


# Inserimento nodi dell'albero

tree2 = Tree()
tree2.insert(root, None)
tree2.insert(n1, root)
tree2.insert(n2, root)
tree2.insert(n3, n1)
tree2.insert(n4, n1)
tree2.insert(n5, n1)
tree2.insert(n6, n2)
tree2.insert(n7, n2)
tree2.insert(n8, n2)
tree2.insert(n9, n3)
tree2.insert(n10, n4)
tree2.insert(n11, n5)
tree2.insert(n12, n6)
tree2.insert(n13, n7)
tree2.insert(n14, n8)

d = Xbwt_Edit_Distance2(tree, tree2)
print(d)
"""