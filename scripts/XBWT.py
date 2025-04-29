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
from networkx.drawing import nx_agraph
#import pygraphviz as pgv
import matplotlib
from matplotlib.pyplot import figure
from tqdm import tqdm
import os
import svgling
import nltk
import os
import graphviz

if os.name == 'nt':
    os.environ["PATH"] = os.join("C:\\Program Files\\Graphviz\\bin", os.getenv("PATH"))

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
            "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

class Node(object):
    """ Node of a Tree """

    def __init__(self, label='root', children=None, parent=None):
        self.label = label
        self.parent = parent
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

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
        if len(self.children) == 0:
            return True
        else:
            return False

    def level(self):
        """ Return the level of a node """
        if self.isRoot():
            return 0
        else:
            return 1 + self.parent.level()

    def isRightmost(self):
        """ 
        Return 1 if node is the rightmost children of the parent, 0
        otherwise
        """
        length_parent = len(self.parent.children)
        if length_parent != 0:
            if (self.parent.children[length_parent-1] == self):
                return 1
        return 0

    def addChild(self, node):
        """ Add a child at node """
        node.parent = self
        assert isinstance(node, Node)
        self.children.append(node)

    def getChildren(self):
        """ Return the children's array of a node"""
        return self.children

    def setChildren(self, children):
        self.children = children


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
        else:
            if self.root is None:
                self.root = node
        self.nodes.append(node)

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

    def addDollarsToLeafs(self):
        for node in self.nodes:
            if node.isLeaf():
                node.addChild(Node("$"))

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
            if len((Stack[len(Stack)-1]).children) == 0:
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
            for i in Par.children:
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
        #print("Pos_first: \n", Pos_first, end="\n\n")
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
        #print("Triplette ordinate:\n", sortedTriplets, end="\n\n")

        lexName, notUnique = self.nameTriplets(sortedTriplets)
        #print("Ranking:\n", lexName, end="\n\n")

        #print("\nLexName: ", lexName)

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
        #print("SA 2: ", SA)

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
    plt.show()





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
    Plot_Exp(swaps_array, distances, "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT", newpath)
    f.close()
    f2.close()
f3 = open(os.path.join(path+"EXP_SST_DATA_AVG_TO_PLOT_"+str(e)+"_DETAILS.txt"),"w+")
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
path = os.path.join(os.getcwd()+"Esperimenti","Scambi sottoalberi 3")
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
f3 = open(os.path.join(path,"EXP_SST_DATA_AVG_TO_PLOT_DETAILS.txt"),"w+")
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

"""
path = os.path.join(os.getcwd()+"Esperimenti","Scambi sottoalberi 4")
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
f3.close()
"""

"""
path = os.path.join(os.getcwd(),"Esperimenti","Scambi simboli")
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
path = os.path.join(os.getcwd(),"Esperimenti 2","Etichette multiple")

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
    f= open(os.path.open(newpath,"EXP_RTC_"+str(e)+"_DETAILS.txt"),"w+")
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


"""
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
                        if e[1].getLabel() != "$":
                            dimTmp+=1
                        
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
            while parent.getParent() != None:
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
                    while parent.getParent() != None:
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

    for node in preorder:
        if node.getParent()!= None:
            print(node.getLabel(), node.getParent().getLabel())
        else:
            print(node.getLabel())
    

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
            while parent.getParent() != None:
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
                while parent.getParent() != None:
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

    
    for node in preorder:
        if node.getParent()!= None:
            print(node.getLabel(), node.getParent().getLabel())
        else:
            print(node.getLabel())
    

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
            while parent.getParent() != None:
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
                while parent.getParent() != None:
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
                    while parent.getParent() != None:
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

    
    for node in preorder:
        if node.getParent()!= None:
            print(node.getLabel(), node.getParent().getLabel())
        else:
            print(node.getLabel())
    

    return newTree, swaps, swaps_array, distances


numero_esperimenti = 50


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

#import mp3treesim as mp3
"""
new_alphabet = copy.deepcopy(alphabet)
tree = Generate_Random_Tree(new_alphabet)           
tree2, swaps, swaps_array, distances = Swap_Subtrees(tree)
t1 = Export_Tree3(tree, "", "")
t2 = Export_Tree3(tree2, "", "2")
Draw_Tree2(t1, "", "", "")
Draw_Tree2(t2, "", "2", "")
f= open("EXP_SCAMBI4"+"_DETAILS.txt","w+")
f.write("Numero sottoalberi scambiati: "+str(swaps)+"\n")
f.write("Numero sottoalberi scambiati: "+str(swaps)+"\n")
f.close()
exportToGraphviz(tree, "tree1")
exportToGraphviz(tree2, "tree2")
"""
"""
f1= open("mp3treesim.txt","w+")

gv1 = mp3.read_dotfile('tree1.gv')
gv2 = mp3.read_dotfile('tree2.gv')
f1.write(str(mp3.similarity(gv1, gv2)))
f1.close()
"""