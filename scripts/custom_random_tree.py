import numpy as np
import copy
import os
import random
from XBWT import Node, Tree
import Dxbw
import mp3treesim
from ResearchLab import exportTreeToGraphviz

# Restituisce il sottoalbero associato al nodo
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

# Restituisce tutti i sottoalberi di un albero T
def get_all_subtree(T):
    nodes = T.getNodes()
    subtrees = []
    for node in nodes:
        subtree = get_subtree(node)
        subtrees.append(subtree)
    return subtrees

# Genera un albero random
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
    print("Path:", path)
    T0 = copy.deepcopy(T)
    # La ricostruzione funziona ma non funziona il preordine
    distances = []
    distances_mp3 = []
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
                if e[1].getLabel() != "$":
                    dimTmp+=1
                #dimTmp += 1
            # print("\n\n")
            sbt_dim_real.append(dimTmp+1)
        else:
            sbt_dim_real.append(0)
    # Non considerare il primo sottoalbero (cioè quello della radice)
    subtrees_dim = [len(e) for e in subtrees]

    # Ricavo la prima posizione delle foglie (0)
    firstEmptySubtreeIndex = list(subtrees_dim).index(0)
    print("Il first: ", firstEmptySubtreeIndex)
    # Ricavo la posizione del sottoalbero da rimuovere (escluso il sottoalbero della radice)
    posSubtree = 1
    # Faccio in modo che vi siano almeno due nodi dopo la rimozione selezionata
    while True:
        posSubtree = random.randint(1, firstEmptySubtreeIndex-1)
        if (subtrees_dim[0] - subtrees_dim[posSubtree] > 2):
            break
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
        # Calcolo le distanze
        print("Sono nel calcolo delle distanze")
        distances.append(Dxbw.dxbw(T0, newTree))
        exportTreeToGraphviz(T0, "tree1", path, True)
        exportTreeToGraphviz(newTree, "tree2_int_"+str(removals), path, True)
        p1 = os.path.join(path, "tree1.gv")
        p2 = os.path.join(path, "tree2_int_"+str(removals)+".gv")
        print("Path 2: ", p2)
        gv1 = mp3treesim.read_dotfile(p1)
        gv2 = mp3treesim.read_dotfile(p2)
        distances_mp3.append(mp3treesim.similarity(gv1, gv2))
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
                            dimTmp += 1
                    # print("\n")
                    sbt_dim_real.append(dimTmp+1) # Aggiungo la radice
                else:
                    sbt_dim_real.append(0)
            # Non considerare il primo sottoalbero (cioè quello della radice)
            subtrees_dim = [len(e) for e in subtrees]
            #print("Dimensioni sottoalberi: ", subtrees_dim)
            # print(subtrees_dim)
            if len(subtrees_dim) != 2:
                firstEmptySubtreeIndex = list(subtrees_dim).index(0)
                # print(firstEmptySubtreeIndex)
                while True:
                    posSubtree = random.randint(1, firstEmptySubtreeIndex-1)
                    print("Sono entrato nel loop")
                    print("Dimensioni reali dei sottoalberi: ", sbt_dim_real)
                    if (sbt_dim_real[0] - sbt_dim_real[posSubtree]) > 2:
                        break
                    if (sbt_dim_real[0] - min(sbt_dim_real[1:firstEmptySubtreeIndex])) < 3:
                        r = nremovals
                        break
                dec = 0
                print("FirstIndex0: ", firstEmptySubtreeIndex)
                print("Subtrees con stampa: ", subtrees)
                for sub in subtrees:
                    for s in sub:
                        print(s[0].getLabel(), s[1].getLabel())
                print("\n Possubtree: ", posSubtree)
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
    return newTree, removals, size_sub_rem, removals_array, distances, distances_mp3

def Swap_Subtrees(T, path):
    #nswaps = random.randint(1, maxSwaps)
    distances = []
    distances_mp3 = []
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
                distances.append(Dxbw.dxbw(T, newTree))

                exportTreeToGraphviz(T, "tree1", path, True)
                exportTreeToGraphviz(newTree, "tree2_int_"+str(swaps), path, True)
                p1 = os.path.join(path,"tree1.gv")
                p2 = os.path.join(path,"tree2_int_"+str(swaps)+".gv")
                gv1 = mp3treesim.read_dotfile(p1)
                gv2 = mp3treesim.read_dotfile(p2)
                distances_mp3.append(mp3treesim.similarity(gv1, gv2))

        i += 1
    #print("Numero di scambi effettuati: ", swaps)
    # print(newTree.preorder(newTree.getRoot()))

    """
    for node in preorder:
        if node.getParent()!= None:
            print(node.getLabel(), node.getParent().getLabel())
        else:
            print(node.getLabel())
    """

    return newTree, swaps, swaps_array, distances, distances_mp3

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
            distances.append(Dxbw.dxbw(T, newTree))

        i += 1
    #print("Numero di scambi effettuati: ", swaps)
    # print(newTree.preorder(newTree.getRoot()))

    """
    for node in preorder:
        if node.getParent()!= None:
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
            distances.append(Dxbw.dxbw(T, newTree))

        # i+=1
    #print("Numero di scambi effettuati: ", swaps)
    # print(newTree.preorder(newTree.getRoot()))

    """
    for node in preorder:
        if node.getParent()!= None:
            print(node.getLabel(), node.getParent().getLabel())
        else:
            print(node.getLabel())
    """

    return newTree, swaps, swaps_array, distances


