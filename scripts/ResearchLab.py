# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:10:24 2022

@author: dolce
"""

from fileinput import filename
import os
import subprocess
from tokenize import String
import networkx as nx
from networkx.drawing import nx_agraph
import mp3treesim as mp3
import XBWT
from XBWT import Node, Tree
import graphviz
from graphviz import Digraph
import random
import Dxbw

# Esporta un Tree nel formato Graphviz
def exportTreeToGraphviz(tree, out_prefix, folder_path, remove_dollars):
    # print("Tipi:", type(out_prefix), type(folder_path))
    nodes = []
    if remove_dollars:
        for n in tree.preorder(tree.getRoot()):
            if n.getLabel() != "$":
                nodes.append(n)
    else:
        nodes = tree.preorder(tree.getRoot())
    tot_nodes = len(tree.getNodes()) # Numero totale di nodi
    final_path = (folder_path+"\\"+out_prefix).replace("\\", "/")
    g = Digraph('G') # Creo un grafo diretto

    # Etichetta della radice
    label = tree.getRoot().getLabel()
    # Aggiunfo la radice al grafo 0 
    g.node(str(0), label)

    dict_labels = {}
    i = 0
    for node in nodes:
        dict_labels[node] = i
        i+=1
    
    for node in nodes:
        for child in node.getChildren():
            if remove_dollars:
                if child.getLabel() == "$":
                    break
            g.node(str(dict_labels[child]), child.getLabel())
            g.edge(str(dict_labels[node]), str(str(dict_labels[child])))

    #g.save()
    g.render(filename=final_path+".gv", format="png")

# Crea un oggetto di tipo Tree da un file di tipo Graphviz
def importTreeFromGraphviz(fileName):
    G = nx.Graph(nx_agraph.read_dot(fileName))
    nodes = G.nodes(data=True)
    node_label = {}
    for node in nodes:
        node_label[node[0]] = Node(node[1]['label'])
    edges = list(G.edges)
    tree = Tree()
    tree.insert(node_label[edges[0][0]], None)
    for edge in edges:
        tree.insert(node_label[edge[1]], node_label[edge[0]])
    return tree

def generateTree():
    num_gen_trees = int(input("Quanti alberi vuoi generare: "))
    num_nodes = int(input("Inserisci il numero totale di nodi (default: 10): "))
    num_label = int(input("Inserisci il numero totale di etichette (default: 10): "))
    max_num_child = input("Inserisci il numero massimo di figli per nodo (default: 3): ")
    complete_tree = False
    while(True):
        complete_tree_choice = input("Vuoi generare un albero completo (Y/N): ")
        complete_tree_choice = complete_tree_choice.upper()
        if complete_tree_choice == "Y" or complete_tree_choice == "N":
            if complete_tree_choice == "Y":
                complete_tree = True
            break
        else:
            print("\nDigita Y (si) o N (no)")
    prefix_file_outpt = input("Inserisci il prefisso del file di output: ")
    
    command = "python generate_tree.py -n "+str(num_nodes)+" -l "+str(num_label)+" -s "+str(max_num_child)
    
    if complete_tree:
        command+=" --full"

    for i in range(num_gen_trees):
        final_prefix_file_outpt=prefix_file_outpt+"_"+str(i)
        final_command = command+" -o "+final_prefix_file_outpt
        subprocess.run(final_command, shell=True)
    
    print("\nGENERAZIONE COMPLETATA !")

def pertubateTree():
    pertubation_dict = {}
    pertubation_dict[1] = "--labelswap"
    pertubation_dict[2] = "--noderemove"
    pertubation_dict[3] = "--labelremove"
    pertubation_dict[4] = "--labelduplication"
    pertubation_dict[5] = "--nodeswap" 
    
    path = input("Inserisci il path dell'albero: ")
    
    scelta = 0
    while(True):
       print("\nDigita una tra le seguenti opzioni per scegliere la perturbazione da applicare all'albero': ")
       print("1 - Scambio etichette")
       print("2 - Rimozione nodi")
       print("3 - Rimozione etichette")
       print("4 - Duplicazione etichette")
       print("5 - Scambio nodi")
       print("6 - Annulla operazione")
       scelta = int(input("Inserisci scelta: "))
       if scelta > 0 and scelta < 7:
           break
    if scelta == 6:
        exit()
        
    prob = int(input("Inserisci numero operazioni: "))
    
    filename = os.path.basename(path)
    path_wo_filename = path.replace(filename, "")
    path_splitted = os.path.splitext(path)
    final_path = path_splitted[0]+"_p"+path_splitted[1]
    
    command = "python perturbation.py -t "+path+ " "+pertubation_dict[scelta]+" "+str(prob)+" --out "+"tree_p2.gv" 
    print(command)
    subprocess.run(command, shell=True)
    
    print("\nPERTURBAZIONE APPLICATA !")
    
def mp3TreeSimComp(treePath1, treePath2):
    f1= open("mp3treesim.txt","w+")
    gv1 = mp3.read_dotfile(treePath1)
    gv2 = mp3.read_dotfile(treePath2)
    f1.write(str(mp3.similarity(gv1, gv2)))
    f1.close()
    print("\nConfronto completato, risultati disponibili in mp3treesim.txt!")

def dxbwComp(treePath1, treePath2):
    f1= open("dxbw.txt","w+")
    tree1 = importTreeFromGraphviz(treePath1)
    tree1.addDollarsToLeafs() # Aggiungo i dollari alle foglie
    tree2 = importTreeFromGraphviz(treePath2)
    tree2.addDollarsToLeafs()
    f1.write(str(Dxbw.dxbw(tree1, tree2)))
    f1.close()
    print("\nConfronto completato, risultati disponibili in dxbw.txt!")

if __name__ == "__main__":
    while(True):
        print("Digita: ")
        print("1 - Generazione alberi random fully-labeled")
        print("2 - Perturba albero")
        print("3 - Confronta alberi con mp3treesim")
        print("4 - Confronta alberi con dxbw")
        print("5 - Esci")
        scelta = int(input("Inserisci scelta: "))
        if scelta > 0 and scelta < 6:
            break
    if scelta == 1:
        generateTree()
    if scelta == 2:
        pertubateTree()
    if scelta == 3:
        treePath1 = input("Inserisci path tree 1: ")
        treePath2 = input("Inserisci path tree 2: ")
        mp3TreeSimComp(treePath1, treePath2)
    if scelta == 4:
        treePath1 = input("Inserisci path tree 1: ")
        treePath2 = input("Inserisci path tree 2: ")
        dxbwComp(treePath1, treePath2)
        
    # Dichiarazione nodi 

    # root=Node('F')
    # n1=Node('Y')
    # n2=Node('D')
    # n3=Node('J')
    # n4=Node('I')
    # n5=Node('B')
    # n6=Node('C')
    # n7=Node('$')
    # n8=Node('S')
    # n9=Node('$')
    # n10=Node('Q')
    # n11=Node('O')
    # n12=Node('$')
    # n13=Node('E')
    # n14=Node('$')
    # n15=Node('Z')
    # n16=Node('P')
    # n17=Node('$')
    # n18=Node('G')
    # n19=Node('$')
    # n20=Node('N')
    # n21=Node('K')
    # n22=Node('$')
    # n23=Node('W')
    # n24=Node('$')
    # n25=Node('L')
    # n26=Node('$')
    # n27=Node('T')
    # n28=Node('U')
    # n29=Node('V')
    # n30=Node('$')
    # n31=Node('M')
    # n32=Node('$')
    # n33=Node('H')
    # n34=Node('X')
    # n35=Node('$')
    # n36=Node('A')
    # n37=Node('R')
    # n38=Node('$')


    # # Inserimento nodi dell'albero

    # tree=Tree()
    # tree.insert(root, None)
    # tree.insert(n1, root)
    # tree.insert(n2, n1)
    # tree.insert(n3, n2)
    # tree.insert(n27, n2)
    # tree.insert(n4, n3)
    # tree.insert(n15, n3)
    # tree.insert(n20, n3)
    # tree.insert(n28, n27)
    # tree.insert(n33, n27)
    # tree.insert(n36, n27)
    # tree.insert(n5, n4)
    # tree.insert(n10, n4)
    # tree.insert(n13, n4)
    # tree.insert(n16, n15)
    # tree.insert(n18, n15)
    # tree.insert(n21, n20)
    # tree.insert(n23, n20)
    # tree.insert(n25, n20)
    # tree.insert(n29, n28)
    # tree.insert(n31, n28)
    # tree.insert(n34, n33)
    # tree.insert(n37, n36)
    # tree.insert(n6, n5)
    # tree.insert(n8, n5)
    # tree.insert(n11, n10)
    # tree.insert(n7, n13)
    # tree.insert(n9, n16)
    # tree.insert(n12, n18)
    # tree.insert(n14, n21)
    # tree.insert(n17, n23)
    # tree.insert(n19, n25)
    # tree.insert(n22, n29)
    # tree.insert(n24, n31)
    # tree.insert(n26, n34)
    # tree.insert(n30, n37)
    # tree.insert(n32, n6)
    # tree.insert(n35, n8)
    # tree.insert(n38, n11)

    #exportTreeToGraphviz(tree, "tree", True)
    