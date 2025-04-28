import os
from tqdm import tqdm
from custom_random_tree import Generate_Random_Tree, Remove_Subtrees, Swap_Subtrees
from ResearchLab import exportTreeToGraphviz
import Dxbw
import pylab as plt
import numpy as np

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
            "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

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
    if numexp == "":
        path =  savepath+"\\"+prelabel+"_PLOT.png"
    else:
        path = savepath+"\\"+prelabel+"_"+str(numexp)+"_PLOT.png"
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.close()
    #plt.show()

def Plot_Exp2(xdata, ydata, title, xlabel, ylabel, numexp, prelabel, savepath):

    # print(xdata)
    # print(ydata)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.vlines(xdata, 0, ydata, color="black", linestyle = "dashed", linewidth=0.5)
    #plt.hlines(ydata, 0, xdata, color="black", linestyle = "dashed", linewidth=0.5)

    #plt.yticks(np.arange(min(ydata), max(ydata)+2, 2))

    plt.scatter(xdata, ydata, color='r', zorder=2)
    plt.plot(xdata, ydata, color='c', zorder=1)
    path = savepath+"\\"+prelabel
    if str(numexp) != "":
        path=path+"_"+str(numexp)
    path = path+"_PLOT.png"
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.close()
    #plt.show()

if __name__ == "__main__":
    numero_esperimenti = 2

    # path = os.path.dirname(os.path.abspath(__file__))+"\Esperimenti\Rimozioni"
    # print("Path corrente: ", path)

    # # Rimozioni
    # print("ESPERIMENTI - RIMOZIONI SOTTOALBERI")
    # for e in tqdm(range(1, numero_esperimenti+1)):
    #     newpath = path+"\\"+str(e)
    #     if not os.path.exists(newpath):
    #         os.makedirs(newpath)
    #     tree = Generate_Random_Tree(alphabet)
    #     tree2, removals, size_sub_rem, removals_array, distances, distances_mp3 = Remove_Subtrees(tree, 10, newpath)
    #     print("Nuovo path:", newpath, type(newpath))
    #     #exportTreeToGraphviz(tree, "tree1", newpath, True)
    #     exportTreeToGraphviz(tree2, "tree2", newpath, True)
    #     asp_distance = sum(size_sub_rem)
    #     real_distance = Dxbw.dxbw(tree, tree2)
    #     f= open(newpath+"\EXP_REM_"+str(e)+"_DETAILS.txt","w+")
    #     f2 = open(newpath+"\EXP_REM_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt","w+")
    #     f3 = open(newpath+"\EXP_REM_DATA_TO_PLOT_MP3_"+str(e)+"_DETAILS.txt","w+")
    #     f.write("***** ESPERIMENTO "+str(e)+" - RIMOZIONI SOTTOALBERI *****\n\n")
    #     f.write("Dimensione albero 1: "+str(len(tree.getNodes()))+"\n")
    #     f.write("Numero sottoalberi rimossi: "+str(removals)+"\n")
    #     f.write("Dimensioni sottoalberi rimossi: "+str(size_sub_rem)+"\n")
    #     f.write("Dimensione albero 2: "+str(len(tree2.getNodes()))+"\n")
    #     f.write("Misura dxbw aspettata: "+str(asp_distance)+"\n")
    #     #f.write("-------------------------------------------------------")
    #     f.write("d_xbw: "+str(real_distance))
    #     f.close()
    #     f2.write("***** ESPERIMENTO "+str(e)+" - RIMOZIONE SOTTOALBERI - DATI DA PLOTTARE *****\n\n")
    #     f2.write("Numero di rimozioni | Valore misura\n")
    #     for i in range(len(distances)):
    #         f2.write(str(removals_array[i])+" "+str(distances[i])+"\n")
    #     f2.close()
    #     f3.write("***** ESPERIMENTO "+str(e)+" - RIMOZIONE SOTTOALBERI - DATI DA PLOTTARE - MP3TREESIM *****\n\n")
    #     f3.write("Numero di rimozioni | Valore misura\n")
    #     for i in range(len(distances_mp3)):
    #         f3.write(str(removals_array[i])+" "+str(distances_mp3[i])+"\n")
    #     f3.close()
    #     Plot_Exp(removals_array, distances, "ESPERIMENTO "+str(e)+" - RIMOZIONE SOTTOALBERI", "NUMERO RIMOZIONI", "VALORE MISURA", e, "EXP_REM", newpath)
    #     Plot_Exp2(removals_array, distances_mp3, "ESPERIMENTO "+str(e)+" - RIMOZIONE SOTTOALBERI - MP3TREESIM", "NUMERO RIMOZIONI", "VALORE MISURA", e, "EXP_REM_MP3", newpath)
    #     Plot_Exp2(removals_array, [1-v for v in distances_mp3], "ESPERIMENTO "+str(e)+" - RIMOZIONE SOTTOALBERI - MP3TREESIM (REVERSE)", "NUMERO RIMOZIONI", "VALORE MISURA", e, "EXP_REM_REV_MP3", newpath)

    path = os.path.dirname(os.path.abspath(__file__))+"\Esperimenti\Scambi sottoalberi"
    print(path)

    # Scambi di sottoalberi
    print("ESPERIMENTI - SCAMBI SOTTOALBERI 1")
    dictAverage={}
    dictAverageMP3={}
    dictSwaps= {}
    for e in tqdm(range(1, numero_esperimenti+1)):
        newpath = path+"\\"+str(e)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        tree = Generate_Random_Tree(alphabet)           
        tree2, swaps, swaps_array, distances, distances_mp3 = Swap_Subtrees(tree, newpath)
        exportTreeToGraphviz(tree2, "tree2", newpath, True)
        asp_distance = 2*swaps
        real_distance = Dxbw.dxbw(tree, tree2)
        dimTree2 = 0
        for n in tree2.getNodes():
            if n.getLabel() != "$":
                dimTree2+=1
        f= open(newpath+"\EXP_SST_"+str(e)+"_DETAILS.txt","w+")
        f2 = open(newpath+"\EXP_SST_DATA_TO_PLOT_"+str(e)+"_DETAILS.txt","w+")
        f4 = open(newpath+"\EXP_SST_MP3_DATA_TO_PLOT_MP3_"+str(e)+"_DETAILS.txt","w+")
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
                dictAverageMP3[i+1] = dictAverageMP3[i+1] + distances_mp3[i]
            else:
                dictAverage[i+1] = distances[i]
                dictAverageMP3[i+1] = distances_mp3[i]
            f2.write(str(swaps_array[i])+" "+str(distances[i])+"\n")
        Plot_Exp(swaps_array, distances, "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT", newpath)
        Plot_Exp2(swaps_array, distances_mp3, "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1 - MP3TREESIM", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_MP3", newpath)
        Plot_Exp2(swaps_array, [1-v for v in distances_mp3], "ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1 - MP3TREESIM (REV)", "NUMERO SCAMBI", "VALORE MISURA", e, "EXP_SBT_MP3_REV", newpath)
        f.close()
        f2.close()
        f4.write("***** ESPERIMENTO "+str(e)+" - RIMOZIONE SOTTOALBERI - DATI DA PLOTTARE - MP3TREESIM *****\n\n")
        f4.write("Numero di rimozioni | Valore misura\n")
        for i in range(len(distances_mp3)):
            f4.write(str(swaps_array[i])+" "+str(distances_mp3[i])+"\n")
        f4.close()
    f3 = open(path+"\EXP_SST_DATA_AVG_TO_PLOT_"+str(e)+"_DETAILS.txt","w+")
    f5 = open(path+"\EXP_SST_DATA_MP3_AVG_TO_PLOT_"+str(e)+"_DETAILS.txt","w+")
    f3.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1 - DATI DA PLOTTARE (AVG) *****\n\n")
    f3.write("Numero di esperimenti: "+str(numero_esperimenti)+"\n\n")
    f3.write("Numero di scambi | Somma misure | Media | Media 2\n")
    f5.write("***** ESPERIMENTO "+str(e)+" - SCAMBI SOTTOALBERI 1 - MP3TREESIM - DATI DA PLOTTARE (AVG) *****\n\n")
    f5.write("Numero di esperimenti: "+str(numero_esperimenti)+"\n\n")
    f5.write("Numero di scambi | Somma misure | Media | Media 2\n")
    final_swaps_array = []
    avg_distances = []
    avg_distances2 = []
    avg_distances2_mp3 = []
    for k in dictAverage.keys():
        final_swaps_array.append(k)
        avg_distances.append(dictAverage[k]/numero_esperimenti)
        avg_distances2.append(dictAverage[k]/dictSwaps[k])
        avg_distances2_mp3.append(dictAverageMP3[k]/dictSwaps[k])
        f3.write(str(k)+" "+str(dictAverage[k])+" "+str(dictAverage[k]/numero_esperimenti)+" "+str(dictAverage[k]/dictSwaps[k])+"\n")
        f5.write(str(k)+" "+str(dictAverageMP3[k])+" "+str(dictAverageMP3[k]/numero_esperimenti)+" "+str(dictAverageMP3[k]/dictSwaps[k])+"\n")
    Plot_Exp(final_swaps_array, avg_distances, "ESPERIMENTO - SCAMBI SOTTOALBERI 1 (AVG)", "NUMERO SCAMBI", "VALORE MISURA", "", "EXP_SBT_AVG", path)
    Plot_Exp(final_swaps_array, avg_distances2, "ESPERIMENTO - SCAMBI SOTTOALBERI 1 (AVG) - 2", "NUMERO SCAMBI", "VALORE MISURA", "", "EXP_SBT_AVG_2", path)
    Plot_Exp2(final_swaps_array, avg_distances2_mp3, "ESPERIMENTO - SCAMBI SOTTOALBERI 1 - MP3TREESIM - (AVG) - 2", "NUMERO SCAMBI", "VALORE MISURA", "", "EXP_SBT_MP3_AVG_2", path)
    f3.close()
    f5.close()