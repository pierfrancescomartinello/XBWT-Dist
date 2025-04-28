from XBWT import XBWT
import numpy as np

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
                                   dp[i-1][j-1])      # Replace
    return dp[m][n]

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

def dxbw(tree1, tree2):
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

    # print("TREE, S_LAST, S_ALPHA, S_PI, LCP", end="\n\n")
    # j = 0
    # for i in m_lcp:
    #     print(j, i)
    #     j+=1

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
        dist = editDistDP(str1, str2, len(str1), len(str2))
        total = total+dist
        dists[str(t)] = dist

    for p in other_partitions:
        dists[str(p)] = p[1]+1-p[0]
        total += p[1]+1-p[0]

    #print(dists, end="\n\n")
    return total