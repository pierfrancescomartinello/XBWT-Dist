from copy import deepcopy

def calculate_ds(SA:list[list[str]], Child:list[list[int]], F:dict[str, int], n_0:int, n_1:int, k:int) -> tuple[list[int], list[int]]:
    '''
    Calculates the necessary structures to compute the Jaccard distance
   
    Args:
        SA (tuple of lists of string): The arrays SA0 and SA1, representing the labels of the two XBWTs
        Child (tuple of lists of int): The arrays Child0 and Child1, representing the number of children each node in the two XBWTs has
        F (dictionary of str and int): The dictionary that gives the indexes on where in the fictitious merged XBWT the prefixes that start with a particular character must be placed
        n_0 (int): The size of SA0
        n_1 (int): The size of SA1
    
    Returns:
        Flag (list of int): The array showing the contribution of each one of the two XBWTs
        LCP  (list of int): The array representing the longest common prefix between the element at index i and the one at index i-1 (LCP[0] is 0 per convention as LCP('',\\epsilon) = 0)
    '''
    
    n:int = n_0+n_1     
    h:int = 0           # The current iteration
    
    Flagh:list[int] = [0,1] + [0]*(n_0-1) + [1]*(n_1-1)         # That's how Flag is defined
    LCP:list[int] = [0]+[-1]*(n-1) + [0]                        # That's how LCP is defined
    
    
    # TODO: use this to implement a k-invariant algorithm
    # working:bool = True     # This boolean variable checks if the algorithm has to end
    # prev_2:bool = False     # This boolean variable checks if the previous two iteration of the array Flag are equal
    
    for _ in range(k):
        h += 1
        id:int = 0
        kn:list[int] = [0,0]
        
        temp_F:dict[str, int] = deepcopy(F)
        # The first two indexes will contain the roots of the XBWTs
        Flag:list[int] = [0,1] + [-1]*(n-2)
        Block_Id:dict[str, int] = {i:-1 for i in F.keys()}
        
        for i in range(n):
            if LCP[i] not in (-1, h-1):
                id:int = i          # This means a new block is starting--
                
            b:int = Flagh[i]        
            c:str = SA[b][kn[b]]    # Fetching the correct character
            if c != "$":
                
                # The node is responsible to place its children in the correct position in the array Flag
                children:int = Child[b][kn[b]]
                Flag[temp_F[c]:temp_F[c]+children] = [b]*children
                
                # If the last time that the character has been seen is on a different block...
                if Block_Id[c] != id:
                    Block_Id[c] = id                # ...update Block_Id...
                    if LCP[temp_F[c]] == -1:        
                        LCP[temp_F[c]] = h-1        # Update LCP for only the leftmost child of the node

                temp_F[c] += children
                
            kn[b] += 1
        
        
        # TODO: implement a k-invariant algorithm
        #if Flagh == Flag:       # When Flag is finally stable...
        #    if not prev_2:      # ...we wait for a third iteration
        #        prev_2 = True
        #    else:
        #        working = False # ...and that's when the algorithm stops
        #else:
            #    # Preparing for the next iteration
            #    Flagh = Flag
        Flagh = Flag

    
    return Flagh, [i if i != -1 else k for i in LCP]
   
def calculate_partition_contributions(Flag:list[int], LCP:list[int], SA:list[list[str]], alphabet:set[str]) -> list[int | float]:
    index_list:list[int] = []
    C:dict[str,int] = {c:0 for c in alphabet}
    LCP_part:list[list[int]] = []
    Flag_part:list[list[int]] = []
    kn:list[int] = [0,0]
    contributions:list[int | float] = []
    
    def dict_reset(dictionary):
        return {c:0 for c in dictionary.keys()}
            
    for i, value in enumerate(LCP):
        if i == 0 or value < LCP[i-1]:
            index_list.append(i)
    
    index_list.append(len(LCP))
    for i in range(len(index_list)-1):
        # is LCP useful? I don't think so
        LCP_part.append(LCP[index_list[i]:index_list[i+1]])
        Flag_part.append(Flag[index_list[i]:index_list[i+1]])

    #for partition in Flag_part:
    #    for b in partition:
        #        if (c:= SA[b][kn[b]]) != '$':
            #            C[c] = C.get(c, 0) + (1 if b == 0 else -1)
            #        kn[b]+=1
            #    contributions.append(sum([abs(i) for i in C.values()]))
            #    C.clear()
    
    for partition in Flag_part:
        S:list[list[str]] = [[],[]]
        for b in partition:
            if (c:=SA[b][kn[b]]) != '$':
                S[b].append(c)
        
        #Why is the distance this way?
        contributions.append((len(set(S[0]+S[1]))-len(set(S[0])&set(S[1])))/len(set(S[0]+S[1])))
    
    
    
    print(LCP_part)
    return contributions


def calculate_distance(contributions:list[int]) -> int:
    return sum(contributions)
    
