from copy import deepcopy
from collections import Counter

def calculate_ds(SA:list[list[str]], Child:list[list[int]], F:dict[str, int], n_0:int, n_1:int, k:int) -> tuple[list[int], list[int]]:
    '''
    Calculates the necessary structures to compute the distance described in Dolce et Al.,2024
   
    Args:
        SA (tuple of lists of string):  The arrays SA0 and SA1, representing the labels of the two XBWTs
        Child (tuple of lists of int):  The arrays Child0 and Child1, representing the number of children each node in the two XBWTs has
        F (dictionary of str and int):  The dictionary that gives the indexes on where in the fictitious merged XBWT the prefixes 
                                        that start with a particular character must be placed
        n_0 (int): The size of SA0
        n_1 (int): The size of SA1
    
    Returns:
        Flag (list of int): The array showing the contribution of each one of the two XBWTs
        LCP  (list of int): The array representing the longest common prefix between the element at index i and the one at index i-1
                            (LCP[0] is 0 per convention since LCP('',\\epsilon) = 0)
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


def calculate_partition_contributions(Flag:list[int], LCP:list[int], SA:list[list[str]], alphabet:set[str]) -> list[float]:
    '''
    Calculates how much each set of the partition contributes to the final pseudometric.
    The contribution (calculated following Dolce et Al, 2024) is calculated starting from the Jaccard distance
    on the contributions (multiset of characters) from the two trees
    
    
    Args:
        Flag (list of int): The array showing the contribution of each one of the two XBWTs
        LCP  (list of int): The array representing the longest common prefix between the element at index i and the one at index i-1
                            (LCP[0] is 0 per convention since LCP('',\\epsilon) = 0)
        SA (tuple of lists of string):  The arrays SA0 and SA1, representing the labels of the two XBWTs
        alphabet (set of string): The union of the alphabets of the two XBWTs
        
    Returns:
        contributions (list of float): The contribution given by each set in the partition
     '''
    
    
    Flag_part:list[list[int]] = []      # Here the partition of the Flag array will be stored
    kn:list[int] = [0,0]                # Two variables to keep track of the current reading index for the two trees
    contributions:list[float] = []      # Here the contributions related to each set in the partition will be stored
    
    
    # TODO: Consider substitution with https://pypi.org/project/multiset/
    def multiset_union(a:Counter, b:Counter)-> list:
        '''
            A method to calculate the union of two multisets.
            The union of two multisets is defined as a multiset on which an element
            will appear with multiplicity equal to the maximum between the multiplicity of the element in the original input
            
            Args:
                a (Counter object): This represents the first multiset
                b (Counter object): This represents the second multiset
                
            Returns:
                union (list): This will be the union of the two multisets
        '''
        return [i for i in set(a.keys()).union(set(b.keys())) for _ in range(max(a.get(i,0), b.get(i,0)))]
    
    def multiset_intersection(a:Counter, b:Counter) ->list:
        '''
            A method to calculate the intersection of two multisets.
            The union of two multisets is defined as a multiset on which an element
            will appear with multiplicity equal to the maximum between the multiplicity of the element in the original input
            
            Args:
                a (Counter object): This represents the first multiset
                b (Counter object): This represents the second multiset
                
            Returns:
                union (list): This will be the intersection of the two multisets
        '''
        return [i for i in set(a.keys()).union(set(b.keys())) for _ in range(min(a.get(i,0), b.get(i,0)))]
    
    
    def dict_reset(d:dict)->dict:
        '''
            A method to reset a dictionary
            
            Args:
                d (dict): The dictionary set to be reset
            
            Returns:
                d (dict): The same dictionary where the values of each key is set to zero
        '''
        return {c:0 for c in d.keys()}
            
    # This is where all the bounds of each set in the partition is stored
    index_list:list = [i for i,value in enumerate(LCP) if i == 0 or value < LCP[i-1]] + [len(LCP)]

    #Splitting the orginal Flag array into a list of subarrays
    for i in range(len(index_list)-1):
        # is LCP useful? I don't think so
        # LCP_part.append(LCP[index_list[i]:index_list[i+1]])
        Flag_part.append(Flag[index_list[i]:index_list[i+1]])

    # Navigating through each partition
    for i, partition in enumerate(Flag_part):
        #This is where the elements with their multiplicity from each XBWT will be stored
        C:list[Counter] = [Counter(), Counter()]
        for b in partition:
            c:str =  SA[b][kn[b]]
            C[b][c] = C[b].get(c,0) + 1
            kn[b]+=1
        
        #Calculating the union and intersection of the multiset and using them to calculate the contribution
        uni:list[int] = multiset_union(*C)
        inter:list[int] = multiset_intersection(*C)
        contributions.append((len(uni) - len(inter))/len(uni))    

    return contributions


def calculate_distance(contributions:list[float]) -> float:
    '''
        A simple wrapper of the sum function.
        Useful should the definition be modified in future
        
        Args:
            contributions (list of float): The contributions from each subset
            
        Returns:
            distance (float): The sum of the contributions
    '''
    return sum(contributions)
    



