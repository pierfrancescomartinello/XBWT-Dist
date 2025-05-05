from copy import deepcopy
from colorama import Fore, Style


def flag_experiment(SA:list[list[str]], Child:list[list[int]], F:dict[str, int], n_0:int, n_1:int) -> tuple[list[int], list[int]]:
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
    
    working:bool = True     # This boolean variable checks if the algorithm has to end
    prev_2:bool = False     # This boolean variable checks if the previous two iteration of the array Flag are equal
    
    while working:
        h += 1
        id:int = 0
        kn:list[int] = [0,0]
        
        temp_F:dict[str, int] = deepcopy(F)
        # The first two indexes will contain the roots of the XBWTs
        Flag:list[int] = [0,1] + [-1]*(n-2)
        Block_Id:dict[str, int] = {i:-1 for i in F.keys()}
        
        for k in range(n):
            if LCP[k] not in (-1, h-1):
                id:int = k          # This means a new block is starting--
                
            b:int = Flagh[k]        
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
                        #print("Element {} at position {} from tree {} has been updated at iteration {}, step {}".format(c,kn[b],b,h,k))

                temp_F[c] += children
                
            kn[b] += 1
        
        
        
        if Flagh == Flag:       # When Flag is finally stable...
            if not prev_2:      # ...we wait for a third iteration
                prev_2 = True
            else:
                working = False # ...and that's when the algorithm stops
            print("Flag at iteration " + Fore.GREEN + "{}".format(h) +Style.RESET_ALL+ ": {}".format(Flag))
        else:
            # Preparing for the next iteration
            Flagh = Flag
            print("Flag at iteration {}: {}".format(h,Flag))
        #input()
    return Flagh, LCP
    
def insert_SA() -> tuple[list[str], list[str]]:
    '''
    Allows the insertion of the two XBWTs from stdin
    
    Returns:
        tuple[list[str],list[str]]: The two arrays (SA0, SA1)
    '''
    
    SA0:list[str] = []
    SA1:list[str] = []
    
    xbwt0 = input("Insert all the SA array in a single string, ENTER to stop: ")
    SA0 = [c for c in xbwt0]
    print("Now the other list")
    xbwt1 = input("Insert all the SA array in a single string, ENTER to stop: ")
    SA1 = [c for c in xbwt1]
        
    return SA0, SA1
    
def insert_Child(n_0:int, n_1:int) -> tuple[list[int], list[int]]:
    '''
    Allows the insertions of the Child arrays relative to the previously inserted XBWTs, from stdin
    
    Args:
        n_0 (int): The size of the array SA0
        n_1 (int): The size of the array SA1
    
    Returns:
        tuple[list[int],list[int]] : The two arrays (Child0, Child1)
    '''
    
    
    Child0:list[int] = []
    Child1:list[int]= []
    
    for i in range(n_0):
        Child0.append(int(input("# of children of element {} if T0: ".format(i))))
    print("Now the other list")
    for j in range(n_1):
        Child1.append(int(input("# of children of element {} if T1: ".format(j))))
    
    return Child0, Child1
      
def insert_F(alphabet:list[str]) ->dict[str, int]:
    '''
    Allows the insertion of the array F relative to the combined alphabets of the two previously inserted XBWTs, from stdin
    
    Args:
        alphabet(list of strings): The combined alphabet of the two XBWTs
    
    Returns:
        dict[str, int]: The dictionary F that counts the number of children of nodes with label lexicographically smaller than the key plus 2 in the two XBWTs
    
    '''
    
    F:dict[str, int] = {}
    
    for c in alphabet:
        F[c] = int(input("F[{}]?: ".format(c)))
    return F
    
if __name__ == "__main__":
    '''
    Main function
    '''
    
    
    if (g:=int(input("Automatic(0), Manual(1), From File(2)[NOT IMPLEMENTED]: "))) == 0:
        # Here there is an automatic example, below as a comment there is another one
        SA0:list[str] = ["A", "B", "C", "B", "B", "D", "$", "$", "A", "E", "$", "F", "$"]
        SA1:list[str] = ["A", "B", "C", "D", "$", "$", "A", "C", "$", "B", "D", "E", "$", "A", "$"]
        Child0:list[int] = [2,2,2,1,1,1,0,0,1,1,0,1,0]
        Child1:list[int] = [3,2,2,1,0,0,1,1,0,1,1,1,0,1,0]
        F:dict[str, int] = {"A":2, "B":10, "C":17, "D":22, "E":25, "F":27}
    elif g == 1:
        # Manual insertion from the stdin
        SA0,SA1 = insert_SA()
        Child0, Child1 = insert_Child(len(SA0),len(SA1))
        F = insert_F(sorted(set(SA0+SA1))[1:])
        
    else:
        # Here's where the File implementation/API to work with Dolce's code will go
        # TODO implement from file insertion
        print("The option is not comtemplated.")
        exit(1)
        
    #Prepping the file structure to pass to the function
    SA:list[list[str]] = [SA0, SA1]
    Child:list[list[int]] = [Child0, Child1]
    
    flag:list[int] = []
    lcp:list[int] = []
    
    flag,lcp = flag_experiment(SA, Child, F, len(SA0), len(SA1))
    print(flag)
    print([i if i != -1 else 'w' for i in lcp])


# SA0:list[str] = ["A","B","C","$","$","A","G","$","F"]
# SA1:list[str] = ["A", "C", "D","$", "A","G","$", "$","F"]
# Child0:list[int] = [2,1,2,0,0,1,1,0,1]
# Child1:list[int] = [2,2,1,0,1,1,0,0,1]
# F:dict[str, int] = {"A":2, "B":8, "C":9,"D":13,"F":14,"G":16}
# 
# 
# 
# 
# 
#         SA0:list[str] = ["A", "B", "C", "B", "B", "D", "$", "$", "A", "E", "$", "F", "$"]
#         SA1:list[str] = ["A", "B", "C", "D", "$", "$", "A", "C", "$", "B", "D", "E", "$", "A", "$"]
#         Child0:list[int] = [2,2,2,1,1,1,0,0,1,1,0,1,0]
#         Child1:list[int] = [3,2,2,1,0,0,1,1,0,1,1,1,0,1,0]
#         F:dict[str, int] = {"A":2, "B":10, "C":17, "D":22, "E":25, "F":27}
#            