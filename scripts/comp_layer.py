from typing import Counter
from XBWT import Tree, Node
from ResearchLab import importTreeFromGraphviz
from pathlib import Path

class XBWT():
    
    def __init__(self, tree):
        '''
            Constructor for the class XBWT
            
            Args:
                tree (Tree): the tree structure of which we want to calculate the XBWT
        '''
        
        self.__tree:Tree = tree
        self.__Child:list[int]      # This array contains the informations about the number of children of each entry
        self.__SA:list[str]         # The list of labels sorted by their parent-to-root linearization
        
        # These are not the data structure of the original XBWT (that is SA and Last). The array last is the third returned data structure 
        self.__SA, self.__Child, _= self.__compute_XBWT__()
        
        # This is basically a counter for the number of children of nodes with the same tag
        self.__F:dict[str, int] = self.__compute_F__()
        
        # This is the tree height
        self.__tree_height:int = self.__tree.getHeight()
      
    def __compute_F__(self)-> Counter:
        '''
            Calculates the counter of the number of children of nodes with the same label
            
            Returns:
                F (Counter): The aforementioned counter
        '''
        F:Counter = Counter()
        for i, label in enumerate(self.__SA):
            # The entry of the counter F must be incremented by the number of children of the entry at position i
            F[label] = F.get(label,0) + self.__Child[i] 
        return F
    
    def __str__(self) -> str:
        '''
            __str__ function
        '''
        return '''
            SA = {}\n
            Child = {}
        '''.format(self.__SA, self.__Child)
        pass

    @staticmethod
    def __get_parent_to_root(i:Node) -> str:
        '''
            Calculate the parent to root linearization of the node i
            
            Args:
                i (Node): The node of which we want to calculate the PTR_linearization
                
            Returns:
                PTR_linearization (str): The concatenation of the labels of the parent-to-root path
        '''
        
        to_ret:str = ''
        while((p:= i.getParent()) is not None):
            to_ret += p.getLabel()
            i = p

        return to_ret
        
    def __compute_XBWT__(self)-> tuple[list[str],list[int], list[bool]]:
        '''
            A function to calculate the XBWT. This is not the original XBWT, 
            since the second returned data structure gives us the information about the number of children of each entry
                
            Returns:
                SA (list of str): The label of each node sorted by their PTR_linearization
                Child (list of int): The number of children of each entry
                Last (list of bool): A boolean flag that tells us if the node is the rightmost child of its parent
                
        '''
        
        entries:list[tuple[str,str,bool,int]] = []
        for i in self.__tree.getNodes(): # For each entry...
            # ...we calculate the parent to root linearization...
            PTR_linearization:str = XBWT.__get_parent_to_root(i)
            # ...and store the label, PTR_linearization, the information about its position and the number of children
            entries.append((i.getLabel(), PTR_linearization, i.isRightmost(), len(i.getChildren())))
        
        # This list is sorted by the lexicographical order of the PTR_linearization
        entries = sorted(entries, key=lambda x: x[1])
        
        # We fetch what we need
        SA:list[str] = [i[0] for i in entries]
        Child:list[int] = [i[2] for i in entries]
        
        return SA, Child, [i[2] for i in entries]
        
    # A series of Getters methods  
    def get_tree(self)-> Tree:
        return self.__tree
    
    def get_F(self)-> dict[str, int]:
        return self.__F
        
    def get_child(self)-> list[int]:
        return self.__Child
        
    def get_SA(self)-> list[str]:
        return self.__SA
        


# def string_to_tree(string:str)->Tree:
    tree:Tree = Tree()
    stack = []
    parent = None
    
    for c in string:
        match c:
            case '(':
                stack.append(parent)
            case ')':
                stack.pop()
            case ',':
                continue
            case ' ':
                continue
            case _:
                parent = stack[-1] if stack else None
                node = Node(c)
                tree.insert(node, parent)
                parent = node
    
    return tree
# 
# def file_to_tree(path:str)->list[Tree]:
    try:
        with open(path, "r+") as f:
            input = [tree.strip() for tree in f.read().split(FILE_SEPARATOR) if tree.strip()]
            return [string_to_tree(i) for i in input]
    except FileNotFoundError:
        print("File not found")
        exit(1)
#         
# def input_to_tree()-> Tree:
    try:
        return string_to_tree(input())
    except Exception:
        print("The input is empty")
        exit(1)
# 


if __name__ == "__main__":
    
    tree_str:str = "(A(B(D,E)S,F(L,N,G))"
    
    #print(os.getcwd())
    tree:Tree = importTreeFromGraphviz(Path.cwd() / "scripts" / "tree_p2.gv")
    tree.addDollarsToLeaves()
    
    # print([(i.label, [j.label for j in i.getChildren()]) for i in tree.nodes])
    # realTrees:list[Tree] = file_to_tree(os.path.join(os.getcwd(),"tree.txt"))
    
    x = XBWT(tree)
    SA, Child = x.__compute_XBWT__()
    F = x.__compute_F__()
    print(SA, "\n", Child, "\n", F, "\n\n")
