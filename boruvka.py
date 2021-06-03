
# Boruvka's algorithm to find Minimum Spanning 
# Tree of a given connected, undirected and weighted graph 
  
from collections import defaultdict 
  
class Graph: 
# functions used in main Boruvkas function
     # union of two sets x and y with rank
    def union(self, parent, rank, x, y): 
        x_root = self.find(parent, x) 
        y_root = self.find(parent, y) 
        if rank[x_root] < rank[y_root]:  
            parent[x_root] = y_root 
        elif rank[x_root] > rank[y_root]: 
            parent[y_root] = x_root 
        else : 
            parent[y_root] = x_root #Make one as root and increment.
            rank[x_root] += 1

    def __init__(self,vertices): 
        self.V= vertices 
        self.graph = []


    # add edge to the graph 
    def addEdge(self,u,v,w): 
        self.graph.append([u,v,w]) 

    # find set of an element i 
    def find(self, parent, i): 
        if parent[i] == i: 
            return i 
        return self.find(parent, parent[i]) 

   
   #constructing MST
    def boruvkaMST(self): 
        parent = []
        rank = [] 
        cheapest =[]      
        numTrees = self.V 
        MSTweight = 0     
        for node in range(self.V): 
            parent.append(node) 
            rank.append(0) 
            cheapest =[-1] * self.V 

        while numTrees > 1: 
            for i in range(len(self.graph)):
                u,v,w = self.graph[i] 
                set1 = self.find(parent, u)
                set2 = self.find(parent, v)
                if set1 != set2:
                    if cheapest[set1] == -1 or cheapest[set1][2] > w :
                        cheapest[set1] = [u,v,w]
                    if cheapest[set2] == -1 or cheapest[set2][2] > w :
                        cheapest[set2] = [u,v,w]
            for node in range(self.V): 
                if cheapest[node] != -1: 
                    u,v,w = cheapest[node] 
                    set1 = self.find(parent, u) 
                    set2 = self.find(parent, v) 
                    if set1 != set2 : 
                        MSTweight += w 
                        self.union(parent, rank, set1, set2) 
                        numTrees = numTrees - 1  
            cheapest =[-1] * self.V 
        print (MSTweight) 
      
g = Graph(4) 
g.addEdge(0, 1, 10) 
g.addEdge(0, 2, 6) 
g.addEdge(0, 3, 5) 
g.addEdge(1, 3, 15) 
g.addEdge(2, 3, 4) 
  
g.boruvkaMST() 