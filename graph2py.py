import numpy as np
import bisect as bis
import heapq as hpq
from sys import getsizeof
from collections import deque

def getIndex(L,x):
    i = 0
    while L[i] != x:
        i += 1
    return i        

def getSrtdSeqMedian(seq):
    n = len(seq)
    if n % 2:
        return seq[n // 2]
    else:
        return (seq[n // 2] + seq[(n // 2) - 1]) / 2
        
# ---------------------------- 1st Class ------------------------------

class AbstractGraph:
    def __init__(self, filename):
        f = open(filename, 'r')

        n_nodes = int(f.readline())

        self._initialize(n_nodes)

        self.n_nodes = n_nodes
        self.n_edges = 0

        for l in f:
            self._update(l)

        f.close()

        self._finalize()

    def _initialize(self, n_nodes):
        self.graph = self._emptygraph(n_nodes)

    def _update(self, l):
        v,u = l.split()
        v = int(v)
        u = int(u)
        self._addedge(v, u)

    def _getdegrees(self):
        degrees = []

        for v in range(self.n_nodes):
            d = self._getdegree(v + 1)
            bis.insort(degrees, (d,v))

        degrees = list(zip(*degrees))
        
        return degrees
    
    def _finalize(self):
        self._savedegreeinfo()

    def _savedegreeinfo(self):
        degrees = self._getdegrees()

        self.degree_min    = degrees[0][0],degrees[1][0]
        self.degree_median = getSrtdSeqMedian(degrees[0]),getSrtdSeqMedian(degrees[1])
        self.degree_max    = degrees[0][-1],degrees[1][-1]
        self.degree_mean   = 2*self.n_edges/self.n_nodes

    def _writedegreeinfo(self, f):
        f.write('Número de vértices = {}\n'.format(self.n_nodes))
        f.write('Número de arestas  = {}\n'.format(self.n_edges))
        f.write('Grau mínimo        = {}\n'.format(self.degree_min))
        f.write('Grau mediano       = {}\n'.format(self.degree_median))
        f.write('Grau máximo        = {}\n'.format(self.degree_max))
        f.write('Grau médio         = {}\n'.format(self.degree_mean))

    def BFS(self, root, filename = None):
        ' breadth first search '
        
        n_nodes = self.n_nodes

        parent = np.full(n_nodes, -1, int)
        level  = np.full(n_nodes, -1, int)

        queue = deque()
        queue.append(root)
        
        level[root - 1] = 0

        while len(queue) >= 1:
            v = queue.popleft()
            neighbors = self._getneighbors(v)
            for u in neighbors:
                if level[u - 1] == -1:
                    level[u - 1] = level[v - 1] + 1
                    parent[u - 1] = v
                    queue.append(u)

        if filename:

             with open(filename, 'w') as f:

                bfs_tree = [(i,j) for i,j in zip(parent,level)]
                f.write(repr(n_nodes) + '\n')
                for node in bfs_tree:

                    f.write(repr(node) + '\n')

        return parent,level
    

    def DFS(graph, root, filename = None):
        ' depth first search '
        
        n_nodes = graph.n_nodes

        parent = np.full(n_nodes, -1, int)
        level  = np.full(n_nodes, -1, int)

        discovered = np.full(n_nodes, False)
        stack = [root]

        discovered[root - 1] = True
        level[root - 1] = 0

        while len(stack) >= 1:
            v = stack.pop(0)
            for u in range(n_nodes)[::-1]:
                if graph._isedge(v, u + 1) and not discovered[u]:
                    discovered[u] = True
                    stack.append(u + 1)


        if filename:

             with open(filename, 'w') as f:

                dfs_tree = [(i,j) for i,j in zip(parent,level)]
                f.write(repr(n_nodes) + '\n')
                for node in dfs_tree:

                    f.write(repr(node) + '\n')

        return parent,level

    def ConnectedComponents(self, filename = None):
        discovered = 0

        n_nodes = self.n_nodes

        parent    = np.full(n_nodes, -1, int)
        level     = np.full(n_nodes, -1, int)
        component = np.full(n_nodes, -1, int)
        components = []

        while discovered < n_nodes:
            root = getIndex(component, -1)

            queue = deque()
            queue.append(root + 1)

            level[root] = 0
            component[root] = root

            discovered += 1

            this = deque()
            this.append(root)

            while len(queue) >= 1:

                v = queue.popleft()

                neighbors = self._getneighbors(v)
                for u in neighbors:
                        if level[u - 1] == -1:
                            discovered   += 1
                            level[u - 1]     = level[v - 1] + 1
                            parent[u - 1]    = v
                            component[u - 1] = root
                            this.append(u - 1)
                            queue.append(u)

            this.appendleft(len(this))
            bis.insort(components, this)

        if filename:

             with open(filename, 'w') as f:
                f.write(str(len(components)) + '\n \n')
                for c in components[::-1]:
                    for l in c:
                        f.write(str(l) + '\n')
                    f.write('\n')
                f.write('\n')

        return discovered, parent, level, component

    def Diameter(self):

        diameter = 0

        for v in range(self.n_nodes):

            temp     = max(self.BFS(v+1)[1])
            diameter = max(temp, diameter)

        return diameter

    def PseudoDiameter(self):

        v = np.random.randint(1,self.n_nodes+1)

        bfs_v = self.BFS(v)[1]

        u = np.argmax(bfs_v) + 1
        pseudo_diam = [0,0,bfs_v[u - 1]]

        while pseudo_diam[-1] > pseudo_diam[-2] and pseudo_diam[-2] >= pseudo_diam[-3]:

            bfs_u = self.BFS(self,u)[1]
            u     = np.argmax(bfs_u) + 1
            pseudo_diam += [bfs_u[u - 1]]

        return pseudo_diam[-1]

    def save(self, filename):
        f = open(filename, 'w')

        self._writedegreeinfo(f)

        f.close()
  
# ---------------------------- 2nd Class ------------------------------

class ArrayGraph(AbstractGraph):  
    
    def _emptygraph(self, n_nodes):
        return np.full((n_nodes, n_nodes), False, dtype=bool)
    
    def _getdegree(self, v): # Use _isedge
        d = 0
        
        for u in range(self.n_nodes):
            if self._isedge(v, u + 1):
                d += 1
        
        return d
    
    def _addedge(self, v, u):
        if not (self.graph[v - 1, u - 1] and self.graph[u - 1, v - 1]):
            self.graph[v - 1, u - 1] = True
            self.graph[u - 1, v - 1] = True
            self.n_edges += 1

    def _isedge(self, v, u):
        return self.graph[v - 1, u - 1] and self.graph[v - 1, u - 1]
    
    def _getneighbors(self, v): # Use _isedge
        return [u + 1 for u in range(self.n_nodes) if self._isedge(v, u + 1)]

# ---------------------------- 3rd Class ------------------------------

class ListGraph(AbstractGraph):

    def _emptygraph(self, n_nodes):
        return [[] for _ in range(n_nodes)]
    
    def _addedge(self, v, u):            
        v_edges = self.graph[v - 1]
        u_edges = self.graph[u - 1]

        if not self._isedge(v, u):
            bis.insort(v_edges, u)
            bis.insort(u_edges, v)
            self.n_edges += 1

    def _finalize(self):
        self._casttondarray()
        self._savedegreeinfo()
    
    def _casttondarray(self):
        for i in range(len(self.graph)):
            self.graph[i] = np.array(self.graph[i])
        self.graph = np.array(self.graph)
    
    def _getdegree(self, v):
        d = len(self.graph[v - 1])
        return d
    
    def _isedge(self, v, u):
        return (u in self.graph[v - 1]) and (v in self.graph[u - 1])
    
    def _getneighbors(self, v):
        return self.graph[v - 1]

# ---------------------------- 4th Class ------------------------------

class AbstractWeightedGraph(AbstractGraph):
    
    totalweight = 0
    
    def _update(self, l):
        v,u,w = l.split()
        v = int(v)
        u = int(u)
        w = float(w)
        if not self._isedge(u,v):
            self.totalweight += w
        self._addedge(v, u, w)
            
    def _initialize(self, n_nodes):
        self.graph   = self._emptygraph(n_nodes)
        self.weights = self._emptyweights(n_nodes)
    
    def Dijkstra(self, root, target = False, path = False):
        n_nodes = self.n_nodes

        distance = np.full(n_nodes, np.Inf, float)
        parent   = np.full(n_nodes, -1, int) 
        explored = np.full(n_nodes, False, bool)
        
        parent[root - 1]   = root 
        distance[root - 1] = 0
        
        priority_queue = []
        hpq.heappush(priority_queue, (0, root))
    
        while len(priority_queue) >= 1:
            d, v = hpq.heappop(priority_queue)
            if not explored[v - 1]:
                explored[v - 1] = True
                neighbors = self._getneighbors(v)
                for u in neighbors:
                    w_vu = self._getweight(v, u)
                    
                    if w_vu < 0:
                        raise Exception("Edge ({}, {}) has negative weight.".format(v,u))
                    
                    if distance[u - 1] > distance[v - 1] + w_vu:
                        distance[u - 1] = distance[v - 1] + w_vu
                        parent[u - 1] = v 
                        hpq.heappush(priority_queue, (distance[u - 1], u))
                        
        if target:
            
            if path == True:
                u = target
                caminho = [u]
                if  parent[u - 1] == -1:
                    raise Exception("There is no path between ({} and {})." .format(root,target))
                while(parent[u - 1] != root) :
                    u = parent[u - 1]
                    caminho.append(u)
                caminho.append(root)
                return distance[target-1],caminho[::-1]
                
            return distance[target-1]

        return distance, parent 
    
    def MSTPrim2(self, root, filename = None):
    
        n_nodes  = self.n_nodes

        costs    = np.full(n_nodes, np.Inf, float)
        parent   = np.full(n_nodes, -1, int) 
        explored = np.full(n_nodes, False, bool)

        parent[root - 1]   = root
        costs[root - 1] = 0
        
        priority_queue = []
        hpq.heappush(priority_queue, (0, root))

        while len(priority_queue) >= 1:

            w, v = hpq.heappop(priority_queue)

            if not explored[v - 1]:

                explored[v - 1] = True
                v_neighbors = self._getneighbors(v)

                for u in v_neighbors:

                    w_uv = self._getweight(u, v)

                    if  costs[u - 1] > w_uv and explored[u-1] == False:
                        parent[u - 1] = v 
                        costs[u - 1] = w_uv
                        hpq.heappush(priority_queue, (costs[u - 1], u))
                        
        if filename == True:
            
            pass
            
                        
        return costs, sum(costs)
    
    def MSTPrim(self, root, filename = None):
    
        n_nodes  = self.n_nodes

        costs    = np.full(n_nodes, np.Inf, float)
        explored = np.full(n_nodes, False, bool)

        costs[root - 1] = 0
        
        priority_queue = []
        hpq.heappush(priority_queue, (0, root))
        
        if filename is not None:
            
            parent   = np.full(n_nodes, -1, int) 
            parent[root - 1]   = root
            
            with open(filename + ".txt", 'w') as f:
                f.write(str(n_nodes) + "\n")
                
                while len(priority_queue) >= 1:
                    w, v = hpq.heappop(priority_queue)

                    if not explored[v - 1]:
                        explored[v - 1] = True
                        neighbors = self._getneighbors(v)
                        
                        p = parent[v - 1]

                        for u in neighbors:
                            w_uv = self._getweight(u, v)
                            if u == p and p != v:
                                f.write(str(p) + " " + str(v) + " " + str(w_uv) + " " + "\n")
                            if  costs[u - 1] > w_uv and explored[u - 1] == False:
                                parent[u - 1] = v 
                                costs[u - 1] = w_uv
                                hpq.heappush(priority_queue, (costs[u - 1], u))
        
        else:
            while len(priority_queue) >= 1:
                    w, v = hpq.heappop(priority_queue)

                    if not explored[v - 1]:
                        explored[v - 1] = True
                        v_neighbors = self._getneighbors(v)

                        for u in v_neighbors:
                            w_uv = self._getweight(u, v)

                            if  costs[u - 1] > w_uv and explored[u-1] == False:
                                costs[u - 1] = w_uv
                                hpq.heappush(priority_queue, (costs[u - 1], u))
        return costs, sum(costs)

    def Eccentricity_slow(self, root):
        # Do Bellman-Ford then take max
        n_nodes = self.n_nodes

        distance = np.full(n_nodes, np.Inf, float)
        distance[root - 1] = 0.
        for length in range(1, n_nodes):
            for v in range(1, n_nodes + 1):
                neighbors = self._getneighbors(v)
                for u in neighbors:
                    distance[v - 1] = min(distance[v - 1], distance[u - 1] + self._getweight(v, u))
        
        verification = np.copy(distance)
        for v in range(1, n_nodes + 1):
                neighbors = self._getneighbors(v)
                for u in neighbors:
                    verification[v - 1] = min(verification[v - 1], verification[u - 1] + self._getweight(v, u))
                    
        if not np.array_equal(distance, verification):
            raise Exception("Graph contains negative cycle.")
            
        return max(distance)
    
    
    def Eccentricity(self, root):
        n_nodes = self.n_nodes

        distance = np.full(n_nodes, np.Inf, float)
        explored = np.full(n_nodes, False, bool)
        
        distance[root - 1] = 0
        
        priority_queue = []
        hpq.heappush(priority_queue, (0, root))
    
        while len(priority_queue) >= 1:
            d, v = hpq.heappop(priority_queue)
            if not explored[v - 1]:
                explored[v - 1] = True
                neighbors = self._getneighbors(v)
                for u in neighbors:
                    w_vu = self._getweight(v, u)
                    
                    if w_vu < 0:
                        raise Exception("Edge ({}, {}) has negative weight.".format(v,u))
                    
                    if distance[u - 1] > distance[v - 1] + w_vu:
                        distance[u - 1] = distance[v - 1] + w_vu
                        
                        hpq.heappush(priority_queue, (distance[u - 1], u))

        return max(distance)
    
    def Diameter(self):
        raise Exception("Undefined for Weighted Graphs.")
    
    def PseudoDiameter(self):
        raise Exception("Undefined for Weighted Graphs.")
    
    def BFS(self, root, filename = None):
        raise Exception("Undefined for Weighted Graphs.")
    
    def DFS(self, root, filename = None):
        raise Exception("Undefined for Weighted Graphs.")

# ---------------------------- 5th Class ------------------------------

class WeightedArrayGraph(ArrayGraph, AbstractWeightedGraph):

    def _emptygraph(self, n_nodes):
        return np.full((n_nodes, n_nodes), np.NaN, dtype=float)

    def _emptyweights(self, n_nodes):
        return self.graph

    def _addedge(self, v, u, w):
        if not self._isedge(v, u):
            self.graph[v - 1, u - 1] = w
            self.graph[u - 1, v - 1] = w
            self.n_edges += 1

    def _isedge(self, v, u):
        return (not np.isnan(self.graph[v - 1, u - 1])) and (not np.isnan(self.graph[v - 1, u - 1]))

    def _getweight(self, v, u):
        return self.weights[v - 1, u - 1]

# ---------------------------- 6th Class ------------------------------

class WeightedListGraph(ListGraph, AbstractWeightedGraph):

    def _emptygraph(self, n_nodes):
        return [[] for _ in range(n_nodes)]
    
    def _emptyweights(self, n_nodes):
        return [[] for _ in range(n_nodes)]
    
    def _addedge(self, v, u, w):
        v_edges = self.graph[v - 1]
        u_edges = self.graph[u - 1]
            
        v_weights = self.weights[v - 1]
        u_weights = self.weights[u - 1]

        if not self._isedge(v, u):
            bis.insort(v_edges, u)
            bis.insort(u_edges, v)
        self.n_edges += 1

        u_idx = getIndex(v_edges, u)
        v_idx = getIndex(u_edges, v)

        v_weights.insert(u_idx, w)
        u_weights.insert(v_idx, w)

    def _finalize(self):
        self._casttondarray()
        self._savedegreeinfo()
    
    def _casttondarray(self):
        for i in range(len(self.graph)):
            self.graph[i]   = np.array(self.graph[i])
            self.weights[i] = np.array(self.weights[i])
        self.graph   = np.array(self.graph)
        self.weights = np.array(self.weights)
        
    def _getweight(self, v, u):
        if self._isedge(v, u):
            u_idx = getIndex(self._getneighbors(v), u)
            return self.weights[v - 1][u_idx]
        else:
            raise Exception("({}, {}) isn't an edge.".format(v, u))