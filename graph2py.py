import numpy as np
import bisect as bis
from collections import deque

bool_to_int = lambda x: 1 if x else 0
b2i = bool_to_int

def getIndex(L,x):
    i = 0
    while L[i] != x:
        i += 1
    return i

def fixSrtdSeqRight(seq, idx):
    i   = idx
    val = seq[i]
    n = len(seq)

    while i < (n - 1) and seq[i+1] < seq[i]:
        seq[i]   = seq[i+1]
        seq[i+1] = val

        i += 1

    return i

def nudgeSeqRight(seq,idx_str,idx_end):
    val = seq[idx_str]
    seq[idx_str:idx_end] = seq[idx_str + 1:idx_end + 1]
    seq[idx_end] = val

def getSrtdSeqMedian(seq):
    n = len(seq)
    if n % 2:
        return seq[n // 2]
    else:
        return (seq[n // 2] + seq[(n // 2) - 1]) / 2

## Here's the First Class - AbstractGraph

class AbstractGraph:
    def __init__(self, filename):
        f = open(filename, 'r')

        n_nodes = int(f.readline())

        self.graph = self._initialize(n_nodes)

        self.n_nodes = n_nodes
        self.n_edges = 0

        for l in f:
            v,u = l.split()
            v = int(v)
            u = int(u)
            self._update(v, u)

        f.close()

        self._finalize()

    def _initialize(self, n_nodes):
        return self._emptygraph(n_nodes)

    def _emptygraph(self, n_nodes):
        pass

    def _update(self, v, u):
        self._addedge(v, u)

    def _addedge(self, v, u):
        pass

    def _getdegrees(self):
        degrees = []

        for v in range(self.n_nodes):
            d = self._getdegree(v + 1)
            bis.insort(degrees, d)

        return degrees

    def _getdegree(self, v):
        pass

    def _finalize(self):
        self._savedegreeinfo()

    def _savedegreeinfo(self):
        degrees = self._getdegrees()

        self.degree_min    = degrees[0]
        self.degree_median = getSrtdSeqMedian(degrees)
        self.degree_max    = degrees[-1]
        self.degree_mean   = 2*self.n_edges/self.n_nodes

    def _isedge(self, v, u):
        pass

    def _getneighbors(self, v):
        pass

    def _writedegreeinfo(self, f):
        f.write('Número de vértices = {}\n'.format(self.n_nodes))
        f.write('Número de arestas  = {}\n'.format(self.n_edges))
        f.write('Grau mínimo        = {}\n'.format(self.degree_min))
        f.write('Grau mediano       = {}\n'.format(self.degree_median))
        f.write('Grau máximo        = {}\n'.format(self.degree_max))
        f.write('Grau médio         = {}\n'.format(self.degree_mean))

    def BFS(self, root, filename = None):
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

    def Pseudo_diameter(self):

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


## Here's the second Class - ArrayGraph

class ArrayGraph(AbstractGraph):

    def _emptygraph(self, n_nodes):
        return np.full((n_nodes, n_nodes), False, dtype=bool)

    def _getdegree(self, v):
        d = 0

        for u in range(self.n_nodes):
            d += b2i(self.graph[v - 1, u])

        return d

    def _addedge(self, v, u):
        if not (self.graph[v - 1, u - 1] and self.graph[u - 1, v - 1]):
            self.graph[v - 1, u - 1] = True
            self.graph[u - 1, v - 1] = True
            self.n_edges += 1

    def _isedge(self, v, u):
        return self.graph[v - 1, u - 1] and self.graph[v - 1, u - 1]

    def _getneighbors(self, v):
        return [u + 1 for u in range(self.n_nodes) if self._isedge(v, u + 1)]

## Here's the third Class - ListGraph

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
