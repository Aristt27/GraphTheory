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

    def _writedegreeinfo(self, f):
        f.write('Número de vértices = {}\n'.format(self.n_nodes))
        f.write('Número de arestas  = {}\n'.format(self.n_edges))
        f.write('Grau mínimo        = {}\n'.format(self.degree_min))
        f.write('Grau mediano       = {}\n'.format(self.degree_median))
        f.write('Grau máximo        = {}\n'.format(self.degree_max))
        f.write('Grau médio         = {}\n'.format(self.degree_mean))

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
