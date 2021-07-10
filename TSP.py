
import networkx as nx
from networkx.algorithms.matching import max_weight_matching
from networkx.algorithms.euler import eulerian_circuit

import numpy as np

import itertools

#import sys
#print (sys.version)


class TSP:


        

        def __init__(self,dis):
                self.dis = dis # dis is adjacency matrix
                self.n = len(dis)
                self.edges = []
                self.p = []
                self.rank = []
                for i in range(self.n):
                        self.p.append(0)
                        self.rank.append(1)

                self.mst_edges = []
                for i in range(0,self.n):
                        self.p[i] = i
                        for j in range(i+1,self.n):
                                self.edges.append((dis[i][j],(i,j)))

                self.edges.sort(key = lambda e: e[0])
        def root(self,u):
                if(self.p[u] == u): return u
                self.p[u] = self.root(self.p[u])
                return self.p[u]
        def merge(self,u,v):
                pu = self.root(u)
                pv = self.root(v)
                if(pu == pv):
                        return 0
                if(self.rank[pu] < self.rank[pv]): pu,pv = pv,pu
                self.p[pv] = pu
                self.rank[pu] += self.rank[pv]  
                return 1        
        def MST(self):
                for w,(u,v) in self.edges:
                        if(self.merge(u,v)):
                                self.mst_edges.append((u,v))    

          


                    


        def Build(self):
                deg = []
                for i in range(self.n):
                        deg.append(0)       
                for u,v in self.mst_edges:
                        deg[u] += 1
                        deg[v] += 1

                odd = []
                for i in range(self.n):
                        if(deg[i] % 2): odd.append(i)

                m = len(odd)    

                dis_odd = np.zeros((m,m))



                for i in range(m):
                        for j in range(m):
                                dis_odd[i][j] = -self.dis[odd[i]][odd[j]]


                nx_graph = nx.from_numpy_array(dis_odd)   
                matching = max_weight_matching(nx_graph, maxcardinality=True)   


                for u,v in matching:
                        self.mst_edges.append((odd[u],odd[v]))

                G = nx.MultiGraph()

                G.add_edges_from(self.mst_edges)   


                euler_tour = list(eulerian_circuit(G, source = 0))


                euler_tour = list(itertools.chain.from_iterable(euler_tour))
                
                
                path = []

                visited = []

                for i in range(self.n):
                        visited.append(0)

                for i in euler_tour:
                        if(not visited[i]):
                                path.append(i)
                                visited[i]  = 1


                                
                return path
                




