import networkx as nx
import numpy as np
from geopy.distance import geodesic

tj = 1    # 1, 2, 3, 4
capacity = 2000_000   # ci=2000k req/s
exp_distri_mean = 200_000   # mean=200k req/s
alpha = 10
beta = 7
gml_file = "Sprint.gml"   # DFN.gml

G = nx.read_gml(gml_file, label=None)
n = len(G.nodes)  # switch number
largest_distance = 0
for e in G.edges:
    a = e[0]
    b = e[1]
    lat1, lng1 = G.nodes[a]['Latitude'], G.nodes[a]['Longitude']
    lat2, lng2 = G.nodes[b]['Latitude'], G.nodes[b]['Longitude']
    distance = geodesic((lat1, lng1), (lat2, lng2)).km
    value = {(a, b): distance}
    nx.set_edge_attributes(G, value, "distance")
    if (distance > largest_distance):
        largest_distance = distance
SC_max = 0.4*largest_distance  # 0.4 x d_max
CC_max = 0.8*largest_distance  # 0.8 x d_max

def kk(i, x, load):
    total_load = np.sum(x[i]*load)
    return 1 if total_load <= capacity else 0
def Z(i, j):
    if i==j:
        return 1
    if j in G[i]:
        #print(i,j,G.edges[(i,j)]['distance'],SC_max)
        if G.edges[(i, j)]['distance'] <= SC_max:
            return 1
    return 0
def v(j, x):
    tmp = 0
    for k in range(n):
        tmp += x[k][j]*Z(k, j)
    return tmp
def g(j, x):
    return alpha if v(j, x) <= tj else 0
def Y(i, k):
    if k in G[i]:
        if G.edges[(i, k)]['distance'] > CC_max:
            return 1
    return 0
def Con(i, j, x):
    tmp = 0
    for k in range(n):
        tmp += x[k][j]*Z(k, j)*Y(i, k)
    return 1 if tmp == 0 else 0
def u(i, x, load):
    uti = 0
    for j in range(n):
        if (x[i][j] != 0):
            if kk(i, x, load) == 0:
                #print("kk")
                uti -= beta
            elif Z(i, j) == 0:
                #print("Z")
                uti -= beta
            elif g(j, x) == 0:
                #print("g")
                uti -= beta
            elif Con(i, j, x) == 0:
                #print("con")
                uti -= beta
            else:
                uti += alpha-beta
    return uti

class Environment():
    def __init__(self):
        self.x=np.random.randint(2,size=(n,n))
        self.observation_space = (2**n)
        self.action_space = 2**n
        self.n=n
        
        self.load=np.random.exponential(exp_distri_mean,(n)) # load

    def step(self, action, p):
        '''next_state'''
        action_array=self.inverse_extract(action)
        self.x[p] = action_array
        next_state=self.extract_state(p)
        '''reward'''
        # tmp_controller_placed = np.ones(n, dtype=int)
        # if np.sum(self.x[p]) == 0:
        #     tmp_controller_placed[p] = 0
        # else:
        #     tmp_controller_placed[p] = 1
        # if np.sum(self.controller_placed) > np.sum(tmp_controller_placed):
        #     reward = 10
        # elif np.sum(self.controller_placed) < np.sum(tmp_controller_placed):
        #     reward = -10
        # else:
        #     reward = u(p, self.x, self.load)
        reward = u(p, self.x, self.load)

        return next_state, reward
    
    def reset(self):
        p=np.random.randint(0,n)
        self.x=np.random.randint(2,size=(n,n))
        for i in range(n):
            for j in range(n):
                if j in G[i]:
                    if G.edges[(i,j)]['distance']>SC_max:
                        self.x[i][j]=0
                else:
                    self.x[i][j]=0
        s1=self.extract_state(p)
        state=np.array((s1))
        return state, p
    
    def extract_state(self,p):
        s1 = self.x[p]
        # 0 for stay, 1 for change
        s2 = [0 if np.sum(self.x, axis=0)[j]==tj else 1 for j in range(n)]
        cvt1=0
        cvt2=0
        for i in range(n):
            cvt1*=2; cvt2*=2
            cvt1+=s1[i]
            cvt2+=s2[i]
        return cvt1
    
    def inverse_extract(self,v):
        s=np.zeros(n,dtype=int)
        for idx in range(n):
            i=n-1-idx
            s[i]=v%2
            v/=2
        return s

    def render(self):
        pass

    def close(self):
        pass