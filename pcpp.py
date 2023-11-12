import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
'''Simulation Setup'''
tj=1    # 1, 2, 3, 4
capacity=2000_000   # ci=2000k req/s
exp_distri_mean=200_000   # mean=200k req/s
alpha=10
beta=7
gml_file="Sprint.gml"   # DFN.gml
trials=200  # 200
'''Read GML file, Setup'''
G=nx.read_gml(gml_file,label=None)
n=len(G.nodes)  #switch number
largest_distance=0
for e in G.edges:
    a=e[0]; b=e[1]
    lat1,lng1=G.nodes[a]['Latitude'], G.nodes[a]['Longitude']
    lat2,lng2=G.nodes[b]['Latitude'], G.nodes[b]['Longitude']
    distance=geodesic((lat1,lng1),(lat2,lng2)).km
    value={(a,b):distance}
    nx.set_edge_attributes(G,value,"distance")
    if(distance>largest_distance):
        largest_distance=distance
SC_max=0.4*largest_distance # 0.4 x d_max
CC_max=0.8*largest_distance # 0.8 x d_max
'''Formulation'''
manage_more_set=np.zeros(n,dtype=int)
def kk(i,x):
    total_load=np.sum(x[i]*load)
    return 1 if total_load<=capacity else 0
def Z(i,j):
    if j in G[i]:
        if G.edges[(i,j)]['distance']<=SC_max:
            return 1
    return 0
def v(i,j,x):
    tmp=0
    for k in range(n):
        if(manage_more_set[k]):
            tmp+=x[k][j]*Z(k,j)
    return tmp
def g(i,j,X):
    return alpha if v(i,j,x)<=tj else 0
def Y(i,k):
    if k in G[i]:
        if G.edges[(i,k)]['distance']>CC_max:
            return 1
    return 0
def Con(i,j,x):
    tmp=0
    for k in range(n):
        if(manage_more_set[k]):
            tmp+=x[k][j]*Z(k,j)*Y(i,k)
    return 1 if tmp==0 else 0
def u(i,x):
    uti=0
    for j in range(n):
        if(x[i][j]!=0):
            if kk(i,x)==0:
                uti-=beta
            elif Z(i,j)==0:
                uti-=beta
            elif g(i,j,x)==0:
                uti-=beta
            elif Con(i,j,x)==0:
                uti-=beta
            else:
                uti+=alpha-beta
        #uti+=x[i][j]*(kk(i,x)*Z(i,j)*g(i,j,x)*Con(i,j,x)-beta)
    return uti
neighbor=np.zeros((n,n),dtype=int)
for j in range(n):
    for i in range(n):
        for k in range(n):
            if Z(i,j)*Z(k,j)==1:
                neighbor[i][k]=1
                neighbor[k][i]=1
'''200 Trials'''
np.random.seed(0)
active_controllers=0
state_transitions=0
for trail in tqdm(range(trials)):
    notNE_list=np.arange(n) # notNE controller
    load=np.random.exponential(exp_distri_mean,(n)) # load
    controller_placed=np.ones(n,dtype=int)  # yi
    x=np.random.randint(2,size=(n,n))   # Initialize Strategy
    convergence_time=0  # state transition
    while(len(notNE_list)):
        p=np.random.choice(notNE_list)
        notNE_list=np.delete(notNE_list,np.argwhere(notNE_list==p))
        origin_strategy=np.copy(x[p])
        '''get the best strategy'''
        max_utility=0
        max_strategy=np.zeros(n,dtype=int)
        for b in range(2**n):
            x[p]=np.zeros(n,dtype=int)
            bit=bin(b)
            length=len(bit)-2
            for i in range(length):
                x[p][i]=bit[-(i+1)]

            for j in range(n):
                jk=0; ik=0
                for k in range(n):
                    jk+=Z(j,k)*x[j][k]
                    ik+=Z(p,k)*x[p][k]
                manage_more_set[j]=1 if jk>=ik else 0

            utility=u(p,x)
            if utility>max_utility:
                max_utility=utility
                max_strategy=np.copy(x[p])
        
        x[p]=np.copy(max_strategy)
        if np.sum(x[p])==0:
            controller_placed[p]=0
        else:
            controller_placed[p]=1
        if np.sum(np.not_equal(origin_strategy,x[p]))!=0:
            '''print change of strategy'''
            # print("change" ,convergence_time+1)
            # print("old:",origin_strategy)
            # print("new:",x[p])
            convergence_time+=1
            for k in range(n):
                if neighbor[p][k]:
                    notNE_list=np.append(notNE_list,k)
                        
    ac=np.sum(controller_placed)
    '''print info in every trial'''
    print(x)
    print("Number of active controllers: ",ac)
    print("Number of state transitions: ",convergence_time)
    active_controllers+=ac
    state_transitions+=convergence_time
print("trials:",trials,", number of tj:",tj)
print("avg. active controllers:",active_controllers/trials)
print("avg. state transitions:",state_transitions/trials)

'''show graph part'''
# pos={}
# for i in range (n):
#     pos[i]=(G.nodes[i]['Longitude'],G.nodes[i]['Latitude'])
# nx.draw_networkx(G,pos=pos)
# plt.show()