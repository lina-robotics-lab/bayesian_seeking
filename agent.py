import numpy as np
from estimator import NormalBayesianEstimator
import networkx as nx
class Agent:
    def __init__(self,G,m=50,epsilon=0.01):
        """
           m,epsilon: exploration strength(var_1) drops below epsilon after m updates.
        """

        self.G = G.copy()

        self.curr_s = 0

        for n in self.G.nodes:
            self.G.nodes[n]['est']= NormalBayesianEstimator(m = m,epsilon=epsilon)
            self.G.nodes[n]['r_hist']=[]
    
    def next_s(self):
        raise NotImplementedError
    
    def update(self,s,r):
        raise NotImplementedError

class LocalThompsonSamplingAgent(Agent):
    def __init__(self,G,m=50,epsilon=0.01):
        super().__init__(G=G,m=m,epsilon=epsilon)
         
    def next_s(self):
        # Deciding next s using Thompson Sampling.
        muhats = []
        zs = []
        # Sample muhat from the posterior of NormalBayesianEstimation, for all s in the neighborhood(including curr_s).
        for z in list(self.G[self.curr_s])+[self.curr_s]:
            mu_1,var_1 = self.G.nodes[z]['est'].get_param()
            muhats.append(np.random.randn()*np.sqrt(var_1)+mu_1)
            zs.append(z)
        
        z_star = zs[np.argmax(muhats)]

        return z_star
    
    def update(self,s,r):
        
        self.curr_s = s
        
        # We assume the reward at node s is determined completely by its mean and variance, i.e., a Normal r.v.
        self.G.nodes[s]['r_hist'].append(r)
        self.G.nodes[s]['est'].update(r)

class LocalUCBAgent(Agent):
    def __init__(self,G,beta=0.1,m=50,epsilon=0.01):
        """
           beta: high-level eploration strength
           m,epsilon: low-level exploration strength(var_1) drops below epsilon after m updates.
        """
        
        super().__init__(G=G,m=m,epsilon=epsilon)
        self.beta = beta

        

    def next_s(self):
        # Deciding next s using UCB = mu_s + beta * sigma_s, for all s in the neighborhood(including curr_s).
        muhats = []
        zs = []
        for z in list(self.G[self.curr_s])+[self.curr_s]:
            
            mu_1,var_1 = self.G.nodes[z]['est'].get_param()

            muhats.append(mu_1+self.beta * np.sqrt(var_1))
            zs.append(z)
        
        z_star = zs[np.argmax(muhats)]

        return z_star
    
    def update(self,s,r):
        
        self.curr_s = s
        
        # We assume the reward at node s is determined completely by its mean and variance, i.e., a Normal r.v.
        self.G.nodes[s]['r_hist'].append(r)
        self.G.nodes[s]['est'].update(r)
class LocalRandomWalkAgent(Agent):
    def __init__(self,G):
        """
            
        """
        
        self.G = G.copy()
        
        self.curr_s = 0
      
        
    def next_s(self):
        # Deciding next s uniformly from the neighborhood.
        
        z = np.random.choice(list(self.G[self.curr_s])+[self.curr_s])
        
        return z
    
    def update(self,s,r):
        
        self.curr_s = s

class MultiStepLookAheadAgent(Agent):
    def __init__(self,G,T = 100,beta=0.1,m=50,epsilon=0.01,merit = 'UCB'):
        """
           beta: high-level eploration strength
           m,epsilon: low-level exploration strength(var_1) drops below epsilon after m updates.
           T: lookahead time horizon.
        """
        
        super().__init__(G=G,m=m,epsilon=epsilon)
        self.beta = beta
        self.T = T
        A = nx.adj_matrix(G).todense()
        self.A = A+np.eye(len(A)) # Adjacency matrix with self loops on each node. 
        # The todense operation is quite time-consuming, so we only do it once in the initialization.

        # print('self.T',self.T)
        
        if merit == 'UCB':
            self.merit = self._ucb
        elif merit == 'TS':
            self.merit = self._ts
        else:
            print('merit {} is not yet supported.'.format(merit))
            raise TypeError

        
    def _ucb(self):
        muhats = []
        zs = []
        for z in self.G.nodes:    
            mu_1,var_1 = self.G.nodes[z]['est'].get_param()
            muhats.append(mu_1+self.beta * np.sqrt(var_1))
            zs.append(z)
        
        return {z:muhat for (z,muhat) in zip(zs,muhats)}
    
    def _ts(self):
        muhats = []
        zs = []
        for z in self.G.nodes:    
            mu_1,var_1 = self.G.nodes[z]['est'].get_param()
            muhats.append(np.random.randn()*np.sqrt(var_1)+mu_1)
            zs.append(z)
        
        return {z:muhat for (z,muhat) in zip(zs,muhats)}
    

    def next_s(self):
        
        # Get merits in the form of {z:muhat for (z,muhat) in zip(G.nodes,muhats)}
        merits = self.merit()    
        
        # Find the optimal path given the merits
        
        ss = self.path_search(merits)

        return ss[0]
    
    def update(self,s,r):
        
        self.curr_s = s
 
        
        # We assume the reward at node s is determined completely by its mean and variance, i.e., a Normal r.v.
        self.G.nodes[s]['r_hist'].append(r)
        self.G.nodes[s]['est'].update(r)
    
    def path_search(self,merits):
        # The DP method of optimal path search. 
        
        #(1) V0 (s) = μ̂_s , Vl (s) = μ̂_s + max_{z in G[s]} Vl−1(z)
        #(2)  for m = 1, ..., l do
        #         s_m ← arg max_{z in G[s_{m-1}]} Vl−m (z)
        
        S = len(self.G.nodes)
        V = np.zeros((self.T+1,S))
        
        zs = np.array(list(merits.keys()))
        muhats = np.array(list(merits.values()))
        
        V[0,zs] = muhats

        for t in range(1,self.T+1):  
            V[t,:] = np.max(np.multiply(self.A,V[t-1,:]),axis=1).flatten()+muhats
        
        s = [self.curr_s]
#         for t in range(1,self.T+1):
#             nb =list(G[s[-1]])+[s[-1]]       
#             s.append(nb[np.argmax(V[self.T-t,nb])])
        
        nb =list(self.G[self.curr_s])+[self.curr_s]  
        s.append(nb[np.argmax(V[self.T-1,nb])])
        # print(V[self.T-1,nb],'V_T-1',self.T,'T')
        # print(muhats[nb],'muhats')
        return s[1:]