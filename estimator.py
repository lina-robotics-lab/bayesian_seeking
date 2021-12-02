import numpy as np
class NormalBayesianEstimator:
    def __init__(self,mu_lim=(-1,1),m=50,epsilon=0.01):
        """
            mu_lim = The upper and lower bounds of the unknown mean parameter mu to be estimated.
            m,epsilon: exploration strength(var_1) drops below epsilon after m updates.
        """
        self.mu_0 = np.mean(mu_lim)
        self.var_0= (mu_lim[0]-mu_lim[1])**2/4
        self.var = m*epsilon/(1-epsilon*self.var_0)
        self.xsum = 0
        self.n = 0
        
        self.mu_1 = self.mu_0
        self.var_1 = self.var_0
        
    def get_param(self):
        return self.mu_1,self.var_1
    
    def update(self,x):
        self.xsum += x
        self.n +=1
        
        self.var_1 = 1/(1/self.var_0 + self.n/self.var) 
        self.mu_1 = self.var_1 * (self.mu_0/self.var_0 + self.xsum/self.var)

        

        