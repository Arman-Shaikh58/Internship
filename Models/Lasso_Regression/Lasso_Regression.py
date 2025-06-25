import numpy as np

class LassoRegression:
    def __init__(self,lr=0.01,n_iter=1000,lamb=0.01):
        self.lr=lr
        self.n_iter=n_iter
        self.lamb=lamb
        self.w=None
        self.b=0
        
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.w=np.zeros(n_features)
        
        for _ in range(self.n_iter):
            pred=np.dot(X,self.w)+self.b
            
            dw=(-1/n_samples)*(np.dot(X.T,(y-pred)))+self.lamb*np.sign(self.w)
            db= (-1/n_samples)*(np.sum(y-pred))      
            
            self.w=self.w-self.lr*dw
            self.b=self.b-self.lr*db 
            
    def predict(self,X):
        return np.dot(X,self.w)+self.b
    