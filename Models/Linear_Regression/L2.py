import numpy as np

class L2:
    def __init__(self,lr=0.001,n_iter=1000):
        self.lr=lr
        self.n_iter=n_iter
        self.w=None
        self.b=None
    
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.w=np.zeros(n_features)
        self.b=0
        
        for _ in range(self.n_iter):
            y_pred=np.dot(X,self.w)+self.b
            
            dw=(1/n_samples)*np.dot(X.T,(y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)
            
            self.w=self.w-self.lr*dw
            self.b=self.b-self.lr*db
            
    def pridict(self,X):
        return np.dot(X,self.w)+self.b
        