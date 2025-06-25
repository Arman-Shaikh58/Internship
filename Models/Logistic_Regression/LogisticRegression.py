import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression():
    def __init__(self,lr=0.001,n_iter=1000):
        self.lr=lr
        self.n_iter=n_iter
        self.weight=None
        self.bias=None
        
    def fit(self,X,y):
        n_smaple,n_features=X.shape
        self.weight=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iter):
            prob=np.dot(X,self.weight)+self.bias
            predictions=sigmoid(prob)
            dw=(1/n_smaple)*np.dot(X.T,(predictions-y))
            db=(1/n_smaple)*np.sum(predictions-y)
            self.weight=self.weight-self.lr*dw
            self.bias=self.bias-self.lr*db

        
    def pridect(self,X):
        prob=np.dot(X,self.weight)+self.bias
        predictions=sigmoid(prob)
        pre=[0 if y<=0.5 else 1 for y in predictions]
        return pre