import numpy as np
import pandas as pd

class linearRegression:
    def __init__(self,lr=0.011,n_iter=1000):
        self.lr=lr
        self.n_iter=n_iter
        self.w=0
        self.b=0
    
    def fit(self,X,y):
        n_saples,n_features=X.shape
        self.w=np.zeros(n_features)
        
        for _ in range(self.n_iter):
            y_pred=np.dot(X,self.w)+self.b
            
            dw=(1/n_saples)*(np.dot(X.T,(y_pred-y)))
            db=(1/n_saples)*(np.sum(y_pred-y))
            
            self.w=self.w-self.lr*dw
            self.b=self.b-self.lr*db
        
    def predict(self,X):
        y_pred=np.dot(X,self.w)+self.b
        return y_pred
    
        
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

X,y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1234)
# fig=plt.figure(figsize=(8,6))
    # plt.scatter(X[:,0],y,s=30)
    # plt.show()

reg=linearRegression()
reg.fit(X_train,y_train)
preds=reg.predict(X_test)
def mse(y,y_pred):
    return np.mean((y-y_pred)**2)
print(mse(y_test,preds))
print(r2_score(y_test,preds))

y_pred_line=reg.predict(X)
cmap=plt.get_cmap('viridis')
fig=plt.figure(figsize=(8,6))
m1=plt.scatter(X_train,y_train,color=cmap(0.9),s=10)
m2=plt.scatter(X_test,y_test,color=cmap(0.5),s=10)
plt.plot(X,y_pred_line,color="black")
plt.show()