import numpy as np

class KNNRegressor:
    def __init__(self,k=3):
        self.k=k
        
    def fit(self,X,y):
        self.X=np.array(X)
        self.y=np.array(y)
    
    def predict(self,X):
        X=np.array(X)
        predictions=[self._predict(x) for x in X]
        return predictions
    
    def _predict(self,X):
        
        distances=np.linalg.norm(self.X-X,axis=1)
        k_indices=np.argsort(distances)[:self.k]
        k_nearest_tars=self.y[k_indices]
        return np.mean(k_nearest_tars)
        