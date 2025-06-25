import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from Lasso_Regression import LassoRegression

X, y = make_regression(n_samples=100, n_features=1, noise=20)
model = LassoRegression(lr=0.01, n_iter=1000, lamb=0.1)
model.fit(X, y)
y_pred = model.pridict(X)

plt.scatter(X, y, color='green')
plt.plot(X, y_pred, color='black')
plt.title("Lasso Regression (from Scratch)")
plt.show()
