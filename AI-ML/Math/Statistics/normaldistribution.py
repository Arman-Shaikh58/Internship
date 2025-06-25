import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(loc=50, scale=10, size=1000)
plt.hist(data, bins=30, density=True)
plt.title("Normal Distribution with μ=50 and σ=10")
plt.show()
