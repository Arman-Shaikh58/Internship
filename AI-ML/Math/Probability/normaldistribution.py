from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
pdf = norm.pdf(x, loc=0, scale=1)
plt.plot(x, pdf)
plt.title("Normal Distribution PDF")
plt.show()
