import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

p = 0.7  # Probability of success
x = [0, 1]
pmf = bernoulli.pmf(x, p)# here pmf means Pobability Mass Function this finds probability of both success and falis in the probability

print(pmf)
plt.bar(x, pmf, tick_label=["0 (Fail)", "1 (Success)"])
plt.title(f"Bernoulli Distribution (p = {p})")
plt.ylabel("Probability")
plt.show()
