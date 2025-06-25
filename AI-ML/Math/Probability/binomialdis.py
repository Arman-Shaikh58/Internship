import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters for the binomial distribution
n = 10       # number of trials (e.g., tossing a coin 10 times)
p = 0.5      # probability of success on each trial (e.g., probability of heads = 0.5)

# Generate x values from 0 to n (number of possible successes)
x = range(0, n + 1)

# Compute binomial probabilities (PMF - Probability Mass Function)
probs = [binom.pmf(k, n, p) for k in x]

# Plotting
plt.bar(x, probs, color='skyblue', edgecolor='black')
plt.title(f"Binomial Distribution (n={n}, p={p})")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
