import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
np.random.seed(0)
data = np.random.normal(loc=50, scale=10, size=1000)  # Normal distribution

# Calculate statistics
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data, keepdims=True).mode[0]
std = np.std(data)

# Plot
plt.figure(figsize=(12, 6))
sns.histplot(data, bins=30, kde=True, color='skyblue')

# Mark lines for mean, median, mode
plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2f}')
plt.axvline(mode, color='blue', linestyle='-.', label=f'Mode: {mode:.2f}')

# Shade 1 standard deviation range
plt.axvspan(mean - std, mean + std, alpha=0.2, color='orange', label='±1 Std Dev')

# Labels and legend
plt.title('Understanding Mean, Median, Mode and Standard Deviation')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
