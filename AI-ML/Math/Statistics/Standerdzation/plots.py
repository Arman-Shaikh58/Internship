import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Original data
data = np.array([[100], [200], [300], [400], [500]])

# Standardize
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

# Flatten arrays for plotting
original_values = data.flatten()
standardized_values = standardized_data.flatten()

# Plot
plt.figure(figsize=(12, 5))

# Before Standardization
plt.subplot(1, 2, 1)
plt.plot(original_values, marker='o', color='orange')
plt.title("Before Standardization")
plt.xlabel("Index")
plt.ylabel("Original Value")
plt.grid(True)

# After Standardization
plt.subplot(1, 2, 2)
plt.plot(standardized_values, marker='o', color='teal')
plt.axhline(0, color='gray', linestyle='--', label='Mean = 0')
plt.title("After Standardization")
plt.xlabel("Index")
plt.ylabel("Z-score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
