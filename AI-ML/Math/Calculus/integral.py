import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Standard Normal PDF
def normal_pdf(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

# Integrate from -1 to 1
result, error = quad(normal_pdf, -1, 1)

print(f"Integral (probability between -1 and 1): {result:.4f}")
print(f"Estimated error: {error:.2e}")

# Plot for visualization
x = np.linspace(-4, 4, 500)
y = normal_pdf(x)

plt.plot(x, y, label='Normal PDF')
plt.fill_between(x, y, where=(x >= -1) & (x <= 1), color='skyblue', alpha=0.5, label='Area [-1,1]')
plt.title("Standard Normal Distribution")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True)
plt.show()
