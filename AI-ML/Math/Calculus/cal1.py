import numpy as np
import matplotlib.pyplot as plt

def get_y(r):
    return 2 * np.pi * r

x_data = np.linspace(0, 3, 100)
y_data = [get_y(i) for i in x_data]

bar_width = x_data[1] - x_data[0]  # remove space between bars

plt.bar(x_data, y_data, width=bar_width, align='center')
plt.plot(x_data, y_data, "r-")
plt.grid(True)
plt.xlabel("Radius")
plt.ylabel("Area")
plt.title("This is how area of circle is finded")
plt.show()
