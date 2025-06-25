import matplotlib.pyplot as plt
import numpy as np

def get_y(x):
    return 2*x+3

x=np.random.rand(5)*10
y=list(get_y(x1) for x1 in x)
plt.plot(x,y,marker='.',c='blue')
plt.title("Plot of y=2x+3")
plt.xlabel("x")
plt.xticks(np.arange(0, 11, 1))  # Set x-ticks from 0 to 10 with a step of 1
plt.yticks(np.arange(0, 25, 1))  # Set y-ticks from 0 to 24 with a step of 1
plt.ylabel("y") 
plt.grid(True)
plt.show()