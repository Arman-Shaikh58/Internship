import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
bios = pd.read_csv("../Pandas/complete-pandas-tutorial/data/bios.csv")
bios["height_category"]=pd.cut(bios["height_cm"],bins=[0,150,180,210],labels=["short","medium","tall"])
height_counts=bios.groupby("height_category")["height_category"].count().sort_values(ascending=False)
plt.bar(height_counts.index, height_counts.values, color='skyblue')
plt.show()