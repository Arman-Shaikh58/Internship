import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


bios=pd.read_csv("../Pandas/complete-pandas-tutorial/data/bios.csv")
bios["born_year"]=pd.to_datetime(bios["born_date"]).dt.year
values=bios.groupby("born_year")["born_year"].count().sort_values(ascending=False)
plt.bar(values.index, np.log1p(values.values), color='skyblue')
plt.title("Number of People Born Each Year")
plt.xlabel("Year")
plt.ylabel("Number of People")
plt.show()