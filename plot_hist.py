import pandas as pd
import os
import matplotlib.pyplot as plt

PATH = "C:\\Users\\Andrew\\Documents\MASc\\LoadCoincidence\\load_data\\"

df = pd.read_parquet(PATH+"residential_loads.parquet")

loads = []

for d in df.index:
    if d.month==4 and d.hour==13 and d.weekday()<5:
        loads += df.loc[df.index==d].values.flatten().tolist()
        

fig1, ax1 = plt.subplots()
ax1.hist(df[df.columns[0]], 500)
ax1.set_title(df.columns[0])

fig2, ax2 = plt.subplots()
ax2.hist(df[df.columns[45]], 500)
ax2.set_title(df.columns[45])

fig3, ax3 = plt.subplots()
ax3.hist(df[df.columns[99]], 500)
ax3.set_title(df.columns[99])

fig4, ax4 = plt.subplots()

one_household_loads = [p for d, p in zip(df.index, df[df.columns[0]]) if d.month==4 and d.hour==13 and d.weekday()<5]
ax4.hist(one_household_loads, 20)
ax4.set_title(df.columns[0]+" weekdays at 1pm in April")

fig5, ax5 = plt.subplots() 
ax5.hist(loads, 500)
ax5.set_title("All, weekdays at 1pm in April")
plt.show()     