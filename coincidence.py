import pandas as pd
import os
import matplotlib.pyplot as plt

PATH = "C:\\Users\\Andrew\\Documents\MASc\\LoadCoincidence\\load_data\\"

df = pd.read_parquet(PATH+"residential_loads.parquet")

cols = df.columns

nsamples = 1000
nmax = 75

cf = [0 for n in range(1, nmax+1)]
mcl = [1 for n in range(1, nmax+1)]

for n in range(1, nmax+1):
    print("Coincidence factor for {}".format(n))
    for i in range(0, nsamples):
        subset = df.sample(n=n, axis=1)
        cf_sample = subset.sum(axis=1).max()/subset.max().sum()
        mcl_sample = subset.sum(axis=1).min()/subset.max().sum()
        cf[n-1] = max(cf_sample, cf[n-1])
        mcl[n-1] = min(mcl_sample, mcl[n-1])

print(cf)

fig1, ax1 = plt.subplots()
ax1.plot(cf, '.')

frg2, ax2 = plt.subplots()
ax2.plot(mcl, ".")

plt.show()