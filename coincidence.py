import pandas as pd
import os
from math import floor
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def decay(x, k, gamma, c):
    return k*(1/(x**gamma)) + c

PATH = "C:\\Users\\Andrew\\Documents\MASc\\LoadCoincidence\\load_data\\"

df = pd.read_parquet(PATH+"residential_loads.parquet")

households = df.max()[df.max()<30].index.tolist()

df = df[households]

df = df[((df.index.month==6)|(df.index.month==7)|(df.index.month==8))&(df.index.hour==13)&(df.index.weekday<5)]

maxdf = df.max()
mindf = df.min()

nsamples = 1000
nmax = 50

q = 0.995
nout = floor((1-q)*nsamples)

cf = [[0 for i in range(0, nout)] for n in range(1, nmax+1)]
mcl = [[1 for i in range(0, nout)] for n in range(1, nmax+1)]

for n in range(1, nmax+1):
    print("Coincidence factor for {}".format(n))
    for i in range(0, nsamples):
        subset = df.sample(n=n, axis=1)
        cf_sample = subset.sum(axis=1).max()/sum(maxdf[maxdf.index.isin(subset.columns)])
        mcl_sample = subset.sum(axis=1).min()/sum(maxdf[maxdf.index.isin(subset.columns)])
        if cf_sample > cf[n-1][0]:
            cf[n-1].append(cf_sample)
            cf[n-1].sort()
            cf[n-1] = cf[n-1][1:]
        if mcl_sample < mcl[n-1][-1]:
            mcl[n-1].append(mcl_sample)
            mcl[n-1].sort()
            mcl[n-1] = mcl[n-1][0:-1]

print(cf)

cf_q = [cf_i[0] for cf_i in cf]
mcl_q = [mcl_i[-1] for mcl_i in mcl]

cf_fit, pcov = curve_fit(decay, range(2, nmax+1), cf_q[1:])
mcl_fit, pcov = curve_fit(decay, range(2, nmax+1), mcl_q[1:])

fig1, ax1 = plt.subplots()
ax1.plot(cf_q, '.')
ax1.plot(decay(range(1, n+1), *cf_fit))

fig2, ax2 = plt.subplots()
ax2.plot(mcl_q, ".")
ax2.plot(decay(range(1, n+1), *mcl_fit))

print("CF fit:")
print(cf_fit)

print("MCL fit")
print(mcl_fit)

plt.show()