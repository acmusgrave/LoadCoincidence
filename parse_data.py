import pandas as pd
import os

PATH = "C:\\Users\\Andrew\\Documents\MASc\\LoadCoincidence\\load_data\\"

# take only total kw from each commercial load and combine to one file
commercial_list = os.listdir(PATH+"commercial")

df = pd.read_parquet(PATH+"commercial\\"+commercial_list[0])
df = df[['Time', 'total_site_electricity_kw']]
df = df.set_index('Time')
identifier=commercial_list[0].split(".")[0]
df = df.rename(columns={'total_site_electricity_kw': identifier})

for file in commercial_list[1:]:
   other = pd.read_parquet(PATH+"commercial\\"+file)
   other = other[['Time', 'total_site_electricity_kw']]
   other=other.set_index('Time')
   identifier=file.split(".")[0]
   other = other.rename(columns={'total_site_electricity_kw': identifier})
   df=df.join(other, how='inner')
    
df.to_parquet(PATH+"commercial_loads.parquet")

# take only total kw from each residential load and combine to one file
residential_list = os.listdir(PATH+"residential")

df = pd.read_parquet(PATH+"residential\\"+residential_list[0])
df = df[['Time', 'total_site_electricity_kw']]
df = df.set_index('Time')
identifier=residential_list[0].split(".")[0]
df = df.rename(columns={'total_site_electricity_kw': identifier})

for file in residential_list[1:]:
   other = pd.read_parquet(PATH+"residential\\"+file)
   other = other[['Time', 'total_site_electricity_kw']]
   other=other.set_index('Time')
   identifier=file.split(".")[0]
   other = other.rename(columns={'total_site_electricity_kw': identifier})
   df=df.join(other, how='inner')
    
df.to_parquet(PATH+"residential_loads.parquet")