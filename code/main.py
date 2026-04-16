import pandas as pd # Pandas
import numpy as np # Numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.decomposition import PCA


#%%
df = pd.read_csv(r"C:\Users\isabe\OneDrive\Documents\BME2315\Grosso-Orlando-Module-4-Cancer\data\GSE62944_metadata_percent_nonNA_by_cancer_type.csv")
df = pd.DataFrame(df)
df = df.drop(columns = ["Unnamed: 0"])
df['cluster'] = df['cluster'].astype(int)
g = sns.PairGrid(df, hue = "cluster", palette = "Set2")
g.map(sns.scatterplot)