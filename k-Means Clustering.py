#Uploading Dataset
import pandas as pd
df=pd.read_csv('Dataset (Grid_Stability).csv')

#KMeans Clustering
from sklearn.cluster import KMeans
df.dropna()
df.fillna(0)
km=KMeans(n_clusters=4)
yp=km.fit_predict(df)

df['Pred']=yp

