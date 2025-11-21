import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

df = pd.read_csv("INST414/Warehouse_and_Retail_Sales.csv")
numeric_df = df.select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

inertia = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

clean_df = numeric_df.copy()
clean_df['Cluster'] = clusters

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
clean_df['PCA1'] = pca_data[:, 0]
clean_df['PCA2'] = pca_data[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=clean_df, palette='tab10')
plt.title('Cluster Visualization (PCA Projection)')
plt.show()

cluster_summary = clean_df.groupby('Cluster').mean(numeric_only=True)
print("\nCluster Summary:")
print(cluster_summary)


