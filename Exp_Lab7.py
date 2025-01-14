from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load Digits dataset
data = load_digits()
df = pd.DataFrame(data.data, columns=data.feature_names)  # Extract the features (64 pixels)

# Preprocess data: Scaling for better clustering performance
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)  # 10 clusters for 10 digits
kmeans_labels = kmeans.fit_predict(data_scaled)

# Apply EM (Gaussian Mixture Model) clustering
gmm = GaussianMixture(n_components=10, random_state=42)  # 10 components for 10 digits
gmm_labels = gmm.fit_predict(data_scaled)

# Compare clustering results using silhouette score
kmeans_silhouette = silhouette_score(data_scaled, kmeans_labels)
gmm_silhouette = silhouette_score(data_scaled, gmm_labels)

# Visualize the clusters: We'll only visualize the first two principal components
# to better understand the clustering in a 2D space
from sklearn.decomposition import PCA

# Apply PCA for dimensionality reduction to 2D for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Visualize the clusters
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.subplot(1, 2, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=gmm_labels, cmap='viridis')
plt.title('EM (Gaussian Mixture) Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()

# Print the comparison results
print(f'K-Means Silhouette Score: {kmeans_silhouette:.4f}')
print(f'EM (Gaussian Mixture) Silhouette Score: {gmm_silhouette:.4f}')

