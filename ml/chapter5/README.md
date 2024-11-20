
# Chapter 5: Unsupervised Learning
---

## 5.1 Clustering

### 5.1.1 k-Means Clustering
- **Description**: k-Means clustering partitions data into k clusters where each data point belongs to the cluster with the nearest mean.
- **Key Concepts**:
  - Centroid initialization
  - Cluster assignment and centroid update
  - Elbow method for choosing k
- **Example**:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.cluster import KMeans
  
  # Sample data
  X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
  
  # Creating and training the model
  kmeans = KMeans(n_clusters=2, random_state=0)
  kmeans.fit(X)
  
  # Predictions and plotting
  y_kmeans = kmeans.predict(X)
  plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
  plt.show()
  ```

### 5.1.2 Hierarchical Clustering
- **Description**: Hierarchical clustering builds clusters by recursively merging or splitting existing clusters.
- **Key Concepts**:
  - Agglomerative (bottom-up) and Divisive (top-down) approaches
  - Dendrograms to visualize cluster formation
  - Linkage criteria: Single, Complete, Average
- **Example**:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.cluster.hierarchy import dendrogram, linkage
  
  # Sample data
  X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [8, 8]])
  
  # Perform hierarchical clustering
  Z = linkage(X, 'ward')
  
  # Plot the dendrogram
  plt.figure(figsize=(10, 7))
  dendrogram(Z)
  plt.show()
  ```

### 5.1.3 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **Description**: DBSCAN clusters data based on density, grouping closely packed points, and marking points in low-density regions as outliers.
- **Key Concepts**:
  - Parameters: eps (maximum distance between two samples), min_samples (minimum number of samples in a neighborhood for a point to be considered a core point)
  - Identifies core, border, and noise points
- **Example**:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.cluster import DBSCAN
  
  # Sample data
  X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
  
  # Creating and training the model
  dbscan = DBSCAN(eps=3, min_samples=2)
  dbscan.fit(X)
  
  # Plot the results
  labels = dbscan.labels_
  unique_labels = set(labels)
  for label in unique_labels:
    label_mask = (labels == label)
    plt.scatter(X[label_mask, 0], X[label_mask, 1], label=f'Cluster {label}')
  plt.legend()
  plt.show()
  ```

## 5.2 Dimensionality Reduction

### 5.2.1 Principal Component Analysis (PCA)
- **Description**: PCA reduces the dimensionality of data while retaining as much variance as possible.
- **Key Concepts**:
  - Eigenvalues and eigenvectors
  - Explained variance ratio
- **Example**:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.decomposition import PCA
  
  # Sample data
  np.random.seed(0)
  X = np.random.randn(100, 5)
  
  # Perform PCA
  pca = PCA(n_components=2)
  principalComponents = pca.fit_transform(X)
  
  # Plotting the reduced dimension data
  plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.show()
  
  # Explained variance ratio
  print(pca.explained_variance_ratio_)
  ```

### 5.2.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)
- **Description**: t-SNE is a non-linear dimensionality reduction technique primarily used for data visualization in two or three dimensions.
- **Key Concepts**:
  - Perplexity
  - Optimization using gradient descent
- **Example**:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.manifold import TSNE
  from sklearn.datasets import load_digits
  
  # Load sample data
  digits = load_digits()
  X = digits.data
  y = digits.target
  
  # Perform t-SNE
  tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
  X_embedded = tsne.fit_transform(X)
  
  # Plotting the results
  plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
  plt.colorbar()
  plt.show()
  ```
