
# Chapter 1: Introduction to Machine Learning

## 1.1 What is Machine Learning?

### 1.1.1 Definition and Scope of Machine Learning
- **Definition**: Machine learning (ML) is a subset of artificial intelligence (AI) focused on building systems that learn from data. These systems can improve their performance over time without being explicitly programmed.

- **Scope**:
  - **Patterns and Inferences**: Algorithms are designed to identify patterns in data, make predictions, and inferences.
  - **Automation of Analytical Model Building**: Involves the use of statistical techniques to enable computers to learn from historical data.

### 1.1.2 Differences between AI, Machine Learning, and Deep Learning

- **Artificial Intelligence (AI)**:
  - Simulates human intelligence processes.
  - Encompasses machine learning, expert systems, logic, and more.

- **Machine Learning (ML)**:
  - Algorithms learn from and make predictions based on data.
  - Types include supervised learning, unsupervised learning, and reinforcement learning.

- **Deep Learning (DL)**:
  - A subset of ML with neural networks having many layers (deep networks).
  - Particularly useful for image and speech recognition tasks.

## 1.2 Types of Machine Learning

### 1.2.1 Supervised Learning
- **Definition**: Training a model on labeled data to predict outcomes for new, unseen data.
  
#### 1.2.1.1 Classification
- **Mathematical Framework**:
  - **Logistic Regression**:
    - The logistic function (sigmoid): \(\sigma(z) = \frac{1}{1 + e^{-z}}\)
    - Probability that \( y = 1 \):
      \[
      P(y=1|x) = \sigma(W^Tx + b) = \frac{1}{1 + e^{-(W^Tx + b)}}
      \]
    - Cost Function (Cross-Entropy Loss):
      \[
      J(\theta) = -\frac{1}{m}\sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
      \]
  - **Decision Trees**:
    - Recursive binary splitting.
    - Gini Index for node impurity:
      \[
      G = \sum_{i=1}^{n}\sum_{j=1}^{n} p_i(1 - p_j)
      \]
  - **Support Vector Machines (SVM)**:
    - Optimization objective:
      \[
      \min_{\alpha} \left( \frac{1}{2} \sum_{i}\sum_{j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \sum_{i} \alpha_i \right)
      \]
    - Subject to:
      \[
      \sum_{i} \alpha_i y_i = 0 \quad and \quad 0 \leq \alpha_i \leq C
      \]

#### 1.2.1.2 Regression
- **Mathematical Framework**:
  - **Linear Regression**:
    - Model:
      \[
      y = W^Tx + b
      \]
    - Cost Function (Mean Squared Error):
      \[
      J(W, b) = \frac{1}{m} \sum_{i=1}^{m} (W^Tx^{(i)} + b - y^{(i)})^2
      \]
  - **Polynomial Regression**:
    - Model:
      \[
      y = \sum_{i=0}^{n} \theta_i x^i
      \]
    - Using the same loss function as linear regression.

### 1.2.2 Unsupervised Learning
- **Definition**: Learning from unlabeled data to find hidden patterns.

#### 1.2.2.1 Clustering
- **Mathematical Framework**:
  - **k-Means Clustering**:
    - Objectives to minimize intra-cluster variance:
      \[
      J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
      \]
    - Steps:
      1. Initialize \(k\) cluster centroids.
      2. Assign each point to the nearest centroid.
      3. Update centroids by averaging assigned points.
  - **Hierarchical Clustering**:
    - Agglomerative method starts with each point as its own cluster.
    - Merge clusters iteratively based on closeness until one cluster remains.
    - **Distance Metrics**:
      - Single Linkage (Minimum distance between points in clusters).
      - Complete Linkage (Maximum distance between points in clusters).
      - Average Linkage (Average distance between points).

#### 1.2.2.2 Dimensionality Reduction
- **Mathematical Framework**:
  - **Principal Component Analysis (PCA)**:
    - Steps:
      1. Standardize the data.
      2. Compute the covariance matrix.
      3. Compute eigenvalues and eigenvectors of the covariance matrix.
      4. Project the data onto the top \(k\) eigenvectors.
    - Maximizes variance in projected space:
      \[
      \text{maximize} \quad \| Wx \|^2 \quad \text{subject to} \quad W^TW = I
      \]
  - **t-Distributed Stochastic Neighbor Embedding (t-SNE)**:
    - Converts high-dimensional Euclidean distances into conditional probabilities.
    - Minimizes Kullback-Leibler divergence between joint probabilities of the low-dimensional and high-dimensional data.
    - Cost function involving distribution similarities:
      \[
      KL(P \| Q) = \sum_i \sum_j p_{ij} \log \left( \frac{p_{ij}}{q_{ij}} \right)
      \]
    - Misfit minimization to better-preserve the input neighborhood crowding:
      \[
      q_{ij} = \frac{\left( 1 + \| y_i - y_j \|^2 \right)^{-1}}{\sum_{k \neq l} \left( 1 + \| y_k - y_l \|^2 \right)^{-1}}
      \]

### 1.2.3 Reinforcement Learning
- **Mathematical Framework**:
  - **Markov Decision Process (MDP)**:
    - Defined by states \(S\), actions \(A\), transition probabilities \(P\), and reward function \(R\).
  - **Objective**:
    - Maximize the cumulative reward.
    - **Value Function** (Expectation of cumulative reward starting from state \(s\)):
      \[
      V(s) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) | s_0 = s \right]
      \]
    - **Bellman Equation**:
      \[
      V(s) = \max_{a} \left[ R(s, a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]
      \]
  - **Q-Learning**:
    - Q-value for state-action pair:
      \[
      Q(s, a) = \mathbb{E} \left[ R_s^a + \gamma \max_{a'} Q(s', a') \right]
      \]
    - Update rule:
      \[
      Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right)
      \]

## 1.3 Real-world Applications of Machine Learning

### 1.3.1 Healthcare
- **Applications**:
  - **Disease Diagnosis and Prediction**:
    - Using statistical models and deep learning to analyze patient data.
    - Examples: Predicting diabetes using logistic regression.
  - **Personalized Treatment Plans**:
    - Recommender systems based on patient history.
  - **Medical Imaging**:
    - CNNs for classifying medical images (e.g., MRI, CT scans).

### 1.3.2 Finance
- **Applications**:
  - **Fraud Detection**:
    - Anomaly detection algorithms highlight abnormal transaction patterns.
  - **Algorithmic Trading**:
    - Time series analysis for predicting stock prices.
  - **Credit Scoring**:
    - Classification models to determine credit risk based on history.

### 1.3.3 Marketing
- **Applications**:
  - **Customer Segmentation**:
    - Clustering techniques to identify similar customer groups.
  - **Personalized Marketing**:
    - Predictive modeling and recommendation engines to tailor marketing messages.
  - **Sentiment Analysis**:
    - NLP techniques to analyze customer feedback.

### 1.3.4 Autonomous Vehicles
- **Applications**:
  - **Object Detection and Recognition**:
    - Convolutional neural networks (CNNs) are used to identify objects like pedestrians and vehicles.
  - **Path Planning and Navigation**:
    - Reinforcement learning algorithms to plan optimal routes and navigation paths.
  - **Sensor Fusion**:
    - Combining data from multiple sensors (LiDAR, cameras, radar) for a comprehensive understanding of the environment.
```

This in-depth content includes the mathematical foundations and algorithms for various types of machine learning, providing a solid theoretical foundation accompanied by real-world applications.