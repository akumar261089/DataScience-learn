# Chapter 6: Model Evaluation and Selection
---

## 6.1 Cross-validation
- **Description**: Cross-validation is a technique for assessing the performance of a model by splitting the data into training and validation sets multiple times and averaging the results.
- **Significance**:
  - Provides a more accurate estimate of model performance compared to a single train-test split.
  - Helps in detecting overfitting.
- **Usage**: Common methods include k-fold cross-validation, stratified k-fold, leave-one-out.
- **Example**:
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import cross_val_score
  from sklearn.linear_model import LogisticRegression
  
  # Sample data
  X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
  y = np.array([0, 1, 0, 1, 0])
  
  # Creating the model
  model = LogisticRegression()
  
  # Perform 5-fold cross-validation
  scores = cross_val_score(model, X, y, cv=5)
  print(f'Cross-validation scores: {scores}')
  print(f'Mean score: {scores.mean()}')
  ```

## 6.2 Bias-variance tradeoff
- **Description**: The bias-variance tradeoff is the balance between the error due to bias (error from overly simplistic models) and the error due to variance (error from overly complex models).
- **Significance**:
  - Understanding this tradeoff helps to choose models that are neither too simple (high bias) nor too complex (high variance).
- **Usage**: Analyze learning curves to understand the bias-variance tradeoff and select appropriate models.
- **Example**:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.model_selection import learning_curve
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.datasets import load_iris
  
  # Load sample data
  iris = load_iris()
  X, y = iris.data, iris.target
  
  # Create the model
  model = DecisionTreeClassifier()
  
  # Compute learning curves
  train_sizes, train_scores, validation_scores = learning_curve(model, X, y, cv=5)
  
  # Plot learning curves
  plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
  plt.plot(train_sizes, np.mean(validation_scores, axis=1), label='Validation score')
  plt.xlabel('Training set size')
  plt.ylabel('Score')
  plt.legend()
  plt.show()
  ```

## 6.3 Hyperparameter tuning
### 6.3.1 Grid Search
- **Description**: Grid Search is an exhaustive search over a specified parameter grid to find the best combination of hyperparameters.
- **Significance**:
  - Helps in finding the optimal hyperparameters that maximize model performance.
- **Usage**: Define a parameter grid and use `GridSearchCV` to perform the search.
- **Example**:
  ```python
  import numpy as np
  from sklearn.model_selection import GridSearchCV
  from sklearn.ensemble import RandomForestClassifier
  
  # Sample data
  X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
  y = np.array([0, 1, 0, 1, 0])
  
  # Model and parameter grid
  model = RandomForestClassifier()
  param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 2, 3, 4]}
  
  # Perform grid search
  grid_search = GridSearchCV(model, param_grid, cv=5)
  grid_search.fit(X, y)
  
  # Best parameters
  print(f'Best parameters: {grid_search.best_params_}')
  ```

### 6.3.2 Random Search
- **Description**: Random Search is a method of hyperparameter tuning that searches over a specified parameter grid by sampling from a distribution of hyperparameters.
- **Significance**:
  - More efficient than Grid Search for large parameter spaces.
  - Can find good hyperparameter combinations faster.
- **Usage**: Define a parameter distribution and use `RandomizedSearchCV` to perform the search.
- **Example**:
  ```python
  import numpy as np
  from sklearn.model_selection import RandomizedSearchCV
  from sklearn.ensemble import RandomForestClassifier
  from scipy.stats import randint
  
  # Sample data
  X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
  y = np.array([0, 1, 0, 1, 0])
  
  # Model and parameter distribution
  model = RandomForestClassifier()
  param_dist = {'n_estimators': randint(10, 200), 'max_depth': randint(1, 10)}
  
  # Perform random search
  random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
  random_search.fit(X, y)
  
  # Best parameters
  print(f'Best parameters: {random_search.best_params_}')
  ```

## 6.4 Model selection criteria
- **Description**: Evaluate and select the best model based on various criteria.
- **Significance**:
  - Helps in choosing the most suitable model that generalizes well to unseen data.
- **Usage**: Criteria include accuracy, precision, recall, F1-score, ROC-AUC, and computational efficiency.
- **Example**:
  ```python
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

  # Sample data
  y_true = [0, 1, 0, 1, 0]
  y_pred = [0, 1, 0, 0, 1]
  y_pred_proba = [0.1, 0.9, 0.2, 0.3, 0.7]
  
  # Calculate various metrics
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  roc_auc = roc_auc_score(y_true, y_pred_proba)
  
  # Print the metrics
  print(f'Accuracy: {accuracy}')
  print(f'Precision: {precision}')
  print(f'Recall: {recall}')
  print(f'F1-Score: {f1}')
  print(f'ROC-AUC: {roc_auc}')
  ```

