
# Chapter 4: Supervised Learning

## 4.1 Regression

### 4.1.1 Linear Regression
- **Description**: Linear Regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables.
- **Key Concepts**:
  - Equation: \( y = \beta_0 + \beta_1x \)
  - Ordinary Least Squares (OLS) method
- **Example**:
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.linear_model import LinearRegression
  
  # Sample data
  X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
  y = np.array([1, 3, 2, 3, 5])
  
  # Creating and training the model
  model = LinearRegression()
  model.fit(X, y)
  
  # Making predictions
  y_pred = model.predict(X)
  print(y_pred)
  ```

### 4.1.2 Multiple Linear Regression
- **Description**: Multiple Linear Regression models the relationship between two or more independent variables and a dependent variable by fitting a linear equation.
- **Key Concepts**:
  - Equation: \( y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n \)
  - Feature importance using coefficients
- **Example**:
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.linear_model import LinearRegression
  
  # Sample data
  X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
  y = np.array([1, 2, 3, 4])
  
  # Creating and training the model
  model = LinearRegression()
  model.fit(X, y)
  
  # Making predictions
  y_pred = model.predict(X)
  print(y_pred)
  ```

### 4.1.3 Polynomial Regression
- **Description**: Polynomial Regression is a type of regression that models the relationship between the independent variable(s) and the dependent variable as an nth degree polynomial.
- **Key Concepts**:
  - Equation: \( y = \beta_0 + \beta_1x + \beta_2x^2 + \ldots + \beta_nx^n \)
  - Overfitting risk with high-degree polynomials
- **Example**:
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.preprocessing import PolynomialFeatures
  from sklearn.linear_model import LinearRegression
  
  # Sample data
  X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
  y = np.array([1, 4, 9, 16, 25])
  
  # Transforming the features to polynomial features
  poly = PolynomialFeatures(degree=2)
  X_poly = poly.fit_transform(X)
  
  # Creating and training the model
  model = LinearRegression()
  model.fit(X_poly, y)
  
  # Making predictions
  y_pred = model.predict(X_poly)
  print(y_pred)
  ```

### 4.1.4 Metrics: MSE, RMSE, R²
- **Description**: Evaluate the performance of regression models using different metrics.
- **Key Metrics**:
  - Mean Squared Error (MSE): 
    ```python
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    ```
  - Root Mean Squared Error (RMSE): 
    ```python
    rmse = np.sqrt(mse)
    ```
  - R-squared (R²): 
    ```python
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ```

## 4.2 Classification

### 4.2.1 Logistic Regression
- **Description**: Logistic Regression is used for binary classification problems where the response variable is categorical.
- **Key Concepts**:
  - Sigmoid function: \( \sigma(z) = \frac{1}{1 + e^{-z}} \)
  - Binary cross-entropy loss
- **Example**:
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.linear_model import LogisticRegression
  
  # Sample data
  X = np.array([[1,2], [3,4], [5,6], [7,8]])
  y = np.array([0, 0, 1, 1])
  
  # Creating and training the model
  model = LogisticRegression()
  model.fit(X, y)
  
  # Making predictions
  y_pred = model.predict(X)
  print(y_pred)
  ```

### 4.2.2 k-Nearest Neighbors (k-NN)
- **Description**: k-NN is a non-parametric method used for classification and regression by measuring the distance between points.
- **Key Concepts**:
  - Distance metrics: Euclidean, Manhattan
  - Choosing k value
- **Example**:
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.neighbors import KNeighborsClassifier
  
  # Sample data
  X = np.array([[1,2], [3,4], [5,6], [7,8]])
  y = np.array([0, 0, 1, 1])
  
  # Creating and training the model
  model = KNeighborsClassifier(n_neighbors=3)
  model.fit(X, y)
  
  # Making predictions
  y_pred = model.predict(X)
  print(y_pred)
  ```

### 4.2.3 Support Vector Machine (SVM)
- **Description**: SVM is a supervised learning model used for classification by finding the hyperplane that best separates the classes.
- **Key Concepts**:
  - Kernel trick: linear, polynomial, RBF
  - Support vectors
- **Example**:
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.svm import SVC
  
  # Sample data
  X = np.array([[1,2], [3,4], [5,6], [7,8]])
  y = np.array([0, 0, 1, 1])
  
  # Creating and training the model
  model = SVC(kernel='linear')
  model.fit(X, y)
  
  # Making predictions
  y_pred = model.predict(X)
  print(y_pred)
  ```

### 4.2.4 Decision Trees
- **Description**: Decision Trees classify data by splitting the dataset into branches based on feature values.
- **Key Concepts**:
  - Gini impurity
  - Information gain
- **Example**:
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.tree import DecisionTreeClassifier
  
  # Sample data
  X = np.array([[1,2], [3,4], [5,6], [7,8]])
  y = np.array([0, 0, 1, 1])
  
  # Creating and training the model
  model = DecisionTreeClassifier()
  model.fit(X, y)
  
  # Making predictions
  y_pred = model.predict(X)
  print(y_pred)
  ```

### 4.2.5 Random Forests
- **Description**: Random Forests are an ensemble method that combines multiple decision trees to improve classification performance.
- **Key Concepts**:
  - Bootstrap aggregation (bagging)
  - Feature randomness
- **Example**:
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.ensemble import RandomForestClassifier
  
  # Sample data
  X = np.array([[1,2], [3,4], [5,6], [7,8]])
  y = np.array([0, 0, 1, 1])
  
  # Creating and training the model
  model = RandomForestClassifier(n_estimators=100)
  model.fit(X, y)
  
  # Making predictions
  y_pred = model.predict(X)
  print(y_pred)
  ```

### 4.2.6 Gradient Boosting (XGBoost, LightGBM)
- **Description**: Gradient Boosting models build successive models that progressively correct errors made by previous models.
- **Key Concepts**:
  - Boosting trees
  - Learning rate
- **Example with XGBoost**:
  ```python
  import numpy as np
  import pandas as pd
  import xgboost as xgb
  
  # Sample data
  X = np.array([[1,2], [3,4], [5,6], [7,8]])
  y = np.array([0, 0, 1, 1])
  
  # Creating and training the model
  model = xgb.XGBClassifier()
  model.fit(X, y)
  
  # Making predictions
  y_pred = model.predict(X)
  print(y_pred)
  ```
- **Example with LightGBM**:
  ```python
  import numpy as np
  import pandas as pd
  import lightgbm as lgb
  
  # Sample data
  X = np.array([[1,2], [3,4], [5,6], [7,8]])
  y = np.array([0, 0, 1, 1])
  
  # Creating and training the model
  model = lgb.LGBMClassifier()
  model.fit(X, y)
  
  # Making predictions
  y_pred = model.predict(X)
  print(y_pred)
  ```

### 4.2.7 Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Description**: Evaluating the performance of classification models using various metrics.
- **Key Metrics**:
  - Accuracy: 
    ```python
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    ```
  - Precision: 
    ```python
    from sklearn.metrics import precision_score
    precision = precision_score(y_true, y_pred)
    ```
  - Recall: 
    ```python
    from sklearn.metrics import recall_score
    recall = recall_score(y_true, y_pred)
    ```
  - F1-Score: 
    ```python
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred)
    ```
  - ROC-AUC:
    ```python
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    ```
- **Example**:
  ```python
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
  
  # Sample data
  y_true = np.array([0, 0, 1, 1])
  y_pred = np.array([0, 1, 1, 1])
  y_pred_proba = np.array([0.1, 0.4, 0.35, 0.8])
  
  # Calculating metrics
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  roc_auc = roc_auc_score(y_true, y_pred_proba)
  
  print(f'Accuracy: {accuracy}')
  print(f'Precision: {precision}')
  print(f'Recall: {recall}')
  print(f'F1-Score: {f1}')
  print(f'ROC-AUC: {roc_auc}')
  ```
