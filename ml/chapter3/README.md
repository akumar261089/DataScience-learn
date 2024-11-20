
# Chapter 3: Data Preprocessing

## 3.1 Importing and Understanding Data

### 3.1.1 Loading Data with Pandas
- **Description**: Learn the techniques to load data into Python using Pandas.
- **Key Functions**:
  - `pd.read_csv()`: Read data from a CSV file.
    ```python
    import pandas as pd
    df = pd.read_csv('data.csv')
    ```
  - `pd.read_excel()`: Read data from an Excel file.
    ```python
    df = pd.read_excel('data.xlsx')
    ```
  - `pd.read_sql()`: Read data from a SQL database.
    ```python
    import sqlite3
    conn = sqlite3.connect('database.db')
    df = pd.read_sql('SELECT * FROM table_name', conn)
    ```
- **Examples**:
  ```python
  # Example of reading CSV file
  df = pd.read_csv('data.csv')
  print(df.head())
  ```

### 3.1.2 Exploring Datasets
- **Description**: Understand how to explore and summarize datasets.
- **Key Functions**:
  - `df.head()`: View the first few rows of the dataset.
  - `df.info()`: Summary of the dataframe, including the data types.
  - `df.describe()`: Statistical summary of numerical columns.
  - `df.shape`: Dimensions of the dataframe.
  - `df.columns`: Column names.
- **Examples**:
  ```python
  # Viewing the first few rows
  print(df.head())
  
  # Summary of the dataframe
  print(df.info())
  
  # Statistical summary
  print(df.describe())
  ```

### 3.1.3 Handling Missing Data
- **Description**: Methods to handle missing values in the dataset.
- **Key Functions**:
  - `df.isna().sum()`: Count the number of missing values in each column.
  - `df.dropna()`: Remove rows with missing values.
  - `df.fillna()`: Fill missing values with specified values.
- **Examples**:
  ```python
  # Counting missing values
  print(df.isna().sum())
  
  # Removing rows with missing values
  df_clean = df.dropna()
  
  # Filling missing values with median
  df_filled = df.fillna(df.median())
  ```

## 3.2 Data Cleaning

### 3.2.1 Removing Duplicates
- **Description**: Identifying and removing duplicate rows.
- **Key Functions**:
  - `df.duplicated()`: Identify duplicate rows.
  - `df.drop_duplicates()`: Remove duplicate rows.
- **Examples**:
  ```python
  # Identifying duplicate rows
  duplicates = df.duplicated()
  print(duplicates)
  
  # Removing duplicate rows
  df_no_duplicates = df.drop_duplicates()
  ```

### 3.2.2 Handling Outliers
- **Description**: Detect and handle outliers in numerical data.
- **Key Techniques**:
  - Z-score for identifying outliers.
    ```python
    from scipy import stats
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    ```
  - IQR (Interquartile Range) method.
    ```python
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_outliers_removed = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    ```
- **Examples**:
  ```python
  # Using Z-score to remove outliers
  from scipy import stats
  import numpy as np
  df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
  
  # Using IQR method to remove outliers
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3 - Q1
  df_outliers_removed = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
  ```

### 3.2.3 Scaling and Normalization
- **Description**: Standardize or normalize features for better performance in machine learning algorithms.
- **Key Techniques**:
  - Standardization (mean = 0, standard deviation = 1).
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    ```
  - Min-Max Scaling (values between 0 and 1).
    ```python
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    ```
- **Examples**:
  ```python
  # Standardizing the data
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df)
  
  # Normalizing the data
  from sklearn.preprocessing import MinMaxScaler
  normalizer = MinMaxScaler()
  df_normalized = normalizer.fit_transform(df)
  ```

## 3.3 Feature Engineering

### 3.3.1 Encoding Categorical Variables
- **Description**: Converting categorical data into numerical format.
- **Key Techniques**:
  - One-Hot Encoding.
    ```python
    df_encoded = pd.get_dummies(df, columns=['categorical_column'])
    ```
  - Label Encoding.
    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['categorical_column'] = le.fit_transform(df['categorical_column'])
    ```
- **Examples**:
  ```python
  # One-Hot Encoding
  df_encoded = pd.get_dummies(df, columns=['category'])
  
  # Label Encoding
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  df['category'] = le.fit_transform(df['category'])
  ```

### 3.3.2 Creating New Features
- **Description**: Deriving new features from existing ones to better represent the underlying data patterns.
- **Techniques**:
  - Date and Time Features: Extracting year, month, day, week, hour from datetime.
    ```python
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    ```
  - Mathematical Transformations: Creating interaction terms, polynomial features.
    ```python
    df['feature_sum'] = df['feature1'] + df['feature2']
    df['feature_product'] = df['feature1'] * df['feature2']
    ```
- **Examples**:
  ```python
  # Extracting date features
  df['date'] = pd.to_datetime(df['date'])
  df['year'] = df['date'].dt.year
  df['month'] = df['date'].dt.month
  
  # Feature interactions
  df['feature_sum'] = df['feature1'] + df['feature2']
  df['feature_product'] = df['feature1'] * df['feature2']
  ```

### 3.3.3 Feature Selection
- **Description**: Selecting the most important features to improve model performance.
- **Techniques**:
  - Filter Methods (e.g., correlation, chi-square).
    ```python
    from sklearn.feature_selection import SelectKBest, f_classif
    X_new = SelectKBest(f_classif, k=5).fit_transform(X, y)
    ```
  - Wrapper Methods (e.g., forward, backward selection).
    ```python
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    rfe = RFE(model, 5)
    X_rfe = rfe.fit_transform(X, y)
    ```
  - Embedded Methods (e.g., LASSO, tree-based methods).
    ```python
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=0.01)
    model.fit(X, y)
    importance = model.coef_
    ```
- **Examples**:
  ```python
  # Filter Method: Selecting top 5 features by ANOVA F-value
  from sklearn.feature_selection import SelectKBest, f_classif
  X_new = SelectKBest(f_classif, k=5).fit_transform(X, y)
  
  # Wrapper Method: Recursive Feature Elimination (RFE)
  from sklearn.feature_selection import RFE
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression()
  rfe = RFE(model, 5)
  X_rfe = rfe.fit_transform(X, y)
  
  # Embedded Method: LASSO
  from sklearn.linear_model import Lasso
  model = Lasso(alpha=0.01)
  model.fit(X, y)
  importance = model.coef_
  print(importance)
  ```
