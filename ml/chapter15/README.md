
# Chapter 15: Real-world Projects and Case Studies
---

## Project 1: Predicting House Prices
- **Description**: In this project, we will predict house prices using historical data. This project will involve data preprocessing, exploration, building regression models, and evaluating and fine-tuning these models.
- **Significance**:
  - **Real Estate Market Insights**: Helps understand factors influencing house prices, valuable for real estate agencies and buyers.
  - **Skill Development**: Enhances skills in regression analysis and model evaluation.
- **Usage**:
  - Predicting property prices for potential buyers and sellers.
  - Assisting financial institutions in property valuation for mortgage purposes.
- **Example**:
  ```markdown
  - Using a dataset like the Boston Housing Dataset to predict house prices.
  ```

### 15.1.1 Data Preprocessing and Exploration
- **Steps**:
  - Loading the dataset.
  - Handling missing values.
  - Feature engineering and selection.
  - Exploratory Data Analysis (EDA) to understand data distributions.
- **Example**:
  ```python
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  # Load dataset
  df = pd.read_csv('boston_housing.csv')

  # Handle missing values
  df = df.dropna()

  # Feature engineering
  df['price_per_room'] = df['MEDV'] / df['RM']

  # Exploratory Data Analysis
  sns.pairplot(df)
  plt.show()
  ```

### 15.1.2 Building Regression Models
- **Steps**:
  - Splitting the data into training and testing sets.
  - Building multiple regression models (e.g., Linear Regression, Decision Trees, Random Forests).
- **Example**:
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LinearRegression
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.ensemble import RandomForestRegressor

  X = df.drop('MEDV', axis=1)
  y = df['MEDV']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Build Linear Regression model
  lr_model = LinearRegression()
  lr_model.fit(X_train, y_train)

  # Build Decision Tree model
  dt_model = DecisionTreeRegressor()
  dt_model.fit(X_train, y_train)

  # Build Random Forest model
  rf_model = RandomForestRegressor()
  rf_model.fit(X_train, y_train)
  ```

### 15.1.3 Evaluating and Fine-tuning Models
- **Steps**:
  - Evaluating model performance using metrics like RMSE, MAE.
  - Fine-tuning models using GridSearchCV or RandomizedSearchCV.
- **Example**:
  ```python
  from sklearn.metrics import mean_squared_error, mean_absolute_error
  from sklearn.model_selection import GridSearchCV

  # Model evaluation
  y_pred_lr = lr_model.predict(X_test)
  y_pred_rf = rf_model.predict(X_test)

  print('Linear Regression RMSE:', mean_squared_error(y_test, y_pred_lr, squared=False))
  print('Random Forest RMSE:', mean_squared_error(y_test, y_pred_rf, squared=False))

  # Fine-tuning Random Forest model
  param_grid = {'n_estimators': [50, 100, 200]}
  grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='neg_mean_squared_error')
  grid_search.fit(X_train, y_train)
  best_model = grid_search.best_estimator_
  ```

## Project 2: Sentiment Analysis
- **Description**: This project involves analyzing text data to determine the sentiment (positive, negative, neutral) expressed in it. We will perform data collection, preprocessing, build, and evaluate a sentiment classifier.
- **Significance**:
  - **Customer Insights**: Helps businesses understand customer sentiment through reviews and feedback.
  - **Market Analysis**: Useful in gauging public opinion on products and services.
- **Usage**:
  - Sentiment analysis on product reviews, social media posts, and customer feedback.
- **Example**:
  ```markdown
  - Conducting sentiment analysis on a dataset like IMDb movie reviews.
  ```

### 15.2.1 Data Collection and Preprocessing
- **Steps**:
  - Collecting text data from sources like APIs or datasets.
  - Cleaning and preprocessing text data (tokenization, removing stop words, stemming/lemmatization).
- **Example**:
  ```python
  import pandas as pd
  import re
  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize
  from nltk.stem import PorterStemmer

  # Load dataset
  df = pd.read_csv('imdb_reviews.csv')

  # Text preprocessing
  stop_words = set(stopwords.words('english'))
  stemmer = PorterStemmer()

  def preprocess_text(text):
      text = re.sub(r'\W', ' ', text)
      text = text.lower()
      tokens = word_tokenize(text)
      tokens = [word for word in tokens if word not in stop_words]
      tokens = [stemmer.stem(word) for word in tokens]
      return ' '.join(tokens)

  df['cleaned_review'] = df['review'].apply(preprocess_text)
  ```

### 15.2.2 Building and Evaluating a Sentiment Classifier
- **Steps**:
  - Splitting data into training and testing sets.
  - Building a sentiment classifier using algorithms like Logistic Regression, Naive Bayes, or neural networks.
  - Evaluating model performance using metrics like accuracy, F1-score, confusion matrix.
- **Example**:
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

  X = df['cleaned_review']
  y = df['sentiment']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Vectorize text data
  vectorizer = TfidfVectorizer(max_features=5000)
  X_train_tfidf = vectorizer.fit_transform(X_train)
  X_test_tfidf = vectorizer.transform(X_test)

  # Build sentiment classifier
  model = LogisticRegression()
  model.fit(X_train_tfidf, y_train)

  # Evaluate model
  y_pred = model.predict(X_test_tfidf)
  print('Accuracy:', accuracy_score(y_test, y_pred))
  print('Classification Report:\n', classification_report(y_test, y_pred))
  print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
  ```

## Project 3: Image Classification with Deep Learning
- **Description**: Classifying images into predefined categories using deep learning, specifically Convolutional Neural Networks (CNNs). We will explore building models from scratch and using transfer learning with pre-trained models.
- **Significance**:
  - **Automation**: Automates complex visual tasks like object recognition.
  - **Advanced AI Techniques**: Demonstrates the application of advanced deep learning techniques.
- **Usage**:
  - Applications in medical imaging, autonomous vehicles, and image-based research areas.
- **Example**:
  ```markdown
  - Classifying images from the CIFAR-10 dataset.
  ```

### 15.3.1 Using CNNs to Classify Images
- **Steps**:
  - Loading and preprocessing image data.
  - Building and training a CNN model to classify images.
- **Example**:
  ```python
  import tensorflow as tf
  from tensorflow.keras.datasets import cifar10
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

  # Load CIFAR-10 dataset
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()

  # Normalize data
  X_train, X_test = X_train / 255.0, X_test / 255.0

  # Build CNN model
  model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
      MaxPooling2D((2, 2)),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D((2, 2)),
      Flatten(),
      Dense(64, activation='relu'),
      Dense(10, activation='softmax')
  ])

  # Compile and train model
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
  ```

### 15.3.2 Transfer Learning with Pre-trained Models
- **Steps**:
  - Loading a pre-trained model (e.g., VGG16, ResNet).
  - Fine-tuning the pre-trained model for image classification.
- **Example**:
  ```python
  from tensorflow.keras.applications import VGG16
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Dense, Flatten

  # Load pre-trained VGG16 model
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

  # Fine-tune the model
  x = base_model.output
  x = Flatten()(x)
  x = Dense(64, activation='relu')(x)
  predictions = Dense(10, activation='softmax')(x)

  model = Model(inputs=base_model.input, outputs=predictions)

  for layer in base_model.layers:
      layer.trainable = False

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
  ```

## Project 4: Deploying a Model to Production
- **Description**: This project involves creating a web service to deploy a machine learning model to production using Flask, Docker, and Kubernetes.
- **Significance**:
  - **Practical Implementation**: Covers practical aspects of deploying ML models, essential for real-world applications.
  - **Scalability**: Demonstrates how to make ML models accessible and scalable.
- **Usage**:
  - Deploying predictive models as APIs for consumption by applications.
- **Example**:
  ```markdown
  - Deploying a model predicting house prices as a REST API.
  ```

### 15.4.1 Creating a Web Service with Flask
- **Steps**:
  - Setting up a Flask web application.
  - Creating API endpoints for model predictions.
- **Example**:
  ```python
  from flask import Flask, request, jsonify
  import pickle

  app = Flask(__name__)

  # Load pre-trained model
  model = pickle.load(open('model.pkl', 'rb'))

  @app.route('/predict', methods=['POST'])
  def predict():
      data = request.get_json(force=True)
      prediction = model.predict([data['features']])
      return jsonify({'prediction': prediction.tolist()})

  if __name__ == '__main__':
      app.run(debug=True)
  ```

### 15.4.2 Deploying the Model Using Docker and Kubernetes
- **Steps**:
  - Creating a Docker image for the Flask application.
  - Deploying the Docker container to a Kubernetes cluster.
- **Example**:
  ```dockerfile
  # Dockerfile
  FROM python:3.8-slim

  WORKDIR /app

  COPY requirements.txt requirements.txt
  RUN pip install -r requirements.txt

  COPY . .

  CMD ["python", "app.py"]
  ```

  ```yaml
  # Kubernetes Deployment
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: house-predictor-deployment
  spec:
    replicas: 2
    selector:
      matchLabels:
        app: house-predictor
    template:
      metadata:
        labels:
          app: house-predictor
      spec:
        containers:
        - name: house-predictor
          image: myregistry/house-predictor:latest
          ports:
          - containerPort: 5000
  ---
  apiVersion: v1
  kind: Service
  metadata:
    name: house-predictor-service
  spec:
    type: LoadBalancer
    selector:
      app: house-predictor
    ports:
      - protocol: TCP
        port: 80
        targetPort: 5000
  ```

## Capstone Project: End-to-End Machine Learning Pipeline
- **Description**: In this capstone project, we will build an end-to-end machine learning pipeline covering problem selection, data collection, model building, evaluation, deployment, monitoring, and continuous integration.
- **Significance**:
  - **Comprehensive Experience**: Provides hands-on experience with all stages of the machine learning lifecycle.
  - **Industry Relevance**: Prepares for real-world machine learning challenges.
- **Usage**:
  - Implementing a full-fledged machine learning solution for business problems.
- **Example**:
  ```markdown
  - Building an end-to-end pipeline for a retail sales prediction model.
  ```

### 15.5.1 Problem Selection and Data Collection
- **Steps**:
  - Defining the problem statement.
  - Collecting and preprocessing relevant data.
- **Example**:
  ```python
  # Defining the problem statement
  problem_statement = "Predict the future sales of a retail store."

  # Collecting data
  import pandas as pd
  df = pd.read_csv('retail_sales_data.csv')

  # Preprocessing data
  df = df.dropna()
  df['sales_per_day'] = df['total_sales'] / df['num_days_open']
  ```

### 15.5.2 Model Building, Evaluation, and Tuning
- **Steps**:
  - Splitting data into training and testing sets.
  - Building and training models.
  - Evaluating model performance and fine-tuning.
- **Example**:
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import mean_squared_error

  X = df.drop('total_sales', axis=1)
  y = df['total_sales']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  model = LinearRegression()
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
  ```

### 15.5.3 Deployment and Monitoring
- **Steps**:
  - Deploying the trained model as an API.
  - Setting up monitoring and logging for the deployed model.
- **Example**:
  ```python
  from flask import Flask, request, jsonify
  from prometheus_client import start_http_server, Summary

  app = Flask(__name__)
  model = pickle.load(open('sales_model.pkl', 'rb'))

  REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

  @app.route('/predict', methods=['POST'])
  @REQUEST_TIME.time()
  def predict():
      data = request.get_json(force=True)
      prediction = model.predict([data['features']])
      return jsonify({'prediction': prediction.tolist()})

  if __name__ == '__main__':
      start_http_server(8000)
      app.run(debug=True)
  ```

### 15.5.4 Continuous Integration and Retraining Setup
- **Steps**:
  - Setting up CI/CD pipelines for continuous integration and deployment.
  - Automating model retraining based on new data.
- **Example**:
  ```yaml
  # CI/CD Pipeline configuration using GitHub Actions
  name: CI/CD Pipeline

  on: [push]

  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest

  - name: Build and push Docker image
    run: |
      docker build -t myregistry/sales-predictor:latest .
      docker push myregistry/sales-predictor:latest

  - name: Deploy to Kubernetes
    run: |
      kubectl apply -f k8s/deployment.yaml
  ```

