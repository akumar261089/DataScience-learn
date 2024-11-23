## [4. Text Classification](chapter4/README.md)

Text classification is a fundamental task in natural language processing (NLP) where the goal is to assign a label or category to a given text. This chapter explores various traditional machine learning algorithms and evaluation metrics used for text classification.

### [4.1 Traditional Machine Learning Algorithms](chapter4/section4.1/README.md)

Traditional machine learning algorithms have been widely used for text classification tasks. They often rely on vectorized text data, such as Bag of Words (BoW) or TF-IDF representations.

#### [Naive Bayes](chapter4/section4.1/naive_bayes.md)

Naive Bayes is a probabilistic classifier based on Bayes' Theorem, assuming independence between features.

**Theory:**
- Bayes' Theorem: 
  \[
  P(C|X) = \frac{P(X|C) P(C)}{P(X)}
  \]
- Naive Bayes assumes that the presence (or absence) of a particular feature of a class is unrelated to the presence (or absence) of any other feature.

**Coding Example using sklearn:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample data
data = ['I love this product', 'This is a terrible product', 'Absolutely wonderful', 'Do not buy this']
labels = ['positive', 'negative', 'positive', 'negative']

# Creating the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Training the model
model.fit(data, labels)

# Predicting
predicted_label = model.predict(['This product is great'])[0]
print(predicted_label)
# Output: 'positive'
```

**Uses:**
- Spam detection
- Sentiment analysis
- Document classification

**Significance:**
- Simple, fast, and effective for large datasets.
- Works well with high-dimensional data.

#### [Support Vector Machines (SVM)](chapter4/section4.1/svm.md)

Support Vector Machines (SVM) are supervised learning models that can be used for classification and regression tasks.

**Theory:**
- SVM finds the hyperplane that best separates the data points of different classes in a high-dimensional space.
- Maximizes the margin (distance) between the closest points of the classes (support vectors).

**Coding Example using sklearn:**
```python
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Sample data
data = ['I love this product', 'This is a terrible product', 'Absolutely wonderful', 'Do not buy this']
labels = ['positive', 'negative', 'positive', 'negative']

# Creating the model
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

# Training the model
model.fit(data, labels)

# Predicting
predicted_label = model.predict(['This product is terrible'])[0]
print(predicted_label)
# Output: 'negative'
```

**Uses:**
- Text categorization (e.g., news articles, emails).
- Image classification when extended to higher dimensions with kernels.

**Significance:**
- Powerful classifier that works well with clear margin of separation.
- Effective in high-dimensional spaces and with unstructured data like text.

#### [Decision Trees](chapter4/section4.1/decision_trees.md)

Decision Trees are non-parametric supervised learning methods used for classification and regression.

**Theory:**
- Decision tree builds classification models based on decisions made by splitting the dataset into subsets based on the most significant feature that provides maximum information gain.

**Coding Example using sklearn:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline

# Sample data
data = ['I love this product', 'This is a terrible product', 'Absolutely wonderful', 'Do not buy this']
labels = ['positive', 'negative', 'positive', 'negative']

# Creating the model
model = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())

# Training the model
model.fit(data, labels)

# Predicting
predicted_label = model.predict(['Absolutely terrible'])[0]
print(predicted_label)
# Output: 'negative'
```

**Uses:**
- Rule-based classification problems.
- Easily interpretable models for decision-making processes.

**Significance:**
- Simple to understand and interpret.
- Handles both numerical and categorical data.

#### [Ensemble Methods (Random Forest, Gradient Boosting)](chapter4/section4.1/ensemble_methods.md)

Ensemble methods combine multiple machine learning models to improve the overall performance of the system. Common ensemble methods include Random Forest and Gradient Boosting.

**Random Forest**

**Theory:**
- A Random Forest is an ensemble of decision trees.
- Combines multiple decision trees to reduce the risk of overfitting and improve generalization.

**Coding Example using sklearn:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# Sample data
data = ['I love this product', 'This is a terrible product', 'Absolutely wonderful', 'Do not buy this']
labels = ['positive', 'negative', 'positive', 'negative']

# Creating the model
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())

# Training the model
model.fit(data, labels)

# Predicting
predicted_label = model.predict(['Do not buy this product'])[0]
print(predicted_label)
# Output: 'negative'
```

**Gradient Boosting**

**Theory:**
- Gradient Boosting constructs additive models iteratively by optimizing arbitrary differentiable loss functions.
- It builds trees sequentially, each one correcting the errors of the previous one.

**Coding Example using sklearn:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline

# Sample data
data = ['I love this product', 'This is a terrible product', 'Absolutely wonderful', 'Do not buy this']
labels = ['positive', 'negative', 'positive', 'negative']

# Creating the model
model = make_pipeline(TfidfVectorizer(), GradientBoostingClassifier())

# Training the model
model.fit(data, labels)

# Predicting
predicted_label = model.predict(['I love this'])[0]
print(predicted_label)
# Output: 'positive'
```

**Uses:**
- Classification tasks with complex and noisy data.
- Kaggle competitions and real-world applications requiring high accuracy.

**Significance:**
- High accuracy and robustness to overfitting.
- Effective in handling a mix of different features
### [4.2 Evaluation Metrics](chapter4/section4.2/README.md)

Evaluating the performance of text classification models is crucial to understanding their effectiveness. Various metrics are used to measure different aspects of model performance.

#### [Accuracy](chapter4/section4.2/accuracy.md)

Accuracy is one of the most basic evaluation metrics for classification models. It measures the proportion of correctly classified instances among the total instances.

**Theory:**
- Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

**Coding Example using sklearn:**
```python
from sklearn.metrics import accuracy_score

# Sample true labels and predicted labels
true_labels = ['positive', 'negative', 'positive', 'negative']
predicted_labels = ['positive', 'negative', 'negative', 'negative']

# Calculating accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Accuracy: {accuracy}')
# Output: Accuracy: 0.75
```

**Uses:**
- Basic measure of model performance.
- Suitable when class distribution is balanced.

**Significance:**
- Provides an overall effectiveness of the model.
- May not be enough for imbalanced datasets.

#### [Precision, Recall, F1-score](chapter4/section4.2/precision_recall_f1.md)

Precision, recall, and F1-score provide a more detailed evaluation than accuracy, especially important for imbalanced datasets.

**Theory:**

- **Precision:** Measures how many of the instances predicted as positive are actually positive.
  - Precision = True Positives / (True Positives + False Positives)

- **Recall:** Measures how many of the actual positives are correctly predicted.
  - Recall = True Positives / (True Positives + False Negatives)

- **F1-score:** Harmonic mean of precision and recall, providing a single performance metric that balances both.
  - F1-score = 2 * (Precision * Recall) / (Precision + Recall)

**Coding Example using sklearn:**
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Sample true labels and predicted labels
true_labels = ['positive', 'negative', 'positive', 'negative']
predicted_labels = ['positive', 'negative', 'negative', 'negative']

# Calculating precision, recall, and F1-score
precision = precision_score(true_labels, predicted_labels, pos_label='positive')
recall = recall_score(true_labels, predicted_labels, pos_label='positive')
f1 = f1_score(true_labels, predicted_labels, pos_label='positive')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
# Output:
# Precision: 1.0
# Recall: 0.5
# F1-score: 0.6666666666666666
```

**Uses:**
- Detailed evaluation of model performance.
- Balances importance between precision and recall.

**Significance:**
- Useful for imbalanced datasets.
- Provides insights into false positives and false negatives.

#### [Confusion Matrix](chapter4/section4.2/confusion_matrix.md)

A confusion matrix provides a detailed breakdown of correct and incorrect classifications by showing actual versus predicted classes.

**Theory:**
- Contains four quadrants: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
- Helps visualize the performance of a classification model.

**Coding Example using sklearn:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample true labels and predicted labels
true_labels = ['positive', 'negative', 'positive', 'negative']
predicted_labels = ['positive', 'negative', 'negative', 'negative']

# Generating the confusion matrix
matrix = confusion_matrix(true_labels, predicted_labels, labels=['positive', 'negative'])

# Visualizing the confusion matrix
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred. Positive', 'Pred. Negative'], yticklabels=['True Positive', 'True Negative'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Output:
# Confusion matrix plot will be displayed with the following values in the cells:
# [[1, 1],
#  [0, 2]]
```

**Uses:**
- Provides a comprehensive view of a model's performance.
- Useful for understanding types of errors made by the classifier.

**Significance:**
- Helps in diagnosing problems like class imbalance.
- Offers insights beyond overall accuracy by showing the distribution of errors.

### Summarizing Evaluation Metrics:

**Accuracy:** Simple yet sometimes misleading, especially for imbalanced datasets.

**Precision, Recall, F1-score:** Offer a balanced, detailed assessment of classifier performance, balancing false positives and false negatives.

**Confusion Matrix:** Visual and intuitive, helps to understand the exact types of errors your model is making.

---

### [4.3 Summary](chapter4/section4.3/README.md)

In advanced text processing and classification tasks, mastering the use of traditional machine learning algorithms and understanding evaluation metrics is key to building effective models. As exemplified above:

- **Traditional Machine Learning Algorithms:** Employ methods like Naive Bayes, SVMs, Decision Trees, and Ensemble methods to develop robust text classifiers.
- **Evaluation Metrics:** Measure performance using a variety of metrics to ensure a well-rounded understanding of model capabilities and limitations.

By effectively implementing and evaluating these techniques, one can build text classification models that not only perform well on training data but also generalize effectively to unseen data, providing valuable insights into real-world text datasets.

---
