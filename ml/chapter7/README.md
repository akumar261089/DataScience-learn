
# Chapter 7: Introduction to Deep Learning
---

## 7.1 Neural Networks

### 7.1.1 Perceptrons and Multilayer Perceptrons
- **Description**: A perceptron is a basic unit of a neural network that performs a binary classification. A Multilayer Perceptron (MLP) consists of multiple layers of perceptrons (neurons) and can classify data that is not linearly separable.
- **Significance**:
  - The perceptron is the foundational building block of neural networks.
  - MLPs can model complex functions and solve problems that simple linear models cannot.
- **Usage**: Used in various applications such as image and speech recognition, among others.
- **Example**:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.neural_network import MLPClassifier
  from sklearn.datasets import make_moons

  # Generate sample data
  X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
  
  # Create and train the model
  mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
  mlp.fit(X, y)
  
  # Plot the decision boundary
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
  Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
  plt.show()
  ```

### 7.1.2 Activation Functions
- **Description**: Activation functions introduce non-linearity into the network, allowing it to learn complex patterns.
- **Significance**:
  - Essential for enabling neural networks to approximate complex functions.
  - Different activation functions (e.g., sigmoid, tanh, ReLU) have different properties and are used based on the specific problem.
- **Usage**: Commonly used activation functions include sigmoid, tanh, ReLU, and variants like Leaky ReLU.
- **Example**:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt

  # Define activation functions
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))
  
  def tanh(x):
      return np.tanh(x)
  
  def relu(x):
      return np.maximum(0, x)
  
  # Generate sample data
  x = np.linspace(-10, 10, 100)
  
  # Apply activation functions
  y_sigmoid = sigmoid(x)
  y_tanh = tanh(x)
  y_relu = relu(x)
  
  # Plot activation functions
  plt.figure(figsize=(12, 6))
  plt.plot(x, y_sigmoid, label='Sigmoid')
  plt.plot(x, y_tanh, label='tanh')
  plt.plot(x, y_relu, label='ReLU')
  plt.legend()
  plt.xlabel('Input')
  plt.ylabel('Output')
  plt.title('Activation Functions')
  plt.show()
  ```

## 7.2 Training Neural Networks

### 7.2.1 Forward and Backward Propagation
- **Description**: Forward propagation computes the output of the neural network given the input. Backward propagation calculates the gradients of the loss function with respect to each weight in the network.
- **Significance**:
  - Forward propagation allows the network to make predictions.
  - Backward propagation is used for learning by updating the weights in the network.
- **Usage**: Integral part of the training process of neural networks.
- **Example**:
  ```python
  import numpy as np

  # Example of forward and backward propagation in a simple neural network

  # Define activation functions and their derivatives
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(x):
      return x * (1 - x)

  # Define input features and target
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y = np.array([[0], [1], [1], [0]])

  # Initialize weights
  np.random.seed(42)
  weights_input_hidden = np.random.random((2, 2))
  weights_hidden_output = np.random.random((2, 1))

  # Define learning rate
  learning_rate = 0.1

  # Training (forward and backward propagation)
  for epoch in range(1000):
      # Forward propagation
      hidden_layer_input = np.dot(X, weights_input_hidden)
      hidden_layer_output = sigmoid(hidden_layer_input)
      final_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
      y_pred = sigmoid(final_layer_input)

      # Calculate error
      error = y - y_pred
      if (epoch + 1) % 200 == 0:
          print(f'Epoch {epoch+1}, Error: {np.mean(np.abs(error))}')

      # Backward propagation
      d_y_pred = error * sigmoid_derivative(y_pred)
      d_hidden_layer_output = d_y_pred.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

      # Update weights
      weights_hidden_output += hidden_layer_output.T.dot(d_y_pred) * learning_rate
      weights_input_hidden += X.T.dot(d_hidden_layer_output) * learning_rate
  ```

### 7.2.2 Gradient Descent and Optimization Algorithms
- **Description**: Gradient descent is an optimization algorithm used to minimize the loss function by iteratively updating the model's parameters. Variants include Stochastic Gradient Descent (SGD), Mini-batch Gradient Descent, Adam, RMSprop, etc.
- **Significance**:
  - Essential for training neural networks efficiently.
  - Different optimization algorithms can significantly affect the convergence speed and performance.
- **Usage**: Adam, RMSprop, and SGD are commonly used in practice.
- **Example**:
  ```python
  import numpy as np
  from sklearn.datasets import make_classification
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.optimizers import SGD, Adam, RMSprop

  # Generate sample data
  X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

  # Define the neural network model
  model = Sequential()
  model.add(Dense(10, input_dim=20, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  # Compile the model with different optimizers
  optimizer = Adam(learning_rate=0.01)
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  # Train the model
  model.fit(X, y, epochs=50, batch_size=10)
  ```

### 7.2.3 Overfitting and Regularization
- **Description**: Overfitting occurs when a model learns the training data too well, including its noise and outliers, leading to poor performance on unseen data. Regularization techniques like L1, L2 (Ridge), and Dropout are used to prevent overfitting.
- **Significance**:
  - Regularization helps in generalizing the model to new data, making it robust.
- **Usage**: Regularization techniques are applied depending on the specific problem.
- **Example**:
  ```python
  import numpy as np
  from sklearn.datasets import make_classification
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout
  from tensorflow.keras.regularizers import l2

  # Generate sample data
  X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

  # Define the neural network model with regularization
  model = Sequential()
  model.add(Dense(64, input_dim=20, activation='relu', kernel_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))

  # Compile the model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Train the model
  model.fit(X, y, epochs=50, batch_size=10, validation_split=0.2)
  ```
  
