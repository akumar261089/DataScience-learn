## Module 6: Practical Machine Learning Applications

### **6.1 Image Classification**
#### Building an Image Classifier
- **Theory**: Image classification involves training a model to recognize different classes of images.
- **Example**:
  ```javascript
  import * as tf from '@tensorflow/tfjs';

  // Define a simple convolutional neural network
  const model = tf.sequential();
  model.add(tf.layers.conv2d({inputShape: [64, 64, 3], filters: 32, kernelSize: 3, activation: 'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize: 2}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Dummy data for example
  const xs = tf.randomNormal([100, 64, 64, 3]);
  const ys = tf.randomUniform([100, 10]);

  model.fit(xs, ys, {epochs: 10}).then(() => {
    console.log('Model trained');
  });
  ```
- **Usage**: This model can recognize distinct classes of images such as different animals, vehicles, or objects.
- **Significance**: Automates visual recognition tasks, which is applicable in fields like self-driving cars, healthcare diagnostics, and retail.

#### Using Pre-trained Models (e.g., MobileNet)
- **Theory**: Pre-trained models come already trained on large datasets, providing a strong foundation for a wide variety of tasks.
- **Example**:
  ```javascript
  import * as mobilenet from '@tensorflow-models/mobilenet';

  const img = document.getElementById('img'); // HTML element for the image

  mobilenet.load().then(model => {
    model.classify(img).then(predictions => {
      console.log('Predictions: ', predictions);
    });
  });
  ```
- **Usage**: Efficiently classify images using highly optimized, pre-trained models.
- **Significance**: Speeds up development time and improves accuracy by utilizing models trained on extensive datasets.

#### Fine-tuning Models in the Browser
- **Example**:
  ```javascript
  import * as mobilenet from '@tensorflow-models/mobilenet';
  import * as tf from '@tensorflow/tfjs';

  async function fineTune() {
    const baseModel = await mobilenet.load({ version: 2, alpha: 1.0 });
    const model = tf.sequential();
    for (const layer of baseModel.layers) {
      layer.trainable = false;
      model.add(layer);
    }
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    const xs = tf.randomNormal([100, 64, 64, 3]);
    const ys = tf.randomUniform([100, 10]);

    await model.fit(xs, ys, { epochs: 10 });
    console.log('Model fine-tuned');
  }
  fineTune();
  ```
- **Usage**: Adjust pre-trained models with new datasets to suit specific classification tasks.
- **Significance**: Leverages existing models to quickly adapt to new tasks, reducing the need for extensive data and training time.

### **6.2 Text Analysis**
#### Text Preprocessing (Tokenization, Normalization)
- **Theory**: Preprocessing text involves converting raw text into a structured form suitable for machine learning models.
- **Example**:
  ```javascript
  // Tokenization example
  const text = "TensorFlow.js is amazing!";
  const words = text.toLowerCase().split(' ');

  // Create a simple vocabulary index
  const vocab = {};
  words.forEach((word, index) => {
    vocab[word] = index;
  });

  // Convert words to indices
  const wordIndices = words.map(word => vocab[word]);
  console.log(wordIndices); // [0, 1, 2]
  ```
- **Usage**: Prepare text data for sentiment analysis or text classification tasks.
- **Significance**: Ensures text is in a format that machine learning models can understand and process effectively.

#### Sentiment Analysis with TensorFlow.js
- **Example**:
  ```javascript
  import * as tf from '@tensorflow/tfjs';

  // Define a simple model for sentiment analysis
  const model = tf.sequential();
  model.add(tf.layers.embedding({inputDim: 10000, outputDim: 16, inputLength: 100}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  // Dummy data for example
  const xs = tf.randomUniform([100, 100], 0, 10000, 'int32');
  const ys = tf.randomUniform([100, 1]);

  model.fit(xs, ys, {epochs: 10}).then(() => {
    console.log('Model trained for sentiment analysis');
  });
  ```
- **Usage**: Classify text as positive or negative sentiment.
- **Significance**: Useful for applications like social media monitoring, customer feedback analysis, and product reviews.

#### Building a Text Classifier
- **Example**:
  ```javascript
  import * as tf from '@tensorflow/tfjs';

  // Define a text classifier
  const model = tf.sequential();
  model.add(tf.layers.embedding({inputDim: 5000, outputDim: 128, inputLength: 200}));
  model.add(tf.layers.conv1d({kernelSize: 5, filters: 128, activation: 'relu'}));
  model.add(tf.layers.globalMaxPool1d());
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 5, activation: 'softmax'}));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Dummy data for example
  const xs = tf.randomUniform([100, 200], 0, 5000, 'int32');
  const ys = tf.oneHot(tf.randomUniform([100], 0, 5, 'int32'), 5);

  model.fit(xs, ys, {epochs: 10}).then(() => {
    console.log('Text classifier model trained');
  });
  ```
- **Usage**: Categorize text into multiple classes, such as topics or genres.
- **Significance**: Facilitates tasks like email categorization, document tagging, and news categorization.

### **6.3 Time Series Forecasting**
#### Preparing Time Series Data
- **Theory**: Time series data involves observations recorded at sequential time intervals, requiring special preprocessing techniques.
- **Example**:
  ```javascript
  // Example time series data preparation
  const timeSeriesData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const windowSize = 3;

  function createWindows(data, windowSize) {
    const windows = [];
    for (let i = 0; i < data.length - windowSize; i++) {
      windows.push(data.slice(i, i + windowSize));
    }
    return windows;
  }

  const windows = createWindows(timeSeriesData, windowSize);
  console.log(windows);
  ```
- **Usage**: Convert raw time series data into a format suitable for training forecasting models.
- **Significance**: Ensures models can capture temporal dependencies in the data, which are crucial for accurate forecasting.

#### Building and Training a Forecasting Model
- **Example**:
  ```javascript
  import * as tf from '@tensorflow/tfjs';

  // Define a simple LSTM model for time series forecasting
  const model = tf.sequential();
  model.add(tf.layers.lstm({units: 50, returnSequences: true, inputShape: [10, 1]}));
  model.add(tf.layers.lstm({units: 50}));
  model.add(tf.layers.dense({units: 1}));

  model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError',
  });

  // Dummy data for example
  const xs = tf.randomUniform([100, 10, 1]);
  const ys = tf.randomUniform([100, 1]);

  model.fit(xs, ys, {epochs: 10}).then(() => {
    console.log('Time series forecasting model trained');
  });
  ```
- **Usage**: Predict future values in a time series, such as stock prices, weather data, or sales figures.
- **Significance**: Supports proactive decision-making and planning in various applications, from finance to operations.

#### Evaluating Time Series Models
- **Example**:
  ```javascript
  import * as tf from '@tensorflow/tfjs';

  // Split time series data into training and testing sets
  const splitAt = Math.floor(timeSeriesData.length * 0.8);
  const trainData = timeSeriesData.slice(0, splitAt);
  const testData = timeSeriesData.slice(splitAt);

  // Evaluation function
  function evaluateModel(model, data) {
    const xs = tf.tensor(data.inputs);
    const ys = tf.tensor(data.targets);
    const preds = model.predict(xs);

    preds.print(); // Output model predictions

    const error = preds.sub(ys).square().mean().sqrt();
    error.print(); // Output mean squared error
  }

  evaluateModel(model, {inputs: testData, targets: testDataTargets});
  ```
- **Usage**: Measure model accuracy using metrics such as Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).
- **Significance**: Ensures model validity and reliability, crucial for making dependable forecasts.

### Conclusion
Practical machine learning applications with TensorFlow.js cover a broad range of scenarios, including image classification, text analysis, and time series forecasting. Mastering these areas allows developers to build intelligent applications that can automatically understand and respond to diverse data types. Using TensorFlow.js, these powerful capabilities can be harnessed directly in the browser or on the server, providing flexibility and performance to modern web applications.