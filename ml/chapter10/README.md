
# Chapter 10: Large Language Models (LLMs) and Generative AI
---

## 10.1 Understanding LLMs

### 10.1.1 Introduction to LLMs (e.g., GPT, BERT)
- **Description**: Large Language Models (LLMs) are deep learning models trained on vast amounts of text data. They are designed to understand, generate, and manipulate human language.
- **Significance**:
  - LLMs have revolutionized natural language processing (NLP) by achieving state-of-the-art performance on various tasks.
  - Examples include language translation, text generation, sentiment analysis, and more.
- **Usage**:
  - GPT (Generative Pre-trained Transformer) is designed for generating human-like text.
  - BERT (Bidirectional Encoder Representations from Transformers) is focused on understanding the context of words in a sentence.
- **Example**:
  ```python
  # GPT-3 usage example
  from transformers import GPT3Tokenizer, GPT3LMHeadModel

  tokenizer = GPT3Tokenizer.from_pretrained("gpt3")
  model = GPT3LMHeadModel.from_pretrained("gpt3")

  input_text = "Once upon a time"
  input_ids = tokenizer.encode(input_text, return_tensors='pt')

  output = model.generate(input_ids, max_length=50, num_return_sequences=1)
  print(tokenizer.decode(output[0], skip_special_tokens=True))
  ```

  ```python
  # BERT usage example
  from transformers import BertTokenizer, BertModel

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')

  input_text = "Hello, my dog is cute"
  input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']

  outputs = model(input_ids)
  print(outputs.last_hidden_state)
  ```

### 10.1.2 Applications of LLMs
- **Description**: LLMs are applied in various real-world applications where understanding and generating human language is crucial.
- **Significance**:
  - LLMs add immense value to industries such as finance, healthcare, customer support, content creation, and more.
- **Usage**: Examples include chatbots, automated content generation, code generation, sentiment analysis, translation, etc.
- **Example**:
  ```markdown
  - **Chatbots**: LLMs like GPT can generate human-like responses in conversational agents, improving user interaction.
  - **Content Creation**: Tools like Copy.ai use LLMs to generate marketing content, blog posts, and more.
  - **Sentiment Analysis**: BERT can be fine-tuned to analyze sentiments in social media posts or customer reviews.
  - **Code Generation**: OpenAI Codex can generate code snippets, assisting programmers.
  ```

## 10.2 Generative AI

### 10.2.1 Basics of Generative Adversarial Networks (GANs)
- **Description**: GANs consist of two neural networks, a Generator and a Discriminator, that compete with each other to create new, synthetic data samples that resemble the training data.
- **Significance**:
  - GANs have opened new avenues in generating realistic images, videos, and even music.
  - They are used in applications like image synthesis, super-resolution, and data augmentation.
- **Usage**: GANs are used in creative arts, gaming, fashion design, and enhancing training datasets.
- **Example**:
  ```markdown
  - **Image Synthesis**: GANs can generate photorealistic images of objects, animals, and even people who don't exist in reality.
  - **Super-Resolution**: Enhancing the resolution of images by generating high-quality details.
  - **Data Augmentation**: Generating additional samples to augment small datasets for training other models.
  ```

### 10.2.2 Building Simple GANs with TensorFlow/PyTorch
- **Description**: Building simple GANs involves defining the Generator and Discriminator networks, the training loop, and the loss functions.
- **Significance**:
  - Provides a practical understanding of how GANs work.
- **Usage**: Implementing GANs to create new data samples based on the domain of interest.
- **Example**:
  ```python
  # Simple GAN implementation in PyTorch
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torchvision import datasets, transforms

  # Define the Generator
  class Generator(nn.Module):
      def __init__(self):
          super(Generator, self).init()
          self.main = nn.Sequential(
              nn.Linear(100, 256),
              nn.ReLU(True),
              nn.Linear(256, 512),
              nn.ReLU(True),
              nn.Linear(512, 1024),
              nn.ReLU(True),
              nn.Linear(1024, 28*28),
              nn.Tanh()
          )

      def forward(self, x):
          return self.main(x).view(-1, 1, 28, 28)

  # Define the Discriminator
  class Discriminator(nn.Module):
      def __init__(self):
          super(Discriminator, self).init()
          self.main = nn.Sequential(
              nn.Linear(28*28, 1024),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(1024, 512),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(512, 256),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(256, 1),
              nn.Sigmoid()
          )

      def forward(self, x):
          return self.main(x.view(-1, 28*28))

  # Initialize models
  generator = Generator()
  discriminator = Discriminator()

  # Loss function and optimizers
  criterion = nn.BCELoss()
  optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
  optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

  # Dataset and DataLoader
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])
  dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

  # Training loop
  num_epochs = 30
  for epoch in range(num_epochs):
      for i, (real_images, _) in enumerate(dataloader):
          # Training Discriminator
          optimizer_d.zero_grad()
          
          # Real images
          real_labels = torch.ones(real_images.size(0), 1)
          real_output = discriminator(real_images)
          loss_real = criterion(real_output, real_labels)
          
          # Fake images
          z = torch.randn(real_images.size(0), 100)
          fake_images = generator(z)
          fake_labels = torch.zeros(real_images.size(0), 1)
          fake_output = discriminator(fake_images.detach())
          loss_fake = criterion(fake_output, fake_labels)
          
          loss_d = loss_real + loss_fake
          loss_d.backward()
          optimizer_d.step()

          # Training Generator
          optimizer_g.zero_grad()
          fake_output = discriminator(fake_images)
          loss_g = criterion(fake_output, real_labels)
          loss_g.backward()
          optimizer_g.step()

      print(f'Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}')

  # Generating images
  z = torch.randn(16, 100)
  fake_images = generator(z)
  ```
  ```python
  # Simple GAN implementation in TensorFlow
  import tensorflow as tf
  from tensorflow.keras import layers
  import numpy as np

  # Define the Generator
  def build_generator():
      model = tf.keras.Sequential()
      model.add(layers.Dense(256, input_dim=100, activation='relu'))
      model.add(layers.Dense(512, activation='relu'))
      model.add(layers.Dense(1024, activation='relu'))
      model.add(layers.Dense(28*28, activation='tanh'))
      model.add(layers.Reshape((28, 28, 1)))
      return model

  # Define the Discriminator
  def build_discriminator():
      model = tf.keras.Sequential()
      model.add(layers.Flatten(input_shape=(28, 28, 1)))
      model.add(layers.Dense(512, activation=layers.LeakyReLU(0.2)))
      model.add(layers.Dense(256, activation=layers.LeakyReLU(0.2)))
      model.add(layers.Dense(1, activation='sigmoid'))
      return model

  # Create the models
  generator = build_generator()
  discriminator = build_discriminator()
  discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  z = layers.Input(shape=(100,))
  img = generator(z)
  discriminator.trainable = False
  real_or_fake = discriminator(img)
  combined = tf.keras.Model(z, real_or_fake)
  combined.compile(loss='binary_crossentropy', optimizer='adam')

  # Load and preprocess the dataset
  (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
  X_train = (X_train / 127.5) - 1.0
  X_train = np.expand_dims(X_train, axis=3)

  # Training the GAN
  batch_size = 64
  num_epochs = 30000
  half_batch = int(batch_size / 2)
  
  for epoch in range(num_epochs):
      # Train Discriminator
      idx = np.random.randint(0, X_train.shape[0], half_batch)
      real_images = X_train[idx]
      real_labels = np.ones((half_batch, 1))
      d_loss_real = discriminator.train_on_batch(real_images, real_labels)

      noise = np.random.normal(0, 1, (half_batch, 100))
      fake_images = generator.predict(noise)
      fake_labels = np.zeros((half_batch, 1))
      d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

      d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

      # Train Generator
      noise = np.random.normal(0, 1, (batch_size, 100))
      valid_y = np.ones((batch_size, 1))
      g_loss = combined.train_on_batch(noise, valid_y)

      if epoch % 1000 == 0:
          print(f'{epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]')
  ```

