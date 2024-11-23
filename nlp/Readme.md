# Natural Language Processing (NLP) with Python Course

## 📚 Chapters

### [1. Introduction to NLP](chapter1/README.md)
#### [1.1 What is NLP?](chapter1/README.md)
- [History and applications of NLP](chapter1/README.md)
- [Basic terminology (tokens, corpus, etc.)](chapter1/README.md)

#### [1.2 Setting up the Environment](chapter1/README.md)
- [Installing Python and Jupyter Notebook](chapter1/section1.2/installing_python_jupyter.md)
- [Installing essential libraries (nltk, spaCy, transformers, etc.)](chapter1/section1.2/installing_libraries.md)

### [2. Text Preprocessing](chapter2/README.md)
#### [2.1 Tokenization](chapter2/section2.1/README.md)
- [Word tokenization](chapter2/section2.1/word_tokenization.md)
- [Sentence tokenization](chapter2/section2.1/sentence_tokenization.md)

#### [2.2 Removing Stop Words, Punctuation, and Special Characters](chapter2/section2.2/README.md)

#### [2.3 Stemming and Lemmatization](chapter2/section2.3/README.md)
- [Porter stemmer](chapter2/section2.3/porter_stemmer.md)
- [Snowball stemmer](chapter2/section2.3/snowball_stemmer.md)
- [Using spaCy for lemmatization](chapter2/section2.3/using_spacy_lemmatization.md)

#### [2.4 Text Normalization](chapter2/section2.4/README.md)
- [Lowercasing](chapter2/section2.4/lowercasing.md)
- [Handling contractions](chapter2/section2.4/handling_contractions.md)
- [Removing extra whitespace](chapter2/section2.4/removing_whitespace.md)

### [3. Working with Text Data](chapter3/README.md)
#### [3.1 Bag of Words (BoW)](chapter3/section3.1/README.md)
- [Vectorization using BoW](chapter3/section3.1/vectorization_bow.md)
- [CountVectorizer with sklearn](chapter3/section3.1/countvectorizer_sklearn.md)

#### [3.2 TF-IDF](chapter3/section3.2/README.md)
- [Understanding Term Frequency-Inverse Document Frequency](chapter3/section3.2/tfidf_understanding.md)
- [TF-IDF Vectorizer with sklearn](chapter3/section3.2/tfidf_vectorizer_sklearn.md)

#### [3.3 Word Embeddings](chapter3/section3.3/README.md)
- [Word2Vec](chapter3/section3.3/word2vec.md)
- [GloVe](chapter3/section3.3/glove.md)
- [FastText](chapter3/section3.3/fasttext.md)

### [4. Text Classification](chapter4/README.md)
#### [4.1 Traditional Machine Learning Algorithms](chapter4/README.md)
- [Naive Bayes](chapter4/README.md)
- [Support Vector Machines (SVM)](chapter4/README.md)
- [Decision Trees](chapter4/README.md)
- [Ensemble methods (Random Forest, Gradient Boosting)](chapter4/README.md)

#### [4.2 Evaluation Metrics](chapter4/README.md)
- [Accuracy](chapter4/README.md)
- [Precision, Recall, F1-score](chapter4/README.md)
- [Confusion Matrix](chapter4/README.md)

### [5. Advanced NLP Techniques](chapter5/README.md)
#### [5.1 Named Entity Recognition (NER)](chapter5/README.md)
- [Using spaCy for NER](chapter5/README.md)
- [Custom NER models](chapter5/README.md)

#### [5.2 Part-of-Speech Tagging](chapter5/README.md)
- [Using nltk and spaCy for POS tagging](chapter5/README.md)

#### [5.3 Dependency Parsing](chapter5/README.md)
- [Using spaCy for dependency parsing](chapter5/README.md)

#### [5.4 Coreference Resolution](chapter5/README.md)
- [Understanding coreference](chapter5/README.md)
- [Using spaCy and other libraries for coreference resolution](chapter5/README.md)

### [6. Topic Modeling](chapter6/README.md)
#### [6.1 Latent Dirichlet Allocation (LDA)](chapter6/README.md)
- [Understanding LDA](chapter6/README.md)
- [Implementing LDA with gensim](chapter6/README.md)

#### [6.2 Latent Semantic Analysis (LSA)](chapter6/README.md)
- [Understanding LSA](chapter6/README.md)
- [Implementing LSA with sklearn](chapter6/README.md)

### [7. Sentiment Analysis](chapter7/README.md)
- [Rule-based sentiment analysis](chapter7/README.md)
- [Machine learning-based sentiment analysis](chapter7/README.md)
- [Using pre-trained models (e.g., VADER, TextBlob)](chapter7/README.md)

### [8. Sequence Models](chapter8/README.md)
#### [8.1 Recurrent Neural Networks (RNN)](chapter8/README.md)
- [Understanding RNNs](chapter8/README.md)
- [Implementing RNNs using Keras/TensorFlow](chapter8/README.md)

#### [8.2 Long Short-Term Memory Networks (LSTM)](chapter8/README.md)
- [Understanding LSTM](chapter8/README.md)
- [Implementing LSTM using Keras/TensorFlow](chapter8/README.md)

#### [8.3 Gated Recurrent Units (GRU)](chapter8/README.md)
- [Understanding GRU](chapter8/README.md)
- [Implementing GRU using Keras/TensorFlow](chapter8/README.md)

### [9. Transformer Models](chapter9/README.md)
#### [9.1 Attention Mechanism](chapter9/README.md)
- [Understanding attention mechanism](chapter9/README.md)
- [Self-attention](chapter9/README.md)

#### [9.2 Introduction to Transformer Architecture](chapter9/README.md)
- [Understanding the transformer architecture](chapter9/README.md)
- [Encoder-Decoder structure](chapter9/README.md)

#### [9.3 Implementing Transformers](chapter9/README.md)
- [Using Hugging Face's transformers library](chapter9/README.md)

#### [9.4 BERT and its Variants](chapter9/README.md)
- [Understanding BERT](chapter9/README.md)
- [Fine-tuning BERT for specific tasks](chapter9/README.md)

### [10. Large Language Models (LLMs)](chapter10/README.md)
#### [10.1 Introduction to LLMs](chapter10/README.md)
- [GPT, GPT-2, GPT-3 overview](chapter10/README.md)
- [Understanding the advantages and challenges](chapter10/README.md)

#### [10.2 Fine-tuning LLMs](chapter10/README.md)
- [Fine-tuning GPT models for specific tasks](chapter10/README.md)
- [Prompt engineering](chapter10/README.md)

#### [10.3 Deploying LLMs](chapter10/README.md)
- [Using APIs (OpenAI, Hugging Face)](chapter10/README.md)
- [Building simple applications with LLMs](chapter10/README.md)

### [11. Projects and Applications](chapter11/README.md)
#### [Project 1: Spam Detection](chapter11/project1_spam_detection.md)
- [Dataset collection and preprocessing](chapter11/project1_section1.md)
- [Building and evaluating a classification model](chapter11/project1_section2.md)

#### [Project 2: Sentiment Analysis of Movie Reviews](chapter11/project2_sentiment_analysis.md)
- [Data preprocessing](chapter11/project2_section1.md)
- [Building a sentiment analysis model](chapter11/project2_section2.md)
- [Visualizing results](chapter11/project2_section3.md)

#### [Project 3: News Article Classification](chapter11/project3_news_classification.md)
- [Data collection and preprocessing](chapter11/project3_section1.md)
- [Implementing and evaluating multiple classification models](chapter11/project3_section2.md)

#### [Project 4: Named Entity Recognition (NER) System](chapter11/project4_ner_system.md)
- [Using pre-trained models](chapter11/project4_section1.md)
- [Fine-tuning and custom NER models](chapter11/project4_section2.md)

#### [Project 5: Topic Modeling on a Large Text Corpus](chapter11/project5_topic_modeling.md)
- [Preprocessing a large text dataset](chapter11/project5_section1.md)
- [Implementing LDA for topic modeling](chapter11/project5_section2.md)
- [Visualizing top trends](chapter11/project5_section3.md)

#### [Project 6: Chatbot Development](chapter11/project6_chatbot.md)
- [Using RNN/LSTM for intent recognition](chapter11/project6_section1.md)
- [Building a simple rule-based chatbot](chapter11/project6_section2.md)
- [Enhancing using Transformer models](chapter11/project6_section3.md)

#### [Project 7: Text Summarization](chapter11/project7_text_summarization.md)
- [Implementing extractive text summarization](chapter11/project7_section1.md)
- [Using transformer models for abstractive summarization](chapter11/project7_section2.md)

#### [Project 8: Language Generation with GPT-3](chapter11/project8_language_generation.md)
- [Prompt engineering with GPT-3](chapter11/project8_section1.md)
- [Fine-tuning GPT-3 for specific applications](chapter11/project8_section2.md)
- [Building a simple application using GPT-3](chapter11/project8_section3.md)

### [12. Ethics and Best Practices in NLP](chapter12/README.md)
- [Understanding bias in data and models](chapter12/section12.1/understanding_bias.md)
- [Importance of data privacy](chapter12/section12.2/data_privacy.md)
- [Building fair and interpretable models](chapter12/section12.3/fair_interpretable.md)

### 13. [Understanding Security Concerns](chapter13/README.md)
    - Types of Security Risks
    - Examples of Security Risks
- Strategies to Mitigate Security Risks
- Implementing Security Measures
    - Input Validation and Moderation
    - Output Filtering
    - Model Retraining and Updates
- **Case Studies and Real-World Applications**
    - Case Study 1: Preventing Misinformation
    - Case Study 2: Protecting Sensitive Data
- **Best Practices for Developers**
    - Ethical Considerations
    - Continuous Monitoring and Improvement
- **Exercises**
    - Input Validation and Moderation Implementation
    - Output Filtering Techniques
    - Creating a Monitoring System for LLM Outputs

### [14. Final Capstone Project](chapter14/README.md)
#### [Capstone Project: End-to-End NLP Application](chapter14/README.md)
- [Choose a real-world problem](chapter14/README.md)
- [Data collection and preprocessing](chapter14/README.md)
- [Model selection, training, and evaluation](chapter14/README.md)
- [Deployment and presentation](chapter14/README.md)

## 📦 Python Libraries

### [nltk](libraries/nltk/README.md)
#### [Overview](libraries/nltk/overview.md)
nltk (NaturalLanguage Toolkit) is a powerful library for working with human language data in Python.

#### [Installation](libraries/nltk/installation.md)
```bash
pip install nltk
```

#### [Examples](libraries/nltk/examples.md)
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Natural Language Processing with Python is fun!"
tokens = word_tokenize(text)
print(tokens)
```

### [spaCy](libraries/spacy/README.md)
#### [Overview](libraries/spacy/overview.md)
spaCy is an open-source library for advanced NLP with Python, designed for efficiency.

#### [Installation](libraries/spacy/installation.md)
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

#### [Examples](libraries/spacy/examples.md)
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Natural Language Processing with Python is fun!")
for token in doc:
    print(token.text, token.pos_, token.dep_)
```

### [transformers](libraries/transformers/README.md)
#### [Overview](libraries/transformers/overview.md)
transformers by Hugging Face provides thousands of pretrained models to perform tasks on different modalities.

#### [Installation](libraries/transformers/installation.md)
```bash
pip install transformers
```

#### [Examples](libraries/transformers/examples.md)
```python
from transformers import pipeline

nlp_pipeline = pipeline("sentiment-analysis")
result = nlp_pipeline("I love learning new things about NLP!")
print(result)
```

### [gensim](libraries/gensim/README.md)
#### [Overview](libraries/gensim/overview.md)
gensim is a robust library for topic modeling and document similarity analysis.

#### [Installation](libraries/gensim/installation.md)
```bash
pip install gensim
```

#### [Examples](libraries/gensim/examples.md)
```python
from gensim import corpora, models

documents = [
    "Natural Language Processing with Python is fun!",
    "NLTK and spaCy are useful libraries."
]
texts = [[word for word in document.lower().split()] for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train the model
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
```

