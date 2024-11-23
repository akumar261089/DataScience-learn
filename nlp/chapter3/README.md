## [3. Working with Text Data](chapter3/README.md)

Working with text data involves converting raw text into a numerical form that machine learning algorithms can process. This chapter covers fundamental techniques for representing and extracting valuable features from text data.

#### [3.1 Bag of Words (BoW)](chapter3/section3.1/README.md)

Bag of Words (BoW) is a simple and widely-used method for text representation. In the BoW model, a text document is represented as a collection (or bag) of its words, disregarding grammar and word order but maintaining multiplicity.

- **[Vectorization using BoW](chapter3/section3.1/vectorization_bow.md):** Vectorization refers to converting text into numerical vectors. The BoW vectorization creates a vocabulary of known words, assigns each a unique index, and represents each document as a vector of word occurrence counts.

  **Theory:**
  - Each unique word in the text corpus forms a feature in the vector space.
  - Each document is then described by a vector of word counts, possibly term frequencies.

  **Coding Example:**
  ```python
  from sklearn.feature_extraction.text import CountVectorizer

  documents = ["Machine learning is great", "Natural language processing is a part of machine learning"]
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(documents)

  print(vectorizer.get_feature_names_out())
  # Output: ['great', 'is', 'language', 'learning', 'machine', 'natural', 'of', 'part', 'processing']

  print(X.toarray())
  # Output: 
  # [[1 1 0 1 1 0 0 0 0]
  #  [0 2 1 1 1 1 1 1 1]]
  ```

- **[CountVectorizer with sklearn](chapter3/section3.1/countvectorizer_sklearn.md):** The `CountVectorizer` class in `scikit-learn` provides a simple way to convert text documents to a matrix of token counts.

#### [3.2 TF-IDF](chapter3/section3.2/README.md)

TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. Unlike the BoW model, it reflects the importance of a word in a document relative to the entire corpus.

- **[Understanding Term Frequency-Inverse Document Frequency](chapter3/section3.2/tfidf_understanding.md):** 
  - Term Frequency (TF): Measures how frequently a term appears in a document.
  - Inverse Document Frequency (IDF): Measures how important a term is by reflecting how common or rare it is across the entire corpus.

  **Formula:**
  \[
  \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log \left( \frac{N}{\text{DF}(t)} \right)
  \]
  where:
  - \( t \) is the term
  - \( d \) is the document
  - \( N \) is the total number of documents
  - \( \text{DF}(t) \) is the number of documents containing the term t

- ****[TF-IDF Vectorizer with sklearn](chapter3/section3.2/tfidf_vectorizer_sklearn.md):** The `TfidfVectorizer` class in `scikit-learn` allows us to transform a collection of raw documents into a matrix of TF-IDF features.

**Coding Example:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["Machine learning is great", "Natural language processing is a part of machine learning"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
# Output: ['great', 'is', 'language', 'learning', 'machine', 'natural', 'of', 'part', 'processing']

print(X.toarray())
# Output:
# [[0.         0.46941728 0.         0.58028582 0.58028582 0.         0.         0.         0.        ]
#  [0.         0.24187323 0.38754468 0.29926894 0.29926894 0.38754468 0.38754468 0.38754468 0.38754468]]
```

The significance of TF-IDF lies in its ability to filter out commonly found words across documents while emphasizing words that are more unique and relevant to a specific document.

#### [3.3 Word Embeddings](chapter3/section3.3/README.md)

Word embeddings are a type of word representation that allows words to be represented as vectors in a continuous vector space where semantically similar words are close to each other. Unlike BoW and TF-IDF models, word embeddings capture the context of words, which allows the meaning and relationship between words to be preserved.

- **[Word2Vec](chapter3/section3.3/word2vec.md):** 
  - Developed by Google, Word2Vec uses neural networks to learn distributed representations of words.
  - It comes in two flavors: Continuous Bag of Words (CBOW) and Skip-gram.
  - `CBOW` predicts the target word from the context, while `Skip-gram` predicts the context words from the target word.

  **Coding Example:**

  ```python
  import gensim
  from gensim.models import Word2Vec

  sentences = [["machine", "learning", "is", "great"], ["natural", "language", "processing"]]
  model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

  # Accessing the vector for a word
  vector = model.wv['machine']
  print(vector)

  # Finding similar words
  similar_words = model.wv.most_similar('machine')
  print(similar_words)
  ```

- **[GloVe](chapter3/section3.3/glove.md):** 
  - GloVe (Global Vectors for Word Representation) is a word embedding model developed by Stanford.
    - It constructs a co-occurrence matrix from a corpus, which captures how frequently pairs of words appear together within a set window size. It then uses matrix factorization techniques to derive word vectors.

  **Coding Example:**

  ```python
  from glove import Corpus, Glove

  sentences = [["machine", "learning", "is", "great"], ["natural", "language", "processing"]]
  corpus = Corpus()
  corpus.fit(sentences, window=10)

  glove = Glove(no_components=100, learning_rate=0.05)
  glove.fit(corpus.matrix, epochs=10, no_threads=4, verbose=True)
  glove.add_dictionary(corpus.dictionary)

  # Accessing the vector for a word
  vector = glove.word_vectors[glove.dictionary['machine']]
  print(vector)

  # Finding similar words
  similar_words = glove.most_similar('machine')
  print(similar_words)
  ```

- **[FastText](chapter3/section3.3/fasttext.md):**
  - Developed by Facebook, FastText extends the Word2Vec model by representing each word as a collection of subwords (n-grams). This allows it to generate embeddings even for words that were not seen during training.
  - It is particularly effective for morphologically rich languages and can handle rare words better.

  **Coding Example:**

  ```python
  from gensim.models import FastText

  sentences = [["machine", "learning", "is", "great"], ["natural", "language", "processing"]]
  model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

  # Accessing the vector for a word
  vector = model.wv['machine']
  print(vector)

  # Finding similar words
  similar_words = model.wv.most_similar('machine')
  print(similar_words)
  ```

### Uses and Significance

1. **Bag of Words (BoW):**
  - **Uses:** Simple and effective for text classification tasks when context and word order are not critical.
  - **Significance:** Fast to implement and interpret but limited by its disregard for word order and context.

2. **TF-IDF:**
  - **Uses:** Information retrieval, text classification, and clustering by highlighting important words in documents.
  - **Significance:** Captures the importance of words in a document relative to the entire corpus, providing better features than BoW for many tasks.

3. **Word Embeddings (Word2Vec, GloVe, FastText):**
  - **Uses:** Wide range of NLP tasks, such as sentiment analysis, machine translation, named entity recognition, and more.
  - **Significance:** Preserves semantic relationships between words, allowing models to leverage word meanings and context for improved performance.

### Conclusion

This chapter has explored essential techniques for representing text data, ranging from simple methods like Bag of Words to advanced techniques like word embeddings. Each method has its own strengths and applications. By understanding and employing these methods, you can effectively work with text data to build powerful NLP models.

**Next Chapter:** [Advanced Text Processing Techniques](chapter4/README.md)