### [2. Text Preprocessing](chapter2/README.md)

Text preprocessing is a crucial step in preparing textual data for machine learning and natural language processing (NLP) tasks. It involves various techniques to clean and transform raw text into a format that models can easily work with. This chapter delves into the essential methods and algorithms used to preprocess text data effectively.

#### [2.1 Tokenization](chapter2/section2.1/README.md)

Tokenization is the process of breaking down text into smaller units called tokens. These tokens can be words, sentences, or even subwords. Tokenization plays a critical role in converting textual data into manageable and analyzable formats.

- **[Word tokenization](chapter2/section2.1/word_tokenization.md):** This method involves splitting text into individual words. Below is an example using Python with the NLTK library:

  ```python
  import nltk
  nltk.download('punkt')
  from nltk.tokenize import word_tokenize

  text = "Machine learning is fascinating."
  word_tokens = word_tokenize(text)
  print(word_tokens)
  # Output: ['Machine', 'learning', 'is', 'fascinating', '.']
  ```

- **[Sentence tokenization](chapter2/section2.1/sentence_tokenization.md):** This method involves splitting text into sentences. Below is an example using Python with the NLTK library:

  ```python
  import nltk
  nltk.download('punkt')
  from nltk.tokenize import sent_tokenize

  text = "Machine learning is fascinating. It has many applications."
  sentence_tokens = sent_tokenize(text)
  print(sentence_tokens)
  # Output: ['Machine learning is fascinating.', 'It has many applications.']
  ```

#### [2.2 Removing Stop Words, Punctuation, and Special Characters](chapter2/section2.2/README.md)

Stop words are common words that carry little meaningful information, such as "and", "the", and "in". Punctuation and special characters can also be removed to streamline the text for analysis. Removing these elements helps focus on the more meaningful words in the text.

Example:
- Original text: "This is a sample text, with stop words and punctuation!"
- Processed text: "sample text stop words punctuation"

Coding example using NLTK to remove stop words:

```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

text = "This is a sample text, with stop words and punctuation!"
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(text)

filtered_words = [word for word in word_tokens if word.lower() not in stop_words and word not in string.punctuation]
print(filtered_words)
# Output: ['This', 'sample', 'text', 'stop', 'words', 'punctuation']
```

#### [2.3 Stemming and Lemmatization](chapter2/section2.3/README.md)

Stemming and lemmatization are techniques to reduce words to their base or root form. This helps in normalizing the text and reducing the vocabulary size.

- **[Porter stemmer](chapter2/section2.3/porter_stemmer.md):** Developed by Martin Porter, this algorithm removes common morphological and inflectional endings from words.

  ```python
  from nltk.stem import PorterStemmer

  porter = PorterStemmer()
  words = ["running", "jumps", "easily", "faster"]
  stemmed_words = [porter.stem(word) for word in words]
  print(stemmed_words)
  # Output: ['run', 'jump', 'easili', 'faster']
  ```

- **[Snowball stemmer](chapter2/section2.3/snowball_stemmer.md):** An improvement over the Porter stemmer, this algorithm provides more accurate stemming.

  ```python
  from nltk.stem.snowball import SnowballStemmer

  snowball = SnowballStemmer("english")
  words = ["running", "jumps", "easily", "faster"]
  stemmed_words = [snowball.stem(word) for word in words]
  print(stemmed_words)
  # Output: ['run', 'jump', 'easili', 'faster']
  ```

- **[Using spaCy for lemmatization](chapter2/section2.3/using_spacy_lemmatization.md):** SpaCy is a popular NLP library for Python. Lemmatizationwith spaCy involves converting words to their base or dictionary form.

```python
import spacy

nlp = spacy.load('en_core_web_sm')
text = "running runs ran easily better"
doc = nlp(text)

lemmatized_words = [token.lemma_ for token in doc]
print(lemmatized_words)
# Output: ['run', 'run', 'run', 'easily', 'well']
```

#### [2.4 Text Normalization](chapter2/section2.4/README.md)

Text normalization is the process of transforming text into a standard format. It includes various techniques to homogenize the text data.

- **[Lowercasing](chapter2/section2.4/lowercasing.md):** Converting all characters in the text to lowercase. This standardizes the text and helps in reducing the dimensionality of the data.

  ```python
  text = "Text Processing With Python"
  lowercased_text = text.lower()
  print(lowercased_text)
  # Output: "text processing with python"
  ```

- **[Handling contractions](chapter2/section2.4/handling_contractions.md):** Expanding contractions into their full forms to maintain consistency in the text.

  ```python
  import contractions

  text = "I'm learning NLP and it's very interesting!"
  expanded_text = contractions.fix(text)
  print(expanded_text)
  # Output: "I am learning NLP and it is very interesting!"
  ```

  Alternatively, you can handle contractions manually using a dictionary:
  
  ```python
  contraction_mapping = {
      "don't": "do not",
      "I'm": "I am",
      "it's": "it is",
      "you're": "you are",
      # Add more contractions as needed
  }

  text = "I'm learning NLP and it's very interesting!"
  expanded_text = ' '.join([contraction_mapping.get(word, word) for word in text.split()])
  print(expanded_text)
  # Output: "I am learning NLP and it is very interesting!"
  ```

- **[Removing extra whitespace](chapter2/section2.4/removing_whitespace.md):** Eliminating additional spaces, tabs, or newlines from the text. This cleans up the text for better readability and processing.

  ```python
  text = "This is    a text   with     extra spaces."
  normalized_text = ' '.join(text.split())
  print(normalized_text)
  # Output: "This is a text with extra spaces."
  ```
