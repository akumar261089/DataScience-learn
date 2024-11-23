# [5. Advanced NLP Techniques](chapter5/README.md)

Natural Language Processing (NLP) encompasses a range of techniques to understand and manipulate human language. This chapter delves into advanced NLP techniques including Named Entity Recognition (NER), Part-of-Speech (POS) tagging, Dependency Parsing, and Coreference Resolution.

## [5.1 Named Entity Recognition (NER)](chapter5/section5.1/README.md)

Named Entity Recognition (NER) is the process of identifying and classifying named entities (e.g., person names, organizations, locations) within a text.

### [Using spaCy for NER](chapter5/section5.1/spacy.md)

**Theory:**
- NER models use statistical and machine learning approaches to recognize entities.
- Common entities include PERSON, ORG (organization), LOC (location), DATE, etc.

**Coding Example using spaCy:**
```python
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "Apple is looking at buying U.K. startup for $1 billion"

# Process the text
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(f'{ent.text}: {ent.label_}')
# Output:
# Apple: ORG
# U.K.: GPE
# $1 billion: MONEY
```

**Uses:**
- Information extraction.
- Content categorization.
- Simplifying data preprocessing by identifying and normalizing entities.

**Significance:**
- Helps in creating structured data from unstructured text.
- Crucial for applications in search engines, recommendation systems, and knowledge graphs.

### [Custom NER models](chapter5/section5.1/custom_ner.md)

**Theory:**
- Custom NER models are trained on domain-specific data to recognize entities specific to that domain.
- Annotation of training data is required, specifying entity spans and labels.

**Coding Example using spaCy:**
```python
import spacy
from spacy.training import Example
from spacy.tokens import DocBin

# Load a pre-trained model
nlp = spacy.load('en_core_web_sm')

# Training data
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]

# Convert the training data to spaCy's format
db = DocBin()
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    db.add(example.reference)

db.to_disk("./training_data.spacy")

# Train the custom model
import spacy.cli
spacy.cli.train("path/to/config.cfg", output_path="./output")
```

**Uses:**
- Extracting medical terms in healthcare.
- Identifying financial entities in financial documents.

**Significance:**
- Tailored to specific needs, providing higher accuracy for industry-specific applications.

## [5.2 Part-of-Speech Tagging](chapter5/section5.2/README.md)

Part-of-Speech (POS) tagging assigns grammatical tags (e.g., noun, verb) to each word in a sentence.

### [Using nltk and spaCy for POS tagging](chapter5/section5.2/pos_tagging.md)

**Theory:**
- POS tagging helps in understanding the syntactic structure of text.
- Essential for various NLP tasks, including parsing, information extraction, and sentiment analysis.

**Coding Example using nltk:**
```python
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample text
text = "Apple is looking at buying a U.K. startup for $1 billion."

# Tokenize the text
words = word_tokenize(text)

# POS tagging
pos_tags = pos_tag(words)
print(pos_tags)
# Output: [('Apple', 'NNP'), ('is', 'VBZ'), ('looking', 'VBG'), ('at', 'IN'), ('buying', 'VBG'), ('a', 'DT'), ('U.K.', 'NNP'), ('startup', 'NN'), ('for', 'IN'), ('$1', 'JJ'), ('billion', 'CD'), ('.', '.')]
```

**Coding Example using spaCy:**
```python
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "Apple is looking at buying a U.K. startup for $1 billion."

# Process the text
doc = nlp(text)

# POS tagging
for token in doc:
    print(f'{token.text}: {token.pos_}')
# Output:
# Apple: PROPN
# is: AUX
# looking: VERB
# at: ADP
# buying: VERB
# a: DET
# U.K.: PROPN
# startup: NOUN
# for: ADP
# $1: ADJ
# billion: NUM
# .: PUNCT
```

**Uses:**
- Grammar checking.
- Text-to-speech systems.
- Syntactic parsing.

**Significance:**
- Enhances understanding of sentence structure.
- Helps disambiguate the meaning of words.

## [5.3 Dependency Parsing](chapter5/section5.3/README.md)

Dependency parsing involves analyzing the grammatical structure of a sentence to establish relationships between "head" words and words that modify those heads.

### [Using spaCy for dependency parsing](chapter5/section5.3/dependency_parsing.md)

**Theory:**
- Dependency parsing helps to understand the syntactic structure and how words in a sentence relate to each other.
- Each word is linked to its dependents, creating a tree structure.

**Coding Example using spaCy:**
```python
import spacy
from spacy import displacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "Apple is looking at buying a U.K. startup for $1 billion."

# Process the text
doc = nlp(text)

# Dependency parsing
for token in doc:
    print(f'{token.text} ({token.dep_}): {token.head.text}')
# Output (example):
# Apple (nsubj): is
# is (ROOT): is
# looking (xcomp): is
# at (prep): looking
# buying (pcomp): at
# a (det): startup
# U.K. (compound): startup
# startup (pobj): at
# for (prep): startup
# $1 (quantmod): billion
# billion (pobj): for
# . (punct): is

# Visualize the dependency parse
displacy.render(doc, style='dep', jupyter=True)
```

**Uses:**
- Understanding syntactic structure for complex sentence processing.
- Improving machine translation quality.
- Enhancing sentiment analysis by understanding context.

**Significance:**
- Provides deep insights into sentence structure and relationships between words.
- Useful for parsing complex sentences and improving the performance of various NLP applications.

## [5.4 Coreference Resolution](chapter5/section5.4/README.md)

Coreference resolution is the process of determining which words in a text refer to the same entity.

### [Understanding coreference](chapter5/section5.4/understanding_coreference.md)

**Theory:**
- Coreference resolution finds all expressions that refer to the same entity in a text.
- Improves text understanding by linking pronouns and nouns to their antecedents.

**Example:**
```text
John said he would come. (Here, "he" refers to "John")
```

### [Using spaCy and other libraries for coreference resolution](chapter5/section5.4/coreference_resolution.md)

**Theory:**
- Coreference resolution uses algorithms to establish links between pronouns and the nouns they refer to.
- This process often involves complex heuristics and machine learning models.

**Coding Example using spaCy with neuralcoref:**
```python
import spacy
import neuralcoref

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Add neuralcoref to the spaCy pipeline
neuralcoref.add_to_pipe(nlp)

# Sample text
text = "John said he would come to the party. He wasn't sure if he would enjoy it."

# Process the text
doc = nlp(text)

# Print coreferences
print(doc._.coref_clusters)
# Output:
# [John: [John, he, He, he], party: [the party, it]]
```

**Uses:**
- Improving the accuracy of question answering systems.
- Enhancing information extraction and summarization.
- Better understanding references in narrative texts.

**Significance:**
- Resolves ambiguities in text, leading to better understanding and interpretation.
- Essential for tasks that require deep text understanding, such as dialogue systems and complex text analytics.

---

By mastering these advanced NLP techniques, practitioners can significantly enhance the capabilities of text processing applications. Whether building sophisticated chatbots, improving search engines, or conducting in-depth text analysis, these tools and concepts form the foundation for high-quality NLP applications.