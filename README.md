# Semantic Textual Similarity

Semantic Textual Similarity (STS) is a measure of how similar two pieces of
text are in meaning, regardless of their surface-level differences. It aims to
capture the underlying semantic relationship between texts, which can be
crucial for various natural language processing tasks.

## Datasets used

1. Mono-lingual SemEval Datasets(Train, Validation & Test) - Data STS 2017 Trial Data
2. Cross-Lingual SemEval Datasets (Training, Validation and Testing Sets) - STS Cross-lingual English-Spanish

## Data Preprocessing

Data cleaning was done in many stages:
1. Removal of URLs, hashtags, punctuations, etc.
2. Removed the stop words.
3. Lemmatization and Stemming.
4. For the Cross-Lingual case, stopwords of both English and Spanish language are removed.

## Models used

1. Word2Vec
2. BERT Model (both untrained and Fine-tuned)
3. RoBERTa Model (both untrained and Fine-tuned)
4. Doc2Vec (both untrained and Fine-tuned)
5. Sentence Transformers (both untrained and Fine-tuned)
6. Siamese BiLSTMs
7. Universal Sentence Encoder (USE) (both untrained and Fine-tuned)