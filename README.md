# NLU-Assignment-02
This repository contains solutions for two problems:

1. Word Embedding Learning using Word2Vec (CBOW & Skip-gram)  
2. Character-Level Name Generation using RNN-based models

# Requirements

1. Install the following dependencies before running:
    pip install torch numpy nltk gensim matplotlib scikit-learn pdfminer.six
2. Also download NLTK tokenizer:
    import nltk
    nltk.download('punkt')


# How to Run

1. Problem 1 (Word2Vec)

a. Run preprocessing to generate cleaned corpus:
    python your_preprocessing_script.py
b. Train Word2Vec models:
    python your_word2vec_script.py
c. Run analysis (nearest neighbors, analogies):
    python your_analysis_script.py
d. Generate visualizations:
    python your_visualization_script.py


2. Problem 2 (Name Generation)
a. enerate Dataset
    python create_names.py
b. Preprocess Data
    python preprocess.py
c. Train Models
i. Vanilla RNN:
    python rnn_model.py
ii. BLSTM:
    python blstm_model.py
iii. Attention RNN:
    python attention_rnn.py

d. Run evaluation for generated names:
    python evaluate.py
This will output:
  a. Total generated names
  b. Novelty rate
  c. Diversity score
