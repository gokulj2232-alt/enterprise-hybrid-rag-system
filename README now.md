# NLP Semantic Search System

## Project Overview

This project is an NLP-based semantic search system that searches a collection of 50,000 documents and retrieves the most relevant documents for a user query.

The system uses Word2Vec word embeddings, TF-IDF weighting, document vectors, and cosine similarity to perform semantic search.

## Dataset

Dataset file:

NLP\_50K\_Document\_Dataset.xlsx

Number of documents:

50,000

Original columns:

* document\_id
* category
* title
* content
* keyword

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* BeautifulSoup
* Gensim
* Scikit-learn
* Streamlit
* Excel

## NLP Preprocessing

The text preprocessing pipeline includes:

1. Load Excel dataset
2. Validate dataset columns
3. Handle missing values
4. Combine title, content, and keyword
5. Convert text to lowercase
6. Remove HTML
7. Remove URLs
8. Remove special characters
9. Normalize whitespace
10. Tokenize text
11. Remove English stopwords

## Semantic Search Method

The system uses the following process:

User Query
→ Text Preprocessing
→ TF-IDF Weighting
→ Word2Vec Embeddings
→ Query Vector
→ Cosine Similarity
→ Ranking
→ Top-K Results

## Word2Vec Configuration

* Vector size: 100
* Window size: 5
* Minimum word count: 1
* Training method: Skip-gram
* Epochs: 10
* Random seed: 42

## TF-IDF

TF-IDF is used to assign importance weights to words.

The TF-IDF weights are combined with Word2Vec word vectors to create weighted document vectors and query vectors.

## Similarity Measurement

Cosine similarity is used to compare the query vector with all document vectors.

Documents are ranked according to their similarity score.

## Saved Models

The trained system stores the following files:

* models/word2vec.model
* models/document\_vectors.npy
* models/tfidf\_vectorizer.pkl
* models/processed\_data.pkl

These files allow the application to load the trained system without retraining Word2Vec every time.

## Streamlit Application

The Streamlit application provides:

* Search box
* Top-K result selection
* Processed query display
* Known vocabulary display
* Similarity scores
* Document details
* Category and keyword information

## How to Run

Activate the virtual environment:

.venv\\Scripts\\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

## Example Queries

* machine learning
* cloud computing
* python programming
* information technology

## Project Structure

semantic\_search\_project/

├── app.py

├── requirements.txt

├── README.md

├── data/

│   └── NLP\_50K\_Document\_Dataset.xlsx

├── models/

│   ├── word2vec.model

│   ├── document\_vectors.npy

│   ├── tfidf\_vectorizer.pkl

│   └── processed\_data.pkl

└── .venv/

## Project Status

The NLP semantic search system is successfully implemented and integrated with a Streamlit user interface.

