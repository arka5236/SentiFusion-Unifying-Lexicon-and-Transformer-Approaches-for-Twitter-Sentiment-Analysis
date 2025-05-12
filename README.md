Overview


SentiFusion is a comprehensive sentiment analysis pipeline that processes a large Twitter dataset to extract insights using both traditional lexicon-based and modern transformer-based approaches. The project integrates data preprocessing, TF-IDF feature extraction, sentiment scoring using VADER and RoBERTa, and even extends to training and evaluating machine learning classifiers. Comparative visualizations and error analysis further highlight the strengths and limitations of each method, paving the way for future improvements.
Features
Data Preprocessing:

Robust cleaning of tweets including removal of unwanted characters, URLs, and mentions.

Trimming and normalization of sentiment labels for binary classification.

Feature Extraction:

Transformation of cleaned text using TF-IDF vectorization.

Sentiment Analysis:

VADER: Computes compound sentiment scores to quickly gauge overall sentiment.

RoBERTa: Uses a fine-tuned transformer model for context-aware sentiment predictions.

Comparative Analysis:

Generates visualizations (e.g., bar plots, box plots, pairplots) to compare outputs from VADER and RoBERTa.
Machine Learning Modeling:

Implements traditional classifiers (e.g., Naive Bayes, Logistic Regression) on TF-IDF features.

Evaluates model performance using metrics like accuracy, precision, recall, and confusion matrices.

Error and Feature Analysis:

Performs error analysis by examining misclassified examples.

Attempts feature importance exploration where appropriate.


                          ┌─────────────────────────────┐
                          │  Twitter Dataset            │
                          │ "training.1600000.processed │
                          │  .noemoticon.csv"           │
                          └─────────────┬───────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────────┐
                          │    Data Preprocessing       │
                          │  (Cleaning, Normalization,  │
                          │   Filtering Binary Labels)  │
                          └─────────────┬───────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────────┐
                          │  Feature Extraction         │
                          │     (TF-IDF Vectorization)  │
                          └─────────────┬───────────────┘
                                        │
                                        ▼
                           ┌────────────┴─────────────┐
                           │                          │
                           ▼                          ▼
           ┌─────────────────────────┐    ┌─────────────────────────┐
           │ VADER Sentiment Analysis│    │  RoBERTa Sentiment      │
           │ (Compound Score,        │    │  Analysis (Labels and   │
           │  Lexicon-based)         │    │  Confidence Scores)     │
           └────────────┬────────────┘    └────────────┬────────────┘
                        │                             │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────────────┐
                          │ Comparative Analysis &      │
                          │ Visualization               │
                          │ (Heatmaps, Pairplots, etc.) │
                          └────────────┬───────────────┘
                                     │
                                     ▼
                          ┌─────────────────────────────┐
                          │ Machine Learning Modeling   │
                          │ (Classifier Training &       │
                          │  Evaluation)                │
                          └────────────┬───────────────┘
                                     │
                                     ▼
                          ┌─────────────────────────────┐
                          │ Error Analysis &            │
                          │ Feature Exploration         │
                          └─────────────────────────────┘
Requirements
Python 3.x

Pandas

NumPy

scikit-learn

NLTK

Transformers (Hugging Face)

Seaborn

Matplotlib

Results & Analysis
Evaluation Metrics: The pipeline outputs classification reports, confusion matrices, and accuracy scores based on model predictions.

Visual Insights: Comparative visualizations are generated to compare sentiment outputs from VADER and RoBERTa.

Error Analysis: Sample misclassified tweets are highlighted to understand model limitations and drive further improvements.

Future Work
Explore ensemble methods merging VADER and RoBERTa outputs.

Experiment with deep learning models or advanced transformer architectures.

Refine preprocessing steps to better capture the intricacies of social media text.
