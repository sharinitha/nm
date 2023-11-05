# NAAN-MUDHALVAN-AI
Here we have used some machine learning algorithms to create a chatbot using python....
# DATA-SOURCE
https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot
# DEPENDENCIES
import os
import pickle
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Masking
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
# CODE
for the entire code refer chatbot.ipynb
