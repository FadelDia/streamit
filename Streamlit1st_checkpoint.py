# To install the necessary libraries, you can use the following command in your command prompt or terminal:

pip install nltk streamlit

# Once we have installed the libraries, we can import them in your Python script using the following code:

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st

# Loading and Preprocessing Data:

