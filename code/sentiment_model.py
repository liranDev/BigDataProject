import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
import pickle
warnings.filterwarnings('ignore')
import sentiment_analysis_1 as sa
from config import *
from kernals import *

# load the model from disk
prediction_model = pickle.load(open(LOGISTIC_REGRESSION_MODEL, 'rb'))
# load the model from disk
vector_model = pickle.load(open(VECTORS_MODEL, 'rb'))
sentiments = []
length = 50
def predict(sentences):
    for sent in sentences:
        sentiment = sa.Sentiment(sent, prediction_model, vector_model)
        sentiments.append(sentiment.predictSentiment())
    weighted_sentiment = exponent(sentiments)
    return weighted_sentiment