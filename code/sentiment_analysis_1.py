import pandas as pd
import re
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import warnings
warnings.filterwarnings('ignore')

class Sentiment:
    dict_sentiments = {'Positive': 1, 'Negative': -1, 'Irrelevant': 0, 'Neutral': 0}
    def __init__(self, line, model, vect):
        self.line = line
        self.model = model
        self.vect = vect

    def data_processing(self):
        text = self.line.lower()
        text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)

    def stemming(self):
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in self.processed]
        return self.processed

    def predictSentiment(self):
        self.processed = self.data_processing()
        text = self.stemming()
        vec = self.vect.transform(pd.Series(text))
        pred = self.model.predict(vec)
        return self.dict_sentiments[pred[0]]
