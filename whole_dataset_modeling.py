import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words("English"))
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import pickle
df = pd.read_csv('C:/Users/Windows10/python files/tripadvisor_hotel_reviews.csv')
df['Review'] = df['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df['Review'] = df['Review'].str.split()
df['Review'] = df['Review'].apply(lambda x: [stemmer.stem(y) for y in x])


def make_sentences(data, name):
    data[name] = data[name].apply(lambda x: ' '.join([i + ' ' for i in x]))
    data[name] = data[name].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))


make_sentences(df, 'Review')


def sentiment(review):
    if review >= 3:
        return 1
    else:
        return 0


df['sentiment'] = df['Rating'].apply(sentiment)
X = df['Review']
Y = df['sentiment']

tfIdfVectorizer = TfidfVectorizer(use_idf=True)
X = tfIdfVectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X, Y)
pickle.dump(model, open('C:/Users/Windows10/python files/trip_review.sav', 'wb'))
pickle.dump(tfIdfVectorizer, open('C:/Users/Windows10/python files/tfidf.sav', 'wb'))
results = pickle.load(open('C:/Users/Windows10/python files/trip_review.sav', 'rb')).score(X, Y)
print(results)
