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
model = pickle.load(open('C:/Users/Windows10/python files/trip_review.sav','rb'))
tfidf = pickle.load(open('C:/Users/Windows10/python files/tfidf.sav','rb'))
def review_predictor(text):
    df =pd.DataFrame([[text]],columns=['Review'])
    df['Review'] = df['Review'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
    df['Review'] = df['Review'].str.split()
    df['Review'] = df['Review'].apply(lambda x: [stemmer.stem(y) for y in x])
    df['Review'] = df['Review'].apply(lambda x:' '.join([i+' ' for i in x]))
    df['Review'] = df['Review'].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    X = df['Review']
    X = tfidf.transform(X)
    X = model.predict(X)
    X = X[0]
    return X
import streamlit as st
st.write("""
# Hotel Predictions  
Input text Below to get sentiment.
""")
test = st.text_input("Please enter a Review:\n")
result = review_predictor(test)
if result == 1:
    st.success('Positive Review')
else:
    st.error('Negative Review')