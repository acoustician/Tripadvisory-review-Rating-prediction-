# Hotel Sentiment Analysis

Hotels play a crucial role in traveling and with the increased access to information new pathways of selecting the best ones emerged.
With this model, you can explore what makes a great hotel and maybe even use this model in your travels!


Table of contents   
1. Description and the aim of Project
2. Packages used  
3. Text Analysis
4. Pre-processing of text
5. Vectorization and Modeling
6. Deployment and testing
7. Conclusion.


# Description and the aim of Project
The aim of the model is to predict the Rating of hotel by Review.These Model is trained by dataset of hotel consisting of 20k reviews crawled from Tripadvisor. These Dataset has two features, first one is Review and another one is Rating.Review is the opinion of the customer in form of text and Rating is the opinion of customers in form of number from 1 to 5.These model gives Binary Rating when we pass Text Review in it.  


# Packages Used

1. [Pandas](https://pandas.pydata.org/about/)
2. [Numpy](https://numpy.org/)
3. [Seaborn](https://seaborn.pydata.org/)
4. [Matplotlib](https://matplotlib.org/)
5. [TextBlob](https://textblob.readthedocs.io/en/dev/)
6. [Natural Language Toolkit(NLTK)](https://www.nltk.org/)
7. [SnowballStemmer](https://snowballstem.org/) 
8. [Regular Expression(re)](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions)
9. [WordCloud](https://www.wordclouds.com/)
10. [Scikitlearn(sklearn)](https://scikit-learn.org/stable/)
11. [pickle](https://docs.python.org/3/library/pickle.html#:~:text=%E2%80%9CPickling%E2%80%9D%20is%20the%20process%20whereby,back%20into%20an%20object%20hierarchy)  


# Text Analysis

Exploratory data analysis (EDA) to analyze and investigate dataset and summarize their main characteristics, often employing data visualization methods using [Seaborn](https://seaborn.pydata.org/) and [Matplotlib](https://matplotlib.org/)
Then, Sentiment analysis to gain the sentiment of customer by 'Polarity' and 'Subjectivity' using  [TextBlob](https://textblob.readthedocs.io/en/dev/)

  Polarity - It is the expression that determines the sentimental aspect of an opinion. In textual data, the result of sentiment analysis can be determined for each entity         in the sentence, document or sentence. The sentiment polarity can be determined as positive, negative and neutral.  

  Subjectivity - Subjectivity generally refer to personal opinion, emotion or judgment whereas objective refers to factual information of the writer.
  
  
  
  
# Pre-Processing of text

1. Removing Stopwords - Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. For example, the words like the, he, have etc. Such words are already captured this in corpus named corpus.Using [Natural Language Toolkit(NLTK)](https://www.nltk.org/) package 
2.  Stemming/lemmatization - Stemming and Lemmatization both generate the root form of the inflected words, The difference is that stem might not be an actual word whereas, lemma is an actual language word.But, when for that specific problem stemming works better then lemmatization using [Natural Language Toolkit(NLTK)](https://www.nltk.org/) package
3. Applied [Regular Expression(re)](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions) to remove the punctuation and unwanted symbols if any using.
4. Ploted [WordCloud](https://www.wordclouds.com/) to check the freequent words used in all reviews.


# Vectorization and Modeling.
Vectorization - Word vectorization is the process of encoding individual words into vectors so that the text can be easily analyzed or consumed by the machine learning algorithm. It’s difficult to analyse the raw corpus therefore a need to be convert it in to integers(best format is vectors) where we can apply mathematical operations and get insights from the data using [Scikitlearn(sklearn)](https://scikit-learn.org/stable/).

Modeling - The process of modeling means training a machine learning algorithm to predict the labels from the features, tuning it for the business need, and validating it on holdout data.To choose best performing model.

Tested six combination of algorithm and vectorization techniques using [Scikitlearn(sklearn)](https://scikit-learn.org/stable/) which are as follows:-

1. [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) with [Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
2. [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) with [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
3. [TFIDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) with [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
4. [Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) Vectorization with [Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
5. [Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) with [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
6. [Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) with [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html).

# Deployment and its testing
  At first, Best model choosen out of the six combination of vectorizer and algorithm than perform the same for whole dataset without spliting the dtaset.The best model and vectorizer method is stored by using pickle.                                                                                                                                       For deployment testing a function was defined by using stored model from pickle, when a single sentence review passed to that function it returns sentiment of customer(you can also get that files which uploaded with these repository).  
  These model is deployed using [Streamlit](https://streamlit.io/), screenshot is also attached to it.
  

# Conclusion
  Performed modeling by split the dataset, Perfromed six diffrent combination of vectorization and algorithm for train the model then found that the Logistic Regression with tfidf vectorization gives best accuracy among all. Than, applied those combination of Vectorizer and algorithm on whole dataset. These model gave 96.4% accuracy
  
  
                
