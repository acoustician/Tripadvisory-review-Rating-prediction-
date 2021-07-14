# Hotel Sentiment Analysis

Hotels play a crucial role in traveling and with the increased access to information new pathways of selecting the best ones emerged.
With this model, you can explore what makes a great hotel and maybe even use this model in your travels!


Table of contents   
1. Description and the aim of Project
2. Packages used  
3. Text Analysis
4. Pre-processing of text
5. Modeling
6. Deployment Testing
7. Deployment


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


# Text Analysis

At first, I perform Exploratory data analysis (EDA) to analyze and investigate data sets and summarize their main characteristics, often employing data visualization methods.  
Then, I perform Sentiment analysis to gain the sentiment of customer by 'Polarity' and 'Subjectivity'.  

  Polarity - It is the expression that determines the sentimental aspect of an opinion. In textual data, the result of sentiment analysis can be determined for each entity         in the sentence, document or sentence. The sentiment polarity can be determined as positive, negative and neutral.  

  Subjectivity - Subjectivity generally refer to personal opinion, emotion or judgment whereas objective refers to factual information pf the writer.
  
  
  
  
# Pre-Processing of text

1. Removing Stopwords - Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. For example, the words like the, he, have etc. Such words are already captured this in corpus named corpus.Using [TextBlob](https://textblob.readthedocs.io/en/dev/) library, i remove Stopwords 
2.  Stemming/lemmatization - Stemming and Lemmatization both generate the root form of the inflected words, The difference is that stem might not be an actual word whereas, lemma is an actual language word.But, when i perform both than i found that stemming works better then lemmatization for particular problem.so, i go for stemming
3. Applied Regular Expression to remove the punctuation and unwanted symbols if any.
4. Ploted Word CLoud to check the freequent words used in all reviews


#Modeling
The process of modeling means training a machine learning algorithm to predict the labels from the features, tuning it for the business need, and validating it on holdout data.To 
