# Hotel Sentiment Analysis

Hotels play a crucial role in traveling and with the increased access to information new pathways of selecting the best ones emerged.
With this model, you can explore what makes a great hotel and maybe even use this model in your travels!


Table of contents   
1. Description and the aim of Project
2. Packages used  
3. Text Analysis
4. Pre-processing of text
5. Vectorization and Modeling
6. Conclusion.


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

At first, I perform Exploratory data analysis (EDA) to analyze and investigate data sets and summarize their main characteristics, often employing data visualization methods.  
Then, I perform Sentiment analysis to gain the sentiment of customer by 'Polarity' and 'Subjectivity'.  

  Polarity - It is the expression that determines the sentimental aspect of an opinion. In textual data, the result of sentiment analysis can be determined for each entity         in the sentence, document or sentence. The sentiment polarity can be determined as positive, negative and neutral.  

  Subjectivity - Subjectivity generally refer to personal opinion, emotion or judgment whereas objective refers to factual information pf the writer.
  
  
  
  
# Pre-Processing of text

1. Removing Stopwords - Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. For example, the words like the, he, have etc. Such words are already captured this in corpus named corpus.Using [TextBlob](https://textblob.readthedocs.io/en/dev/) library, i remove Stopwords 
2.  Stemming/lemmatization - Stemming and Lemmatization both generate the root form of the inflected words, The difference is that stem might not be an actual word whereas, lemma is an actual language word.But, when i perform both than i found that stemming works better then lemmatization for particular problem.so, i go for stemming
3. Applied Regular Expression to remove the punctuation and unwanted symbols if any.
4. Ploted Word CLoud to check the freequent words used in all reviews


# Vectorization and Modeling.
Vectorization - Word vectorization is the process of encoding individual words into vectors so that the text can be easily analyzed or consumed by the machine learning algorithm. It’s difficult to analyse the raw corpus therefore a need to be convert it in to integers(best format is vectors) where we can apply mathematical operations and get insights from the data.

Modeling - The process of modeling means training a machine learning algorithm to predict the labels from the features, tuning it for the business need, and validating it on holdout data.To choose best performing model.

I tested six combination of algorithm and vectorization techniques which are as follows:-

1. TF-IDF with Logistic regression
2. TF-IDF with Random Forest
3. TFIDF with Naive Bayes
4. Count Vectorization with Logistic regression
5. Count Vectorization with Random forest
6. Count Vectorization with Naive Bayes

TFIDF Vectorization - TFIDF short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently.

Count Vectorization - CountVectorizer creates a matrix in which each unique word is represented by a column of the matrix, and each text sample from the document is a row in the matrix. The value of each cell is nothing but the count of the word in that particular text sample. 

Logistic Regression - Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist.

Random Forest - Random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time.

Naive Bayes - Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems.It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.

# Conclusion
  when i perform six diffrent combination of vectorization and algorithm for train the model then i found that, the Logistic Regression with tfidf vectorization gives best accuracy among all. 
  When we pass statement in the defined function of model then we get the sentiment of that review in from of binary(0 and 1).   
