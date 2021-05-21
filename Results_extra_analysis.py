#!/usr/bin/env python
# coding: utf-8

# # Notebook for Master Thesis
# Author: Youri Senders <br/>
# Student Number: 2018966 <br/>
# ANR: 895590 <br/>
# 
# Title: The Impact of Stemming and Lemmatization in Predicting Sentiment Polarity of Twitter data <br/>
# Dataset: Sentiment140 <br/>
# Date: Feb 2021 - June 2021
# 
# This notebook is used for conducting the extra results. It is a seperate notebook because running the main notebook again is time-consuming!

# # Part I: Loading models and preprocess test data
# The first part consists of loading all the W2V and SVM models trained in the main notebook. The test set needs to be cleaned again. Hence, most of this part is repetitive to the main document. See part II for the results

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
warnings.simplefilter(action='ignore')


# In[2]:


from gensim.models import Word2Vec, KeyedVectors
w2v_baseline = Word2Vec.load("model_w2v_baseline")
w2v_porter = Word2Vec.load("model_w2v_porter")
w2v_snowball = Word2Vec.load("model_w2v_snowball")
w2v_lancaster = Word2Vec.load("model_w2v_lancaster")
w2v_lemma = Word2Vec.load("model_w2v_lemma")
w2v_lovins = Word2Vec.load("model_w2v_lovins")


# In[3]:


import joblib as joblib
SVM_baseline = joblib.load('model_SVM_baseline.pkl')
SVM_porter = joblib.load('model_SVM_porter.pkl')
SVM_snowball = joblib.load('model_SVM_snowball.pkl')
SVM_lancaster = joblib.load('model_SVM_lancaster.pkl')
SVM_lemma = joblib.load('model_SVM_lemma.pkl')
SVM_lovins = joblib.load('model_SVM_lovins.pkl')


# In[4]:


header_list = ["label", "id", "date", "flag", "username", "content"]


# In[5]:


df_test = pd.read_csv("testdata.manual.2009.06.14.csv", encoding='latin-1', names=header_list)
df_test


# In[6]:


# drop neutral tweets
def drop_neutral(labels):
    count = 0
    list_neutral = []
    for label in labels:
        print(label)
        if label == 2:
            list_neutral.append(count)
        count += 1
    print(count)
    return list_neutral

neutral_index = drop_neutral(df_test['label'])
print(neutral_index)


# In[7]:


df_test = df_test.drop(neutral_index)
df_test


# In[8]:


# transform labels to 0 and 1
# 0 = negative tweet
# 1 = positive tweet
df_test.label = df_test.label.replace({4: 1})
df_test.drop(["id", "date", "flag", "username"], axis=1, inplace=True)
df_test


# In[9]:


df_test['label'].value_counts()


# In[10]:


tweets_test = df_test['content']
y_test = df_test['label'].values
print(tweets_test[0:10])
print(y_test[0:10])


# In[11]:


from url_hashtag import cleaner_url
from emoticon import cleaner_emoji
from punctuation import cleaner_punc
from final import cleaner_final

def cleaned_tweets(tweets):
    cleaned_tweet = []
    for tweet in tweets:
        tweet = cleaner_url(tweet) # remove URLs, hastags, mentions, replace tabs and line breaks.
        tweet = cleaner_emoji(tweet) # convert emoticons to tags
        tweet = cleaner_punc(tweet) # remove punctuation
        tweet = cleaner_final(tweet) # lowercasing and remove extra blank spaces
        cleaned_tweet.append(tweet)
    return cleaned_tweet

cleaned_test = cleaned_tweets(tweets_test)


# In[12]:


from nltk import word_tokenize

def tokenization(tweets):
    tokens = []
    for tweet in tweets:
        tweet = word_tokenize(tweet)
        tokens.append(tweet)
        
    return tokens

tokenized_test = tokenization(cleaned_test)


# In[13]:


from nltk.stem import PorterStemmer
porter = PorterStemmer()

def porter_stemmer(tokenized_text):
    stemmed_tokens = [[porter.stem(word) for word in tweet] for tweet in tokenized_text]
    
    return stemmed_tokens

porter_stemmed_test = porter_stemmer(tokenized_test)


# In[14]:


from nltk.stem.snowball import SnowballStemmer
snowball = SnowballStemmer(language='english')

def snowball_stemmer(tokenized_text):
    stemmed_tokens = [[snowball.stem(word) for word in tweet] for tweet in tokenized_text]
    
    return stemmed_tokens

snowball_stemmed_test = snowball_stemmer(tokenized_test)


# In[15]:


from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()

def lancaster_stemmer(tokenized_text):
    stemmed_tokens = [[lancaster.stem(word) for word in tweet] for tweet in tokenized_text]
    
    return stemmed_tokens

lancaster_stemmed_test = lancaster_stemmer(tokenized_test)


# In[16]:


from nltk.stem import WordNetLemmatizer
WNlemmatizer = WordNetLemmatizer()

def lemmatization(tokenized_text):
    lemma_tokens = [[WNlemmatizer.lemmatize(word) for word in tweet] for tweet in tokenized_text]
    
    return lemma_tokens

lemma_test = lemmatization(tokenized_test)


# In[17]:


from stemming.lovins import stem
def lovins_stemmer(tokenized_text):
    """The lovins stemmer needs more attention.
    Error words are changed for stemmed words in WEKA."""
    
    # first create an empty list
    words = []
    
    # iterate over each tweet
    for index, tweet in enumerate(tokenized_text):
        # create a new list per tweet
        words.append([])
        # iterate over each word in a tweet
        for word in tweet:
            # change error word for WEKA stemmed word
            if word in lovins_error:
                ix = lovins_error.index(word)
                error_word = lovins_error[ix]
                stemmed_word = error_stemmed[ix]
                words[index].append(stemmed_word)
            # stem word via python Lovins stemmer
            else:
                d = stem(word)
                words[index].append(d)
        
    return words


# In[18]:


import gensim
def word_averaging(model, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in model.wv.key_to_index:
            mean.append(model.wv.get_vector(word, norm=True))
            all_words.add(model.wv.key_to_index[word])

    if not mean:
        print(mean)
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(model.wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(model, text_list):
    return np.vstack([word_averaging(model, review) for review in text_list ])


# In[19]:


df_errorwords_stemmed = pd.read_csv("df_errorwords_stemmed.csv")
lovins_error = df_errorwords_stemmed['error_words'].values
error_stemmed = df_errorwords_stemmed['stemmed_words'].values
# convert back to list
lovins_error = lovins_error.tolist()
error_stemmed = error_stemmed.tolist()


# In[20]:


#clean test tweets
cleaned_test = cleaned_tweets(tweets_test)


# In[21]:


#tokenize test tweets
tokenized_test = tokenization(cleaned_test)


# In[22]:


#stem/lemmatize test tweets
porter_stemmed_test = porter_stemmer(tokenized_test)
snowball_stemmed_test = snowball_stemmer(tokenized_test)
lancaster_stemmed_test = lancaster_stemmer(tokenized_test)
lemma_test = lemmatization(tokenized_test)
lovins_stemmed_test = lovins_stemmer(tokenized_test)


# In[23]:


# create x_test 
x_test_baseline = word_averaging_list(w2v_baseline, tokenized_test)
x_test_porter = word_averaging_list(w2v_porter, porter_stemmed_test)
x_test_snowball = word_averaging_list(w2v_snowball, snowball_stemmed_test)
x_test_lancaster = word_averaging_list(w2v_lancaster, lancaster_stemmed_test)
x_test_lemma = word_averaging_list(w2v_lemma, lemma_test)
x_test_lovins = word_averaging_list(w2v_lovins, lovins_stemmed_test)


# In[24]:


# add all stemmed versions to dataset
df_test['tokenized'] = tokenized_test
df_test['porter'] = porter_stemmed_test
df_test['snowball'] = snowball_stemmed_test
df_test['lancaster'] = lancaster_stemmed_test
df_test['lemmatization'] = lemma_test


# In[25]:


df_test


# # Part II: Results
# This section provides the extra results of the research. These are the accuracy score, precision score, recall score, and confusion matrix for each method.

# In[26]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, plot_confusion_matrix

def evaluation_metrics(SVM_model, X_test, y_test):
    y_pred = SVM_model.predict(X_test)
    
    # accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score: {:.3}".format(acc))
    
    # confusion matrix
    cf = confusion_matrix(y_test, y_pred)
    cf2 = plot_confusion_matrix(SVM_model, X_test, y_test)
    print("Confusion matrix:")
    print(cf)
    
    # precision
    precision = precision_score(y_test, y_pred)
    print("Precision score: {:.3}".format(precision))
    
    # recall
    recall = recall_score(y_test, y_pred)
    print("Recall score: {:.3}".format(recall))


# In[27]:


evaluation_baseline = evaluation_metrics(SVM_baseline, x_test_baseline, y_test)


# In[28]:


evaluation_porter = evaluation_metrics(SVM_porter, x_test_porter, y_test)


# In[29]:


evaluation_snowball = evaluation_metrics(SVM_snowball, x_test_snowball, y_test)


# In[30]:


evaluation_lancaster = evaluation_metrics(SVM_lancaster, x_test_lancaster, y_test)


# In[31]:


evaluation_lemma = evaluation_metrics(SVM_lemma, x_test_lemma, y_test)


# In[32]:


evaluation_lovins = evaluation_metrics(SVM_lovins, x_test_lovins, y_test)

