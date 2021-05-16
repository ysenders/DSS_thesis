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

# # Part I: Exploratory Data Analysis
# The first part of the notebook is about exploring the data. The dataset is loaded, and some standard/basic EDA functions are used in order to get to know the data. Some examples:
# - Import necessary packages
# - Load data
# - Print dataframe
# - Print head/tail of data
# - Check for missing values
# - Check the labels of the tweets
# - Visualise by plotting

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
warnings.simplefilter(action='ignore')


# In[2]:


header_list = ["label", "id", "date", "flag", "username", "content"]


# In[3]:


df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', names=header_list)
df


# In[4]:


#check for missing data

missing_data = df.isna().sum().sort_values(ascending=False)
percentage_missing = round((df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100,2)
missing_info = pd.concat([missing_data,percentage_missing],keys=['Missing values','Percentage'],axis=1)
missing_info.style.background_gradient()


# In[5]:


print(df['content'].head(3))


# In[6]:


print(df['content'].tail(3))


# In[7]:


fig = plt.figure(figsize=(8,8))
targets = df.groupby('label').size()
targets.plot(kind='pie', subplots=True, figsize=(10, 8), autopct = "%.2f%%", colors=['red','green'])
plt.title("Pie chart of different classes of tweets",fontsize=16)
plt.ylabel("")
plt.legend()
plt.show()


# In[8]:


df['label'].value_counts()


# # Part II: Data Cleaning process
# This part is about performing the basic cleaning steps of the paper of Magliani et al. (2016). The basic cleaning steps are:
# - Remove URLs
# - Remove hashtags
# - Remove Mentions (@username)
# - Tabs and line breaks are replaced with a blank, followed by a quotation mark with apexes
# - Words with a sequence of three or more vowels are shortened to two vowels
# - Emoticons are converted to tags
# - Removing extra blank spaces
# - Lowercase text

# In[9]:


# transform labels to 0 and 1
# 0 = negative tweet
# 1 = positive tweet
df.label = df.label.replace({4: 1})


# In[10]:


# drop unnecessary columns
df.drop(["id", "date", "flag", "username"], axis=1, inplace=True)
df


# In[11]:


random_neg = df[:799999].sample(frac = .125, replace = False, random_state=2)
random_pos = df[800000:].sample(frac = .125, replace = False, random_state=2)


# In[12]:


print(random_neg['label'].value_counts())
print(random_pos['label'].value_counts())


# In[13]:


df = pd.concat([random_neg, random_pos])
df


# In[14]:


df['label'].value_counts()


# In[15]:


tweets_train = df['content'] #x_train
y_train = df['label'].values #y_train
print(tweets_train.shape)
print(y_train.shape)


# In[16]:


from url_hashtag import cleaner_url
from emoticon import cleaner_emoji
from punctuation import cleaner_punc
from final import cleaner_final

def cleaned_tweets(tweets):
    """Applies basic cleaning operations and prepare to tokenize"""
    
    cleaned_tweet = []
    for tweet in tweets:
        tweet = cleaner_url(tweet) # remove URLs, hastags, mentions, replace tabs and line breaks.
        tweet = cleaner_emoji(tweet) # convert emoticons into tags
        tweet = cleaner_punc(tweet) # remove punctuation
        tweet = cleaner_final(tweet) # lowercasing and remove extra blank spaces
        cleaned_tweet.append(tweet)
    return cleaned_tweet


# In[17]:


cleaned_train = cleaned_tweets(tweets_train)


# # Part III: Stemming & Lemmatization
# The third part is what this thesis is about: Stemming and Lemmatization. This part contains:
# - Tokenize tweets
# - Apply Porter Stemming filter
# - Apply Snowball Stemming filter
# - Apply Lancaster Stemming filter
# - Apply Lovins Stemming filter
# - Apply Lemmatization filter
# 
# This means we have have five versions of the dataset. We need to keep the original tokenized data as well, because we need this as baseline!

# In[18]:


pip install stemming==1.0.1


# In[19]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')


# In[20]:


from nltk import word_tokenize

def tokenization(tweets):
    """Create tokens of each word in the corpus"""
    
    tokens = []
    for tweet in tweets:
        tweet = word_tokenize(tweet)
        tokens.append(tweet)
        
    return tokens


# In[21]:


tokenized_train = tokenization(cleaned_train)


# In[22]:


def filter_docs(corpus, texts, labels, condition_on_doc):
    """Filter corpus, texts and labels given the function condition_on_doc 
    which takes a doc.
    The document doc is kept if condition_on_doc(doc) is true."""
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} tweets removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels)


# In[23]:


tokenized_train, tweets_train, y_train = filter_docs(tokenized_train, tweets_train, y_train, lambda doc: (len(doc) != 0))


# In[24]:


from nltk.stem import PorterStemmer
porter = PorterStemmer()

def porter_stemmer(tokenized_text):
    stemmed_tokens = [[porter.stem(word) for word in tweet] for tweet in tokenized_text]
    
    return stemmed_tokens


# In[25]:


porter_stemmed_train = porter_stemmer(tokenized_train)


# In[26]:


print(porter_stemmed_train[0:3])


# In[27]:


from nltk.stem.snowball import SnowballStemmer
snowball = SnowballStemmer(language='english')

def snowball_stemmer(tokenized_text):
    stemmed_tokens = [[snowball.stem(word) for word in tweet] for tweet in tokenized_text]
    
    return stemmed_tokens


# In[28]:


snowball_stemmed_train = snowball_stemmer(tokenized_train)


# In[29]:


print(snowball_stemmed_train[0:3])


# In[30]:


from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()

def lancaster_stemmer(tokenized_text):
    stemmed_tokens = [[lancaster.stem(word) for word in tweet] for tweet in tokenized_text]
    
    return stemmed_tokens


# In[31]:


lancaster_stemmed_train = lancaster_stemmer(tokenized_train)


# In[32]:


print(lancaster_stemmed_train[0:3])


# In[33]:


from nltk.stem import WordNetLemmatizer
WNlemmatizer = WordNetLemmatizer()

def lemmatization(tokenized_text):
    lemma_tokens = [[WNlemmatizer.lemmatize(word) for word in tweet] for tweet in tokenized_text]
    
    return lemma_tokens


# In[34]:


lemma_train = lemmatization(tokenized_train)


# In[35]:


print(lemma_train[0:3])


# In[36]:


from stemming.lovins import stem
from text_sanitizer import sanitize_weka

def lovins_error_words(tokenized_text):
    """"This function finds words which the python lovins stemmer cannot stem.
    Unfortunetaly, this stemmer sometimes gives an IndexError.
    The list of words will be stemmed in WEKA GUI."""
    
    # make empty list
    not_stemmed = []
    
    # iterate over each tweet
    for tweet in tokenized_text:
        # iterate over each word in a tweet
        for word in tweet:
            # try to stem the word
            try:
                stem(word)
            # find error words and store them in the empty list
            except IndexError:
                if word not in not_stemmed:
                    word = sanitize_weka(word) # get ready for WEKA
                    not_stemmed.append(word)
            
    # sort list alphabetically for WEKA        
    sorted_list = sorted(not_stemmed) 
    
    sanitize_weka(word)
    return sorted_list

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


# In[37]:


error_words = lovins_error_words(tokenized_train)


# In[38]:


# save the error words to csv for WEKA
dic_errorwords = {'words': error_words}
df_errorwords = pd.DataFrame(dic_errorwords)
df_errorwords.to_csv('df_errorwords.csv', index=False)
df_errorwords


# In[39]:


# load new df with lists of error words and stemmed words from WEKA
df_errorwords_stemmed = pd.read_csv("df_errorwords_stemmed.csv")
lovins_error = df_errorwords_stemmed['error_words'].values
error_stemmed = df_errorwords_stemmed['stemmed_words'].values
df_errorwords_stemmed


# In[40]:


# convert back to list
lovins_error = lovins_error.tolist()
error_stemmed = error_stemmed.tolist()


# In[41]:


# finally, ready to use the lovins stemmer
lovins_stemmed_train = lovins_stemmer(tokenized_train)


# In[42]:


# save stemmed data for later use
data = {'label': y_train, 'tokens': tokenized_train, 'porter': porter_stemmed_train, 
        'snowball': snowball_stemmed_train, 'lancaster': lancaster_stemmed_train, 
        'lovins': lovins_stemmed_train, 'lemmatization': lemma_train}

df_stemmed = pd.DataFrame(data)
df_stemmed.to_csv('df_stemmed.csv', index=False)
df_stemmed


# # Part IV: Word Embeddings
# Before we can train a machine learning classification algorithm, tweets need to be encoded into a (numerical) vector representation. The skip-gram method of Word2Vec will be used. Meaning we have six versions (baseline + five filters)

# In[43]:


from gensim.models import Word2Vec, KeyedVectors
# Importing the built-in logging module
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[44]:


# Set values for various parameters
num_features = 100   # Word vector dimensionality                      
min_word_count = 5   # Minimum word count                        
context = 10         # Window size                                                                                    
hs = 1               # Hierarchical softmax evaluation method
sg = 1               # Skip-gram method


# In[45]:


def train_word2vec(X_train, num_features, min_word_count, context, hs, sg):
    print("Training model...")
    model_W2V = Word2Vec(X_train, 
                     vector_size = num_features,
                     min_count = min_word_count,
                     window = context,
                     hs = hs,
                     sg = sg)
    
    return model_W2V


# In[46]:


w2v_baseline = train_word2vec(tokenized_train, num_features, min_word_count, context, hs, sg)


# In[47]:


w2v_porter = train_word2vec(porter_stemmed_train, num_features, min_word_count, context, hs, sg)


# In[48]:


w2v_snowball = train_word2vec(snowball_stemmed_train, num_features, min_word_count, context, hs, sg)


# In[49]:


w2v_lancaster = train_word2vec(lancaster_stemmed_train, num_features, min_word_count, context, hs, sg)


# In[50]:


w2v_lemma = train_word2vec(lemma_train, num_features, min_word_count, context, hs, sg)


# In[51]:


w2v_lovins = train_word2vec(lovins_stemmed_train, num_features, min_word_count, context, hs, sg)


# In[52]:


# save models for later use
w2v_baseline.save("model_w2v_baseline")
w2v_porter.save("model_w2v_porter")
w2v_snowball.save("model_w2v_snowball")
w2v_lancaster.save("model_w2v_lancaster")
w2v_lemma.save("model_w2v_lemma")
w2v_lovins.save("model_w2v_lovins")


# In[53]:


print(w2v_baseline)
print(w2v_porter)
print(w2v_snowball)
print(w2v_lancaster)
print(w2v_lemma)
print(w2v_lovins)


# In[54]:


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


# In[55]:


x_train_baseline = word_averaging_list(w2v_baseline, tokenized_train)


# In[56]:


x_train_porter = word_averaging_list(w2v_porter, porter_stemmed_train)


# In[57]:


x_train_snowball = word_averaging_list(w2v_snowball, snowball_stemmed_train)


# In[58]:


x_train_lancaster = word_averaging_list(w2v_lancaster, lancaster_stemmed_train)


# In[59]:


x_train_lemma = word_averaging_list(w2v_lemma, lemma_train)


# In[60]:


x_train_lovins = word_averaging_list(w2v_lovins, lovins_stemmed_train)


# In[61]:


print(len(y_train))
print(len(x_train_baseline))
print(len(x_train_porter))
print(len(x_train_snowball))
print(len(x_train_lancaster))
print(len(x_train_lemma))
print(len(x_train_lovins))
print(x_train_baseline.shape)


# # Part V: Grid search
# It could be possible that each method has its preferred parameter settings. Lets try this out via grid search

# In[62]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def grid_search(X_train, Y_train):
    # set svm
    clf = svm.SVC(kernel="linear")
    
    # set all hyperparameters for tuning
    # for a linear kernel we only need top optimize the c parameter
    param_grid = {'C': [0.1, 1, 5, 10]}
    
    # split data
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.20, random_state = 666)
    
    
    # perform grid search on a subset of the data
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, refit=True, verbose=1)
    grid.fit(x_train[0:25000], y_train[0:25000])
    
    return grid.best_params_['C']


# In[63]:


param_c_baseline = grid_search(x_train_baseline, y_train)


# In[64]:


param_c_porter = grid_search(x_train_porter, y_train)


# In[65]:


param_c_snowball = grid_search(x_train_snowball, y_train)


# In[66]:


param_c_lancaster = grid_search(x_train_lancaster, y_train)


# In[67]:


param_c_lemma = grid_search(x_train_lemma, y_train)


# In[68]:


param_c_lovins = grid_search(x_train_lovins, y_train)


# In[69]:


print("baseline: ", param_c_baseline)
print("porter: ", param_c_porter)
print("snowball: ", param_c_snowball)
print("lancaster: ", param_c_lancaster)
print("lemma: ", param_c_lemma)
print("lovins: ", param_c_lovins)


# # Part VI: Training the Machine Learning Classifier
# The machine learning classifier used in this research is the Support Vector Machines (SVM). The SVM will be trained on the baseline + five filters

# In[70]:


from sklearn import svm

def train_SVM(X_train, Y_train, param_c):
    clf = svm.SVC(kernel='linear', C=param_c)
    model_SVM = clf.fit(X_train, Y_train)
    return model_SVM


# In[71]:


SVM_baseline = train_SVM(x_train_baseline, y_train, param_c_baseline)


# In[72]:


SVM_porter = train_SVM(x_train_porter, y_train, param_c_porter)


# In[73]:


SVM_snowball = train_SVM(x_train_snowball, y_train, param_c_snowball)


# In[74]:


SVM_lancaster = train_SVM(x_train_lancaster, y_train, param_c_lancaster)


# In[75]:


SVM_lemma = train_SVM(x_train_lemma, y_train, param_c_lemma)


# In[76]:


SVM_lovins = train_SVM(x_train_lovins, y_train, param_c_lovins)


# In[77]:


print(SVM_baseline)
print(SVM_porter)
print(SVM_snowball)
print(SVM_lancaster)
print(SVM_lemma)
print(SVM_lovins)


# In[78]:


import joblib as joblib
joblib.dump(SVM_baseline, 'model_SVM_baseline.pkl')
joblib.dump(SVM_porter, 'model_SVM_porter.pkl')
joblib.dump(SVM_snowball, 'model_SVM_snowball.pkl')
joblib.dump(SVM_lancaster, 'model_SVM_lancaster.pkl')
joblib.dump(SVM_lemma, 'model_SVM_lemma.pkl')
joblib.dump(SVM_lovins, 'model_SVM_lovins.pkl')


# # Part VII: Testing and evaluation
# All versions will be tested on the testset. We're looking for the accuracy scores of each version. Then, we plot them in a table in order to compare. Which one will get the highest score?

# In[79]:


df_test = pd.read_csv("testdata.manual.2009.06.14.csv", encoding='latin-1', names=header_list)
df_test


# In[80]:


# drop neutral tweets
def drop_neutral(labels):
    count = 0
    list_neutral = []
    for label in labels:
        if label == 2:
            list_neutral.append(count)
        count += 1
    return list_neutral

neutral_index = drop_neutral(df_test['label'])


# In[81]:


df_test = df_test.drop(neutral_index)
df_test


# In[82]:


# transform labels to 0 and 1
# 0 = negative tweet
# 1 = positive tweet
df_test.label = df_test.label.replace({4: 1})
df_test.drop(["id", "date", "flag", "username"], axis=1, inplace=True)
df_test


# In[83]:


df_test['label'].value_counts()


# In[84]:


tweets_test = df_test['content']
y_test = df_test['label'].values


# In[85]:


#clean test tweets
cleaned_test = cleaned_tweets(tweets_test)


# In[86]:


#tokenize test tweets
tokenized_test = tokenization(cleaned_test)


# In[87]:


#stem/lemmatize test tweets
porter_stemmed_test = porter_stemmer(tokenized_test)
snowball_stemmed_test = snowball_stemmer(tokenized_test)
lancaster_stemmed_test = lancaster_stemmer(tokenized_test)
lemma_test = lemmatization(tokenized_test)
lovins_stemmed_test = lovins_stemmer(tokenized_test)


# In[88]:


# create x_test 
x_test_baseline = word_averaging_list(w2v_baseline, tokenized_test)
x_test_porter = word_averaging_list(w2v_porter, porter_stemmed_test)
x_test_snowball = word_averaging_list(w2v_snowball, snowball_stemmed_test)
x_test_lancaster = word_averaging_list(w2v_lancaster, lancaster_stemmed_test)
x_test_lemma = word_averaging_list(w2v_lemma, lemma_test)
x_test_lovins = word_averaging_list(w2v_lovins, lovins_stemmed_test)


# In[91]:


from sklearn.metrics import accuracy_score

def calculate_accuracy(SVM_model, X_test, y_test):
    y_pred = SVM_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    score = print("Accuracy: {:.3}".format(acc))
    
    return score


# In[92]:


acc_baseline = calculate_accuracy(SVM_baseline, x_test_baseline, y_test)


# In[93]:


acc_porter = calculate_accuracy(SVM_porter, x_test_porter, y_test)


# In[94]:


acc_snowball = calculate_accuracy(SVM_snowball, x_test_snowball, y_test)


# In[95]:


acc_lancaster = calculate_accuracy(SVM_lancaster, x_test_lancaster, y_test)


# In[96]:


acc_lemma = calculate_accuracy(SVM_lemma, x_test_lemma, y_test)


# In[97]:


acc_lovins = calculate_accuracy(SVM_lovins, x_test_lovins, y_test)

