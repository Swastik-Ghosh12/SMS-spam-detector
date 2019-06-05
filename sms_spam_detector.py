

import nltk

import pandas as pd
import numpy as np

import string

from sklearn.feature_extraction.text import CountVectorizer


nltk.download_shell()


sms=[line for line in open('SMSSpamCollection')]

print(len(sms))
type(sms)

sms[43]

df=pd.read_csv('SMSSpamCollection', sep='\t', names=['target','sms'])

df.head()

df.describe()

df['length']=df['sms'].apply(len)

df.head()

s="this is a sample message. Please remove the punctuations ? ! Please ! !"
s
npunc=[char for char in s if char not in string.punctuation]

npunc=[char for char in s if char not in string.punctuation]

npunc

npunc=''.join(npunc)

npunc

from nltk.corpus import stopwords

clean_s=[word for word in npunc.split()
        if word.lower() not in stopwords.words('english')]

clean_s

def process_text(raw_text):
  #check for punctuations
  npunc=[char for char in raw_text if char not in string.punctuation]
  
  #form a sentence again
  npunc=''.join(npunc)
  
  #removing stopwords
  return [word for word in npunc.split()
   if word.lower() not in stopwords.words('english')]

df['sms'].head(2)

df['sms'].apply(process_text)

bow_transformer=CountVectorizer(analyzer=process_text).fit(df['sms'])

df_bow=bow_transformer.transform(df['sms'])

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_trans=TfidfTransformer()

tfidf_trans.fit(df_bow)

df_tfidf=tfidf_trans.transform(df_bow)

print(df_tfidf)

from sklearn.naive_bayes import MultinomialNB

spam_ham_detector=MultinomialNB()

from sklearn.model_selection import train_test_split

x=df_tfidf
y=df['target']

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3, random_state=42)

spam_ham_detector.fit(x_train, y_train)

accuracy=spam_ham_detector.score(x_test, y_test)

print(accuracy*100)

