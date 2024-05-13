import pandas as pd
df = pd.read_csv('C:\\Users\\Musae\\Documents\\GitHub-REPOs\\NLP-Project\\data\\ar_reviews_100k.tsv', sep='\t')
df.head()
df.isnull().sum()
df['label'].value_counts()
df = df[df['label']!='Mixed']
df.duplicated().sum()
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = list(set(stopwords.words('arabic')))
print(stop_words)

import re
import string
import sys
import argparse
from nltk.tokenize import word_tokenize

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

def remove_diacritics(text):
    arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(arabic_diacritics, '', str(text))
    return text

def remove_emoji(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    text = remove_emoji(text)
    text = remove_diacritics(text)
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text

df['cleanedtext'] = df['text'].apply(clean_text)
df.head()
def process_text(text):
    stemmer = nltk.ISRIStemmer()
    word_list = nltk.word_tokenize(text)
    #stemming
    word_list = [stemmer.stem(w) for w in  word_list]
    return ' '.join(word_list)

df['cleanedtextnew']=df['cleanedtext'].apply(process_text)
df.head()
### split data to train and test 
from sklearn.model_selection import train_test_split
x,y=df['cleanedtext'],df['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
vectorizer=TfidfVectorizer() #(analyzer='char_wb',ngram_range=(3,5),min_df=0.01,max_df=0.3)

clf=SVC(kernel='rbf')
from sklearn.pipeline import make_pipeline
pipe=make_pipeline(vectorizer,clf)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,y_pred)